from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PseudoMaskDataset
from .infer import predict_array
from .losses import pseudo_supervised_loss, teacher_student_consistency
from .models import build_model
from .utils import ensure_dir, list_images, read_image_rgb, save_mask, seed_everything, stem_key, write_json, match_mask_path, read_mask_any


@dataclass
class TrainConfig:
    images: str
    pseudo_dir: str
    workdir: str
    model: str = "unet"
    base_channels: int = 32
    dropout: float = 0.05
    crop_size: int = 512
    epochs: int = 80
    rounds: int = 1
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-5
    boundary_weight: float = 0.2
    consistency_weight: float = 0.1
    consistency_threshold: float = 0.70
    ema_decay: float = 0.99
    num_workers: int = 2
    amp: bool = False
    device: str = "cuda"
    seed: int = 2026
    val_fraction: float = 0.10
    val_images: str | None = None
    recursive: bool = False
    refine_after_round: bool = True
    refine_alpha: float = 0.65
    refine_threshold: float = 0.5


def split_train_val(paths: List[Path], val_fraction: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    n_val = int(round(len(paths) * val_fraction))
    n_val = max(1, n_val) if len(paths) >= 5 and val_fraction > 0 else 0
    val_idx = set(idx[:n_val].tolist())
    train = [p for i, p in enumerate(paths) if i not in val_idx]
    val = [p for i, p in enumerate(paths) if i in val_idx]
    return train, val


def update_ema(student: torch.nn.Module, teacher: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(decay).add_(ps.data, alpha=1.0 - decay)
        for bs, bt in zip(student.buffers(), teacher.buffers()):
            bt.copy_(bs)


@torch.no_grad()
def evaluate_pseudo(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    dices = []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        logits = model(x)
        pred = (torch.sigmoid(logits) >= 0.5).float()
        inter = (pred * y).sum(dim=(1, 2, 3))
        denom = pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (denom + 1e-6)
        dices.extend(dice.detach().cpu().numpy().tolist())
    return float(np.mean(dices)) if dices else 0.0


def refine_pseudo_masks(
    model: torch.nn.Module,
    image_paths: List[Path],
    current_pseudo_dir: Path,
    out_dir: Path,
    alpha: float,
    threshold: float,
    device: str,
) -> Path:
    ensure_dir(out_dir)
    model.eval()
    for p in tqdm(image_paths, desc="refine-pseudo"):
        img = read_image_rgb(p)
        prob = predict_array(model, img, device=device)
        old_path = match_mask_path(current_pseudo_dir, p)
        old = read_mask_any(old_path)
        if old.shape != prob.shape:
            old = cv2.resize(old, (prob.shape[1], prob.shape[0]), interpolation=cv2.INTER_NEAREST)
        fused = alpha * prob + (1.0 - alpha) * old.astype(np.float32)
        mask = (fused >= threshold).astype(np.uint8)
        save_mask(out_dir / f"{stem_key(p)}.png", mask)
    return out_dir


def save_checkpoint(path: Path, model: torch.nn.Module, cfg: TrainConfig, epoch: int, round_id: int, metric: float) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "round": round_id,
            "pseudo_val_dice": metric,
            "model_config": {"name": cfg.model, "base": cfg.base_channels, "dropout": cfg.dropout},
            "train_config": asdict(cfg),
        },
        path,
    )


def run_training(cfg: TrainConfig) -> Path:
    seed_everything(cfg.seed)
    workdir = ensure_dir(cfg.workdir)
    ckpt_dir = ensure_dir(workdir / "checkpoints")
    log_path = workdir / "train_log.jsonl"
    write_json(workdir / "train_config.json", asdict(cfg))

    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
    image_paths = list_images(cfg.images, recursive=cfg.recursive)
    if cfg.val_images:
        train_paths = image_paths
        val_paths = list_images(cfg.val_images, recursive=cfg.recursive)
    elif cfg.val_fraction <= 0:
        train_paths = image_paths
        val_paths: List[Path] = []
    else:
        train_paths, val_paths = split_train_val(image_paths, cfg.val_fraction, cfg.seed)
        if not val_paths:
            val_paths = train_paths[: min(len(train_paths), max(1, cfg.batch_size))]
    print(f"[train] train-images={len(image_paths)}, train={len(train_paths)}, pseudo-val={len(val_paths)}")

    student = build_model(cfg.model, base=cfg.base_channels, dropout=cfg.dropout).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    best_metric = float("-inf")
    current_pseudo = Path(cfg.pseudo_dir)

    for round_id in range(cfg.rounds):
        print(f"[train] Round {round_id + 1}/{cfg.rounds}; pseudo_dir={current_pseudo}")
        train_ds = PseudoMaskDataset(
            train_paths,
            current_pseudo,
            crop_size=cfg.crop_size,
            mode="train",
            seed=cfg.seed + round_id * 101,
            boundary_weight=cfg.boundary_weight,
        )
        val_ds = None
        val_loader = None
        if val_paths:
            val_ds = PseudoMaskDataset(
                val_paths,
                current_pseudo,
                crop_size=cfg.crop_size,
                mode="val",
                seed=cfg.seed,
                boundary_weight=cfg.boundary_weight,
            )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        if val_ds is not None:
            val_loader = DataLoader(val_ds, batch_size=max(1, min(cfg.batch_size, len(val_ds))), shuffle=False, num_workers=cfg.num_workers)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))

        for epoch in range(cfg.epochs):
            student.train()
            total = 0.0
            total_pseudo = 0.0
            total_cons = 0.0
            n = 0
            pbar = tqdm(train_loader, desc=f"round {round_id+1} epoch {epoch+1}/{cfg.epochs}")
            for batch in pbar:
                x = batch["image"].to(device, non_blocking=True)
                xw = batch["weak_image"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True)
                bw = batch["boundary_weight"].to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
                    logits = student(x)
                    loss_pseudo = pseudo_supervised_loss(logits, y, boundary_weight=bw, boundary_lambda=cfg.boundary_weight)
                    if cfg.consistency_weight > 0:
                        with torch.no_grad():
                            t_logits = teacher(xw)
                        loss_cons = teacher_student_consistency(logits, t_logits, cfg.consistency_threshold)
                    else:
                        loss_cons = torch.zeros((), device=device)
                    loss = loss_pseudo + cfg.consistency_weight * loss_cons
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                update_ema(student, teacher, cfg.ema_decay)
                total += float(loss.detach().cpu())
                total_pseudo += float(loss_pseudo.detach().cpu())
                total_cons += float(loss_cons.detach().cpu())
                n += 1
                pbar.set_postfix(loss=total / max(1, n), pseudo=total_pseudo / max(1, n), cons=total_cons / max(1, n))
            scheduler.step()
            train_loss = total / max(1, n)
            if val_loader is not None:
                metric = evaluate_pseudo(teacher, val_loader, device)
                print(f"[train] pseudo-val Dice={metric:.4f}")
            else:
                metric = -train_loss
                print(f"[train] loss={train_loss:.4f}; pseudo-val disabled")
            log = {
                "round": round_id,
                "epoch": epoch,
                "loss": train_loss,
                "pseudo_loss": total_pseudo / max(1, n),
                "consistency_loss": total_cons / max(1, n),
                "pseudo_val_dice": metric,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")
            save_checkpoint(ckpt_dir / "last.pt", teacher, cfg, epoch=epoch, round_id=round_id, metric=metric)
            if metric > best_metric:
                best_metric = metric
                save_checkpoint(ckpt_dir / "best.pt", teacher, cfg, epoch=epoch, round_id=round_id, metric=metric)
        if cfg.refine_after_round and round_id < cfg.rounds - 1:
            refined_dir = workdir / f"pseudo_masks_round{round_id + 2}"
            current_pseudo = refine_pseudo_masks(
                teacher,
                image_paths,
                current_pseudo,
                refined_dir,
                alpha=cfg.refine_alpha,
                threshold=cfg.refine_threshold,
                device=cfg.device,
            )
    print(f"[train] Best checkpoint: {ckpt_dir / 'best.pt'}")
    return ckpt_dir / "best.pt"
