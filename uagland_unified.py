"""Unified adapter for Gland_Segmentation/UAGlandSeg."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
UAGLAND_ROOT = ROOT / 'Gland_Segmentation' / 'UAGlandSeg'
DATA_ROOT = ROOT / 'data'

if str(UAGLAND_ROOT) not in sys.path:
    sys.path.insert(0, str(UAGLAND_ROOT))

from uagland.infer import run_inference  # noqa: E402
from uagland.metrics import binary_metrics, run_evaluation  # noqa: E402
from uagland.pseudo import PseudoConfig, run_pseudo_generation  # noqa: E402
from uagland.train import TrainConfig, run_training  # noqa: E402
from uagland.utils import ensure_dir, list_images, match_mask_path, read_mask_any  # noqa: E402

DATASET_DIRS = {
    'ccg': 'CCG',
    'glas': 'Glas',
    'pglandseg': 'PGlandSeg',
}


def dataset_dir(dataset: str) -> Path:
    key = dataset.lower()
    if key not in DATASET_DIRS:
        raise ValueError(f'Unknown dataset: {dataset}')
    path = DATA_ROOT / DATASET_DIRS[key]
    if not path.exists():
        raise FileNotFoundError(f'Missing dataset directory: {path}')
    return path


def workdir(dataset: str, method: str) -> Path:
    return UAGLAND_ROOT / 'train_out' / dataset.lower() / method.lower()


def read_split_ids(ds_dir: Path, split: str) -> list[str]:
    split_file = ds_dir / 'annotations' / f'{split}.txt'
    if not split_file.exists():
        raise FileNotFoundError(f'Missing split file: {split_file}')
    with split_file.open('r', encoding='utf-8') as handle:
        ids = [line.strip().split()[0] for line in handle if line.strip()]
    if not ids:
        raise RuntimeError(f'No samples found in split file: {split_file}')
    return [Path(sample_id).stem for sample_id in ids]


def source_paths(ds_dir: Path, split: str) -> tuple[list[Path], Path]:
    image_dir = ds_dir / 'images'
    label_dir = ds_dir / 'labels'
    if not image_dir.exists():
        raise FileNotFoundError(f'Missing image directory: {image_dir}')
    if not label_dir.exists():
        raise FileNotFoundError(f'Missing label directory: {label_dir}')

    images = []
    for sample_id in read_split_ids(ds_dir, split):
        image_path = image_dir / f'{sample_id}.png'
        if not image_path.exists():
            raise FileNotFoundError(f'Missing image listed in {split}.txt: {image_path}')
        images.append(image_path)
    return images, label_dir


def reset_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize_split(ds_dir: Path, split: str, out_dir: Path, labels: bool = False) -> Path:
    images, label_dir = source_paths(ds_dir, split)
    out = reset_dir(out_dir)
    for image_path in images:
        copy_or_link(image_path, out / image_path.name)
        if labels:
            label_path = match_mask_path(label_dir, image_path)
            copy_or_link(label_path, out / label_path.name)
    return out


def materialize_labels(ds_dir: Path, split: str, out_dir: Path) -> Path:
    images, label_dir = source_paths(ds_dir, split)
    out = reset_dir(out_dir)
    for image_path in images:
        label_path = match_mask_path(label_dir, image_path)
        copy_or_link(label_path, out / label_path.name)
    return out



def build_pseudo_config(args: argparse.Namespace, images: Path, workdir_path: Path, force_refit: bool) -> PseudoConfig:
    return PseudoConfig(
        images=str(images),
        workdir=str(workdir_path),
        encoder=args.encoder,
        dinov2_model=args.dinov2_model,
        dinov2_local_repo=args.dinov2_local_repo,
        dinov2_weights=args.dinov2_weights,
        pretrained=not args.no_pretrained,
        device=args.device,
        image_size=args.image_size,
        num_prototypes=args.num_prototypes,
        foreground_prototypes=args.foreground_prototypes,
        max_tokens_per_image=args.max_tokens_per_image,
        aux_weight=args.aux_weight,
        prototype_temperature=args.prototype_temperature,
        blend_gamma=args.blend_gamma,
        threshold=args.pseudo_threshold,
        use_crf=args.use_crf,
        min_area=args.min_area,
        hole_area=args.hole_area,
        closing_radius=args.closing_radius,
        seed=args.seed,
        recursive=False,
        force_refit_prototypes=force_refit,
    )


def train(args: argparse.Namespace) -> None:
    ds_dir = dataset_dir(args.dataset)
    out_root = workdir(args.dataset, args.method)
    train_images = materialize_split(ds_dir, 'train', out_root / 'splits' / 'train_images')
    val_images = materialize_split(ds_dir, 'val', out_root / 'splits' / 'val_images')

    pseudo_cfg = build_pseudo_config(args, train_images, out_root / 'pseudo', args.force_refit_prototypes)
    train_pseudo_dir = run_pseudo_generation(pseudo_cfg)

    val_pseudo_workdir = out_root / 'pseudo_val'
    val_pseudo_workdir.mkdir(parents=True, exist_ok=True)
    prototype_model = out_root / 'pseudo' / 'prototype_model.joblib'
    if prototype_model.exists():
        shutil.copy2(prototype_model, val_pseudo_workdir / 'prototype_model.joblib')
    val_pseudo_cfg = build_pseudo_config(args, val_images, val_pseudo_workdir, False)
    val_pseudo_dir = run_pseudo_generation(val_pseudo_cfg)

    combined_pseudo_dir = reset_dir(out_root / 'pseudo_train_val' / 'pseudo_masks')
    for pseudo_dir in (train_pseudo_dir, val_pseudo_dir):
        for mask_path in list_images(pseudo_dir):
            copy_or_link(mask_path, combined_pseudo_dir / mask_path.name)

    train_cfg = TrainConfig(
        images=str(train_images),
        pseudo_dir=str(combined_pseudo_dir),
        workdir=str(out_root),
        crop_size=args.crop_size,
        epochs=args.epochs,
        rounds=args.rounds,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        boundary_weight=args.boundary_weight,
        consistency_weight=args.consistency_weight,
        consistency_threshold=args.consistency_threshold,
        ema_decay=args.ema_decay,
        num_workers=args.workers,
        amp=args.amp,
        device=args.device,
        seed=args.seed,
        val_fraction=0.0,
        val_images=str(val_images),
        recursive=False,
        refine_after_round=not args.no_refine_after_round,
        refine_alpha=args.refine_alpha,
        refine_threshold=args.refine_threshold,
    )
    best = run_training(train_cfg)
    print(f'[uagland_unified] Best checkpoint: {best}')


def summarize_binary_metrics(pred_dir: Path, gt_dir: Path) -> dict[str, float]:
    totals = {'tp': 0.0, 'fp': 0.0, 'fn': 0.0, 'tn': 0.0}
    for pred_path in list_images(pred_dir):
        gt_path = match_mask_path(gt_dir, pred_path)
        pred = read_mask_any(pred_path)
        gt = read_mask_any(gt_path)
        if pred.shape != gt.shape:
            import cv2
            pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        p = pred.astype(bool)
        g = gt.astype(bool)
        totals['tp'] += float(np.logical_and(p, g).sum())
        totals['fp'] += float(np.logical_and(p, ~g).sum())
        totals['fn'] += float(np.logical_and(~p, g).sum())
        totals['tn'] += float(np.logical_and(~p, ~g).sum())
    total_pixels = sum(totals.values())
    if total_pixels == 0:
        raise RuntimeError(f'No prediction pixels found in {pred_dir}')
    return totals


def _pct(value: float) -> float:
    return value * 100.0


def print_unified_metrics(counts: dict[str, float]) -> None:
    tp = counts['tp']
    fp = counts['fp']
    fn = counts['fn']
    tn = counts['tn']
    eps = 1e-6
    gland_dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    gland_iou = (tp + eps) / (tp + fp + fn + eps)
    gland_recall = (tp + eps) / (tp + fn + eps)
    background_dice = (2 * tn + eps) / (2 * tn + fp + fn + eps)
    background_iou = (tn + eps) / (tn + fp + fn + eps)
    background_recall = (tn + eps) / (tn + fp + eps)
    accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    print(f'Dice class background is {_pct(background_dice):.2f}')
    print(f'IoU class background is {_pct(background_iou):.2f}')
    print(f'Recall class background is {_pct(background_recall):.2f}')
    print(f'Dice class gland is {_pct(gland_dice):.2f}')
    print(f'IoU class gland is {_pct(gland_iou):.2f}')
    print(f'Recall class gland is {_pct(gland_recall):.2f}')
    print(f'mDice is {_pct((background_dice + gland_dice) / 2.0):.2f}')
    print(f'mIoU is {_pct((background_iou + gland_iou) / 2.0):.2f}')
    print(f'mRecall is {_pct((background_recall + gland_recall) / 2.0):.2f}')
    print(f'Accuracy is {_pct(accuracy):.2f}')


def test(args: argparse.Namespace) -> None:
    ds_dir = dataset_dir(args.dataset)
    out_root = workdir(args.dataset, args.method)
    test_images = materialize_split(ds_dir, 'test', out_root / 'splits' / 'test_images')
    test_labels = materialize_labels(ds_dir, 'test', out_root / 'splits' / 'test_labels')
    ckpt = Path(args.checkpoint) if args.checkpoint else out_root / 'checkpoints' / 'best.pt'
    if not ckpt.exists():
        raise FileNotFoundError(f'Missing checkpoint: {ckpt}')

    vis_dir = Path(args.vis_dir) if args.vis_dir else out_root / 'test_predictions'
    pred_dir = run_inference(
        test_images,
        ckpt,
        vis_dir,
        threshold=args.threshold,
        device=args.device,
        recursive=False,
    )
    eval_dir = out_root / 'evaluation'
    run_evaluation(pred_dir, test_labels, eval_dir)
    metrics = summarize_binary_metrics(pred_dir, test_labels)
    print_unified_metrics(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Unified UAGlandSeg adapter')
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument('--method', default='uagland')
    parser.add_argument('--dataset', choices=sorted(DATASET_DIRS), required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', '--batch-size', dest='batch_size', type=int, default=4)
    parser.add_argument('--crop_size', '--crop-size', dest='crop_size', type=int, default=512)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--encoder', default='dinov2', choices=['dinov2', 'handcrafted'])
    parser.add_argument('--dinov2-model', default='dinov2_vits14')
    parser.add_argument('--dinov2-local-repo', default=None)
    parser.add_argument('--dinov2-weights', default=None)
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--image-size', dest='image_size', type=int, default=518)
    parser.add_argument('--num-prototypes', dest='num_prototypes', type=int, default=8)
    parser.add_argument('--foreground-prototypes', dest='foreground_prototypes', type=int, default=3)
    parser.add_argument('--max-tokens-per-image', dest='max_tokens_per_image', type=int, default=512)
    parser.add_argument('--aux-weight', dest='aux_weight', type=float, default=0.30)
    parser.add_argument('--prototype-temperature', dest='prototype_temperature', type=float, default=1.0)
    parser.add_argument('--blend-gamma', dest='blend_gamma', type=float, default=0.75)
    parser.add_argument('--pseudo-threshold', dest='pseudo_threshold', default='otsu')
    parser.add_argument('--use-crf', dest='use_crf', action='store_true')
    parser.add_argument('--min-area', dest='min_area', type=int, default=80)
    parser.add_argument('--hole-area', dest='hole_area', type=int, default=512)
    parser.add_argument('--closing-radius', dest='closing_radius', type=int, default=4)
    parser.add_argument('--force-refit-prototypes', dest='force_refit_prototypes', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '--weight-decay', dest='weight_decay', type=float, default=1e-5)
    parser.add_argument('--boundary_weight', '--boundary-weight', dest='boundary_weight', type=float, default=0.2)
    parser.add_argument('--consistency_weight', '--consistency-weight', dest='consistency_weight', type=float, default=0.1)
    parser.add_argument('--consistency_threshold', '--consistency-threshold', dest='consistency_threshold', type=float, default=0.60)
    parser.add_argument('--ema_decay', '--ema-decay', dest='ema_decay', type=float, default=0.99)
    parser.add_argument('--val_fraction', '--val-fraction', dest='val_fraction', type=float, default=0.10,
                        help='Kept for compatibility; unified training uses annotations/val.txt.')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no_refine_after_round', '--no-refine-after-round', dest='no_refine_after_round', action='store_true')
    parser.add_argument('--refine_alpha', '--refine-alpha', dest='refine_alpha', type=float, default=0.65)
    parser.add_argument('--refine_threshold', '--refine-threshold', dest='refine_threshold', type=float, default=0.5)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--vis_dir', default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser.parse_args()


if __name__ == '__main__':
    parsed = parse_args()
    if parsed.mode == 'train':
        train(parsed)
    else:
        test(parsed)
