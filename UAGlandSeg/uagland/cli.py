from __future__ import annotations

import argparse
from pathlib import Path

from .infer import run_inference
from .metrics import run_evaluation
from .pseudo import PseudoConfig, run_pseudo_generation
from .train import TrainConfig, run_training


def add_pseudo_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--images", required=True, help="Directory containing unlabeled training images.")
    p.add_argument("--workdir", required=True, help="Experiment output directory.")
    p.add_argument("--encoder", default="dinov2", choices=["dinov2", "handcrafted"])
    p.add_argument("--dinov2-model", default="dinov2_vits14")
    p.add_argument("--dinov2-local-repo", default=None, help="Path to local DINOv2 repo, e.g. dinov2-main.")
    p.add_argument("--dinov2-weights", default=None, help="Optional fine-tuned DINOv2 state dict.")
    p.add_argument("--no-pretrained", action="store_true", help="Instantiate DINOv2 without pretrained hub weights.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--num-prototypes", type=int, default=8)
    p.add_argument("--foreground-prototypes", type=int, default=3)
    p.add_argument("--max-tokens-per-image", type=int, default=512)
    p.add_argument("--aux-weight", type=float, default=0.30)
    p.add_argument("--prototype-temperature", type=float, default=1.0)
    p.add_argument("--blend-gamma", type=float, default=0.75)
    p.add_argument("--threshold", default="otsu", help="otsu, yen, q70, q80, or numeric string.")
    p.add_argument("--use-crf", action="store_true")
    p.add_argument("--min-area", type=int, default=80)
    p.add_argument("--hole-area", type=int, default=512)
    p.add_argument("--closing-radius", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--recursive", action="store_true")


def add_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--images", required=True)
    p.add_argument("--pseudo-dir", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--model", default="unet")
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--crop-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--boundary-weight", type=float, default=0.2)
    p.add_argument("--consistency-weight", type=float, default=0.1)
    p.add_argument("--consistency-threshold", type=float, default=0.70)
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--no-refine-after-round", action="store_true")
    p.add_argument("--refine-alpha", type=float, default=0.65)
    p.add_argument("--refine-threshold", type=float, default=0.5)


def add_infer_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--images", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--recursive", action="store_true")


def add_eval_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--object-iou-thr", type=float, default=0.5)


def parse_numeric_threshold(x: str):
    try:
        return float(x)
    except ValueError:
        return x


def main() -> None:
    parser = argparse.ArgumentParser(
        description="UAGlandSeg: unsupervised attention-prototype colorectal gland segmentation."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pseudo = sub.add_parser("pseudo", help="Generate unsupervised pseudo masks from unlabeled images.")
    add_pseudo_args(p_pseudo)

    p_train = sub.add_parser("train", help="Train a segmentation network on pseudo masks with boundary-aware and consistency losses.")
    add_train_args(p_train)

    p_infer = sub.add_parser("infer", help="Run inference on images.")
    add_infer_args(p_infer)

    p_eval = sub.add_parser("eval", help="Evaluate prediction masks against held-out ground truth masks.")
    add_eval_args(p_eval)

    p_all = sub.add_parser("all", help="Run pseudo-label generation, training, inference, and optional evaluation.")
    add_pseudo_args(p_all)
    p_all.add_argument("--test-images", default=None)
    p_all.add_argument("--test-masks", default=None)
    p_all.add_argument("--infer-out", default=None)
    # training overrides for all
    p_all.add_argument("--crop-size", type=int, default=512)
    p_all.add_argument("--epochs", type=int, default=80)
    p_all.add_argument("--rounds", type=int, default=1)
    p_all.add_argument("--batch-size", type=int, default=4)
    p_all.add_argument("--lr", type=float, default=1e-4)
    p_all.add_argument("--boundary-weight", type=float, default=0.2)
    p_all.add_argument("--consistency-weight", type=float, default=0.1)
    p_all.add_argument("--ema-decay", type=float, default=0.99)
    p_all.add_argument("--num-workers", type=int, default=2)
    p_all.add_argument("--amp", action="store_true")

    args = parser.parse_args()
    if args.command == "pseudo":
        cfg = PseudoConfig(
            images=args.images,
            workdir=args.workdir,
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
            threshold=parse_numeric_threshold(args.threshold),
            use_crf=args.use_crf,
            min_area=args.min_area,
            hole_area=args.hole_area,
            closing_radius=args.closing_radius,
            seed=args.seed,
            recursive=args.recursive,
        )
        run_pseudo_generation(cfg)
    elif args.command == "train":
        cfg = TrainConfig(
            images=args.images,
            pseudo_dir=args.pseudo_dir,
            workdir=args.workdir,
            model=args.model,
            base_channels=args.base_channels,
            dropout=args.dropout,
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
            num_workers=args.num_workers,
            amp=args.amp,
            device=args.device,
            seed=args.seed,
            val_fraction=args.val_fraction,
            recursive=args.recursive,
            refine_after_round=not args.no_refine_after_round,
            refine_alpha=args.refine_alpha,
            refine_threshold=args.refine_threshold,
        )
        run_training(cfg)
    elif args.command == "infer":
        run_inference(args.images, args.ckpt, args.out, threshold=args.threshold, device=args.device, recursive=args.recursive)
    elif args.command == "eval":
        run_evaluation(args.pred_dir, args.gt_dir, args.out, iou_thr=args.object_iou_thr)
    elif args.command == "all":
        pseudo_cfg = PseudoConfig(
            images=args.images,
            workdir=args.workdir,
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
            threshold=parse_numeric_threshold(args.threshold),
            use_crf=args.use_crf,
            min_area=args.min_area,
            hole_area=args.hole_area,
            closing_radius=args.closing_radius,
            seed=args.seed,
            recursive=args.recursive,
        )
        pseudo_dir = run_pseudo_generation(pseudo_cfg)
        train_cfg = TrainConfig(
            images=args.images,
            pseudo_dir=str(pseudo_dir),
            workdir=args.workdir,
            crop_size=args.crop_size,
            epochs=args.epochs,
            rounds=args.rounds,
            batch_size=args.batch_size,
            lr=args.lr,
            boundary_weight=args.boundary_weight,
            consistency_weight=args.consistency_weight,
            ema_decay=args.ema_decay,
            num_workers=args.num_workers,
            amp=args.amp,
            device=args.device,
            seed=args.seed,
            recursive=args.recursive,
        )
        best = run_training(train_cfg)
        if args.test_images:
            infer_out = args.infer_out or str(Path(args.workdir) / "test_predictions")
            pred_dir = run_inference(args.test_images, best, infer_out, device=args.device, recursive=args.recursive)
            if args.test_masks:
                run_evaluation(pred_dir, args.test_masks, Path(args.workdir) / "evaluation")


if __name__ == "__main__":
    main()
