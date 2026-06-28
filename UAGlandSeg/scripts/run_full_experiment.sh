#!/usr/bin/env bash
set -euo pipefail

TRAIN_IMAGES=${1:-"SSL_Segmentation/Semantic Segmentation/Data/train/thumb"}
TEST_IMAGES=${2:-"SSL_Segmentation/Semantic Segmentation/Data/test/thumb"}
TEST_MASKS=${3:-"SSL_Segmentation/Semantic Segmentation/Data/test/mask"}
WORKDIR=${4:-"runs/uaglandseg_formal"}
DINO_LOCAL_REPO=${5:-"dinov2-main"}

python -m uagland.cli pseudo \
  --images "$TRAIN_IMAGES" \
  --workdir "$WORKDIR" \
  --encoder dinov2 \
  --dinov2-local-repo "$DINO_LOCAL_REPO" \
  --dinov2-model dinov2_vits14 \
  --image-size 518 \
  --num-prototypes 8 \
  --foreground-prototypes 3 \
  --threshold otsu \
  --use-crf

python -m uagland.cli train \
  --images "$TRAIN_IMAGES" \
  --pseudo-dir "$WORKDIR/pseudo_masks" \
  --workdir "$WORKDIR" \
  --crop-size 512 \
  --epochs 80 \
  --rounds 2 \
  --batch-size 4 \
  --lr 1e-4 \
  --boundary-weight 0.2 \
  --consistency-weight 0.1 \
  --ema-decay 0.99 \
  --amp

python -m uagland.cli infer \
  --images "$TEST_IMAGES" \
  --ckpt "$WORKDIR/checkpoints/best.pt" \
  --out "$WORKDIR/test_predictions" \
  --threshold 0.5

python -m uagland.cli eval \
  --pred-dir "$WORKDIR/test_predictions/masks" \
  --gt-dir "$TEST_MASKS" \
  --out "$WORKDIR/evaluation"
