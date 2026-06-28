# UAGlandSeg：根据 WSSS-Gland 论文思路改造的无监督腺体分割实验代码

本代码**不使用 image-level label，也不使用 pixel-level mask 训练**。人工 mask 只允许用于最终 held-out test 评价。

原论文的核心路线是：

1. 自监督/领域适配的 DINOv2 特征编码器；
2. 注意力图生成伪标签；
3. CRF/阈值/融合后处理；
4. 用边界敏感损失训练最终分割网络。

这里将第 2 步从“image-level label 监督的 attention MIL 分类器”改成了**无标签 prototype attention**：

```text
unlabeled H&E image
    -> DINOv2 patch token feature
    -> unsupervised prototype clustering
    -> H&E tissue/stain/boundary prior selects gland-like prototypes
    -> prototype attention map
    -> stain-prior blending + threshold + optional DenseCRF + morphology
    -> pseudo mask
    -> U-Net + EMA teacher-student + boundary-aware loss
    -> final gland mask
```

## 1. 文件结构

```text
UAGlandSeg/
  requirements.txt
  README_UAGlandSeg.md
  configs/glas_unsup.yaml
  scripts/run_full_experiment.sh
  uagland/
    cli.py
    pseudo.py
    dino.py
    crf.py
    data.py
    models.py
    losses.py
    train.py
    infer.py
    metrics.py
    utils.py
```

## 2. 安装

建议把 `UAGlandSeg` 放在 `WSSS-Gland-Segmentation` 仓库根目录下：

```bash
cd WSSS-Gland-Segmentation
cp -r /path/to/UAGlandSeg .
cd UAGlandSeg
pip install -r requirements.txt
```

为了使用本仓库自带的 `dinov2-main`，运行时从 `WSSS-Gland-Segmentation` 根目录调用：

```bash
export PYTHONPATH=$PWD/UAGlandSeg:$PYTHONPATH
```

如果要启用 DenseCRF：

```bash
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

`pydensecrf` 没装时，代码会自动跳过 CRF，仍可运行。

## 3. 推荐正式实验命令

从 `WSSS-Gland-Segmentation` 根目录执行：

```bash
export PYTHONPATH=$PWD/UAGlandSeg:$PYTHONPATH
bash UAGlandSeg/scripts/run_full_experiment.sh \
  "SSL_Segmentation/Semantic Segmentation/Data/train/thumb" \
  "SSL_Segmentation/Semantic Segmentation/Data/test/thumb" \
  "SSL_Segmentation/Semantic Segmentation/Data/test/mask" \
  "runs/uaglandseg_formal" \
  "dinov2-main"
```

如果你的数据目录不同：

```bash
bash UAGlandSeg/scripts/run_full_experiment.sh TRAIN_IMAGE_DIR TEST_IMAGE_DIR TEST_MASK_DIR runs/uaglandseg_formal dinov2-main
```

## 4. 分阶段运行

### 4.1 生成无监督伪标签

```bash
python -m uagland.cli pseudo \
  --images "SSL_Segmentation/Semantic Segmentation/Data/train/thumb" \
  --workdir runs/uaglandseg_formal \
  --encoder dinov2 \
  --dinov2-local-repo dinov2-main \
  --dinov2-model dinov2_vits14 \
  --image-size 518 \
  --num-prototypes 8 \
  --foreground-prototypes 3 \
  --threshold otsu \
  --use-crf
```

输出：

```text
runs/uaglandseg_formal/prototype_model.joblib
runs/uaglandseg_formal/prototype_scores.csv
runs/uaglandseg_formal/pseudo_masks/
runs/uaglandseg_formal/attention_maps/
runs/uaglandseg_formal/stain_priors/
runs/uaglandseg_formal/pseudo_overlays/
runs/uaglandseg_formal/pseudo_summary.csv
```

先查看 `pseudo_overlays/`。如果伪标签明显不合理，先调：

```bash
--num-prototypes 10
--foreground-prototypes 2 或 4
--blend-gamma 0.65 到 0.85
--threshold q70 / q75 / otsu
--min-area 100
--hole-area 800
```

### 4.2 用伪标签训练分割网络

```bash
python -m uagland.cli train \
  --images "SSL_Segmentation/Semantic Segmentation/Data/train/thumb" \
  --pseudo-dir runs/uaglandseg_formal/pseudo_masks \
  --workdir runs/uaglandseg_formal \
  --crop-size 512 \
  --epochs 80 \
  --rounds 2 \
  --batch-size 4 \
  --lr 1e-4 \
  --boundary-weight 0.2 \
  --consistency-weight 0.1 \
  --ema-decay 0.99 \
  --amp
```

输出：

```text
runs/uaglandseg_formal/checkpoints/best.pt
runs/uaglandseg_formal/checkpoints/last.pt
runs/uaglandseg_formal/train_log.jsonl
runs/uaglandseg_formal/pseudo_masks_round2/
```

### 4.3 推理

```bash
python -m uagland.cli infer \
  --images "SSL_Segmentation/Semantic Segmentation/Data/test/thumb" \
  --ckpt runs/uaglandseg_formal/checkpoints/best.pt \
  --out runs/uaglandseg_formal/test_predictions \
  --threshold 0.5
```

输出：

```text
runs/uaglandseg_formal/test_predictions/masks/
runs/uaglandseg_formal/test_predictions/probabilities/
runs/uaglandseg_formal/test_predictions/overlays/
```

### 4.4 用 held-out GT mask 评价

```bash
python -m uagland.cli eval \
  --pred-dir runs/uaglandseg_formal/test_predictions/masks \
  --gt-dir "SSL_Segmentation/Semantic Segmentation/Data/test/mask" \
  --out runs/uaglandseg_formal/evaluation
```

输出：

```text
runs/uaglandseg_formal/evaluation/per_image_metrics.csv
runs/uaglandseg_formal/evaluation/summary_metrics.csv
```

## 5. Debug 模式

如果 DINOv2 权重或网络环境暂时不可用，可用 handcrafted encoder 检查数据路径和后续训练流程：

```bash
python -m uagland.cli pseudo \
  --images "SSL_Segmentation/Semantic Segmentation/Data/train/thumb" \
  --workdir runs/uaglandseg_debug \
  --encoder handcrafted \
  --image-size 518
```

注意：`handcrafted` 只能作为代码/debug ablation，不建议作为论文主方法。

## 6. 论文实验规范建议

### 训练阶段

- 不读取 `train/mask`。
- 不读取 image-level benign/malignant label。
- 仅使用 `train/thumb` 图像生成伪标签。

### 模型选择

- `pseudo_val_dice` 只表示拟合伪标签的程度，不等于真实 Dice。
- 正式论文应固定 seed，并报告 3 次以上重复实验的均值和标准差。

### 测试阶段

- 只在最终测试阶段使用 GT mask。
- 至少报告：Dice、IoU、Precision、Recall、HD95、Object F1、Object Dice。

### 消融实验

建议至少包含：

```text
A1: handcrafted pseudo labels
A2: off-the-shelf DINOv2 prototype attention
A3: DINOv2 prototype attention + stain-prior blending
A4: A3 + CRF
A5: A4 + boundary-aware loss
A6: A5 + EMA teacher-student self-training
```

## 7. 方法名称建议

论文中可以命名为：

```text
UAGlandSeg: Unsupervised Attention-Prototype Learning for Colorectal Gland Segmentation
```

方法描述可以写成：

```text
We reformulate the weakly supervised attention-based pseudo-labeling pipeline as a fully unsupervised framework. Instead of training the attention module with image-level labels, we estimate gland-related prototype attention maps by clustering self-supervised DINOv2 patch tokens and selecting gland-like prototypes using label-free histomorphological priors. The generated attention maps are fused with stain and boundary priors, refined by thresholding, morphology, and optional DenseCRF, and then used to train a segmentation network with a boundary-aware objective and EMA teacher-student consistency.
```
