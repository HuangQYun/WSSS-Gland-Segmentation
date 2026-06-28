# UAGlandSeg 正式实验方案

## 1. 研究问题重定义

原弱监督问题：

```text
训练输入：图像 x_i + 图像级标签 y_i
输出：腺体像素 mask
```

无监督改造后：

```text
训练输入：仅图像 x_i
输出：腺体像素 mask
```

训练阶段严禁使用：

```text
train/mask
val/mask
benign/malignant image-level label
任何人工点、框、scribble
```

最终测试阶段允许使用 GT mask 计算指标。

## 2. 方法概述

UAGlandSeg 包含三个阶段：

### Stage 1：自监督特征编码

默认使用 DINOv2 patch token 作为无标签视觉表征：

```text
F_i = E(x_i), F_i ∈ R^{h×w×d}
```

如果已有病理领域自监督 fine-tuned DINOv2 权重，可通过：

```bash
--dinov2-weights /path/to/fine_tuned_dinov2.pth
```

加载。

### Stage 2：无标签 prototype attention 伪标签

将所有训练图像的 patch token 与 H&E 辅助先验拼接：

```text
z_p = concat(norm(DINO_token_p), α · histology_prior_p)
```

其中 `histology_prior` 包括：

```text
Lab intensity/chroma
HSV saturation/value
hematoxylin-like score
edge score
tissue occupancy
```

然后使用 MiniBatch K-means 得到 K 个 prototype。每个 prototype 的腺体样得分为：

```text
score_k = 0.35·hematoxylin + 0.20·saturation + 0.20·edge
          + 0.20·tissue + 0.05·darkness - 0.15·near_white_penalty
```

选择得分最高的若干 prototype 作为 gland-like prototypes。对每个 patch 计算其属于 gland-like prototypes 的概率和，得到 prototype attention map。

最终伪标签：

```text
A_proto = prototype_attention(x)
A_stain = stain/boundary prior(x)
A_final = γ·A_proto + (1-γ)·A_stain
M_pseudo = Morphology(CRF(Threshold(A_final)))
```

### Stage 3：边界敏感 teacher-student 分割训练

分割网络使用 U-Net decoder。训练损失：

```text
L = L_BCE_boundary + L_Dice + λ_consistency · L_teacher_student
```

其中：

```text
L_BCE_boundary: 用伪标签边界距离图加权
L_Dice: 缓解前景/背景不平衡
L_teacher_student: EMA teacher 对强弱增强输出的一致性约束
```

多轮训练时，EMA teacher 会生成下一轮 refined pseudo masks：

```text
M_{r+1} = threshold(α·P_teacher + (1-α)·M_r)
```

## 3. 推荐实验表

| 实验编号 | DINO 特征 | stain prior | CRF | boundary loss | EMA self-training |
|---|---|---|---|---|---|
| A1 | 否，handcrafted | 是 | 否 | 否 | 否 |
| A2 | DINOv2 off-the-shelf | 否 | 否 | 否 | 否 |
| A3 | DINOv2 off-the-shelf | 是 | 否 | 否 | 否 |
| A4 | DINOv2 off-the-shelf | 是 | 是 | 否 | 否 |
| A5 | DINOv2 off-the-shelf | 是 | 是 | 是 | 否 |
| A6 | DINOv2 off-the-shelf | 是 | 是 | 是 | 是 |
| A7 | DINOv2 histology fine-tuned | 是 | 是 | 是 | 是 |

A6 可作为主实验；A7 如果你有足够无标注 WSI/tile 和计算资源，可作为更强主实验。

## 4. 报告指标

建议报告：

```text
Dice
IoU
Precision
Recall
HD95
Object F1
Object Dice
```

并报告：

```text
mean ± std over 3 seeds
per-image CSV
summary CSV
qualitative overlays
pseudo-label quality analysis if train GT is only used for analysis, not selection
```

## 5. 论文中需要写清楚的限制

1. 该方法训练阶段完全无人工标签。
2. prototype selection 使用的是 stain/tissue/boundary 先验，不是人工标签。
3. DINOv2 权重是否经过领域自监督 fine-tuning 要单独报告。
4. 如果使用任何 train mask 调参，方法就不能再称为 pure unsupervised。
5. `pseudo_val_dice` 不是 GT Dice，只用于训练稳定性监控。
