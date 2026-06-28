from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi


def soft_dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def boundary_weight_map(mask: np.ndarray, boundary_weight: float = 0.2, sigma: float = 5.0) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.ones_like(m, dtype=np.float32)
    er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    dl = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=1)
    boundary = (dl != er)
    dist = ndi.distance_transform_edt(~boundary)
    w = 1.0 + boundary_weight * np.exp(-dist / max(sigma, 1e-6))
    return w.astype(np.float32)


def pseudo_supervised_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_weight: torch.Tensor | None = None,
    boundary_lambda: float = 0.2,
) -> torch.Tensor:
    if boundary_weight is None:
        bce = F.binary_cross_entropy_with_logits(logits, target)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, target, weight=boundary_weight)
    dice = soft_dice_loss_from_logits(logits, target)
    return bce + dice


def teacher_student_consistency(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    confidence_threshold: float = 0.70,
) -> torch.Tensor:
    with torch.no_grad():
        t = torch.sigmoid(teacher_logits)
        conf = torch.abs(t - 0.5) * 2.0
        mask = (conf >= confidence_threshold).float()
    s = torch.sigmoid(student_logits)
    if mask.sum() < 1:
        return torch.zeros((), device=student_logits.device, dtype=student_logits.dtype)
    return (((s - t) ** 2) * mask).sum() / (mask.sum() + 1e-6)
