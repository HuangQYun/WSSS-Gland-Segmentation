from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure
from tqdm import tqdm

from .utils import IMG_EXTENSIONS, ensure_dir, match_mask_path, read_mask_any, stem_key


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    tn = float(np.logical_and(~p, ~g).sum())
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    specificity = (tn + 1e-6) / (tn + fp + 1e-6)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall, "specificity": specificity}


def _surface(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if m.sum() == 0:
        return m
    eroded = ndi.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)
    return np.logical_xor(m, eroded)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    if p.sum() == 0 and g.sum() == 0:
        return 0.0
    if p.sum() == 0 or g.sum() == 0:
        return float("inf")
    ps = _surface(p)
    gs = _surface(g)
    if ps.sum() == 0 or gs.sum() == 0:
        return float("inf")
    dt_g = ndi.distance_transform_edt(~gs)
    dt_p = ndi.distance_transform_edt(~ps)
    d1 = dt_g[ps]
    d2 = dt_p[gs]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def _iou_matrix(pred_lab: np.ndarray, gt_lab: np.ndarray) -> np.ndarray:
    pred_ids = np.arange(1, int(pred_lab.max()) + 1)
    gt_ids = np.arange(1, int(gt_lab.max()) + 1)
    mat = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.float32)
    pred_masks = [(pred_lab == i) for i in pred_ids]
    gt_masks = [(gt_lab == j) for j in gt_ids]
    for i, pm in enumerate(pred_masks):
        for j, gm in enumerate(gt_masks):
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            mat[i, j] = inter / union if union > 0 else 0.0
    return mat


def object_metrics(pred: np.ndarray, gt: np.ndarray, iou_thr: float = 0.5) -> Dict[str, float]:
    pl = measure.label(pred.astype(bool), connectivity=2)
    gl = measure.label(gt.astype(bool), connectivity=2)
    n_p = int(pl.max())
    n_g = int(gl.max())
    if n_p == 0 and n_g == 0:
        return {"object_f1": 1.0, "object_precision": 1.0, "object_recall": 1.0, "object_dice": 1.0, "n_pred": 0, "n_gt": 0}
    if n_p == 0 or n_g == 0:
        return {"object_f1": 0.0, "object_precision": 0.0, "object_recall": 0.0, "object_dice": 0.0, "n_pred": n_p, "n_gt": n_g}
    mat = _iou_matrix(pl, gl)
    # Greedy matching is enough for reporting; official challenge scripts may differ.
    matches = []
    used_p, used_g = set(), set()
    while True:
        i, j = np.unravel_index(int(np.argmax(mat)), mat.shape)
        val = float(mat[i, j])
        if val < iou_thr:
            break
        if i not in used_p and j not in used_g:
            matches.append((i + 1, j + 1, val))
            used_p.add(i)
            used_g.add(j)
        mat[i, :] = -1
        mat[:, j] = -1
    tp = len(matches)
    precision = tp / max(1, n_p)
    recall = tp / max(1, n_g)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)
    dices = []
    for pi, gj, _ in matches:
        pm = pl == pi
        gm = gl == gj
        inter = np.logical_and(pm, gm).sum()
        d = (2 * inter + 1e-6) / (pm.sum() + gm.sum() + 1e-6)
        dices.append(float(d))
    obj_dice = float(np.mean(dices)) if dices else 0.0
    return {
        "object_f1": float(f1),
        "object_precision": float(precision),
        "object_recall": float(recall),
        "object_dice": obj_dice,
        "n_pred": n_p,
        "n_gt": n_g,
    }


def list_prediction_masks(pred_dir: str | Path) -> List[Path]:
    p = Path(pred_dir)
    files = [x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMG_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"No prediction masks found in {p}")
    return sorted(files)


def run_evaluation(pred_dir: str | Path, gt_dir: str | Path, out: str | Path, iou_thr: float = 0.5) -> Path:
    out = ensure_dir(out)
    rows = []
    for p in tqdm(list_prediction_masks(pred_dir), desc="evaluation"):
        gt_path = match_mask_path(gt_dir, p)
        pred = read_mask_any(p)
        gt = read_mask_any(gt_path)
        if pred.shape != gt.shape:
            import cv2

            pred = cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        row = {"name": stem_key(p), "pred": str(p), "gt": str(gt_path)}
        row.update(binary_metrics(pred, gt))
        row["hd95"] = hd95(pred, gt)
        row.update(object_metrics(pred, gt, iou_thr=iou_thr))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_image_metrics.csv", index=False)
    numeric = df.select_dtypes(include=[np.number])
    summary = numeric.replace([np.inf, -np.inf], np.nan).agg(["mean", "std", "median"]).T
    summary.to_csv(out / "summary_metrics.csv")
    print(summary)
    print(f"[eval] Metrics saved to: {out}")
    return out
