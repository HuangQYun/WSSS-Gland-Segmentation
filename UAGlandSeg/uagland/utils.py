from __future__ import annotations

import json
import os
import random
import warnings
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from skimage import color, filters, measure, morphology, segmentation

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def seed_everything(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(paths: str | Path | Sequence[str | Path], recursive: bool = False) -> List[Path]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out: List[Path] = []
    for item in paths:
        p = Path(item)
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
            out.append(p)
        elif p.is_dir():
            iterator = p.rglob("*") if recursive else p.iterdir()
            out.extend([x for x in iterator if x.is_file() and x.suffix.lower() in IMG_EXTENSIONS])
    out = sorted(set(out), key=lambda x: str(x))
    if not out:
        raise FileNotFoundError(f"No image files found in: {paths}")
    return out


def stem_key(path: str | Path) -> str:
    return Path(path).stem


def read_image_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def save_mask(path: str | Path, mask: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = (mask > 0).astype(np.uint8) * 255
    Image.fromarray(m).save(path)


def save_gray(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a = normalize01(arr)
    Image.fromarray((a * 255).astype(np.uint8)).save(path)


def normalize01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.nanmin(x))
    mx = float(np.nanmax(x))
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def rgb_to_tensor(image: np.ndarray) -> torch.Tensor:
    x = image.astype(np.float32) / 255.0
    return torch.from_numpy(x.transpose(2, 0, 1)).float()


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((mask > 0).astype(np.float32)[None, ...])


def pad_to_multiple_tensor(x: torch.Tensor, multiple: int = 16, value: float = 0.0) -> Tuple[torch.Tensor, Tuple[int, int]]:
    # x: BCHW
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), value=value)
    return x, (ph, pw)


def unpad_tensor(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    ph, pw = pad_hw
    if ph:
        x = x[..., :-ph, :]
    if pw:
        x = x[..., :-pw]
    return x


def tissue_mask_rgb(
    image: np.ndarray,
    saturation_min: int = 18,
    value_max: int = 245,
    min_area: int = 256,
) -> np.ndarray:
    """Return a conservative tissue mask for H&E RGB images."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    s = hsv[..., 1]
    v = hsv[..., 2]
    mask = (s >= saturation_min) & (v <= value_max)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        mask = morphology.remove_small_objects(mask.astype(bool), min_size=min_area)
        mask = morphology.remove_small_holes(mask, area_threshold=min_area)
        mask = morphology.binary_closing(mask, morphology.disk(3))
    return mask.astype(bool)


def hematoxylin_like_score(image: np.ndarray) -> np.ndarray:
    """Approximate nuclear/epithelial stain score without requiring stain deconvolution."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    L = lab[..., 0] / 255.0
    A = lab[..., 1] / 255.0
    B = lab[..., 2] / 255.0
    S = hsv[..., 1] / 255.0
    # darker, bluish-purple, saturated tissue tends to correspond to epithelium/nuclei.
    score = (1.0 - L) * 0.50 + S * 0.35 + np.maximum(0.0, A - B + 0.1) * 0.15
    return normalize01(score)


def edge_score(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return normalize01(np.sqrt(gx * gx + gy * gy))


def stain_prior(image: np.ndarray) -> np.ndarray:
    tissue = tissue_mask_rgb(image).astype(np.float32)
    hema = hematoxylin_like_score(image)
    edge = edge_score(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    prior = 0.55 * hema + 0.20 * edge + 0.20 * sat + 0.05 * tissue
    prior *= tissue
    return normalize01(prior)


def clean_binary_mask(
    mask: np.ndarray,
    min_area: int = 80,
    hole_area: int = 512,
    closing_radius: int = 4,
    opening_radius: int = 1,
) -> np.ndarray:
    m = mask.astype(bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        if opening_radius > 0:
            m = morphology.binary_opening(m, morphology.disk(opening_radius))
        if closing_radius > 0:
            m = morphology.binary_closing(m, morphology.disk(closing_radius))
        m = morphology.remove_small_objects(m, min_size=max(1, min_area))
        m = morphology.remove_small_holes(m, area_threshold=max(1, hole_area))
    return m.astype(np.uint8)


def threshold_score(score: np.ndarray, tissue: np.ndarray, threshold: str | float = "otsu", foreground_quantile: float = 0.72) -> np.ndarray:
    valid = tissue.astype(bool) & np.isfinite(score)
    if valid.sum() < 32:
        return np.zeros_like(score, dtype=np.uint8)
    values = score[valid]
    if isinstance(threshold, str):
        if threshold.lower() == "otsu":
            try:
                tau = float(filters.threshold_otsu(values))
            except Exception:
                tau = float(np.quantile(values, foreground_quantile))
        elif threshold.lower() == "yen":
            tau = float(filters.threshold_yen(values))
        elif threshold.lower().startswith("q"):
            tau = float(np.quantile(values, float(threshold[1:]) / 100.0))
        else:
            raise ValueError(f"Unknown threshold mode: {threshold}")
    else:
        tau = float(threshold)
    m = (score >= tau) & valid
    # Guard against degenerate masks. This is only a stabilizer, not supervised tuning.
    area_ratio = float(m.sum()) / float(max(1, valid.sum()))
    if area_ratio < 0.02:
        tau = float(np.quantile(values, 0.90))
        m = (score >= tau) & valid
    elif area_ratio > 0.85:
        tau = float(np.quantile(values, 0.55))
        m = (score >= tau) & valid
    return m.astype(np.uint8)


def save_overlay(image: np.ndarray, mask: np.ndarray, path: str | Path, alpha: float = 0.25) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = image.astype(np.float32) / 255.0
    m = mask.astype(bool)
    overlay = img.copy()
    overlay[m, 1] = np.maximum(overlay[m, 1], 0.95)
    overlay[m, 0] *= 1.0 - alpha
    overlay[m, 2] *= 1.0 - alpha
    bnd = segmentation.find_boundaries(m, mode="outer")
    overlay[bnd] = np.array([1.0, 0.0, 0.0])
    Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8)).save(path)


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_mask_any(path: str | Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def match_mask_path(mask_dir: str | Path, image_path: str | Path) -> Path:
    mask_dir = Path(mask_dir)
    key = stem_key(image_path)
    candidates = []
    for ext in IMG_EXTENSIONS:
        candidates.append(mask_dir / f"{key}{ext}")
        candidates.append(mask_dir / f"{key}_mask{ext}")
        candidates.append(mask_dir / f"{key}-mask{ext}")
    for c in candidates:
        if c.exists():
            return c
    all_masks = list(mask_dir.glob(f"{key}*"))
    all_masks = [p for p in all_masks if p.suffix.lower() in IMG_EXTENSIONS]
    if all_masks:
        return sorted(all_masks)[0]
    raise FileNotFoundError(f"No mask found for image {image_path} in {mask_dir}")


def connected_components(mask: np.ndarray) -> np.ndarray:
    return measure.label(mask.astype(bool), connectivity=2).astype(np.int32)
