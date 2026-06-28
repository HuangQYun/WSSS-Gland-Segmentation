from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .losses import boundary_weight_map
from .utils import list_images, match_mask_path, mask_to_tensor, read_image_rgb, read_mask_any, rgb_to_tensor, stem_key


def _pad_to_minimum(image: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    ph = max(0, size - h)
    pw = max(0, size - w)
    if ph or pw:
        top = ph // 2
        bottom = ph - top
        left = pw // 2
        right = pw - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=0)
    return image, mask


def _random_crop(image: np.ndarray, mask: np.ndarray, size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    image, mask = _pad_to_minimum(image, mask, size)
    h, w = image.shape[:2]
    if h == size and w == size:
        return image, mask
    y = int(rng.integers(0, h - size + 1))
    x = int(rng.integers(0, w - size + 1))
    return image[y : y + size, x : x + size], mask[y : y + size, x : x + size]


def _center_resize(image: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return image, mask


def _geometric_aug(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[::-1])
        mask = np.ascontiguousarray(mask[::-1])
    k = int(rng.integers(0, 4))
    if k:
        image = np.ascontiguousarray(np.rot90(image, k))
        mask = np.ascontiguousarray(np.rot90(mask, k))
    return image, mask


def _color_jitter(image: np.ndarray, rng: np.random.Generator, strength: float = 0.15) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    brightness = 1.0 + rng.uniform(-strength, strength)
    contrast = 1.0 + rng.uniform(-strength, strength)
    x = (x - 0.5) * contrast + 0.5
    x = x * brightness
    hsv = cv2.cvtColor(np.clip(x * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= 1.0 + rng.uniform(-strength, strength)
    hsv[..., 2] *= 1.0 + rng.uniform(-strength, strength)
    x = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _strong_aug(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = _color_jitter(image, rng, strength=0.25)
    if rng.random() < 0.35:
        k = int(rng.choice([3, 5]))
        x = cv2.GaussianBlur(x, (k, k), 0)
    if rng.random() < 0.35:
        noise = rng.normal(0, 5.0, size=x.shape).astype(np.float32)
        x = np.clip(x.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return x


class PseudoMaskDataset(Dataset):
    def __init__(
        self,
        images: str | Path | List[str | Path],
        pseudo_dir: str | Path,
        crop_size: int = 512,
        mode: str = "train",
        seed: int = 2026,
        recursive: bool = False,
        boundary_weight: float = 0.2,
    ):
        self.image_paths = list_images(images, recursive=recursive) if not isinstance(images, list) else [Path(p) for p in images]
        self.pseudo_dir = Path(pseudo_dir)
        self.crop_size = int(crop_size)
        self.mode = mode
        self.seed = seed
        self.boundary_weight = float(boundary_weight)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        p = self.image_paths[idx]
        img = read_image_rgb(p)
        mask_path = match_mask_path(self.pseudo_dir, p)
        mask = read_mask_any(mask_path)
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        return img, mask, stem_key(p)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed + idx + (0 if self.mode != "train" else torch.initial_seed() % 1000000))
        img, mask, name = self._load_pair(idx)
        if self.mode == "train":
            # Random scaling encourages robustness under pseudo-label noise.
            if rng.random() < 0.50:
                scale = float(rng.uniform(0.75, 1.25))
                new_w = max(32, int(img.shape[1] * scale))
                new_h = max(32, int(img.shape[0] * scale))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            img, mask = _random_crop(img, mask, self.crop_size, rng)
            img, mask = _geometric_aug(img, mask, rng)
            weak = _color_jitter(img, rng, strength=0.08)
            strong = _strong_aug(img, rng)
        else:
            img, mask = _center_resize(img, mask, self.crop_size)
            weak = img
            strong = img
        bw = boundary_weight_map(mask, boundary_weight=self.boundary_weight)
        return {
            "image": rgb_to_tensor(strong),
            "weak_image": rgb_to_tensor(weak),
            "mask": mask_to_tensor(mask),
            "boundary_weight": torch.from_numpy(bw[None, ...]).float(),
            "name": name,
        }
