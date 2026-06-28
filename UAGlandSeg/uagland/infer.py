from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from .models import build_model
from .utils import ensure_dir, list_images, pad_to_multiple_tensor, read_image_rgb, rgb_to_tensor, save_gray, save_mask, save_overlay, stem_key, unpad_tensor


@torch.no_grad()
def predict_array(model: torch.nn.Module, image_rgb: np.ndarray, device: str = "cuda") -> np.ndarray:
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    x = rgb_to_tensor(image_rgb).unsqueeze(0).to(dev)
    x, pad_hw = pad_to_multiple_tensor(x, multiple=16, value=0.0)
    logits = model(x)
    logits = unpad_tensor(logits, pad_hw)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return prob.astype(np.float32)


def load_checkpoint(ckpt: str | Path, device: str = "cuda") -> torch.nn.Module:
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    obj = torch.load(ckpt, map_location=dev)
    cfg = obj.get("model_config", {}) if isinstance(obj, dict) else {}
    model = build_model(cfg.get("name", "unet"), base=int(cfg.get("base", 32)), dropout=float(cfg.get("dropout", 0.05))).to(dev)
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def run_inference(
    images: str | Path,
    ckpt: str | Path,
    out: str | Path,
    threshold: float = 0.5,
    device: str = "cuda",
    recursive: bool = False,
) -> Path:
    out = ensure_dir(out)
    mask_dir = ensure_dir(out / "masks")
    prob_dir = ensure_dir(out / "probabilities")
    overlay_dir = ensure_dir(out / "overlays")
    model = load_checkpoint(ckpt, device=device)
    image_paths = list_images(images, recursive=recursive)
    rows = []
    for p in tqdm(image_paths, desc="inference"):
        img = read_image_rgb(p)
        prob = predict_array(model, img, device=device)
        mask = (prob >= threshold).astype(np.uint8)
        key = stem_key(p)
        save_mask(mask_dir / f"{key}.png", mask)
        save_gray(prob_dir / f"{key}.png", prob)
        save_overlay(img, mask, overlay_dir / f"{key}.png")
        rows.append({"image": str(p), "name": key, "foreground_ratio": float(mask.mean())})
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(out / "inference_summary.csv", index=False)
    except Exception:
        pass
    print(f"[infer] Masks saved to: {mask_dir}")
    return mask_dir
