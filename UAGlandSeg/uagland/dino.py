from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .utils import edge_score, hematoxylin_like_score, normalize01, tissue_mask_rgb


def resize_feature_map(arr: np.ndarray, size: tuple[int, int], interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Resize HWC feature maps with any channel count."""
    if arr.ndim == 2:
        return cv2.resize(arr, size, interpolation=interpolation)
    channels = [cv2.resize(arr[..., c], size, interpolation=interpolation) for c in range(arr.shape[-1])]
    return np.stack(channels, axis=-1)


@dataclass
class DinoConfig:
    model_name: str = "dinov2_vits14"
    local_repo: Optional[str] = None
    weights: Optional[str] = None
    pretrained: bool = True
    image_size: int = 518
    device: str = "cuda"


class DinoFeatureExtractor:
    """Small wrapper around DINOv2 torch.hub models.

    It supports both a local DINOv2 repository (`--dinov2-local-repo dinov2-main`)
    and the public torch.hub repository. The local path is preferable for formal
    experiments because it avoids silent code changes after the experiment date.
    """

    def __init__(self, cfg: DinoConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device.startswith("cuda") else "cpu")
        self.model = self._load_model()
        self.model.eval().to(self.device)
        self.patch_size = 14
        self.transform = transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _load_model(self):
        if self.cfg.local_repo:
            repo = str(Path(self.cfg.local_repo).resolve())
            model = torch.hub.load(repo, self.cfg.model_name, source="local", pretrained=self.cfg.pretrained)
        else:
            model = torch.hub.load("facebookresearch/dinov2", self.cfg.model_name, pretrained=self.cfg.pretrained)
        if self.cfg.weights:
            state = torch.load(self.cfg.weights, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            # remove common prefixes
            cleaned = {}
            for k, v in state.items():
                nk = k.replace("module.", "").replace("backbone.", "")
                cleaned[nk] = v
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            if missing:
                print(f"[DINO] Missing keys when loading weights: {len(missing)}")
            if unexpected:
                print(f"[DINO] Unexpected keys when loading weights: {len(unexpected)}")
        return model

    @torch.no_grad()
    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return patch token feature map with shape [Ht, Wt, C]."""
        pil = Image.fromarray(image_rgb).convert("RGB")
        x = self.transform(pil).unsqueeze(0).to(self.device)
        tokens = None
        if hasattr(self.model, "get_intermediate_layers"):
            try:
                out = self.model.get_intermediate_layers(x, n=1, reshape=True, return_class_token=False)
                tokens = out[0]
            except TypeError:
                out = self.model.get_intermediate_layers(x, n=1)
                tokens = out[0]
        if tokens is None:
            out = self.model.forward_features(x)
            if isinstance(out, dict):
                tokens = out.get("x_norm_patchtokens", None)
                if tokens is None:
                    tokens = out.get("patch_tokens", None)
                if tokens is None:
                    for v in out.values():
                        if torch.is_tensor(v) and v.ndim in (3, 4):
                            tokens = v
                            break
            elif torch.is_tensor(out):
                tokens = out
        if tokens is None:
            raise RuntimeError("Could not extract DINOv2 patch tokens from the model output.")
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if tokens.ndim == 4:
            # B, C, H, W
            fmap = tokens[0].permute(1, 2, 0).contiguous()
        elif tokens.ndim == 3:
            # B, N, C; remove CLS token if present and infer grid size.
            t = tokens[0]
            n = t.shape[0]
            side = int(round(n ** 0.5))
            if side * side != n and n > 1:
                t = t[1:]
                n = t.shape[0]
                side = int(round(n ** 0.5))
            if side * side != n:
                raise RuntimeError(f"Patch token count {n} is not a square grid.")
            fmap = t.reshape(side, side, t.shape[-1])
        else:
            raise RuntimeError(f"Unsupported DINO token shape: {tuple(tokens.shape)}")
        return fmap.detach().float().cpu().numpy()


class HandcraftedFeatureExtractor:
    """Deterministic feature extractor used for debugging or ablation.

    This is not the recommended encoder for the final paper experiment. It keeps
    the code executable on machines where DINOv2 weights are not yet available.
    """

    def __init__(self, image_size: int = 518, grid_size: int = 37):
        self.image_size = image_size
        self.grid_size = grid_size

    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        img = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        hema = hematoxylin_like_score(img)[..., None]
        edge = edge_score(img)[..., None]
        tissue = tissue_mask_rgb(img).astype(np.float32)[..., None]
        feats = np.concatenate([lab, hsv, hema, edge, tissue], axis=-1)
        out = resize_feature_map(feats, (self.grid_size, self.grid_size), interpolation=cv2.INTER_AREA)
        return out.astype(np.float32)


def build_extractor(
    encoder: str,
    device: str = "cuda",
    image_size: int = 518,
    dinov2_model: str = "dinov2_vits14",
    dinov2_local_repo: str | None = None,
    dinov2_weights: str | None = None,
    pretrained: bool = True,
):
    encoder = encoder.lower()
    if encoder == "dinov2":
        return DinoFeatureExtractor(
            DinoConfig(
                model_name=dinov2_model,
                local_repo=dinov2_local_repo,
                weights=dinov2_weights,
                pretrained=pretrained,
                image_size=image_size,
                device=device,
            )
        )
    if encoder == "handcrafted":
        return HandcraftedFeatureExtractor(image_size=image_size)
    raise ValueError(f"Unsupported encoder: {encoder}")


def patch_aux_features(image_rgb: np.ndarray, grid_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return auxiliary H&E priors per patch and tissue occupancy per patch.

    Output:
      aux: [Ht, Wt, 7] = L, A, B, S, V, hematoxylin-like, edge
      tissue: [Ht, Wt]
    """
    h, w = grid_hw
    img = image_rgb
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    hema = hematoxylin_like_score(img)[..., None]
    edge = edge_score(img)[..., None]
    aux_full = np.concatenate([lab, hsv[..., 1:3], hema, edge], axis=-1)
    aux = resize_feature_map(aux_full, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    tissue_full = tissue_mask_rgb(img).astype(np.float32)
    tissue = cv2.resize(tissue_full, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    return aux, tissue


def l2_normalize_features(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)
