from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .crf import dense_crf_refine
from .dino import build_extractor, l2_normalize_features, patch_aux_features
from .utils import (
    clean_binary_mask,
    edge_score,
    ensure_dir,
    hematoxylin_like_score,
    list_images,
    normalize01,
    read_image_rgb,
    save_gray,
    save_mask,
    save_overlay,
    stain_prior,
    stem_key,
    threshold_score,
    tissue_mask_rgb,
    write_json,
)


@dataclass
class PseudoConfig:
    images: str
    workdir: str
    encoder: str = "dinov2"
    dinov2_model: str = "dinov2_vits14"
    dinov2_local_repo: str | None = None
    dinov2_weights: str | None = None
    pretrained: bool = True
    device: str = "cuda"
    image_size: int = 518
    num_prototypes: int = 8
    foreground_prototypes: int = 3
    max_tokens_per_image: int = 512
    max_fit_images: int = 1000000
    aux_weight: float = 0.30
    prototype_temperature: float = 1.0
    blend_gamma: float = 0.75
    threshold: str = "otsu"
    use_crf: bool = False
    min_area: int = 80
    hole_area: int = 512
    closing_radius: int = 4
    seed: int = 2026
    recursive: bool = False


def _combined_features(token_map: np.ndarray, aux: np.ndarray, aux_weight: float) -> np.ndarray:
    dino = l2_normalize_features(token_map.astype(np.float32))
    aux_n = aux.astype(np.float32)
    feats = np.concatenate([dino, aux_weight * aux_n], axis=-1)
    return feats.reshape(-1, feats.shape[-1]).astype(np.float32)


def _cluster_scores(aux_samples: np.ndarray, tissue_samples: np.ndarray, labels: np.ndarray, n_clusters: int) -> pd.DataFrame:
    rows = []
    # aux: L,A,B,S,V,hema,edge
    for k in range(n_clusters):
        idx = labels == k
        if idx.sum() == 0:
            rows.append({"cluster": k, "count": 0, "score": -999.0})
            continue
        a = aux_samples[idx]
        tissue = tissue_samples[idx]
        L = float(a[:, 0].mean())
        S = float(a[:, 3].mean())
        V = float(a[:, 4].mean())
        hema = float(a[:, 5].mean())
        edge = float(a[:, 6].mean())
        tiss = float(tissue.mean())
        # Unsupervised gland-epithelium score.
        # It favors tissue-rich, hematoxylin/nuclear, moderately saturated and edge-rich prototypes,
        # while penalizing near-white background/lumen-like prototypes.
        score = 0.35 * hema + 0.20 * S + 0.20 * edge + 0.20 * tiss + 0.05 * (1.0 - L) - 0.15 * max(0.0, V - 0.90)
        rows.append(
            {
                "cluster": k,
                "count": int(idx.sum()),
                "score": score,
                "L": L,
                "S": S,
                "V": V,
                "hema": hema,
                "edge": edge,
                "tissue": tiss,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def fit_prototype_model(image_paths: List[Path], cfg: PseudoConfig, extractor: Any, workdir: Path) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    X_parts: List[np.ndarray] = []
    Aux_parts: List[np.ndarray] = []
    Tissue_parts: List[np.ndarray] = []
    used = image_paths[: cfg.max_fit_images]
    for p in tqdm(used, desc="fit-prototypes"):
        img = read_image_rgb(p)
        token_map = extractor.extract(img)
        gh, gw = token_map.shape[:2]
        aux, tissue = patch_aux_features(img, (gh, gw))
        feats = _combined_features(token_map, aux, cfg.aux_weight)
        tissue_flat = tissue.reshape(-1)
        valid = np.where(tissue_flat >= 0.20)[0]
        if len(valid) < 8:
            valid = np.arange(feats.shape[0])
        if len(valid) > cfg.max_tokens_per_image:
            valid = rng.choice(valid, size=cfg.max_tokens_per_image, replace=False)
        X_parts.append(feats[valid])
        Aux_parts.append(aux.reshape(-1, aux.shape[-1])[valid])
        Tissue_parts.append(tissue_flat[valid])
    X = np.concatenate(X_parts, axis=0)
    Aux = np.concatenate(Aux_parts, axis=0)
    Tissue = np.concatenate(Tissue_parts, axis=0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = MiniBatchKMeans(
        n_clusters=cfg.num_prototypes,
        random_state=cfg.seed,
        batch_size=min(4096, max(256, Xs.shape[0])),
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(Xs)
    score_df = _cluster_scores(Aux, Tissue, labels, cfg.num_prototypes)
    eligible = score_df[score_df["tissue"] >= 0.15]
    if len(eligible) == 0:
        eligible = score_df
    selected = eligible.head(cfg.foreground_prototypes)["cluster"].astype(int).tolist()
    score_df.to_csv(workdir / "prototype_scores.csv", index=False)
    model = {
        "scaler": scaler,
        "kmeans": kmeans,
        "selected_clusters": selected,
        "config": asdict(cfg),
        "score_table": score_df.to_dict(orient="records"),
    }
    joblib.dump(model, workdir / "prototype_model.joblib")
    print(f"[pseudo] Selected foreground prototypes: {selected}")
    return model


def prototype_attention(
    image_rgb: np.ndarray,
    extractor: Any,
    model: Dict[str, Any],
    aux_weight: float,
    temperature: float,
) -> np.ndarray:
    token_map = extractor.extract(image_rgb)
    gh, gw = token_map.shape[:2]
    aux, tissue_grid = patch_aux_features(image_rgb, (gh, gw))
    feats = _combined_features(token_map, aux, aux_weight)
    xs = model["scaler"].transform(feats)
    dist = model["kmeans"].transform(xs).astype(np.float32)
    temp = max(float(temperature), 1e-6)
    z = -dist / temp
    z -= z.max(axis=1, keepdims=True)
    prob = np.exp(z)
    prob /= np.clip(prob.sum(axis=1, keepdims=True), 1e-6, None)
    selected = model["selected_clusters"]
    score_grid = prob[:, selected].sum(axis=1).reshape(gh, gw)
    score_grid *= normalize01(tissue_grid)
    h, w = image_rgb.shape[:2]
    score = cv2.resize(score_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    return normalize01(score)


def generate_one_pseudo(
    image_path: Path,
    extractor: Any,
    model: Dict[str, Any],
    cfg: PseudoConfig,
    out_dirs: Dict[str, Path],
) -> Dict[str, Any]:
    img = read_image_rgb(image_path)
    tissue = tissue_mask_rgb(img)
    proto = prototype_attention(img, extractor, model, cfg.aux_weight, cfg.prototype_temperature)
    prior = stain_prior(img)
    score = cfg.blend_gamma * proto + (1.0 - cfg.blend_gamma) * prior
    score = normalize01(score) * tissue.astype(np.float32)
    if cfg.use_crf:
        score = dense_crf_refine(img, np.clip(score, 1e-4, 1 - 1e-4))
    mask = threshold_score(score, tissue, cfg.threshold)
    mask = clean_binary_mask(
        mask,
        min_area=cfg.min_area,
        hole_area=cfg.hole_area,
        closing_radius=cfg.closing_radius,
        opening_radius=1,
    )
    key = stem_key(image_path)
    save_mask(out_dirs["masks"] / f"{key}.png", mask)
    save_gray(out_dirs["attention"] / f"{key}.png", score)
    save_gray(out_dirs["prior"] / f"{key}.png", prior)
    save_overlay(img, mask, out_dirs["overlays"] / f"{key}.png")
    area_ratio = float(mask.sum()) / float(mask.size)
    tissue_ratio = float(tissue.sum()) / float(tissue.size)
    mean_score_fg = float(score[mask.astype(bool)].mean()) if mask.sum() > 0 else 0.0
    return {
        "image": str(image_path),
        "name": key,
        "area_ratio": area_ratio,
        "tissue_ratio": tissue_ratio,
        "mean_score_fg": mean_score_fg,
    }


def run_pseudo_generation(cfg: PseudoConfig) -> Path:
    workdir = ensure_dir(cfg.workdir)
    out_dirs = {
        "masks": ensure_dir(workdir / "pseudo_masks"),
        "attention": ensure_dir(workdir / "attention_maps"),
        "prior": ensure_dir(workdir / "stain_priors"),
        "overlays": ensure_dir(workdir / "pseudo_overlays"),
    }
    write_json(workdir / "pseudo_config.json", asdict(cfg))
    image_paths = list_images(cfg.images, recursive=cfg.recursive)
    extractor = build_extractor(
        encoder=cfg.encoder,
        device=cfg.device,
        image_size=cfg.image_size,
        dinov2_model=cfg.dinov2_model,
        dinov2_local_repo=cfg.dinov2_local_repo,
        dinov2_weights=cfg.dinov2_weights,
        pretrained=cfg.pretrained,
    )
    model_path = workdir / "prototype_model.joblib"
    if model_path.exists():
        print(f"[pseudo] Loading existing prototype model: {model_path}")
        proto_model = joblib.load(model_path)
    else:
        proto_model = fit_prototype_model(image_paths, cfg, extractor, workdir)
    rows = []
    for p in tqdm(image_paths, desc="pseudo-labels"):
        rows.append(generate_one_pseudo(p, extractor, proto_model, cfg, out_dirs))
    pd.DataFrame(rows).to_csv(workdir / "pseudo_summary.csv", index=False)
    print(f"[pseudo] Pseudo masks saved to: {out_dirs['masks']}")
    return out_dirs["masks"]
