from __future__ import annotations

import numpy as np


def dense_crf_refine(
    image_rgb: np.ndarray,
    prob_fg: np.ndarray,
    n_iter: int = 5,
    sxy_gaussian: int = 3,
    compat_gaussian: int = 3,
    sxy_bilateral: int = 40,
    srgb_bilateral: int = 8,
    compat_bilateral: int = 5,
) -> np.ndarray:
    """DenseCRF binary refinement.

    If pydensecrf is unavailable, the function returns the original probability
    map. This keeps the pipeline executable while preserving CRF when installed.
    """
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except Exception:
        return prob_fg.astype(np.float32)

    h, w = prob_fg.shape
    p_fg = np.clip(prob_fg.astype(np.float32), 1e-4, 1.0 - 1e-4)
    probs = np.stack([1.0 - p_fg, p_fg], axis=0)
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)
    d.addPairwiseBilateral(
        sxy=sxy_bilateral,
        srgb=srgb_bilateral,
        rgbim=np.ascontiguousarray(image_rgb),
        compat=compat_bilateral,
    )
    q = d.inference(n_iter)
    refined = np.array(q, dtype=np.float32).reshape((2, h, w))[1]
    return refined
