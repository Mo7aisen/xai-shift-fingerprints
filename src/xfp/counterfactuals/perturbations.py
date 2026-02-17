"""Low-level perturbation primitives for counterfactual experiments."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from skimage import morphology


def _ensure_binary(mask: np.ndarray) -> np.ndarray:
    return (mask > 0.5).astype(np.uint8)


def dilate_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Dilate the binary mask by a disk of `radius` pixels."""
    if radius <= 0:
        return _ensure_binary(mask)
    selem = morphology.disk(radius)
    dilated = morphology.binary_dilation(_ensure_binary(mask), selem)
    return dilated.astype(np.uint8)


def erode_mask(mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Erode the binary mask by a disk of `radius` pixels."""
    if radius <= 0:
        return _ensure_binary(mask)
    selem = morphology.disk(radius)
    eroded = morphology.binary_erosion(_ensure_binary(mask), selem)
    return eroded.astype(np.uint8)


def apply_intensity_offset(image: np.ndarray, offset: float = 0.1) -> np.ndarray:
    """Apply a uniform intensity offset, clipping to [0, 1]."""
    result = np.clip(image + offset, 0.0, 1.0)
    return result.astype(np.float32)


def insert_gaussian_nodule(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    center: Tuple[int, int] | None = None,
    radius: int = 12,
    intensity: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Inject a synthetic Gaussian nodule near the mask border."""
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w]

    if center is None:
        border_pixels = np.argwhere(np.logical_xor(dilate_mask(mask, 1), erode_mask(mask, 1)))
        if border_pixels.size == 0:
            center = (h // 2, w // 2)
        else:
            idx = np.random.default_rng().integers(0, border_pixels.shape[0])
            center_y, center_x = border_pixels[idx]
            center = (int(center_y), int(center_x))

    cy, cx = center
    gaussian = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / float(2 * radius ** 2))
    scaled = np.clip(image + intensity * gaussian, 0.0, 1.0).astype(np.float32)
    metadata = {
        "type": "gaussian_nodule",
        "center_y": cy,
        "center_x": cx,
        "radius": radius,
        "intensity": intensity,
    }
    return scaled, mask.astype(np.uint8), metadata
