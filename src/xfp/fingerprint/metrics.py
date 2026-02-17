"""Attribution fingerprint metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from scipy import ndimage


CurvePoints = Sequence[float]


@dataclass
class BorderStats:
    abs_sum: float
    signed_sum: float
    pixel_count: int


def compute_border_stats(
    attribution_map: np.ndarray,
    segmentation_mask: np.ndarray,
    *,
    dilation_radius: int = 2,
) -> BorderStats:
    """Summarize attribution intensity along the segmentation border."""

    if attribution_map.shape != segmentation_mask.shape:
        raise ValueError("Attribution map and segmentation mask must share shape.")

    border_mask = _dilated_border(segmentation_mask.astype(bool), dilation_radius)
    if not np.any(border_mask):
        return BorderStats(abs_sum=0.0, signed_sum=0.0, pixel_count=0)

    values = attribution_map[border_mask]
    return BorderStats(
        abs_sum=float(np.sum(np.abs(values))),
        signed_sum=float(np.sum(values)),
        pixel_count=int(border_mask.sum()),
    )


def compute_coverage_curve(
    attribution_map: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    quantiles: CurvePoints = tuple(np.linspace(0.0, 1.0, 11)),
) -> Dict[str, float]:
    """Summarize coverage curve by area under curve and sampled quantiles."""

    if attribution_map.ndim != 2:
        raise ValueError("Attribution map must be 2D.")

    weights = np.abs(attribution_map)
    if mask is not None:
        if mask.shape != attribution_map.shape:
            raise ValueError("Mask must match attribution_map shape.")
        weights = np.where(mask, weights, 0.0)

    flat = weights.reshape(-1)
    total_pixels = flat.size
    if total_pixels == 0:
        return {"coverage_auc": 0.0}

    if np.all(flat == 0):
        coverage = np.linspace(0.0, 1.0, total_pixels + 1)
        cumulative = coverage.copy()
    else:
        sorted_weights = np.sort(flat)[::-1]
        cumulative = np.cumsum(sorted_weights)
        total_weight = float(cumulative[-1]) if cumulative.size else 0.0
        if total_weight <= 0:
            coverage = np.linspace(0.0, 1.0, total_pixels + 1)
            cumulative = coverage.copy()
        else:
            cumulative = cumulative / total_weight
            coverage = np.arange(1, total_pixels + 1, dtype=float) / total_pixels
            coverage = np.concatenate(([0.0], coverage))
            cumulative = np.concatenate(([0.0], cumulative))

    auc = float(np.trapz(cumulative, coverage))
    quantile_values = _sample_curve(coverage, cumulative, quantiles)
    result = {"coverage_auc": auc}
    for fraction, value in zip(quantiles, quantile_values):
        result[f"coverage_q_{fraction:.2f}"] = value
    return result


def compute_histogram_features(
    attribution_map: np.ndarray,
    *,
    bins: int = 32,
    return_distribution: bool = False,
) -> Dict[str, float]:
    """Return histogram-based statistics for attribution magnitudes."""
    abs_values = np.abs(attribution_map.reshape(-1))
    if abs_values.size == 0:
        result = {"hist_entropy": 0.0}
        if return_distribution:
            result.update({f"hist_bin_{idx:02d}": 0.0 for idx in range(bins)})
        return result

    max_val = abs_values.max()
    normalised = abs_values / max_val if max_val > 0 else abs_values
    hist, _ = np.histogram(normalised, bins=bins, range=(0.0, 1.0), density=False)
    total = hist.sum()
    if total <= 0:
        result = {"hist_entropy": 0.0}
        if return_distribution:
            result.update({f"hist_bin_{idx:02d}": 0.0 for idx in range(bins)})
        return result

    prob = hist.astype(np.float64) / total
    prob = prob[prob > 0]
    entropy = -float(np.sum(prob * np.log(prob)))
    result = {"hist_entropy": entropy}
    if return_distribution:
        full_prob = hist.astype(np.float64) / total
        for idx, value in enumerate(full_prob):
            result[f"hist_bin_{idx:02d}"] = float(value)
    return result


def _sample_curve(xs: np.ndarray, ys: np.ndarray, points: CurvePoints) -> np.ndarray:
    return np.interp(points, xs, ys, left=ys[0], right=ys[-1])


def _dilated_border(segmentation_mask: np.ndarray, radius: int) -> np.ndarray:
    if segmentation_mask.ndim != 2:
        raise ValueError("segmentation_mask must be 2D")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if not np.any(segmentation_mask):
        return np.zeros_like(segmentation_mask, dtype=bool)

    structure = ndimage.generate_binary_structure(2, 2)
    eroded = ndimage.binary_erosion(segmentation_mask, structure=structure, border_value=0)
    outline = segmentation_mask & (~eroded)
    if radius == 0:
        return outline
    disk = _disk_structure(radius)
    return ndimage.binary_dilation(outline, structure=disk)


def _disk_structure(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.array([[True]], dtype=bool)
    diameter = radius * 2 + 1
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = (xx * xx + yy * yy) <= radius * radius
    return disk.astype(bool, copy=False)


def compute_component_stats(
    attribution_map: np.ndarray,
    segmentation_mask: np.ndarray,
    *,
    quantile: float = 0.95,
) -> Dict[str, float]:
    """Derive topology features from high-attribution components within the mask ROI."""

    if attribution_map.shape != segmentation_mask.shape:
        raise ValueError("Attribution map and segmentation mask must share shape.")

    mask_bool = segmentation_mask.astype(bool)
    abs_attr = np.abs(attribution_map)

    roi_values = abs_attr[mask_bool] if np.any(mask_bool) else abs_attr.reshape(-1)
    if roi_values.size == 0:
        return {
            "component_count": 0.0,
            "component_mean_size": 0.0,
            "component_median_size": 0.0,
            "component_largest_size": 0.0,
            "component_border_fraction": 0.0,
            "component_border_mass_fraction": 0.0,
        }

    threshold = np.quantile(roi_values, quantile)
    if threshold <= 0:
        active = abs_attr > 0
    else:
        active = abs_attr >= threshold
    active &= mask_bool

    structure = ndimage.generate_binary_structure(2, 2)
    labeled, component_count = ndimage.label(active, structure=structure)
    if component_count == 0:
        return {
            "component_count": 0.0,
            "component_mean_size": 0.0,
            "component_median_size": 0.0,
            "component_largest_size": 0.0,
            "component_border_fraction": 0.0,
            "component_border_mass_fraction": 0.0,
        }

    sizes = ndimage.sum(active, labeled, index=range(1, component_count + 1))
    sizes = np.asarray(sizes, dtype=float)
    total_mass = float(sizes.sum()) if sizes.size else 0.0

    eroded = ndimage.binary_erosion(mask_bool, structure=structure, border_value=0)
    border_mask = mask_bool & (~eroded)
    border_labels = np.unique(labeled[border_mask])
    border_labels = border_labels[(border_labels > 0)]
    border_fraction = float(border_labels.size) / float(component_count)
    if border_labels.size and total_mass > 0:
        border_mass = float(np.sum(sizes[border_labels - 1])) / total_mass
    else:
        border_mass = 0.0

    return {
        "component_count": float(component_count),
        "component_mean_size": float(sizes.mean()),
        "component_median_size": float(np.median(sizes)),
        "component_largest_size": float(sizes.max()),
        "component_border_fraction": border_fraction,
        "component_border_mass_fraction": border_mass,
    }
