"""Metric utilities for counterfactual attribution comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class CounterfactualMetrics:
    attribution_l1: float
    full_mask_counterfactual: float
    coverage_auc_delta: float


def compute_basic_metrics(
    original_attr: np.ndarray,
    perturbed_attr: np.ndarray,
    mask: np.ndarray,
) -> CounterfactualMetrics:
    """Compute preliminary Δ metrics between original and perturbed attributions."""
    if original_attr.shape != perturbed_attr.shape:
        raise ValueError("Attribution maps must have identical shape.")

    l1 = float(np.mean(np.abs(original_attr - perturbed_attr)))
    mask_roi = mask.astype(bool)
    mask_delta = (
        float(np.mean(np.abs(original_attr[mask_roi] - perturbed_attr[mask_roi])))
        if np.any(mask_roi)
        else 0.0
    )

    def coverage_curve(attribution: np.ndarray) -> np.ndarray:
        weights = np.abs(attribution.reshape(-1))
        if np.all(weights == 0):
            return np.zeros(11, dtype=float)
        sorted_weights = np.sort(weights)[::-1]
        cumulative = np.cumsum(sorted_weights)
        cumulative /= cumulative[-1]
        idx = np.linspace(0, cumulative.size - 1, 11).astype(int)
        return cumulative[idx]

    curve_orig = coverage_curve(original_attr)
    curve_pert = coverage_curve(perturbed_attr)
    coverage_delta = float(np.mean(np.abs(curve_orig - curve_pert)))

    return CounterfactualMetrics(
        attribution_l1=l1,
        full_mask_counterfactual=mask_delta,
        coverage_auc_delta=coverage_delta,
    )


def metrics_as_dict(metrics: CounterfactualMetrics, extra: Dict[str, float] | None = None) -> Dict[str, float]:
    payload: Dict[str, float] = {
        "attribution_l1": metrics.attribution_l1,
        "full_mask_counterfactual": metrics.full_mask_counterfactual,
        "coverage_auc_delta": metrics.coverage_auc_delta,
    }
    if extra:
        payload.update(extra)
    return payload
