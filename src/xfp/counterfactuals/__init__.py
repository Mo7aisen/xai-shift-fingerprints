"""Synthetic counterfactual generation utilities."""

from .dataset import CounterfactualSample, CounterfactualBatch
from .perturbations import (
    dilate_mask,
    erode_mask,
    insert_gaussian_nodule,
    apply_intensity_offset,
)

__all__ = [
    "CounterfactualSample",
    "CounterfactualBatch",
    "dilate_mask",
    "erode_mask",
    "insert_gaussian_nodule",
    "apply_intensity_offset",
]
