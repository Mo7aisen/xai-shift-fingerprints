"""Unit tests for counterfactual perturbation utilities."""

import pytest

pytest.importorskip("skimage")

import numpy as np

from xfp.counterfactuals.metrics import compute_basic_metrics
from xfp.counterfactuals.perturbations import dilate_mask, erode_mask, insert_gaussian_nodule


def test_dilate_and_erode_mask() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 4:6] = 1

    dilated = dilate_mask(mask, radius=1)
    eroded = erode_mask(mask, radius=1)

    assert dilated.sum() >= mask.sum()
    assert eroded.sum() <= mask.sum()


def test_insert_gaussian_nodule() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((32, 32), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:22, 10:22] = 1

    perturbed_image, perturbed_mask, meta = insert_gaussian_nodule(image, mask, radius=5, intensity=0.5)

    assert perturbed_image.shape == image.shape
    assert perturbed_mask.shape == mask.shape
    assert 0.0 <= perturbed_image.min() <= perturbed_image.max() <= 1.0
    assert meta["type"] == "gaussian_nodule"


def test_compute_basic_metrics() -> None:
    attr_a = np.zeros((8, 8), dtype=np.float32)
    attr_b = np.ones((8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1

    metrics = compute_basic_metrics(attr_a, attr_b, mask)
    assert metrics.attribution_l1 == 1.0
    assert metrics.full_mask_counterfactual >= 0.0
