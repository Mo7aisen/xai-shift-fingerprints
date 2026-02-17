"""Dataset abstractions for counterfactual experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Optional

import numpy as np


@dataclass
class CounterfactualSample:
    """Container for a single counterfactual tuple."""

    sample_id: str
    dataset_key: str
    original_image: np.ndarray
    original_mask: np.ndarray
    perturbed_image: np.ndarray
    perturbed_mask: np.ndarray
    perturbation_metadata: Dict[str, object]


@dataclass
class CounterfactualBatch:
    """Metadata describing a batch of counterfactuals located on disk."""

    dataset_key: str
    perturbation_name: str
    output_dir: Path
    samples: Iterable[CounterfactualSample]


def iterate_cache(
    cache_dir: Path,
    perturb_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, Dict[str, object]]],
    *,
    limit: Optional[int] = None,
) -> Iterator[CounterfactualSample]:
    """Yield counterfactual samples from cached NPZ tensors."""

    npz_files = sorted(cache_dir.glob("*.npz"))
    if limit is not None:
        npz_files = npz_files[:limit]

    for npz_path in npz_files:
        data = np.load(npz_path)
        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.uint8)
        perturbed_image, perturbed_mask, meta = perturb_fn(image, mask)
        yield CounterfactualSample(
            sample_id=npz_path.stem,
            dataset_key=cache_dir.parent.name,
            original_image=image,
            original_mask=mask,
            perturbed_image=perturbed_image,
            perturbed_mask=perturbed_mask,
            perturbation_metadata=meta,
        )
