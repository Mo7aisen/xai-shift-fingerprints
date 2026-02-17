"""Utilities to save counterfactual inspection panels."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def save_triptych(
    *,
    original_image: np.ndarray,
    perturbed_image: np.ndarray,
    repaired_image: np.ndarray | None = None,
    original_attr: np.ndarray | None = None,
    perturbed_attr: np.ndarray | None = None,
    output_path: Path,
    metadata: Dict[str, object] | None = None,
) -> None:
    """Save a multi-panel comparison figure for QA/UI integration."""

    cols = 3 if repaired_image is not None else 2
    rows = 2 if original_attr is not None and perturbed_attr is not None else 1
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if rows == 1 and cols == 2:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    axes[0, 0].imshow(original_image, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(perturbed_image, cmap="gray")
    axes[0, 1].set_title("Perturbed")
    axes[0, 1].axis("off")

    if repaired_image is not None:
        axes[0, 2].imshow(repaired_image, cmap="gray")
        axes[0, 2].set_title("Repaired")
        axes[0, 2].axis("off")

    if rows > 1:
        axes[1, 0].imshow(original_attr, cmap="bwr")
        axes[1, 0].set_title("Attr (orig)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(perturbed_attr, cmap="bwr")
        axes[1, 1].set_title("Attr (perturbed)")
        axes[1, 1].axis("off")

        if repaired_image is not None:
            diff = perturbed_attr - original_attr
            axes[1, 2].imshow(diff, cmap="bwr")
            axes[1, 2].set_title("Attr Δ")
            axes[1, 2].axis("off")

    if metadata:
        fig.suptitle("\n".join(f"{k}: {v}" for k, v in metadata.items()), fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
