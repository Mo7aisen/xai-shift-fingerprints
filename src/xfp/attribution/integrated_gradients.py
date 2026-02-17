"""Integrated Gradients attribution helpers."""

from __future__ import annotations

from typing import Callable

import torch
from captum.attr import IntegratedGradients


def _build_forward_fn(model: torch.nn.Module) -> Callable[[torch.Tensor], torch.Tensor]:
    def forward_fn(inputs: torch.Tensor) -> torch.Tensor:
        logits = model(inputs)
        probs = torch.sigmoid(logits)
        return probs.mean(dim=(1, 2, 3), keepdim=False)

    return forward_fn


def compute_integrated_gradients(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    *,
    n_steps: int = 32,
    internal_batch_size: int | None = None,
    baseline: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Return Integrated Gradients attribution map for a segmentation model.

    The internal forward function averages the sigmoid probabilities across the mask
    so that higher attributions correspond to stronger predicted mask activation.

    Args:
        model: The segmentation model to attribute
        input_tensor: Input image tensor
        n_steps: Number of interpolation steps for IG
        internal_batch_size: Batch size for processing interpolation steps.
            Use smaller values (e.g., 4-8) to reduce GPU memory usage.
            If None, processes all steps in one batch (may cause OOM).
        baseline: Optional baseline tensor or scalar. Defaults to -1.0 to align
            with inputs normalised via mean=0.5, std=0.5 (range [-1, 1]).
    """

    model.eval()

    if baseline is None:
        baseline_tensor = torch.full_like(input_tensor, -1.0)
    elif isinstance(baseline, torch.Tensor):
        baseline_tensor = baseline.to(device=input_tensor.device, dtype=input_tensor.dtype)
    else:
        baseline_tensor = torch.full_like(input_tensor, float(baseline))

    # Captum objects are not picklable; reuse across invocations by keying on id(model).
    forward_fn = _build_forward_fn(model)
    ig = IntegratedGradients(forward_fn)
    attributions = ig.attribute(
        input_tensor,
        baselines=baseline_tensor,
        n_steps=n_steps,
        method="gausslegendre",
        internal_batch_size=internal_batch_size,
    )
    return attributions
