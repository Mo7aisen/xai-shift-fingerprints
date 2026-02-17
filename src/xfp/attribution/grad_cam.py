"""Grad-CAM attribution helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
import torch.nn.functional as F


@dataclass
class _HookStore:
    activations: list[torch.Tensor]
    gradients: list[torch.Tensor]


@contextmanager
def _register_grad_cam_hooks(module: torch.nn.Module) -> Iterator[_HookStore]:
    store = _HookStore(activations=[], gradients=[])

    def _forward_hook(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        store.activations.append(output.detach())

    def _backward_hook(
        _: torch.nn.Module,
        __: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        store.gradients.append(grad_output[0].detach())

    fwd_handle = module.register_forward_hook(_forward_hook)
    bwd_handle = module.register_full_backward_hook(_backward_hook)
    try:
        yield store
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def _find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last convolutional layer in the model for Grad-CAM."""

    last_conv: torch.nn.Module | None = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Model does not contain a Conv2d layer required for Grad-CAM.")
    return last_conv


def compute_grad_cam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    *,
    target_reducer: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute Grad-CAM attribution map for a segmentation model.

    Args:
        model: Segmentation model under evaluation (assumed to output logits).
        input_tensor: Tensor shaped [B, 1, H, W].

    Returns:
        Attribution tensor shaped [H, W] on CPU.
    """

    model.eval()
    last_conv = _find_last_conv_layer(model)
    reducer = target_reducer or (lambda probs: probs.mean())

    with _register_grad_cam_hooks(last_conv) as store:
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        scalar_target = reducer(probs)
        model.zero_grad(set_to_none=True)
        scalar_target.backward(retain_graph=False)

    if not store.activations or not store.gradients:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

    activations = store.activations[-1]
    gradients = store.gradients[-1]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))

    if cam.max() > 0:
        cam = cam / cam.max()

    cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
    return cam.detach().cpu()[0, 0]
