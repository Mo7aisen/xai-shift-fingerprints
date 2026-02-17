"""Unit tests for attribution helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from xfp.attribution import compute_grad_cam, compute_integrated_gradients


class _TinyUNet(nn.Module):
    """Minimal convolutional network for attribution smoke tests."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)


def _dummy_input(device: torch.device) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(1, 1, 32, 32, device=device, requires_grad=True)


def test_integrated_gradients_returns_map() -> None:
    model = _TinyUNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tensor = _dummy_input(device)
    baseline = torch.full_like(tensor, -1.0)
    attributions = compute_integrated_gradients(
        model,
        tensor,
        n_steps=8,
        internal_batch_size=4,
        baseline=baseline,
    )
    assert attributions.shape == tensor.shape
    assert torch.isfinite(attributions).all()


def test_grad_cam_returns_normalised_map() -> None:
    model = _TinyUNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tensor = _dummy_input(device)
    cam = compute_grad_cam(model, tensor)
    assert cam.shape == torch.Size([32, 32])
    assert torch.isfinite(cam).all()
    assert cam.max().item() <= 1.0 + 1e-6
