#!/usr/bin/env python3
"""Train UNet directly from cached NPZ tensors (image/mask)."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import _path_setup  # noqa: F401
from xfp.models import UNet

LOGGER = logging.getLogger("xfp.train_unet_from_cache")


class CacheDataset(Dataset):
    def __init__(self, files: List[Path]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        arr = np.load(self.files[idx])
        image = arr["image"].astype(np.float32)
        mask = arr["mask"].astype(np.float32)
        image = (image - 0.5) / 0.5
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def split_indices(n: int, seed: int, train_ratio: float) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = int(train_ratio * n)
    return idx[:cut], idx[cut:]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UNet from NPZ cache files.")
    p.add_argument("--cache-dir", required=True, help="Directory containing cached *.npz files.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output", required=True, help="Checkpoint output path (*.pt).")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--loss",
        choices=["dice", "bce_dice"],
        default="bce_dice",
        help="Training loss. bce_dice matches historical checkpoints.",
    )
    p.add_argument(
        "--features",
        default="32,64,128,256",
        help="Comma-separated UNet channel widths, e.g. 24,48,96,192.",
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    features = tuple(int(v.strip()) for v in args.features.split(",") if v.strip())
    if len(features) != 4:
        raise ValueError("--features must contain exactly four comma-separated integers.")
    if any(v <= 0 for v in features):
        raise ValueError("--features values must be positive.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    cache_dir = Path(args.cache_dir)
    files = sorted(cache_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No cache npz files found in {cache_dir}")

    train_idx, val_idx = split_indices(len(files), args.seed, args.train_ratio)
    train_files = [files[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    LOGGER.info("cache files: total=%d train=%d val=%d", len(files), len(train_files), len(val_files))

    train_ds = CacheDataset(train_files)
    val_ds = CacheDataset(val_files)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = UNet(features=features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            dice_term = dice_loss(logits, y)
            if args.loss == "dice":
                loss = dice_term
            else:
                loss = bce(logits, y) + dice_term
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                dice_term = dice_loss(logits, y)
                if args.loss == "dice":
                    loss = dice_term
                else:
                    loss = bce(logits, y) + dice_term
                val_loss += loss.item() * x.size(0)

        train_loss /= max(1, len(train_loader.dataset))
        val_loss /= max(1, len(val_loader.dataset))
        LOGGER.info("epoch %d/%d train=%.4f val=%.4f", epoch, args.epochs, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "source": str(cache_dir),
                    "model_config": {
                        "n_channels": 1,
                        "n_classes": 1,
                        "features": list(features),
                    },
                    "training_config": {
                        "loss": args.loss,
                        "seed": args.seed,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "train_ratio": args.train_ratio,
                    },
                },
                out,
            )

    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "cache_dir": str(cache_dir),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "train_ratio": args.train_ratio,
                "loss": args.loss,
                "features": list(features),
                "best_val_loss": best_val,
                "checkpoint": str(out),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    LOGGER.info("done best_val=%.4f checkpoint=%s", best_val, out)


if __name__ == "__main__":
    main()
