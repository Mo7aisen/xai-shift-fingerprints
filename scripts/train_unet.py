#!/usr/bin/env python3
"""Train a UNet lung segmentation model for JSRT/Montgomery/NIH."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset

import _path_setup  # noqa: F401 - ensures xfp is importable

from xfp.config import load_paths_config
from xfp.models import UNet

LOGGER = logging.getLogger("xfp.train_unet")
TARGET_SIZE = (512, 512)


try:  # Pillow >= 9.1
    Resampling = Image.Resampling
except AttributeError:  # pragma: no cover
    Resampling = Image  # type: ignore[attr-defined]


class LungDataset(Dataset):
    def __init__(self, image_paths: List[Path], mask_paths: List[Path]) -> None:
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image = ImageOps.fit(image, TARGET_SIZE, method=Resampling.BILINEAR)
        mask = ImageOps.fit(mask, TARGET_SIZE, method=Resampling.NEAREST)

        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = (np.asarray(mask) > 0).astype(np.float32)

        # Normalize to match fingerprint runner expectation
        image_arr = (image_arr - 0.5) / 0.5

        image_tensor = torch.from_numpy(image_arr).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        return image_tensor, mask_tensor


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def build_dataset(paths_cfg, dataset_key: str) -> Tuple[List[Path], List[Path]]:
    dataset_cfg = paths_cfg.datasets[dataset_key]
    images_dir = dataset_cfg.images.resolve()
    masks_dir = dataset_cfg.masks.resolve()

    if dataset_key == "montgomery":
        left = sorted((masks_dir / "leftMask").glob("*.png"))
        right = sorted((masks_dir / "rightMask").glob("*.png"))
        left_map = {p.stem: p for p in left}
        right_map = {p.stem: p for p in right}
        images = sorted(images_dir.glob("*.png"))
        image_paths = []
        mask_paths = []
        for img in images:
            if img.stem in left_map and img.stem in right_map:
                # Merge on the fly by stacking and max
                merged = masks_dir / f"{img.stem}_merged_mask.png"
                if not merged.exists():
                    left_img = Image.open(left_map[img.stem]).convert("L")
                    right_img = Image.open(right_map[img.stem]).convert("L")
                    combined = Image.fromarray(np.maximum(np.asarray(left_img), np.asarray(right_img)))
                    combined.save(merged)
                image_paths.append(img)
                mask_paths.append(merged)
        return image_paths, mask_paths

    if dataset_key == "shenzhen":
        images = sorted(images_dir.glob("*.png"))
        image_paths = []
        mask_paths = []
        for img in images:
            mask = masks_dir / f"{img.stem}_mask.png"
            if mask.exists():
                image_paths.append(img)
                mask_paths.append(mask)
        return image_paths, mask_paths

    if dataset_key == "nih_chestxray14":
        images = sorted(images_dir.rglob("*.png"))
        image_paths = []
        mask_paths = []
        for img in images:
            mask = masks_dir / f"{img.stem}_mask.png"
            if mask.exists():
                image_paths.append(img)
                mask_paths.append(mask)
        return image_paths, mask_paths

    # JSRT
    images = sorted(images_dir.glob("*.png"))
    masks = {p.stem: p for p in masks_dir.glob("*.png")}
    image_paths = []
    mask_paths = []
    for img in images:
        mask = masks.get(img.stem)
        if mask:
            image_paths.append(img)
            mask_paths.append(mask)
    return image_paths, mask_paths


def split_data(
    image_paths: List[Path],
    mask_paths: List[Path],
    *,
    splits_map: Dict[str, str] | None,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    if splits_map:
        train_idx = []
        val_idx = []
        for idx, image_path in enumerate(image_paths):
            split = splits_map.get(image_path.stem)
            if split == "train":
                train_idx.append(idx)
            elif split == "val":
                val_idx.append(idx)
        if train_idx and val_idx:
            return train_idx, val_idx

    indices = list(range(len(image_paths)))
    random.Random(seed).shuffle(indices)
    split = int(0.8 * len(indices))
    return indices[:split], indices[split:]


def load_splits(dataset_root: Path) -> Dict[str, str] | None:
    splits_path = dataset_root / "splits.csv"
    if not splits_path.exists():
        return None
    df = pd.read_csv(splits_path)
    if "patient_id" not in df.columns or "split" not in df.columns:
        return None
    return {str(row["patient_id"]): str(row["split"]) for _, row in df.iterrows()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNet for lung segmentation.")
    parser.add_argument("--dataset", required=True, choices=["jsrt", "montgomery", "shenzhen", "nih_chestxray14"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True, help="Checkpoint output path.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    paths_cfg = load_paths_config(Path("configs/paths.yaml"), validate=False)
    image_paths, mask_paths = build_dataset(paths_cfg, args.dataset)
    if not image_paths:
        raise RuntimeError("No images found for training.")

    dataset_root = paths_cfg.datasets[args.dataset].images.resolve().parent
    splits_map = load_splits(dataset_root)
    train_idx, val_idx = split_data(image_paths, mask_paths, splits_map=splits_map, seed=args.seed)

    train_ds = LungDataset([image_paths[i] for i in train_idx], [mask_paths[i] for i in train_idx])
    val_ds = LungDataset([image_paths[i] for i in val_idx], [mask_paths[i] for i in val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                loss = bce(logits, masks) + dice_loss(logits, masks)
                val_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        LOGGER.info("epoch %d/%d train=%.4f val=%.4f", epoch, args.epochs, train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "dataset": args.dataset,
                },
                output_path,
            )

    LOGGER.info("training complete. best_val=%.4f checkpoint=%s", best_val, output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "best_val_loss": best_val,
                "checkpoint": str(output_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
