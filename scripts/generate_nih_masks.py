#!/usr/bin/env python3
"""
NIH ChestXray14 Mask Generation
===============================

Purpose & Output
----------------
Generates lung segmentation masks for the NIH ChestXray14 cohort using an
existing U-Net checkpoint (default: Montgomery model). This script mirrors the
Shenzhen mask-generation flow but operates on the NIH dataset paths defined in
`configs/paths.yaml`. Masks are saved to the configured `nih_chestxray14.masks`
directory, one PNG per image (`<Image Index>_mask.png`).
"""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable

import argparse
import logging
from pathlib import Path
import sys
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xfp.config import load_paths_config
from xfp.models import load_unet_checkpoint

LOGGER = logging.getLogger("nih_mask_generation")
TARGET_SIZE = (512, 512)


try:  # Pillow >= 9.1
    Resampling = Image.Resampling
except AttributeError:
    Resampling = Image  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NIH ChestXray14 lung masks.")
    parser.add_argument(
        "--model-key",
        type=str,
        default="unet_montgomery_full",
        help="Model key inside configs/paths.yaml under `models`.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional explicit checkpoint path (overrides --model-key).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of images (for smoke tests).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have masks.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output directory for generated masks.",
    )
    parser.add_argument(
        "--sample-ids-file",
        type=str,
        default=None,
        help="Optional text file with one NIH sample id per line (e.g., 00001234_001).",
    )
    return parser.parse_args()


def generate_mask(image_path: Path, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    """Run U-Net inference on a single image."""
    image = Image.open(image_path).convert("L")
    original_size = image.size
    image_resized = ImageOps.fit(image, TARGET_SIZE, method=Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image_resized, dtype=np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        mask_pred = (probs.cpu().numpy()[0, 0] > 0.5).astype(np.uint8)

    mask_image = Image.fromarray((mask_pred * 255).astype(np.uint8), mode="L")
    mask_resized = mask_image.resize(original_size, resample=Resampling.NEAREST)
    return np.asarray(mask_resized)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    config_path = Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"
    paths_cfg = load_paths_config(config_path, validate=False)

    nih_cfg = paths_cfg.datasets.get("nih_chestxray14")
    if nih_cfg is None:
        raise KeyError("Add nih_chestxray14 entry to configs/paths.yaml under datasets.")

    images_dir = nih_cfg.images.resolve()
    masks_dir = Path(args.output_dir).resolve() if args.output_dir else nih_cfg.masks.resolve()
    masks_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.rglob("*.png"))
    if args.sample_ids_file:
        requested = {
            line.strip()
            for line in Path(args.sample_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        image_files = [p for p in image_files if p.stem in requested]
    if args.limit:
        image_files = image_files[: args.limit]
    LOGGER.info("Found %d NIH PNG files (limit=%s).", len(image_files), args.limit)

    # Resolve model checkpoint
    model_path = Path(args.model_path) if args.model_path else paths_cfg.models.get(args.model_key)
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model checkpoint not found for key `{args.model_key}`.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Loading model %s on %s", model_path, device)
    model = load_unet_checkpoint(Path(model_path), device=device)
    model.eval()

    generated = 0
    skipped = 0
    failed = 0

    for image_path in tqdm(image_files, desc="Generating NIH masks"):
        mask_path = masks_dir / f"{image_path.stem}_mask.png"
        if args.skip_existing and mask_path.exists():
            skipped += 1
            continue
        try:
            mask_array = generate_mask(image_path, model, device)
            Image.fromarray(mask_array, mode="L").save(mask_path)
            generated += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            LOGGER.error("Failed to generate mask for %s: %s", image_path.name, exc)

    LOGGER.info(
        "NIH mask generation finished: generated=%d skipped=%d failed=%d (masks dir: %s)",
        generated,
        skipped,
        failed,
        masks_dir,
    )


if __name__ == "__main__":
    main()
