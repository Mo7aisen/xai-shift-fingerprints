#!/usr/bin/env python3
"""
Targeted fixer for NIH ChestXray14 lung masks.

Reads the QA findings CSV, re-segments flagged cases with stricter thresholds,
and applies lightweight morphological cleanup so over-filled or fragmented masks
can be corrected without regenerating the entire cohort.
"""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

import sys

sys.path.insert(0, str(SRC_ROOT))

from xfp.config import load_paths_config  # noqa: E402
from xfp.models import load_unet_checkpoint  # noqa: E402

LOGGER = logging.getLogger("nih_mask_fixer")
TARGET_SIZE = (512, 512)


try:  # Pillow >= 9.1
    Resampling = Image.Resampling
except AttributeError:  # pragma: no cover - legacy Pillow
    Resampling = Image  # type: ignore[attr-defined]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair flagged NIH lung masks.")
    parser.add_argument(
        "--qc-csv",
        type=Path,
        default=PROJECT_ROOT / "reports" / "external_validation" / "nih_mask_qc_findings.csv",
        help="Path to the QA findings CSV produced by run_nih_mask_qc.py.",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="unet_nih_full",
        help="Model alias inside configs/paths.yaml (falls back to --model-path).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit checkpoint override.",
    )
    parser.add_argument(
        "--flags",
        nargs="*",
        default=["coverage_high", "fragmented"],
        help="Which QA flags should trigger a fix (default: coverage_high + fragmented).",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=0.5,
        help="Base sigmoid threshold for other flags.",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.85,
        help="Sigmoid threshold used for coverage_high cases.",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=0.90,
        help="Shrink masks until coverage falls below this value for coverage_high cases.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.45,
        help="Dilate masks until coverage exceeds this floor (guards against empty masks).",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Optional directory to store originals (default: masks_dir/fix_backup_<timestamp>).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit the number of flagged masks to process (for dry runs).",
    )
    return parser.parse_args()


def load_flagged_cases(csv_path: Path, flags: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["flags"] = df["flags"].fillna("")
    df["flags_list"] = df["flags"].apply(lambda text: [token.strip() for token in text.split(";") if token.strip()])
    if flags:
        df = df[df["flags_list"].map(lambda lst: any(flag in lst for flag in flags))]
    return df.reset_index(drop=True)


def load_paths() -> Dict[str, Path]:
    cfg = load_paths_config(PROJECT_ROOT / "configs" / "paths.yaml")
    nih_cfg = cfg.datasets.get("nih_chestxray14")
    if nih_cfg is None:
        raise KeyError("nih_chestxray14 missing from configs/paths.yaml")
    return {
        "images": nih_cfg.images.resolve(),
        "masks": nih_cfg.masks.resolve(),
        "models": cfg.models,
    }


def resolve_checkpoint(model_key: str, model_path: Path | None, model_map: Dict[str, Path]) -> Path:
    if model_path:
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint {model_path} not found.")
        return model_path
    checkpoint = model_map.get(model_key)
    if not checkpoint or not checkpoint.exists():
        raise FileNotFoundError(f"Model path missing for key {model_key}.")
    return checkpoint


def infer_mask(image_path: Path, model: torch.nn.Module, device: torch.device, threshold: float) -> np.ndarray:
    image = Image.open(image_path).convert("L")
    original_size = image.size
    image_resized = ImageOps.fit(image, TARGET_SIZE, method=Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image_resized, dtype=np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0]
        mask_pred = (probs > threshold).cpu().numpy().astype(np.uint8)
    mask_image = Image.fromarray(mask_pred * 255, mode="L")
    mask_resized = mask_image.resize(original_size, resample=Resampling.NEAREST)
    return np.asarray(mask_resized, dtype=np.uint8)


def keep_largest_component(mask_bool: np.ndarray) -> np.ndarray:
    visited = np.zeros(mask_bool.shape, dtype=bool)
    best_mask = np.zeros_like(mask_bool)
    best_area = 0
    rows, cols = mask_bool.shape
    for r in range(rows):
        for c in range(cols):
            if not mask_bool[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask_bool[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if len(coords) > best_area:
                best_area = len(coords)
                best_mask.fill(False)
                for rr, cc in coords:
                    best_mask[rr, cc] = True
    return best_mask


def dilate(mask_bool: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    image = Image.fromarray((mask_bool * 255).astype(np.uint8), mode="L")
    for _ in range(iterations):
        image = image.filter(ImageFilter.MaxFilter(size=kernel_size))
    return np.array(image) > 0


def shrink(mask_bool: np.ndarray, target_coverage: float, max_iters: int = 12, kernel_size: int = 5) -> np.ndarray:
    image = Image.fromarray((mask_bool * 255).astype(np.uint8), mode="L")
    for _ in range(max_iters):
        coverage = compute_coverage(np.array(image))
        if coverage <= target_coverage:
            break
        image = image.filter(ImageFilter.MinFilter(size=kernel_size))
    return np.array(image) > 0


def grow(mask_bool: np.ndarray, target_coverage: float, max_iters: int = 12, kernel_size: int = 5) -> np.ndarray:
    image = Image.fromarray((mask_bool * 255).astype(np.uint8), mode="L")
    for _ in range(max_iters):
        coverage = compute_coverage(np.array(image))
        if coverage >= target_coverage:
            break
        image = image.filter(ImageFilter.MaxFilter(size=kernel_size))
    return np.array(image) > 0


def save_mask(mask_path: Path, mask_array: np.ndarray, backup_dir: Path | None) -> None:
    if backup_dir:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mask_path, backup_dir / mask_path.name)
    Image.fromarray(mask_array, mode="L").save(mask_path)


def compute_coverage(mask_array: np.ndarray) -> float:
    return float((mask_array > 0).mean())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    paths = load_paths()
    masks_dir: Path = paths["masks"]
    images_dir: Path = paths["images"]
    backup_dir = args.backup_dir
    if backup_dir is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_dir = masks_dir / f"fix_backup_{timestamp}"

    flagged = load_flagged_cases(args.qc_csv, args.flags)
    if args.max_cases:
        flagged = flagged.head(args.max_cases)
    if flagged.empty:
        LOGGER.info("No flagged masks found matching %s", args.flags)
        return

    checkpoint = resolve_checkpoint(args.model_key, args.model_path, paths["models"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Loading checkpoint %s on %s", checkpoint, device)
    model = load_unet_checkpoint(checkpoint, device=device)
    model.eval()

    summary: List[Dict[str, float]] = []
    for _, row in flagged.iterrows():
        mask_path = Path(row["mask_path"])
        image_name = mask_path.name.replace("_mask.png", ".png")
        image_path = images_dir / image_name
        if not image_path.exists():
            LOGGER.warning("Image %s missing for mask %s; skipping.", image_name, mask_path)
            continue
        flags = row["flags_list"]
        threshold = args.high_threshold if "coverage_high" in flags else args.default_threshold
        raw_mask = infer_mask(image_path, model, device, threshold=threshold)
        mask_bool = raw_mask > 0
        if "fragmented" in flags:
            mask_bool = keep_largest_component(mask_bool)
            mask_bool = dilate(mask_bool, kernel_size=5, iterations=2)
        current_cov = compute_coverage(mask_bool.astype(np.uint8))
        if current_cov < args.min_coverage:
            mask_bool = grow(mask_bool, target_coverage=args.min_coverage, kernel_size=5)
            current_cov = compute_coverage(mask_bool.astype(np.uint8))
        if "coverage_high" in flags or current_cov > args.target_coverage:
            mask_bool = shrink(mask_bool, target_coverage=args.target_coverage, kernel_size=7)
            current_cov = compute_coverage(mask_bool.astype(np.uint8))
        cleaned_mask = (mask_bool.astype(np.uint8) * 255)
        if cleaned_mask.sum() == 0:
            LOGGER.warning("Model produced empty mask for %s; restoring original.", mask_path.name)
            continue
        before_cov = float(row.get("coverage", 0.0))
        after_cov = compute_coverage(cleaned_mask)
        LOGGER.info(
            "Updated %s (flags=%s): coverage %.3f -> %.3f",
            mask_path.name,
            ",".join(flags) or "n/a",
            before_cov,
            after_cov,
        )
        save_mask(mask_path, cleaned_mask, backup_dir=backup_dir)
        summary.append(
            {
                "mask": mask_path.name,
                "flags": ",".join(flags),
                "coverage_before": before_cov,
                "coverage_after": after_cov,
            }
        )

    if summary:
        log_path = PROJECT_ROOT / "logs" / "nih_mask_fix_summary.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        LOGGER.info("Wrote fix summary to %s", log_path)
    LOGGER.info("Mask fixes complete (%d cases).", len(summary))


if __name__ == "__main__":
    main()
