#!/usr/bin/env python3
"""
NIH ChestXray14 Mask QA Toolkit
================================

Features
--------
* Stratified sampling across finding label / sex / age buckets
* Optional watch mode for continuous monitoring
* Advanced metrics (bbox IoU, precision/recall proxies)
* Publication-ready plots saved under reports/external_validation/nih_mask_qc_plots/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from PIL import Image

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - headless environments
    plt = None

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"
DEFAULT_FINDINGS = PROJECT_ROOT / "reports" / "external_validation" / "nih_mask_qc_findings.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports" / "external_validation" / "nih_mask_qc_summary.json"
DEFAULT_FIGURE = PROJECT_ROOT / "reports" / "external_validation" / "nih_mask_qc_samples.png"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "reports" / "external_validation" / "nih_mask_qc_plots"
MANIFEST_PATH = PROJECT_ROOT / "data" / "nih_chestxray14_manifest.csv"

LOGGER = logging.getLogger("nih_mask_qc")
AGE_BUCKETS = [0, 40, 55, 70, 120]
AGE_LABELS = ["<40", "40-55", "55-70", "70+"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def list_masks(masks_dir: Path) -> pd.DataFrame:
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory {masks_dir} missing.")
    files = sorted(p for p in masks_dir.glob("*.png"))
    if not files:
        raise RuntimeError("No masks discovered.")
    return pd.DataFrame(
        {
            "mask_path": files,
            "image_name": [p.name.replace("_mask", "") for p in files],
        }
    )


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest {path} missing. Run prepare_nih_chestxray14.py first.")
    manifest = pd.read_csv(path)
    manifest["Patient Sex"] = manifest["Patient Sex"].fillna("Unknown")
    manifest["Finding Labels"] = manifest["Finding Labels"].fillna("No Finding")
    manifest["Primary Finding"] = manifest["Finding Labels"].apply(lambda text: text.split("|")[0] if text else "No Finding")
    manifest["Age Bucket"] = pd.cut(
        manifest["Patient Age"].fillna(-1), AGE_BUCKETS, labels=AGE_LABELS, right=False
    ).astype(str)
    manifest["BBox Records"] = manifest["BBox Records"].fillna("[]")
    manifest["Has BBox"] = manifest["Has BBox"].fillna(False)
    manifest["bbox_area_norm"] = manifest.apply(
        lambda row: compute_bbox_area_ratio(row), axis=1
    )
    manifest["Lesion Size"] = manifest["bbox_area_norm"].apply(bin_size)
    manifest["Lesion Laterality"] = manifest.apply(determine_laterality, axis=1)
    return manifest


def compute_bbox_area_ratio(row: pd.Series) -> float:
    if not row.get("Has BBox"):
        return 0.0
    width = row.get("Original Width") or 0
    height = row.get("Original Height") or 0
    if not width or not height:
        return 0.0
    area = float(width * height)
    boxes = json.loads(row.get("BBox Records", "[]"))
    total = 0.0
    for box in boxes:
        total += max(box.get("w", 0), 0) * max(box.get("h", 0), 0)
    return total / area if area > 0 else 0.0


def bin_size(ratio: float) -> str:
    if ratio <= 0:
        return "no_bbox"
    if ratio < 0.05:
        return "small"
    if ratio < 0.15:
        return "medium"
    return "large"


def determine_laterality(row: pd.Series) -> str:
    if not row.get("Has BBox"):
        return "no_bbox"
    width = row.get("Original Width") or 0
    if not width:
        return "unknown"
    centers = []
    for box in json.loads(row.get("BBox Records", "[]")):
        centers.append((box.get("x", 0) + box.get("w", 0) / 2) / width)
    if not centers:
        return "unknown"
    mean_center = float(np.mean(centers))
    if mean_center < 0.45:
        return "left"
    if mean_center > 0.55:
        return "right"
    return "central"


def connected_components(mask_bool: np.ndarray) -> tuple[int, np.ndarray]:
    if ndi is not None:
        labels, count = ndi.label(mask_bool)
        if count == 0:
            return 0, np.array([])
        areas = np.bincount(labels.ravel())[1:]
        return int(count), areas
    visited = np.zeros(mask_bool.shape, dtype=bool)
    areas: List[int] = []
    rows, cols = mask_bool.shape
    for r in range(rows):
        for c in range(cols):
            if not mask_bool[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            size = 0
            while stack:
                cr, cc = stack.pop()
                size += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask_bool[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            areas.append(size)
    return len(areas), np.array(areas, dtype=int)


def build_bbox_mask(record: pd.Series, shape: tuple[int, int]) -> np.ndarray:
    boxes = json.loads(record.get("BBox Records", "[]"))
    mask = np.zeros(shape, dtype=bool)
    if not boxes:
        return mask
    for box in boxes:
        x = int(max(box.get("x", 0), 0))
        y = int(max(box.get("y", 0), 0))
        w = int(max(box.get("w", 0), 0))
        h = int(max(box.get("h", 0), 0))
        mask[y : y + h, x : x + w] = True
    return mask


def analyze_sample(mask_path: Path, metadata: pd.Series) -> Dict[str, float | str]:
    mask_arr = np.array(Image.open(mask_path).convert("L"))
    mask_bool = mask_arr > 0
    total_pixels = mask_bool.size
    coverage = float(mask_bool.sum()) / total_pixels
    components, areas = connected_components(mask_bool)
    largest_component_ratio = float(areas.max() / mask_bool.sum()) if areas.size else 0.0
    bbox_area_ratio = compute_bbox_area_ratio(metadata)
    touches_edge = bool(
        mask_bool[0, :].any()
        or mask_bool[-1, :].any()
        or mask_bool[:, 0].any()
        or mask_bool[:, -1].any()
    )
    hole_frac = 0.0
    if ndi is not None:
        filled = ndi.binary_fill_holes(mask_bool)
        hole_frac = float(filled.sum() - mask_bool.sum()) / total_pixels

    bbox_mask = build_bbox_mask(metadata, mask_bool.shape)
    intersection = float(np.logical_and(mask_bool, bbox_mask).sum())
    union = float(np.logical_or(mask_bool, bbox_mask).sum())
    bbox_pixels = float(bbox_mask.sum())
    iou = intersection / union if union > 0 else 0.0
    precision = intersection / mask_bool.sum() if mask_bool.sum() > 0 else 0.0
    recall = intersection / bbox_pixels if bbox_pixels > 0 else 0.0

    flags: List[str] = []
    if coverage == 0:
        flags.append("empty_mask")
    elif coverage < 0.45:
        flags.append("coverage_low")
    if coverage > 0.90:
        flags.append("coverage_high")
    if components > 6:
        flags.append("fragmented")
    if areas.size and largest_component_ratio < 0.40:
        flags.append("scattered")

    return {
        "mask_path": str(mask_path),
        "image_name": metadata["Image Index"],
        "coverage": coverage,
        "bbox_area_ratio": bbox_area_ratio,
        "component_count": components,
        "largest_component_ratio": largest_component_ratio,
        "touches_edge": touches_edge,
        "hole_ratio": hole_frac,
        "flags": ";".join(flags),
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "primary_finding": metadata["Primary Finding"],
        "sex": metadata["Patient Sex"],
        "age_bucket": metadata["Age Bucket"],
    }


def stratified_sample(population: pd.DataFrame, sample_size: int, strat_cols: Sequence[str], seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    df = population.copy()
    if not strat_cols:
        return df.sample(n=min(sample_size, len(df)), random_state=seed)
    df["stratum"] = df[strat_cols].fillna("unknown").agg("|".join, axis=1)
    strata = df["stratum"].unique()
    samples: List[pd.DataFrame] = []
    remaining = sample_size
    for stratum in strata:
        subset = df[df["stratum"] == stratum]
        if subset.empty:
            continue
        proportion = len(subset) / len(df)
        target = max(1, math.floor(proportion * sample_size))
        target = min(target, len(subset))
        selected = subset.sample(n=target, random_state=rng.randint(0, 1_000_000))
        samples.append(selected)
        remaining -= len(selected)
    combined = pd.concat(samples, ignore_index=True)
    if remaining > 0:
        leftovers = df[~df.index.isin(combined.index)]
        extra = leftovers.sample(n=min(remaining, len(leftovers)), random_state=rng.randint(0, 1_000_000))
        combined = pd.concat([combined, extra], ignore_index=True)
    return combined.head(sample_size)


def render_overlays(rows: pd.DataFrame, images_dir: Path, figure_path: Path, max_samples: int = 16) -> None:
    if plt is None:
        LOGGER.warning("matplotlib unavailable; skipping overlays.")
        return
    samples = rows.head(max_samples)
    cols = 4
    n = len(samples)
    rows_count = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(cols * 4, rows_count * 4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, row) in zip(axes, samples.iterrows()):
        image_path = images_dir / row["image_name"]
        mask_path = Path(row["mask_path"])
        if not image_path.exists():
            ax.set_title(f"Missing {image_path.name}")
            continue
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")
        img_arr = np.array(image)
        mask_arr = np.array(mask)
        ax.imshow(img_arr, cmap="gray")
        ax.imshow(mask_arr, cmap="viridis", alpha=0.35)
        ax.set_title(f"{row['image_name']} | cov={row['coverage']:.2f}", fontsize=9)
        ax.axis("off")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=250)
    plt.close(fig)
    LOGGER.info("Saved overlay figure to %s", figure_path)


def render_plots(df: pd.DataFrame, output_dir: Path) -> None:
    if plt is None:
        LOGGER.warning("matplotlib unavailable; skipping QA plots.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df["coverage"], bins=30, kde=True, ax=ax)
    ax.set_title("Lung-mask coverage distribution")
    ax.set_xlabel("Coverage")
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_hist.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="primary_finding", y="coverage", data=df, ax=ax)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.set_title("Coverage by primary finding")
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_by_finding.png", dpi=200)
    plt.close(fig)

    if (df["iou"] > 0).any():
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[df["iou"] > 0]["iou"], bins=30, kde=True, ax=ax)
        ax.set_title("IoU vs. NIH BBoxes")
        fig.tight_layout()
        fig.savefig(output_dir / "iou_hist.png", dpi=200)
        plt.close(fig)


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    flagged = df[df["flags"] != ""]
    pass_rate = 1 - len(flagged) / len(df)
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "sample_size": int(len(df)),
        "coverage_mean": float(df["coverage"].mean()),
        "coverage_std": float(df["coverage"].std(ddof=0)),
        "bbox_area_mean": float(df["bbox_area_ratio"].mean()),
        "component_mean": float(df["component_count"].mean()),
        "hole_ratio_mean": float(df["hole_ratio"].mean()),
        "iou_mean": float(df["iou"].mean()),
        "precision_mean": float(df["precision"].mean()),
        "recall_mean": float(df["recall"].mean()),
        "flagged_count": int(len(flagged)),
        "pass_rate": pass_rate,
        "high_coverage": int((df["coverage"] > 0.9).sum()),
        "low_coverage": int((df["coverage"] < 0.45).sum()),
    }
    return summary


def write_outputs(df: pd.DataFrame, summary: Dict[str, float], findings_path: Path, summary_path: Path) -> None:
    findings_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(findings_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote findings to %s", findings_path)
    LOGGER.info("Saved summary to %s", summary_path)


def run_once(args: argparse.Namespace) -> None:
    paths_cfg = load_yaml(CONFIG_PATH)
    datasets_root = Path(paths_cfg["paths"]["datasets_root"])
    nih_cfg = paths_cfg["datasets"]["nih_chestxray14"]
    images_dir = datasets_root / nih_cfg["images"]
    masks_dir = datasets_root / nih_cfg["masks"]

    mask_inventory = list_masks(masks_dir)
    manifest = load_manifest(args.manifest)
    merged = mask_inventory.merge(manifest, left_on="image_name", right_on="Image Index", how="left")
    if merged["Image Index"].isna().any():
        LOGGER.warning("Some masks lack manifest metadata; consider re-running prepare_nih_chestxray14.py.")
    sample = stratified_sample(
        merged,
        sample_size=args.samples,
        strat_cols=args.stratify if args.stratify else [],
        seed=args.seed,
    )
    LOGGER.info("Selected %d samples across %d strata.", len(sample), sample["stratum"].nunique() if "stratum" in sample else 1)

    records = [analyze_sample(Path(row["mask_path"]), row) for _, row in sample.iterrows()]
    df = pd.DataFrame(records)
    write_outputs(df, summarize(df), args.output, args.summary)
    render_overlays(df, images_dir, args.figure, max_samples=16)
    render_plots(df, args.plots_dir)


def get_latest_mtime(paths: Sequence[Path]) -> float:
    mtimes = []
    for path in paths:
        if path.is_dir():
            mtimes.append(path.stat().st_mtime)
        elif path.exists():
            mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIH mask QA with stratification + watch mode.")
    parser.add_argument("--samples", type=int, default=300, help="Number of masks to audit.")
    parser.add_argument("--seed", type=int, default=2025, help="Sampling seed.")
    parser.add_argument(
        "--stratify",
        nargs="*",
        default=["Primary Finding", "Patient Sex", "Age Bucket"],
        help="Manifest columns to stratify on.",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Manifest CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_FINDINGS, help="CSV output path.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Summary JSON path.")
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE, help="Overlay PNG path.")
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR, help="Directory for additional plots.")
    parser.add_argument("--watch", action="store_true", help="Enable continuous monitoring.")
    parser.add_argument("--watch-interval", type=int, default=900, help="Seconds between watch cycles.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not args.watch:
        run_once(args)
        return
    LOGGER.info("Entering watch mode (interval=%ss)", args.watch_interval)
    cfg = load_yaml(CONFIG_PATH)
    datasets_root = Path(cfg["paths"]["datasets_root"])
    masks_dir = datasets_root / cfg["datasets"]["nih_chestxray14"]["masks"]
    watched_paths = [masks_dir, args.manifest]
    last_mtime = 0.0
    while True:
        current_mtime = get_latest_mtime(watched_paths)
        if current_mtime > last_mtime:
            LOGGER.info("Change detected; running QA.")
            try:
                run_once(args)
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("QA run failed: %s", exc)
            last_mtime = current_mtime
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
