#!/usr/bin/env python3
"""
BBox-stratified attribution analysis for NIH fingerprints.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "nih_chestxray14_manifest.csv"
FINGERPRINTS_ROOT = PROJECT_ROOT / "data" / "fingerprints"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "external_validation" / "bbox_stratified"

LOGGER = logging.getLogger("bbox_stratified")

DATASETS = ["nih_baseline", "jsrt_to_nih", "montgomery_to_nih"]
SIZE_BINS = {"no_bbox": (-np.inf, 0.0), "small": (0.0, 0.05), "medium": (0.05, 0.15), "large": (0.15, np.inf)}


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Image Index"] = df["Image Index"].astype(str)
    df["bbox_area_norm"] = df.apply(lambda row: compute_area_ratio(row), axis=1)
    df["Lesion Size"] = df["bbox_area_norm"].apply(classify_size)
    df["Lesion Laterality"] = df.apply(determine_laterality, axis=1)
    return df[
        [
            "Image Index",
            "Lesion Size",
            "Lesion Laterality",
            "bbox_area_norm",
            "Finding Labels",
            "Patient Age",
            "Patient Sex",
        ]
    ]


def compute_area_ratio(row: pd.Series) -> float:
    if not row.get("Has BBox"):
        return 0.0
    width = row.get("Original Width") or 0
    height = row.get("Original Height") or 0
    if not width or not height:
        return 0.0
    boxes = json.loads(row.get("BBox Records", "[]"))
    total = 0.0
    for box in boxes:
        total += max(box.get("w", 0), 0) * max(box.get("h", 0), 0)
    return total / (width * height) if width * height > 0 else 0.0


def classify_size(ratio: float) -> str:
    for label, (low, high) in SIZE_BINS.items():
        if low < ratio <= high:
            return label
    return "no_bbox"


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


def load_fingerprint(dataset: str, root: Path) -> pd.DataFrame:
    parquet_files = list((root / dataset).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for {dataset}.")
    frames = [pd.read_parquet(file) for file in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    df["sample_id"] = df["sample_id"].astype(str)
    return df


def summarise(groups: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    summary = groups[metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def perform_tests(df: pd.DataFrame, metric: str, size_a: str, size_b: str) -> Dict[str, float]:
    group_a = df[df["Lesion Size"] == size_a][metric]
    group_b = df[df["Lesion Size"] == size_b][metric]
    if len(group_a) < 5 or len(group_b) < 5:
        return {"p_value": float("nan"), "effect": float("nan")}
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    effect = group_a.mean() - group_b.mean()
    return {"p_value": float(p_value), "effect": float(effect)}


def analyse_dataset(dataset: str, manifest: pd.DataFrame, root: Path, out_dir: Path) -> None:
    df = load_fingerprint(dataset, root).merge(
        manifest, left_on="sample_id", right_on="Image Index", how="left"
    )
    metrics = ["dice", "coverage_auc", "attribution_abs_sum"]
    size_stats = summarise(df.groupby("Lesion Size"), metrics)
    laterality_stats = summarise(df.groupby("Lesion Laterality"), metrics)
    out_dir.mkdir(parents=True, exist_ok=True)
    size_stats.to_csv(out_dir / f"{dataset}_size_stats.csv", index=False)
    laterality_stats.to_csv(out_dir / f"{dataset}_laterality_stats.csv", index=False)

    tests = perform_tests(df, "dice", "large", "small")
    with (out_dir / f"{dataset}_tests.json").open("w", encoding="utf-8") as handle:
        json.dump(tests, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bbox-stratified attribution analyses.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="Manifest CSV with bbox metadata.")
    parser.add_argument(
        "--fingerprints-root",
        type=Path,
        default=FINGERPRINTS_ROOT,
        help="Root directory containing fingerprint parquet files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DATASETS,
        help="Fingerprint experiment keys to analyse.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for reports.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    manifest = load_manifest(args.manifest)
    for dataset in args.datasets:
        LOGGER.info("Analysing %s ...", dataset)
        try:
            analyse_dataset(dataset, manifest, args.fingerprints_root, args.output_dir)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Failed to analyse %s: %s", dataset, exc)
            raise
    LOGGER.info("BBox-stratified analyses complete. Outputs saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
