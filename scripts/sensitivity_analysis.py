#!/usr/bin/env python3
"""Sensitivity analysis for QA thresholds and bbox binning."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QA_CSV = PROJECT_ROOT / "reports" / "external_validation" / "nih_mask_qc_findings.csv"
MANIFEST_CSV = PROJECT_ROOT / "data" / "nih_chestxray14_manifest.csv"
FINGERPRINT_ROOT = PROJECT_ROOT / "data" / "fingerprints"
TABLE_DIR = PROJECT_ROOT / "results" / "tables" / "sensitivity"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "sensitivity"

DATASETS = ["nih_baseline", "jsrt_to_nih", "montgomery_to_nih"]
COVERAGE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]
BIN_CONFIGS = {
    "default": [("small", 0.0, 0.05), ("medium", 0.05, 0.15), ("large", 0.15, np.inf)],
    "tight": [("small", 0.0, 0.03), ("medium", 0.03, 0.12), ("large", 0.12, np.inf)],
    "wide": [("small", 0.0, 0.07), ("medium", 0.07, 0.20), ("large", 0.20, np.inf)],
}

LOGGER = logging.getLogger("sensitivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sensitivity analyses.")
    parser.add_argument("--qa-csv", type=Path, default=QA_CSV)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--fingerprints-root", type=Path, default=FINGERPRINT_ROOT)
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--table-dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIG_DIR)
    return parser.parse_args()


def coverage_threshold_analysis(df: pd.DataFrame, out_csv: Path, fig_dir: Path) -> None:
    records: List[Dict[str, float]] = []
    n = len(df)
    for thr in COVERAGE_THRESHOLDS:
        flagged = (df["coverage"] > thr).sum()
        successes = n - flagged
        ci_low, ci_high = stats.binomtest(successes, n).proportion_ci(confidence_level=0.95, method="wilson")
        records.append(
            {
                "threshold": thr,
                "flagged": int(flagged),
                "pass_rate": successes / n,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    coverage_df = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    coverage_df.to_csv(out_csv, index=False)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=coverage_df, x="threshold", y="pass_rate", marker="o")
    plt.fill_between(coverage_df["threshold"], coverage_df["ci_low"], coverage_df["ci_high"], alpha=0.2)
    plt.ylim(0.9, 1.01)
    plt.ylabel("QA pass rate")
    plt.xlabel("Coverage upper threshold")
    plt.title("Coverage threshold sensitivity")
    plt.tight_layout()
    plt.savefig(fig_dir / "coverage_thresholds.png", dpi=200)
    plt.close()


def compute_bbox_area_ratio(row: pd.Series) -> float:
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
    return total / (width * height)


def assign_bin(ratio: float, bins: List[Tuple[str, float, float]]) -> str:
    for label, low, high in bins:
        if low <= ratio < high:
            return label
    return "no_bbox"


def load_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    manifest["bbox_ratio"] = manifest.apply(compute_bbox_area_ratio, axis=1)
    manifest["Image Index"] = manifest["Image Index"].astype(str)
    return manifest


def load_fingerprint(dataset: str, root: Path) -> pd.DataFrame:
    files = sorted((root / dataset).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No fingerprint parquet for {dataset}")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["sample_id"] = df["sample_id"].astype(str)
    df["image_key"] = df["sample_id"].apply(lambda s: s if s.endswith(".png") else f"{s}.png")
    return df


def bbox_bin_analysis(manifest: pd.DataFrame, root: Path, datasets: List[str], out_dir: Path, fig_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        df = load_fingerprint(dataset, root).merge(manifest, left_on="image_key", right_on="Image Index", how="left")
        violin_plot_data = []
        summaries: List[Dict[str, float]] = []
        for cfg_name, bins in BIN_CONFIGS.items():
            df_cfg = df.assign(bin_label=df["bbox_ratio"].apply(lambda r, b=bins: assign_bin(r, b)))
            for label in sorted(df_cfg["bin_label"].unique()):
                subset = df_cfg[df_cfg["bin_label"] == label]
                if subset.empty:
                    continue
                summaries.append(
                    {
                        "dataset": dataset,
                        "config": cfg_name,
                        "bin": label,
                        "dice_mean": subset["dice"].mean(),
                        "dice_std": subset["dice"].std(ddof=0),
                        "dice_count": len(subset),
                    }
                )
            large = df_cfg[df_cfg["bin_label"] == "large"]["dice"]
            small = df_cfg[df_cfg["bin_label"] == "small"]["dice"]
            if len(large) >= 5 and len(small) >= 5:
                _, p_val = stats.ttest_ind(large, small, equal_var=False)
                effect = large.mean() - small.mean()
            else:
                p_val = np.nan
                effect = np.nan
            summaries.append(
                {
                    "dataset": dataset,
                    "config": cfg_name,
                    "bin": "large_vs_small_effect",
                    "dice_mean": effect,
                    "dice_std": np.nan,
                    "dice_count": p_val,
                }
            )
            if cfg_name == "default":
                violin_plot_data.append(df_cfg.assign(config=cfg_name))
        pd.DataFrame(summaries).to_csv(out_dir / f"{dataset}_bbox_sensitivity.csv", index=False)
        if violin_plot_data:
            plot_df = pd.concat(violin_plot_data, ignore_index=True)
            plt.figure(figsize=(7, 4))
            sns.violinplot(x="bin_label", y="dice", data=plot_df)
            plt.title(f"Dice vs lesion size ({dataset})")
            plt.tight_layout()
            plt.savefig(fig_dir / f"{dataset}_dice_violin.png", dpi=200)
            plt.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    qa_df = pd.read_csv(args.qa_csv)
    coverage_threshold_analysis(qa_df, args.table_dir / "coverage_thresholds.csv", args.figure_dir)
    manifest = load_manifest(args.manifest)
    bbox_bin_analysis(manifest, args.fingerprints_root, args.datasets, args.table_dir, args.figure_dir)
    LOGGER.info("Sensitivity analysis complete.")


if __name__ == "__main__":
    main()
