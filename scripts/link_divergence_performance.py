#!/usr/bin/env python
"""Link divergence metrics with downstream performance deltas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_ROOT = PROJECT_ROOT / "data" / "fingerprints"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "divergence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = {
    "JSRT → Montgomery": {
        "folder": "jsrt_to_montgomery",
        "target_key": "montgomery",
    },
    "JSRT → Shenzhen": {
        "folder": "jsrt_to_shenzhen",
        "target_key": "shenzhen",
    },
}

METRICS_TO_COMPARE = [
    ("dice_mean", "Dice"),
    ("coverage_auc_mean", "Coverage AUC"),
    ("mask_coverage_mean", "Mask Coverage"),
    ("attribution_abs_sum_mean", "Attribution Sum"),
    ("border_abs_sum_mean", "Border Sum"),
    ("hist_entropy_mean", "Histogram Entropy"),
]


def _load_summary(folder: str) -> Dict[str, Dict[str, float]]:
    summary_path = FINGERPRINT_ROOT / folder / "summary.json"
    return json.loads(summary_path.read_text())


def _load_divergence_table() -> pd.DataFrame:
    table_path = PROJECT_ROOT / "divergence_comparison_table.csv"
    if not table_path.exists():
        raise FileNotFoundError("Run create_comparison_table.py before linking performance deltas.")
    return pd.read_csv(table_path)


def _compute_deltas(summary: Dict[str, Dict[str, float]], target_key: str) -> Dict[str, float]:
    ref_stats = summary["jsrt"]
    tgt_stats = summary[target_key]

    deltas: Dict[str, float] = {}
    for metric_key, label in METRICS_TO_COMPARE:
        ref_value = ref_stats[metric_key]
        tgt_value = tgt_stats[metric_key]
        diff = tgt_value - ref_value
        ratio = (tgt_value / ref_value) if ref_value else float("nan")
        deltas[f"{label} Δ"] = diff
        deltas[f"{label} Ratio"] = ratio
    return deltas


def main() -> None:
    divergence_df = _load_divergence_table()
    rows: List[Dict[str, object]] = []

    for comparison, config in EXPERIMENTS.items():
        summary = _load_summary(config["folder"])
        delta_metrics = _compute_deltas(summary, config["target_key"])
        divergence_slice = divergence_df.loc[divergence_df["Comparison"] == comparison].to_dict(orient="records")[0]
        row = {
            "Comparison": comparison,
            "Reference Samples": divergence_slice["Reference Samples"],
            "Target Samples": divergence_slice["Target Samples"],
            "KL Divergence": divergence_slice["KL Divergence"],
            "EMD": divergence_slice["EMD"],
            "Graph Edit Distance": divergence_slice["Graph Edit Distance"],
        }
        row.update(delta_metrics)
        rows.append(row)

    link_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "divergence_performance_link.csv"
    link_df.to_csv(csv_path, index=False)

    md_lines = [
        "# Divergence vs Performance Deltas",
        "",
        "Derived by `scripts/link_divergence_performance.py`.",
        "",
        link_df.to_markdown(index=False),
        "",
    ]
    md_path = OUTPUT_DIR / "divergence_performance_link.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"✓ Saved performance link CSV → {csv_path}")
    print(f"✓ Saved performance link markdown → {md_path}")


if __name__ == "__main__":
    main()
