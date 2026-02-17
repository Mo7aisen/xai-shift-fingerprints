#!/usr/bin/env python3
"""Build a compact journal-facing results table from Gate3/Gate5/baseline artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile journal results summary tables.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--gate3-csv", type=Path, required=True)
    parser.add_argument("--gate5-json", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    g3 = pd.read_csv(args.gate3_csv)
    g5 = json.loads(args.gate5_json.read_text(encoding="utf-8"))
    base = pd.read_csv(args.baseline_csv)

    rows: list[dict[str, object]] = []
    for endpoint, sub in g3.groupby("endpoint"):
        clinical = g5["endpoints"].get(endpoint, {})
        rows.append(
            {
                "experiment": args.experiment,
                "endpoint": endpoint,
                "auroc_mean": float(sub["auroc"].mean()),
                "aupr_mean": float(sub["aupr"].mean()) if "aupr" in sub.columns else float("nan"),
                "fpr95_mean": float(sub["fpr95"].mean()),
                "ece_mean": float(sub["ece"].mean()),
                "brier_mean": float(sub["brier"].mean()) if "brier" in sub.columns else float("nan"),
                "pearson_all_mean": float(clinical.get("pearson_all_mean", float("nan"))),
                "spearman_all_mean": float(clinical.get("spearman_all_mean", float("nan"))),
                "gate3_rows": int(len(sub)),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("endpoint")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    lines = [
        "# Journal Main Results",
        "",
        f"- Experiment: `{args.experiment}`",
        "",
        "## Fingerprint Endpoints",
        "",
        "| Endpoint | AUROC | AUPR | FPR95 | ECE | Brier | Pearson(all) | Spearman(all) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in out_df.iterrows():
        lines.append(
            f"| {row['endpoint']} | {row['auroc_mean']:.4f} | {row['aupr_mean']:.4f} | {row['fpr95_mean']:.4f} | "
            f"{row['ece_mean']:.4f} | {row['brier_mean']:.4f} | {row['pearson_all_mean']:.4f} | {row['spearman_all_mean']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## OOD Baselines",
            "",
            "| Method | AUROC | 95% CI |",
            "|---|---:|---|",
        ]
    )
    for _, row in base.iterrows():
        lines.append(f"| {row['method']} | {row['auc']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] |")

    lines.extend(["", f"- CSV: `{args.out_csv}`", f"- Baselines source: `{args.baseline_csv}`"])
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[INFO] wrote {args.out_csv}")
    print(f"[INFO] wrote {args.out_md}")


if __name__ == "__main__":
    main()
