#!/usr/bin/env python3
"""Run journal-ready OOD baselines for a configured cross-dataset experiment.

Methods:
- ResNet-50 Mahalanobis feature distance
- UNet entropy score
- UNet MSP score
- UNet MaxLogit proxy
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run OOD baseline suite for one experiment.")
    parser.add_argument("--experiment", required=True, help="Experiment key from configs/experiments.yaml.")
    parser.add_argument("--subset", default="full", help="Subset under cache root.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional per-dataset cap.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--experiments-config", type=Path, default=repo_root / "configs" / "experiments.yaml")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "results" / "baselines")
    return parser.parse_args()


def _load_experiment(experiments_path: Path, experiment: str) -> tuple[str, str]:
    cfg = yaml.safe_load(experiments_path.read_text(encoding="utf-8"))
    item = cfg["experiments"][experiment]
    train = str(item["train_dataset"])
    tests = list(item.get("test_datasets", []))
    if len(tests) != 1:
        raise ValueError(f"Expected exactly one test dataset for '{experiment}', got: {tests}")
    return train, str(tests[0])


def _run(cmd: list[str], cwd: Path) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _select_scope(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if "scope" in df.columns and (df["scope"] == scope).any():
        return df[df["scope"] == scope].copy()
    if "scope" in df.columns and (df["scope"] == "overall").any():
        return df[df["scope"] == "overall"].copy()
    return df.copy()


def _maybe_float(row: pd.Series, key: str) -> float:
    value = row.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    id_dataset, ood_dataset = _load_experiment(args.experiments_config, args.experiment)
    datasets_arg = f"{id_dataset},{ood_dataset}"
    in_datasets_arg = id_dataset
    scope = f"{id_dataset} vs {ood_dataset}"

    out_dir = args.output_dir / f"{args.experiment}_{args.subset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    common_args = [
        "--subset",
        args.subset,
        "--num-workers",
        str(args.num_workers),
        "--max-samples",
        str(args.max_samples),
        "--output-dir",
        str(out_dir),
        "--datasets",
        datasets_arg,
        "--in-datasets",
        in_datasets_arg,
    ]

    _run(
        [
            sys.executable,
            "scripts/run_resnet_feature_baseline.py",
            "--batch-size",
            str(max(1, args.batch_size)),
            *common_args,
        ],
        cwd=repo_root,
    )
    _run(
        [
            sys.executable,
            "scripts/run_energy_ood_baseline.py",
            "--batch-size",
            str(max(1, args.batch_size // 2)),
            *common_args,
        ],
        cwd=repo_root,
    )

    resnet_auc = pd.read_csv(out_dir / "resnet_ood_auc.csv")
    energy_auc = pd.read_csv(out_dir / "energy_ood_auc.csv")

    resnet_row = _select_scope(resnet_auc, scope).iloc[0]
    energy_rows = _select_scope(energy_auc, scope)

    baseline_rows: list[dict[str, object]] = [
        {
            "experiment": args.experiment,
            "subset": args.subset,
            "id_dataset": id_dataset,
            "ood_dataset": ood_dataset,
            "scope": str(resnet_row.get("scope", scope)),
            "method": "mahalanobis_resnet50",
            "auc": float(resnet_row["auc"]),
            "ci_low": float(resnet_row["ci_low"]),
            "ci_high": float(resnet_row["ci_high"]),
            "aupr": _maybe_float(resnet_row, "aupr"),
            "aupr_ci_low": _maybe_float(resnet_row, "aupr_ci_low"),
            "aupr_ci_high": _maybe_float(resnet_row, "aupr_ci_high"),
            "fpr95": _maybe_float(resnet_row, "fpr95"),
            "fpr95_ci_low": _maybe_float(resnet_row, "fpr95_ci_low"),
            "fpr95_ci_high": _maybe_float(resnet_row, "fpr95_ci_high"),
            "tpr_at_fpr05": _maybe_float(resnet_row, "tpr_at_fpr05"),
            "tpr_at_fpr05_ci_low": _maybe_float(resnet_row, "tpr_at_fpr05_ci_low"),
            "tpr_at_fpr05_ci_high": _maybe_float(resnet_row, "tpr_at_fpr05_ci_high"),
            "ece": _maybe_float(resnet_row, "ece"),
            "brier": _maybe_float(resnet_row, "brier"),
        }
    ]

    metric_map = {
        "entropy": "unet_entropy",
        "msp": "unet_msp",
        "maxlogit": "unet_maxlogit",
    }
    for metric, method_name in metric_map.items():
        sub = energy_rows[energy_rows["metric"] == metric]
        if sub.empty:
            continue
        row = sub.iloc[0]
        baseline_rows.append(
            {
                "experiment": args.experiment,
                "subset": args.subset,
                "id_dataset": id_dataset,
                "ood_dataset": ood_dataset,
                "scope": str(row.get("scope", scope)),
                "method": method_name,
                "auc": float(row["auc"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
                "aupr": _maybe_float(row, "aupr"),
                "aupr_ci_low": _maybe_float(row, "aupr_ci_low"),
                "aupr_ci_high": _maybe_float(row, "aupr_ci_high"),
                "fpr95": _maybe_float(row, "fpr95"),
                "fpr95_ci_low": _maybe_float(row, "fpr95_ci_low"),
                "fpr95_ci_high": _maybe_float(row, "fpr95_ci_high"),
                "tpr_at_fpr05": _maybe_float(row, "tpr_at_fpr05"),
                "tpr_at_fpr05_ci_low": _maybe_float(row, "tpr_at_fpr05_ci_low"),
                "tpr_at_fpr05_ci_high": _maybe_float(row, "tpr_at_fpr05_ci_high"),
                "ece": _maybe_float(row, "ece"),
                "brier": _maybe_float(row, "brier"),
            }
        )

    summary_csv = out_dir / f"journal_ood_baselines_{args.experiment}_{args.subset}.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "experiment",
                "subset",
                "id_dataset",
                "ood_dataset",
                "scope",
                "method",
                "auc",
                "ci_low",
                "ci_high",
                "aupr",
                "aupr_ci_low",
                "aupr_ci_high",
                "fpr95",
                "fpr95_ci_low",
                "fpr95_ci_high",
                "tpr_at_fpr05",
                "tpr_at_fpr05_ci_low",
                "tpr_at_fpr05_ci_high",
                "ece",
                "brier",
            ],
        )
        writer.writeheader()
        for row in baseline_rows:
            writer.writerow(row)

    summary_md = out_dir / f"journal_ood_baselines_{args.experiment}_{args.subset}.md"
    lines = [
        "# Journal OOD Baselines",
        "",
        f"- Experiment: `{args.experiment}`",
        f"- ID dataset: `{id_dataset}`",
        f"- OOD dataset: `{ood_dataset}`",
        f"- Subset: `{args.subset}`",
        "",
        "| Method | AUROC | AUPR | FPR95 | TPR@5%FPR | ECE | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in baseline_rows:
        lines.append(
            f"| {row['method']} | {row['auc']:.4f} | {row['aupr']:.4f} | {row['fpr95']:.4f} | "
            f"{row['tpr_at_fpr05']:.4f} | {row['ece']:.4f} | {row['brier']:.4f} |"
        )
    lines.extend(["", f"- CSV: `{summary_csv}`"])
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[INFO] Baseline summary CSV: {summary_csv}")
    print(f"[INFO] Baseline summary MD: {summary_md}")


if __name__ == "__main__":
    main()
