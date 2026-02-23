#!/usr/bin/env python3
"""Compile Gate-4 (IG quality + robustness + ablation) report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Gate-4 report.")
    parser.add_argument("--root", type=Path, default=Path("reports_v2/gate4"))
    parser.add_argument("--experiment", default="jsrt_to_montgomery")
    parser.add_argument("--id-dataset", default="jsrt")
    parser.add_argument("--ood-dataset", default="montgomery")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ig-a", type=int, default=16)
    parser.add_argument("--ig-b", type=int, default=32)
    parser.add_argument("--endpoints", nargs="+", default=["predicted_mask", "mask_free"])
    parser.add_argument("--max-ig-delta", type=float, default=0.02)
    parser.add_argument("--max-robust-degradation", type=float, default=20.0)
    parser.add_argument("--min-ablation-retention", type=float, default=0.95)
    parser.add_argument("--out-json", type=Path, default=Path("reports_v2/audits/GATE4_IG_ROBUSTNESS_SUMMARY.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports_v2/audits/GATE4_IG_ROBUSTNESS_2026-02-17.md"))
    return parser.parse_args()


def _roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return float((rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in {"dataset_key", "subset", "sample_id", "patient_id"} or c.startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _load_pair(base: Path, endpoint: str, experiment: str, id_dataset: str, ood_dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = base / endpoint / experiment
    id_df = pd.read_parquet(path / f"{id_dataset}.parquet")
    ood_df = pd.read_parquet(path / f"{ood_dataset}.parquet")
    return id_df, ood_df


def _scores_and_topk(id_df: pd.DataFrame, ood_df: pd.DataFrame, top_k: int = 20) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = [c for c in _numeric_feature_columns(id_df) if c in ood_df.columns]
    if not cols:
        raise RuntimeError("No shared numeric feature columns for scoring.")
    x_id = id_df[cols]
    x_ood = ood_df[cols]
    mu = x_id.mean(axis=0)
    sd = x_id.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    score_id = ((x_id - mu) / sd).abs().mean(axis=1).to_numpy(dtype=float)
    score_ood = ((x_ood - mu) / sd).abs().mean(axis=1).to_numpy(dtype=float)
    effect = ((x_ood.mean(axis=0) - x_id.mean(axis=0)).abs() / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    topk = effect.sort_values(ascending=False).head(top_k).index.tolist()
    return score_id, score_ood, topk


def _scores_topk(id_df: pd.DataFrame, ood_df: pd.DataFrame, topk: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x_id = id_df[topk]
    x_ood = ood_df[topk]
    mu = x_id.mean(axis=0)
    sd = x_id.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    score_id = ((x_id - mu) / sd).abs().mean(axis=1).to_numpy(dtype=float)
    score_ood = ((x_ood - mu) / sd).abs().mean(axis=1).to_numpy(dtype=float)
    return score_id, score_ood


def _auc_from_scores(score_id: np.ndarray, score_ood: np.ndarray) -> float:
    y = np.concatenate([np.zeros(len(score_id), dtype=int), np.ones(len(score_ood), dtype=int)])
    score = np.concatenate([score_id, score_ood])
    return _roc_auc(y, score)


def main() -> None:
    args = parse_args()
    root = args.root
    artifacts_a = root / "artifacts" / f"ig{args.ig_a}" / f"seed{args.seed}"
    artifacts_b = root / "artifacts" / f"ig{args.ig_b}" / f"seed{args.seed}"

    endpoints = {}
    gate_pass = True
    for endpoint in args.endpoints:
        id_a, ood_a = _load_pair(artifacts_a, endpoint, args.experiment, args.id_dataset, args.ood_dataset)
        id_b, ood_b = _load_pair(artifacts_b, endpoint, args.experiment, args.id_dataset, args.ood_dataset)

        score_id_a, score_ood_a, top20 = _scores_and_topk(id_a, ood_a, top_k=20)
        score_id_b, score_ood_b, _ = _scores_and_topk(id_b, ood_b, top_k=20)
        auc_a = _auc_from_scores(score_id_a, score_ood_a)
        auc_b = _auc_from_scores(score_id_b, score_ood_b)
        ig_delta = abs(auc_b - auc_a)

        score_id_top20, score_ood_top20 = _scores_topk(id_a, ood_a, top20)
        auc_top20 = _auc_from_scores(score_id_top20, score_ood_top20)
        retention = auc_top20 / auc_a if auc_a > 0 else float("nan")

        robust_csv = root / "robustness" / f"{endpoint}_ig{args.ig_a}" / "robustness_summary.csv"
        robust_df = pd.read_csv(robust_csv)
        robust_auc = robust_df[robust_df["metric"] == "auroc_shift_detection"].copy()
        robust_auc["degradation_percent"] = (
            (robust_auc["baseline"] - robust_auc["perturbed"]) / robust_auc["baseline"].abs()
        ) * 100.0
        max_deg = float(robust_auc["degradation_percent"].max())

        pass_ig = bool(ig_delta < args.max_ig_delta)
        pass_ablation = bool(np.isfinite(retention) and retention >= args.min_ablation_retention)
        pass_robust = bool(np.isfinite(max_deg) and max_deg <= args.max_robust_degradation)
        endpoint_pass = pass_ig and pass_ablation and pass_robust
        gate_pass = gate_pass and endpoint_pass

        endpoints[endpoint] = {
            "auc_ig_a": float(auc_a),
            "auc_ig_b": float(auc_b),
            "ig_delta_abs": float(ig_delta),
            "auc_top20": float(auc_top20),
            "ablation_retention": float(retention),
            "robustness_max_degradation_percent": float(max_deg),
            "pass_ig_delta": pass_ig,
            "pass_ablation": pass_ablation,
            "pass_robustness": pass_robust,
            "pass_endpoint": endpoint_pass,
        }

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "experiment": args.experiment,
        "id_dataset": args.id_dataset,
        "ood_dataset": args.ood_dataset,
        "ig_steps_compare": [args.ig_a, args.ig_b],
        "thresholds": {
            "max_ig_delta": args.max_ig_delta,
            "max_robust_degradation_percent": args.max_robust_degradation,
            "min_ablation_retention": args.min_ablation_retention,
        },
        "endpoints": endpoints,
        "gate4_pass": bool(gate_pass),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Gate-4 IG Quality & Robustness Report",
        "",
        f"- Generated UTC: `{summary['generated_utc']}`",
        f"- Experiment: `{args.experiment}`",
        f"- ID dataset: `{args.id_dataset}`",
        f"- OOD dataset: `{args.ood_dataset}`",
        f"- Seed: `{args.seed}`",
        f"- IG compare: `{args.ig_a} vs {args.ig_b}`",
        "",
        "## Endpoint Results",
        "",
        "| Endpoint | AUC(IG16) | AUC(IG32) | |Δ| | Top20 Retention | Max Robust Deg (%) | IG | Ablation | Robust | PASS |",
        "|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for endpoint in args.endpoints:
        e = endpoints[endpoint]
        lines.append(
            "| "
            f"{endpoint} | {e['auc_ig_a']:.4f} | {e['auc_ig_b']:.4f} | {e['ig_delta_abs']:.4f} | "
            f"{e['ablation_retention']:.4f} | {e['robustness_max_degradation_percent']:.2f} | "
            f"{'PASS' if e['pass_ig_delta'] else 'FAIL'} | "
            f"{'PASS' if e['pass_ablation'] else 'FAIL'} | "
            f"{'PASS' if e['pass_robustness'] else 'FAIL'} | "
            f"{'PASS' if e['pass_endpoint'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Final Decision",
            "",
            f"- Gate-4 status: `{'PASS' if gate_pass else 'NO-GO'}`",
            f"- JSON summary: `{args.out_json}`",
        ]
    )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[gate4] completed: {'PASS' if gate_pass else 'NO-GO'}")
    print(f"[gate4] summary -> {args.out_json}")
    print(f"[gate4] report -> {args.out_md}")


if __name__ == "__main__":
    main()
