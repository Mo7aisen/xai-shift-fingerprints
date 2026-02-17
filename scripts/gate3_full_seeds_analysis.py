#!/usr/bin/env python3
"""Gate-3 statistical analysis on full official seeds (42-46).

This script computes the pending statistical criteria for Gate-3:
- AUROC with bootstrap CI per seed and endpoint
- AUROC seed-to-seed CI width
- Feature-stability Jaccard across seeds (top-10 discriminative features)
- Post-hoc power for ID vs OOD score separation

Expected artifact layout:
  <artifacts_root>/seed<SEED>/<endpoint>/<experiment>/{<id_dataset>.parquet,<ood_dataset>.parquet}
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, t


DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_ENDPOINTS = ["predicted_mask", "mask_free"]
DEFAULT_EXPERIMENT = "jsrt_to_montgomery"
DEFAULT_ID_DATASET = "jsrt"
DEFAULT_OOD_DATASET = "montgomery"
N_BOOT = 1000
TOPK = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate-3 full-seeds statistical analysis.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("reports_v2/gate3_seed_artifacts"),
        help="Root directory containing per-seed endpoint artifacts.",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help="Experiment key used during full-seed run.",
    )
    parser.add_argument(
        "--id-dataset",
        default=DEFAULT_ID_DATASET,
        help="In-distribution dataset key used as reference.",
    )
    parser.add_argument(
        "--ood-dataset",
        default=DEFAULT_OOD_DATASET,
        help="Out-of-distribution dataset key used as target.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seed list to analyze.",
    )
    parser.add_argument(
        "--endpoints",
        nargs="+",
        default=DEFAULT_ENDPOINTS,
        help="Endpoint list to analyze.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=N_BOOT,
        help="Number of stratified bootstrap resamples for AUROC CI.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOPK,
        help="Top-k discriminative features for Jaccard stability.",
    )
    parser.add_argument(
        "--seed-bootstrap",
        type=int,
        default=2026,
        help="RNG seed for bootstrap.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports_v2/audits/GATE3_FULL_SEEDS_STATS.csv"),
        help="Output CSV for per-seed metrics.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports_v2/audits/GATE3_FULL_SEEDS_SUMMARY.json"),
        help="Output JSON for endpoint-level summary and pass/fail.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports_v2/audits/GATE3_FULL_SEEDS_STATS_2026-02-17.md"),
        help="Output markdown report.",
    )
    return parser.parse_args()


def _stratified_boot_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    b0 = rng.choice(idx0, size=idx0.size, replace=True)
    b1 = rng.choice(idx1, size=idx1.size, replace=True)
    return np.concatenate([b0, b1])


def _auc_with_ci(
    y: np.ndarray,
    score: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = _roc_auc(y, score)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _stratified_boot_indices(y, rng)
        boots[i] = _roc_auc(y[idx], score[idx])
    low, high = np.quantile(boots, [0.025, 0.975])
    return point, float(low), float(high)


def _aupr(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    ap = 0.0
    prev_recall = 0.0
    for p, r, yi in zip(precision, recall, y_sorted):
        if yi == 1:
            ap += float(p) * float(r - prev_recall)
            prev_recall = float(r)
    return float(ap)


def _aupr_with_ci(
    y: np.ndarray,
    score: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = _aupr(y, score)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = _stratified_boot_indices(y, rng)
        boots[i] = _aupr(y[idx], score[idx])
    low, high = np.quantile(boots, [0.025, 0.975])
    return point, float(low), float(high)


def _roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _fpr95(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    mask = tpr >= 0.95
    if not np.any(mask):
        return float("nan")
    return float(np.min(fpr[mask]))


def _ece(y: np.ndarray, score: np.ndarray, n_bins: int = 10) -> float:
    smin = float(np.min(score))
    smax = float(np.max(score))
    if smax <= smin:
        p = np.full_like(score, 0.5, dtype=float)
    else:
        p = (score - smin) / (smax - smin)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not np.any(m):
            continue
        acc = float(np.mean(y[m]))
        conf = float(np.mean(p[m]))
        w = float(np.mean(m))
        ece += w * abs(acc - conf)
    return float(ece)


def _brier(y: np.ndarray, score: np.ndarray) -> float:
    smin = float(np.min(score))
    smax = float(np.max(score))
    if smax <= smin:
        p = np.full_like(score, 0.5, dtype=float)
    else:
        p = (score - smin) / (smax - smin)
    return float(np.mean((p - y) ** 2))


def _cohen_d(x0: np.ndarray, x1: np.ndarray) -> float:
    n0 = len(x0)
    n1 = len(x1)
    if n0 < 2 or n1 < 2:
        return float("nan")
    v0 = float(np.var(x0, ddof=1))
    v1 = float(np.var(x1, ddof=1))
    pooled = ((n0 - 1) * v0 + (n1 - 1) * v1) / (n0 + n1 - 2)
    if pooled <= 0:
        return float("nan")
    return float((np.mean(x1) - np.mean(x0)) / math.sqrt(pooled))


def _power_from_effect_size(d: float, n0: int, n1: int) -> float:
    if not np.isfinite(d) or d == 0.0 or n0 < 2 or n1 < 2:
        return float("nan")
    # Normal-approximate power for two-sample effect-size test.
    # n_eff is the harmonic-equivalent sample size for two groups.
    n_eff = (n0 * n1) / (n0 + n1)
    if n_eff <= 0:
        return float("nan")
    z_alpha = float(norm.ppf(1.0 - 0.05 / 2.0))
    z = abs(float(d)) * math.sqrt(float(n_eff))
    return float(norm.cdf(z - z_alpha))


def _safe_numeric_features(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    skip = {
        "dataset_key",
        "subset",
        "sample_id",
        "patient_id",
    }
    for c in df.columns:
        if c in skip or c.startswith("_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        out.append(c)
    return out


def _compute_score_and_top_features(
    df_id: pd.DataFrame,
    df_ood: pd.DataFrame,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    common = [c for c in _safe_numeric_features(df_id) if c in df_ood.columns]
    if not common:
        raise RuntimeError("No common numeric features found between ID and OOD tables.")

    x_id = df_id[common].copy()
    x_ood = df_ood[common].copy()

    mu = x_id.mean(axis=0)
    sd = x_id.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)

    z_id = ((x_id - mu) / sd).abs()
    z_ood = ((x_ood - mu) / sd).abs()
    score_id = z_id.mean(axis=1).to_numpy(dtype=float)
    score_ood = z_ood.mean(axis=1).to_numpy(dtype=float)

    effect = (x_ood.mean(axis=0) - x_id.mean(axis=0)).abs() / sd
    effect = effect.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    top_features = effect.sort_values(ascending=False).head(top_k).index.tolist()

    d = _cohen_d(score_id, score_ood)
    return score_id, score_ood, top_features, d


def _pairwise_jaccard(sets: Iterable[set[str]]) -> tuple[float, float, int]:
    s = list(sets)
    if len(s) < 2:
        return float("nan"), float("nan"), 0
    vals: list[float] = []
    for a, b in itertools.combinations(s, 2):
        union = len(a | b)
        if union == 0:
            vals.append(1.0)
        else:
            vals.append(len(a & b) / union)
    return float(np.mean(vals)), float(np.min(vals)), len(vals)


def _seed_ci_width(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    margin = float(t.ppf(0.975, len(arr) - 1) * sem)
    return 2.0 * margin


def main() -> None:
    args = parse_args()
    rng_seed = args.seed_bootstrap

    per_seed_rows: list[dict[str, object]] = []
    top_features_per_endpoint: dict[str, dict[int, set[str]]] = {ep: {} for ep in args.endpoints}

    for endpoint in args.endpoints:
        for seed in args.seeds:
            base = args.artifacts_root / f"seed{seed}" / endpoint / args.experiment
            id_path = base / f"{args.id_dataset}.parquet"
            ood_path = base / f"{args.ood_dataset}.parquet"
            if not id_path.exists() or not ood_path.exists():
                raise FileNotFoundError(
                    f"Missing seed artifact for endpoint={endpoint}, seed={seed}: "
                    f"{id_path} or {ood_path}"
                )

            df_id = pd.read_parquet(id_path)
            df_ood = pd.read_parquet(ood_path)

            score_id, score_ood, top_features, d = _compute_score_and_top_features(
                df_id,
                df_ood,
                top_k=args.top_k,
            )
            top_features_per_endpoint[endpoint][seed] = set(top_features)

            y = np.concatenate([np.zeros(len(score_id), dtype=int), np.ones(len(score_ood), dtype=int)])
            score = np.concatenate([score_id, score_ood])
            auc, ci_low, ci_high = _auc_with_ci(y, score, n_boot=args.n_boot, seed=rng_seed + seed)
            aupr, aupr_ci_low, aupr_ci_high = _aupr_with_ci(
                y,
                score,
                n_boot=args.n_boot,
                seed=rng_seed + 10000 + seed,
            )
            ci_width = ci_high - ci_low
            aupr_ci_width = aupr_ci_high - aupr_ci_low
            fpr95 = _fpr95(y, score)
            ece = _ece(y, score)
            brier = _brier(y, score)
            power = _power_from_effect_size(d, len(score_id), len(score_ood))

            per_seed_rows.append(
                {
                    "endpoint": endpoint,
                    "seed": seed,
                    "n_id": int(len(score_id)),
                    "n_ood": int(len(score_ood)),
                    "auroc": float(auc),
                    "auroc_ci_low": float(ci_low),
                    "auroc_ci_high": float(ci_high),
                    "auroc_ci_width": float(ci_width),
                    "aupr": float(aupr),
                    "aupr_ci_low": float(aupr_ci_low),
                    "aupr_ci_high": float(aupr_ci_high),
                    "aupr_ci_width": float(aupr_ci_width),
                    "fpr95": float(fpr95),
                    "ece": float(ece),
                    "brier": float(brier),
                    "effect_size_d": float(d),
                    "power": float(power),
                    "top_features": ",".join(top_features),
                }
            )

    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["endpoint", "seed"]).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    per_seed_df.to_csv(args.out_csv, index=False)

    endpoint_summary: dict[str, dict[str, object]] = {}
    all_pass = True
    for endpoint in args.endpoints:
        sub = per_seed_df[per_seed_df["endpoint"] == endpoint].copy()
        auc_values = sub["auroc"].astype(float).tolist()
        aupr_values = sub["aupr"].astype(float).tolist()
        seed_ci_w = _seed_ci_width(auc_values)
        mean_boot_ci_w = float(sub["auroc_ci_width"].mean())
        mean_aupr_boot_ci_w = float(sub["aupr_ci_width"].mean())
        mean_power = float(sub["power"].mean())
        mean_fpr95 = float(sub["fpr95"].mean())
        mean_ece = float(sub["ece"].mean())
        mean_brier = float(sub["brier"].mean())
        mean_j, min_j, n_pairs = _pairwise_jaccard(top_features_per_endpoint[endpoint].values())

        # Gate-3 criteria
        pass_auroc_ci = bool(np.isfinite(seed_ci_w) and seed_ci_w < 0.06)
        pass_jaccard = bool(np.isfinite(mean_j) and mean_j > 0.75)
        pass_power = bool(np.isfinite(mean_power) and mean_power > 0.80)
        endpoint_pass = pass_auroc_ci and pass_jaccard and pass_power
        all_pass = all_pass and endpoint_pass

        endpoint_summary[endpoint] = {
            "auroc_mean": float(np.mean(auc_values)),
            "auroc_std": float(np.std(auc_values, ddof=1)) if len(auc_values) > 1 else float("nan"),
            "auroc_seed_ci_width": float(seed_ci_w),
            "auroc_bootstrap_ci_width_mean": float(mean_boot_ci_w),
            "aupr_mean": float(np.mean(aupr_values)),
            "aupr_std": float(np.std(aupr_values, ddof=1)) if len(aupr_values) > 1 else float("nan"),
            "aupr_bootstrap_ci_width_mean": float(mean_aupr_boot_ci_w),
            "fpr95_mean": float(mean_fpr95),
            "ece_mean": float(mean_ece),
            "brier_mean": float(mean_brier),
            "power_mean": float(mean_power),
            "jaccard_mean": float(mean_j),
            "jaccard_min": float(min_j),
            "jaccard_pairs": int(n_pairs),
            "pass_auroc_ci": pass_auroc_ci,
            "pass_jaccard": pass_jaccard,
            "pass_power": pass_power,
            "pass_endpoint": endpoint_pass,
        }

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts_root": str(args.artifacts_root),
        "experiment": args.experiment,
        "id_dataset": args.id_dataset,
        "ood_dataset": args.ood_dataset,
        "seeds": args.seeds,
        "endpoints": args.endpoints,
        "criteria": {
            "auroc_seed_ci_width_lt": 0.06,
            "jaccard_mean_gt": 0.75,
            "power_mean_gt": 0.80,
        },
        "endpoint_summary": endpoint_summary,
        "gate3_statistical_full_pass": bool(all_pass),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown report
    lines: list[str] = []
    lines.append("# Gate-3 Statistical Analysis (Full Seeds)")
    lines.append("")
    lines.append(f"- Generated UTC: `{summary['generated_utc']}`")
    lines.append(f"- Experiment: `{args.experiment}`")
    lines.append(f"- ID dataset: `{args.id_dataset}`")
    lines.append(f"- OOD dataset: `{args.ood_dataset}`")
    lines.append(f"- Seeds: `{args.seeds}`")
    lines.append(f"- Endpoints: `{args.endpoints}`")
    lines.append(f"- Artifacts root: `{args.artifacts_root}`")
    lines.append("")
    lines.append("## Criteria")
    lines.append("")
    lines.append("- AUROC seed-CI width `< 0.06`")
    lines.append("- Feature-stability Jaccard mean `> 0.75`")
    lines.append("- Post-hoc power mean `> 0.80`")
    lines.append("")
    lines.append("## Endpoint Summary")
    lines.append("")
    lines.append("| Endpoint | AUROC mean | AUPR mean | Seed-CI width | Mean AUROC boot CI width | Mean AUPR boot CI width | FPR95 mean | ECE mean | Brier mean | Jaccard mean | Power mean | AUROC CI | Jaccard | Power | PASS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|")
    for endpoint in args.endpoints:
        e = endpoint_summary[endpoint]
        lines.append(
            "| "
            f"{endpoint} | "
            f"{e['auroc_mean']:.4f} | "
            f"{e['aupr_mean']:.4f} | "
            f"{e['auroc_seed_ci_width']:.4f} | "
            f"{e['auroc_bootstrap_ci_width_mean']:.4f} | "
            f"{e['aupr_bootstrap_ci_width_mean']:.4f} | "
            f"{e['fpr95_mean']:.4f} | "
            f"{e['ece_mean']:.4f} | "
            f"{e['brier_mean']:.4f} | "
            f"{e['jaccard_mean']:.4f} | "
            f"{e['power_mean']:.4f} | "
            f"{'PASS' if e['pass_auroc_ci'] else 'FAIL'} | "
            f"{'PASS' if e['pass_jaccard'] else 'FAIL'} | "
            f"{'PASS' if e['pass_power'] else 'FAIL'} | "
            f"{'PASS' if e['pass_endpoint'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## Final Decision")
    lines.append("")
    lines.append(f"- Gate-3 statistical status: `{'FULL PASS' if all_pass else 'NO-GO / CONDITIONAL'}`")
    lines.append("")
    lines.append("## Per-Seed Metrics")
    lines.append("")
    lines.append(f"- CSV: `{args.out_csv}`")
    lines.append(f"- JSON: `{args.out_json}`")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    status = "FULL PASS" if all_pass else "NO-GO / CONDITIONAL"
    print(f"[gate3] completed: {status}")
    print(f"[gate3] per-seed metrics -> {args.out_csv}")
    print(f"[gate3] summary -> {args.out_json}")
    print(f"[gate3] report -> {args.out_md}")


if __name__ == "__main__":
    main()
