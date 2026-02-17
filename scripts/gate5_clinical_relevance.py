#!/usr/bin/env python3
"""Gate-5 clinical relevance audit.

Computes correlation between per-sample shift score and segmentation quality drop.
Shift scores are derived from endpoint-specific fingerprint features (predicted_mask/mask_free),
while Dice is sourced from an analysis-only upper-bound reference table.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

META_COLUMNS = {
    "dataset_key",
    "subset",
    "sample_id",
    "patient_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Gate-5 clinical relevance report.")
    parser.add_argument("--artifacts-root", type=Path, default=Path("reports_v2/gate3_seed_artifacts"))
    parser.add_argument("--dice-reference-dir", type=Path, required=True)
    parser.add_argument("--experiment", default="jsrt_to_montgomery")
    parser.add_argument("--id-dataset", default="jsrt")
    parser.add_argument("--ood-dataset", default="montgomery")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--endpoints", nargs="+", default=["predicted_mask", "mask_free"])
    parser.add_argument(
        "--score-method",
        choices=["all_mean_abs_z", "topk_mean_abs_z", "topk_weighted_abs_z"],
        default="all_mean_abs_z",
        help="Shift-score recipe used for clinical correlation.",
    )
    parser.add_argument(
        "--score-top-k",
        type=int,
        default=20,
        help="Top-k discriminative features used for topk_* score methods.",
    )
    parser.add_argument("--min-corr", type=float, default=0.60)
    parser.add_argument("--top-k-cases", type=int, default=10)
    parser.add_argument("--out-seed-csv", type=Path, default=Path("reports_v2/audits/GATE5_CLINICAL_PER_SEED.csv"))
    parser.add_argument("--out-cases-csv", type=Path, default=Path("reports_v2/audits/GATE5_CLINICAL_CASES.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("reports_v2/audits/GATE5_CLINICAL_SUMMARY.json"))
    parser.add_argument("--out-md", type=Path, default=Path("reports_v2/audits/GATE5_CLINICAL_RELEVANCE_2026-02-17.md"))
    return parser.parse_args()


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in META_COLUMNS or col.startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _load_endpoint_pair(root: Path, seed: int, endpoint: str, experiment: str, id_dataset: str, ood_dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / f"seed{seed}" / endpoint / experiment
    id_path = base / f"{id_dataset}.parquet"
    ood_path = base / f"{ood_dataset}.parquet"
    if not id_path.exists() or not ood_path.exists():
        raise FileNotFoundError(f"Missing endpoint artifacts: {id_path} | {ood_path}")
    return pd.read_parquet(id_path), pd.read_parquet(ood_path)


def _load_dice_reference(root: Path, id_dataset: str, ood_dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_path = root / f"{id_dataset}.parquet"
    ood_path = root / f"{ood_dataset}.parquet"
    if not id_path.exists() or not ood_path.exists():
        raise FileNotFoundError(f"Missing dice reference tables: {id_path} | {ood_path}")
    id_df = pd.read_parquet(id_path)
    ood_df = pd.read_parquet(ood_path)
    if "sample_id" not in id_df.columns or "dice" not in id_df.columns:
        raise RuntimeError(f"Dice reference must include sample_id,dice columns: {id_path}")
    if "sample_id" not in ood_df.columns or "dice" not in ood_df.columns:
        raise RuntimeError(f"Dice reference must include sample_id,dice columns: {ood_path}")
    return id_df[["sample_id", "dice"]].copy(), ood_df[["sample_id", "dice"]].copy()


def _compute_shift_scores(id_df: pd.DataFrame, ood_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    common = [c for c in _numeric_feature_columns(id_df) if c in ood_df.columns]
    if not common:
        raise RuntimeError("No shared numeric columns found for shift scoring.")

    x_id = id_df[common]
    x_ood = ood_df[common]
    mu = x_id.mean(axis=0)
    sd = x_id.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)

    score_id = ((x_id - mu) / sd).abs().mean(axis=1).to_numpy(dtype=float)
    score_ood = ((x_ood - mu) / sd).abs().mean(axis=1).to_numpy(dtype=float)
    return score_id, score_ood


def _compute_shift_scores_topk(
    id_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    *,
    top_k: int,
    weighted: bool,
) -> tuple[np.ndarray, np.ndarray]:
    common = [c for c in _numeric_feature_columns(id_df) if c in ood_df.columns]
    if not common:
        raise RuntimeError("No shared numeric columns found for shift scoring.")

    x_id = id_df[common]
    x_ood = ood_df[common]
    mu = x_id.mean(axis=0)
    sd = x_id.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    effect = ((x_ood.mean(axis=0) - x_id.mean(axis=0)).abs() / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    selected = effect.sort_values(ascending=False).head(top_k)
    if selected.empty:
        raise RuntimeError("No top-k features selected for shift scoring.")

    z_id = ((x_id[selected.index] - mu[selected.index]) / sd[selected.index]).abs().to_numpy(dtype=float)
    z_ood = ((x_ood[selected.index] - mu[selected.index]) / sd[selected.index]).abs().to_numpy(dtype=float)

    if weighted:
        weights = selected.to_numpy(dtype=float)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        score_id = (z_id * weights[None, :]).sum(axis=1)
        score_ood = (z_ood * weights[None, :]).sum(axis=1)
    else:
        score_id = z_id.mean(axis=1)
        score_ood = z_ood.mean(axis=1)

    return score_id.astype(float), score_ood.astype(float)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    pear_r, pear_p = pearsonr(x, y)
    spear_r, spear_p = spearmanr(x, y)
    return float(pear_r), float(pear_p), float(spear_r), float(spear_p)


def _j(case_df: pd.DataFrame, score_col: str, drop_col: str) -> pd.Series:
    rank_score = case_df[score_col].rank(method="dense", ascending=False)
    rank_drop = case_df[drop_col].rank(method="dense", ascending=False)
    return rank_score + rank_drop


def _build_case_table(
    *,
    id_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    score_id: np.ndarray,
    score_ood: np.ndarray,
    dice_id_ref: pd.DataFrame,
    dice_ood_ref: pd.DataFrame,
    top_k: int,
    endpoint: str,
    seed: int,
) -> pd.DataFrame:
    ref_mean_dice = float(dice_id_ref["dice"].mean())

    ood_merge = ood_df[["sample_id"]].copy()
    ood_merge["shift_score"] = score_ood
    ood_merge = ood_merge.merge(dice_ood_ref, on="sample_id", how="inner")
    ood_merge["dice_drop"] = ref_mean_dice - ood_merge["dice"]
    ood_merge["risk_rank"] = _j(ood_merge, "shift_score", "dice_drop")

    top = ood_merge.sort_values("risk_rank", ascending=True).head(top_k).copy()
    top["endpoint"] = endpoint
    top["seed"] = seed
    top["type"] = "high_shift_high_drop"

    low = ood_merge.sort_values("risk_rank", ascending=False).head(top_k).copy()
    low["endpoint"] = endpoint
    low["seed"] = seed
    low["type"] = "low_shift_low_drop"

    out = pd.concat([top, low], ignore_index=True)
    return out[["endpoint", "seed", "type", "sample_id", "shift_score", "dice", "dice_drop", "risk_rank"]]


def _aggregate(values: Iterable[float]) -> tuple[float, float, float]:
    arr = np.array(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v, v
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return float(arr.mean()), float(lo), float(hi)


def main() -> None:
    args = parse_args()
    args.out_seed_csv.parent.mkdir(parents=True, exist_ok=True)

    dice_id_ref, dice_ood_ref = _load_dice_reference(args.dice_reference_dir, args.id_dataset, args.ood_dataset)
    dice_ref_mean = float(dice_id_ref["dice"].mean())

    per_seed_rows: list[dict[str, float | int | str]] = []
    case_tables: list[pd.DataFrame] = []

    for endpoint in args.endpoints:
        for seed in args.seeds:
            id_df, ood_df = _load_endpoint_pair(
                args.artifacts_root,
                seed,
                endpoint,
                args.experiment,
                args.id_dataset,
                args.ood_dataset,
            )
            if args.score_method == "all_mean_abs_z":
                score_id, score_ood = _compute_shift_scores(id_df, ood_df)
            elif args.score_method == "topk_mean_abs_z":
                score_id, score_ood = _compute_shift_scores_topk(
                    id_df,
                    ood_df,
                    top_k=args.score_top_k,
                    weighted=False,
                )
            else:
                score_id, score_ood = _compute_shift_scores_topk(
                    id_df,
                    ood_df,
                    top_k=args.score_top_k,
                    weighted=True,
                )

            id_join = id_df[["sample_id"]].copy()
            id_join["shift_score"] = score_id
            id_join = id_join.merge(dice_id_ref, on="sample_id", how="inner")
            id_join["dice_drop"] = dice_ref_mean - id_join["dice"]

            ood_join = ood_df[["sample_id"]].copy()
            ood_join["shift_score"] = score_ood
            ood_join = ood_join.merge(dice_ood_ref, on="sample_id", how="inner")
            ood_join["dice_drop"] = dice_ref_mean - ood_join["dice"]

            all_join = pd.concat([id_join, ood_join], ignore_index=True)
            pear_all, p_all, spear_all, sp_all = _safe_corr(
                all_join["shift_score"].to_numpy(dtype=float),
                all_join["dice_drop"].to_numpy(dtype=float),
            )
            pear_ood, p_ood, spear_ood, sp_ood = _safe_corr(
                ood_join["shift_score"].to_numpy(dtype=float),
                ood_join["dice_drop"].to_numpy(dtype=float),
            )

            per_seed_rows.append(
                {
                    "endpoint": endpoint,
                    "seed": seed,
                    "n_all": int(len(all_join)),
                    "n_ood": int(len(ood_join)),
                    "pearson_all": pear_all,
                    "pearson_all_p": p_all,
                    "spearman_all": spear_all,
                    "spearman_all_p": sp_all,
                    "pearson_ood": pear_ood,
                    "pearson_ood_p": p_ood,
                    "spearman_ood": spear_ood,
                    "spearman_ood_p": sp_ood,
                    "id_dice_mean": float(id_join["dice"].mean()),
                    "ood_dice_mean": float(ood_join["dice"].mean()),
                    "ood_dice_drop_mean": float((id_join["dice"].mean() - ood_join["dice"].mean())),
                    "id_shift_mean": float(id_join["shift_score"].mean()),
                    "ood_shift_mean": float(ood_join["shift_score"].mean()),
                }
            )

            if seed == args.seeds[0]:
                case_tables.append(
                    _build_case_table(
                        id_df=id_df,
                        ood_df=ood_df,
                        score_id=score_id,
                        score_ood=score_ood,
                        dice_id_ref=dice_id_ref,
                        dice_ood_ref=dice_ood_ref,
                        top_k=args.top_k_cases,
                        endpoint=endpoint,
                        seed=seed,
                    )
                )

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(args.out_seed_csv, index=False)

    if case_tables:
        cases_df = pd.concat(case_tables, ignore_index=True)
    else:
        cases_df = pd.DataFrame(
            columns=["endpoint", "seed", "type", "sample_id", "shift_score", "dice", "dice_drop", "risk_rank"]
        )
    cases_df.to_csv(args.out_cases_csv, index=False)

    endpoint_summary: dict[str, dict[str, float | bool]] = {}
    gate_pass = True
    for endpoint in args.endpoints:
        sub = per_seed_df[per_seed_df["endpoint"] == endpoint].copy()
        pear_mean, pear_lo, pear_hi = _aggregate(sub["pearson_all"].to_list())
        spear_mean, spear_lo, spear_hi = _aggregate(sub["spearman_all"].to_list())
        pear_ood_mean, _, _ = _aggregate(sub["pearson_ood"].to_list())
        spear_ood_mean, _, _ = _aggregate(sub["spearman_ood"].to_list())

        pass_pearson = bool(np.isfinite(pear_mean) and pear_mean >= args.min_corr)
        pass_spearman = bool(np.isfinite(spear_mean) and spear_mean >= args.min_corr)
        endpoint_pass = pass_pearson and pass_spearman
        gate_pass = gate_pass and endpoint_pass

        endpoint_summary[endpoint] = {
            "pearson_all_mean": pear_mean,
            "pearson_all_ci_low": pear_lo,
            "pearson_all_ci_high": pear_hi,
            "spearman_all_mean": spear_mean,
            "spearman_all_ci_low": spear_lo,
            "spearman_all_ci_high": spear_hi,
            "pearson_ood_mean": pear_ood_mean,
            "spearman_ood_mean": spear_ood_mean,
            "pass_pearson": pass_pearson,
            "pass_spearman": pass_spearman,
            "pass_endpoint": endpoint_pass,
        }

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "id_dataset": args.id_dataset,
        "ood_dataset": args.ood_dataset,
        "seeds": args.seeds,
        "endpoints": endpoint_summary,
        "thresholds": {"min_correlation": args.min_corr},
        "score_method": args.score_method,
        "score_top_k": args.score_top_k,
        "dice_reference_dir": str(args.dice_reference_dir),
        "dice_reference_id_mean": dice_ref_mean,
        "gate5_pass": bool(gate_pass),
        "artifacts": {
            "per_seed_csv": str(args.out_seed_csv),
            "cases_csv": str(args.out_cases_csv),
        },
    }

    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Gate-5 Clinical Relevance Report",
        "",
        f"- Generated UTC: `{summary['generated_utc']}`",
        f"- Experiment: `{args.experiment}`",
        f"- Seed set: `{args.seeds}`",
        f"- Correlation threshold: `{args.min_corr:.2f}`",
        f"- Shift score method: `{args.score_method}`",
        f"- Shift score top-k: `{args.score_top_k}`",
        f"- Dice reference mean ({args.id_dataset}): `{dice_ref_mean:.6f}`",
        "",
        "## Endpoint Summary",
        "",
        "| Endpoint | Pearson(all) mean [95% CI] | Spearman(all) mean [95% CI] | Pearson(ood) mean | Spearman(ood) mean | PASS |",
        "|---|---|---|---:|---:|---|",
    ]
    for endpoint in args.endpoints:
        e = endpoint_summary[endpoint]
        lines.append(
            "| "
            f"{endpoint} | "
            f"{e['pearson_all_mean']:.4f} [{e['pearson_all_ci_low']:.4f}, {e['pearson_all_ci_high']:.4f}] | "
            f"{e['spearman_all_mean']:.4f} [{e['spearman_all_ci_low']:.4f}, {e['spearman_all_ci_high']:.4f}] | "
            f"{e['pearson_ood_mean']:.4f} | "
            f"{e['spearman_ood_mean']:.4f} | "
            f"{'PASS' if e['pass_endpoint'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Per-seed metrics: `{args.out_seed_csv}`",
            f"- Qualitative cases: `{args.out_cases_csv}`",
            f"- JSON summary: `{args.out_json}`",
            "",
            "## Final Decision",
            "",
            f"- Gate-5 status: `{'PASS' if gate_pass else 'NO-GO'}`",
        ]
    )

    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[gate5] completed: {'PASS' if gate_pass else 'NO-GO'}")
    print(f"[gate5] summary -> {args.out_json}")
    print(f"[gate5] report -> {args.out_md}")


if __name__ == "__main__":
    main()
