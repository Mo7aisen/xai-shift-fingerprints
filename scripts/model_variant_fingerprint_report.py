#!/usr/bin/env python3
"""Compare canonical vs model-variant fingerprint shift detection on one experiment.

This script is intended for backbone/model-configuration generalization checks:
- reads canonical and variant fingerprint parquet files for endpoint(s)
- computes per-endpoint OOD metrics using the same score recipe
- compares top-k discriminative feature overlap
- compares per-sample score agreement (ID/OOD) between canonical and variant
"""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensure repo imports are available

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from xfp.utils.ood_eval import binary_ood_metrics_with_bootstrap


META_COLUMNS = {"dataset_key", "subset", "sample_id", "patient_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare canonical vs variant fingerprint results.")
    parser.add_argument("--canonical-root", type=Path, required=True, help="Canonical fingerprint root dir.")
    parser.add_argument("--variant-root", type=Path, required=True, help="Variant fingerprint root dir.")
    parser.add_argument("--experiment", default="jsrt_to_montgomery")
    parser.add_argument("--id-dataset", default="jsrt")
    parser.add_argument("--ood-dataset", default="montgomery")
    parser.add_argument("--endpoints", nargs="+", default=["predicted_mask", "mask_free"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--score-method", choices=["all_mean_abs_z", "topk_weighted_abs_z"], default="all_mean_abs_z")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    return parser.parse_args()


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in META_COLUMNS or col.startswith("_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _load_pair(root: Path, endpoint: str, experiment: str, id_dataset: str, ood_dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = root / endpoint / experiment
    id_path = base / f"{id_dataset}.parquet"
    ood_path = base / f"{ood_dataset}.parquet"
    if not id_path.exists() or not ood_path.exists():
        raise FileNotFoundError(f"Missing fingerprint pair for endpoint={endpoint}: {id_path} | {ood_path}")
    return pd.read_parquet(id_path), pd.read_parquet(ood_path)


def _effect_and_stats(df_id: pd.DataFrame, df_ood: pd.DataFrame) -> tuple[list[str], pd.Series, pd.Series, pd.Series]:
    common = [c for c in _numeric_feature_columns(df_id) if c in df_ood.columns]
    if not common:
        raise RuntimeError("No shared numeric features found.")
    x_id = df_id[common].copy()
    x_ood = df_ood[common].copy()
    mu = x_id.mean(axis=0)
    sd = x_id.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    effect = ((x_ood.mean(axis=0) - x_id.mean(axis=0)).abs() / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return common, mu, sd, effect


def _score_pair(
    df_id: pd.DataFrame,
    df_ood: pd.DataFrame,
    *,
    top_k: int,
    score_method: str,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.Series]:
    common, mu, sd, effect = _effect_and_stats(df_id, df_ood)
    x_id = df_id[common].copy()
    x_ood = df_ood[common].copy()

    if score_method == "all_mean_abs_z":
        z_id = ((x_id - mu) / sd).abs()
        z_ood = ((x_ood - mu) / sd).abs()
        score_id = z_id.mean(axis=1).to_numpy(dtype=float)
        score_ood = z_ood.mean(axis=1).to_numpy(dtype=float)
    else:
        selected = effect.sort_values(ascending=False).head(top_k)
        if selected.empty:
            raise RuntimeError("No features selected for top-k weighted score.")
        z_id = ((x_id[selected.index] - mu[selected.index]) / sd[selected.index]).abs().to_numpy(dtype=float)
        z_ood = ((x_ood[selected.index] - mu[selected.index]) / sd[selected.index]).abs().to_numpy(dtype=float)
        weights = selected.to_numpy(dtype=float)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        score_id = (z_id * weights[None, :]).sum(axis=1)
        score_ood = (z_ood * weights[None, :]).sum(axis=1)

    top_features = effect.sort_values(ascending=False).head(top_k).index.tolist()
    return score_id, score_ood, top_features, effect


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return float("nan")
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _score_agreement(df_a: pd.DataFrame, score_a: np.ndarray, df_b: pd.DataFrame, score_b: np.ndarray) -> tuple[float, int]:
    a = pd.DataFrame({"sample_id": df_a["sample_id"].astype(str), "score_a": score_a})
    b = pd.DataFrame({"sample_id": df_b["sample_id"].astype(str), "score_b": score_b})
    merged = a.merge(b, on="sample_id", how="inner")
    if merged.empty:
        return float("nan"), 0
    corr = _safe_corr(merged["score_a"].to_numpy(dtype=float), merged["score_b"].to_numpy(dtype=float))
    return corr, int(len(merged))


def _jaccard(a: list[str], b: list[str]) -> float:
    sa = set(a)
    sb = set(b)
    union = len(sa | sb)
    if union == 0:
        return 1.0
    return float(len(sa & sb) / union)


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []

    for idx, endpoint in enumerate(args.endpoints):
        can_id, can_ood = _load_pair(args.canonical_root, endpoint, args.experiment, args.id_dataset, args.ood_dataset)
        var_id, var_ood = _load_pair(args.variant_root, endpoint, args.experiment, args.id_dataset, args.ood_dataset)

        can_score_id, can_score_ood, can_top, can_effect = _score_pair(
            can_id, can_ood, top_k=args.top_k, score_method=args.score_method
        )
        var_score_id, var_score_ood, var_top, var_effect = _score_pair(
            var_id, var_ood, top_k=args.top_k, score_method=args.score_method
        )

        y_can = np.concatenate([np.zeros(len(can_score_id), dtype=int), np.ones(len(can_score_ood), dtype=int)])
        s_can = np.concatenate([can_score_id, can_score_ood])
        y_var = np.concatenate([np.zeros(len(var_score_id), dtype=int), np.ones(len(var_score_ood), dtype=int)])
        s_var = np.concatenate([var_score_id, var_score_ood])

        can_metrics = binary_ood_metrics_with_bootstrap(y_can, s_can, n_boot=args.n_boot, seed=args.seed + idx * 10)
        var_metrics = binary_ood_metrics_with_bootstrap(y_var, s_var, n_boot=args.n_boot, seed=args.seed + idx * 10 + 1)

        common_effect_cols = sorted(set(can_effect.index).intersection(set(var_effect.index)))
        effect_corr = _safe_corr(
            can_effect.loc[common_effect_cols].to_numpy(dtype=float),
            var_effect.loc[common_effect_cols].to_numpy(dtype=float),
        ) if common_effect_cols else float("nan")

        score_corr_id, n_match_id = _score_agreement(can_id, can_score_id, var_id, var_score_id)
        score_corr_ood, n_match_ood = _score_agreement(can_ood, can_score_ood, var_ood, var_score_ood)

        rows.append(
            {
                "endpoint": endpoint,
                "score_method": args.score_method,
                "top_k": int(args.top_k),
                "canonical_auc": can_metrics["auc"],
                "canonical_aupr": can_metrics["aupr"],
                "canonical_fpr95": can_metrics["fpr95"],
                "canonical_tpr_at_fpr05": can_metrics["tpr_at_fpr05"],
                "variant_auc": var_metrics["auc"],
                "variant_aupr": var_metrics["aupr"],
                "variant_fpr95": var_metrics["fpr95"],
                "variant_tpr_at_fpr05": var_metrics["tpr_at_fpr05"],
                "delta_auc_variant_minus_canonical": var_metrics["auc"] - can_metrics["auc"],
                "delta_aupr_variant_minus_canonical": var_metrics["aupr"] - can_metrics["aupr"],
                "delta_fpr95_variant_minus_canonical": var_metrics["fpr95"] - can_metrics["fpr95"],
                "topk_jaccard": _jaccard(can_top, var_top),
                "effect_corr_pearson": effect_corr,
                "score_corr_id": score_corr_id,
                "score_corr_ood": score_corr_ood,
                "n_match_id": n_match_id,
                "n_match_ood": n_match_ood,
                "canonical_top_features": ",".join(can_top),
                "variant_top_features": ",".join(var_top),
            }
        )

    df = pd.DataFrame(rows).sort_values("endpoint").reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.write_text(df.to_csv(index=False), encoding="utf-8")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.experiment,
        "id_dataset": args.id_dataset,
        "ood_dataset": args.ood_dataset,
        "score_method": args.score_method,
        "top_k": args.top_k,
        "canonical_root": str(args.canonical_root),
        "variant_root": str(args.variant_root),
        "endpoints": args.endpoints,
        "rows": rows,
    }
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Model Variant Fingerprint Comparison",
        "",
        f"- Experiment: `{args.experiment}`",
        f"- ID dataset: `{args.id_dataset}`",
        f"- OOD dataset: `{args.ood_dataset}`",
        f"- Score method: `{args.score_method}`",
        f"- Top-k features: `{args.top_k}`",
        f"- Canonical root: `{args.canonical_root}`",
        f"- Variant root: `{args.variant_root}`",
        "",
        "| Endpoint | Canon AUROC | Var AUROC | ΔAUROC | Canon AUPR | Var AUPR | ΔAUPR | Canon FPR95 | Var FPR95 | Top-k Jaccard | Effect Corr | Score Corr ID | Score Corr OOD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['endpoint']} | {r['canonical_auc']:.4f} | {r['variant_auc']:.4f} | {r['delta_auc_variant_minus_canonical']:.4f} | "
            f"{r['canonical_aupr']:.4f} | {r['variant_aupr']:.4f} | {r['delta_aupr_variant_minus_canonical']:.4f} | "
            f"{r['canonical_fpr95']:.4f} | {r['variant_fpr95']:.4f} | {r['topk_jaccard']:.4f} | {r['effect_corr_pearson']:.4f} | "
            f"{r['score_corr_id']:.4f} | {r['score_corr_ood']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `Top-k Jaccard` compares the top-k discriminative fingerprint features between canonical and variant runs.",
            "- `Effect Corr` is Pearson correlation of standardized feature-effect magnitudes across common features.",
            "- `Score Corr ID/OOD` are per-sample Pearson correlations after matching `sample_id`.",
            "",
            f"- CSV: `{args.out_csv}`",
            f"- JSON: `{args.out_json}`",
        ]
    )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[INFO] wrote {args.out_csv}")
    print(f"[INFO] wrote {args.out_json}")
    print(f"[INFO] wrote {args.out_md}")


if __name__ == "__main__":
    main()
