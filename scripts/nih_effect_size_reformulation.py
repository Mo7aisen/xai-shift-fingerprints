#!/usr/bin/env python3
"""Compute robust NIH effect sizes (Cliff's delta, Glass's Δ) with diagnostics.

Outputs CSV suitable for Table 2 replacement and plotting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "hist_entropy": ("histogram_entropy",),
}


def _resolve_metric_column(df: pd.DataFrame, metric: str) -> str | None:
    candidates = (metric, *METRIC_ALIASES.get(metric, ()))
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _clean(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = _clean(x)
    y = _clean(y)
    if x.size == 0 or y.size == 0:
        return float("nan")
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    return (2.0 * u / (x.size * y.size)) - 1.0


def glass_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = _clean(x)
    y = _clean(y)
    if x.size < 2 or y.size < 2:
        return float("nan")
    sd = np.std(x, ddof=1)
    if sd == 0:
        return float("nan")
    return (np.mean(x) - np.mean(y)) / sd


def bootstrap_ci(effect_fn, x, y, n_boot=1000, seed=42, max_sample=None):
    rng = np.random.default_rng(seed)
    x = _clean(x)
    y = _clean(y)
    if x.size == 0 or y.size == 0:
        return (float("nan"), float("nan"))

    if max_sample:
        x = rng.choice(x, size=min(max_sample, x.size), replace=False)
        y = rng.choice(y, size=min(max_sample, y.size), replace=False)

    stats_samples = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=x.size, replace=True)
        yb = rng.choice(y, size=y.size, replace=True)
        stats_samples.append(effect_fn(xb, yb))

    stats_samples = np.asarray(stats_samples, dtype=float)
    return tuple(np.nanpercentile(stats_samples, [2.5, 97.5]))


def variance_diagnostics(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    x = _clean(x)
    y = _clean(y)
    if x.size == 0 or y.size == 0:
        return (float("nan"), float("nan"))

    levene_raw = stats.levene(x, y, center="median").pvalue
    x_log = np.log1p(np.clip(x, a_min=0, a_max=None))
    y_log = np.log1p(np.clip(y, a_min=0, a_max=None))
    levene_log = stats.levene(x_log, y_log, center="median").pvalue
    return (float(levene_raw), float(levene_log))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute NIH effect sizes (Cliff's delta / Glass's Δ).")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples per metric.")
    parser.add_argument("--max-sample", type=int, default=5000, help="Max samples per group for bootstrap.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Use small subsets for quick CPU sanity check.")
    return parser.parse_args()


def load_fingerprint_data(fingerprint_dir: Path, dataset_key: str) -> pd.DataFrame:
    parquet_file = fingerprint_dir / f"{dataset_key}.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Fingerprint data not found: {parquet_file}")
    return pd.read_parquet(parquet_file)


def main() -> None:
    args = parse_args()
    root = args.root
    fingerprints_root = root / "data" / "fingerprints"
    output_name = "nih_effect_sizes_dry_run.csv" if args.dry_run else "nih_effect_sizes.csv"
    output_path = root / "reports" / "enhanced_statistics" / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparisons = [
        ("JSRT vs NIH", "jsrt_baseline", "jsrt", "jsrt_to_nih", "nih_chestxray14"),
        ("Montgomery vs NIH", "montgomery_baseline", "montgomery", "montgomery_to_nih", "nih_chestxray14"),
        ("Shenzhen vs NIH", "shenzhen_baseline", "shenzhen", "shenzhen_to_nih", "nih_chestxray14"),
    ]

    metrics = [
        "dice",
        "coverage_auc",
        "attribution_abs_sum",
        "border_abs_sum",
        "hist_entropy",
    ]

    rows = []
    for comparison_name, ref_exp, ref_key, target_exp, target_key in comparisons:
        ref_df = load_fingerprint_data(fingerprints_root / ref_exp, ref_key)
        tgt_df = load_fingerprint_data(fingerprints_root / target_exp, target_key)

        for metric in metrics:
            ref_col = _resolve_metric_column(ref_df, metric)
            tgt_col = _resolve_metric_column(tgt_df, metric)
            if ref_col is None or tgt_col is None:
                continue

            ref_vals = ref_df[ref_col].dropna().values
            tgt_vals = tgt_df[tgt_col].dropna().values

            if args.dry_run:
                ref_vals = ref_vals[:200]
                tgt_vals = tgt_vals[:2000]

            cliffs = cliffs_delta(ref_vals, tgt_vals)
            glass = glass_delta(ref_vals, tgt_vals)
            cliffs_ci = bootstrap_ci(cliffs_delta, ref_vals, tgt_vals, args.bootstrap, args.seed, args.max_sample)
            glass_ci = bootstrap_ci(glass_delta, ref_vals, tgt_vals, args.bootstrap, args.seed, args.max_sample)
            levene_raw, levene_log = variance_diagnostics(ref_vals, tgt_vals)

            rows.append(
                {
                    "comparison": comparison_name,
                    "metric": metric,
                    "n_ref": len(ref_vals),
                    "n_tgt": len(tgt_vals),
                    "cliffs_delta": cliffs,
                    "cliffs_ci_low": cliffs_ci[0],
                    "cliffs_ci_high": cliffs_ci[1],
                    "glass_delta": glass,
                    "glass_ci_low": glass_ci[0],
                    "glass_ci_high": glass_ci[1],
                    "levene_p_raw": levene_raw,
                    "levene_p_log": levene_log,
                }
            )

    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[INFO] Saved NIH effect sizes to {output_path}")


if __name__ == "__main__":
    main()
