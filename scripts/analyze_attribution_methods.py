#!/usr/bin/env python
"""Compare attribution methods within fingerprint tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

KEY_METRICS = [
    "attribution_abs_sum",
    "attribution_abs_mean",
    "attribution_abs_sum_log10",
    "attribution_abs_mean_log10",
    "coverage_auc",
    "hist_entropy",
    "component_count",
    "component_largest_size",
]


def _slug_value(value: str) -> str:
    return value.lower().replace(" ", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise and compare attribution methods recorded in fingerprint parquet tables."
    )
    parser.add_argument(
        "--fingerprints-root",
        default="data/fingerprints",
        help="Root directory containing experiment fingerprint outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/attribution_methods",
        help="Directory to store markdown summaries.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=KEY_METRICS,
        help="Metric names (without method prefix) to summarise.",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["projection", "site", "patient_age_bucket"],
        help="Metadata columns to include in stratified summaries.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Persist stratified tables as CSV files alongside markdown outputs.",
    )
    return parser.parse_args()


def _identify_methods(df: pd.DataFrame, metrics: List[str]) -> Dict[str, str]:
    """Return mapping of method name -> column prefix."""

    prefixes: Dict[str, str] = {}
    for metric in metrics:
        suffix = metric
        for col in df.columns:
            if not col.endswith(suffix):
                continue
            prefix = col[: -len(suffix)]
            name = prefix.rstrip("_") or "integrated_gradients"
            prefixes[name] = prefix
    return prefixes


def _metric_column(prefix: str, metric: str) -> str:
    return f"{prefix}{metric}" if prefix else metric


def _method_stats(df: pd.DataFrame, prefix: str, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for metric in metrics:
        column = _metric_column(prefix, metric)
        if column not in df.columns:
            continue
        series = df[column].dropna()
        stats[metric] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "median": float(series.median()),
        }
    return stats


def _correlations(df: pd.DataFrame, prefix_a: str, prefix_b: str, metrics: List[str]) -> Dict[str, float]:
    corrs: Dict[str, float] = {}
    for metric in metrics:
        col_a = _metric_column(prefix_a, metric)
        col_b = _metric_column(prefix_b, metric)
        if col_a in df.columns and col_b in df.columns:
            corrs[metric] = float(df[[col_a, col_b]].corr(method="pearson").iloc[0, 1])
    return corrs


def _dice_correlations(df: pd.DataFrame, prefix: str, metrics: List[str]) -> Dict[str, float]:
    corrs: Dict[str, float] = {}
    if "dice" not in df.columns:
        return corrs
    for metric in metrics:
        column = _metric_column(prefix, metric)
        if column in df.columns:
            corrs[metric] = float(df[[column, "dice"]].corr(method="pearson").iloc[0, 1])
    return corrs


def _top_deltas(
    df: pd.DataFrame,
    base_prefix: str,
    other_prefix: str,
    metric: str,
    *,
    k: int = 5,
) -> pd.DataFrame | None:
    col_base = _metric_column(base_prefix, metric)
    col_other = _metric_column(other_prefix, metric)
    if col_base not in df.columns or col_other not in df.columns or "sample_id" not in df.columns:
        return None
    delta = df[col_other] - df[col_base]
    ranked = delta.abs().sort_values(ascending=False).head(k).index
    extract = df.loc[ranked, ["sample_id", col_base, col_other]].copy()
    extract["delta"] = df.loc[ranked, col_other] - df.loc[ranked, col_base]
    return extract


def _group_summary_table(
    df: pd.DataFrame,
    group_col: str,
    prefix: str,
    metrics: List[str],
) -> pd.DataFrame | None:
    if group_col not in df.columns:
        return None
    working = df.dropna(subset=[group_col])
    if working.empty:
        return None

    rows: List[Dict[str, object]] = []
    for group_value, group_df in working.groupby(group_col):
        row: Dict[str, object] = {group_col: group_value, "count": int(len(group_df))}
        for metric in metrics:
            column = _metric_column(prefix, metric)
            if column not in group_df.columns:
                continue
            series = group_df[column].dropna()
            if series.empty:
                continue
            row[metric] = f"{series.mean():.3f}±{series.std(ddof=0):.3f}"
        if len(row) > 2:
            rows.append(row)

    if not rows:
        return None

    return pd.DataFrame(rows).sort_values(by=group_col).reset_index(drop=True)


def generate_report(table_path: Path, output_dir: Path, metrics: List[str], group_by: List[str], export_csv: bool) -> None:
    df = pd.read_parquet(table_path)
    methods = _identify_methods(df, metrics)
    if not methods:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        f"# Attribution Method Comparison — `{table_path.stem}`",
        "",
        f"Source: `{table_path}`",
        "",
    ]

    base_method = "integrated_gradients" if "integrated_gradients" in methods else next(iter(methods))
    base_prefix = methods[base_method]

    for method, prefix in methods.items():
        stats = _method_stats(df, prefix, metrics)
        lines.append(f"## {method.replace('_', ' ').title()}")
        lines.append("")
        if not stats:
            lines.append("_No metrics available._")
            lines.append("")
            continue
        table = pd.DataFrame(stats).T[["mean", "std", "median"]]
        lines.append(table.to_markdown())
        lines.append("")

        dice_corrs = _dice_correlations(df, prefix, metrics)
        if dice_corrs:
            corr_table = pd.Series(dice_corrs, name="corr_with_dice").to_frame()
            lines.append("Dice Correlations:")
            lines.append("")
            lines.append(corr_table.to_markdown())
            lines.append("")

        if method != base_method:
            corr = _correlations(df, base_prefix, prefix, metrics)
            if corr:
                corr_df = pd.Series(corr, name=f"{method} vs {base_method} (pearson)").to_frame()
                lines.append("Method Concordance:")
                lines.append("")
                lines.append(corr_df.to_markdown())
                lines.append("")

            delta_table = _top_deltas(df, base_prefix, prefix, "attribution_abs_sum", k=5)
            if delta_table is not None:
                lines.append("Largest |Δ attribution_abs_sum| samples:")
                lines.append("")
                lines.append(delta_table.to_markdown(index=False))
                lines.append("")

        for group_col in group_by:
            group_table = _group_summary_table(df, group_col, prefix, metrics)
            if group_table is None:
                continue
            lines.append(f"Stratified by `{group_col}`:")
            lines.append("")
            lines.append(group_table.to_markdown(index=False))
            lines.append("")
            if export_csv:
                csv_filename = (
                    f"{table_path.parent.name}__{table_path.stem}__{_slug_value(method)}__{group_col}.csv"
                )
                group_table.to_csv(output_dir / csv_filename, index=False)

    output_path = output_dir / f"{table_path.parent.name}__{table_path.stem}.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Wrote {output_path}")


def main() -> None:
    args = parse_args()
    root = Path(args.fingerprints_root)
    output_root = Path(args.output_dir)

    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for table_path in sorted(exp_dir.glob("*.parquet")):
            generate_report(table_path, output_root, args.metrics, args.group_by, args.export_csv)


if __name__ == "__main__":
    main()
