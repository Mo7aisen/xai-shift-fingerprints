#!/usr/bin/env python3
"""Generate a journal-ready analysis notes report."""
import _path_setup  # noqa: F401 - ensures xfp is importable

from pathlib import Path
import argparse
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate analysis notes for the manuscript.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Primary significance level.")
    parser.add_argument("--robust-alpha", type=float, default=0.01, help="Robustness check alpha.")
    return parser.parse_args()


def load_sample_counts() -> dict:
    fp_root = PROJECT_ROOT / "data" / "fingerprints"
    datasets = {
        "JSRT": fp_root / "jsrt_to_montgomery" / "jsrt.parquet",
        "Montgomery": fp_root / "jsrt_to_montgomery" / "montgomery.parquet",
        "Shenzhen": fp_root / "jsrt_to_shenzhen" / "shenzhen.parquet",
    }
    counts = {}
    for name, path in datasets.items():
        if not path.exists():
            counts[name] = None
            continue
        counts[name] = len(pd.read_parquet(path))
    return counts


def main() -> None:
    args = parse_args()
    alpha = args.alpha
    robust_alpha = args.robust_alpha

    reports_dir = PROJECT_ROOT / "reports"
    baseline_path = reports_dir / "baseline_comparisons" / "baseline_comparison_table.csv"
    hypothesis_path = PROJECT_ROOT / "results" / "metrics" / "divergence" / "hypothesis_tests.csv"
    corr_path = reports_dir / "error_correlation" / "attribution_mass_dice_correlation_by_dataset.csv"
    feature_meta_path = reports_dir / "feature_analysis" / "analysis_metadata.json"

    lines = []
    lines.append("# Analysis Notes for Manuscript\n\n")

    lines.append("## Dataset Scope\n")
    lines.append("- Primary datasets: JSRT, Montgomery, Shenzhen (fingerprint-level analyses)\n")
    lines.append("- External validation: NIH ChestXray14 (used for transfer/validation; not in primary 3-class analyses)\n\n")

    counts = load_sample_counts()
    if all(v is not None for v in counts.values()):
        total = sum(counts.values())
        lines.append("## Dataset Imbalance\n")
        lines.append("| Dataset | Samples | Share |\n|---|---:|---:|\n")
        for name, count in counts.items():
            share = (count / total) * 100 if total else 0.0
            lines.append(f"| {name} | {count} | {share:.1f}% |\n")
        lines.append("\n")

    if feature_meta_path.exists():
        meta = pd.read_json(feature_meta_path, typ="series")
        lines.append("## Comparability Across Drafts\n")
        lines.append(
            f"- Feature analysis uses {int(meta.get('n_groups', 0))} groups: {', '.join(meta.get('datasets', []))}\n"
        )
        lines.append("- Results are not directly comparable to earlier 2-class drafts without re-running in 2-class mode.\n\n")

    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        lines.append("## Robustness Checks\n")
        lines.append(f"- Baseline comparisons (raw p-values) significant at alpha={robust_alpha}: ")
        lines.append(f"{(baseline_df['p-value'] < robust_alpha).sum()}/{len(baseline_df)}\n")
        if "p_value_fdr" in baseline_df.columns:
            lines.append(f"- Baseline comparisons (FDR) significant at alpha={robust_alpha}: ")
            lines.append(f"{(baseline_df['p_value_fdr'] < robust_alpha).sum()}/{len(baseline_df)}\n")
        lines.append("\n")

    if hypothesis_path.exists():
        hyp_df = pd.read_csv(hypothesis_path)
        if "permutation_p_fdr" in hyp_df.columns:
            sig = (hyp_df["permutation_p_fdr"] < robust_alpha).sum()
            lines.append(f"- Hypothesis tests (Permutation FDR) significant at alpha={robust_alpha}: {sig}/{len(hyp_df)}\n\n")

    if corr_path.exists():
        corr_df = pd.read_csv(corr_path)
        neg = corr_df[corr_df["pearson_r"] < 0]
        lines.append("## Negative Correlations and Clinical Interpretation\n")
        if not neg.empty:
            datasets = ", ".join(neg["dataset"].tolist())
            lines.append(f"- Negative attribution mass vs Dice correlations observed in: {datasets}\n")
            lines.append("- Potential interpretation: higher attribution mass may reflect over-confident or diffuse attributions in harder cases.\n")
            lines.append("- Recommendation: consider dataset-specific calibration or model adaptation for those cohorts.\n\n")
        else:
            lines.append("- No negative correlations observed across datasets.\n\n")

    output_path = reports_dir / "analysis_notes.md"
    output_path.write_text("".join(lines))
    print(f"[INFO] Wrote analysis notes to: {output_path}")


if __name__ == "__main__":
    main()
