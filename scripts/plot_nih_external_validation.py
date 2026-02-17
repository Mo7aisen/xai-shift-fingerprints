"""Plot NIH case-study effect-size summary figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    preferred_csv = repo_root / "reports" / "enhanced_statistics" / "nih_effect_sizes.csv"
    fallback_csv = repo_root / "results" / "metrics" / "divergence" / "hypothesis_tests.csv"
    csv_path = preferred_csv if preferred_csv.exists() else fallback_csv

    output_case_study = repo_root / "manuscript" / "figures" / "nih_case_study.pdf"
    output_legacy = repo_root / "manuscript" / "figures" / "nih_external_validation.pdf"

    df = pd.read_csv(csv_path)
    if "comparison" in df.columns:
        nih_df = df[df["comparison"].str.contains("NIH", na=False)].copy()
    else:
        nih_df = df.copy()

    if nih_df.empty:
        raise ValueError("No NIH comparisons found in input CSV")

    metric_order = [
        "dice",
        "coverage_auc",
        "attribution_abs_sum",
        "border_abs_sum",
        "hist_entropy",
    ]
    metric_labels = ["Dice", "Coverage", "Attr\nMass", "Border\nMass", "Entropy"]

    nih_df = nih_df[nih_df["metric"].isin(metric_order)]
    nih_df["comparison_short"] = nih_df["comparison"].str.replace(" Baseline", "", regex=False)

    preferred = ["JSRT vs NIH", "Montgomery vs NIH", "Shenzhen vs NIH"]
    available = nih_df["comparison_short"].unique().tolist()
    comparisons = [comp for comp in preferred if comp in available]
    if not comparisons:
        raise ValueError("No NIH comparisons available for plotting.")

    value_col = "cliffs_delta" if "cliffs_delta" in nih_df.columns else "permutation_effect_size"
    y_label = "Cliff's delta" if value_col == "cliffs_delta" else "Cohen's d"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    width = 0.24
    x_positions = list(range(len(metric_order)))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    y_min, y_max = 0.0, 0.0
    for idx, comp in enumerate(comparisons):
        values = []
        for metric in metric_order:
            row = nih_df[(nih_df["metric"] == metric) & (nih_df["comparison_short"] == comp)]
            values.append(float(row[value_col].iloc[0]) if not row.empty else 0.0)
        offsets = [x + (idx - (len(comparisons) - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=comp, color=colors[idx], edgecolor="black", linewidth=0.7, alpha=0.9)
        y_min = min(y_min, min(values))
        y_max = max(y_max, max(values))

    y_margin = max(0.08, 0.1 * (y_max - y_min if y_max > y_min else 1.0))
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel(y_label)
    ax.set_title(f"NIH Case Study: Predictive Drift ({y_label})", fontsize=14, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    output_case_study.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_case_study)
    fig.savefig(output_legacy)
    print(f"[INFO] Saved figure to: {output_case_study}")
    print(f"[INFO] Saved compatibility copy to: {output_legacy}")


if __name__ == "__main__":
    main()
