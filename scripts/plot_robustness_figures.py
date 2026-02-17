"""Plot robustness summary figures for manuscript."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_percent_change(csv_path: Path, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
        }
    )

    df = pd.read_csv(csv_path)
    metrics = [
        "dice_cohens_d",
        "iou_cohens_d",
        "attribution_abs_sum_cohens_d",
    ]
    plot_df = df[df["metric"].isin(metrics)].copy()
    if plot_df.empty:
        raise ValueError("No matching metrics found for plotting.")

    summary = (
        plot_df.groupby(["perturbation", "metric"])["percent_change"]
        .median()
        .reset_index()
    )

    metric_labels = {
        "dice_cohens_d": "Dice (Cohen's d)",
        "iou_cohens_d": "IoU (Cohen's d)",
        "attribution_abs_sum_cohens_d": "Attribution Mass (Cohen's d)",
    }
    perturb_order = ["intensity_shift", "gaussian_blur", "salt_pepper"]
    metric_order = ["dice_cohens_d", "iou_cohens_d", "attribution_abs_sum_cohens_d"]

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    width = 0.22
    x_positions = range(len(perturb_order))

    for idx, metric in enumerate(metric_order):
        values = []
        for perturb in perturb_order:
            row = summary[
                (summary["perturbation"] == perturb) & (summary["metric"] == metric)
            ]
            values.append(row["percent_change"].iloc[0] if not row.empty else 0.0)
        offsets = [x + (idx - 1) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=metric_labels[metric])

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(["Intensity", "Blur", "Salt+Pepper"])
    ax.set_ylabel("Median % Change vs Baseline")
    ax.set_title("Robustness of Effect Sizes Under Perturbations")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)


def plot_correlation_stability(csv_path: Path, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
        }
    )

    df = pd.read_csv(csv_path)
    corr_df = df[df["metric"].isin(["pearson_r", "spearman_r"])].copy()
    if corr_df.empty:
        raise ValueError("No correlation metrics found for plotting.")

    corr_df["delta"] = corr_df["perturbed"] - corr_df["baseline"]
    summary = (
        corr_df.groupby(["perturbation", "metric"])["delta"]
        .median()
        .reset_index()
    )

    metric_labels = {
        "pearson_r": "Pearson r",
        "spearman_r": "Spearman r",
    }
    perturb_order = ["intensity_shift", "gaussian_blur", "salt_pepper"]
    metric_order = ["pearson_r", "spearman_r"]

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    width = 0.3
    x_positions = range(len(perturb_order))

    for idx, metric in enumerate(metric_order):
        values = []
        for perturb in perturb_order:
            row = summary[
                (summary["perturbation"] == perturb) & (summary["metric"] == metric)
            ]
            values.append(row["delta"].iloc[0] if not row.empty else 0.0)
        offsets = [x + (idx - 0.5) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=metric_labels[metric])

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(["Intensity", "Blur", "Salt+Pepper"])
    ax.set_ylabel("Median Δ Correlation vs Baseline")
    ax.set_title("Correlation Stability Under Perturbations")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "reports" / "robustness" / "robustness_summary.csv"
    output_path = (
        repo_root / "manuscript" / "figures" / "robustness_percent_change.pdf"
    )
    plot_percent_change(csv_path, output_path)
    print(f"[INFO] Saved figure to: {output_path}")

    corr_output = (
        repo_root / "manuscript" / "figures" / "robustness_correlation_stability.pdf"
    )
    plot_correlation_stability(csv_path, corr_output)
    print(f"[INFO] Saved figure to: {corr_output}")


if __name__ == "__main__":
    main()
