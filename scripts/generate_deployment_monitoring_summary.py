#!/usr/bin/env python3
"""
Generate Deployment Monitoring Summary
======================================

Purpose & Output:
-----------------
Creates Shenzhen-focused monitoring tables/figures that demonstrate
attribution fingerprints triggering alerts before Dice drops.
Outputs CSV, Markdown, LaTeX (booktabs) tables, and paired PNG/PDF figures
for both `reports/` and `manuscript/figures/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import logging
import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("deployment_monitoring")


def _parse_mean(value: Any) -> float:
    """Extract the mean portion from strings like '0.018±0.011'."""
    if isinstance(value, str) and "±" in value:
        value = value.split("±", 1)[0]
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return float("nan")


class DeploymentMonitoringSummary:
    """Produce Shenzhen deployment monitoring artifacts."""

    def __init__(self, root: Path = PROJECT_ROOT) -> None:
        self.root = root
        self.reports_dir = self.root / "reports" / "deployment"
        self.figures_reports_dir = self.reports_dir
        self.figures_manuscript_dir = self.root / "manuscript" / "figures"
        self.tables_manuscript_dir = self.root / "manuscript" / "tables"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_manuscript_dir.mkdir(parents=True, exist_ok=True)
        self.tables_manuscript_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load divergence comparison table and isolate Shenzhen comparisons."""
        table_path = self.root / "divergence_comparison_table.csv"
        LOGGER.info("Loading divergence table from %s", table_path)
        df = pd.read_csv(table_path)
        mask = df["Comparison"].str.contains("Shenzhen", case=False, na=False)
        subset = df[mask].copy()
        if subset.empty:
            raise RuntimeError("No Shenzhen rows found in divergence_comparison_table.csv")
        LOGGER.info("Loaded %d Shenzhen comparisons", len(subset))
        return subset

    @staticmethod
    def _compute_alert(row: pd.Series) -> str:
        """Simple rule to determine alert category."""
        attr_ratio = row["Attr Sum Ratio"]
        kl = row["KL Divergence"]
        emd = row["EMD"]
        dice_drop = row["Dice Drop (%)"]

        reasons = []
        if attr_ratio >= 1.25 or attr_ratio <= 0.75:
            reasons.append("Attr ratio threshold")
        if emd >= 0.10:
            reasons.append("EMD drift")
        if kl >= 0.02:
            reasons.append("KL drift")
        if abs(dice_drop) >= 5:
            reasons.append("Dice change")

        if not reasons:
            return "No Alert"
        return " & ".join(reasons)

    def build_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive monitoring-friendly statistics."""
        LOGGER.info("Building monitoring summary rows")
        summary_rows = []
        for _, row in df.iterrows():
            ref_mean = _parse_mean(row["Ref Dice (mean±std)"])
            tgt_mean = _parse_mean(row["Target Dice (mean±std)"])
            dice_drop = (tgt_mean - ref_mean) / ref_mean * 100 if ref_mean else float("nan")

            summary_rows.append(
                {
                    "Comparison": row["Comparison"],
                    "Reference Dataset": row["Reference Dataset"],
                    "Target Dataset": row["Target Dataset"],
                    "Reference Samples": row["Reference Samples"],
                    "Target Samples": row["Target Samples"],
                    "Dice Drop (%)": round(dice_drop, 2),
                    "Attr Sum Ratio": row["Attr Sum Ratio"],
                    "Border Sum Ratio": row["Border Sum Ratio"],
                    "Entropy Ratio": row["Entropy Ratio"],
                    "KL Divergence": row["KL Divergence"],
                    "EMD": row["EMD"],
                    "Graph Edit Distance": row["Graph Edit Distance"],
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        summary_df["Alert Trigger"] = summary_df.apply(self._compute_alert, axis=1)
        LOGGER.info("Computed summary for %d rows", len(summary_df))
        return summary_df

    def save_tables(self, df: pd.DataFrame) -> Dict[str, Path]:
        """Persist CSV, Markdown, and LaTeX versions."""
        csv_path = self.reports_dir / "deployment_monitoring_summary.csv"
        md_path = self.reports_dir / "deployment_monitoring_summary.md"
        tex_path = self.tables_manuscript_dir / "deployment_monitoring_summary.tex"
        df.to_csv(csv_path, index=False)

        with open(md_path, "w", encoding="utf-8") as handle:
            handle.write("# Deployment Monitoring Summary (Shenzhen)\n\n")
            handle.write("Derived from `divergence_comparison_table.csv`.\n\n")
            handle.write(df.to_markdown(index=False))
            handle.write("\n")

        table_lines = [
            "\\begin{table}[t]",
            "    \\centering",
            "    \\caption{Shenzhen deployment monitoring summary.}",
            "    \\label{tab:deployment-monitoring}",
            "    \\begin{tabular}{lrrrrrr}",
            "        \\toprule",
            "        Comparison & $\\Delta$Dice (\\%) & Attr Ratio & Border Ratio & Entropy Ratio & KL & EMD \\\\",
            "        \\midrule",
        ]
        for _, row in df.iterrows():
            table_lines.append(
                "        {comparison} & {dice:.1f} & {attr:.2f} & {border:.2f} & {entropy:.2f} & {kl:.3f} & {emd:.3f} \\\\".format(
                    comparison=row["Comparison"].replace("→", "\\textrightarrow "),
                    dice=row["Dice Drop (%)"],
                    attr=row["Attr Sum Ratio"],
                    border=row["Border Sum Ratio"],
                    entropy=row["Entropy Ratio"],
                    kl=row["KL Divergence"],
                    emd=row["EMD"],
                )
            )
        table_lines.extend(["        \\bottomrule", "    \\end{tabular}", "\\end{table}"])
        tex_path.write_text("\n".join(table_lines), encoding="utf-8")
        LOGGER.info("Saved monitoring table to %s / %s / %s", csv_path, md_path, tex_path)
        return {"csv": csv_path, "md": md_path, "tex": tex_path}

    def save_figures(self, df: pd.DataFrame) -> Dict[str, Path]:
        """Create monitoring figure showing Dice vs attribution shift."""
        sns.set(style="whitegrid", context="paper")
        plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        sns.barplot(
            data=df,
            x="Comparison",
            y="Dice Drop (%)",
            palette=sns.color_palette("colorblind")[:2],
            ax=axes[0],
        )
        axes[0].axhline(0, color="black", linewidth=1)
        axes[0].set_title("Dice Change vs Shenzhen", fontweight="bold")
        axes[0].set_ylabel("Dice $\\Delta$ (\\%)")
        axes[0].set_xlabel("")
        axes[0].tick_params(axis="x", rotation=20)

        ratios = df.melt(
            id_vars=["Comparison"],
            value_vars=["Attr Sum Ratio", "Border Sum Ratio", "Entropy Ratio"],
            var_name="Metric",
            value_name="Ratio",
        )
        sns.barplot(
            data=ratios,
            x="Comparison",
            y="Ratio",
            hue="Metric",
            palette=sns.color_palette("colorblind")[2:5],
            ax=axes[1],
        )
        axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Attribution Ratios Trigger Early Alerts", fontweight="bold")
        axes[1].set_ylabel("Target / Reference")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=20)
        axes[1].legend(frameon=True, loc="best")

        fig.suptitle("Deployment Hypothesis Evidence – Shenzhen as Incoming Site", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        paths = {
            "report_png": self.figures_reports_dir / "deployment_monitoring.png",
            "report_pdf": self.figures_reports_dir / "deployment_monitoring.pdf",
            "manuscript_png": self.figures_manuscript_dir / "fig_deployment_monitoring.png",
            "manuscript_pdf": self.figures_manuscript_dir / "fig_deployment_monitoring.pdf",
        }
        fig.savefig(paths["report_png"], dpi=300)
        fig.savefig(paths["manuscript_png"], dpi=300)
        fig.savefig(paths["report_pdf"])
        fig.savefig(paths["manuscript_pdf"])
        plt.close(fig)
        LOGGER.info("Saved deployment figures to %s", paths)
        return paths

    def run(self) -> Dict[str, Dict[str, Path]]:
        """Generate all outputs."""
        df = self.load_data()
        summary = self.build_summary(df)
        tables = self.save_tables(summary)
        figures = self.save_figures(summary)
        return {"tables": tables, "figures": figures}


def main() -> None:
    """Entrypoint."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    summary = DeploymentMonitoringSummary()
    outputs = summary.run()
    LOGGER.info(
        "Deployment monitoring summary generated: tables=%s figures=%s",
        outputs["tables"],
        outputs["figures"],
    )


if __name__ == "__main__":
    main()
