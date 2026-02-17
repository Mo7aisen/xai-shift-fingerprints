#!/usr/bin/env python3
"""
Error Correlation Analysis
===========================

This script addresses the expert review's question:
"Does low attribution mass correlate with higher error rates?"

This is critical for clinical interpretation - if attribution shift
predicts errors, it's actionable. Otherwise, "so what?"

Analyses:
1. Correlation between attribution mass and Dice scores
2. Error analysis for low vs high attribution mass samples
3. Boundary error correlation
4. Clinical significance of attribution shifts

Usage:
    python scripts/error_correlation_analysis.py
"""
import _path_setup  # noqa: F401 - ensures xfp is importable

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')



class ErrorCorrelationAnalysis:
    """Analyze correlation between attribution metrics and errors."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.fingerprint_dir = self.root / "data" / "fingerprints"
        self.output_dir = self.root / "reports" / "error_correlation"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

    def log(self, message):
        """Log a message."""
        print(f"[INFO] {message}")

    def load_fingerprints(self):
        """Load fingerprint data."""
        self.log("Loading fingerprint data...")

        jsrt_path = self.fingerprint_dir / "jsrt_to_montgomery" / "jsrt.parquet"
        montgomery_path = self.fingerprint_dir / "jsrt_to_montgomery" / "montgomery.parquet"
        shenzhen_path = self.fingerprint_dir / "jsrt_to_shenzhen" / "shenzhen.parquet"

        self.jsrt_df = pd.read_parquet(jsrt_path)
        self.montgomery_df = pd.read_parquet(montgomery_path)
        self.shenzhen_df = pd.read_parquet(shenzhen_path)

        # Combine for unified analysis
        self.jsrt_df['dataset'] = 'JSRT'
        self.montgomery_df['dataset'] = 'Montgomery'
        self.shenzhen_df['dataset'] = 'Shenzhen'

        self.combined_df = pd.concat(
            [self.jsrt_df, self.montgomery_df, self.shenzhen_df],
            ignore_index=True
        )

        self.log(f"✓ JSRT: {len(self.jsrt_df)} samples")
        self.log(f"✓ Montgomery: {len(self.montgomery_df)} samples")
        self.log(f"✓ Shenzhen: {len(self.shenzhen_df)} samples")
        self.log(f"✓ Combined: {len(self.combined_df)} samples")

    def attribution_mass_dice_correlation(self):
        """
        Analyze correlation between attribution mass and Dice scores.

        Hypothesis: Lower attribution mass → lower confidence → higher errors
        """
        self.log("\n" + "="*80)
        self.log("ATTRIBUTION MASS vs DICE SCORE CORRELATION")
        self.log("="*80)

        # Get attribution mass and Dice scores
        attr_mass = self.combined_df['attribution_abs_sum'].values
        dice_scores = self.combined_df['dice'].values

        # Remove NaN
        mask = np.isfinite(attr_mass) & np.isfinite(dice_scores)
        attr_mass = attr_mass[mask]
        dice_scores = dice_scores[mask]

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(attr_mass, dice_scores)

        # Spearman correlation (non-parametric)
        spearman_r, spearman_p = stats.spearmanr(attr_mass, dice_scores)

        self.log(f"\nCorrelation between attribution mass and Dice score:")
        self.log(f"  Pearson r: {pearson_r:.4f}, p={pearson_p:.6f}")
        self.log(f"  Spearman ρ: {spearman_r:.4f}, p={spearman_p:.6f}")

        # Interpretation
        if abs(pearson_r) < 0.1:
            interp = "negligible correlation"
        elif abs(pearson_r) < 0.3:
            interp = "weak correlation"
        elif abs(pearson_r) < 0.5:
            interp = "moderate correlation"
        else:
            interp = "strong correlation"

        self.log(f"  Interpretation: {interp}")

        # Save combined
        corr_df = pd.DataFrame({
            'metric': ['Pearson', 'Spearman'],
            'correlation': [pearson_r, spearman_r],
            'p_value': [pearson_p, spearman_p],
            'interpretation': [interp, interp],
            'dataset': ['Combined', 'Combined'],
        })

        output_path = self.output_dir / "attribution_mass_dice_correlation.csv"
        corr_df.to_csv(output_path, index=False)

        self.log(f"✓ Saved correlation analysis to: {output_path}")

        # Per-dataset correlations
        per_dataset_rows = []
        for dataset in ['JSRT', 'Montgomery', 'Shenzhen']:
            subset = self.combined_df[self.combined_df['dataset'] == dataset]
            attr_vals = subset['attribution_abs_sum'].values
            dice_vals = subset['dice'].values
            mask = np.isfinite(attr_vals) & np.isfinite(dice_vals)
            if mask.sum() < 3:
                continue
            pearson_r_d, pearson_p_d = stats.pearsonr(attr_vals[mask], dice_vals[mask])
            spearman_r_d, spearman_p_d = stats.spearmanr(attr_vals[mask], dice_vals[mask])
            per_dataset_rows.append({
                'dataset': dataset,
                'pearson_r': pearson_r_d,
                'pearson_p': pearson_p_d,
                'spearman_r': spearman_r_d,
                'spearman_p': spearman_p_d,
            })

        per_dataset_df = pd.DataFrame(per_dataset_rows)
        per_dataset_path = self.output_dir / "attribution_mass_dice_correlation_by_dataset.csv"
        per_dataset_df.to_csv(per_dataset_path, index=False)
        self.log(f"✓ Saved per-dataset correlations to: {per_dataset_path}")

        # Create scatter plot
        self._plot_attribution_dice_scatter()

        return corr_df

    def _plot_attribution_dice_scatter(self):
        """Create scatter plot of attribution mass vs Dice score."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot by dataset
        for dataset in ['JSRT', 'Montgomery', 'Shenzhen']:
            subset = self.combined_df[self.combined_df['dataset'] == dataset]
            ax.scatter(
                subset['attribution_abs_sum'],
                subset['dice'],
                label=dataset,
                alpha=0.6,
                s=50
            )

        ax.set_xlabel('Attribution Mass (IG)', fontsize=12)
        ax.set_ylabel('Dice Score', fontsize=12)
        ax.set_title('Attribution Mass vs Dice Score', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add regression line
        attr_mass = self.combined_df['attribution_abs_sum'].values
        dice_scores = self.combined_df['dice'].values
        mask = np.isfinite(attr_mass) & np.isfinite(dice_scores)

        z = np.polyfit(attr_mass[mask], dice_scores[mask], 1)
        p = np.poly1d(z)
        ax.plot(attr_mass[mask], p(attr_mass[mask]), "r--", alpha=0.8, label=f'Linear fit (r={stats.pearsonr(attr_mass[mask], dice_scores[mask])[0]:.3f})')

        plt.tight_layout()
        output_path = self.output_dir / "attribution_mass_dice_scatter.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.log(f"✓ Saved scatter plot to: {output_path}")

    def low_vs_high_attribution_error_analysis(self):
        """
        Compare error rates for low vs high attribution mass samples.

        Low attribution = bottom 25%
        High attribution = top 25%
        """
        self.log("\n" + "="*80)
        self.log("LOW vs HIGH ATTRIBUTION MASS ERROR ANALYSIS")
        self.log("="*80)

        # Analyze per dataset
        results = []

        for dataset_name in ['JSRT', 'Montgomery', 'Shenzhen']:
            dataset_df = self.combined_df[self.combined_df['dataset'] == dataset_name].copy()

            attr_mass = dataset_df['attribution_abs_sum'].values
            dice_scores = dataset_df['dice'].values

            # Define low/high attribution groups
            q25 = np.percentile(attr_mass, 25)
            q75 = np.percentile(attr_mass, 75)

            low_attr_mask = attr_mass <= q25
            high_attr_mask = attr_mass >= q75

            # Error rate (1 - Dice score)
            low_attr_dice = dice_scores[low_attr_mask]
            high_attr_dice = dice_scores[high_attr_mask]

            low_attr_error = 1 - low_attr_dice
            high_attr_error = 1 - high_attr_dice

            # Statistics
            t_stat, t_pvalue = stats.ttest_ind(low_attr_error, high_attr_error)

            # Cohen's d
            mean_diff = np.mean(low_attr_error) - np.mean(high_attr_error)
            pooled_std = np.sqrt((np.var(low_attr_error, ddof=1) + np.var(high_attr_error, ddof=1)) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            self.log(f"\n{dataset_name} Dataset:")
            self.log(f"  Low attribution (bottom 25%):")
            self.log(f"    n={len(low_attr_dice)}, Dice={np.mean(low_attr_dice):.4f}±{np.std(low_attr_dice):.4f}")
            self.log(f"    Error rate: {np.mean(low_attr_error):.4f}±{np.std(low_attr_error):.4f}")
            self.log(f"  High attribution (top 25%):")
            self.log(f"    n={len(high_attr_dice)}, Dice={np.mean(high_attr_dice):.4f}±{np.std(high_attr_dice):.4f}")
            self.log(f"    Error rate: {np.mean(high_attr_error):.4f}±{np.std(high_attr_error):.4f}")
            self.log(f"  T-test: t={t_stat:.4f}, p={t_pvalue:.6f}")
            self.log(f"  Cohen's d: {cohens_d:.4f}")

            results.append({
                'dataset': dataset_name,
                'low_attr_n': len(low_attr_dice),
                'low_attr_dice_mean': np.mean(low_attr_dice),
                'low_attr_dice_std': np.std(low_attr_dice),
                'low_attr_error_mean': np.mean(low_attr_error),
                'low_attr_error_std': np.std(low_attr_error),
                'high_attr_n': len(high_attr_dice),
                'high_attr_dice_mean': np.mean(high_attr_dice),
                'high_attr_dice_std': np.std(high_attr_dice),
                'high_attr_error_mean': np.mean(high_attr_error),
                'high_attr_error_std': np.std(high_attr_error),
                'ttest_statistic': t_stat,
                'ttest_pvalue': t_pvalue,
                'cohens_d': cohens_d
            })

        results_df = pd.DataFrame(results)

        output_path = self.output_dir / "low_vs_high_attribution_error_analysis.csv"
        results_df.to_csv(output_path, index=False)

        self.log(f"\n✓ Saved error analysis to: {output_path}")

        return results_df

    def border_error_correlation(self):
        """
        Analyze correlation between border attribution and boundary errors.

        Border attribution should correlate with boundary accuracy.
        """
        self.log("\n" + "="*80)
        self.log("BORDER ATTRIBUTION vs BOUNDARY ERROR CORRELATION")
        self.log("="*80)

        # Get border attribution and Dice scores
        if 'border_abs_sum' not in self.combined_df.columns:
            self.log("WARNING: border_abs_sum not found in data")
            return None

        border_attr = self.combined_df['border_abs_sum'].values
        dice_scores = self.combined_df['dice'].values

        # Remove NaN
        mask = np.isfinite(border_attr) & np.isfinite(dice_scores)
        border_attr = border_attr[mask]
        dice_scores = dice_scores[mask]

        # Correlation
        pearson_r, pearson_p = stats.pearsonr(border_attr, dice_scores)
        spearman_r, spearman_p = stats.spearmanr(border_attr, dice_scores)

        self.log(f"\nCorrelation between border attribution and Dice score:")
        self.log(f"  Pearson r: {pearson_r:.4f}, p={pearson_p:.6f}")
        self.log(f"  Spearman ρ: {spearman_r:.4f}, p={spearman_p:.6f}")

        # Save
        corr_df = pd.DataFrame({
            'metric': ['Pearson', 'Spearman'],
            'correlation': [pearson_r, spearman_r],
            'p_value': [pearson_p, spearman_p]
        })

        output_path = self.output_dir / "border_attribution_dice_correlation.csv"
        corr_df.to_csv(output_path, index=False)

        self.log(f"✓ Saved border correlation to: {output_path}")

        return corr_df

    def entropy_error_correlation(self):
        """
        Analyze correlation between attribution entropy and errors.

        High entropy = diffuse, uncertain attribution → higher errors?
        """
        self.log("\n" + "="*80)
        self.log("ATTRIBUTION ENTROPY vs ERROR CORRELATION")
        self.log("="*80)

        if 'hist_entropy' not in self.combined_df.columns:
            self.log("WARNING: hist_entropy not found in data")
            return None

        entropy = self.combined_df['hist_entropy'].values
        dice_scores = self.combined_df['dice'].values

        # Remove NaN
        mask = np.isfinite(entropy) & np.isfinite(dice_scores)
        entropy = entropy[mask]
        dice_scores = dice_scores[mask]

        # Correlation
        pearson_r, pearson_p = stats.pearsonr(entropy, dice_scores)
        spearman_r, spearman_p = stats.spearmanr(entropy, dice_scores)

        self.log(f"\nCorrelation between entropy and Dice score:")
        self.log(f"  Pearson r: {pearson_r:.4f}, p={pearson_p:.6f}")
        self.log(f"  Spearman ρ: {spearman_r:.4f}, p={spearman_p:.6f}")

        # Save
        corr_df = pd.DataFrame({
            'metric': ['Pearson', 'Spearman'],
            'correlation': [pearson_r, spearman_r],
            'p_value': [pearson_p, spearman_p]
        })

        output_path = self.output_dir / "entropy_dice_correlation.csv"
        corr_df.to_csv(output_path, index=False)

        self.log(f"✓ Saved entropy correlation to: {output_path}")

        return corr_df

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        self.log("\n" + "="*80)
        self.log("GENERATING ERROR CORRELATION SUMMARY")
        self.log("="*80)

        summary_lines = []
        summary_lines.append("# Error Correlation Analysis Summary\n")
        summary_lines.append(f"Generated: {pd.Timestamp.now()}\n\n")

        # Load results
        attr_dice_corr = pd.read_csv(self.output_dir / "attribution_mass_dice_correlation.csv")
        per_dataset = pd.read_csv(self.output_dir / "attribution_mass_dice_correlation_by_dataset.csv")
        error_analysis = pd.read_csv(self.output_dir / "low_vs_high_attribution_error_analysis.csv")

        summary_lines.append("## Key Findings\n\n")

        # Attribution mass correlation
        pearson_r = attr_dice_corr[attr_dice_corr['metric'] == 'Pearson']['correlation'].values[0]
        pearson_p = attr_dice_corr[attr_dice_corr['metric'] == 'Pearson']['p_value'].values[0]

        summary_lines.append(f"1. **Attribution Mass vs Dice Score Correlation (Combined)**\n")
        summary_lines.append(f"   - Pearson r = {pearson_r:.4f} (p = {pearson_p:.6f})\n")
        summary_lines.append(f"   - Interpretation: {attr_dice_corr['interpretation'].values[0]}\n\n")

        summary_lines.append("   **Per-dataset correlations:**\n")
        for _, row in per_dataset.iterrows():
            summary_lines.append(
                f"   - {row['dataset']}: Pearson r = {row['pearson_r']:.4f} (p={row['pearson_p']:.6f}), "
                f"Spearman ρ = {row['spearman_r']:.4f} (p={row['spearman_p']:.6f})\n"
            )
        summary_lines.append("\n")

        # Low vs high attribution error analysis
        summary_lines.append(f"2. **Low vs High Attribution Mass Error Rates**\n\n")
        for _, row in error_analysis.iterrows():
            summary_lines.append(f"   **{row['dataset']} Dataset:**\n")
            summary_lines.append(f"   - Low attribution (bottom 25%): Error = {row['low_attr_error_mean']:.4f} ± {row['low_attr_error_std']:.4f}\n")
            summary_lines.append(f"   - High attribution (top 25%): Error = {row['high_attr_error_mean']:.4f} ± {row['high_attr_error_std']:.4f}\n")
            summary_lines.append(f"   - Difference: Cohen's d = {row['cohens_d']:.4f}, p = {row['ttest_pvalue']:.6f}\n\n")

        # Clinical interpretation
        summary_lines.append("## Clinical Interpretation\n\n")
        if pearson_r > 0:
            summary_lines.append("- Higher attribution mass is associated with better segmentation accuracy\n")
            summary_lines.append("- Low attribution mass samples may indicate model uncertainty\n")
            summary_lines.append("- Attribution shift detection can serve as an early warning for potential errors\n\n")
        else:
            summary_lines.append("- Negative correlation suggests higher attribution mass associates with lower Dice\n")
            summary_lines.append("- This may reflect over-confident or diffuse attribution in harder cases (e.g., Shenzhen)\n")
            summary_lines.append("- Consider dataset-specific calibration or model adaptation before deployment\n\n")

        summary_lines.append("## Recommendations\n\n")
        summary_lines.append("1. Monitor attribution mass in deployed models\n")
        summary_lines.append("2. Flag samples with low attribution mass for expert review\n")
        summary_lines.append("3. Use attribution shift as trigger for model retraining\n")
        summary_lines.append("4. Investigate boundary-specific attribution patterns\n")

        # Save summary
        output_path = self.output_dir / "ERROR_CORRELATION_SUMMARY.md"
        with open(output_path, 'w') as f:
            f.writelines(summary_lines)

        self.log(f"✓ Saved summary report to: {output_path}")

    def run_all(self):
        """Run all error correlation analyses."""
        self.log("="*80)
        self.log("ERROR CORRELATION ANALYSIS")
        self.log("="*80)

        self.load_fingerprints()
        self.attribution_mass_dice_correlation()
        self.low_vs_high_attribution_error_analysis()
        self.border_error_correlation()
        self.entropy_error_correlation()
        self.generate_summary_report()

        self.log("\n" + "="*80)
        self.log("ERROR CORRELATION ANALYSIS COMPLETE")
        self.log("="*80)


if __name__ == "__main__":
    analyzer = ErrorCorrelationAnalysis()
    analyzer.run_all()
