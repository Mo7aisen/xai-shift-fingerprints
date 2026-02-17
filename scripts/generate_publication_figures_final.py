import _path_setup  # noqa: F401 - ensures xfp is importable

#!/usr/bin/env python3
"""
Generate Publication-Quality Figures
=====================================

Creates all figures needed for the manuscript revision with proper
naming and formatting for LaTeX inclusion.

Figures generated:
1. fig_baseline_comparison.pdf - Bar chart comparing shift detection methods
2. fig_pca_variance.pdf - Scree plot showing dimensionality reduction
3. fig_attribution_dice_scatter.pdf - Scatter plot with regression line
4. fig_feature_importance.pdf - Top 15 discriminative features
5. fig_error_by_attribution.pdf - Error rates for low vs high attribution
6. fig_distribution_comparison.pdf - Distribution shifts visualization
7. fig_method_overview.pdf - Conceptual overview diagram

Author: PhD Research Project
Date: October 31, 2025
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')



class PublicationFigureGenerator:
    """Generate all publication figures."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.output_dir = self.root / "manuscript" / "figures"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_context("paper", font_scale=1.3)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.05

    def log(self, message):
        """Log progress."""
        print(f"[Figure] {message}")

    def load_data(self):
        """Load all necessary data."""
        self.log("Loading data for figure generation...")

        # Fingerprints
        jsrt_path = self.root / "data/fingerprints/jsrt_to_montgomery/jsrt.parquet"
        mont_path = self.root / "data/fingerprints/jsrt_to_montgomery/montgomery.parquet"
        shenzhen_path = self.root / "data/fingerprints/jsrt_to_shenzhen/shenzhen.parquet"

        self.jsrt_df = pd.read_parquet(jsrt_path)
        self.montgomery_df = pd.read_parquet(mont_path)
        self.shenzhen_df = pd.read_parquet(shenzhen_path)

        # Baseline comparisons
        baseline_path = self.root / "reports/baseline_comparisons/baseline_comparison_table.csv"
        self.baseline_df = pd.read_csv(baseline_path)

        # Feature importance
        feature_path = self.root / "reports/feature_analysis/consensus_feature_ranking.csv"
        self.feature_df = pd.read_csv(feature_path)

        # PCA
        pca_path = self.root / "reports/feature_analysis/pca_variance_explained.csv"
        self.pca_df = pd.read_csv(pca_path)

        # Error correlation
        error_path = self.root / "reports/error_correlation/low_vs_high_attribution_error_analysis.csv"
        self.error_df = pd.read_csv(error_path)

        self.log("Data loaded successfully")

    def fig_baseline_comparison(self):
        """Figure 1: Baseline shift detection comparison."""
        self.log("Generating Figure 1: Baseline comparison...")

        available_comparisons = self.baseline_df['Comparison'].unique().tolist()
        primary_order = ['JSRT vs Montgomery', 'JSRT vs Shenzhen', 'Montgomery vs Shenzhen']
        comparisons = [c for c in primary_order if c in available_comparisons]
        if not comparisons:
            comparisons = available_comparisons
        n_comparisons = len(comparisons)
        ncols = 3
        nrows = int(np.ceil(n_comparisons / ncols))

        method_order_full = [
            "Input MMD (Histograms)",
            "KS Test (Brightness)",
            "Dice Score Shift",
            "Coverage Shift",
            "IoU Shift",
            "Attribution Fingerprints (KL)",
            "Attribution Fingerprints (EMD)",
            "Attribution Fingerprints (GED)",
        ]
        label_map = {
            "Input MMD (Histograms)": "Input MMD",
            "KS Test (Brightness)": "KS Test",
            "Dice Score Shift": "Dice Shift",
            "Coverage Shift": "Coverage",
            "IoU Shift": "IoU Shift",
            "Attribution Fingerprints (KL)": "Attr FP (KL)",
            "Attribution Fingerprints (EMD)": "Attr FP (EMD)",
            "Attribution Fingerprints (GED)": "Attr FP (GED)",
        }

        # First pass: gather p-values and keep only methods with at least one finite value.
        raw_values = {}
        for comparison in comparisons:
            subset = self.baseline_df[self.baseline_df['Comparison'] == comparison]
            vals = []
            for method in method_order_full:
                rows = subset[subset['Method'] == method]
                if rows.empty:
                    vals.append(np.nan)
                    continue
                p_fdr = rows['p_value_fdr'].values[0]
                p_raw = rows['p-value'].values[0]
                vals.append(float(p_fdr) if np.isfinite(p_fdr) else float(p_raw) if np.isfinite(p_raw) else np.nan)
            raw_values[comparison] = np.array(vals, dtype=float)

        valid_indices = [
            i for i in range(len(method_order_full))
            if any(np.isfinite(raw_values[c][i]) for c in comparisons)
        ]
        if not valid_indices:
            valid_indices = list(range(len(method_order_full)))

        method_order = [method_order_full[i] for i in valid_indices]
        method_labels = [label_map[m] for m in method_order]

        has_attr = any(m.startswith('Attribution Fingerprints') for m in method_order)
        colors = [('#E69F00' if m.startswith('Attribution Fingerprints') else '#56B4E9') for m in method_order]

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(16, 9),
            sharex=True,
            sharey=False,
        )
        axes = np.atleast_1d(axes).ravel()

        epsilon = 1e-300
        transformed = {}
        x_max = 0.0
        for comparison in comparisons:
            p_values = raw_values[comparison][valid_indices]
            p_values = np.where(np.isfinite(p_values), p_values, 1.0)
            p_values = np.clip(p_values, epsilon, 1.0)
            neg_log_p = -np.log10(p_values)
            transformed[comparison] = neg_log_p
            x_max = max(x_max, float(np.nanmax(neg_log_p)))

        y_positions = np.arange(len(method_labels))
        for idx, comparison in enumerate(comparisons):
            ax = axes[idx]
            neg_log_p = transformed[comparison]

            ax.barh(
                y_positions,
                neg_log_p,
                color=colors,
                alpha=0.85,
                edgecolor='black',
                linewidth=1.0,
            )
            ax.axvline(
                x=-np.log10(0.05),
                color='black',
                linestyle='--',
                linewidth=1.2,
                alpha=0.7,
            )

            ax.set_title(comparison, fontsize=12, fontweight='bold', pad=8)
            ax.grid(axis='x', alpha=0.25, linestyle='--')
            ax.set_axisbelow(True)
            ax.invert_yaxis()
            ax.tick_params(axis='x', labelsize=9)
            ax.set_xlim(0, x_max * 1.05 if x_max > 0 else 1)

            row, col = divmod(idx, ncols)
            ax.set_yticks(y_positions)
            if col == 0:
                ax.set_yticklabels(method_labels, fontsize=9)
            else:
                ax.tick_params(axis='y', labelleft=False)
            if row == nrows - 1:
                ax.set_xlabel(r'-log$_{10}$(FDR-corrected p-value)', fontsize=10)

        for idx in range(n_comparisons, len(axes)):
            axes[idx].axis('off')

        if has_attr:
            blue_patch = mpatches.Patch(color='#56B4E9', label='Traditional Methods')
            orange_patch = mpatches.Patch(color='#E69F00', label='Attribution Fingerprints')
            fig.legend(
                handles=[blue_patch, orange_patch],
                loc='lower center',
                bbox_to_anchor=(0.5, 0.0),
                ncol=2,
                frameon=True,
                fontsize=10,
            )
            bottom_margin = 0.10
        else:
            bottom_margin = 0.06

        fig.subplots_adjust(left=0.20, right=0.98, top=0.94, bottom=bottom_margin, wspace=0.10, hspace=0.20)
        output_path = self.output_dir / "fig_baseline_comparison.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()

        self.log(f"Saved: {output_path}")

    def fig_pca_variance(self):
        """Figure 2: PCA variance explained (scree plot)."""
        self.log("Generating Figure 2: PCA variance explained...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Scree plot
        components = self.pca_df['component'].values[:50]
        variance = self.pca_df['variance_explained'].values[:50] * 100
        cumsum_var = self.pca_df['cumulative_variance'].values[:50] * 100

        ax1.plot(components, variance, 'o-', linewidth=2, markersize=4, color='#2c3e50')
        ax1.set_xlabel('Principal Component', fontsize=13)
        ax1.set_ylabel('Variance Explained (%)', fontsize=13)
        ax1.set_title('(a) Scree Plot', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 51])

        # Right: Cumulative variance
        ax2.plot(components, cumsum_var, linewidth=2.5, color='#e74c3c')
        ax2.axhline(y=90, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='90% variance')
        ax2.axhline(y=95, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='95% variance')

        # Add markers
        n_90 = np.argmax(cumsum_var >= 90) + 1
        n_95 = np.argmax(cumsum_var >= 95) + 1

        ax2.plot(n_90, 90, 'go', markersize=10, label=f'{n_90} components')
        ax2.plot(n_95, 95, 'ro', markersize=10, label=f'{n_95} components')

        ax2.set_xlabel('Number of Components', fontsize=13)
        ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=13)
        ax2.set_title('(b) Cumulative Variance', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 51])
        ax2.set_ylim([0, 105])

        plt.tight_layout()
        output_path = self.output_dir / "fig_pca_variance.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()

        self.log(f"Saved: {output_path}")

    def fig_attribution_dice_scatter(self):
        """Figure 3: Attribution mass vs Dice score scatter plot."""
        self.log("Generating Figure 3: Attribution mass vs Dice scatter...")

        fig, ax = plt.subplots(figsize=(8, 6))

        # JSRT data
        jsrt_attr = self.jsrt_df['attribution_abs_sum'].values
        jsrt_dice = self.jsrt_df['dice'].values

        # Montgomery data
        mont_attr = self.montgomery_df['attribution_abs_sum'].values
        mont_dice = self.montgomery_df['dice'].values

        # Shenzhen data
        shenzhen_attr = self.shenzhen_df['attribution_abs_sum'].values
        shenzhen_dice = self.shenzhen_df['dice'].values

        # Scatter plots
        ax.scatter(jsrt_attr, jsrt_dice, alpha=0.6, s=50, color='#3498db',
                   label='JSRT', edgecolors='black', linewidth=0.5)
        ax.scatter(mont_attr, mont_dice, alpha=0.6, s=50, color='#e74c3c',
                   label='Montgomery', edgecolors='black', linewidth=0.5)
        ax.scatter(shenzhen_attr, shenzhen_dice, alpha=0.6, s=50, color='#2ecc71',
                   label='Shenzhen', edgecolors='black', linewidth=0.5)

        # Regression line (combined)
        all_attr = np.concatenate([jsrt_attr, mont_attr, shenzhen_attr])
        all_dice = np.concatenate([jsrt_dice, mont_dice, shenzhen_dice])

        mask = np.isfinite(all_attr) & np.isfinite(all_dice)
        z = np.polyfit(all_attr[mask], all_dice[mask], 1)
        p = np.poly1d(z)

        x_line = np.linspace(all_attr[mask].min(), all_attr[mask].max(), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.7,
                label=f'Linear fit (r={stats.pearsonr(all_attr[mask], all_dice[mask])[0]:.3f})')

        # Formatting
        ax.set_xlabel('Attribution Mass (IG)', fontsize=13)
        ax.set_ylabel('Dice Score', fontsize=13)
        ax.set_title('Attribution Mass vs Segmentation Performance', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "fig_attribution_dice_scatter.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()

        self.log(f"Saved: {output_path}")

    def fig_feature_importance(self):
        """Figure 4: Top 15 discriminative features."""
        self.log("Generating Figure 4: Feature importance...")

        fig, ax = plt.subplots(figsize=(10, 7))

        # Get top 15 features (excluding pixel spacing which has extreme values)
        top_features = self.feature_df[
            ~self.feature_df['feature'].str.contains('pixel_spacing')
        ].head(15)

        features = top_features['feature'].values
        ranks = top_features['mean_rank'].values

        # Clean feature names
        feature_labels = []
        for f in features:
            if 'attribution_abs_sum' in f and 'log' not in f:
                feature_labels.append('IG Mass')
            elif 'attribution_abs_mean' in f and 'log' not in f:
                feature_labels.append('IG Mean')
            elif 'hist_entropy' in f:
                feature_labels.append('Histogram Entropy')
            elif 'border_abs_sum' in f:
                feature_labels.append('Border IG Mass')
            elif 'grad_cam_attribution' in f:
                feature_labels.append('Grad-CAM ' + f.split('_')[-1].title())
            else:
                # Keep original but clean up
                feature_labels.append(f.replace('_', ' ').title())

        # Horizontal bar chart
        y_pos = np.arange(len(features))
        colors = plt.cm.RdYlGn_r(ranks / ranks.max())

        bars = ax.barh(y_pos, 1/ranks * 100, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_labels, fontsize=11)
        ax.set_xlabel('Discriminative Power (Normalized)', fontsize=13)
        ax.set_title('Top 15 Most Discriminative Features', fontsize=14, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)

        # Invert y-axis to have rank 1 on top
        ax.invert_yaxis()

        plt.tight_layout()
        output_path = self.output_dir / "fig_feature_importance.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()

        self.log(f"Saved: {output_path}")

    def fig_error_by_attribution(self):
        """Figure 5: Error rates by attribution mass quartiles."""
        self.log("Generating Figure 5: Error rates by attribution mass...")

        fig, ax = plt.subplots(figsize=(9, 6))

        datasets = self.error_df['dataset'].values
        low_error = self.error_df['low_attr_error_mean'].values * 100
        low_error_std = self.error_df['low_attr_error_std'].values * 100
        high_error = self.error_df['high_attr_error_mean'].values * 100
        high_error_std = self.error_df['high_attr_error_std'].values * 100

        x = np.arange(len(datasets))
        width = 0.35

        ax.bar(x - width/2, low_error, width, yerr=low_error_std,
               label='Low Attribution (Bottom 25%)', color='#e74c3c',
               alpha=0.8, edgecolor='black', linewidth=1.2, capsize=5)
        ax.bar(x + width/2, high_error, width, yerr=high_error_std,
               label='High Attribution (Top 25%)', color='#3498db',
               alpha=0.8, edgecolor='black', linewidth=1.2, capsize=5)

        ax.set_ylabel('Error Rate (%)', fontsize=13)
        ax.set_xlabel('Dataset', fontsize=13)
        ax.set_title('Segmentation Error by Attribution Mass Quartile', fontsize=14, fontweight='bold', pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=12)
        ax.legend(loc='upper left', frameon=True)
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

        y_limit_candidate = 0.0
        for i, dataset in enumerate(datasets):
            p_val = self.error_df[self.error_df['dataset'] == dataset]['ttest_pvalue'].values[0]
            y_max = max(low_error[i] + low_error_std[i], high_error[i] + high_error_std[i])
            if p_val < 0.05:
                x1 = i - width / 2
                x2 = i + width / 2
                bracket_y = y_max + 0.5
                ax.plot([x1, x1, x2, x2], [bracket_y, bracket_y + 0.2, bracket_y + 0.2, bracket_y], color='black', linewidth=1.2)
                if p_val < 0.001:
                    star = '***'
                elif p_val < 0.01:
                    star = '**'
                else:
                    star = '*'
                ax.text(i, bracket_y + 0.25, star, ha='center', va='bottom', fontsize=13, fontweight='bold')
                y_limit_candidate = max(y_limit_candidate, bracket_y + 0.9)
            else:
                y_limit_candidate = max(y_limit_candidate, y_max)

        if y_limit_candidate > 0:
            ax.set_ylim(0, y_limit_candidate + 0.6)

        plt.tight_layout()
        output_path = self.output_dir / "fig_error_by_attribution.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()

        self.log(f"Saved: {output_path}")

    def fig_distribution_comparison(self):
        """Figure 6: Distribution comparison (violin plots)."""
        self.log("Generating Figure 6: Distribution comparison...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Prepare data
        self.jsrt_df['dataset'] = 'JSRT'
        self.montgomery_df['dataset'] = 'Montgomery'
        self.shenzhen_df['dataset'] = 'Shenzhen'
        combined_df = pd.concat([self.jsrt_df, self.montgomery_df, self.shenzhen_df], ignore_index=True)

        metrics = [
            ('attribution_abs_sum', 'IG Attribution Mass'),
            ('dice', 'Dice Score'),
            ('hist_entropy', 'Histogram Entropy'),
            ('border_abs_sum', 'Border IG Mass')
        ]

        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            # Violin plot
            dataset_order = ['JSRT', 'Montgomery', 'Shenzhen']
            data = [combined_df[combined_df['dataset'] == name][metric].values for name in dataset_order]

            parts = ax.violinplot(
                data,
                positions=[0, 1, 2],
                showmeans=True,
                showmedians=True,
            )

            # Color the violins
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.2)

            # Box plot overlay
            ax.boxplot(
                data,
                positions=[0, 1, 2],
                widths=0.1,
                patch_artist=False,
                showfliers=False,
            )

            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(dataset_order, fontsize=11)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f'({chr(97+idx)}) {title}', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            ax.set_axisbelow(True)

        plt.tight_layout()
        output_path = self.output_dir / "fig_distribution_comparison.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()

        self.log(f"Saved: {output_path}")

    def fig_dimensionality_reduction(self):
        """Figure 7: Dimensionality reduction summary."""
        self.log("Generating Figure 7: Dimensionality reduction summary...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Data
        original = len(self.pca_df)
        components_90 = (self.pca_df['cumulative_variance'] >= 0.90).idxmax() + 1
        components_95 = (self.pca_df['cumulative_variance'] >= 0.95).idxmax() + 1
        components_99 = (self.pca_df['cumulative_variance'] >= 0.99).idxmax() + 1

        categories = [
            'Original\nFeatures',
            f'90% Variance\n({components_90} PCs)',
            f'95% Variance\n({components_95} PCs)',
            f'99% Variance\n({components_99} PCs)',
        ]
        values = [original, components_90, components_95, components_99]
        reductions = [
            0,
            (1 - components_90 / original) * 100,
            (1 - components_95 / original) * 100,
            (1 - components_99 / original) * 100,
        ]

        colors = ['#95a5a6', '#27ae60', '#e74c3c', '#f39c12']

        bars = ax.bar(categories, values, color=colors, alpha=0.8,
                      edgecolor='black', linewidth=1.5)

        # Add percentage labels
        for i, (bar, reduction) in enumerate(zip(bars, reductions)):
            height = bar.get_height()
            if i > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'-{reduction:.1f}%',
                       ha='center', va='bottom', fontsize=12, fontweight='bold',
                       color='darkgreen')

        ax.set_ylabel('Number of Features/Components', fontsize=13)
        ax.set_title('Dimensionality Reduction via PCA', fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim([0, max(values) * 1.1])
        ax.grid(axis='y', alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        output_path = self.output_dir / "fig_dimensionality_reduction.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()

        self.log(f"Saved: {output_path}")

    def generate_all_figures(self):
        """Generate all publication figures."""
        self.log("="*80)
        self.log("GENERATING ALL PUBLICATION FIGURES")
        self.log("="*80)

        self.load_data()

        self.fig_baseline_comparison()
        self.fig_pca_variance()
        self.fig_attribution_dice_scatter()
        self.fig_feature_importance()
        self.fig_error_by_attribution()
        self.fig_distribution_comparison()
        self.fig_dimensionality_reduction()

        self.log("="*80)
        self.log("ALL FIGURES GENERATED SUCCESSFULLY")
        self.log(f"Output directory: {self.output_dir}")
        self.log("="*80)


if __name__ == "__main__":
    generator = PublicationFigureGenerator()
    generator.generate_all_figures()
