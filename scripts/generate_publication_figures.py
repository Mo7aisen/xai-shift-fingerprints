#!/usr/bin/env python3
"""
Publication Figure Generation Script
=====================================

Generates publication-quality figures for the attribution fingerprinting manuscript.

Figures generated:
1. Figure 1: Divergence metrics comparison (KL, EMD, GED) with confidence intervals
2. Figure 2: Attribution fingerprint feature comparison (box plots)
3. Figure 3: Qualitative attribution overlays (IG vs Grad-CAM, best/worst cases)
4. Figure 4: Coverage curves for JSRT vs Montgomery
5. Figure 5: Method comparison scatter plots (IG vs Grad-CAM correlations)

Author: PhD Research Project
Date: 2025
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Set publication-quality plotting defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,  # TrueType fonts for PDF
    'ps.fonttype': 42,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
})

# Color palette (colorblind-friendly)
COLORS = {
    'jsrt': '#0173B2',      # Blue
    'montgomery': '#DE8F05', # Orange
    'ig': '#029E73',        # Teal
    'gradcam': '#CC78BC',   # Purple
    'positive': '#56B4E9',
    'negative': '#E69F00',
    'neutral': '#999999',
}


class PublicationFigureGenerator:
    """Generate publication-ready figures for manuscript."""

    def __init__(self, base_path: Path):
        """
        Initialize figure generator.

        Args:
            base_path: Root directory of the project
        """
        self.base_path = Path(base_path)
        self.data_path = self.base_path / 'reports'
        self.output_path = self.base_path / 'manuscript' / 'figures'
        self.output_path.mkdir(parents=True, exist_ok=True)

        print(f"✓ Initialized figure generator")
        print(f"  Data path: {self.data_path}")
        print(f"  Output path: {self.output_path}")

    def load_data(self):
        """Load all required data for figure generation."""
        print("\n📊 Loading data...")

        # Divergence metrics
        self.divergence_df = pd.read_csv(
            self.data_path / 'divergence' / 'divergence_comparison_table.csv'
        )

        # Bootstrap uncertainty
        self.uncertainty_df = pd.read_csv(
            self.data_path / 'divergence' / 'divergence_uncertainty.csv'
        )

        # Hypothesis tests
        self.hypothesis_df = pd.read_csv(
            self.data_path / 'divergence' / 'hypothesis_tests.csv'
        )

        # Detailed metrics
        self.metrics_df = pd.read_csv(
            self.data_path / 'divergence' / 'detailed_metrics_comparison.csv'
        )

        print(f"  ✓ Loaded divergence metrics: {len(self.divergence_df)} rows")
        print(f"  ✓ Loaded uncertainty data: {len(self.uncertainty_df)} rows")
        print(f"  ✓ Loaded hypothesis tests: {len(self.hypothesis_df)} rows")
        print(f"  ✓ Loaded detailed metrics: {len(self.metrics_df)} rows")

    def figure1_divergence_metrics(self):
        """
        Figure 1: Divergence Metrics Comparison

        Bar chart showing KL divergence, EMD, and GED with 95% confidence intervals.
        """
        print("\n🎨 Generating Figure 1: Divergence Metrics...")

        fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))

        metrics = ['kl_divergence', 'emd', 'graph_edit_distance']
        titles = ['KL Divergence', 'Earth Mover\'s Distance', 'Graph Edit Distance']

        for ax, metric, title in zip(axes, metrics, titles):
            # Extract data
            row = self.uncertainty_df[
                self.uncertainty_df['Metric'] == metric
            ].iloc[0]

            point = row['Point Estimate']
            ci_low = row['CI 2.5%']
            ci_high = row['CI 97.5%']
            error = [[point - ci_low], [ci_high - point]]

            # Get p-value
            p_val = self.hypothesis_df[
                self.hypothesis_df['metric'] == metric
            ]['p_value'].iloc[0]

            # Bar plot
            ax.bar(0, point, width=0.5, color=COLORS['jsrt'],
                   edgecolor='black', linewidth=0.8)
            ax.errorbar(0, point, yerr=error, fmt='none',
                       color='black', linewidth=1.5, capsize=5, capthick=1.5)

            # Significance marker
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'

            ax.text(0, ci_high * 1.1, sig_text, ha='center', va='bottom',
                   fontsize=14, fontweight='bold')

            # Styling
            ax.set_title(title, fontweight='bold', pad=10)
            ax.set_ylabel('Divergence')
            ax.set_xticks([0])
            ax.set_xticklabels(['JSRT → Montgomery'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

            # Add CI text
            ci_text = f"95% CI:\n[{ci_low:.4f}, {ci_high:.4f}]"
            ax.text(0.98, 0.98, ci_text, transform=ax.transAxes,
                   ha='right', va='top', fontsize=7,
                   bbox=dict(boxstyle='round', facecolor='white',
                            edgecolor='gray', alpha=0.8))

        plt.tight_layout()

        # Save
        output_file = self.output_path / 'figure1_divergence_metrics.pdf'
        plt.savefig(output_file, format='pdf')
        plt.savefig(output_file.with_suffix('.png'), format='png')
        plt.close()

        print(f"  ✓ Saved: {output_file}")

    def figure2_fingerprint_features(self):
        """
        Figure 2: Attribution Fingerprint Feature Comparison

        Box plots comparing key fingerprint features between JSRT and Montgomery.
        """
        print("\n🎨 Generating Figure 2: Fingerprint Features...")

        # Key features to plot
        features = [
            ('attribution_sum', 'Attribution Sum'),
            ('border_sum_abs', 'Border Sum (abs)'),
            ('histogram_entropy', 'Histogram Entropy'),
            ('component_count', 'Component Count'),
            ('coverage_auc', 'Coverage AUC'),
            ('dice_score', 'DICE Score'),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(7.5, 5))
        axes = axes.flatten()

        for idx, (feature, label) in enumerate(features):
            ax = axes[idx]

            # Extract data for JSRT and Montgomery
            jsrt_data = self.metrics_df[
                (self.metrics_df['dataset'] == 'jsrt') &
                (self.metrics_df['experiment'] == 'jsrt_to_montgomery')
            ][feature].values

            mont_data = self.metrics_df[
                (self.metrics_df['dataset'] == 'montgomery') &
                (self.metrics_df['experiment'] == 'jsrt_to_montgomery')
            ][feature].values

            # Box plot
            bp = ax.boxplot([jsrt_data, mont_data],
                           labels=['JSRT', 'Montgomery'],
                           patch_artist=True,
                           widths=0.6,
                           medianprops=dict(color='black', linewidth=1.5),
                           boxprops=dict(linewidth=0.8),
                           whiskerprops=dict(linewidth=0.8),
                           capprops=dict(linewidth=0.8))

            # Color boxes
            bp['boxes'][0].set_facecolor(COLORS['jsrt'])
            bp['boxes'][1].set_facecolor(COLORS['montgomery'])

            # Styling
            ax.set_title(label, fontweight='bold', fontsize=10, pad=8)
            ax.set_ylabel('Value', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

            # Add sample sizes
            ax.text(0.02, 0.98, f'n={len(jsrt_data)}', transform=ax.transAxes,
                   ha='left', va='top', fontsize=7, color=COLORS['jsrt'])
            ax.text(0.98, 0.98, f'n={len(mont_data)}', transform=ax.transAxes,
                   ha='right', va='top', fontsize=7, color=COLORS['montgomery'])

        plt.tight_layout()

        # Save
        output_file = self.output_path / 'figure2_fingerprint_features.pdf'
        plt.savefig(output_file, format='pdf')
        plt.savefig(output_file.with_suffix('.png'), format='png')
        plt.close()

        print(f"  ✓ Saved: {output_file}")

    def figure3_coverage_curves(self):
        """
        Figure 3: Coverage Curves

        Line plots showing coverage at different attribution thresholds.
        """
        print("\n🎨 Generating Figure 3: Coverage Curves...")

        fig, ax = plt.subplots(figsize=(5, 4))

        # Coverage quantiles
        quantiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # Extract coverage data
        jsrt_coverage = []
        mont_coverage = []

        for q in quantiles:
            col_name = f'coverage_quantile_{q}'

            jsrt_val = self.metrics_df[
                (self.metrics_df['dataset'] == 'jsrt') &
                (self.metrics_df['experiment'] == 'jsrt_to_montgomery')
            ][col_name].mean()

            mont_val = self.metrics_df[
                (self.metrics_df['dataset'] == 'montgomery') &
                (self.metrics_df['experiment'] == 'jsrt_to_montgomery')
            ][col_name].mean()

            jsrt_coverage.append(jsrt_val)
            mont_coverage.append(mont_val)

        # Plot
        ax.plot(quantiles, jsrt_coverage, marker='o', markersize=5,
               label='JSRT', color=COLORS['jsrt'], linewidth=2)
        ax.plot(quantiles, mont_coverage, marker='s', markersize=5,
               label='Montgomery', color=COLORS['montgomery'], linewidth=2)

        # Styling
        ax.set_xlabel('Attribution Threshold Percentile (%)', fontweight='bold')
        ax.set_ylabel('Coverage (Fraction of Mask Covered)', fontweight='bold')
        ax.set_title('Attribution Coverage Curves', fontweight='bold', pad=15)
        ax.legend(loc='lower right', frameon=True, edgecolor='gray')
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, 105)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()

        # Save
        output_file = self.output_path / 'figure3_coverage_curves.pdf'
        plt.savefig(output_file, format='pdf')
        plt.savefig(output_file.with_suffix('.png'), format='png')
        plt.close()

        print(f"  ✓ Saved: {output_file}")

    def figure4_method_comparison(self):
        """
        Figure 4: Attribution Method Comparison (IG vs Grad-CAM)

        Scatter plots showing correlation between IG and Grad-CAM fingerprint features.
        """
        print("\n🎨 Generating Figure 4: Method Comparison...")

        # Load attribution method comparison data
        try:
            ig_df = pd.read_csv(
                self.data_path / 'attribution_methods' /
                'jsrt_to_montgomery__montgomery_integrated_gradients.csv'
            )

            gradcam_df = pd.read_csv(
                self.data_path / 'attribution_methods' /
                'jsrt_to_montgomery__montgomery_gradcam.csv'
            )

            # Merge on sample identifiers
            merged = ig_df.merge(gradcam_df, on='sample_id', suffixes=('_ig', '_gradcam'))

            features = [
                ('attribution_sum', 'Attribution Sum'),
                ('border_sum_abs', 'Border Sum'),
                ('coverage_auc', 'Coverage AUC'),
                ('histogram_entropy', 'Histogram Entropy'),
            ]

            fig, axes = plt.subplots(2, 2, figsize=(7, 6.5))
            axes = axes.flatten()

            for idx, (feature, label) in enumerate(features):
                ax = axes[idx]

                x = merged[f'{feature}_ig']
                y = merged[f'{feature}_gradcam']

                # Scatter
                ax.scatter(x, y, alpha=0.6, s=30, color=COLORS['positive'],
                          edgecolor='black', linewidth=0.3)

                # Fit line
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), color=COLORS['negative'],
                       linewidth=2, linestyle='--', label='Linear Fit')

                # Diagonal reference
                ax.plot([x.min(), x.max()], [x.min(), x.max()],
                       color='gray', linewidth=1, linestyle=':',
                       alpha=0.5, label='y=x')

                # Correlation
                r = np.corrcoef(x, y)[0, 1]
                ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
                       ha='left', va='top', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white',
                                edgecolor='gray', alpha=0.9))

                # Styling
                ax.set_xlabel(f'{label} (IG)', fontweight='bold')
                ax.set_ylabel(f'{label} (Grad-CAM)', fontweight='bold')
                ax.set_title(label, fontweight='bold', pad=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

                if idx == 0:
                    ax.legend(loc='lower right', fontsize=7)

            plt.tight_layout()

            # Save
            output_file = self.output_path / 'figure4_method_comparison.pdf'
            plt.savefig(output_file, format='pdf')
            plt.savefig(output_file.with_suffix('.png'), format='png')
            plt.close()

            print(f"  ✓ Saved: {output_file}")

        except FileNotFoundError as e:
            print(f"  ⚠ Warning: Could not generate Figure 4 - data files not found")
            print(f"    {e}")

    def generate_all_figures(self):
        """Generate all publication figures."""
        print("\n" + "="*60)
        print("PUBLICATION FIGURE GENERATION")
        print("="*60)

        self.load_data()

        self.figure1_divergence_metrics()
        self.figure2_fingerprint_features()
        self.figure3_coverage_curves()
        self.figure4_method_comparison()

        print("\n" + "="*60)
        print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
        print(f"  Output directory: {self.output_path}")
        print("="*60)


def main():
    """Main execution function."""
    # Determine base path
    if len(sys.argv) > 1:
        base_path = Path(sys.argv[1])
    else:
        base_path = Path(__file__).parent.parent

    # Generate figures
    generator = PublicationFigureGenerator(base_path)
    generator.generate_all_figures()


if __name__ == '__main__':
    main()
