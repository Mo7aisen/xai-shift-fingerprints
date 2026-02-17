#!/usr/bin/env python3
"""
Generate final PNG figures with NO OVERLAPPING
Fixed spacing, padding, and layout issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Configure matplotlib for better spacing
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titlepad': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.25,
})

class FigureGenerator:
    def __init__(self):
        self.base_path = Path(__file__).resolve().parents[1]
        self.output_dir = self.base_path / 'manuscript' / 'figures_png'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_figures(self):
        """Generate all 5 figures with proper spacing"""
        print("Generating Figure 1: Baseline Comparison...")
        self.generate_baseline_comparison()

        print("Generating Figure 2: Attribution-Dice Paradox...")
        self.generate_paradox_figure()

        print("Generating Figure 3: PCA Dimensionality...")
        self.generate_pca_figure()

        print("Generating Figure 4: Clinical Significance...")
        self.generate_clinical_figure()

        print("Generating Figure 5: Method Summary...")
        self.generate_method_summary()

        print("\nAll figures generated successfully!")

    def generate_baseline_comparison(self):
        """Figure 1: Baseline comparison using empirical p-values."""
        table_path = self.base_path / 'reports' / 'baseline_comparisons' / 'baseline_comparison_table.csv'
        df = pd.read_csv(table_path)

        order = [
            'Input MMD (Histograms)',
            'KS Test (Brightness)',
            'Dice Score Shift',
            'Coverage Shift',
            'Attribution Fingerprints (KL)',
            'Attribution Fingerprints (EMD)',
            'Attribution Fingerprints (GED)',
        ]
        label_map = {
            'Input MMD (Histograms)': 'Input\nMMD',
            'KS Test (Brightness)': 'Input\nKS',
            'Dice Score Shift': 'Pred\nDice Δ',
            'Coverage Shift': 'Pred\nCoverage',
            'Attribution Fingerprints (KL)': 'Expl\nKL',
            'Attribution Fingerprints (EMD)': 'Expl\nEMD',
            'Attribution Fingerprints (GED)': 'Expl\nGED',
        }
        level_colors = {
            'Input': '#5DADE2',
            'Prediction': '#58D68D',
            'Explanation': '#EC7063',
        }

        df = df.set_index('Method').loc[order].reset_index()
        df['neg_log_p'] = -np.log10(np.clip(df['p-value'].astype(float), 1e-300, 1.0))

        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        bars = ax.bar(
            range(len(df)),
            df['neg_log_p'],
            color=[level_colors[level] for level in df['Level']],
            edgecolor='black',
            linewidth=1.2,
            width=0.65,
        )

        threshold = -np.log10(1e-3)
        ax.axhline(threshold, color='gray', linestyle='--', linewidth=1.5,
                   label='p = 10⁻³ threshold', alpha=0.7, zorder=1)

        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([label_map[m] for m in df['Method']], ha='center')
        ax.set_ylabel('−log₁₀(p-value)', fontweight='bold')
        ax.set_xlabel('Shift Detection Method', fontweight='bold')
        ax.set_ylim(0, df['neg_log_p'].max() * 1.15)

        ax.set_title('All Monitors Flag JSRT→Montgomery Shift;\nExplanation-Level Adds Complementary Insight',
                     fontweight='bold')

        legend_elements = [
            mpatches.Patch(facecolor='#5DADE2', edgecolor='black', label='Input-Level'),
            mpatches.Patch(facecolor='#58D68D', edgecolor='black', label='Prediction-Level'),
            mpatches.Patch(facecolor='#EC7063', edgecolor='black', label='Explanation-Level (Ours)')
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  framealpha=0.95, fontsize=9, bbox_to_anchor=(0.98, 0.98))

        ax.grid(axis='y', alpha=0.3, linestyle='--')
        annotations = [
            'MMD = 0.0456',
            'KS = 0.863',
            "ΔDice = -0.205",
            "ΔCoverage = +0.021",
            'KL = 0.069',
            'EMD = 0.415',
            'GED = 462',
        ]
        for bar, text in zip(bars, annotations):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.4, text,
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    rotation=0)

        plt.tight_layout(pad=1.2)
        output_path = self.output_dir / 'fig1_baseline_comparison.png'
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        print(f"  Saved: {output_path}")

    def generate_paradox_figure(self):
        """Figure 2: Attribution-Dice paradox - FIXED overlapping"""
        jsrt_fp, mont_fp = self._load_fingerprints()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5))

        ax1.scatter(jsrt_fp['attribution_abs_sum'], jsrt_fp['dice'],
                    s=30, alpha=0.7, c='#1B9E77', edgecolors='black',
                    linewidth=0.4, label='JSRT')
        ax1.scatter(mont_fp['attribution_abs_sum'], mont_fp['dice'],
                    s=30, alpha=0.7, c='#D95F02', edgecolors='black',
                    linewidth=0.4, label='Montgomery')

        all_attr = np.concatenate([jsrt_fp['attribution_abs_sum'],
                                   mont_fp['attribution_abs_sum']])
        all_dice = np.concatenate([jsrt_fp['dice'], mont_fp['dice']])
        z = np.polyfit(all_attr, all_dice, 1)
        p = np.poly1d(z)
        x_line = np.linspace(all_attr.min(), all_attr.max(), 100)
        ax1.plot(x_line, p(x_line), 'k--', linewidth=1.5, alpha=0.7, zorder=1)

        corr = np.corrcoef(all_attr, all_dice)[0, 1]
        ax1.text(0.04, 0.96, f'r = {corr:.2f}', transform=ax1.transAxes,
                 fontsize=10.5, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white',
                           edgecolor='black', linewidth=1.0),
                 verticalalignment='top')

        ax1.set_xlabel('Integrated Gradients Attribution Mass', fontweight='bold')
        ax1.set_ylabel('Dice Score', fontweight='bold')
        ax1.set_title('(a) Attribution Collapse Tracks Dice Failures', fontweight='bold')
        ax1.legend(loc='lower right', framealpha=0.95, fontsize=9,
                   edgecolor='black', fancybox=False)
        ax1.grid(alpha=0.3, linestyle='--')

        jsrt_attr_mean = jsrt_fp['attribution_abs_sum'].mean()
        mont_attr_mean = mont_fp['attribution_abs_sum'].mean()
        jsrt_dice_mean = jsrt_fp['dice'].mean()
        mont_dice_mean = mont_fp['dice'].mean()

        attr_norm = [100, (mont_attr_mean / jsrt_attr_mean) * 100]
        dice_norm = [100, (mont_dice_mean / jsrt_dice_mean) * 100]

        datasets = ['JSRT (Baseline)', 'Montgomery (Shift)']
        x_pos = np.arange(len(datasets))
        width = 0.36

        bars1 = ax2.bar(x_pos - width / 2, attr_norm, width,
                        label='Attribution Mass (relative)',
                        color='#756BB1', edgecolor='black', linewidth=1.2)
        bars2 = ax2.bar(x_pos + width / 2, dice_norm, width,
                        label='Dice Score (relative)',
                        color='#31A354', edgecolor='black', linewidth=1.2)

        for bar_group, raw_values in zip([bars1, bars2],
                                         [[jsrt_attr_mean, mont_attr_mean],
                                          [jsrt_dice_mean, mont_dice_mean]]):
            for i, bar in enumerate(bar_group):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 2.5,
                    f'{raw_values[i]:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    fontweight='bold'
                )

        attr_drop = 100 - attr_norm[1]
        dice_drop = 100 - dice_norm[1]

        ax2.set_xlabel('Dataset', fontweight='bold')
        ax2.set_ylabel('Relative Value to JSRT (%)', fontweight='bold')
        ax2.set_title(f'(b) Cross-Dataset Shift: IG ↓{attr_drop:.0f}%, Dice ↓{dice_drop:.0f}%',
                      fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(datasets)
        ax2.legend(loc='upper right', framealpha=0.95, fontsize=9,
                   edgecolor='black', fancybox=False)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(60, 115)

        plt.tight_layout(pad=1.4)
        output_path = self.output_dir / 'fig2_attribution_dice_paradox.png'
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        print(f"  Saved: {output_path}")

    def generate_pca_figure(self):
        """Figure 3: PCA dimensionality - FIXED overlapping annotations"""
        variance_path = self.base_path / 'reports' / 'feature_analysis' / 'pca_variance_explained.csv'
        df = pd.read_csv(variance_path)

        cumulative = df['cumulative_variance'].to_numpy()
        n_components = len(df)

        n_90 = int(np.argmax(cumulative >= 0.90) + 1)
        n_95 = int(np.argmax(cumulative >= 0.95) + 1)
        n_99 = int(np.argmax(cumulative >= 0.99) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5))

        ax1.plot(range(1, n_components + 1), cumulative * 100, 'r-', linewidth=2.5)
        ax1.axhline(y=95, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)
        ax1.axhline(y=90, color='green', linestyle='--', linewidth=1.5, alpha=0.8)

        ax1.plot(n_95, cumulative[n_95 - 1] * 100, 'o', color='green',
                 markersize=8, markeredgecolor='black', markeredgewidth=1.2, zorder=5)
        ax1.plot(n_90, cumulative[n_90 - 1] * 100, 'o', color='red',
                 markersize=8, markeredgecolor='black', markeredgewidth=1.2, zorder=5)

        ax1.annotate(f'{n_95} components\n(78% reduction)',
                     xy=(n_95, cumulative[n_95 - 1] * 100),
                     xytext=(n_95 + 10, 82),
                     fontsize=9.5, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='mintcream',
                               edgecolor='black', linewidth=1.0),
                     arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))
        ax1.annotate(f'{n_90} components\n(86% reduction)',
                     xy=(n_90, cumulative[n_90 - 1] * 100),
                     xytext=(n_90 + 8, 73),
                     fontsize=9.5, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='mistyrose',
                               edgecolor='black', linewidth=1.0),
                     arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))

        ax1.set_xlabel('Principal Components Retained', fontweight='bold')
        ax1.set_ylabel('Cumulative Variance Explained (%)', fontweight='bold')
        ax1.set_title('(a) Compact Fingerprint Representation', fontweight='bold')
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_xlim(0, 60)
        ax1.set_ylim(0, 102)

        categories = [
            'Original\nFeatures',
            f'90% Var\n({n_90} PCs)',
            f'95% Var\n({n_95} PCs)',
            f'99% Var\n({n_99} PCs)'
        ]
        n_features = [121, n_90, n_95, n_99]
        reductions = [0,
                      100 * (1 - n_90 / 121),
                      100 * (1 - n_95 / 121),
                      100 * (1 - n_99 / 121)]
        colors = ['#95A5A6', '#27AE60', '#E74C3C', '#F39C12']

        bars = ax2.bar(range(len(categories)), n_features, color=colors,
                       edgecolor='black', linewidth=1.5, width=0.55)

        for bar, red in zip(bars[1:], reductions[1:]):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + 4,
                f'−{red:.0f}%',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold',
                color='darkgreen'
            )

        sample_dim_ratio = 385 / n_95
        ax2.text(0.98, 0.98,
                 f'Sample:Component = 385:{n_95} ≈ {sample_dim_ratio:.1f}:1',
                 transform=ax2.transAxes,
                 fontsize=9.5,
                 fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='aliceblue',
                           edgecolor='black', linewidth=1.0),
                 ha='right', va='top')

        ax2.set_ylabel('Number of Features / Components', fontweight='bold')
        ax2.set_title('(b) Dimensionality Reduction Addresses Overfitting', fontweight='bold')
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(categories)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 140)

        plt.tight_layout(pad=1.4)
        output_path = self.output_dir / 'fig3_pca_dimensionality.png'
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        print(f"  Saved: {output_path}")

    def generate_clinical_figure(self):
        """Figure 4: Clinical significance - FIXED overlapping"""
        jsrt_fp, mont_fp = self._load_fingerprints()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5))

        datasets = ['JSRT', 'Montgomery']
        x_pos = np.arange(len(datasets))
        width = 0.35

        jsrt_q1, jsrt_q3 = jsrt_fp['attribution_abs_sum'].quantile([0.25, 0.75])
        mont_q1, mont_q3 = mont_fp['attribution_abs_sum'].quantile([0.25, 0.75])

        jsrt_low = jsrt_fp[jsrt_fp['attribution_abs_sum'] <= jsrt_q1]
        jsrt_high = jsrt_fp[jsrt_fp['attribution_abs_sum'] >= jsrt_q3]
        mont_low = mont_fp[mont_fp['attribution_abs_sum'] <= mont_q1]
        mont_high = mont_fp[mont_fp['attribution_abs_sum'] >= mont_q3]

        def error_stats(df):
            errors = (1 - df['dice']) * 100
            return errors.mean(), errors.std()

        low_means = []
        high_means = []
        low_stds = []
        high_stds = []
        for low_df, high_df in [(jsrt_low, jsrt_high), (mont_low, mont_high)]:
            mean_low, std_low = error_stats(low_df)
            mean_high, std_high = error_stats(high_df)
            low_means.append(mean_low)
            high_means.append(mean_high)
            low_stds.append(std_low)
            high_stds.append(std_high)

        bars1 = ax1.bar(
            x_pos - width / 2, low_means, width,
            yerr=low_stds, capsize=4,
            label='Low Attribution (Q1)',
            color='#E6550D', edgecolor='black', linewidth=1.2,
            error_kw={'linewidth': 1.2, 'elinewidth': 1.2}
        )

        bars2 = ax1.bar(
            x_pos + width / 2, high_means, width,
            yerr=high_stds, capsize=4,
            label='High Attribution (Q4)',
            color='#3182BD', edgecolor='black', linewidth=1.2,
            error_kw={'linewidth': 1.2, 'elinewidth': 1.2}
        )

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 1.2,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=8.5,
                    fontweight='bold'
                )

        ax1.set_ylabel('Dice Error (%)', fontweight='bold')
        ax1.set_title('(a) Low Attribution Cases Double Cross-Domain Errors', fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(datasets)
        ax1.legend(loc='upper left', framealpha=0.95, fontsize=9,
                   edgecolor='black', fancybox=False)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, max(high_means) + 10)

        jsrt_attr = jsrt_fp['attribution_abs_sum']
        mont_attr = mont_fp['attribution_abs_sum']

        parts = ax2.violinplot(
            [jsrt_attr, mont_attr],
            positions=[0, 1],
            widths=0.6,
            showmeans=True,
            showmedians=True
        )

        colors = ['#3182BD', '#E6550D']
        for idx, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[idx])
            pc.set_edgecolor('black')
            pc.set_linewidth(1.2)
            pc.set_alpha(0.75)

        for stat in ['cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians']:
            if stat in parts:
                parts[stat].set_edgecolor('black')
                parts[stat].set_linewidth(1.0)

        ax2.text(
            0, jsrt_attr.mean() + 0.8,
            f'{jsrt_attr.mean():.2f}',
            ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='black', linewidth=0.8)
        )
        ax2.text(
            1, mont_attr.mean() + 0.8,
            f'{mont_attr.mean():.2f}',
            ha='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='black', linewidth=0.8)
        )

        ax2.set_ylabel('Integrated Gradients Attribution Mass', fontweight='bold')
        ax2.set_xlabel('Dataset', fontweight='bold')
        ax2.set_title('(b) Attribution Mass Contracts Under Distribution Shift', fontweight='bold')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['JSRT', 'Montgomery'])
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(0, max(jsrt_attr.max(), mont_attr.max()) + 2)

        plt.tight_layout(pad=1.4)
        output_path = self.output_dir / 'fig4_clinical_significance.png'
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        print(f"  Saved: {output_path}")

    def generate_method_summary(self):
        """Figure 5: Method summary diagram - FIXED text positioning"""
        fig, ax = plt.subplots(figsize=(12.5, 6.5))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        fig.suptitle('Attribution Fingerprinting for Dataset Shift Detection',
                     fontsize=16, fontweight='bold', y=0.98)
        fig.text(0.5, 0.93, 'Monitoring Model Explanation Stability Beyond Accuracy Metrics',
                 fontsize=11.5, ha='center', style='italic')

        box_height = 2.5
        box_y = 4.5

        rect1 = plt.Rectangle((0.5, box_y), 2.5, box_height,
                              facecolor='#AED6F1', edgecolor='black', linewidth=3)
        ax.add_patch(rect1)
        ax.text(1.75, box_y + box_height - 0.4, '1. Extract Features',
                ha='center', fontsize=12.5, fontweight='bold')
        ax.text(1.75, box_y + 1.7, '121 features from',
                ha='center', fontsize=10)
        ax.text(1.75, box_y + 1.3, 'IG + Grad-CAM',
                ha='center', fontsize=10.5, fontweight='bold')
        ax.text(1.75, box_y + 0.9, 'attribution maps',
                ha='center', fontsize=10)
        ax.text(1.75, box_y + 0.3, '95% variance with 26 PCs (78% fewer dims)',
                ha='center', fontsize=9.5, style='italic')

        ax.arrow(3.1, box_y + box_height/2, 0.6, 0, head_width=0.25,
                 head_length=0.2, fc='black', ec='black', linewidth=2)

        rect2 = plt.Rectangle((3.9, box_y), 2.5, box_height,
                              facecolor='#ABEBC6', edgecolor='black', linewidth=3)
        ax.add_patch(rect2)
        ax.text(5.15, box_y + box_height - 0.4, '2. Detect Shift',
                ha='center', fontsize=12.5, fontweight='bold')
        ax.text(5.15, box_y + 1.7, 'KL = 0.069 (p < 10⁻³)',
                ha='center', fontsize=10.5, fontweight='bold')
        ax.text(5.15, box_y + 1.3, 'EMD = 0.415',
                ha='center', fontsize=10.5, fontweight='bold')
        ax.text(5.15, box_y + 0.9, 'GED = 462',
                ha='center', fontsize=10.5, fontweight='bold')
        ax.text(5.15, box_y + 0.35, 'Input/Prediction monitors also flag shift',
                ha='center', fontsize=9.5, style='italic')

        ax.arrow(6.5, box_y + box_height/2, 0.6, 0, head_width=0.25,
                 head_length=0.2, fc='black', ec='black', linewidth=2)

        rect3 = plt.Rectangle((7.3, box_y), 2.5, box_height,
                              facecolor='#F9E79F', edgecolor='black', linewidth=3)
        ax.add_patch(rect3)
        ax.text(8.55, box_y + box_height - 0.4, '3. Clinical Action',
                ha='center', fontsize=12.5, fontweight='bold')
        ax.text(8.55, box_y + 1.7, 'IG mass ↓36% (24.1 → 15.5)',
                ha='center', fontsize=10.5, fontweight='bold')
        ax.text(8.55, box_y + 1.3, 'Dice ↓0.205 (0.923 → 0.718)',
                ha='center', fontsize=10.5, fontweight='bold')
        ax.text(8.55, box_y + 0.7, 'Low attribution cases double Dice error',
                ha='center', fontsize=9.5, style='italic')
        ax.text(8.55, box_y + 0.3, 'Use for triage alerts & drift monitoring',
                ha='center', fontsize=9.5, style='italic')

        finding_y = 2.2
        finding_rect = plt.Rectangle((1.5, finding_y), 7, 1.3,
                                     facecolor='#FADBD8', edgecolor='red',
                                     linewidth=3, linestyle='-')
        ax.add_patch(finding_rect)
        ax.text(5, finding_y + 0.75,
                'Key Finding: Explanation collapse highlights the highest-risk Montgomery cases before accuracy alarms fire.',
                ha='center', fontsize=10.5, fontweight='bold')

        bottom_y = 0.6
        bottom_rect = plt.Rectangle((0.5, bottom_y), 9, 1.2,
                                    facecolor='#E8F8F5', edgecolor='blue',
                                    linewidth=2, linestyle='--')
        ax.add_patch(bottom_rect)
        ax.text(5, bottom_y + 0.8,
                'Baseline: All monitors flag shift; only explanation-level reveals reasoning change.',
                ha='center', fontsize=10)
        ax.text(5, bottom_y + 0.45,
                'Dimensionality: 26 PCs (95% variance) maintain a 14.8:1 sample-to-dimension ratio.',
                ha='center', fontsize=10)
        ax.text(5, bottom_y + 0.1,
                'Deployment Action: Flag low-attribution cases for review and log attribution drift over time.',
                ha='center', fontsize=10)

        plt.tight_layout(pad=1.0)
        output_path = self.output_dir / 'fig5_method_summary.png'
        plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.4)
        plt.close()
        print(f"  Saved: {output_path}")

    def _load_fingerprints(self):
        """Load JSRT and Montgomery fingerprint data."""
        jsrt_fp = pd.read_parquet(
            self.base_path / 'data' / 'fingerprints' / 'jsrt_to_montgomery' / 'jsrt.parquet'
        )
        mont_fp = pd.read_parquet(
            self.base_path / 'data' / 'fingerprints' / 'jsrt_to_montgomery' / 'montgomery.parquet'
        )
        return jsrt_fp, mont_fp

if __name__ == '__main__':
    generator = FigureGenerator()
    generator.generate_all_figures()
