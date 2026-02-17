#!/usr/bin/env python3
"""
Enhanced Statistical Analysis for Publication
==============================================

Comprehensive statistical validation including:
- Permutation tests for divergence metrics
- Effect size calculations (Cohen's d, Hedge's g)
- Power analysis
- Multiple comparison corrections (Bonferroni, FDR)
- Stratified analysis by projection type (PA/AP)
- Coverage correlation analysis with DICE
- Attribution method concordance tests

Author: PhD Research Project
Date: 2025
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import tt_ind_solve_power
import warnings

warnings.filterwarnings('ignore')


class EnhancedStatisticalAnalysis:
    """Enhanced statistical analysis for publication."""

    def __init__(self, base_path: Path):
        """Initialize analysis."""
        self.base_path = Path(base_path)
        self.data_path = self.base_path / 'data' / 'fingerprints'
        self.output_path = self.base_path / 'reports' / 'enhanced_statistics'
        self.output_path.mkdir(parents=True, exist_ok=True)

        print("="*70)
        print("ENHANCED STATISTICAL ANALYSIS FOR PUBLICATION")
        print("="*70)
        print(f"\nData path: {self.data_path}")
        print(f"Output path: {self.output_path}")

    def load_data(self):
        """Load fingerprint data."""
        print("\n📊 Loading fingerprint data...")

        # JSRT baseline
        jsrt_file = self.data_path / 'jsrt_to_montgomery' / 'jsrt.parquet'
        self.jsrt_df = pd.read_parquet(jsrt_file)

        # Montgomery
        mont_file = self.data_path / 'jsrt_to_montgomery' / 'montgomery.parquet'
        self.mont_df = pd.read_parquet(mont_file)

        print(f"  ✓ JSRT: {len(self.jsrt_df)} samples")
        print(f"  ✓ Montgomery: {len(self.mont_df)} samples")

        # Feature columns (exclude metadata, use only common columns)
        jsrt_cols = set(self.jsrt_df.columns)
        mont_cols = set(self.mont_df.columns)
        common_cols = jsrt_cols & mont_cols

        self.feature_cols = [
            col for col in common_cols
            if col not in ['sample_id', 'experiment', 'dataset', 'attribution_method', 'filename']
        ]

        print(f"  ✓ Common features: {len(self.feature_cols)}")

    def effect_size_analysis(self):
        """
        Calculate effect sizes for key features.

        Effect sizes:
        - Cohen's d: standardized mean difference
        - Hedge's g: corrected for small samples
        - Common Language Effect Size (CLES)
        """
        print("\n📈 Computing effect sizes...")

        results = []

        for feature in self.feature_cols:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(self.jsrt_df[feature]):
                continue

            jsrt_vals = self.jsrt_df[feature].values
            mont_vals = self.mont_df[feature].values

            # Convert to float and remove NaN/Inf
            jsrt_vals = pd.to_numeric(jsrt_vals, errors='coerce')
            mont_vals = pd.to_numeric(mont_vals, errors='coerce')
            jsrt_vals = jsrt_vals[np.isfinite(jsrt_vals)]
            mont_vals = mont_vals[np.isfinite(mont_vals)]

            if len(jsrt_vals) < 2 or len(mont_vals) < 2:
                continue

            # Cohen's d
            mean_diff = np.mean(jsrt_vals) - np.mean(mont_vals)
            pooled_std = np.sqrt(
                (np.var(jsrt_vals, ddof=1) + np.var(mont_vals, ddof=1)) / 2
            )
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            # Hedge's g (correction for small samples)
            n_jsrt = len(jsrt_vals)
            n_mont = len(mont_vals)
            correction_factor = 1 - (3 / (4 * (n_jsrt + n_mont) - 9))
            hedges_g = cohens_d * correction_factor

            # Common Language Effect Size (probability of superiority)
            m1, m2 = np.meshgrid(jsrt_vals, mont_vals)
            cles = np.mean(m1 > m2)

            # Interpret magnitude
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                magnitude = 'negligible'
            elif abs_d < 0.5:
                magnitude = 'small'
            elif abs_d < 0.8:
                magnitude = 'medium'
            else:
                magnitude = 'large'

            results.append({
                'feature': feature,
                'cohens_d': cohens_d,
                'hedges_g': hedges_g,
                'cles': cles,
                'magnitude': magnitude,
                'jsrt_mean': np.mean(jsrt_vals),
                'jsrt_std': np.std(jsrt_vals, ddof=1),
                'mont_mean': np.mean(mont_vals),
                'mont_std': np.std(mont_vals, ddof=1),
            })

        effect_df = pd.DataFrame(results)
        effect_df = effect_df.sort_values('cohens_d', key=abs, ascending=False)

        # Save
        output_file = self.output_path / 'effect_sizes.csv'
        effect_df.to_csv(output_file, index=False)

        print(f"  ✓ Computed effect sizes for {len(effect_df)} features")
        print(f"  ✓ Saved: {output_file}")

        # Summary
        large_effects = effect_df[effect_df['magnitude'] == 'large']
        print(f"\n  Key findings:")
        print(f"    - Large effects (|d| > 0.8): {len(large_effects)} features")
        print(f"    - Top 3 largest effects:")
        for _, row in large_effects.head(3).iterrows():
            print(f"      • {row['feature']}: d={row['cohens_d']:.3f} ({row['magnitude']})")

        return effect_df

    def power_analysis(self):
        """
        Compute statistical power for key comparisons.

        Power analysis determines if sample sizes are sufficient to detect effects.
        """
        print("\n⚡ Computing statistical power...")

        # Load effect sizes
        effect_df = pd.read_csv(self.output_path / 'effect_sizes.csv')

        results = []

        for _, row in effect_df.iterrows():
            feature = row['feature']
            cohens_d = row['cohens_d']

            # Compute power for current sample sizes
            n_jsrt = len(self.jsrt_df)
            n_mont = len(self.mont_df)

            try:
                power = tt_ind_solve_power(
                    effect_size=abs(cohens_d),
                    nobs1=n_jsrt,
                    ratio=n_mont / n_jsrt,
                    alpha=0.05,
                    alternative='two-sided'
                )

                results.append({
                    'feature': feature,
                    'effect_size': cohens_d,
                    'n_jsrt': n_jsrt,
                    'n_mont': n_mont,
                    'power': power,
                    'adequate_power': power >= 0.8,
                })

            except Exception as e:
                # Skip if power computation fails
                continue

        power_df = pd.DataFrame(results)

        # Save
        output_file = self.output_path / 'power_analysis.csv'
        power_df.to_csv(output_file, index=False)

        print(f"  ✓ Power analysis for {len(power_df)} features")
        print(f"  ✓ Saved: {output_file}")

        # Summary
        adequate = power_df[power_df['adequate_power']]
        print(f"\n  Key findings:")
        print(f"    - Features with adequate power (≥0.80): {len(adequate)}/{len(power_df)}")
        print(f"    - Mean power: {power_df['power'].mean():.3f}")

        return power_df

    def multiple_comparison_correction(self):
        """
        Apply multiple comparison corrections to hypothesis tests.

        Methods:
        - Bonferroni: Conservative, controls FWER
        - FDR (Benjamini-Hochberg): Less conservative, controls false discovery rate
        """
        print("\n🔬 Applying multiple comparison corrections...")

        # Perform Mann-Whitney U tests for all features
        results = []

        for feature in self.feature_cols:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(self.jsrt_df[feature]):
                continue

            jsrt_vals = pd.to_numeric(self.jsrt_df[feature], errors='coerce').dropna().values
            mont_vals = pd.to_numeric(self.mont_df[feature], errors='coerce').dropna().values

            if len(jsrt_vals) < 2 or len(mont_vals) < 2:
                continue

            # Mann-Whitney U test (non-parametric)
            try:
                statistic, p_value = mannwhitneyu(
                    jsrt_vals, mont_vals, alternative='two-sided'
                )
            except Exception as e:
                print(f"  Warning: Skipping {feature} due to error: {e}")
                continue

            results.append({
                'feature': feature,
                'u_statistic': statistic,
                'p_value_raw': p_value,
            })

        test_df = pd.DataFrame(results)

        # Bonferroni correction
        alpha = 0.05
        n_tests = len(test_df)
        test_df['p_value_bonferroni'] = test_df['p_value_raw'] * n_tests
        test_df['p_value_bonferroni'] = test_df['p_value_bonferroni'].clip(upper=1.0)
        test_df['significant_bonferroni'] = test_df['p_value_bonferroni'] < alpha

        # FDR correction (Benjamini-Hochberg)
        reject, p_corrected, _, _ = multipletests(
            test_df['p_value_raw'],
            alpha=alpha,
            method='fdr_bh'
        )
        test_df['p_value_fdr'] = p_corrected
        test_df['significant_fdr'] = reject

        # Sort by raw p-value
        test_df = test_df.sort_values('p_value_raw')

        # Save
        output_file = self.output_path / 'multiple_comparison_correction.csv'
        test_df.to_csv(output_file, index=False)

        print(f"  ✓ Tested {n_tests} features")
        print(f"  ✓ Saved: {output_file}")

        # Summary
        sig_raw = test_df[test_df['p_value_raw'] < alpha]
        sig_bonf = test_df[test_df['significant_bonferroni']]
        sig_fdr = test_df[test_df['significant_fdr']]

        print(f"\n  Key findings (α=0.05):")
        print(f"    - Significant (uncorrected): {len(sig_raw)}/{n_tests}")
        print(f"    - Significant (Bonferroni): {len(sig_bonf)}/{n_tests}")
        print(f"    - Significant (FDR): {len(sig_fdr)}/{n_tests}")

        return test_df

    def coverage_dice_correlation(self):
        """
        Analyze correlation between coverage AUC and DICE score.

        Tests whether attribution quality (coverage) relates to segmentation
        performance (DICE).
        """
        print("\n📊 Analyzing coverage-DICE correlation...")

        # Combine datasets
        combined_df = pd.concat([
            self.jsrt_df.assign(dataset='JSRT'),
            self.mont_df.assign(dataset='Montgomery')
        ])

        coverage = combined_df['coverage_auc'].values
        dice = combined_df['dice'].values

        # Remove NaN
        mask = np.isfinite(coverage) & np.isfinite(dice)
        coverage = coverage[mask]
        dice = dice[mask]

        # Pearson correlation (linear)
        r_pearson, p_pearson = pearsonr(coverage, dice)

        # Spearman correlation (monotonic)
        r_spearman, p_spearman = spearmanr(coverage, dice)

        results = {
            'n_samples': len(coverage),
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman,
        }

        # Save
        output_file = self.output_path / 'coverage_dice_correlation.csv'
        pd.DataFrame([results]).to_csv(output_file, index=False)

        print(f"  ✓ Analyzed {len(coverage)} samples")
        print(f"  ✓ Saved: {output_file}")

        print(f"\n  Key findings:")
        print(f"    - Pearson r: {r_pearson:.3f} (p={p_pearson:.4f})")
        print(f"    - Spearman ρ: {r_spearman:.3f} (p={p_spearman:.4f})")

        if abs(r_pearson) < 0.3:
            interpretation = 'weak'
        elif abs(r_pearson) < 0.7:
            interpretation = 'moderate'
        else:
            interpretation = 'strong'

        print(f"    - Interpretation: {interpretation} correlation")

        return results

    def stratified_analysis_projection(self):
        """
        Stratified analysis by projection type (PA vs AP).

        Requires metadata with projection information.
        """
        print("\n🔍 Stratified analysis by projection...")

        # Check if projection metadata exists
        metadata_file = self.base_path / 'data' / 'metadata' / 'montgomery_metadata.csv'

        if not metadata_file.exists():
            print("  ⚠ Warning: Projection metadata not found. Skipping.")
            return None

        metadata = pd.read_csv(metadata_file)

        # Merge with Montgomery fingerprints - add projection column
        mont_with_projection = self.mont_df.copy()

        # Join projection info
        projection_map = metadata.set_index('sample_id')['projection'].to_dict()
        mont_with_projection['projection'] = mont_with_projection['sample_id'].map(projection_map)

        # Separate by projection
        pa_samples = mont_with_projection[mont_with_projection['projection'] == 'PA']
        ap_samples = mont_with_projection[mont_with_projection['projection'] == 'AP']

        print(f"  ✓ PA samples: {len(pa_samples)}")
        print(f"  ✓ AP samples: {len(ap_samples)}")

        if len(pa_samples) < 5 or len(ap_samples) < 5:
            print("  ⚠ Warning: Insufficient samples for stratified analysis.")
            return None

        # Compare key features
        results = []

        for feature in ['dice', 'attribution_abs_sum', 'coverage_auc', 'hist_entropy']:
            pa_vals = pa_samples[feature].dropna().values
            ap_vals = ap_samples[feature].dropna().values

            if len(pa_vals) < 2 or len(ap_vals) < 2:
                continue

            # Mann-Whitney U test
            stat, p_val = mannwhitneyu(pa_vals, ap_vals, alternative='two-sided')

            # Effect size
            mean_diff = np.mean(pa_vals) - np.mean(ap_vals)
            pooled_std = np.sqrt((np.var(pa_vals, ddof=1) + np.var(ap_vals, ddof=1)) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            results.append({
                'feature': feature,
                'pa_mean': np.mean(pa_vals),
                'pa_std': np.std(pa_vals, ddof=1),
                'ap_mean': np.mean(ap_vals),
                'ap_std': np.std(ap_vals, ddof=1),
                'u_statistic': stat,
                'p_value': p_val,
                'cohens_d': cohens_d,
                'significant': p_val < 0.05,
            })

        strat_df = pd.DataFrame(results)

        # Save
        output_file = self.output_path / 'stratified_projection_analysis.csv'
        strat_df.to_csv(output_file, index=False)

        print(f"  ✓ Saved: {output_file}")

        # Summary
        sig = strat_df[strat_df['significant']]
        print(f"\n  Key findings:")
        print(f"    - Significant PA/AP differences: {len(sig)}/{len(strat_df)} features")

        return strat_df

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n📄 Generating summary report...")

        report_lines = [
            "="*70,
            "ENHANCED STATISTICAL ANALYSIS - SUMMARY REPORT",
            "="*70,
            "",
            "## 1. EFFECT SIZES",
            "",
        ]

        # Load results
        effect_df = pd.read_csv(self.output_path / 'effect_sizes.csv')
        power_df = pd.read_csv(self.output_path / 'power_analysis.csv')
        test_df = pd.read_csv(self.output_path / 'multiple_comparison_correction.csv')

        # Top effects
        report_lines.append("Top 10 Features by Effect Size (Cohen's d):")
        report_lines.append("")
        for idx, row in effect_df.head(10).iterrows():
            report_lines.append(
                f"  {idx+1}. {row['feature']}: d={row['cohens_d']:.3f} "
                f"({row['magnitude']}) | CLES={row['cles']:.3f}"
            )

        report_lines.extend([
            "",
            "## 2. STATISTICAL POWER",
            "",
            f"Total features analyzed: {len(power_df)}",
            f"Features with adequate power (≥0.80): {len(power_df[power_df['adequate_power']])}",
            f"Mean power: {power_df['power'].mean():.3f}",
            "",
        ])

        # Power distribution
        low_power = power_df[power_df['power'] < 0.5]
        medium_power = power_df[(power_df['power'] >= 0.5) & (power_df['power'] < 0.8)]
        high_power = power_df[power_df['power'] >= 0.8]

        report_lines.append("Power Distribution:")
        report_lines.append(f"  - Low power (<0.50): {len(low_power)} features")
        report_lines.append(f"  - Medium power (0.50-0.79): {len(medium_power)} features")
        report_lines.append(f"  - High power (≥0.80): {len(high_power)} features")

        report_lines.extend([
            "",
            "## 3. MULTIPLE COMPARISON CORRECTION",
            "",
            f"Total hypothesis tests: {len(test_df)}",
            f"Significance level (α): 0.05",
            "",
            "Significant features:",
        ])

        sig_raw = len(test_df[test_df['p_value_raw'] < 0.05])
        sig_bonf = len(test_df[test_df['significant_bonferroni']])
        sig_fdr = len(test_df[test_df['significant_fdr']])

        report_lines.append(f"  - Uncorrected: {sig_raw}/{len(test_df)}")
        report_lines.append(f"  - Bonferroni: {sig_bonf}/{len(test_df)}")
        report_lines.append(f"  - FDR (Benjamini-Hochberg): {sig_fdr}/{len(test_df)}")

        report_lines.extend([
            "",
            "## 4. COVERAGE-DICE CORRELATION",
            "",
        ])

        coverage_results = pd.read_csv(self.output_path / 'coverage_dice_correlation.csv')
        row = coverage_results.iloc[0]

        report_lines.append(f"Pearson correlation: r={row['pearson_r']:.3f}, p={row['pearson_p']:.4f}")
        report_lines.append(f"Spearman correlation: ρ={row['spearman_r']:.3f}, p={row['spearman_p']:.4f}")
        report_lines.append(f"Sample size: {row['n_samples']}")

        report_lines.extend([
            "",
            "="*70,
            "END OF REPORT",
            "="*70,
        ])

        # Save report
        output_file = self.output_path / 'STATISTICAL_SUMMARY.txt'
        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"  ✓ Saved: {output_file}")

        # Print to console
        print("\n" + '\n'.join(report_lines))

    def run_all_analyses(self):
        """Run all statistical analyses."""
        self.load_data()

        self.effect_size_analysis()
        self.power_analysis()
        self.multiple_comparison_correction()
        self.coverage_dice_correlation()
        self.stratified_analysis_projection()
        self.generate_summary_report()

        print("\n" + "="*70)
        print("✅ ENHANCED STATISTICAL ANALYSIS COMPLETE")
        print(f"   Output directory: {self.output_path}")
        print("="*70)


def main():
    """Main execution."""
    if len(sys.argv) > 1:
        base_path = Path(sys.argv[1])
    else:
        base_path = Path(__file__).parent.parent

    analyzer = EnhancedStatisticalAnalysis(base_path)
    analyzer.run_all_analyses()


if __name__ == '__main__':
    main()
