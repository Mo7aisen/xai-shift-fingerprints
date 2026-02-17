"""Statistical hypothesis tests for divergence metrics.

Implements permutation tests and non-parametric comparisons to validate
attribution shift findings for publication.
"""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "hist_entropy": ("histogram_entropy",),
}


def _resolve_metric_column(df: pd.DataFrame, metric: str) -> str | None:
    """Return the first matching column name for a metric, considering aliases."""

    candidates = (metric, *METRIC_ALIASES.get(metric, ()))
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    statistic_fn: callable = lambda x, y: np.abs(np.mean(x) - np.mean(y)),
    n_permutations: int = 10000,
) -> dict:
    """
    Perform two-sample permutation test.

    Args:
        group1: First sample
        group2: Second sample
        statistic_fn: Function to compute test statistic
        n_permutations: Number of random permutations

    Returns:
        Dictionary with observed statistic, p-value, and null distribution
    """

    observed_stat = statistic_fn(group1, group2)
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    null_distribution = []
    for _ in tqdm(range(n_permutations), desc="Permutations"):
        np.random.shuffle(combined)
        perm_group1 = combined[:n1]
        perm_group2 = combined[n1:]
        null_stat = statistic_fn(perm_group1, perm_group2)
        null_distribution.append(null_stat)

    null_distribution = np.array(null_distribution)
    p_value = np.mean(null_distribution >= observed_stat)

    return {
        "observed_statistic": observed_stat,
        "p_value": p_value,
        "null_mean": np.mean(null_distribution),
        "null_std": np.std(null_distribution),
        "effect_size": (observed_stat - np.mean(null_distribution)) / np.std(null_distribution),
    }


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy()
    adjusted = np.full_like(p, np.nan, dtype=float)
    valid_mask = np.isfinite(p)
    p_valid = p[valid_mask]
    if p_valid.size == 0:
        return pd.Series(adjusted, index=p_values.index)

    order = np.argsort(p_valid)
    ranked = p_valid[order]
    n = len(ranked)
    adjusted_vals = ranked * n / (np.arange(1, n + 1))
    adjusted_vals = np.minimum.accumulate(adjusted_vals[::-1])[::-1]
    adjusted_vals = np.clip(adjusted_vals, 0.0, 1.0)
    adjusted[valid_mask] = adjusted_vals[np.argsort(order)]
    return pd.Series(adjusted, index=p_values.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Statistical hypothesis tests for divergence metrics.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level for FDR.")
    return parser.parse_args()


def load_fingerprint_data(fingerprint_dir: Path, dataset_key: str) -> pd.DataFrame:
    """Load fingerprint data for a specific dataset."""

    parquet_file = fingerprint_dir / f"{dataset_key}.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Fingerprint data not found: {parquet_file}")

    return pd.read_parquet(parquet_file)


def main():
    """Run statistical hypothesis tests on divergence metrics."""
    args = parse_args()
    alpha = args.alpha

    print("="*70)
    print("Statistical Hypothesis Testing for Attribution Divergence")
    print("="*70)

    root = Path(__file__).resolve().parents[1]
    fingerprints_root = root / "data" / "fingerprints"

    # Load fingerprint data
    experiments = {
        "jsrt_baseline": ["jsrt"],
        "jsrt_to_montgomery": ["montgomery"],
        "jsrt_to_shenzhen": ["shenzhen"],
        "jsrt_to_nih": ["nih_chestxray14"],
        "montgomery_baseline": ["montgomery"],
        "montgomery_to_jsrt": ["jsrt"],
        "montgomery_to_shenzhen": ["shenzhen"],
        "montgomery_to_nih": ["nih_chestxray14"],
        "shenzhen_baseline": ["shenzhen"],
        "shenzhen_to_nih": ["nih_chestxray14"],
        "nih_baseline": ["nih_chestxray14"],
    }

    print("\n[1/3] Loading fingerprint data...")
    data = {}
    for exp_name, datasets in experiments.items():
        exp_dir = fingerprints_root / exp_name
        for dataset_key in datasets:
            try:
                df = load_fingerprint_data(exp_dir, dataset_key)
                data[dataset_key] = df
                print(f"  ✓ {dataset_key}: {len(df)} samples")
            except FileNotFoundError as e:
                print(f"  ✗ {dataset_key}: {e}")

    if len(data) < 2:
        print("\n❌ Insufficient data for hypothesis testing. Need at least 2 datasets.")
        return

    # Extract key metrics for testing
    print("\n[2/3] Extracting metrics...")
    metrics_of_interest = [
        "dice",
        "attribution_abs_sum",
        "border_abs_sum",
        "hist_entropy",  # Fixed: fingerprints use hist_entropy not histogram_entropy
        "coverage_auc",
    ]

    results = []

    # Define all cross-dataset comparisons to test
    comparisons = [
        ("JSRT Baseline vs Montgomery", "jsrt_baseline", "jsrt", "jsrt_to_montgomery", "montgomery"),
        ("JSRT Baseline vs Shenzhen", "jsrt_baseline", "jsrt", "jsrt_to_shenzhen", "shenzhen"),
        ("JSRT Baseline vs NIH", "jsrt_baseline", "jsrt", "jsrt_to_nih", "nih_chestxray14"),
        ("Montgomery Baseline vs JSRT", "montgomery_baseline", "montgomery", "montgomery_to_jsrt", "jsrt"),
        ("Montgomery Baseline vs Shenzhen", "montgomery_baseline", "montgomery", "montgomery_to_shenzhen", "shenzhen"),
        ("Montgomery Baseline vs NIH", "montgomery_baseline", "montgomery", "montgomery_to_nih", "nih_chestxray14"),
        ("Shenzhen Baseline vs NIH", "shenzhen_baseline", "shenzhen", "shenzhen_to_nih", "nih_chestxray14"),
    ]

    print("\n[3/3] Running Hypothesis Tests...")

    for comparison_name, ref_exp, ref_key, target_exp, target_key in comparisons:
        ref_dir = fingerprints_root / ref_exp
        target_dir = fingerprints_root / target_exp

        try:
            ref_df = load_fingerprint_data(ref_dir, ref_key)
            target_df = load_fingerprint_data(target_dir, target_key)
        except FileNotFoundError as e:
            print(f"\n  ⚠ Skipping {comparison_name}: {e}")
            continue

        print(f"\n  {comparison_name}...")

        for metric in metrics_of_interest:
            ref_col = _resolve_metric_column(ref_df, metric)
            target_col = _resolve_metric_column(target_df, metric)
            if ref_col is None or target_col is None:
                print(f"    ⚠ Skipping {metric} (not present in both datasets)")
                continue

            ref_values = ref_df[ref_col].dropna().values
            target_values = target_df[target_col].dropna().values

            print(f"\n    Testing: {metric}")
            print(f"      Reference ({ref_key}):  n={len(ref_values)}, mean={np.mean(ref_values):.4f}")
            print(f"      Target ({target_key}): n={len(target_values)}, mean={np.mean(target_values):.4f}")

            perm_result = permutation_test(ref_values, target_values, n_permutations=10000)
            print(f"      Permutation test p-value: {perm_result['p_value']:.6f}")
            print(f"      Effect size (Cohen's d): {perm_result['effect_size']:.4f}")

            u_stat, u_pvalue = stats.mannwhitneyu(ref_values, target_values, alternative="two-sided")
            print(f"      Mann-Whitney U p-value: {u_pvalue:.6f}")

            t_stat, t_pvalue = stats.ttest_ind(ref_values, target_values, equal_var=False)
            print(f"      Welch t-test p-value: {t_pvalue:.6f}")

            results.append({
                "comparison": comparison_name,
                "metric": metric,
                "ref_dataset": ref_key,
                "ref_n": len(ref_values),
                "ref_mean": np.mean(ref_values),
                "ref_std": np.std(ref_values),
                "target_dataset": target_key,
                "target_n": len(target_values),
                "target_mean": np.mean(target_values),
                "target_std": np.std(target_values),
                "mean_difference": np.mean(target_values) - np.mean(ref_values),
                "permutation_p": perm_result["p_value"],
                "permutation_effect_size": perm_result["effect_size"],
                "mann_whitney_p": u_pvalue,
                "t_test_p": t_pvalue,
                "significant_at_0.05": perm_result["p_value"] < 0.05,
                "significant_at_0.01": perm_result["p_value"] < 0.01,
            })

    # Save results
    output_dir = root / "results" / "metrics" / "divergence"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df["permutation_p_fdr"] = benjamini_hochberg(results_df["permutation_p"])
    results_df["mann_whitney_p_fdr"] = benjamini_hochberg(results_df["mann_whitney_p"])
    results_df["t_test_p_fdr"] = benjamini_hochberg(results_df["t_test_p"])
    results_df["significant_perm_fdr"] = results_df["permutation_p_fdr"] < alpha
    results_df["significant_mwu_fdr"] = results_df["mann_whitney_p_fdr"] < alpha
    results_df["significant_ttest_fdr"] = results_df["t_test_p_fdr"] < alpha
    results_path = output_dir / "hypothesis_tests.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n{'='*70}")
    print(f"✅ Hypothesis Testing Complete!")
    print(f"{'='*70}")
    print(f"Results saved: {results_path}")
    print(f"Alpha (FDR): {alpha}")
    print(f"\nTotal tests: {len(results)}")
    print(f"Significant at p<0.05: {results_df['significant_at_0.05'].sum() if 'significant_at_0.05' in results_df else 'N/A'}")
    print(f"Significant (Permutation, FDR): {results_df['significant_perm_fdr'].sum()}")

    # Print summary table
    print("\n" + "="*70)
    print("Summary of Significant Findings (p < 0.05)")
    print("="*70)

    if 'permutation_p' in results_df.columns:
        sig_results = results_df[results_df['significant_perm_fdr'] == True]
        if len(sig_results) > 0:
            for _, row in sig_results.iterrows():
                print(f"\n{row['comparison']}: {row['metric']}")
                if 'mean_difference' in row:
                    print(f"  Mean difference: {row['mean_difference']:.4f}")
                if 'permutation_p' in row:
                    print(f"  Permutation p-value: {row['permutation_p']:.6f}")
                if 'permutation_effect_size' in row:
                    print(f"  Effect size: {row['permutation_effect_size']:.4f}")
        else:
            print("  No significant differences detected at FDR<alpha")


if __name__ == "__main__":
    main()
