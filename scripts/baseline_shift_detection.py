#!/usr/bin/env python3
"""
Baseline Shift Detection Comparisons
======================================

This script implements baseline shift detection methods to compare against
attribution fingerprinting:

1. Input-level shift detection:
   - Maximum Mean Discrepancy (MMD) on pixel intensities
   - Kolmogorov-Smirnov test on pixel distributions

2. Feature-level shift detection:
   - MMD on model embeddings (if available)

3. Prediction-level shift detection:
   - Confidence score distributions
   - Calibration metrics

This addresses the expert review's concern that we need to show attribution
fingerprinting detects shift that OTHER methods miss.

Usage:
    python scripts/baseline_shift_detection.py --alpha 0.05
"""
import _path_setup  # noqa: F401 - ensures xfp is importable

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import argparse
from typing import Optional, List
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')


from xfp.shift.divergence import compute_shift_scores  # noqa: E402


class BaselineShiftDetection:
    """Baseline shift detection methods for comparison."""

    def __init__(self, alpha: float, dataset_keys: List[str], output_dir: Path):
        self.root = PROJECT_ROOT
        self.fingerprint_dir = self.root / "data" / "fingerprints"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alpha = alpha
        self.max_mmd_samples = 1000
        self.mmd_permutations = 1000
        self.mmd_permutations_sampled = 200
        self.mmd_seed = 42

        self.comparisons = self._build_comparisons(dataset_keys)
        self.comparison_data = {}
        self.results = []

    def log(self, message):
        """Log a message."""
        print(f"[INFO] {message}")

    @staticmethod
    def _build_comparisons(dataset_keys: List[str]) -> List[dict]:
        datasets = set(dataset_keys)
        comparisons = []
        if {"jsrt", "montgomery"}.issubset(datasets):
            comparisons.append(
                {
                    "name": "JSRT vs Montgomery",
                    "folder": "jsrt_to_montgomery",
                    "ref_key": "jsrt",
                    "target_key": "montgomery",
                }
            )
        if {"jsrt", "shenzhen"}.issubset(datasets):
            comparisons.append(
                {
                    "name": "JSRT vs Shenzhen",
                    "folder": "jsrt_to_shenzhen",
                    "ref_key": "jsrt",
                    "target_key": "shenzhen",
                }
            )
        if {"montgomery", "shenzhen"}.issubset(datasets):
            comparisons.append(
                {
                    "name": "Montgomery vs Shenzhen",
                    "folder": "montgomery_to_shenzhen",
                    "ref_key": "montgomery",
                    "target_key": "shenzhen",
                }
            )
        if {"jsrt", "nih_chestxray14"}.issubset(datasets):
            comparisons.append(
                {
                    "name": "JSRT vs NIH",
                    "folder": "jsrt_to_nih",
                    "ref_key": "jsrt",
                    "target_key": "nih_chestxray14",
                }
            )
        if {"montgomery", "nih_chestxray14"}.issubset(datasets):
            comparisons.append(
                {
                    "name": "Montgomery vs NIH",
                    "folder": "montgomery_to_nih",
                    "ref_key": "montgomery",
                    "target_key": "nih_chestxray14",
                }
            )
        if {"shenzhen", "nih_chestxray14"}.issubset(datasets):
            comparisons.append(
                {
                    "name": "Shenzhen vs NIH",
                    "folder": "shenzhen_to_nih",
                    "ref_key": "shenzhen",
                    "target_key": "nih_chestxray14",
                }
            )
        return comparisons

    @staticmethod
    def _benjamini_hochberg(p_values: pd.Series) -> pd.Series:
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

    @staticmethod
    def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
        mean_diff = np.mean(x) - np.mean(y)
        pooled_std = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
        return mean_diff / pooled_std if pooled_std > 0 else 0.0

    @staticmethod
    def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
        x = x.reshape(-1, 1)
        y = y.reshape(1, -1)
        greater = np.sum(x > y)
        less = np.sum(x < y)
        denom = x.size * y.size
        return (greater - less) / denom if denom else 0.0

    def _load_bootstrap_samples(self, folder: str) -> Optional[pd.DataFrame]:
        sample_path = self.root / "reports" / "divergence" / f"bootstrap_samples_{folder}.parquet"
        if sample_path.exists():
            return pd.read_parquet(sample_path)
        return None

    def load_fingerprints(self):
        """Load fingerprint data."""
        self.log("Loading fingerprint data...")

        for config in self.comparisons:
            ref_path = self.fingerprint_dir / config["folder"] / f"{config['ref_key']}.parquet"
            tgt_path = self.fingerprint_dir / config["folder"] / f"{config['target_key']}.parquet"
            if not ref_path.exists() or not tgt_path.exists():
                self.log(f"⚠ Missing fingerprints for {config['name']}: {ref_path} or {tgt_path}")
                continue
            ref_df = pd.read_parquet(ref_path)
            tgt_df = pd.read_parquet(tgt_path)
            self.comparison_data[config["name"]] = {
                "ref_df": ref_df,
                "tgt_df": tgt_df,
                "ref_path": ref_path,
                "tgt_path": tgt_path,
                "ref_key": config["ref_key"],
                "target_key": config["target_key"],
            }
            self.log(f"✓ {config['name']} -> {len(ref_df)} vs {len(tgt_df)} samples")

    def compute_mmd(self, X, Y, kernel='rbf', gamma=None):
        """
        Compute Maximum Mean Discrepancy (MMD) between two distributions.

        MMD is a kernel-based measure of distribution difference.
        Higher MMD = more different distributions.
        """
        if gamma is None:
            # Use median heuristic for bandwidth
            gamma = 1.0 / X.shape[1]

        # Compute kernel matrices
        XX = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))
        YY = np.exp(-gamma * cdist(Y, Y, 'sqeuclidean'))
        XY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))

        # MMD^2 = mean(K(X,X)) + mean(K(Y,Y)) - 2*mean(K(X,Y))
        mmd_sq = XX.mean() + YY.mean() - 2 * XY.mean()
        mmd = np.sqrt(max(0, mmd_sq))

        return mmd

    def permutation_test_mmd(self, X, Y, n_permutations=1000, seed=42):
        """
        Permutation test for MMD significance.
        """
        np.random.seed(seed)

        # Observed MMD
        observed_mmd = self.compute_mmd(X, Y)

        # Permutation test
        combined = np.vstack([X, Y])
        n_X = len(X)

        permuted_mmds = []
        for _ in range(n_permutations):
            perm_idx = np.random.permutation(len(combined))
            perm_X = combined[perm_idx[:n_X]]
            perm_Y = combined[perm_idx[n_X:]]
            permuted_mmds.append(self.compute_mmd(perm_X, perm_Y))

        # P-value
        p_value = np.mean(np.array(permuted_mmds) >= observed_mmd)

        return observed_mmd, p_value

    def input_level_shift_detection(self, ref_df, tgt_df):
        """
        Detect shift at input level (pixel distributions).

        Note: We don't have raw images, but we have histogram features
        which are derived from input pixels.
        """
        results = {"n_ref": len(ref_df), "n_target": len(tgt_df)}

        # Extract histogram features (proxy for pixel distributions)
        hist_cols = [col for col in ref_df.columns if col.startswith('hist_bin_')]

        if not hist_cols:
            return results

        ref_hist = ref_df[hist_cols].values
        tgt_hist = tgt_df[hist_cols].values

        # 1. MMD on pixel histogram distributions
        rng = np.random.default_rng(self.mmd_seed)
        ref_hist_mmd = ref_hist
        tgt_hist_mmd = tgt_hist
        if len(ref_hist) > self.max_mmd_samples:
            ref_hist_mmd = ref_hist[rng.choice(len(ref_hist), self.max_mmd_samples, replace=False)]
            self.log(f"Subsampled reference histograms for MMD: {len(ref_hist)} -> {len(ref_hist_mmd)}")
        if len(tgt_hist) > self.max_mmd_samples:
            tgt_hist_mmd = tgt_hist[rng.choice(len(tgt_hist), self.max_mmd_samples, replace=False)]
            self.log(f"Subsampled target histograms for MMD: {len(tgt_hist)} -> {len(tgt_hist_mmd)}")

        n_permutations = self.mmd_permutations
        if len(ref_hist_mmd) < len(ref_hist) or len(tgt_hist_mmd) < len(tgt_hist):
            n_permutations = self.mmd_permutations_sampled

        mmd, p_value = self.permutation_test_mmd(
            ref_hist_mmd,
            tgt_hist_mmd,
            n_permutations=n_permutations,
            seed=self.mmd_seed,
        )

        results["histogram_mmd"] = mmd
        results["histogram_mmd_pvalue"] = p_value
        results["histogram_mmd_n_ref"] = len(ref_hist_mmd)
        results["histogram_mmd_n_target"] = len(tgt_hist_mmd)
        results["histogram_mmd_permutations"] = n_permutations

        # 2. Kolmogorov-Smirnov test on mean pixel intensity
        # Use hist_bin_00 as proxy for overall brightness
        if 'hist_bin_00' in ref_df.columns:
            ref_brightness = ref_df['hist_bin_00'].values
            tgt_brightness = tgt_df['hist_bin_00'].values

            ks_stat, ks_pvalue = stats.ks_2samp(ref_brightness, tgt_brightness)

            results["ks_statistic"] = ks_stat
            results["ks_pvalue"] = ks_pvalue

        # 3. T-test on mean histogram features
        mean_hist_ref = ref_hist.mean(axis=1)
        mean_hist_tgt = tgt_hist.mean(axis=1)

        t_stat, t_pvalue = stats.ttest_ind(mean_hist_ref, mean_hist_tgt)

        results["ttest_statistic"] = t_stat
        results["ttest_pvalue"] = t_pvalue
        return results

    def prediction_level_shift_detection(self, ref_df, tgt_df):
        """
        Detect shift at prediction level (model outputs).
        """
        results = {}

        # 1. Dice score distribution shift
        if 'dice' in ref_df.columns and 'dice' in tgt_df.columns:
            ref_dice = ref_df['dice'].values
            tgt_dice = tgt_df['dice'].values

            # T-test
            t_stat, t_pvalue = stats.ttest_ind(ref_dice, tgt_dice)

            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_dice, tgt_dice)

            # Effect sizes
            cohens_d = self._cohens_d(ref_dice, tgt_dice)
            cliffs_delta = self._cliffs_delta(ref_dice, tgt_dice)

            results["dice_ttest_stat"] = t_stat
            results["dice_ttest_pvalue"] = t_pvalue
            results["dice_ks_stat"] = ks_stat
            results["dice_ks_pvalue"] = ks_pvalue
            results["dice_cohens_d"] = cohens_d
            results["dice_cliffs_delta"] = cliffs_delta

        # 2. Coverage (calibration proxy) shift
        if 'coverage_auc' in ref_df.columns and 'coverage_auc' in tgt_df.columns:
            ref_cov = ref_df['coverage_auc'].values
            tgt_cov = tgt_df['coverage_auc'].values

            t_stat, t_pvalue = stats.ttest_ind(ref_cov, tgt_cov)
            cohens_d = self._cohens_d(ref_cov, tgt_cov)
            cliffs_delta = self._cliffs_delta(ref_cov, tgt_cov)

            results["coverage_ttest_stat"] = t_stat
            results["coverage_ttest_pvalue"] = t_pvalue
            results["coverage_cohens_d"] = cohens_d
            results["coverage_cliffs_delta"] = cliffs_delta

        # 3. IoU shift (native or derived from Dice)
        iou_source = None
        if 'iou' in ref_df.columns and 'iou' in tgt_df.columns:
            ref_iou = ref_df['iou'].values
            tgt_iou = tgt_df['iou'].values
            iou_source = "native"
        elif 'dice' in ref_df.columns and 'dice' in tgt_df.columns:
            ref_dice = ref_df['dice'].values
            tgt_dice = tgt_df['dice'].values
            ref_iou = ref_dice / np.clip(2 - ref_dice, 1e-6, None)
            tgt_iou = tgt_dice / np.clip(2 - tgt_dice, 1e-6, None)
            iou_source = "derived_from_dice"

        if iou_source is not None:
            t_stat, t_pvalue = stats.ttest_ind(ref_iou, tgt_iou)
            cohens_d = self._cohens_d(ref_iou, tgt_iou)
            cliffs_delta = self._cliffs_delta(ref_iou, tgt_iou)
            results["iou_ttest_stat"] = t_stat
            results["iou_ttest_pvalue"] = t_pvalue
            results["iou_cohens_d"] = cohens_d
            results["iou_cliffs_delta"] = cliffs_delta
            results["iou_source"] = iou_source
        return results

    def explanation_level_shift_detection(self, ref_path, tgt_path):
        """Detect shift at the explanation level (attribution fingerprints)."""
        scores = compute_shift_scores(
            reference=ref_path,
            target=tgt_path,
            metrics=["kl_divergence", "emd", "graph_edit_distance"],
        ).scores
        return scores

    def generate_comparison_table(self):
        """
        Generate comparison table showing all shift detection methods.
        """
        self.log("\n" + "=" * 80)
        self.log("GENERATING COMPARISON TABLE")
        self.log("=" * 80)

        rows = []
        for result in self.results:
            comparison = result["comparison"]
            input_level = result["input_level"]
            prediction_level = result["prediction_level"]
            explanation_level = result["explanation_level"]
            ref_n = input_level.get("n_ref")
            tgt_n = input_level.get("n_target")

            if "histogram_mmd_pvalue" in input_level:
                rows.append({
                    "Comparison": comparison,
                    "Method": "Input MMD (Histograms)",
                    "Level": "Input",
                    "Shift Detected": "Yes" if input_level["histogram_mmd_pvalue"] < self.alpha else "No",
                    "Alpha": self.alpha,
                    "p-value": input_level["histogram_mmd_pvalue"],
                    "Statistic": input_level["histogram_mmd"],
                    "Effect Size": np.nan,
                    "Effect Size Type": "",
                    "Ref N": ref_n,
                    "Target N": tgt_n,
                    "CI 2.5%": np.nan,
                    "CI 97.5%": np.nan,
                    "P-Value Method": "Permutation (MMD)",
                    "Interpretation": "Distribution difference in pixel histograms",
                })

            if "ks_pvalue" in input_level:
                rows.append({
                    "Comparison": comparison,
                    "Method": "KS Test (Brightness)",
                    "Level": "Input",
                    "Shift Detected": "Yes" if input_level["ks_pvalue"] < self.alpha else "No",
                    "Alpha": self.alpha,
                    "p-value": input_level["ks_pvalue"],
                    "Statistic": input_level["ks_statistic"],
                    "Effect Size": np.nan,
                    "Effect Size Type": "",
                    "Ref N": ref_n,
                    "Target N": tgt_n,
                    "CI 2.5%": np.nan,
                    "CI 97.5%": np.nan,
                    "P-Value Method": "KS",
                    "Interpretation": "Difference in brightness distribution",
                })

            if "dice_ttest_pvalue" in prediction_level:
                rows.append({
                    "Comparison": comparison,
                    "Method": "Dice Score Shift",
                    "Level": "Prediction",
                    "Shift Detected": "Yes" if prediction_level["dice_ttest_pvalue"] < self.alpha else "No",
                    "Alpha": self.alpha,
                    "p-value": prediction_level["dice_ttest_pvalue"],
                    "Statistic": prediction_level["dice_cohens_d"],
                    "Effect Size": prediction_level.get("dice_cliffs_delta"),
                    "Effect Size Type": "Cliff's delta",
                    "Ref N": ref_n,
                    "Target N": tgt_n,
                    "CI 2.5%": np.nan,
                    "CI 97.5%": np.nan,
                    "P-Value Method": "t-test",
                    "Interpretation": "Performance difference (Cohen's d)",
                })

            if "coverage_ttest_pvalue" in prediction_level:
                rows.append({
                    "Comparison": comparison,
                    "Method": "Coverage Shift",
                    "Level": "Prediction",
                    "Shift Detected": "Yes" if prediction_level["coverage_ttest_pvalue"] < self.alpha else "No",
                    "Alpha": self.alpha,
                    "p-value": prediction_level["coverage_ttest_pvalue"],
                    "Statistic": prediction_level.get("coverage_cohens_d"),
                    "Effect Size": prediction_level.get("coverage_cliffs_delta"),
                    "Effect Size Type": "Cliff's delta",
                    "Ref N": ref_n,
                    "Target N": tgt_n,
                    "CI 2.5%": np.nan,
                    "CI 97.5%": np.nan,
                    "P-Value Method": "t-test",
                    "Interpretation": "Calibration proxy difference (Cohen's d)",
                })

            if "iou_ttest_pvalue" in prediction_level:
                rows.append({
                    "Comparison": comparison,
                    "Method": "IoU Shift",
                    "Level": "Prediction",
                    "Shift Detected": "Yes" if prediction_level["iou_ttest_pvalue"] < self.alpha else "No",
                    "Alpha": self.alpha,
                    "p-value": prediction_level["iou_ttest_pvalue"],
                    "Statistic": prediction_level.get("iou_cohens_d"),
                    "Effect Size": prediction_level.get("iou_cliffs_delta"),
                    "Effect Size Type": "Cliff's delta",
                    "Ref N": ref_n,
                    "Target N": tgt_n,
                    "CI 2.5%": np.nan,
                    "CI 97.5%": np.nan,
                    "P-Value Method": "t-test",
                    "Interpretation": f"IoU shift ({prediction_level.get('iou_source', 'unknown')})",
                })

            if explanation_level:
                folder = result["folder"]
                samples = self._load_bootstrap_samples(folder)
                for metric, label in [
                    ("kl_divergence", "Attribution Fingerprints (KL)"),
                    ("emd", "Attribution Fingerprints (EMD)"),
                    ("graph_edit_distance", "Attribution Fingerprints (GED)"),
                ]:
                    metric_value = explanation_level.get(metric, float("nan"))
                    p_value = np.nan
                    ci_low = np.nan
                    ci_high = np.nan
                    shift_detected = "Unknown"
                    if samples is not None and metric in samples.columns:
                        metric_samples = samples[metric].to_numpy(dtype=float)
                        ci_low = float(np.quantile(metric_samples, 0.025))
                        ci_high = float(np.quantile(metric_samples, 0.975))
                        # One-sided bootstrap p-value against 0 (positive divergences)
                        p_hat = np.mean(metric_samples <= 0.0)
                        p_value = max(p_hat, 1e-12)
                        shift_detected = "Yes" if ci_low > 0.0 else "No"

                    rows.append({
                        "Comparison": comparison,
                        "Method": label,
                        "Level": "Explanation",
                        "Shift Detected": shift_detected,
                        "Alpha": self.alpha,
                        "p-value": p_value,
                        "Statistic": metric_value,
                        "Effect Size": np.nan,
                        "Effect Size Type": "",
                        "Ref N": ref_n,
                        "Target N": tgt_n,
                        "CI 2.5%": ci_low,
                        "CI 97.5%": ci_high,
                        "P-Value Method": "Bootstrap (CI vs 0)",
                        "Interpretation": "Explanation pattern divergence"
                        if metric == "kl_divergence"
                        else "Spatial distribution difference"
                        if metric == "emd"
                        else "Topological structure difference",
                    })

        df = pd.DataFrame(rows)
        df["p_value_fdr"] = self._benjamini_hochberg(df["p-value"])
        df["Shift Detected (FDR)"] = np.where(
            df["p_value_fdr"].notna(),
            np.where(df["p_value_fdr"] < self.alpha, "Yes", "No"),
            df["Shift Detected"],
        )
        output_path = self.output_dir / "baseline_comparison_table.csv"
        df.to_csv(output_path, index=False)

        self.log(f"✓ Saved comparison table to: {output_path}")
        self.log("\n" + "=" * 80)
        self.log("BASELINE COMPARISON RESULTS")
        self.log("=" * 80)
        print(df.to_string(index=False))
        return df

    def run_all(self):
        """Run all baseline shift detection methods."""
        self.log("="*80)
        self.log("BASELINE SHIFT DETECTION COMPARISONS")
        self.log("="*80)

        self.load_fingerprints()
        for comparison, data in self.comparison_data.items():
            self.log("\n" + "=" * 80)
            self.log(f"COMPARISON: {comparison}")
            self.log("=" * 80)

            input_level = self.input_level_shift_detection(data["ref_df"], data["tgt_df"])
            prediction_level = self.prediction_level_shift_detection(data["ref_df"], data["tgt_df"])
            explanation_level = self.explanation_level_shift_detection(data["ref_path"], data["tgt_path"])
            self.results.append({
                "comparison": comparison,
                "folder": data["ref_path"].parent.name,
                "input_level": input_level,
                "prediction_level": prediction_level,
                "explanation_level": explanation_level,
            })

        self.generate_comparison_table()

        self.log("\n" + "="*80)
        self.log("BASELINE SHIFT DETECTION COMPLETE")
        self.log("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline shift detection comparisons.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument(
        "--datasets",
        default="jsrt,montgomery,shenzhen,nih_chestxray14",
        help="Comma-separated dataset keys for comparisons.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports" / "baseline_comparisons"),
        help="Output directory for comparison table.",
    )
    args = parser.parse_args()

    dataset_keys = [key.strip().lower() for key in args.datasets.split(",") if key.strip()]
    detector = BaselineShiftDetection(
        alpha=args.alpha,
        dataset_keys=dataset_keys,
        output_dir=Path(args.output_dir),
    )
    detector.run_all()
