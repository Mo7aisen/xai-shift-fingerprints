#!/usr/bin/env python3
"""
Feature Importance and Dimensionality Reduction Analysis
=========================================================

This script addresses the expert review's concern about high dimensionality
(61-dimensional fingerprint) with limited samples (385 total).

Analyses:
1. Feature importance ranking (which features discriminate best?)
2. Principal Component Analysis (PCA)
3. Dimensionality reduction validation
4. Feature selection for parsimonious shift detection

Usage:
    python scripts/feature_importance_pca.py --datasets jsrt,montgomery,shenzhen
"""
import _path_setup  # noqa: F401 - ensures xfp is importable

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import argparse
from typing import List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

warnings.filterwarnings('ignore')



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature importance and PCA analysis.")
    parser.add_argument(
        "--datasets",
        default="jsrt,montgomery,shenzhen",
        help="Comma-separated dataset keys (e.g., jsrt,montgomery or jsrt,montgomery,shenzhen).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports" / "feature_analysis"),
        help="Output directory for analysis artifacts.",
    )
    return parser.parse_args()


class FeatureAnalysis:
    """Feature importance and dimensionality reduction."""

    def __init__(self, dataset_keys: List[str], output_dir: Path):
        self.root = PROJECT_ROOT
        self.fingerprint_dir = self.root / "data" / "fingerprints"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_keys = dataset_keys

    def log(self, message):
        """Log a message."""
        print(f"[INFO] {message}")

    def load_fingerprints(self):
        """Load fingerprint data."""
        self.log("Loading fingerprint data...")

        dataset_map = {
            "jsrt": ("JSRT", self.fingerprint_dir / "jsrt_to_montgomery" / "jsrt.parquet"),
            "montgomery": ("Montgomery", self.fingerprint_dir / "jsrt_to_montgomery" / "montgomery.parquet"),
            "shenzhen": ("Shenzhen", self.fingerprint_dir / "jsrt_to_shenzhen" / "shenzhen.parquet"),
        }

        self.datasets = {}
        for key in self.dataset_keys:
            if key not in dataset_map:
                raise ValueError(f"Unsupported dataset key: {key}")
            name, path = dataset_map[key]
            self.datasets[name] = pd.read_parquet(path)

        for name, df in self.datasets.items():
            self.log(f"✓ {name}: {len(df)} samples")

        # Get feature columns - only common columns between datasets
        exclude_cols = ['sample_id', 'experiment', 'dataset', 'attribution_method', 'filename']

        common_cols = None
        for df in self.datasets.values():
            cols = set(df.columns)
            common_cols = cols if common_cols is None else common_cols & cols

        self.feature_cols = [col for col in common_cols if col not in exclude_cols]

        # Filter to numeric features
        numeric_features = []
        for col in self.feature_cols:
            if all(pd.api.types.is_numeric_dtype(df[col]) for df in self.datasets.values()):
                numeric_features.append(col)

        self.feature_cols = numeric_features
        self.log(f"✓ Numeric features (common to all datasets): {len(self.feature_cols)}")
        self._write_metadata()

    def _write_metadata(self):
        metadata = {
            "datasets": list(self.datasets.keys()),
            "dataset_keys": self.dataset_keys,
            "n_groups": len(self.datasets),
            "feature_count": len(self.feature_cols),
            "output_dir": str(self.output_dir),
        }
        output_path = self.output_dir / "analysis_metadata.json"
        pd.Series(metadata).to_json(output_path)
        self.log(f"✓ Saved analysis metadata to: {output_path}")

    def univariate_feature_importance(self):
        """
        Rank features by univariate discriminative power.

        Uses effect size (Cohen's d) and statistical significance.
        """
        self.log("\n" + "="*80)
        self.log("UNIVARIATE FEATURE IMPORTANCE")
        self.log("="*80)

        importance_scores = []

        for feature in self.feature_cols:
            group_vals = {}
            valid = True
            for name, df in self.datasets.items():
                vals = pd.to_numeric(df[feature].values, errors='coerce')
                vals = vals[np.isfinite(vals)]
                if len(vals) < 2:
                    valid = False
                    break
                group_vals[name] = vals
            if not valid:
                continue

            groups = list(group_vals.values())
            all_vals = np.concatenate(groups)

            if len(groups) == 2:
                x, y = groups
                if np.allclose(all_vals, all_vals[0]):
                    t_stat, t_pvalue = 0.0, 1.0
                    u_stat, u_pvalue = 0.0, 1.0
                    cohens_d = 0.0
                    cliffs_delta = 0.0
                else:
                    t_stat, t_pvalue = stats.ttest_ind(x, y, equal_var=False)
                    u_stat, u_pvalue = stats.mannwhitneyu(x, y, alternative='two-sided')
                    mean_diff = np.mean(x) - np.mean(y)
                    pooled_std = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
                    x_ = x.reshape(-1, 1)
                    y_ = y.reshape(1, -1)
                    greater = np.sum(x_ > y_)
                    less = np.sum(x_ < y_)
                    cliffs_delta = (greater - less) / (x_.size * y_.size)

                row = {
                    'feature': feature,
                    'effect_size': abs(cohens_d),
                    'effect_size_type': 'Cohen d',
                    'primary_pvalue': t_pvalue,
                    'secondary_pvalue': u_pvalue,
                    'ttest_pvalue': t_pvalue,
                    'mannwhitney_pvalue': u_pvalue,
                    'cohens_d': cohens_d,
                    'cliffs_delta': cliffs_delta,
                }
            else:
                # One-way ANOVA and Kruskal-Wallis (guard against constant features)
                if np.allclose(all_vals, all_vals[0]):
                    f_stat, p_value = 0.0, 1.0
                    h_stat, h_pvalue = 0.0, 1.0
                else:
                    f_stat, p_value = stats.f_oneway(*groups)
                    try:
                        h_stat, h_pvalue = stats.kruskal(*groups)
                    except ValueError:
                        h_stat, h_pvalue = 0.0, 1.0

                overall_mean = np.mean(all_vals)
                ss_total = np.sum((all_vals - overall_mean) ** 2)
                ss_between = sum(len(vals) * (np.mean(vals) - overall_mean) ** 2 for vals in groups)
                eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

                row = {
                    'feature': feature,
                    'effect_size': eta_sq,
                    'effect_size_type': 'Eta squared',
                    'primary_pvalue': p_value,
                    'secondary_pvalue': h_pvalue,
                    'eta_squared': eta_sq,
                    'anova_pvalue': p_value,
                    'kruskal_pvalue': h_pvalue,
                }

            for name, vals in group_vals.items():
                row[f"{name.lower()}_mean"] = np.mean(vals)

            importance_scores.append(row)

        importance_df = pd.DataFrame(importance_scores)
        importance_df = importance_df.sort_values('effect_size', ascending=False)

        # Save
        output_path = self.output_dir / "univariate_feature_importance.csv"
        importance_df.to_csv(output_path, index=False)

        self.log(f"✓ Saved univariate importance to: {output_path}")

        # Show top 10
        self.log("\nTop 10 most discriminative features:")
        for i, row in importance_df.head(10).iterrows():
            self.log(f"  {i+1}. {row['feature']}: {row['effect_size_type']}={row['effect_size']:.3f}, p={row['primary_pvalue']:.6f}")

        return importance_df

    def random_forest_importance(self):
        """
        Use Random Forest for multivariate feature importance.

        This captures feature interactions that univariate methods miss.
        """
        self.log("\n" + "="*80)
        self.log("RANDOM FOREST FEATURE IMPORTANCE")
        self.log("="*80)

        # Prepare data
        X_list = []
        y_list = []
        for idx, (name, df) in enumerate(self.datasets.items()):
            X_list.append(df[self.feature_cols].values)
            y_list.append(np.full(len(df), idx))

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Train Random Forest
        self.log("Training Random Forest classifier...")
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)

        # Get feature importances
        importances = rf.feature_importances_

        rf_importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        })
        rf_importance_df = rf_importance_df.sort_values('importance', ascending=False)

        # Save
        output_path = self.output_dir / "random_forest_feature_importance.csv"
        rf_importance_df.to_csv(output_path, index=False)

        self.log(f"✓ Saved RF importance to: {output_path}")
        self.log(f"✓ RF accuracy: {rf.score(X, y):.4f}")

        # Show top 10
        self.log("\nTop 10 features by Random Forest importance:")
        for i, row in rf_importance_df.head(10).iterrows():
            self.log(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")

        return rf_importance_df

    def pca_analysis(self):
        """
        Perform Principal Component Analysis.

        This addresses the concern about 61-dimensional fingerprint
        with only 385 samples.
        """
        self.log("\n" + "="*80)
        self.log("PRINCIPAL COMPONENT ANALYSIS (PCA)")
        self.log("="*80)

        # Prepare data
        X_list = [df[self.feature_cols].values for df in self.datasets.values()]
        X = np.vstack(X_list)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        self.log("Computing PCA...")
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        # Variance explained
        var_explained = pca.explained_variance_ratio_
        cumsum_var = np.cumsum(var_explained)

        # Find number of components for 90%, 95%, 99% variance
        n_90 = np.argmax(cumsum_var >= 0.90) + 1
        n_95 = np.argmax(cumsum_var >= 0.95) + 1
        n_99 = np.argmax(cumsum_var >= 0.99) + 1

        self.log(f"\n✓ Components for 90% variance: {n_90}/{len(self.feature_cols)}")
        self.log(f"✓ Components for 95% variance: {n_95}/{len(self.feature_cols)}")
        self.log(f"✓ Components for 99% variance: {n_99}/{len(self.feature_cols)}")

        # Save variance explained
        pca_df = pd.DataFrame({
            'component': np.arange(1, len(var_explained) + 1),
            'variance_explained': var_explained,
            'cumulative_variance': cumsum_var
        })

        output_path = self.output_dir / "pca_variance_explained.csv"
        pca_df.to_csv(output_path, index=False)

        self.log(f"✓ Saved PCA variance to: {output_path}")

        # Save PCA loadings (which features contribute to each PC)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(
            loadings[:, :10],  # First 10 PCs
            columns=[f'PC{i+1}' for i in range(10)],
            index=self.feature_cols
        )

        output_path = self.output_dir / "pca_loadings.csv"
        loadings_df.to_csv(output_path)

        self.log(f"✓ Saved PCA loadings to: {output_path}")

        # Show top features for first 3 PCs
        self.log("\nTop features for first 3 principal components:")
        for i in range(min(3, len(var_explained))):
            top_idx = np.argsort(np.abs(loadings[:, i]))[::-1][:5]
            self.log(f"\n  PC{i+1} (explains {var_explained[i]*100:.1f}% variance):")
            for idx in top_idx:
                self.log(f"    - {self.feature_cols[idx]}: {loadings[idx, i]:.3f}")

        return pca_df, loadings_df

    def lda_analysis(self):
        """
        Linear Discriminant Analysis for supervised dimensionality reduction.

        LDA finds the linear combination of features that best separates
        JSRT from Montgomery.
        """
        self.log("\n" + "="*80)
        self.log("LINEAR DISCRIMINANT ANALYSIS (LDA)")
        self.log("="*80)

        # Prepare data
        X_list = []
        y_list = []
        for idx, df in enumerate(self.datasets.values()):
            X_list.append(df[self.feature_cols].values)
            y_list.append(np.full(len(df), idx))

        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # LDA (multi-class)
        self.log("Computing LDA...")
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_scaled, y)

        # Get feature weights (mean absolute weight across classes)
        weights = lda.coef_
        mean_abs_weights = np.mean(np.abs(weights), axis=0)
        lda_df = pd.DataFrame({
            'feature': self.feature_cols,
            'lda_weight': mean_abs_weights,
            'abs_lda_weight': mean_abs_weights
        })
        lda_df = lda_df.sort_values('abs_lda_weight', ascending=False)

        # Save
        output_path = self.output_dir / "lda_feature_weights.csv"
        lda_df.to_csv(output_path, index=False)

        self.log(f"✓ Saved LDA weights to: {output_path}")

        # Show top 10
        self.log("\nTop 10 features by LDA weight:")
        for i, row in lda_df.head(10).iterrows():
            self.log(f"  {i+1}. {row['feature']}: {row['lda_weight']:.4f}")

        return lda_df

    def generate_summary_report(self):
        """Generate summary report."""
        self.log("\n" + "="*80)
        self.log("GENERATING SUMMARY REPORT")
        self.log("="*80)

        # Load all results
        univar_df = pd.read_csv(self.output_dir / "univariate_feature_importance.csv")
        rf_df = pd.read_csv(self.output_dir / "random_forest_feature_importance.csv")
        pca_df = pd.read_csv(self.output_dir / "pca_variance_explained.csv")
        lda_df = pd.read_csv(self.output_dir / "lda_feature_weights.csv")

        # Create consensus ranking
        # Normalize ranks across methods
        univar_df['univar_rank'] = univar_df['effect_size'].rank(ascending=False)
        rf_df['rf_rank'] = rf_df['importance'].rank(ascending=False)
        lda_df['lda_rank'] = lda_df['abs_lda_weight'].rank(ascending=False)

        # Merge
        consensus = univar_df[['feature', 'univar_rank']].copy()
        consensus = consensus.merge(rf_df[['feature', 'rf_rank']], on='feature', how='outer')
        consensus = consensus.merge(lda_df[['feature', 'lda_rank']], on='feature', how='outer')

        # Average rank
        consensus['mean_rank'] = consensus[['univar_rank', 'rf_rank', 'lda_rank']].mean(axis=1)
        consensus = consensus.sort_values('mean_rank')

        # Save
        output_path = self.output_dir / "consensus_feature_ranking.csv"
        consensus.to_csv(output_path, index=False)

        self.log(f"✓ Saved consensus ranking to: {output_path}")

        # Show top 15
        self.log("\nTop 15 features by consensus ranking:")
        for i, row in consensus.head(15).iterrows():
            self.log(f"  {i+1}. {row['feature']}: mean_rank={row['mean_rank']:.1f}")

        # PCA summary
        n_90 = (pca_df['cumulative_variance'] >= 0.90).idxmax() + 1
        n_95 = (pca_df['cumulative_variance'] >= 0.95).idxmax() + 1

        self.log(f"\nDimensionality Reduction Summary:")
        self.log(f"  Original features: {len(self.feature_cols)}")
        self.log(f"  Components for 90% variance: {n_90}")
        self.log(f"  Components for 95% variance: {n_95}")
        self.log(f"  Dimensionality reduction: {(1 - n_95/len(self.feature_cols))*100:.1f}%")

    def run_all(self):
        """Run all feature analyses."""
        self.log("="*80)
        self.log("FEATURE IMPORTANCE AND PCA ANALYSIS")
        self.log("="*80)

        self.load_fingerprints()
        self.univariate_feature_importance()
        self.random_forest_importance()
        self.pca_analysis()
        self.lda_analysis()
        self.generate_summary_report()

        self.log("\n" + "="*80)
        self.log("FEATURE ANALYSIS COMPLETE")
        self.log("="*80)


if __name__ == "__main__":
    args = parse_args()
    dataset_keys = [key.strip().lower() for key in args.datasets.split(",") if key.strip()]
    analyzer = FeatureAnalysis(dataset_keys=dataset_keys, output_dir=Path(args.output_dir))
    analyzer.run_all()
