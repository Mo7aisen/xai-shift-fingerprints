import _path_setup  # noqa: F401 - ensures xfp is importable

#!/usr/bin/env python3
"""
Comprehensive Data Correction Script
=====================================

This script fixes all data inconsistencies identified in the expert review:
1. Recalculates divergence metrics from source fingerprints
2. Verifies and fixes Dice scores
3. Regenerates all summary tables with consistent data
4. Creates a verification report

Usage:
    python scripts/fix_data_inconsistencies.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import pandas as pd
import numpy as np
import json


from xfp.shift.divergence import compute_shift_scores


class DataCorrector:
    def __init__(self):
        self.root = PROJECT_ROOT
        self.fingerprint_dir = self.root / "data" / "fingerprints"
        self.reports_dir = self.root / "reports"
        self.manuscript_dir = self.root / "manuscript"

        self.results = {
            "divergence_metrics": {},
            "dice_scores": {},
            "attribution_mass": {},
            "issues_found": [],
            "fixes_applied": []
        }

    def log(self, message, level="INFO"):
        """Log a message with timestamp."""
        print(f"[{level}] {message}")
        if level == "ERROR":
            self.results["issues_found"].append(message)
        elif level == "FIX":
            self.results["fixes_applied"].append(message)

    def load_fingerprints(self, experiment="jsrt_to_montgomery"):
        """Load fingerprint data for an experiment."""
        self.log(f"Loading fingerprints for {experiment}...")

        exp_dir = self.fingerprint_dir / experiment
        jsrt_path = exp_dir / "jsrt.parquet"
        montgomery_path = exp_dir / "montgomery.parquet"

        if not jsrt_path.exists():
            self.log(f"JSRT fingerprint not found: {jsrt_path}", "ERROR")
            return None, None

        if not montgomery_path.exists():
            self.log(f"Montgomery fingerprint not found: {montgomery_path}", "ERROR")
            return None, None

        jsrt_df = pd.read_parquet(jsrt_path)
        montgomery_df = pd.read_parquet(montgomery_path)

        self.log(f"✓ Loaded JSRT: {len(jsrt_df)} samples")
        self.log(f"✓ Loaded Montgomery: {len(montgomery_df)} samples")

        return jsrt_df, montgomery_df

    def compute_correct_divergence(self):
        """Recompute divergence metrics from source data."""
        self.log("\n" + "="*80)
        self.log("RECOMPUTING DIVERGENCE METRICS FROM SOURCE")
        self.log("="*80)

        # Paths to fingerprint files
        ref_path = self.fingerprint_dir / "jsrt_to_montgomery" / "jsrt.parquet"
        tgt_path = self.fingerprint_dir / "jsrt_to_montgomery" / "montgomery.parquet"

        if not ref_path.exists() or not tgt_path.exists():
            self.log("Fingerprint files not found - cannot recompute divergence", "ERROR")
            return None

        # Compute divergence using the library function
        metrics = ["kl_divergence", "emd", "graph_edit_distance"]
        self.log(f"Computing metrics: {metrics}")

        try:
            scores = compute_shift_scores(
                reference=ref_path,
                target=tgt_path,
                metrics=metrics
            )

            kl_div = scores.scores.get("kl_divergence", 0.0)
            emd = scores.scores.get("emd", 0.0)
            ged = scores.scores.get("graph_edit_distance", 0.0)

            self.log(f"\n✓ CORRECT DIVERGENCE METRICS:")
            self.log(f"  KL Divergence: {kl_div:.6f}")
            self.log(f"  EMD: {emd:.6f}")
            self.log(f"  Graph Edit Distance: {ged:.2f}")

            self.results["divergence_metrics"] = {
                "kl_divergence": kl_div,
                "emd": emd,
                "graph_edit_distance": ged
            }

            return scores

        except Exception as e:
            self.log(f"Error computing divergence: {e}", "ERROR")
            return None

    def verify_dice_scores(self):
        """Verify Dice scores from fingerprint data."""
        self.log("\n" + "="*80)
        self.log("VERIFYING DICE SCORES")
        self.log("="*80)

        jsrt_df, montgomery_df = self.load_fingerprints()

        if jsrt_df is None or montgomery_df is None:
            return None

        # Extract Dice scores
        if 'dice' in jsrt_df.columns:
            jsrt_dice_mean = jsrt_df['dice'].mean()
            jsrt_dice_std = jsrt_df['dice'].std()
            self.log(f"✓ JSRT Dice: {jsrt_dice_mean:.4f} ± {jsrt_dice_std:.4f}")

            self.results["dice_scores"]["jsrt_mean"] = jsrt_dice_mean
            self.results["dice_scores"]["jsrt_std"] = jsrt_dice_std
        else:
            self.log("'dice' column not found in JSRT fingerprints", "ERROR")
            jsrt_dice_mean, jsrt_dice_std = None, None

        if 'dice' in montgomery_df.columns:
            mont_dice_mean = montgomery_df['dice'].mean()
            mont_dice_std = montgomery_df['dice'].std()
            self.log(f"✓ Montgomery Dice: {mont_dice_mean:.4f} ± {mont_dice_std:.4f}")

            self.results["dice_scores"]["montgomery_mean"] = mont_dice_mean
            self.results["dice_scores"]["montgomery_std"] = mont_dice_std
        else:
            self.log("'dice' column not found in Montgomery fingerprints", "ERROR")
            mont_dice_mean, mont_dice_std = None, None

        return {
            "jsrt": (jsrt_dice_mean, jsrt_dice_std),
            "montgomery": (mont_dice_mean, mont_dice_std)
        }

    def verify_attribution_mass(self):
        """Verify attribution mass values."""
        self.log("\n" + "="*80)
        self.log("VERIFYING ATTRIBUTION MASS")
        self.log("="*80)

        jsrt_df, montgomery_df = self.load_fingerprints()

        if jsrt_df is None or montgomery_df is None:
            return None

        # Check for attribution mass columns
        mass_cols = [col for col in jsrt_df.columns if 'attribution_abs_sum' in col.lower()]

        if not mass_cols:
            self.log("No attribution mass columns found", "ERROR")
            return None

        for col in mass_cols:
            if col in jsrt_df.columns and col in montgomery_df.columns:
                jsrt_mean = jsrt_df[col].mean()
                jsrt_std = jsrt_df[col].std()
                mont_mean = montgomery_df[col].mean()
                mont_std = montgomery_df[col].std()

                collapse_pct = (1 - mont_mean/jsrt_mean) * 100

                self.log(f"\n{col}:")
                self.log(f"  JSRT: {jsrt_mean:.2f} ± {jsrt_std:.2f}")
                self.log(f"  Montgomery: {mont_mean:.2f} ± {mont_std:.2f}")
                self.log(f"  Collapse: {collapse_pct:.1f}%")

                self.results["attribution_mass"][col] = {
                    "jsrt_mean": jsrt_mean,
                    "jsrt_std": jsrt_std,
                    "montgomery_mean": mont_mean,
                    "montgomery_std": mont_std,
                    "collapse_percent": collapse_pct
                }

    def regenerate_divergence_table(self):
        """Regenerate the correct divergence comparison table."""
        self.log("\n" + "="*80)
        self.log("REGENERATING DIVERGENCE COMPARISON TABLE")
        self.log("="*80)

        if not self.results["divergence_metrics"]:
            self.log("No divergence metrics computed - skipping table generation", "ERROR")
            return

        jsrt_df, montgomery_df = self.load_fingerprints()

        if jsrt_df is None or montgomery_df is None:
            return

        # Get Dice scores
        jsrt_dice_mean = jsrt_df['dice'].mean() if 'dice' in jsrt_df.columns else 0.0
        jsrt_dice_std = jsrt_df['dice'].std() if 'dice' in jsrt_df.columns else 0.0
        mont_dice_mean = montgomery_df['dice'].mean() if 'dice' in montgomery_df.columns else 0.0
        mont_dice_std = montgomery_df['dice'].std() if 'dice' in montgomery_df.columns else 0.0

        # Get attribution mass (IG)
        jsrt_attr_mean = jsrt_df['attribution_abs_sum'].mean() if 'attribution_abs_sum' in jsrt_df.columns else 0.0
        jsrt_attr_std = jsrt_df['attribution_abs_sum'].std() if 'attribution_abs_sum' in jsrt_df.columns else 0.0
        mont_attr_mean = montgomery_df['attribution_abs_sum'].mean() if 'attribution_abs_sum' in montgomery_df.columns else 0.0
        mont_attr_std = montgomery_df['attribution_abs_sum'].std() if 'attribution_abs_sum' in montgomery_df.columns else 0.0

        # Get border mass
        jsrt_border_mean = jsrt_df['border_abs_sum'].mean() if 'border_abs_sum' in jsrt_df.columns else 0.0
        jsrt_border_std = jsrt_df['border_abs_sum'].std() if 'border_abs_sum' in jsrt_df.columns else 0.0
        mont_border_mean = montgomery_df['border_abs_sum'].mean() if 'border_abs_sum' in montgomery_df.columns else 0.0
        mont_border_std = montgomery_df['border_abs_sum'].std() if 'border_abs_sum' in montgomery_df.columns else 0.0

        # Get entropy
        jsrt_entropy_mean = jsrt_df['hist_entropy'].mean() if 'hist_entropy' in jsrt_df.columns else 0.0
        jsrt_entropy_std = jsrt_df['hist_entropy'].std() if 'hist_entropy' in jsrt_df.columns else 0.0
        mont_entropy_mean = montgomery_df['hist_entropy'].mean() if 'hist_entropy' in montgomery_df.columns else 0.0
        mont_entropy_std = montgomery_df['hist_entropy'].std() if 'hist_entropy' in montgomery_df.columns else 0.0

        # Create table
        table_data = {
            "Comparison": ["JSRT → Montgomery"],
            "Reference Dataset": ["JSRT"],
            "Target Dataset": ["Montgomery"],
            "Reference Samples": [len(jsrt_df)],
            "Target Samples": [len(montgomery_df)],
            "Ref Dice (mean±std)": [f"{jsrt_dice_mean:.3f}±{jsrt_dice_std:.3f}"],
            "Target Dice (mean±std)": [f"{mont_dice_mean:.3f}±{mont_dice_std:.3f}"],
            "Ref Attribution Sum": [f"{jsrt_attr_mean:.2f}±{jsrt_attr_std:.2f}"],
            "Target Attribution Sum": [f"{mont_attr_mean:.2f}±{mont_attr_std:.2f}"],
            "Ref Border Sum": [f"{jsrt_border_mean:.3f}±{jsrt_border_std:.3f}"],
            "Target Border Sum": [f"{mont_border_mean:.3f}±{mont_border_std:.3f}"],
            "Ref Histogram Entropy": [f"{jsrt_entropy_mean:.3f}±{jsrt_entropy_std:.3f}"],
            "Target Histogram Entropy": [f"{mont_entropy_mean:.3f}±{mont_entropy_std:.3f}"],
            "Dice Δ": [mont_dice_mean - jsrt_dice_mean],
            "Attr Sum Ratio": [mont_attr_mean / jsrt_attr_mean if jsrt_attr_mean > 0 else 0.0],
            "Border Sum Ratio": [mont_border_mean / jsrt_border_mean if jsrt_border_mean > 0 else 0.0],
            "Entropy Ratio": [mont_entropy_mean / jsrt_entropy_mean if jsrt_entropy_mean > 0 else 0.0],
            "KL Divergence": [self.results["divergence_metrics"]["kl_divergence"]],
            "EMD": [self.results["divergence_metrics"]["emd"]],
            "Graph Edit Distance": [self.results["divergence_metrics"]["graph_edit_distance"]]
        }

        df = pd.DataFrame(table_data)

        # Save to both locations
        output_path1 = self.root / "divergence_comparison_table.csv"
        output_path2 = self.reports_dir / "divergence" / "divergence_comparison_table.csv"

        output_path2.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path1, index=False)
        df.to_csv(output_path2, index=False)

        self.log(f"✓ Saved corrected table to: {output_path1}", "FIX")
        self.log(f"✓ Saved corrected table to: {output_path2}", "FIX")

        return df

    def regenerate_fingerprint_summary(self):
        """Regenerate the fingerprint summary table."""
        self.log("\n" + "="*80)
        self.log("REGENERATING FINGERPRINT SUMMARY TABLE")
        self.log("="*80)

        jsrt_df, montgomery_df = self.load_fingerprints()

        if jsrt_df is None or montgomery_df is None:
            return

        # Metrics to include
        metrics = [
            'dice',
            'coverage_auc',
            'attribution_abs_sum',  # This is IG Mass
            'border_abs_sum',
            'hist_entropy'
        ]

        metric_labels = {
            'dice': 'Dice',
            'coverage_auc': 'Coverage AUC',
            'attribution_abs_sum': 'IG Mass',
            'border_abs_sum': 'Border IG Mass',
            'hist_entropy': 'Entropy'
        }

        summary_data = []
        for metric in metrics:
            if metric in jsrt_df.columns and metric in montgomery_df.columns:
                row = {
                    "Metric": metric_labels.get(metric, metric),
                    "JSRT Mean": jsrt_df[metric].mean(),
                    "Montgomery Mean": montgomery_df[metric].mean(),
                    "JSRT Std": jsrt_df[metric].std(),
                    "Mont Std": montgomery_df[metric].std()
                }
                summary_data.append(row)

        df = pd.DataFrame(summary_data)

        # Save
        output_path = self.manuscript_dir / "tables" / "fingerprint_summary.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        self.log(f"✓ Saved corrected summary to: {output_path}", "FIX")

        return df

    def generate_report(self):
        """Generate final verification report."""
        self.log("\n" + "="*80)
        self.log("GENERATING CORRECTION REPORT")
        self.log("="*80)

        report_path = self.root / "DATA_CORRECTION_REPORT.md"

        with open(report_path, 'w') as f:
            f.write("# Data Correction Report\n\n")
            f.write(f"**Generated**: {pd.Timestamp.now()}\n\n")

            f.write("## Corrected Divergence Metrics\n\n")
            if self.results["divergence_metrics"]:
                f.write("| Metric | Corrected Value |\n")
                f.write("|--------|----------------|\n")
                f.write(f"| KL Divergence | {self.results['divergence_metrics']['kl_divergence']:.6f} |\n")
                f.write(f"| EMD | {self.results['divergence_metrics']['emd']:.6f} |\n")
                f.write(f"| Graph Edit Distance | {self.results['divergence_metrics']['graph_edit_distance']:.2f} |\n")

            f.write("\n## Verified Dice Scores\n\n")
            if self.results["dice_scores"]:
                f.write("| Dataset | Mean | Std |\n")
                f.write("|---------|------|-----|\n")
                f.write(f"| JSRT | {self.results['dice_scores'].get('jsrt_mean', 0):.4f} | "
                       f"{self.results['dice_scores'].get('jsrt_std', 0):.4f} |\n")
                f.write(f"| Montgomery | {self.results['dice_scores'].get('montgomery_mean', 0):.4f} | "
                       f"{self.results['dice_scores'].get('montgomery_std', 0):.4f} |\n")

            f.write("\n## Issues Found\n\n")
            if self.results["issues_found"]:
                for issue in self.results["issues_found"]:
                    f.write(f"- {issue}\n")
            else:
                f.write("No issues found.\n")

            f.write("\n## Fixes Applied\n\n")
            if self.results["fixes_applied"]:
                for fix in self.results["fixes_applied"]:
                    f.write(f"- {fix}\n")
            else:
                f.write("No fixes applied.\n")

            f.write("\n## Recommendations\n\n")
            f.write("1. Update manuscript with corrected divergence metrics\n")
            f.write("2. Verify all tables use data from corrected CSV files\n")
            f.write("3. Re-run statistical tests with corrected data\n")
            f.write("4. Proceed with additional experiments (single-model shift detection, baselines)\n")

        self.log(f"✓ Saved correction report to: {report_path}")

        # Also save JSON for programmatic access
        json_path = self.root / "DATA_CORRECTION_REPORT.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        self.log(f"✓ Saved JSON report to: {json_path}")

    def run_all_corrections(self):
        """Run all correction steps."""
        self.log("="*80)
        self.log("STARTING COMPREHENSIVE DATA CORRECTION")
        self.log("="*80)

        # Step 1: Recompute divergence
        self.compute_correct_divergence()

        # Step 2: Verify Dice scores
        self.verify_dice_scores()

        # Step 3: Verify attribution mass
        self.verify_attribution_mass()

        # Step 4: Regenerate tables
        self.regenerate_divergence_table()
        self.regenerate_fingerprint_summary()

        # Step 5: Generate report
        self.generate_report()

        self.log("\n" + "="*80)
        self.log("DATA CORRECTION COMPLETE")
        self.log("="*80)

        return self.results


if __name__ == "__main__":
    corrector = DataCorrector()
    corrector.run_all_corrections()
