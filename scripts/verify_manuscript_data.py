#!/usr/bin/env python3
"""
Data Verification Script for Attribution Fingerprints Manuscript
=================================================================

This script checks for data consistency issues and verifies that
manuscript numbers match the actual experimental results.

Usage:
    python scripts/verify_manuscript_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

class DataVerifier:
    def __init__(self, root_dir=None):
        if root_dir is None:
            root_dir = Path(__file__).resolve().parents[1]
        self.root = Path(root_dir)
        self.issues = []
        self.warnings = []
        self.manuscript_claims = self.load_manuscript_claims()

    def load_manuscript_claims(self):
        """Load manuscript claims from JSON so the check stays in sync with text."""
        default_claims = {
            "divergence": {
                "kl": 0.0690,
                "emd": 0.4153,
                "ged": 462.3
            },
            "dice": {
                "jsrt": 0.923,
                "montgomery": 0.958
            },
            "ig_mass": {
                "jsrt": 24.10,
                "montgomery": 3.89
            }
        }

        claims_path = self.root / 'manuscript' / 'claims' / 'metrics.json'
        if not claims_path.exists():
            self.warnings.append(
                f"⚠️  Manuscript claims file missing ({claims_path}); using defaults."
            )
            return default_claims

        try:
            with open(claims_path, 'r') as f:
                loaded = json.load(f)
        except Exception as exc:  # pylint: disable=broad-except
            self.warnings.append(
                f"⚠️  Failed to parse manuscript claims ({exc}); using defaults."
            )
            return default_claims

        def merge_dict(base, updates):
            for key, value in updates.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    base[key] = merge_dict(base[key], value)
                else:
                    base[key] = value
            return base

        return merge_dict(default_claims, loaded)

    def check_file_exists(self, filepath):
        """Check if required file exists."""
        full_path = self.root / filepath
        if not full_path.exists():
            self.issues.append(f"❌ MISSING FILE: {filepath}")
            return False
        return True

    def verify_divergence_metrics(self):
        """Verify divergence metrics consistency."""
        print("\n" + "="*80)
        print("VERIFYING DIVERGENCE METRICS")
        print("="*80)

        # Load actual data
        div_file = 'reports/divergence/divergence_comparison_table.csv'
        if not self.check_file_exists(div_file):
            return

        div_df = pd.read_csv(self.root / div_file)

        actual_kl = div_df['KL Divergence'].values[0]
        actual_emd = div_df['EMD'].values[0]
        actual_ged = div_df['Graph Edit Distance'].values[0]

        print(f"\n📊 Actual Data (from CSV):")
        print(f"  KL Divergence: {actual_kl:.4f}")
        print(f"  EMD: {actual_emd:.4f}")
        print(f"  GED: {actual_ged:.2f}")

        claims = self.manuscript_claims.get("divergence", {})
        manuscript_kl = claims.get("kl", actual_kl)
        manuscript_emd = claims.get("emd", actual_emd)
        manuscript_ged = claims.get("ged", actual_ged)

        print(f"\n📄 Manuscript Claims:")
        print(f"  KL Divergence: {manuscript_kl:.4f}")
        print(f"  EMD: {manuscript_emd:.4f}")
        print(f"  GED: {manuscript_ged:.2f}")

        # Check discrepancies
        kl_diff_pct = abs((manuscript_kl - actual_kl) / actual_kl * 100)
        emd_diff_pct = abs((manuscript_emd - actual_emd) / actual_emd * 100)
        ged_diff_pct = abs((manuscript_ged - actual_ged) / actual_ged * 100)

        print(f"\n⚠️  Discrepancies:")
        print(f"  KL: {kl_diff_pct:.1f}% difference")
        print(f"  EMD: {emd_diff_pct:.1f}% difference")
        print(f"  GED: {ged_diff_pct:.1f}% difference")

        if kl_diff_pct > 5:
            self.issues.append(
                f"🔴 CRITICAL: KL divergence mismatch ({kl_diff_pct:.1f}% off)"
            )
        if emd_diff_pct > 5:
            self.issues.append(
                f"🔴 CRITICAL: EMD mismatch ({emd_diff_pct:.1f}% off)"
            )
        if ged_diff_pct > 5:
            self.issues.append(
                f"🔴 CRITICAL: GED mismatch ({ged_diff_pct:.1f}% off)"
            )

    def verify_dice_scores(self):
        """Verify Dice score consistency across files."""
        print("\n" + "="*80)
        print("VERIFYING DICE SCORES")
        print("="*80)

        jsrt_dice_1 = None
        mont_dice_1 = None

        # Check fingerprint summary
        fprint_file = 'manuscript/tables/fingerprint_summary.csv'
        if self.check_file_exists(fprint_file):
            fprint_df = pd.read_csv(self.root / fprint_file)
            dice_row = fprint_df[fprint_df['Metric'] == 'Dice']
            if not dice_row.empty:
                jsrt_dice_1 = dice_row['JSRT Mean'].values[0]
                mont_dice_1 = dice_row['Montgomery Mean'].values[0]
                print(f"\n📊 fingerprint_summary.csv:")
                print(f"  JSRT Dice: {jsrt_dice_1:.4f}")
                print(f"  Montgomery Dice: {mont_dice_1:.4f}")

        # Check effect sizes
        effect_file = 'reports/enhanced_statistics/effect_sizes.csv'
        if self.check_file_exists(effect_file):
            effect_df = pd.read_csv(self.root / effect_file)
            dice_row = effect_df[effect_df['feature'] == 'dice']
            if not dice_row.empty:
                jsrt_dice_2 = dice_row['jsrt_mean'].values[0]
                mont_dice_2 = dice_row['mont_mean'].values[0]
                print(f"\n📊 effect_sizes.csv:")
                print(f"  JSRT Dice: {jsrt_dice_2:.4f}")
                print(f"  Montgomery Dice: {mont_dice_2:.4f}")

                # Check consistency
                if jsrt_dice_1 is not None and abs(jsrt_dice_1 - jsrt_dice_2) > 0.01:
                    diff_pct = abs((jsrt_dice_1 - jsrt_dice_2) / jsrt_dice_1 * 100)
                    self.issues.append(
                        f"🔴 CRITICAL: JSRT Dice mismatch between files "
                        f"({jsrt_dice_1:.3f} vs {jsrt_dice_2:.3f}, {diff_pct:.1f}% diff)"
                    )

                if mont_dice_1 is not None and abs(mont_dice_1 - mont_dice_2) > 0.01:
                    self.warnings.append(
                        f"⚠️  Montgomery Dice mismatch between files "
                        f"({mont_dice_1:.3f} vs {mont_dice_2:.3f})"
                    )

        # Manuscript claim
        dice_claims = self.manuscript_claims.get("dice", {})
        manuscript_jsrt_dice = dice_claims.get("jsrt", 0.0)
        manuscript_mont_dice = dice_claims.get("montgomery", 0.0)

        print(f"\n📄 Manuscript Claims:")
        print(f"  JSRT Dice: {manuscript_jsrt_dice:.3f}")
        print(f"  Montgomery Dice: {manuscript_mont_dice:.3f}")

    def verify_attribution_mass(self):
        """Verify attribution mass values."""
        print("\n" + "="*80)
        print("VERIFYING ATTRIBUTION MASS")
        print("="*80)

        jsrt_mass_1 = None
        mont_mass_1 = None

        # Check fingerprint summary
        fprint_file = 'manuscript/tables/fingerprint_summary.csv'
        if self.check_file_exists(fprint_file):
            fprint_df = pd.read_csv(self.root / fprint_file)
            mass_row = fprint_df[fprint_df['Metric'] == 'IG Mass']
            if not mass_row.empty:
                jsrt_mass_1 = mass_row['JSRT Mean'].values[0]
                mont_mass_1 = mass_row['Montgomery Mean'].values[0]
                print(f"\n📊 fingerprint_summary.csv (IG Mass):")
                print(f"  JSRT: {jsrt_mass_1:.2f}")
                print(f"  Montgomery: {mont_mass_1:.2f}")
                print(f"  Collapse: {(1 - mont_mass_1/jsrt_mass_1)*100:.1f}%")

        # Check effect sizes
        effect_file = 'reports/enhanced_statistics/effect_sizes.csv'
        if self.check_file_exists(effect_file):
            effect_df = pd.read_csv(self.root / effect_file)
            mass_row = effect_df[effect_df['feature'] == 'attribution_abs_sum']
            if not mass_row.empty:
                jsrt_mass_2 = mass_row['jsrt_mean'].values[0]
                mont_mass_2 = mass_row['mont_mean'].values[0]
                print(f"\n📊 effect_sizes.csv (attribution_abs_sum):")
                print(f"  JSRT: {jsrt_mass_2:.2f}")
                print(f"  Montgomery: {mont_mass_2:.2f}")

                # These might be different features - just flag large differences
                if jsrt_mass_1 is not None and abs(jsrt_mass_1 - jsrt_mass_2) > 5:
                    self.warnings.append(
                        f"⚠️  Large difference in attribution mass between files. "
                        f"Verify these are the same feature (IG vs Grad-CAM?)"
                    )

        # Manuscript claim
        mass_claims = self.manuscript_claims.get("ig_mass", {})
        manuscript_jsrt_mass = mass_claims.get("jsrt", 0.0)
        manuscript_mont_mass = mass_claims.get("montgomery", 0.0)

        print(f"\n📄 Manuscript Claims:")
        print(f"  JSRT: {manuscript_jsrt_mass:.2f}")
        print(f"  Montgomery: {manuscript_mont_mass:.2f}")
        print(f"  Collapse: {(1 - manuscript_mont_mass/manuscript_jsrt_mass)*100:.1f}%")

    def check_sample_sizes(self):
        """Verify sample sizes."""
        print("\n" + "="*80)
        print("VERIFYING SAMPLE SIZES")
        print("="*80)

        effect_file = 'reports/enhanced_statistics/effect_sizes.csv'
        if self.check_file_exists(effect_file):
            effect_df = pd.read_csv(self.root / effect_file)

            # Count non-null samples for Dice
            dice_row = effect_df[effect_df['feature'] == 'dice']
            if not dice_row.empty:
                # Infer sample size from reported statistics
                # This is approximate - ideally we'd have raw data
                print(f"\n📊 Reported in manuscript:")
                print(f"  JSRT: 247 samples")
                print(f"  Montgomery: 138 samples")
                print(f"  Total: 385 samples")

                # Check if this matches
                corr_file = 'reports/enhanced_statistics/coverage_dice_correlation.csv'
                if self.check_file_exists(corr_file):
                    corr_df = pd.read_csv(self.root / corr_file)
                    n_samples = corr_df['n_samples'].values[0]
                    print(f"\n📊 Actual (from coverage_dice_correlation.csv):")
                    print(f"  Total samples: {n_samples}")

                    if n_samples != 385:
                        self.warnings.append(
                            f"⚠️  Sample size mismatch: manuscript claims 385, "
                            f"found {n_samples}"
                        )

    def generate_report(self):
        """Generate final verification report."""
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)

        if not self.issues and not self.warnings:
            print("\n✅ ALL CHECKS PASSED!")
            print("\nYour manuscript data is consistent with your experimental results.")
            return 0

        if self.issues:
            print(f"\n🔴 CRITICAL ISSUES FOUND ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  {issue}")
            print("\n❌ YOU MUST FIX THESE BEFORE SUBMISSION")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
            print("\n⚠️  These should be investigated but may not be critical")

        return 1 if self.issues else 0

    def run_all_checks(self):
        """Run all verification checks."""
        print("\n" + "="*80)
        print("MANUSCRIPT DATA VERIFICATION")
        print("="*80)
        print("\nChecking consistency between experimental data and manuscript...")

        self.verify_divergence_metrics()
        self.verify_dice_scores()
        self.verify_attribution_mass()
        self.check_sample_sizes()

        return self.generate_report()


if __name__ == '__main__':
    verifier = DataVerifier()
    exit_code = verifier.run_all_checks()
    sys.exit(exit_code)
