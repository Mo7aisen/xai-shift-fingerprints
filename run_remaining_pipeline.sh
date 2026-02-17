#!/bin/bash
# Complete the fingerprint pipeline - Run remaining analysis steps
# Created: 2025-11-29
#
# This script runs the remaining pipeline steps after fingerprint extraction:
# 1. Bootstrap divergence analysis
# 2. Statistical hypothesis tests
# 3. Publication figure generation

set -euo pipefail

cd /home/ubuntu/xai_shift_fingerprints_code

echo "=========================================================================="
echo "FINGERPRINT PIPELINE - REMAINING STEPS"
echo "=========================================================================="
echo "Start time: $(date)"
echo ""

# Activate environment if needed
if [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "✓ Conda environment active: $CONDA_DEFAULT_ENV"
else
    echo "⚠️  No conda environment detected"
    echo "   If needed, run: conda activate xai-shift"
fi
echo ""

# Check if we're in the right directory
if [ ! -f "scripts/bootstrap_divergence.py" ]; then
    echo "❌ Error: Not in xai_shift_fingerprints directory"
    exit 1
fi

echo "=========================================================================="
echo "STEP 1: Bootstrap Divergence Analysis"
echo "=========================================================================="
echo "Computing divergence with bootstrap confidence intervals..."
echo "Parameters: seed=2025, n-resamples=1000"
echo ""

python scripts/bootstrap_divergence.py --seed 2025 --n-resamples 1000

if [ $? -eq 0 ]; then
    echo "✅ Bootstrap divergence complete"
else
    echo "❌ Bootstrap divergence failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STEP 2: Statistical Hypothesis Tests"
echo "=========================================================================="
echo "Running statistical significance tests..."
echo ""

python scripts/statistical_hypothesis_tests.py

if [ $? -eq 0 ]; then
    echo "✅ Statistical tests complete"
else
    echo "❌ Statistical tests failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STEP 3: Update Shenzhen vs NIH Summary"
echo "=========================================================================="
echo "Updating analysis notes with Shenzhen vs NIH summary..."
echo ""

python scripts/update_shenzhen_nih_summary.py

if [ $? -eq 0 ]; then
    echo "✅ Shenzhen vs NIH summary updated"
else
    echo "❌ Shenzhen vs NIH summary update failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STEP 4: Baseline Shift Detection"
echo "=========================================================================="
echo "Computing baseline shift detection comparisons..."
echo ""

python scripts/baseline_shift_detection.py

if [ $? -eq 0 ]; then
    echo "✅ Baseline shift detection complete"
else
    echo "❌ Baseline shift detection failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STEP 5: Feature Importance + PCA"
echo "=========================================================================="
echo "Generating feature importance and PCA summaries..."
echo ""

python scripts/feature_importance_pca.py

if [ $? -eq 0 ]; then
    echo "✅ Feature importance + PCA complete"
else
    echo "❌ Feature importance + PCA failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STEP 6: Error Correlation Analysis"
echo "=========================================================================="
echo "Analyzing attribution mass vs Dice error correlations..."
echo ""

python scripts/error_correlation_analysis.py

if [ $? -eq 0 ]; then
    echo "✅ Error correlation analysis complete"
else
    echo "❌ Error correlation analysis failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STEP 7: Enhanced Statistical Analysis"
echo "=========================================================================="
echo "Running enhanced statistical summaries..."
echo ""

python scripts/enhanced_statistical_analysis.py

if [ $? -eq 0 ]; then
    echo "✅ Enhanced statistical analysis complete"
else
    echo "❌ Enhanced statistical analysis failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STEP 8: Fix Data Inconsistencies + Summary Tables"
echo "=========================================================================="
echo "Reconciling metrics tables..."
echo ""

python scripts/fix_data_inconsistencies.py

if [ $? -eq 0 ]; then
    echo "✅ Data reconciliation complete"
else
    echo "❌ Data reconciliation failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STEP 9: Divergence Summary Report"
echo "=========================================================================="
if [ -f "reports/divergence/divergence_comparison_table.csv" ] && [ -f "reports/divergence/detailed_metrics_comparison.csv" ]; then
    echo "Generating divergence summary report..."
    python scripts/analyze_divergence.py \
        --divergence-table reports/divergence/divergence_comparison_table.csv \
        --detailed-table reports/divergence/detailed_metrics_comparison.csv \
        --uncertainty-table reports/divergence/divergence_uncertainty.csv \
        --output-dir reports/divergence

    if [ $? -eq 0 ]; then
        echo "✅ Divergence summary report complete"
    else
        echo "❌ Divergence summary report failed"
        exit 1
    fi
else
    echo "⚠️  Skipping divergence summary report; required tables missing"
fi

echo ""
echo "=========================================================================="
echo "STEP 10: Generate Publication Figures"
echo "=========================================================================="
echo "Creating publication-ready figures..."
echo ""

python scripts/generate_publication_figures_final.py

if [ $? -eq 0 ]; then
    echo "✅ Publication figures complete"
else
    echo "❌ Figure generation failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STEP 11: NIH External Validation Figure"
echo "=========================================================================="
echo "Regenerating NIH external validation figure..."
echo ""

python scripts/plot_nih_external_validation.py

if [ $? -eq 0 ]; then
    echo "✅ NIH external validation figure complete"
else
    echo "❌ NIH external validation figure failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STEP 12: Verify Manuscript Data"
echo "=========================================================================="
echo "Running manuscript data checks..."
echo ""

python scripts/verify_manuscript_data.py

if [ $? -eq 0 ]; then
    echo "✅ Manuscript data verification complete"
else
    echo "❌ Manuscript data verification failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STEP 13: Rebuild Journal Submission Bundle"
echo "=========================================================================="
echo "Refreshing bundle contents and rebuilding zip..."
echo ""

bash scripts/rebuild_journal_submission_bundle.sh

if [ $? -eq 0 ]; then
    echo "✅ Journal submission bundle rebuilt"
else
    echo "❌ Journal submission bundle rebuild failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "PIPELINE COMPLETE"
echo "=========================================================================="
echo "End time: $(date)"
echo ""
echo "Output locations:"
echo "  - Bootstrap results: results/*bootstrap*.npz"
echo "  - Statistical tests: results/*statistical*.json"
echo "  - Publication figures: results/figures/*.png"
echo ""
echo "✅ All steps completed successfully!"
echo "=========================================================================="
