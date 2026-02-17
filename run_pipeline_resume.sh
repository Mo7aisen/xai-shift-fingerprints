#!/bin/bash
# Resume the full pipeline after an interrupted run.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="/home/ubuntu/xai-env/bin/activate"

log() {
    local msg="$1"
    printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "${msg}"
}

log "Starting resume pipeline"
log "Project: ${PROJECT_DIR}"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    log "Activated virtualenv at ${ENV_FILE}"
else
    log "WARNING: virtualenv not found at ${ENV_FILE}; using system Python"
fi

export XFP_ALLOW_MISSING_MODELS=1
log "XFP_ALLOW_MISSING_MODELS=${XFP_ALLOW_MISSING_MODELS}"

cd "${PROJECT_DIR}"

log "Running remaining GPU fingerprints (NIH-related)"
"${PROJECT_DIR}/scripts/run_gpu_fingerprint_batch.sh" \
    jsrt_to_nih \
    montgomery_to_nih \
    shenzhen_to_nih \
    nih_baseline

log "Running bootstrap divergence"
python scripts/bootstrap_divergence.py --seed 2025 --n-resamples 1000

log "Running statistical hypothesis tests"
python scripts/statistical_hypothesis_tests.py

log "Running baseline shift detection"
python scripts/baseline_shift_detection.py

log "Running feature importance + PCA"
python scripts/feature_importance_pca.py

log "Running error correlation analysis"
python scripts/error_correlation_analysis.py

log "Running enhanced statistical analysis"
python scripts/enhanced_statistical_analysis.py

log "Fixing data inconsistencies + regenerating summary tables"
python scripts/fix_data_inconsistencies.py

if [[ -f "reports/divergence/divergence_comparison_table.csv" && -f "reports/divergence/detailed_metrics_comparison.csv" ]]; then
    log "Generating divergence summary report"
    python scripts/analyze_divergence.py \
        --divergence-table reports/divergence/divergence_comparison_table.csv \
        --detailed-table reports/divergence/detailed_metrics_comparison.csv \
        --uncertainty-table reports/divergence/divergence_uncertainty.csv \
        --output-dir reports/divergence
else
    log "Skipping divergence summary report; required tables missing"
fi

log "Running robustness perturbations (GPU)"
python scripts/robustness_perturbations.py --subset-size 200 --device cuda

log "Plotting robustness figures"
python scripts/plot_robustness_figures.py

log "Generating analysis notes"
python scripts/generate_analysis_notes.py --alpha 0.05 --robust-alpha 0.01

log "Generating publication figures"
python scripts/generate_publication_figures_final.py

log "Verifying manuscript data"
python scripts/verify_manuscript_data.py

log "Rebuilding journal submission bundle"
./scripts/rebuild_journal_submission_bundle.sh

log "Rebuilding manuscript PDF"
if command -v latexmk >/dev/null 2>&1; then
    (cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex)
else
    log "latexmk not found; skipping PDF build"
fi

log "Resume pipeline complete"
