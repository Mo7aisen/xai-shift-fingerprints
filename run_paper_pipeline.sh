#!/bin/bash
# End-to-end pipeline for paper-ready results.
# Runs fingerprint generation, analysis, and figure synthesis.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="/home/ubuntu/xai-env/bin/activate"
LOG_DIR="${PROJECT_DIR}/logs/paper_pipeline_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOG_DIR}"

log() {
    local msg="$1"
    printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "${msg}"
}

log "Starting paper pipeline"
log "Project: ${PROJECT_DIR}"
log "Logs: ${LOG_DIR}"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    log "Activated virtualenv at ${ENV_FILE}"
else
    log "WARNING: virtualenv not found at ${ENV_FILE}; using system Python"
fi

cd "${PROJECT_DIR}"
export XFP_ALLOW_MISSING_MODELS=1
log "XFP_ALLOW_MISSING_MODELS=${XFP_ALLOW_MISSING_MODELS}"

log "GPU status snapshot:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
else
    log "nvidia-smi not found"
fi

DATASETS=(jsrt montgomery shenzhen nih_chestxray14)
for dataset in "${DATASETS[@]}"; do
    metadata_path="${PROJECT_DIR}/data/interim/${dataset}/full/metadata.parquet"
    if [[ ! -f "${metadata_path}" ]]; then
        log "Preparing dataset cache for ${dataset} (missing ${metadata_path})"
        python scripts/prepare_data.py --dataset "${dataset}" --subset full
    else
        log "Dataset cache present for ${dataset}"
    fi
done

EXPERIMENTS=(
    jsrt_baseline
    jsrt_to_montgomery
    montgomery_baseline
    montgomery_to_jsrt
    shenzhen_baseline
    jsrt_to_shenzhen
    montgomery_to_shenzhen
    jsrt_to_nih
    montgomery_to_nih
    shenzhen_to_nih
)

if [[ -f "/home/ubuntu/models/nih_unet_transfer.pt" ]]; then
    EXPERIMENTS+=(nih_baseline)
else
    log "NIH baseline model missing (/home/ubuntu/models/nih_unet_transfer.pt); skipping nih_baseline"
fi

log "Running fingerprint batch: ${EXPERIMENTS[*]}"
"${PROJECT_DIR}/scripts/run_gpu_fingerprint_batch.sh" "${EXPERIMENTS[@]}"

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

log "Generating publication figures"
python scripts/generate_publication_figures_final.py

log "Verifying manuscript data"
python scripts/verify_manuscript_data.py

log "Pipeline complete"
