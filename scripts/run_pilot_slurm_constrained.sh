#!/usr/bin/env bash
#SBATCH --job-name=xfp_pilot_cstr
#SBATCH --partition=batch_gpu_g2.large_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_pilot_%j.out
#SBATCH --error=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_pilot_%j.err

set -euo pipefail

PROJECT_ROOT="/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full"
ENV_ACTIVATE="/storage/xai_cxr_safety/xai-al-env/bin/activate"
REGISTRY_PATH="${PROJECT_ROOT}/reports_v2/run_registry.csv"
AUDIT_DIR="${PROJECT_ROOT}/reports_v2/audits"
SEED="${XFP_SEED:-42}"
EXPERIMENT="${XFP_EXPERIMENT:-jsrt_baseline}"
SUBSET="${XFP_SUBSET:-pilot5}"
RUN_ID="slurm_pilot_seed${SEED}_jsrt_${SUBSET}_${SLURM_JOB_ID:-nojob}"
ENDPOINT_DESC="predicted_mask+mask_free"

mkdir -p "${AUDIT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

append_registry_row() {
  local status="$1"
  local gate_passed="$2"
  local start_utc="$3"
  local end_utc="$4"
  local notes="$5"

  python - <<PY
from pathlib import Path
import csv

path = Path("${REGISTRY_PATH}")
row = {
    "run_id": "${RUN_ID}",
    "endpoint": "${ENDPOINT_DESC}",
    "seed": "${SEED}",
    "commit_hash": "${COMMIT_HASH}",
    "config_hash": "${CONFIG_HASH}",
    "input_data_hash": "${INPUT_DATA_HASH}",
    "status": "${status}",
    "gate_passed": "${gate_passed}",
    "start_utc": "${start_utc}",
    "end_utc": "${end_utc}",
    "notes": "${notes}",
}
with path.open("a", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(
        fh,
        fieldnames=[
            "run_id",
            "endpoint",
            "seed",
            "commit_hash",
            "config_hash",
            "input_data_hash",
            "status",
            "gate_passed",
            "start_utc",
            "end_utc",
            "notes",
        ],
    )
    writer.writerow(row)
PY
}

incident_report() {
  local rc="$1"
  local start_utc="$2"
  local end_utc="$3"
  local note="$4"
  local report="${AUDIT_DIR}/PILOT_SLURM_INCIDENT_${SLURM_JOB_ID:-nojob}.md"
  cat > "${report}" <<MD
# Slurm Pilot Incident Report

Date: ${end_utc}
Job ID: ${SLURM_JOB_ID:-unknown}
Run ID: ${RUN_ID}
Status: FAILED
Exit code: ${rc}

## Context
- Experiment: ${EXPERIMENT}
- Subset: ${SUBSET}
- Seed: ${SEED}
- Endpoints: ${ENDPOINT_DESC}
- Time limit: 00:30:00

## Note
${note}

## Timing
- Start: ${start_utc}
- End: ${end_utc}
MD
}

cd "${PROJECT_ROOT}"

# Cluster-path config overrides
export XFP_DATASETS_ROOT="/storage/xai_cxr_safety/Datasets"
export XFP_MODELS_ROOT="/storage/xai_cxr_safety/models"
export XFP_FINGERPRINTS_ROOT="${PROJECT_ROOT}/data/fingerprints"
export XFP_CACHE_ROOT="${PROJECT_ROOT}/data/interim"
export XFP_MODEL_UNET_JSRT_FULL="/storage/xai_cxr_safety/models/jsrt_unet_baseline.pt"
export XFP_MODEL_UNET_MONTGOMERY_FULL="/storage/xai_cxr_safety/models/montgomery_unet_baseline.pt"
export XFP_MODEL_UNET_SHENZHEN_FULL="/storage/xai_cxr_safety/models/shenzhen_unet_baseline.pt"
export XFP_MODEL_UNET_NIH_FULL="/storage/xai_cxr_safety/models/nih_unet_transfer.pt"

if [[ ! -f "${ENV_ACTIVATE}" ]]; then
  echo "[ERROR] Missing virtualenv: ${ENV_ACTIVATE}"
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_ACTIVATE}"

if [[ ! -f "data/interim/jsrt/${SUBSET}/metadata.parquet" ]]; then
  echo "[ERROR] Missing subset metadata: data/interim/jsrt/${SUBSET}/metadata.parquet"
  exit 1
fi

COMMIT_HASH="$(git rev-parse --short=12 HEAD 2>/dev/null || echo 'nogit')"
CONFIG_HASH="$(sha256sum configs/protocol_lock_v1.yaml | awk '{print $1}')"
INPUT_DATA_HASH="$({ find data/interim/jsrt/pilot5 -maxdepth 1 -type f \( -name '*.npz' -o -name 'metadata.parquet' \) | LC_ALL=C sort | xargs -r sha256sum; } | sha256sum | awk '{print $1}')"

START_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
FINAL_STATUS="failed"
FINAL_NOTE="Unhandled error"

cleanup() {
  local rc=$?
  local end_utc
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ "${FINAL_STATUS}" == "completed" ]]; then
    append_registry_row "completed" "true" "${START_UTC}" "${end_utc}" "${FINAL_NOTE}"
    log "Pilot completed successfully"
  else
    append_registry_row "failed" "false" "${START_UTC}" "${end_utc}" "${FINAL_NOTE}"
    incident_report "${rc}" "${START_UTC}" "${end_utc}" "${FINAL_NOTE}"
    log "Pilot failed (rc=${rc})"
  fi
}
trap cleanup EXIT

append_registry_row "running" "false" "${START_UTC}" "" "slurm pilot started on job_id=${SLURM_JOB_ID:-unknown}"

log "Starting constrained Slurm pilot"
log "Run ID: ${RUN_ID}"
log "Commit: ${COMMIT_HASH}"
log "Config hash: ${CONFIG_HASH}"
log "Input hash: ${INPUT_DATA_HASH}"

# Strict preflight constraints
: "${PRECHECK_MAX_VRAM_MIB:=8000}"
export PRECHECK_MAX_VRAM_MIB
./scripts/preflight_gpu.sh

# Keep execution strictly under Slurm walltime with headroom for finalization
export MAX_RUNTIME_SEC="${MAX_RUNTIME_SEC:-1740}"
./scripts/run_pilot_gpu_constrained.sh "${SUBSET}" "${EXPERIMENT}"

FINAL_STATUS="completed"
FINAL_NOTE="slurm constrained pilot completed; preflight pass + post-tests pass"
