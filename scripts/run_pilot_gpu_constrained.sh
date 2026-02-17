#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SUBSET="${1:-pilot5}"
EXPERIMENT="${2:-jsrt_baseline}"
SEED="42"
MAX_RUNTIME_SEC="${MAX_RUNTIME_SEC:-1800}" # 30 min hard limit
ENDPOINTS=("predicted_mask" "mask_free")
REGISTRY_PATH="${ROOT_DIR}/reports_v2/run_registry.csv"
LOG_DIR="${ROOT_DIR}/logs_v2"
mkdir -p "${LOG_DIR}"

if [[ "${EXPERIMENT}" != "jsrt_baseline" ]]; then
  echo "[ERROR] Constrained pilot allows experiment=jsrt_baseline only."
  exit 2
fi
if [[ "${SUBSET}" != "pilot5" ]]; then
  echo "[ERROR] Constrained pilot allows subset=pilot5 only."
  exit 2
fi

./scripts/preflight_gpu.sh

CONFIG_HASH="$(sha256sum configs/protocol_lock_v1.yaml | awk '{print $1}')"
COMMIT_HASH="$(git rev-parse --short=12 HEAD 2>/dev/null || echo "nogit")"
INPUT_DATA_HASH="$(
  {
    find "data/interim/jsrt/${SUBSET}" -maxdepth 1 -type f \( -name '*.npz' -o -name 'metadata.parquet' \) \
      | LC_ALL=C sort | xargs -r sha256sum
  } | sha256sum | awk '{print $1}'
)"

append_registry_row() {
  local run_id="$1"
  local endpoint="$2"
  local status="$3"
  local gate_passed="$4"
  local start_utc="$5"
  local end_utc="$6"
  local notes="$7"
  python - <<PY
from pathlib import Path
import csv

path = Path("${REGISTRY_PATH}")
row = {
    "run_id": "${run_id}",
    "endpoint": "${endpoint}",
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

GLOBAL_START_EPOCH="$(date +%s)"

run_endpoint() {
  local endpoint="$1"
  local now_epoch elapsed remaining
  now_epoch="$(date +%s)"
  elapsed=$((now_epoch - GLOBAL_START_EPOCH))
  remaining=$((MAX_RUNTIME_SEC - elapsed))
  if (( remaining <= 0 )); then
    echo "[ERROR] Runtime budget exceeded before endpoint=${endpoint}"
    return 124
  fi

  local ts run_id log_file start_utc
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  run_id="gpu_pilot_jsrt_${SUBSET}_${endpoint}_seed${SEED}_${ts}"
  log_file="${LOG_DIR}/${run_id}.log"
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  append_registry_row "${run_id}" "${endpoint}" "running" "false" "${start_utc}" "" "gpu_pilot started; budget_sec=${remaining}; log=${log_file}"

  echo "[INFO] Running endpoint=${endpoint} on CUDA (remaining_budget_sec=${remaining})"
  set +e
  timeout "${remaining}" python scripts/run_fingerprint.py \
    --experiment "${EXPERIMENT}" \
    --subset "${SUBSET}" \
    --device cuda \
    --endpoint-mode "${endpoint}" >"${log_file}" 2>&1
  local rc=$?
  set -e

  local end_utc output_dir output_hash
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  output_dir="data/fingerprints/${endpoint}/${EXPERIMENT}"
  output_hash=""
  if [[ -d "${output_dir}" ]]; then
    output_hash="$(
      {
        find "${output_dir}" -type f | LC_ALL=C sort | xargs -r sha256sum
      } | sha256sum | awk '{print $1}'
    )"
  fi

  if [[ ${rc} -eq 0 ]]; then
    append_registry_row "${run_id}" "${endpoint}" "completed" "true" "${start_utc}" "${end_utc}" "gpu_pilot completed; output_hash=${output_hash}; log=${log_file}"
    echo "[OK] endpoint=${endpoint} completed"
    return 0
  fi

  if [[ ${rc} -eq 124 ]]; then
    append_registry_row "${run_id}" "${endpoint}" "aborted_timeout" "false" "${start_utc}" "${end_utc}" "gpu_pilot timeout; output_hash=${output_hash}; log=${log_file}"
    echo "[ERROR] endpoint=${endpoint} aborted due timeout"
    return 124
  fi

  append_registry_row "${run_id}" "${endpoint}" "failed" "false" "${start_utc}" "${end_utc}" "gpu_pilot failed(rc=${rc}); output_hash=${output_hash}; log=${log_file}"
  echo "[ERROR] endpoint=${endpoint} failed (rc=${rc})"
  return ${rc}
}

for endpoint in "${ENDPOINTS[@]}"; do
  run_endpoint "${endpoint}"
done

env \
  -u XFP_DATASETS_ROOT \
  -u XFP_MODELS_ROOT \
  -u XFP_FINGERPRINTS_ROOT \
  -u XFP_CACHE_ROOT \
  -u XFP_MODEL_UNET_JSRT_FULL \
  -u XFP_MODEL_UNET_MONTGOMERY_FULL \
  -u XFP_MODEL_UNET_SHENZHEN_FULL \
  -u XFP_MODEL_UNET_NIH_FULL \
  pytest -q \
    tests/test_all_metadata_have_patient_id.py \
    tests/test_gt_leakage.py \
    tests/test_smoke.py \
    tests/test_metadata.py \
    tests/test_shift_metrics.py

echo "[DONE] Constrained GPU pilot completed."
