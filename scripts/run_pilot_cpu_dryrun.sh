#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SUBSET="${1:-pilot5}"            # e.g. pilot5 or pilot10
EXPERIMENT="${2:-jsrt_baseline}" # jsrt baseline only for constrained pilot
SEED="42"
ENDPOINTS=("predicted_mask" "mask_free")

REGISTRY_PATH="${ROOT_DIR}/reports_v2/run_registry.csv"
LOG_DIR="${ROOT_DIR}/logs_v2"
mkdir -p "${LOG_DIR}"

if [[ "${EXPERIMENT}" != "jsrt_baseline" ]]; then
  echo "[ERROR] Constrained CPU dry-run only allows experiment=jsrt_baseline"
  exit 2
fi

if [[ "${SUBSET}" == "pilot10" && ! -f "${ROOT_DIR}/configs/subsets/jsrt_pilot10.txt" ]]; then
  echo "[INFO] Creating deterministic subset file configs/subsets/jsrt_pilot10.txt from jsrt/full metadata"
  python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path(".")
meta = root / "data" / "interim" / "jsrt" / "full" / "metadata.parquet"
if not meta.exists():
    raise FileNotFoundError(
        "Missing data/interim/jsrt/full/metadata.parquet. Run prepare_data for jsrt/full first."
    )
df = pd.read_parquet(meta)
ids = sorted(df["sample_id"].astype(str).tolist())[:10]
out = root / "configs" / "subsets" / "jsrt_pilot10.txt"
out.write_text("\n".join(ids) + "\n", encoding="utf-8")
print(out)
PY
fi

echo "[INFO] Preparing cache for jsrt/${SUBSET}"
python scripts/prepare_data.py --dataset jsrt --subset "${SUBSET}"

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

run_endpoint() {
  local endpoint="$1"
  local ts
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  local run_id="cpu_dryrun_jsrt_${SUBSET}_${endpoint}_seed${SEED}_${ts}"
  local log_file="${LOG_DIR}/${run_id}.log"
  local start_utc
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  append_registry_row "${run_id}" "${endpoint}" "running" "false" "${start_utc}" "" "cpu_dryrun started; log=${log_file}"

  echo "[INFO] Running endpoint=${endpoint} subset=${SUBSET} (CPU)"
  set +e
  CUDA_VISIBLE_DEVICES="" python scripts/run_fingerprint.py \
    --experiment "${EXPERIMENT}" \
    --subset "${SUBSET}" \
    --device cpu \
    --endpoint-mode "${endpoint}" >"${log_file}" 2>&1
  local rc=$?
  set -e

  local end_utc
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local output_dir="data/fingerprints/${endpoint}/${EXPERIMENT}"
  local output_hash=""
  if [[ -d "${output_dir}" ]]; then
    output_hash="$(
      {
        find "${output_dir}" -type f | LC_ALL=C sort | xargs -r sha256sum
      } | sha256sum | awk '{print $1}'
    )"
  fi

  if [[ ${rc} -eq 0 ]]; then
    append_registry_row "${run_id}" "${endpoint}" "completed" "true" "${start_utc}" "${end_utc}" "cpu_dryrun completed; output_hash=${output_hash}; log=${log_file}"
    echo "[OK] endpoint=${endpoint} completed"
  else
    append_registry_row "${run_id}" "${endpoint}" "failed" "false" "${start_utc}" "${end_utc}" "cpu_dryrun failed(rc=${rc}); output_hash=${output_hash}; log=${log_file}"
    echo "[ERROR] endpoint=${endpoint} failed (rc=${rc}); see ${log_file}"
    return ${rc}
  fi
}

for endpoint in "${ENDPOINTS[@]}"; do
  run_endpoint "${endpoint}"
done

echo "[INFO] Running post-run validation tests"
pytest -q \
  tests/test_all_metadata_have_patient_id.py \
  tests/test_gt_leakage.py \
  tests/test_smoke.py \
  tests/test_metadata.py \
  tests/test_shift_metrics.py

echo "[DONE] CPU dry-run completed for endpoints: ${ENDPOINTS[*]}"
