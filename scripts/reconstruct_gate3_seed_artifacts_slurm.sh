#!/usr/bin/env bash
#SBATCH --job-name=xfp_gate3_reconstruct
#SBATCH --partition=batch_gpu_g2.xlarge_16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=02:30:00
#SBATCH --output=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_gate3_reconstruct_%j.out
#SBATCH --error=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_gate3_reconstruct_%j.err

set -euo pipefail

PROJECT_ROOT="/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full"
ENV_ACTIVATE="/storage/xai_cxr_safety/xai-al-env/bin/activate"
BATCH_TAG="gate3_official_20260217"
EXPERIMENT="jsrt_to_montgomery"
SUBSET="full"
SEEDS=(42 43 44 45 46)
ENDPOINTS=(predicted_mask mask_free)
AUDIT_DIR="${PROJECT_ROOT}/reports_v2/audits"
ARTIFACT_ROOT="${PROJECT_ROOT}/reports_v2/gate3_seed_artifacts"
MATCH_CSV="${AUDIT_DIR}/GATE3_RECONSTRUCT_HASH_MATCH_${SLURM_JOB_ID:-nojob}.csv"
STRICT_HASH_MATCH="${STRICT_HASH_MATCH:-0}"

mkdir -p "${AUDIT_DIR}" "${ARTIFACT_ROOT}" "${PROJECT_ROOT}/logs"

cd "${PROJECT_ROOT}"

if [[ ! -f "${ENV_ACTIVATE}" ]]; then
  echo "[ERROR] Missing virtualenv: ${ENV_ACTIVATE}"
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_ACTIVATE}"

export XFP_DATASETS_ROOT="/storage/xai_cxr_safety/Datasets"
export XFP_MODELS_ROOT="/storage/xai_cxr_safety/models"
export XFP_CACHE_ROOT="${PROJECT_ROOT}/data/interim"
export XFP_MODEL_UNET_JSRT_FULL="/storage/xai_cxr_safety/models/jsrt_unet_baseline.pt"
export XFP_MODEL_UNET_MONTGOMERY_FULL="/storage/xai_cxr_safety/models/montgomery_unet_baseline.pt"
export XFP_MODEL_UNET_SHENZHEN_FULL="/storage/xai_cxr_safety/models/shenzhen_unet_baseline.pt"
export XFP_MODEL_UNET_NIH_FULL="/storage/xai_cxr_safety/models/nih_unet_transfer.pt"

echo "seed,endpoint,expected_hash,actual_hash,match" > "${MATCH_CSV}"

extract_expected_hash() {
  local seed="$1"
  local endpoint="$2"
  python - <<PY
from pathlib import Path
import csv
import re

seed = ${seed}
endpoint = "${endpoint}"
batch_tag = "${BATCH_TAG}"
path = Path("reports_v2/run_registry.csv")
fieldnames = [
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
]
expected = ""
with path.open("r", encoding="utf-8", newline="") as fh:
    reader = csv.DictReader(fh, fieldnames=fieldnames)
    for row in reader:
        run_id = row["run_id"] or ""
        notes = row["notes"] or ""
        if f"gpu_pilot_{batch_tag}" not in run_id:
            continue
        if row["endpoint"] != endpoint:
            continue
        if int(row["seed"]) != seed:
            continue
        if row["status"] != "completed":
            continue
        m = re.search(r"output_hash=([0-9a-f]{64})", notes)
        if m:
            expected = m.group(1)
print(expected)
PY
}

compute_output_hash() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo ""
    return 0
  fi
  find "${dir}" -type f | LC_ALL=C sort | xargs -r sha256sum | sha256sum | awk '{print $1}'
}

for seed in "${SEEDS[@]}"; do
  for endpoint in "${ENDPOINTS[@]}"; do
    expected_hash="$(extract_expected_hash "${seed}" "${endpoint}")"
    if [[ -z "${expected_hash}" ]]; then
      echo "[ERROR] expected hash not found for seed=${seed}, endpoint=${endpoint}"
      exit 1
    fi

    export XFP_FINGERPRINTS_ROOT="${ARTIFACT_ROOT}/seed${seed}"
    out_dir="${XFP_FINGERPRINTS_ROOT}/${endpoint}/${EXPERIMENT}"
    rm -rf "${out_dir}"

    echo "[INFO] Running seed=${seed} endpoint=${endpoint}"
    python scripts/run_fingerprint.py \
      --experiment "${EXPERIMENT}" \
      --subset "${SUBSET}" \
      --device cuda \
      --endpoint-mode "${endpoint}" \
      --seed "${seed}" \
      --skip-validate

    actual_hash="$(compute_output_hash "${out_dir}")"
    match="false"
    if [[ "${actual_hash}" == "${expected_hash}" ]]; then
      match="true"
    fi
    echo "${seed},${endpoint},${expected_hash},${actual_hash},${match}" >> "${MATCH_CSV}"
    echo "[INFO] hash_check seed=${seed} endpoint=${endpoint} match=${match}"
    if [[ "${match}" != "true" && "${STRICT_HASH_MATCH}" == "1" ]]; then
      echo "[ERROR] hash mismatch for seed=${seed}, endpoint=${endpoint}"
      exit 1
    fi
    if [[ "${match}" != "true" ]]; then
      echo "[WARN] hash mismatch for seed=${seed}, endpoint=${endpoint}; continuing (STRICT_HASH_MATCH=${STRICT_HASH_MATCH})"
    fi
  done
done

python scripts/gate3_full_seeds_analysis.py \
  --artifacts-root "${ARTIFACT_ROOT}" \
  --experiment "${EXPERIMENT}" \
  --seeds 42 43 44 45 46 \
  --endpoints predicted_mask mask_free \
  --n-boot 1000 \
  --top-k 10 \
  --out-csv "${AUDIT_DIR}/GATE3_FULL_SEEDS_STATS.csv" \
  --out-json "${AUDIT_DIR}/GATE3_FULL_SEEDS_SUMMARY.json" \
  --out-md "${AUDIT_DIR}/GATE3_FULL_SEEDS_STATS_2026-02-17.md"

echo "[DONE] Gate-3 reconstruction + statistical analysis completed."
echo "[DONE] Hash match table: ${MATCH_CSV}"
