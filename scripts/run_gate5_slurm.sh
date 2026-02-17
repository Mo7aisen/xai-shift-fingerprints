#!/usr/bin/env bash
#SBATCH --job-name=xfp_gate5
#SBATCH --partition=batch_gpu_g2.xlarge_16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_gate5_%j.out
#SBATCH --error=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_gate5_%j.err

set -euo pipefail

PROJECT_ROOT="/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full"
ENV_ACTIVATE="/storage/xai_cxr_safety/xai-al-env/bin/activate"
AUDIT_DIR="${PROJECT_ROOT}/reports_v2/audits"
GATE5_ROOT="${PROJECT_ROOT}/reports_v2/gate5"

EXPERIMENT="${XFP_EXPERIMENT:-jsrt_to_montgomery}"
SUBSET="${XFP_SUBSET:-full}"
SEEDS=( ${XFP_SEEDS:-42 43 44 45 46} )
REF_SEED="${XFP_GATE5_REF_SEED:-42}"
DET_SEED="${XFP_GATE5_DET_SEED:-42}"
DATE_TAG="${XFP_DATE_TAG:-2026-02-17}"
OUTPUT_TAG="${XFP_OUTPUT_TAG:-}"

if [[ -n "${OUTPUT_TAG}" ]]; then
  SAFE_OUTPUT_TAG="$(echo "${OUTPUT_TAG}" | tr '/: ' '___')"
  OUT_SEED_CSV="${AUDIT_DIR}/GATE5_CLINICAL_PER_SEED_${SAFE_OUTPUT_TAG}.csv"
  OUT_CASES_CSV="${AUDIT_DIR}/GATE5_CLINICAL_CASES_${SAFE_OUTPUT_TAG}.csv"
  OUT_CLINICAL_JSON="${AUDIT_DIR}/GATE5_CLINICAL_SUMMARY_${SAFE_OUTPUT_TAG}.json"
  OUT_CLINICAL_MD="${AUDIT_DIR}/GATE5_CLINICAL_RELEVANCE_${SAFE_OUTPUT_TAG}.md"
  OUT_DET_CSV="${AUDIT_DIR}/GATE5_BITWISE_DETERMINISM_${SAFE_OUTPUT_TAG}.csv"
  OUT_DET_JSON="${AUDIT_DIR}/GATE5_BITWISE_DETERMINISM_SUMMARY_${SAFE_OUTPUT_TAG}.json"
  OUT_DET_MD="${AUDIT_DIR}/GATE5_BITWISE_DETERMINISM_${SAFE_OUTPUT_TAG}.md"
  OUT_FINAL_JSON="${AUDIT_DIR}/GATE5_FINAL_SUMMARY_${SAFE_OUTPUT_TAG}.json"
  OUT_FINAL_MD="${AUDIT_DIR}/GATE5_FINAL_${SAFE_OUTPUT_TAG}.md"
else
  OUT_SEED_CSV="${AUDIT_DIR}/GATE5_CLINICAL_PER_SEED.csv"
  OUT_CASES_CSV="${AUDIT_DIR}/GATE5_CLINICAL_CASES.csv"
  OUT_CLINICAL_JSON="${AUDIT_DIR}/GATE5_CLINICAL_SUMMARY.json"
  OUT_CLINICAL_MD="${AUDIT_DIR}/GATE5_CLINICAL_RELEVANCE_${DATE_TAG}.md"
  OUT_DET_CSV="${AUDIT_DIR}/GATE5_BITWISE_DETERMINISM.csv"
  OUT_DET_JSON="${AUDIT_DIR}/GATE5_BITWISE_DETERMINISM_SUMMARY.json"
  OUT_DET_MD="${AUDIT_DIR}/GATE5_BITWISE_DETERMINISM_${DATE_TAG}.md"
  OUT_FINAL_JSON="${AUDIT_DIR}/GATE5_FINAL_SUMMARY.json"
  OUT_FINAL_MD="${AUDIT_DIR}/GATE5_FINAL_${DATE_TAG}.md"
fi

mkdir -p "${PROJECT_ROOT}/logs" "${AUDIT_DIR}" "${GATE5_ROOT}"
cd "${PROJECT_ROOT}"

if [[ ! -f "${ENV_ACTIVATE}" ]]; then
  echo "[ERROR] Missing virtualenv: ${ENV_ACTIVATE}"
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_ACTIVATE}"

# Cluster-path config overrides
export XFP_DATASETS_ROOT="/storage/xai_cxr_safety/Datasets"
export XFP_MODELS_ROOT="/storage/xai_cxr_safety/models"
export XFP_CACHE_ROOT="${PROJECT_ROOT}/data/interim"
export XFP_MODEL_UNET_JSRT_FULL="/storage/xai_cxr_safety/models/jsrt_unet_baseline.pt"
export XFP_MODEL_UNET_MONTGOMERY_FULL="/storage/xai_cxr_safety/models/montgomery_unet_baseline.pt"
export XFP_MODEL_UNET_SHENZHEN_FULL="/storage/xai_cxr_safety/models/shenzhen_unet_baseline.pt"
export XFP_MODEL_UNET_NIH_FULL="/storage/xai_cxr_safety/models/nih_unet_transfer.pt"

# Determinism hardening
export XFP_DETERMINISTIC=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

read -r ID_DATASET OOD_DATASET < <(
  python - <<PY
import yaml
exp = "${EXPERIMENT}"
with open("configs/experiments.yaml", "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
item = cfg["experiments"][exp]
print(item["train_dataset"], item["test_datasets"][0])
PY
)

echo "[GATE5] experiment=${EXPERIMENT} subset=${SUBSET} seeds=${SEEDS[*]}"
echo "[GATE5] datasets id=${ID_DATASET} ood=${OOD_DATASET}"
echo "[GATE5] outputs: ${OUT_CLINICAL_JSON} | ${OUT_DET_JSON}"

# Step 1: generate analysis-only upper-bound dice reference (GT masks)
REF_ROOT="${GATE5_ROOT}/reference_upper_bound/seed${REF_SEED}"
export XFP_FINGERPRINTS_ROOT="${REF_ROOT}"

echo "[GATE5] building upper_bound_gt reference (seed=${REF_SEED})"
python scripts/run_fingerprint.py \
  --experiment "${EXPERIMENT}" \
  --subset "${SUBSET}" \
  --device cuda \
  --endpoint-mode upper_bound_gt \
  --seed "${REF_SEED}" \
  --deterministic \
  --skip-validate

DICE_REF_DIR="${REF_ROOT}/${EXPERIMENT}"

# Step 2: clinical relevance analysis over official Gate-3 seed artifacts
echo "[GATE5] compiling clinical relevance report"
python scripts/gate5_clinical_relevance.py \
  --artifacts-root "${PROJECT_ROOT}/reports_v2/gate3_seed_artifacts" \
  --dice-reference-dir "${DICE_REF_DIR}" \
  --experiment "${EXPERIMENT}" \
  --id-dataset "${ID_DATASET}" \
  --ood-dataset "${OOD_DATASET}" \
  --seeds "${SEEDS[@]}" \
  --endpoints predicted_mask mask_free \
  --min-corr 0.60 \
  --out-seed-csv "${OUT_SEED_CSV}" \
  --out-cases-csv "${OUT_CASES_CSV}" \
  --out-json "${OUT_CLINICAL_JSON}" \
  --out-md "${OUT_CLINICAL_MD}"

# Step 3: determinism re-audit: same config run twice and compare hashes
DET_ROOT="${GATE5_ROOT}/determinism"
for replay in run1 run2; do
  export XFP_FINGERPRINTS_ROOT="${DET_ROOT}/${replay}"
  for endpoint in predicted_mask mask_free; do
    echo "[GATE5] determinism replay=${replay} endpoint=${endpoint} seed=${DET_SEED}"
    python scripts/run_fingerprint.py \
      --experiment "${EXPERIMENT}" \
      --subset "${SUBSET}" \
      --device cuda \
      --endpoint-mode "${endpoint}" \
      --seed "${DET_SEED}" \
      --deterministic \
      --skip-validate
  done
done

python scripts/audit_bitwise_determinism.py \
  --root "${DET_ROOT}" \
  --run-a run1 \
  --run-b run2 \
  --experiment "${EXPERIMENT}" \
  --endpoints predicted_mask mask_free \
  --out-csv "${OUT_DET_CSV}" \
  --out-json "${OUT_DET_JSON}" \
  --out-md "${OUT_DET_MD}"

python scripts/gate5_finalize_decision.py \
  --clinical-json "${OUT_CLINICAL_JSON}" \
  --determinism-json "${OUT_DET_JSON}" \
  --out-json "${OUT_FINAL_JSON}" \
  --out-md "${OUT_FINAL_MD}"

echo "[DONE] Gate-5 clinical relevance + determinism re-audit complete."
echo "[DONE] final decision -> ${OUT_FINAL_JSON}"
