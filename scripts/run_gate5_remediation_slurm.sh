#!/usr/bin/env bash
#SBATCH --job-name=xfp_gate5_fix
#SBATCH --partition=batch_gpu_g2.large_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_gate5_fix_%j.out
#SBATCH --error=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_gate5_fix_%j.err

set -euo pipefail

PROJECT_ROOT="/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full"
ENV_ACTIVATE="/storage/xai_cxr_safety/xai-al-env/bin/activate"
AUDIT_DIR="${PROJECT_ROOT}/reports_v2/audits"
EXPERIMENT="${XFP_EXPERIMENT:-jsrt_to_montgomery}"
SEEDS=( ${XFP_SEEDS:-42 43 44 45 46} )
DATE_TAG="${XFP_DATE_TAG:-2026-02-17}"
OUTPUT_TAG="${XFP_OUTPUT_TAG:-}"

if [[ -n "${OUTPUT_TAG}" ]]; then
  SAFE_OUTPUT_TAG="$(echo "${OUTPUT_TAG}" | tr '/: ' '___')"
  OUT_SEED_CSV="${AUDIT_DIR}/GATE5_CLINICAL_PER_SEED_REMEDIATED_${SAFE_OUTPUT_TAG}.csv"
  OUT_CASES_CSV="${AUDIT_DIR}/GATE5_CLINICAL_CASES_REMEDIATED_${SAFE_OUTPUT_TAG}.csv"
  OUT_CLINICAL_JSON="${AUDIT_DIR}/GATE5_CLINICAL_SUMMARY_REMEDIATED_${SAFE_OUTPUT_TAG}.json"
  OUT_CLINICAL_MD="${AUDIT_DIR}/GATE5_CLINICAL_RELEVANCE_REMEDIATED_${SAFE_OUTPUT_TAG}.md"
  OUT_FINAL_JSON="${AUDIT_DIR}/GATE5_FINAL_SUMMARY_REMEDIATED_${SAFE_OUTPUT_TAG}.json"
  OUT_FINAL_MD="${AUDIT_DIR}/GATE5_FINAL_REMEDIATED_${SAFE_OUTPUT_TAG}.md"
else
  OUT_SEED_CSV="${AUDIT_DIR}/GATE5_CLINICAL_PER_SEED_REMEDIATED.csv"
  OUT_CASES_CSV="${AUDIT_DIR}/GATE5_CLINICAL_CASES_REMEDIATED.csv"
  OUT_CLINICAL_JSON="${AUDIT_DIR}/GATE5_CLINICAL_SUMMARY_REMEDIATED.json"
  OUT_CLINICAL_MD="${AUDIT_DIR}/GATE5_CLINICAL_RELEVANCE_REMEDIATED_${DATE_TAG}.md"
  OUT_FINAL_JSON="${AUDIT_DIR}/GATE5_FINAL_SUMMARY_REMEDIATED.json"
  OUT_FINAL_MD="${AUDIT_DIR}/GATE5_FINAL_REMEDIATED_${DATE_TAG}.md"
fi

mkdir -p "${AUDIT_DIR}" "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

if [[ ! -f "${ENV_ACTIVATE}" ]]; then
  echo "[ERROR] Missing virtualenv: ${ENV_ACTIVATE}"
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_ACTIVATE}"

ARTIFACTS_ROOT="${PROJECT_ROOT}/reports_v2/gate3_seed_artifacts"
DICE_REF_DIR="${PROJECT_ROOT}/reports_v2/gate5/reference_upper_bound/seed42/${EXPERIMENT}"
if [[ -n "${OUTPUT_TAG}" ]]; then
  DETERMINISM_JSON="${AUDIT_DIR}/GATE5_BITWISE_DETERMINISM_SUMMARY_${SAFE_OUTPUT_TAG}.json"
else
  DETERMINISM_JSON="${AUDIT_DIR}/GATE5_BITWISE_DETERMINISM_SUMMARY.json"
fi

if [[ ! -d "${ARTIFACTS_ROOT}" ]]; then
  echo "[ERROR] Missing artifacts root: ${ARTIFACTS_ROOT}"
  exit 1
fi
if [[ ! -d "${DICE_REF_DIR}" ]]; then
  echo "[ERROR] Missing dice reference from previous Gate-5: ${DICE_REF_DIR}"
  exit 1
fi
if [[ ! -f "${DETERMINISM_JSON}" ]]; then
  echo "[ERROR] Missing determinism summary: ${DETERMINISM_JSON}"
  exit 1
fi

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

echo "[GATE5-FIX] experiment=${EXPERIMENT} seeds=${SEEDS[*]}"
echo "[GATE5-FIX] datasets id=${ID_DATASET} ood=${OOD_DATASET}"
echo "[GATE5-FIX] score_method=topk_weighted_abs_z top_k=5"

python scripts/gate5_clinical_relevance.py \
  --artifacts-root "${ARTIFACTS_ROOT}" \
  --dice-reference-dir "${DICE_REF_DIR}" \
  --experiment "${EXPERIMENT}" \
  --id-dataset "${ID_DATASET}" \
  --ood-dataset "${OOD_DATASET}" \
  --seeds "${SEEDS[@]}" \
  --endpoints predicted_mask mask_free \
  --score-method topk_weighted_abs_z \
  --score-top-k 5 \
  --min-corr 0.60 \
  --out-seed-csv "${OUT_SEED_CSV}" \
  --out-cases-csv "${OUT_CASES_CSV}" \
  --out-json "${OUT_CLINICAL_JSON}" \
  --out-md "${OUT_CLINICAL_MD}"

python scripts/gate5_finalize_decision.py \
  --clinical-json "${OUT_CLINICAL_JSON}" \
  --determinism-json "${DETERMINISM_JSON}" \
  --out-json "${OUT_FINAL_JSON}" \
  --out-md "${OUT_FINAL_MD}"

echo "[DONE] Gate-5 remediation analysis complete."
