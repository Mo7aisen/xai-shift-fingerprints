#!/usr/bin/env bash
#SBATCH --job-name=xfp_gate4
#SBATCH --partition=batch_gpu_g2.xlarge_16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_gate4_%j.out
#SBATCH --error=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_gate4_%j.err

set -euo pipefail

PROJECT_ROOT="/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full"
ENV_ACTIVATE="/storage/xai_cxr_safety/xai-al-env/bin/activate"
AUDIT_DIR="${PROJECT_ROOT}/reports_v2/audits"
GATE4_ROOT="${PROJECT_ROOT}/reports_v2/gate4"

SEED="${XFP_SEED:-42}"
EXPERIMENT="${XFP_EXPERIMENT:-jsrt_to_montgomery}"
IG_STEPS_A="${XFP_GATE4_IG_A:-16}"
IG_STEPS_B="${XFP_GATE4_IG_B:-32}"
ROBUST_SUBSET_SIZE="${XFP_GATE4_ROBUST_SUBSET_SIZE:-50}"
DATE_TAG="${XFP_DATE_TAG:-2026-02-17}"
OUTPUT_TAG="${XFP_OUTPUT_TAG:-}"

if [[ -n "${OUTPUT_TAG}" ]]; then
  SAFE_OUTPUT_TAG="$(echo "${OUTPUT_TAG}" | tr '/: ' '___')"
  OUT_JSON="${AUDIT_DIR}/GATE4_IG_ROBUSTNESS_${SAFE_OUTPUT_TAG}.json"
  OUT_MD="${AUDIT_DIR}/GATE4_IG_ROBUSTNESS_${SAFE_OUTPUT_TAG}.md"
else
  OUT_JSON="${AUDIT_DIR}/GATE4_IG_ROBUSTNESS_SUMMARY.json"
  OUT_MD="${AUDIT_DIR}/GATE4_IG_ROBUSTNESS_${DATE_TAG}.md"
fi

mkdir -p "${PROJECT_ROOT}/logs" "${AUDIT_DIR}" "${GATE4_ROOT}"
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

echo "[GATE4] seed=${SEED} experiment=${EXPERIMENT} ig_compare=${IG_STEPS_A}/${IG_STEPS_B}"
echo "[GATE4] datasets id=${ID_DATASET} ood=${OOD_DATASET}"
echo "[GATE4] deterministic: XFP_DETERMINISTIC=${XFP_DETERMINISTIC}, CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}, PYTHONHASHSEED=${PYTHONHASHSEED}"
echo "[GATE4] outputs: ${OUT_JSON} | ${OUT_MD}"

for ig_steps in "${IG_STEPS_A}" "${IG_STEPS_B}"; do
  export XFP_FINGERPRINTS_ROOT="${GATE4_ROOT}/artifacts/ig${ig_steps}/seed${SEED}"
  for endpoint in predicted_mask mask_free; do
    echo "[GATE4] running fingerprints endpoint=${endpoint} ig_steps=${ig_steps}"
    python scripts/run_fingerprint.py \
      --experiment "${EXPERIMENT}" \
      --subset full \
      --device cuda \
      --endpoint-mode "${endpoint}" \
      --seed "${SEED}" \
      --ig-steps "${ig_steps}" \
      --deterministic \
      --skip-validate
  done
done

for endpoint in predicted_mask mask_free; do
  robust_out="${GATE4_ROOT}/robustness/${endpoint}_ig${IG_STEPS_A}"
  echo "[GATE4] robustness endpoint=${endpoint} ig_steps=${IG_STEPS_A} out=${robust_out}"
  python scripts/robustness_perturbations.py \
    --datasets "${ID_DATASET},${OOD_DATASET}" \
    --subset-size "${ROBUST_SUBSET_SIZE}" \
    --seed "${SEED}" \
    --device cuda \
    --output-dir "${robust_out}" \
    --experiment "${EXPERIMENT}" \
    --endpoint-mode "${endpoint}" \
    --ig-steps "${IG_STEPS_A}" \
    --deterministic
done

python scripts/gate4_compile_report.py \
  --root "${GATE4_ROOT}" \
  --experiment "${EXPERIMENT}" \
  --seed "${SEED}" \
  --ig-a "${IG_STEPS_A}" \
  --ig-b "${IG_STEPS_B}" \
  --out-json "${OUT_JSON}" \
  --out-md "${OUT_MD}"

echo "[DONE] Gate-4 run complete."
