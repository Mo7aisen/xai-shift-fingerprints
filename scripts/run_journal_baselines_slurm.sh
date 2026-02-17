#!/usr/bin/env bash
#SBATCH --job-name=xfp_baselines
#SBATCH --partition=batch_gpu_g2.xlarge_16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_baselines_%j.out
#SBATCH --error=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_baselines_%j.err

set -euo pipefail

PROJECT_ROOT="/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full"
ENV_ACTIVATE="/storage/xai_cxr_safety/xai-al-env/bin/activate"

EXPERIMENT="${XFP_EXPERIMENT:-jsrt_to_montgomery}"
SUBSET="${XFP_SUBSET:-full}"
MAX_SAMPLES="${XFP_BASELINE_MAX_SAMPLES:-0}"
BATCH_SIZE="${XFP_BASELINE_BATCH_SIZE:-32}"
NUM_WORKERS="${XFP_BASELINE_NUM_WORKERS:-4}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results/baselines"
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

echo "[BASELINES] experiment=${EXPERIMENT} subset=${SUBSET}"
python scripts/run_journal_ood_baselines.py \
  --experiment "${EXPERIMENT}" \
  --subset "${SUBSET}" \
  --max-samples "${MAX_SAMPLES}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}"

echo "[DONE] baseline suite complete for ${EXPERIMENT}"
