#!/usr/bin/env bash
#SBATCH --job-name=xfp_phase1_sz
#SBATCH --partition=batch_gpu_g2.large_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_phase1_%j.out
#SBATCH --error=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_phase1_%j.err

set -euo pipefail

PROJECT_ROOT="/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full"
MODEL_DIR="/storage/xai_cxr_safety/models"
MODEL_PATH="${MODEL_DIR}/shenzhen_unet_baseline.pt"

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

log "Starting Phase 1: dedicated Shenzhen model + fingerprints"

source /storage/xai_cxr_safety/xai-al-env/bin/activate
cd "${PROJECT_ROOT}"

export XFP_DATASETS_ROOT="/storage/xai_cxr_safety/Datasets"
export XFP_MODELS_ROOT="${MODEL_DIR}"
export XFP_FINGERPRINTS_ROOT="${PROJECT_ROOT}/data/fingerprints"
export XFP_CACHE_ROOT="${PROJECT_ROOT}/data/interim"
export XFP_MODEL_UNET_JSRT_FULL="${MODEL_DIR}/jsrt_unet_baseline.pt"
export XFP_MODEL_UNET_MONTGOMERY_FULL="${MODEL_DIR}/montgomery_unet_baseline.pt"
export XFP_MODEL_UNET_SHENZHEN_FULL="${MODEL_PATH}"
export XFP_MODEL_UNET_NIH_FULL="${MODEL_DIR}/nih_unet_transfer.pt"

mkdir -p "${MODEL_DIR}"
mkdir -p "${PROJECT_ROOT}/logs"

log "Environment check"
python - <<'PY'
import torch, pandas, scipy, yaml
print("torch", torch.__version__, "cuda=", torch.cuda.is_available())
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
PY

log "GPU info"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

log "Train dedicated Shenzhen model"
python scripts/train_unet_from_cache.py \
  --cache-dir "${PROJECT_ROOT}/data/interim/shenzhen/full" \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --num-workers 4 \
  --output "${MODEL_PATH}" \
  --device cuda

if [ ! -f "${MODEL_PATH}" ]; then
  log "ERROR: training did not produce ${MODEL_PATH}"
  exit 1
fi

log "Run integrity check"
python scripts/pre_submission_integrity_check.py || true

log "Recompute Shenzhen baseline fingerprints with dedicated model"
python scripts/run_fingerprint.py --experiment shenzhen_baseline --skip-validate --device cuda

log "Recompute JSRT->Shenzhen fingerprints for comparison"
python scripts/run_fingerprint.py --experiment jsrt_to_shenzhen --skip-validate --device cuda

log "Phase 1 complete"
