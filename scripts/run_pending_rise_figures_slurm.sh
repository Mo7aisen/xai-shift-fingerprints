#!/usr/bin/env bash
#SBATCH --job-name=xfp_rise_figs
#SBATCH --partition=batch_gpu_g2.large_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_%j.out
#SBATCH --error=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full"

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

log "Starting pending RISE + figure regeneration run"

source /storage/xai_cxr_safety/xai-al-env/bin/activate
cd "${PROJECT_ROOT}"

# Keep config portable across local/cluster roots without changing tracked YAML.
export XFP_DATASETS_ROOT="/storage/xai_cxr_safety/Datasets"
export XFP_MODELS_ROOT="/storage/xai_cxr_safety/models"
export XFP_FINGERPRINTS_ROOT="${PROJECT_ROOT}/data/fingerprints"
export XFP_CACHE_ROOT="${PROJECT_ROOT}/data/interim"
export XFP_ALLOW_MISSING_MODELS=1
export XFP_MODEL_UNET_JSRT_FULL="/storage/xai_cxr_safety/models/jsrt_unet_baseline.pt"
export XFP_MODEL_UNET_MONTGOMERY_FULL="/storage/xai_cxr_safety/models/montgomery_unet_baseline.pt"
export XFP_MODEL_UNET_SHENZHEN_FULL="/storage/xai_cxr_safety/models/shenzhen_unet_baseline.pt"
export XFP_MODEL_UNET_NIH_FULL="/storage/xai_cxr_safety/models/nih_unet_transfer.pt"

if [ ! -f "${XFP_MODEL_UNET_SHENZHEN_FULL}" ]; then
  log "ERROR: Missing dedicated Shenzhen checkpoint: ${XFP_MODEL_UNET_SHENZHEN_FULL}"
  log "Train and upload Shenzhen model before running this workflow."
  exit 1
fi

log "Environment check"
python - <<'PY'
import torch, pandas, pyarrow, seaborn, scipy
print("torch", torch.__version__, "cuda=", torch.cuda.is_available())
print("pandas", pandas.__version__)
print("pyarrow", pyarrow.__version__)
PY

log "GPU check"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

log "Run RISE robustness analysis"
python scripts/run_rise_segmentation.py \
  --max-samples 200 \
  --n-masks 1000 \
  --mask-batch 4 \
  --num-workers 2

log "Regenerate publication figures"
python scripts/generate_publication_figures_final.py

log "Sync updated figures into submission folders"
mkdir -p submission_medical_image_analysis/figures final_elsevier_submission/figures
rsync -a manuscript/figures/ submission_medical_image_analysis/figures/
rsync -a manuscript/figures/ final_elsevier_submission/figures/

log "Repack final submission zip"
rm -f final_elsevier_submission.zip
(cd final_elsevier_submission && zip -rq ../final_elsevier_submission.zip .)

log "Done"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
