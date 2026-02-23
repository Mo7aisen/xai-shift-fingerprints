#!/usr/bin/env bash
# Train an alternative JSRT UNet configuration and compare fingerprint stability vs canonical.
#
# Outputs are written under:
#   reports_v2/audits/model_variants/<VARIANT_TAG>/
#   data/fingerprints_model_variants/<VARIANT_TAG>/
#
# Environment overrides:
#   XFP_VARIANT_TAG
#   XFP_VARIANT_FEATURES
#   XFP_VARIANT_LOSS
#   XFP_VARIANT_SEED
#   XFP_VARIANT_EPOCHS
#   XFP_VARIANT_BATCH_SIZE
#   XFP_VARIANT_NUM_WORKERS
#   XFP_VARIANT_EXPERIMENT   (default: jsrt_to_montgomery)
#   XFP_VARIANT_IG_STEPS     (default: 16)

#SBATCH --job-name=xfp_model_var
#SBATCH --partition=batch_gpu_g2.xlarge_16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_model_variant_%j.out
#SBATCH --error=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_model_variant_%j.err

set -euo pipefail

PROJECT_ROOT="/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full"
ENV_ACTIVATE="/storage/xai_cxr_safety/xai-al-env/bin/activate"
MODEL_DIR="/storage/xai_cxr_safety/models"

# NOTE: when passing via `sbatch --export`, commas split variables.
# You can pass `XFP_VARIANT_FEATURES=24_48_96_192`; underscores are normalized to commas.
VARIANT_FEATURES_RAW="${XFP_VARIANT_FEATURES:-24,48,96,192}"
VARIANT_FEATURES="${VARIANT_FEATURES_RAW//_/,}"
VARIANT_LOSS="${XFP_VARIANT_LOSS:-dice}"
VARIANT_SEED="${XFP_VARIANT_SEED:-314159}"
VARIANT_EPOCHS="${XFP_VARIANT_EPOCHS:-50}"
VARIANT_BATCH_SIZE="${XFP_VARIANT_BATCH_SIZE:-8}"
VARIANT_NUM_WORKERS="${XFP_VARIANT_NUM_WORKERS:-4}"
VARIANT_EXPERIMENT="${XFP_VARIANT_EXPERIMENT:-jsrt_to_montgomery}"
VARIANT_IG_STEPS="${XFP_VARIANT_IG_STEPS:-16}"

VARIANT_TAG="${XFP_VARIANT_TAG:-jsrt_unet_${VARIANT_LOSS}_f$(echo "${VARIANT_FEATURES}" | tr ',' '_')_seed${VARIANT_SEED}}"
VARIANT_MODEL_PATH="${MODEL_DIR}/${VARIANT_TAG}.pt"
VARIANT_FP_ROOT="${PROJECT_ROOT}/data/fingerprints_model_variants/${VARIANT_TAG}"
AUDIT_DIR="${PROJECT_ROOT}/reports_v2/audits/model_variants/${VARIANT_TAG}"

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

mkdir -p "${PROJECT_ROOT}/logs" "${MODEL_DIR}" "${VARIANT_FP_ROOT}" "${AUDIT_DIR}"
cd "${PROJECT_ROOT}"

if [[ ! -f "${ENV_ACTIVATE}" ]]; then
  echo "[ERROR] Missing virtualenv: ${ENV_ACTIVATE}"
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_ACTIVATE}"

export XFP_DATASETS_ROOT="/storage/xai_cxr_safety/Datasets"
export XFP_MODELS_ROOT="${MODEL_DIR}"
export XFP_CACHE_ROOT="${PROJECT_ROOT}/data/interim"
export XFP_FINGERPRINTS_ROOT="${VARIANT_FP_ROOT}"
export XFP_MODEL_UNET_JSRT_FULL="${VARIANT_MODEL_PATH}"
export XFP_MODEL_UNET_MONTGOMERY_FULL="${MODEL_DIR}/montgomery_unet_baseline.pt"
export XFP_MODEL_UNET_SHENZHEN_FULL="${MODEL_DIR}/shenzhen_unet_baseline.pt"
export XFP_MODEL_UNET_NIH_FULL="${MODEL_DIR}/nih_unet_transfer.pt"

log "Variant generalization job started"
log "VARIANT_TAG=${VARIANT_TAG}"
log "features=${VARIANT_FEATURES} loss=${VARIANT_LOSS} seed=${VARIANT_SEED}"
log "experiment=${VARIANT_EXPERIMENT} ig_steps=${VARIANT_IG_STEPS}"

python - <<'PY'
import torch, pandas, scipy
print("torch", torch.__version__, "cuda=", torch.cuda.is_available())
print("pandas", pandas.__version__)
print("scipy", scipy.__version__)
PY

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

log "Training alternate JSRT UNet variant"
python scripts/train_unet_from_cache.py \
  --cache-dir "${PROJECT_ROOT}/data/interim/jsrt/full" \
  --epochs "${VARIANT_EPOCHS}" \
  --batch-size "${VARIANT_BATCH_SIZE}" \
  --lr 1e-3 \
  --seed "${VARIANT_SEED}" \
  --num-workers "${VARIANT_NUM_WORKERS}" \
  --output "${VARIANT_MODEL_PATH}" \
  --device cuda \
  --loss "${VARIANT_LOSS}" \
  --features "${VARIANT_FEATURES}"

test -f "${VARIANT_MODEL_PATH}"
test -f "${VARIANT_MODEL_PATH%.pt}.json"

log "Running fingerprint experiment with variant model (predicted_mask)"
python scripts/run_fingerprint.py \
  --experiment "${VARIANT_EXPERIMENT}" \
  --skip-validate \
  --device cuda \
  --endpoint-mode predicted_mask \
  --seed 42 \
  --ig-steps "${VARIANT_IG_STEPS}" \
  --deterministic

log "Running fingerprint experiment with variant model (mask_free)"
python scripts/run_fingerprint.py \
  --experiment "${VARIANT_EXPERIMENT}" \
  --skip-validate \
  --device cuda \
  --endpoint-mode mask_free \
  --seed 42 \
  --ig-steps "${VARIANT_IG_STEPS}" \
  --deterministic

log "Comparing variant fingerprints vs canonical"
python scripts/model_variant_fingerprint_report.py \
  --canonical-root "${PROJECT_ROOT}/data/fingerprints" \
  --variant-root "${VARIANT_FP_ROOT}" \
  --experiment "${VARIANT_EXPERIMENT}" \
  --id-dataset jsrt \
  --ood-dataset montgomery \
  --endpoints predicted_mask mask_free \
  --top-k 20 \
  --score-method all_mean_abs_z \
  --out-csv "${AUDIT_DIR}/model_variant_fingerprint_comparison_${VARIANT_EXPERIMENT}.csv" \
  --out-json "${AUDIT_DIR}/model_variant_fingerprint_comparison_${VARIANT_EXPERIMENT}.json" \
  --out-md "${AUDIT_DIR}/model_variant_fingerprint_comparison_${VARIANT_EXPERIMENT}.md"

cp -f "${VARIANT_MODEL_PATH%.pt}.json" "${AUDIT_DIR}/training_metadata.json"

cat > "${AUDIT_DIR}/RUN_INFO.txt" <<EOF
variant_tag=${VARIANT_TAG}
variant_model_path=${VARIANT_MODEL_PATH}
variant_fingerprint_root=${VARIANT_FP_ROOT}
experiment=${VARIANT_EXPERIMENT}
features=${VARIANT_FEATURES}
loss=${VARIANT_LOSS}
seed=${VARIANT_SEED}
epochs=${VARIANT_EPOCHS}
batch_size=${VARIANT_BATCH_SIZE}
num_workers=${VARIANT_NUM_WORKERS}
ig_steps=${VARIANT_IG_STEPS}
EOF

log "DONE variant generalization workflow complete"
