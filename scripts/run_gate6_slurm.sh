#!/usr/bin/env bash
#SBATCH --job-name=xfp_gate6
#SBATCH --partition=batch_gpu_g2.large_8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_gate6_%j.out
#SBATCH --error=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full/logs/slurm_gate6_%j.err

set -euo pipefail

PROJECT_ROOT="/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full"
ENV_ACTIVATE="/storage/xai_cxr_safety/xai-al-env/bin/activate"
DATE_TAG="${XFP_DATE_TAG:-2026-02-17}"
BUNDLE_TAG="${XFP_BUNDLE_TAG:-gate6_final}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/reports_v2/audits" "${PROJECT_ROOT}/reports_v2/releases"
cd "${PROJECT_ROOT}"

if [[ ! -f "${ENV_ACTIVATE}" ]]; then
  echo "[ERROR] Missing virtualenv: ${ENV_ACTIVATE}"
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_ACTIVATE}"

echo "[GATE6] date_tag=${DATE_TAG} bundle_tag=${BUNDLE_TAG}"

# Keep safeguards active; CPU-only tests.
pytest -q \
  tests/test_gt_leakage.py \
  tests/test_all_metadata_have_patient_id.py \
  tests/test_metadata.py \
  tests/test_shift_metrics.py

# Refresh baseline freeze manifests (legacy + code/config).
./scripts/generate_freeze_manifests.sh "${DATE_TAG}"

# Build Gate-6 reproducibility package and final audit report.
python scripts/gate6_repro_bundle.py \
  --root . \
  --date-tag "${DATE_TAG}" \
  --bundle-tag "${BUNDLE_TAG}" \
  --out-audit-json reports_v2/audits/GATE6_REPRO_SUMMARY.json \
  --out-audit-md reports_v2/audits/GATE6_REPRODUCIBILITY_2026-02-17.md

echo "[DONE] Gate-6 packaging complete."
