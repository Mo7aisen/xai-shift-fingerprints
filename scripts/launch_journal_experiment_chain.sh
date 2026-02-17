#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full}"
EXPERIMENT="${2:-jsrt_to_shenzhen}"
DATE_TAG="${3:-$(date -u +%F)}"
SEEDS="${XFP_SEEDS:-42 43 44 45 46}"
SUBSET="${XFP_SUBSET:-full}"
OUTPUT_TAG="${EXPERIMENT}_${DATE_TAG}"

cd "${PROJECT_ROOT}"

echo "[CHAIN] project_root=${PROJECT_ROOT}"
echo "[CHAIN] experiment=${EXPERIMENT} subset=${SUBSET} seeds=${SEEDS}"
echo "[CHAIN] output_tag=${OUTPUT_TAG}"

gate3_job=$(sbatch --parsable \
  --export=ALL,XFP_EXPERIMENT="${EXPERIMENT}",XFP_SUBSET="${SUBSET}",XFP_SEEDS="${SEEDS}",XFP_HASH_SOURCE=none,XFP_BATCH_TAG="gate3_${OUTPUT_TAG}",XFP_OUTPUT_TAG="${OUTPUT_TAG}",XFP_DATE_TAG="${DATE_TAG}" \
  scripts/reconstruct_gate3_seed_artifacts_slurm.sh)
echo "[CHAIN] submitted gate3 job=${gate3_job}"

gate4_job=$(sbatch --parsable \
  --dependency=afterok:"${gate3_job}" \
  --export=ALL,XFP_EXPERIMENT="${EXPERIMENT}",XFP_SUBSET="${SUBSET}",XFP_OUTPUT_TAG="${OUTPUT_TAG}",XFP_DATE_TAG="${DATE_TAG}" \
  scripts/run_gate4_slurm.sh)
echo "[CHAIN] submitted gate4 job=${gate4_job} (afterok:${gate3_job})"

gate5_job=$(sbatch --parsable \
  --dependency=afterok:"${gate4_job}" \
  --export=ALL,XFP_EXPERIMENT="${EXPERIMENT}",XFP_SUBSET="${SUBSET}",XFP_SEEDS="${SEEDS}",XFP_OUTPUT_TAG="${OUTPUT_TAG}",XFP_DATE_TAG="${DATE_TAG}" \
  scripts/run_gate5_slurm.sh)
echo "[CHAIN] submitted gate5 job=${gate5_job} (afterok:${gate4_job})"

baseline_job=""
set +e
baseline_job=$(sbatch --parsable \
  --dependency=afterok:"${gate5_job}" \
  --export=ALL,XFP_EXPERIMENT="${EXPERIMENT}",XFP_SUBSET="${SUBSET}" \
  scripts/run_journal_baselines_slurm.sh 2>/tmp/xfp_baseline_submit.err)
baseline_rc=$?
set -e
if [[ ${baseline_rc} -eq 0 ]]; then
  echo "[CHAIN] submitted baselines job=${baseline_job} (afterok:${gate5_job})"
else
  echo "[CHAIN][WARN] baseline submit deferred (likely submit-limit)."
  cat /tmp/xfp_baseline_submit.err
fi

echo "[CHAIN] done"
echo "  gate3=${gate3_job}"
echo "  gate4=${gate4_job}"
echo "  gate5=${gate5_job}"
echo "  baselines=${baseline_job:-NOT_SUBMITTED}"
