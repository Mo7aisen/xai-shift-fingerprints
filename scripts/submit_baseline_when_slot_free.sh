#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dependency_job_id> [experiment] [subset]"
  exit 1
fi

DEPENDENCY_JOB_ID="$1"
EXPERIMENT="${2:-jsrt_to_shenzhen}"
SUBSET="${3:-full}"

PROJECT_ROOT="/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full"
LOG="${PROJECT_ROOT}/logs/baseline_submit_watch_${DEPENDENCY_JOB_ID}.log"

cd "${PROJECT_ROOT}"
echo "[$(date -u +%FT%TZ)] watcher start dep=${DEPENDENCY_JOB_ID} exp=${EXPERIMENT} subset=${SUBSET}" >> "${LOG}"

while true; do
  if jid=$(sbatch --parsable \
    --dependency=afterok:"${DEPENDENCY_JOB_ID}" \
    --export=ALL,XFP_EXPERIMENT="${EXPERIMENT}",XFP_SUBSET="${SUBSET}" \
    scripts/run_journal_baselines_slurm.sh 2>>"${LOG}"); then
    echo "[$(date -u +%FT%TZ)] baseline submitted jid=${jid}" >> "${LOG}"
    echo "${jid}"
    break
  fi
  echo "[$(date -u +%FT%TZ)] submit retry in 120s" >> "${LOG}"
  sleep 120
done
