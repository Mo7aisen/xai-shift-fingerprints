#!/usr/bin/env bash
# Watch a Slurm job on HUN-REN and pull validated outputs on completion.
#
# Usage:
#   bash scripts/watch_cluster_job_and_pull.sh 2593

set -euo pipefail

JOB_ID="${1:-}"
if [[ -z "${JOB_ID}" ]]; then
  echo "Usage: $0 <job_id>"
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/cluster_watch_${JOB_ID}.log"

REMOTE_USER="${REMOTE_USER:-xai_cxr_safety}"
REMOTE_HOST="${REMOTE_HOST:-slurm.science-cloud.hu}"
REMOTE_KEY="${REMOTE_KEY:-/home/ubuntu/xai_key.pem}"

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1" | tee -a "${LOG_FILE}"
}

queue_state() {
  ssh -i "${REMOTE_KEY}" -o BatchMode=yes "${REMOTE_USER}@${REMOTE_HOST}" \
    "squeue -j ${JOB_ID} -h -o '%t %M %R'" 2>/dev/null || true
}

log "Watching Slurm job ${JOB_ID}"

while true; do
  state="$(queue_state)"
  if [[ -z "${state}" ]]; then
    log "Job ${JOB_ID} no longer in queue. Fetching sacct state."
    final_state="$(ssh -i "${REMOTE_KEY}" -o BatchMode=yes "${REMOTE_USER}@${REMOTE_HOST}" \
      "sacct -j ${JOB_ID} --format=JobID,State,Elapsed -n -P | head -n 3" 2>/dev/null || true)"
    log "sacct: ${final_state}"
    break
  fi
  log "state: ${state}"
  sleep 60
done

log "Pulling outputs and running local quality gate"
bash "${PROJECT_ROOT}/scripts/pull_cluster_updates.sh" | tee -a "${LOG_FILE}"

log "Watcher complete"
