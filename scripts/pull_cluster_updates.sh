#!/usr/bin/env bash
# Pull cluster outputs to local repo and run a lightweight quality gate.
#
# Usage:
#   bash scripts/pull_cluster_updates.sh
# Optional env overrides:
#   REMOTE_USER=xai_cxr_safety
#   REMOTE_HOST=slurm.science-cloud.hu
#   REMOTE_KEY=/home/ubuntu/xai_key.pem
#   REMOTE_PROJECT=/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_USER="${REMOTE_USER:-xai_cxr_safety}"
REMOTE_HOST="${REMOTE_HOST:-slurm.science-cloud.hu}"
REMOTE_KEY="${REMOTE_KEY:-/home/ubuntu/xai_key.pem}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/storage/xai_cxr_safety/xai_shift_fingerprints_reproduction_20251225_full}"
SSH_OPTS="-i ${REMOTE_KEY} -o StrictHostKeyChecking=accept-new"

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

cd "${PROJECT_ROOT}"

log "Pulling cluster outputs into local repo"
rsync -av -e "ssh ${SSH_OPTS}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT}/reports/rise/" \
  "reports/rise/"

rsync -av -e "ssh ${SSH_OPTS}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT}/manuscript/figures/" \
  "manuscript/figures/"

rsync -av -e "ssh ${SSH_OPTS}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT}/submission_medical_image_analysis/figures/" \
  "submission_medical_image_analysis/figures/"

rsync -av -e "ssh ${SSH_OPTS}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT}/final_elsevier_submission/figures/" \
  "final_elsevier_submission/figures/"

rsync -av -e "ssh ${SSH_OPTS}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT}/final_elsevier_submission.zip" \
  "final_elsevier_submission.zip"

rsync -av -e "ssh ${SSH_OPTS}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT}/logs/slurm_"*".out" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PROJECT}/logs/slurm_"*".err" \
  "logs/" || true

log "Running manuscript data verification"
if [[ -f "/home/ubuntu/xai-env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source /home/ubuntu/xai-env/bin/activate
fi
python scripts/verify_manuscript_data.py

log "Done"
