#!/bin/bash
#
# Launch the GPU fingerprint experiments sequentially with logging.
# This helper is intended to be wrapped by tmux/nohup so runs survive logouts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="/home/ubuntu/xai-env/bin/activate"
LOG_ROOT="${PROJECT_DIR}/logs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_ROOT}/gpu_${RUN_TS}"
MASTER_LOG="${LOG_DIR}/master.log"
EXPERIMENTS=("$@")

mkdir -p "${LOG_DIR}"

log() {
    local msg="$1"
    printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "${msg}" | tee -a "${MASTER_LOG}"
}

log "GPU fingerprint batch starting (logs in ${LOG_DIR})"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    log "Activated virtualenv at ${ENV_FILE}"
else
    log "WARNING: Virtualenv ${ENV_FILE} not found; using system Python"
fi

# NOTE: expandable_segments can trigger CUDA driver errors on some vGPU setups.
# Leave unset unless explicitly needed for memory fragmentation issues.

log "GPU status snapshot:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | tee -a "${MASTER_LOG}"

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
    EXPERIMENTS=(
        jsrt_baseline
        jsrt_to_montgomery
        montgomery_baseline
        montgomery_to_jsrt
        shenzhen_baseline
        jsrt_to_shenzhen
        montgomery_to_shenzhen
    )
fi

for exp in "${EXPERIMENTS[@]}"; do
    log "Launching experiment: ${exp}"
    if python "${PROJECT_DIR}/scripts/run_fingerprint.py" \
        --experiment "${exp}" \
        --device cuda \
        > "${LOG_DIR}/${exp}.log" 2>&1; then
        log "✓ Completed ${exp}"
    else
        status=$?
        log "✗ ${exp} failed (exit code ${status}). See ${LOG_DIR}/${exp}.log"
        exit "${status}"
    fi
    python - <<'PY' >> "${MASTER_LOG}" 2>&1
import torch
if not torch.cuda.is_available():
    print("[GPU MEM] CUDA unavailable")
else:
    stats = torch.cuda.memory_stats()
    allocated = stats.get("allocated_bytes.all.current", 0) / 1e6
    print(f"[GPU MEM] allocated={allocated:.2f}MB")
PY
    log "Sleeping 10s before next experiment"
    sleep 10
done

log "All requested experiments completed successfully"
echo "${LOG_DIR}" > "${LOG_ROOT}/latest_gpu_batch_dir.txt"
