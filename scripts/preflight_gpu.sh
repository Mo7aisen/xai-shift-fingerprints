#!/usr/bin/env bash
set -euo pipefail

MAX_VRAM_MIB="${PRECHECK_MAX_VRAM_MIB:-8000}"
NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

fail() {
  echo "[PREFLIGHT][FAIL] $1"
  exit 1
}

pass() {
  echo "[PREFLIGHT][PASS] $1"
}

echo "[PREFLIGHT] timestamp_utc=${NOW_UTC}"
echo "[PREFLIGHT] max_vram_mib=${MAX_VRAM_MIB}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not found"
fi

GPU_ROWS="$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)"
if [[ -z "${GPU_ROWS}" ]]; then
  fail "No GPU rows returned by nvidia-smi"
fi

echo "[PREFLIGHT] gpu_snapshot"
echo "${GPU_ROWS}" | sed 's/^/  - /'

# Check VRAM threshold on all visible GPUs
while IFS=',' read -r idx name used total util; do
  idx="$(echo "${idx}" | xargs)"
  name="$(echo "${name}" | xargs)"
  used="$(echo "${used}" | xargs)"
  total="$(echo "${total}" | xargs)"
  util="$(echo "${util}" | xargs)"

  if [[ -z "${used}" || ! "${used}" =~ ^[0-9]+$ ]]; then
    fail "Could not parse memory.used for GPU ${idx} (${name})"
  fi

  if (( used > MAX_VRAM_MIB )); then
    fail "GPU ${idx} (${name}) used=${used}MiB exceeds threshold=${MAX_VRAM_MIB}MiB"
  fi

done <<< "${GPU_ROWS}"

APPS_RAW="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true)"

if [[ -n "${APPS_RAW}" ]]; then
  echo "[PREFLIGHT] active_compute_apps"
  echo "${APPS_RAW}" | sed 's/^/  - /'
  fail "Active compute process(es) detected"
fi

pass "GPU preflight checks passed; pilot can start"
