#!/bin/bash
# Refresh journal_submission_bundle contents and rebuild the zip archive.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="${PROJECT_ROOT}/journal_submission_bundle"
ZIP_PATH="${PROJECT_ROOT}/journal_submission_bundle.zip"

log() {
    local msg="$1"
    printf "[bundle] %s\n" "$msg"
}

if [[ ! -d "${BUNDLE_DIR}" ]]; then
    log "Bundle directory missing: ${BUNDLE_DIR}"
    exit 1
fi

if [[ "$(basename "${PROJECT_ROOT}")" == "journal_submission_bundle" ]]; then
    log "Run this script from the main repo, not inside the bundle copy."
    exit 1
fi

log "Syncing reports..."
rsync -a "${PROJECT_ROOT}/reports/" "${BUNDLE_DIR}/reports/"

log "Syncing manuscript directory..."
rsync -a "${PROJECT_ROOT}/manuscript/" "${BUNDLE_DIR}/manuscript/"

log "Syncing scripts, src, configs..."
rsync -a "${PROJECT_ROOT}/scripts/" "${BUNDLE_DIR}/scripts/"
rsync -a "${PROJECT_ROOT}/src/" "${BUNDLE_DIR}/src/"
rsync -a "${PROJECT_ROOT}/configs/" "${BUNDLE_DIR}/configs/"

log "Syncing hypothesis tests..."
mkdir -p "${BUNDLE_DIR}/results/metrics/divergence"
if [[ -f "${PROJECT_ROOT}/results/metrics/divergence/hypothesis_tests.csv" ]]; then
    cp "${PROJECT_ROOT}/results/metrics/divergence/hypothesis_tests.csv" \
        "${BUNDLE_DIR}/results/metrics/divergence/hypothesis_tests.csv"
else
    log "WARN: hypothesis_tests.csv missing; skipping copy"
fi

log "Syncing analysis notes..."
if [[ -f "${PROJECT_ROOT}/reports/analysis_notes.md" ]]; then
    cp "${PROJECT_ROOT}/reports/analysis_notes.md" "${BUNDLE_DIR}/analysis_notes.md"
    cp "${PROJECT_ROOT}/reports/analysis_notes.md" "${BUNDLE_DIR}/reports/analysis_notes.md"
else
    log "WARN: reports/analysis_notes.md missing; skipping copy"
fi

log "Rebuilding zip: ${ZIP_PATH}"
cd "${PROJECT_ROOT}"
if command -v zip >/dev/null 2>&1; then
    zip -r "${ZIP_PATH}" "journal_submission_bundle" >/dev/null
else
    log "zip not found; using python zipfile fallback"
    python - <<'PY'
import os
import zipfile

root = os.path.abspath(os.getcwd())
bundle_dir = os.path.join(root, "journal_submission_bundle")
zip_path = os.path.join(root, "journal_submission_bundle.zip")

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for dirpath, _, filenames in os.walk(bundle_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            arcname = os.path.relpath(full_path, root)
            zf.write(full_path, arcname)
PY
fi
log "Zip rebuild complete"
