#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/reports_v2/manifests"
DATE_TAG="${1:-$(date +%F)}"

mkdir -p "${OUT_DIR}"

legacy_manifest="${OUT_DIR}/freeze_legacy_artifacts_${DATE_TAG}.sha256"
code_manifest="${OUT_DIR}/freeze_code_configs_${DATE_TAG}.sha256"

cd "${ROOT_DIR}"

{
  find manuscript final_elsevier_submission submission_medical_image_analysis results reports -type f \
    ! -path '*/__pycache__/*' \
    ! -name '*.log' \
    ! -name '*.tmp' \
    ! -name '*.temp' \
    -print
  [ -f divergence_comparison_table.csv ] && echo divergence_comparison_table.csv
} | LC_ALL=C sort | xargs -r sha256sum > "${legacy_manifest}"

{
  find src scripts tests configs -type f \
    ! -path '*/__pycache__/*' \
    -print
  for f in README.md requirements.txt environment.yml pyproject.toml; do
    [ -f "${f}" ] && echo "${f}"
  done
} | LC_ALL=C sort | xargs -r sha256sum > "${code_manifest}"

printf 'Wrote %s (%s files)\n' "${legacy_manifest}" "$(wc -l < "${legacy_manifest}")"
printf 'Wrote %s (%s files)\n' "${code_manifest}" "$(wc -l < "${code_manifest}")"
