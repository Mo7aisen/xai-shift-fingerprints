#!/usr/bin/env python3
"""
NIH ChestXray14 Checksum Verifier
=================================

Purpose & Output
----------------
Verifies that all required NIH archive files exist under
`${datasets_root}/NIH_ChestXray14/raw/` (as configured in `configs/paths.yaml`)
and optionally validates SHA256 hashes when a `CHECKSUMS.sha256` file is present.

Produces a Markdown report `reports/external_validation/nih_download_status.md`
listing each archive, its size, and checksum result. This keeps the NIH onboarding
pipeline auditable without attempting any prohibited downloads.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "external_validation" / "nih_download_status.md"

EXPECTED_FILES = [f"images_{i:03d}.tar.gz" for i in range(1, 13)] + [
    "Data_Entry_2017.csv",
    "BBox_List_2017.csv",
]

LOGGER = logging.getLogger("nih_checksum")


def load_config() -> Path:
    """Return the raw NIH directory from configs/paths.yaml."""
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    datasets_root = Path(config["paths"]["datasets_root"])
    return datasets_root / "NIH_ChestXray14" / "raw"


def sha256sum(path: Path) -> str:
    """Compute SHA256 for a file."""
    hash_obj = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def load_reference_checksums(raw_dir: Path) -> Dict[str, str]:
    """Parse CHECKSUMS.sha256 if present."""
    checksum_path = raw_dir / "CHECKSUMS.sha256"
    if not checksum_path.exists():
        LOGGER.warning("Checksum file %s not found.", checksum_path)
        return {}
    refs: Dict[str, str] = {}
    with checksum_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            checksum, filename = line.split(maxsplit=1)
            refs[filename.strip()] = checksum
    return refs


def generate_report(raw_dir: Path, results: List[Dict[str, str]]) -> None:
    """Write status report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        handle.write("# NIH Download Status\n\n")
        handle.write(f"Raw directory: `{raw_dir}`\n\n")
        handle.write("| File | Exists | Size (MB) | SHA256 | Status |\n")
        handle.write("|------|--------|-----------|--------|--------|\n")
        for row in results:
            handle.write(
                f"| {row['file']} | {row['exists']} | {row['size_mb']} | {row['sha256']} | {row['status']} |\n"
            )
    LOGGER.info("Saved report to %s", REPORT_PATH)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw_dir = load_config()
    raw_dir.mkdir(parents=True, exist_ok=True)
    ref_checksums = load_reference_checksums(raw_dir)
    results: List[Dict[str, str]] = []
    for filename in EXPECTED_FILES:
        path = raw_dir / filename
        exists = path.exists()
        size_mb = f"{path.stat().st_size / (1024**2):.1f}" if exists else "-"
        computed_hash = sha256sum(path) if exists else "-"
        status = "missing"
        if exists:
            if filename in ref_checksums:
                status = "match" if computed_hash == ref_checksums[filename] else "mismatch"
            else:
                status = "present (no ref)"
        results.append(
            {
                "file": filename,
                "exists": "yes" if exists else "no",
                "size_mb": size_mb,
                "sha256": computed_hash if exists else "-",
                "status": status,
            }
        )
        LOGGER.info("%s: %s", filename, status)
    generate_report(raw_dir, results)


if __name__ == "__main__":
    main()
