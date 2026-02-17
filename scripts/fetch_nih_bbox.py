#!/usr/bin/env python3
"""
NIH ChestXray14 Bounding-Box Downloader
======================================

Automates staging of `BBox_List_2017.csv` inside
`${datasets_root}/NIH_ChestXray14/raw/` as configured in `configs/paths.yaml`.

Usage:
    python scripts/fetch_nih_bbox.py --source <CSV/ZIP/GZ path or URL>
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import logging
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"
TARGET_FILENAME = "BBox_List_2017.csv"


def load_raw_dir() -> Path:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    datasets_root = Path(config["paths"]["datasets_root"])
    raw_dir = datasets_root / "NIH_ChestXray14" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_url(source: str) -> bool:
    parsed = urllib.parse.urlparse(source)
    return parsed.scheme in {"http", "https"}


def download_to_temp(url: str) -> Path:
    logging.info("Downloading %s", url)
    with urllib.request.urlopen(url) as response:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(response, tmp)
            tmp_path = Path(tmp.name)
    logging.info("Downloaded %s bytes", tmp_path.stat().st_size)
    return tmp_path


def extract_csv(source: Path) -> Path:
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return source
    tmp_dir = Path(tempfile.mkdtemp())
    if suffix == ".gz":
        target = tmp_dir / TARGET_FILENAME
        with gzip.open(source, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return target
    if suffix == ".zip":
        with zipfile.ZipFile(source, "r") as archive:
            members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
            if not members:
                raise RuntimeError("ZIP archive does not contain CSV files.")
            member = members[0]
            archive.extract(member, path=tmp_dir)
            return tmp_dir / member
    raise RuntimeError(f"Unsupported source format: {source}")


def stage(csv_path: Path, destination: Path, overwrite: bool, dry_run: bool) -> None:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists. Use --overwrite to replace it.")
    if dry_run:
        logging.info("Dry run enabled: skipping move for %s", csv_path)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(csv_path), destination)
    logging.info("Staged %s (%.2f MB)", destination, destination.stat().st_size / (1024 * 1024))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage NIH ChestXray14 bounding boxes.")
    parser.add_argument("--source", required=True, help="Local path or HTTPS URL to the archive/CSV.")
    parser.add_argument("--sha256", help="Optional SHA256 checksum to validate the CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Extract but skip moving to the raw dir.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary downloads for debugging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    raw_dir = load_raw_dir()
    target_path = raw_dir / TARGET_FILENAME

    temp_paths: list[Path] = []
    try:
        if is_url(args.source):
            source_path = download_to_temp(args.source)
            temp_paths.append(source_path)
        else:
            source_path = Path(args.source).expanduser().resolve()
            if not source_path.exists():
                raise FileNotFoundError(f"Source {source_path} not found.")

        csv_path = extract_csv(source_path)
        if csv_path not in temp_paths:
            temp_paths.append(csv_path)

        sha = compute_sha256(csv_path)
        logging.info("SHA256: %s", sha)
        if args.sha256 and sha.lower() != args.sha256.lower():
            raise RuntimeError(f"Checksum mismatch. Expected {args.sha256}, got {sha}.")

        stage(csv_path, target_path, overwrite=args.overwrite, dry_run=args.dry_run)
    finally:
        if not args.keep_temp:
            for path in temp_paths:
                if path.exists():
                    if path.is_file():
                        path.unlink(missing_ok=True)
                    else:
                        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        logging.error("Failed to fetch NIH BBox file: %s", exc)
        sys.exit(1)
