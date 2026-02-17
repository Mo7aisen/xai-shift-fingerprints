#!/usr/bin/env python3
"""Backfill patient_id in cached metadata parquet files.

Non-destructive by default:
- creates timestamped backups before writing
- writes updated parquet atomically via a temporary file then replace
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

import pandas as pd

import _path_setup  # noqa: F401
from xfp.data import infer_patient_id


@dataclass
class BackfillResult:
    metadata_path: Path
    backup_path: Path | None
    rows: int
    missing_before: int
    missing_after: int
    changed: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill patient_id in metadata.parquet files under cache root.")
    p.add_argument("--cache-root", type=Path, default=Path("data/interim"))
    p.add_argument("--reports-dir", type=Path, default=Path("reports_v2/audits"))
    p.add_argument("--dry-run", action="store_true", help="Inspect only; do not write changes.")
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="Disable backup creation (not recommended).",
    )
    return p.parse_args()


def infer_dataset_key_from_path(metadata_path: Path) -> str:
    # expected layout: data/interim/<dataset>/<subset>/metadata.parquet
    parts = metadata_path.parts
    if "interim" in parts:
        idx = parts.index("interim")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return metadata_path.parent.parent.name


def backfill_file(metadata_path: Path, *, dry_run: bool, backup: bool) -> BackfillResult:
    df = pd.read_parquet(metadata_path)
    rows = len(df)
    if rows == 0:
        return BackfillResult(metadata_path, None, 0, 0, 0, False)

    if "sample_id" not in df.columns:
        raise ValueError(f"Missing sample_id column: {metadata_path}")

    dataset_key = (
        str(df["dataset_key"].iloc[0])
        if "dataset_key" in df.columns and df["dataset_key"].notna().any()
        else infer_dataset_key_from_path(metadata_path)
    )

    if "patient_id" not in df.columns:
        df["patient_id"] = pd.NA

    before_missing = int(df["patient_id"].isna().sum() + (df["patient_id"].astype(str).str.strip() == "").sum())

    inferred = df["sample_id"].astype(str).map(lambda sid: infer_patient_id(dataset_key, sid))
    patient_series = df["patient_id"].astype("string")
    fill_mask = patient_series.isna() | (patient_series.str.strip() == "")
    df.loc[fill_mask, "patient_id"] = inferred[fill_mask]

    after_missing = int(df["patient_id"].isna().sum() + (df["patient_id"].astype(str).str.strip() == "").sum())
    changed = before_missing != after_missing or "patient_id" not in pd.read_parquet(metadata_path).columns

    backup_path: Path | None = None
    if changed and not dry_run:
        if backup:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_path = metadata_path.with_name(f"{metadata_path.name}.bak.{ts}")
            shutil.copy2(metadata_path, backup_path)

        with tempfile.NamedTemporaryFile(prefix="metadata_backfill_", suffix=".parquet", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            df.to_parquet(tmp_path, index=False)
            tmp_path.replace(metadata_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    return BackfillResult(metadata_path, backup_path, rows, before_missing, after_missing, changed)


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root
    reports_dir = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(cache_root.glob("*/**/metadata.parquet"))
    if not files:
        raise FileNotFoundError(f"No metadata.parquet files found under {cache_root}")

    results: list[BackfillResult] = []
    for path in files:
        result = backfill_file(path, dry_run=args.dry_run, backup=not args.no_backup)
        results.append(result)

    now_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = reports_dir / f"backfill_patient_id_{now_tag}.csv"
    json_path = reports_dir / f"backfill_patient_id_{now_tag}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "metadata_path",
                "backup_path",
                "rows",
                "missing_before",
                "missing_after",
                "changed",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "metadata_path": str(r.metadata_path),
                    "backup_path": str(r.backup_path) if r.backup_path else "",
                    "rows": r.rows,
                    "missing_before": r.missing_before,
                    "missing_after": r.missing_after,
                    "changed": r.changed,
                }
            )

    summary = {
        "cache_root": str(cache_root),
        "dry_run": args.dry_run,
        "backup_enabled": not args.no_backup,
        "files_scanned": len(results),
        "files_changed": int(sum(1 for r in results if r.changed)),
        "total_rows": int(sum(r.rows for r in results)),
        "total_missing_before": int(sum(r.missing_before for r in results)),
        "total_missing_after": int(sum(r.missing_after for r in results)),
        "csv_report": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
