#!/usr/bin/env python3
"""
NIH ChestXray14 Preprocessing Harness
=====================================

Purpose & Output
----------------
This script documents and automates the local steps required to ingest the NIH
ChestXray14 dataset into the attribution-fingerprint pipeline. It performs the
following reproducible actions:

1. Loads `configs/paths.yaml` to determine dataset locations.
2. Verifies that the raw download archives (`images_001.tar` … `images_012.tar`,
   `Data_Entry_2017.csv`, etc.) exist under `${datasets_root}/NIH_ChestXray14/raw`.
3. Confirms that images have been extracted under `.../images/` (if available)
   and reports how many files are currently present.
4. Creates a metadata manifest (Parquet + CSV) summarizing patient IDs, study IDs,
   and label counts from `Data_Entry_2017.csv`.

The script is intentionally idempotent and purely preparatory: it does not attempt
to download NIH data (the DUA forbids automated scraping) but provides clear
logging so we can trace the onboarding status.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"
OUTPUT_MANIFEST = PROJECT_ROOT / "data" / "nih_chestxray14_manifest.parquet"
OUTPUT_MANIFEST_CSV = PROJECT_ROOT / "data" / "nih_chestxray14_manifest.csv"

LOGGER = logging.getLogger("nih_preprocess")


def load_paths(config_path: Path) -> Dict[str, str]:
    """Load dataset paths from configs/paths.yaml."""
    LOGGER.info("Loading configuration from %s", config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    datasets_root = Path(config["paths"]["datasets_root"])
    nih_cfg = config["datasets"].get("nih_chestxray14")
    if nih_cfg is None:
        raise KeyError("Add `nih_chestxray14` entry to configs/paths.yaml under datasets.")

    dataset_paths = {
        "datasets_root": str(datasets_root),
        "raw_dir": str(datasets_root / "NIH_ChestXray14" / "raw"),
        "images_dir": str(datasets_root / Path(nih_cfg["images"])),
        "masks_dir": str(datasets_root / Path(nih_cfg["masks"])),
        "metadata_dir": str(datasets_root / Path(nih_cfg.get("metadata", "NIH_ChestXray14/metadata"))),
    }
    LOGGER.debug("Resolved dataset paths: %s", dataset_paths)
    return dataset_paths


def check_required_archives(raw_dir: Path) -> Dict[str, bool]:
    """Return existence flags for expected NIH archives."""
    tar_entries = {}
    for i in range(1, 13):
        stem = f"images_{i:03d}"
        tar_entries[f"{stem}.tar(.gz)"] = (raw_dir / f"{stem}.tar").exists() or (raw_dir / f"{stem}.tar.gz").exists()
    other_files = ["Data_Entry_2017.csv", "BBox_List_2017.csv"]
    records: Dict[str, bool] = {}
    records.update(tar_entries)
    for name in other_files:
        records[name] = (raw_dir / name).exists()
    return records


def summarize_images(images_dir: Path) -> int:
    """Count extracted PNG images if the directory exists."""
    if not images_dir.exists():
        LOGGER.warning("Images directory %s does not exist yet.", images_dir)
        return 0
    num_files = sum(1 for _ in images_dir.rglob("*.png"))
    LOGGER.info("Found %d PNG files under %s", num_files, images_dir)
    return num_files


def load_metadata_csv(raw_dir: Path) -> Optional[pd.DataFrame]:
    """Load Data_Entry_2017.csv if present."""
    csv_candidates = [
        raw_dir / "Data_Entry_2017.csv",
        raw_dir / "Data_Entry_2017_v2020.csv",
    ]
    csv_path = next((path for path in csv_candidates if path.exists()), None)
    if csv_path is None:
        LOGGER.warning("Metadata CSV not found (expected one of %s); manifest skipped.", csv_candidates)
        return None
    LOGGER.info("Loading NIH metadata CSV from %s", csv_path)
    df = pd.read_csv(csv_path)
    return df


def load_bbox_csv(raw_dir: Path) -> Optional[pd.DataFrame]:
    """Load BBox_List_2017.csv and normalise column names."""
    bbox_path = raw_dir / "BBox_List_2017.csv"
    if not bbox_path.exists():
        LOGGER.warning("Bounding-box CSV %s not found; bbox metadata skipped.", bbox_path)
        return None
    LOGGER.info("Loading NIH bounding boxes from %s", bbox_path)
    bbox_df = pd.read_csv(bbox_path)
    rename_map = {
        "Bbox [x": "bbox_x",
        "y": "bbox_y",
        "w": "bbox_w",
        "h]": "bbox_h",
    }
    bbox_df = bbox_df.rename(columns=rename_map)
    required_cols = ["Image Index", "Finding Label", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    missing = [col for col in required_cols if col not in bbox_df.columns]
    if missing:
        LOGGER.error("BBox CSV missing columns: %s", ", ".join(missing))
        return None
    bbox_df = bbox_df[required_cols].dropna(subset=["bbox_x", "bbox_y", "bbox_w", "bbox_h"])
    for col in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]:
        bbox_df[col] = bbox_df[col].astype(float)
    return bbox_df


def parse_original_dims(df: pd.DataFrame) -> pd.DataFrame:
    """Extract original image width/height regardless of CSV column naming."""
    widths: pd.Series
    heights: pd.Series
    if "OriginalImage[Width,Height]" in df.columns:
        raw_series = df["OriginalImage[Width,Height]"].fillna("0,0")
        width_vals: List[float] = []
        height_vals: List[float] = []
        for value in raw_series:
            try:
                stripped = str(value).replace("[", "").replace("]", "")
                width_str, height_str = stripped.split(",")
                width_vals.append(float(width_str))
                height_vals.append(float(height_str))
            except Exception:  # pragma: no cover
                width_vals.append(float("nan"))
                height_vals.append(float("nan"))
        widths = pd.Series(width_vals)
        heights = pd.Series(height_vals)
    else:
        width_col = pd.to_numeric(df.get("OriginalImage[Width", pd.Series(dtype=float)), errors="coerce")
        height_col = pd.to_numeric(df.get("Height]", pd.Series(dtype=float)), errors="coerce")
        widths = width_col
        heights = height_col
    return pd.DataFrame({"Original Width": widths, "Original Height": heights})


def aggregate_bboxes(bbox_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bounding boxes per image index."""

    def to_records(group: pd.DataFrame) -> List[Dict[str, float]]:
        return [
            {
                "label": row["Finding Label"],
                "x": row["bbox_x"],
                "y": row["bbox_y"],
                "w": row["bbox_w"],
                "h": row["bbox_h"],
            }
            for _, row in group.iterrows()
        ]

    group_cols = ["Finding Label", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    record_series = (
        bbox_df.groupby("Image Index", group_keys=False)[group_cols].apply(to_records).rename("bbox_records")
    )
    grouped = (
        bbox_df.groupby("Image Index")
        .agg(
            bbox_count=("bbox_x", "count"),
            bbox_labels=("Finding Label", lambda labels: "|".join(sorted(set(labels)))),
        )
        .join(record_series)
        .reset_index()
    )
    grouped["bbox_records"] = grouped["bbox_records"].apply(lambda records: json.dumps(records, ensure_ascii=False))
    return grouped


def build_manifest(df: pd.DataFrame, bbox_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Compute per-study summary statistics."""
    LOGGER.info("Building manifest from %d entries", len(df))
    labels = df["Finding Labels"].str.split("|")
    dims = parse_original_dims(df)
    manifest = pd.DataFrame(
        {
            "Image Index": df["Image Index"],
            "Patient ID": df["Patient ID"],
            "Study Date": df["Follow-up #"],
            "View Position": df["View Position"],
            "Finding Count": labels.apply(lambda x: 0 if x == [""] else len(x)),
            "Finding Labels": df["Finding Labels"],
            "Patient Age": pd.to_numeric(df["Patient Age"], errors="coerce"),
            "Patient Sex": df["Patient Sex"],
        }
    )
    manifest["Original Width"] = dims["Original Width"]
    manifest["Original Height"] = dims["Original Height"]
    manifest["Has BBox"] = False
    manifest["BBox Count"] = 0
    manifest["BBox Labels"] = ""
    manifest["BBox Records"] = "[]"
    if bbox_df is not None and not bbox_df.empty:
        agg = aggregate_bboxes(bbox_df)
        manifest = manifest.merge(agg, on="Image Index", how="left")
        manifest["Has BBox"] = manifest["bbox_count"].fillna(0).astype(int) > 0
        manifest["BBox Count"] = manifest["bbox_count"].fillna(0).astype(int)
        manifest["BBox Labels"] = manifest["bbox_labels"].fillna("")
        manifest["BBox Records"] = manifest["bbox_records"].fillna("[]")
        manifest = manifest.drop(columns=["bbox_count", "bbox_labels", "bbox_records"])
    return manifest


def save_manifest(manifest: pd.DataFrame) -> None:
    """Persist manifest to Parquet and CSV."""
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(OUTPUT_MANIFEST, index=False)
    manifest.to_csv(OUTPUT_MANIFEST_CSV, index=False)
    LOGGER.info("Saved manifest to %s and %s", OUTPUT_MANIFEST, OUTPUT_MANIFEST_CSV)


def main() -> None:
    """Entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        paths = load_paths(CONFIG_PATH)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load configuration: %s", exc)
        sys.exit(1)

    raw_dir = Path(paths["raw_dir"])
    images_dir = Path(paths["images_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    archive_status = check_required_archives(raw_dir)
    missing_archives = [name for name, exists in archive_status.items() if not exists]
    if missing_archives:
        LOGGER.warning("Missing %d raw files: %s", len(missing_archives), ", ".join(missing_archives))
    else:
        LOGGER.info("All expected NIH tar archives detected.")

    summarize_images(images_dir)

    metadata_df = load_metadata_csv(raw_dir)
    bbox_df = load_bbox_csv(raw_dir)
    if metadata_df is not None:
        manifest = build_manifest(metadata_df, bbox_df)
        save_manifest(manifest)
    else:
        LOGGER.warning("Manifest not generated because metadata CSV was unavailable.")

    LOGGER.info("NIH ChestXray14 preprocessing check completed.")


if __name__ == "__main__":
    main()
