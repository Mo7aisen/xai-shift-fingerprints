#!/usr/bin/env python
"""Create standardized metadata CSVs for each cohort."""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd



from xfp.config import load_paths_config

DEFAULT_COLUMNS = [
    "sample_id",
    "patient_id",
    "split",
    "projection",
    "site",
    "patient_age",
    "patient_sex",
    "acquisition_year",
    "notes",
]


@dataclass(frozen=True)
class DatasetSpec:
    projection: str
    site: str
    extra_candidates: List[str]


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "jsrt": DatasetSpec(
        projection="PA",
        site="JSRT",
        extra_candidates=[
            "metadata.csv",
            "jsrt_metadata.csv",
            "clinical_metadata.csv",
        ],
    ),
    "montgomery": DatasetSpec(
        projection="PA",
        site="Montgomery",
        extra_candidates=[
            "MontgomerySetMetadata.csv",
            "metadata.csv",
        ],
    ),
    "shenzhen": DatasetSpec(
        projection="PA",
        site="Shenzhen",
        extra_candidates=[
            "Shenzhen_Chest_X-ray_Set_metadata.csv",
            "metadata.csv",
        ],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset metadata tables.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_SPECS.keys()),
        help="Dataset key defined in configs/paths.yaml.",
    )
    parser.add_argument(
        "--config",
        default="configs/paths.yaml",
        help="Workspace paths configuration YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/metadata",
        help="Directory to store standardized metadata CSVs.",
    )
    parser.add_argument(
        "--extra-metadata",
        default=None,
        help="Optional path to an additional metadata CSV/TSV/XLSX file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing metadata CSV.",
    )
    return parser.parse_args()


def _dataset_root(images_path: Path) -> Path:
    """Return dataset root directory given an images directory."""
    return images_path.parent


def _load_splits(dataset_root: Path) -> pd.DataFrame:
    """Load split annotations if available; otherwise generate defaults."""
    candidates = [
        dataset_root / "splits.csv",
        dataset_root / "splits.tsv",
        dataset_root / "annotations" / "splits.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            df = pd.read_csv(candidate)
            if "patient_id" not in df.columns:
                raise KeyError(f"'patient_id' column missing in {candidate}")
            df["sample_id"] = df["patient_id"].astype(str)
            if "split" not in df.columns:
                df["split"] = "full"
            return df[["sample_id", "patient_id", "split"]]
    # Fallback: derive from image filenames
    image_dir = dataset_root / "CXR_png"
    if not image_dir.exists():
        image_dir = dataset_root / "images"
    image_files = sorted(p for p in image_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(
            f"No splits file found and no .png images to infer sample IDs in {dataset_root}"
        )
    records = [
        {"sample_id": path.stem, "patient_id": path.stem, "split": "full"}
        for path in image_files
    ]
    return pd.DataFrame.from_records(records)


def _load_extra_metadata(extra_path: Path) -> pd.DataFrame:
    """Load cohort-specific clinical metadata if provided."""
    if extra_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(extra_path)
    else:
        with extra_path.open("r", encoding="utf-8") as handle:
            sample = handle.read(1024)
            handle.seek(0)
            dialect = csv.Sniffer().sniff(sample) if "," not in sample else None
            df = pd.read_csv(handle, dialect=dialect)
    df.columns = [col.strip().lower() for col in df.columns]
    return df


def _normalise_extra_metadata(
    dataset: str,
    extra: pd.DataFrame,
) -> pd.DataFrame:
    """Map cohort-specific column names into the common schema."""
    mapping_candidates = [
        {
            "patient_id": "patient_id",
            "sample_id": "sample_id",
            "sex": "patient_sex",
            "gender": "patient_sex",
            "age": "patient_age",
            "projection": "projection",
            "viewposition": "projection",
            "site": "site",
            "hospital": "site",
            "year": "acquisition_year",
        }
    ]
    df = extra.rename(columns=mapping_candidates[0])
    if "sample_id" not in df.columns:
        if "patient_id" in df.columns:
            df["sample_id"] = df["patient_id"]
    if "sample_id" not in df.columns:
        raise KeyError(
            f"Unable to identify a 'sample_id' column in extra metadata for {dataset}."
        )
    return df


def _find_extra_metadata(dataset_root: Path, spec: DatasetSpec) -> Optional[Path]:
    for candidate in spec.extra_candidates:
        path = dataset_root / candidate
        if path.exists():
            return path
    return None


def _merge_metadata(
    base: pd.DataFrame,
    extra: Optional[pd.DataFrame],
    spec: DatasetSpec,
) -> pd.DataFrame:
    df = base.copy()
    df["projection"] = spec.projection
    df["site"] = spec.site
    df["patient_age"] = pd.NA
    df["patient_sex"] = pd.NA
    df["acquisition_year"] = pd.NA
    df["notes"] = pd.NA

    if extra is not None:
        extra = extra.copy()
        extra["sample_id"] = extra["sample_id"].astype(str)
        df = df.merge(extra, on="sample_id", how="left", suffixes=("", "_extra"))
        for column in ["patient_id", "patient_age", "patient_sex", "projection", "site", "acquisition_year"]:
            extra_column = f"{column}_extra"
            if extra_column in df.columns:
                df[column] = df[column].fillna(df[extra_column])
                df.drop(columns=[extra_column], inplace=True)
        if "notes_extra" in df.columns:
            df["notes"] = df["notes"].fillna(df["notes_extra"])
            df.drop(columns=["notes_extra"], inplace=True)

    missing_columns = [col for col in DEFAULT_COLUMNS if col not in df.columns]
    for column in missing_columns:
        df[column] = pd.NA

    df = df[DEFAULT_COLUMNS]
    df.sort_values("sample_id", inplace=True)
    return df


def main() -> None:
    args = parse_args()
    paths_cfg = load_paths_config(Path(args.config))

    dataset_entry = paths_cfg.datasets.get(args.dataset)
    if dataset_entry is None:
        raise KeyError(f"Dataset '{args.dataset}' not defined in {args.config}")

    dataset_root = _dataset_root(dataset_entry.images)
    spec = DATASET_SPECS[args.dataset]

    splits_df = _load_splits(dataset_root)

    extra_path = Path(args.extra_metadata) if args.extra_metadata else _find_extra_metadata(dataset_root, spec)
    extra_df = None
    if extra_path:
        extra_df = _normalise_extra_metadata(args.dataset, _load_extra_metadata(extra_path))
        print(f"[metadata] Loaded extra metadata from {extra_path}")
    else:
        print(f"[metadata] No extra metadata source found for {args.dataset}; using defaults.")

    combined = _merge_metadata(splits_df, extra_df, spec)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.dataset}_metadata.csv"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to replace it."
        )
    combined.to_csv(output_path, index=False)
    print(f"[metadata] Wrote {output_path} ({combined.shape[0]} rows)")


if __name__ == "__main__":
    main()
