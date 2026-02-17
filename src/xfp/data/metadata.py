"""Dataset metadata utilities for attribution fingerprinting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


METADATA_VERSION = "1.0"


@dataclass(frozen=True)
class DatasetDefaults:
    projection: str
    site: str
    pixel_spacing_x_mm: float | None = None
    pixel_spacing_y_mm: float | None = None


DATASET_DEFAULTS: Mapping[str, DatasetDefaults] = {
    "jsrt": DatasetDefaults(projection="PA", site="JSRT", pixel_spacing_x_mm=0.175, pixel_spacing_y_mm=0.175),
    "montgomery": DatasetDefaults(projection="PA", site="Montgomery", pixel_spacing_x_mm=0.175, pixel_spacing_y_mm=0.175),
    "shenzhen": DatasetDefaults(projection="PA", site="Shenzhen"),
    "nih_chestxray14": DatasetDefaults(projection="PA", site="NIH"),
}


def get_metadata_path(dataset: str, metadata_root: Path | str = Path("data/metadata")) -> Path:
    """Return the expected metadata CSV path for a dataset."""
    root = Path(metadata_root)
    return root / f"{dataset}_metadata.csv"


def load_dataset_metadata(
    dataset: str,
    sample_ids: Iterable[str],
    *,
    metadata_root: Path | str = Path("data/metadata"),
) -> pd.DataFrame:
    """Load dataset-level metadata and align to provided sample IDs.

    If a metadata CSV exists, its values override defaults. Otherwise defaults are used.
    """
    dataset_key = dataset.lower()
    defaults = DATASET_DEFAULTS.get(dataset_key, DatasetDefaults(projection="PA", site=dataset_key.upper()))

    sample_list = [str(sample_id) for sample_id in sample_ids]
    base = pd.DataFrame({"sample_id": sample_list})
    base["projection"] = defaults.projection
    base["site"] = defaults.site
    base["patient_age"] = pd.NA
    base["patient_sex"] = pd.NA
    base["pixel_spacing_x_mm"] = defaults.pixel_spacing_x_mm
    base["pixel_spacing_y_mm"] = defaults.pixel_spacing_y_mm

    metadata_path = get_metadata_path(dataset_key, metadata_root)
    if metadata_path.exists():
        metadata_df = pd.read_csv(metadata_path)
        metadata_df["sample_id"] = metadata_df["sample_id"].astype(str)
        merged = base.merge(metadata_df, on="sample_id", how="left", suffixes=("", "_meta"))
        for column in metadata_df.columns:
            if column == "sample_id":
                continue
            meta_column = f"{column}_meta" if f"{column}_meta" in merged.columns else column
            if meta_column in merged.columns and meta_column != column:
                merged[column] = merged[meta_column].combine_first(merged[column])
                merged.drop(columns=[meta_column], inplace=True)
        return merged

    return base


def derive_age_bucket(series: pd.Series) -> pd.Series:
    """Bucket patient ages into common clinical ranges."""
    ages = pd.to_numeric(series, errors="coerce")
    bins = [-float("inf"), 30, 50, 70, float("inf")]
    labels = ["≤30", "31-50", "51-70", "70+"]
    return pd.cut(ages, bins=bins, labels=labels, right=True)


def summarize_metadata(df: pd.DataFrame) -> dict:
    """Return a lightweight summary for reporting and QA."""
    return {
        "rows": int(len(df)),
        "missing_patient_age": int(df.get("patient_age", pd.Series(dtype="float")).isna().sum()),
        "missing_patient_sex": int(df.get("patient_sex", pd.Series(dtype="object")).isna().sum()),
        "sites": sorted(df.get("site", pd.Series(dtype="object")).dropna().unique().tolist()),
        "metadata_version": METADATA_VERSION,
    }
