"""Dataset-wide integrity check for patient_id in interim metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_all_metadata_have_patient_id_column_and_values() -> None:
    root = Path("data/interim")
    files = sorted(root.glob("*/**/metadata.parquet"))
    assert files, "No metadata.parquet files found under data/interim"

    missing_column = []
    missing_values = []

    for path in files:
        df = pd.read_parquet(path)
        if "patient_id" not in df.columns:
            missing_column.append(str(path))
            continue

        patient = df["patient_id"].astype("string")
        invalid_mask = patient.isna() | (patient.str.strip() == "")
        if bool(invalid_mask.any()):
            missing_values.append(f"{path} (missing={int(invalid_mask.sum())})")

    assert not missing_column, "Missing patient_id column in: " + ", ".join(missing_column)
    assert not missing_values, "Missing/empty patient_id values in: " + ", ".join(missing_values)
