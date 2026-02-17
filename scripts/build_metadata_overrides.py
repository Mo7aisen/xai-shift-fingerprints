#!/usr/bin/env python
"""Normalise raw clinical metadata into workspace overrides."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "metadata"


def _normalise_jsrt(raw_path: Path, output_path: Path) -> None:
    df = pd.read_csv(raw_path)
    if "study_id" not in df.columns:
        raise KeyError(f"'study_id' column missing from {raw_path}")

    df["sample_id"] = df["study_id"].str.replace(".png", "", case=False)
    df["pixel_spacing_x_mm"] = 0.175
    df["pixel_spacing_y_mm"] = 0.175
    df["projection"] = "PA"
    df["site"] = "JSRT"
    df["patient_age"] = pd.to_numeric(df.get("age"), errors="coerce")
    df["patient_sex"] = df.get("gender", "").str.title()

    extras = {
        "subtlety": "nodule_subtlety",
        "size": "nodule_size_mm",
        "state": "diagnosis_state",
        "position": "lesion_position",
        "diagnosis": "diagnosis_notes",
        "x": "nodule_center_x",
        "y": "nodule_center_y",
    }
    for src, dst in extras.items():
        if src in df.columns:
            df[dst] = df[src]

    keep_cols = [
        "sample_id",
        "pixel_spacing_x_mm",
        "pixel_spacing_y_mm",
        "projection",
        "site",
        "patient_age",
        "patient_sex",
        "nodule_subtlety",
        "nodule_size_mm",
        "diagnosis_state",
        "lesion_position",
        "diagnosis_notes",
        "nodule_center_x",
        "nodule_center_y",
    ]
    available = [col for col in keep_cols if col in df.columns]
    df_out = df[available].sort_values("sample_id")
    df_out.to_csv(output_path, index=False)
    print(f"✓ JSRT metadata → {output_path}")


def _parse_montgomery_clinical(clinical_dir: Path) -> pd.DataFrame:
    records = []
    pattern = re.compile(r"(\d+)")
    for path in sorted(clinical_dir.glob("MCUCXR_*.txt")):
        content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        lines = [line.strip() for line in content if line.strip()]
        sex: Optional[str] = None
        age: Optional[float] = None
        findings: Optional[str] = None
        for line in lines:
            lower = line.lower()
            if "patient's sex" in lower:
                value = line.split(":", 1)[-1].strip()
                sex = value.title() if value else None
            elif "patient's age" in lower:
                match = pattern.search(line)
                if match:
                    age = float(match.group(1))
            else:
                findings = line
        sample_id = path.stem
        records.append({
            "sample_id": sample_id,
            "patient_age_clinical": age,
            "patient_sex_clinical": sex,
            "clinical_findings": findings,
        })
    return pd.DataFrame(records)


def _normalise_montgomery(raw_path: Path, clinical_dir: Path, output_path: Path) -> None:
    if raw_path.exists():
        df_raw = pd.read_csv(raw_path)
        if "study_id" not in df_raw.columns:
            raise KeyError(f"'study_id' column missing from {raw_path}")
        df_raw["sample_id"] = df_raw["study_id"].str.replace(".png", "", case=False)
    else:
        df_raw = pd.DataFrame(columns=["sample_id"])

    df_clinical = _parse_montgomery_clinical(clinical_dir) if clinical_dir.exists() else pd.DataFrame(columns=["sample_id"])

    df = pd.merge(df_raw, df_clinical, on="sample_id", how="outer")

    df["pixel_spacing_x_mm"] = 0.0875
    df["pixel_spacing_y_mm"] = 0.0875
    df["projection"] = "PA"
    df["site"] = "Montgomery"

    age_cols = [col for col in ["age", "patient_age", "patient_age_clinical"] if col in df.columns]
    if age_cols:
        age_df = df[age_cols].apply(pd.to_numeric, errors="coerce")
        df["patient_age"] = age_df.bfill(axis=1).iloc[:, 0]
    else:
        df["patient_age"] = pd.NA

    sex_cols = [col for col in ["gender", "patient_sex", "patient_sex_clinical"] if col in df.columns]
    if sex_cols:
        sex_series = df[sex_cols].apply(lambda col: col.astype(str).where(col.notna()))
        df["patient_sex"] = sex_series.bfill(axis=1).iloc[:, 0].str.title().replace({'F': 'Female', 'M': 'Male'})
    else:
        df["patient_sex"] = pd.NA

    if "findings" in df.columns:
        df["findings"] = df["findings"].fillna(df.get("clinical_findings"))
    else:
        df["findings"] = df.get("clinical_findings")

    keep_cols = [
        "sample_id",
        "pixel_spacing_x_mm",
        "pixel_spacing_y_mm",
        "projection",
        "site",
        "patient_age",
        "patient_sex",
        "findings",
        "clinical_findings",
    ]
    available = [col for col in keep_cols if col in df.columns]
    df_out = df[available].sort_values("sample_id")
    df_out.to_csv(output_path, index=False)
    print(f"✓ Montgomery metadata → {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build metadata override CSVs from raw clinical files.")
    parser.add_argument(
        "--jsrt-raw",
        default=RAW_DIR / "jsrt_metadata.csv",
        type=Path,
        help="Source CSV for JSRT clinical metadata.",
    )
    parser.add_argument(
        "--montgomery-raw",
        default=RAW_DIR / "montgomery_metadata.csv",
        type=Path,
        help="Source CSV for Montgomery clinical metadata (optional).",
    )
    parser.add_argument(
        "--montgomery-clinical-dir",
        default=RAW_DIR / "montgomery_clinical",
        type=Path,
        help="Directory containing per-study Montgomery clinical readings (txt files).",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        type=Path,
        help="Directory to write normalised metadata overrides.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.jsrt_raw.exists():
        raise FileNotFoundError(f"JSRT raw metadata not found: {args.jsrt_raw}")
    _normalise_jsrt(args.jsrt_raw, args.output_dir / "jsrt_metadata.csv")

    _normalise_montgomery(args.montgomery_raw, args.montgomery_clinical_dir, args.output_dir / "montgomery_metadata.csv")


if __name__ == "__main__":
    main()
