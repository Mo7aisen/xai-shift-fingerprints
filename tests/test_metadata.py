import pandas as pd

from xfp.data.metadata import load_dataset_metadata, derive_age_bucket


def test_load_dataset_metadata_defaults(tmp_path):
    sample_ids = ["sample_a", "sample_b"]
    df = load_dataset_metadata("jsrt", sample_ids, metadata_root=tmp_path)
    assert list(df["sample_id"]) == sample_ids
    assert df.loc[0, "pixel_spacing_x_mm"] == 0.175
    assert df.loc[0, "projection"] == "PA"
    assert df["site"].unique().tolist() == ["JSRT"]


def test_load_dataset_metadata_overrides(tmp_path):
    override = pd.DataFrame({
        "sample_id": ["sample_a"],
        "projection": ["AP"],
        "patient_age": [45],
        "patient_sex": ["F"],
    })
    override.to_csv(tmp_path / "montgomery_metadata.csv", index=False)

    df = load_dataset_metadata("montgomery", ["sample_a", "sample_b"], metadata_root=tmp_path)
    row_a = df[df["sample_id"] == "sample_a"].iloc[0]
    assert row_a["projection"] == "AP"
    assert row_a["patient_age"] == 45
    assert row_a["patient_sex"] == "F"

    row_b = df[df["sample_id"] == "sample_b"].iloc[0]
    # Default projection carries through for samples without overrides
    assert row_b["projection"] == "PA"
    assert pd.isna(row_b["patient_sex"])


def test_derive_age_bucket_handles_missing():
    series = pd.Series([10, 35, 67, 82, pd.NA], dtype="Float64")
    buckets = derive_age_bucket(series)
    assert buckets.iloc[0] == "≤30"
    assert buckets.iloc[1] == "31-50"
    assert buckets.iloc[2] == "51-70"
    assert buckets.iloc[3] == "70+"
    assert pd.isna(buckets.iloc[4])
