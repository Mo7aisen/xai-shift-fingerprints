#!/usr/bin/env python
"""Ensure every cached sample has metadata."""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable

import argparse
from pathlib import Path

import pandas as pd


from xfp.config import load_paths_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate metadata coverage.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["jsrt", "montgomery", "shenzhen"],
        help="Dataset key to validate.",
    )
    parser.add_argument(
        "--subset",
        default="full",
        help="Cache subset to inspect.",
    )
    parser.add_argument(
        "--config",
        default="configs/paths.yaml",
        help="Paths configuration YAML.",
    )
    parser.add_argument(
        "--metadata-dir",
        default="data/metadata",
        help="Directory containing standardized metadata CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths_cfg = load_paths_config(Path(args.config))

    cache_dir = paths_cfg.cache_root / args.dataset / args.subset
    metadata_path = cache_dir / "metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cache metadata missing at {metadata_path}")

    cache_df = pd.read_parquet(metadata_path, columns=["sample_id"])
    csv_path = Path(args.metadata_dir) / f"{args.dataset}_metadata.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected metadata CSV at {csv_path}")

    meta_df = pd.read_csv(csv_path)

    cache_ids = set(cache_df["sample_id"].astype(str))
    csv_ids = set(meta_df["sample_id"].astype(str))

    missing = sorted(cache_ids - csv_ids)
    extra = sorted(csv_ids - cache_ids)

    print(f"[validation] {args.dataset}:{args.subset} cache rows: {len(cache_ids)}")
    print(f"[validation] {args.dataset} metadata rows: {len(csv_ids)}")

    if missing:
        print(f"[validation][ERROR] Missing {len(missing)} metadata rows, e.g., {missing[:5]}")
    else:
        print("[validation] All cached samples have metadata entries ✅")

    if extra:
        print(f"[validation][WARN] Metadata CSV contains {len(extra)} unused entries, e.g., {extra[:5]}")


if __name__ == "__main__":
    main()
