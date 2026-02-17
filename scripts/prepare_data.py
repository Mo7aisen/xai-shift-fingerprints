#!/usr/bin/env python
"""Prepare harmonised datasets for attribution fingerprinting."""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable

import argparse
from pathlib import Path


from xfp.config import load_paths_config
from xfp.data.pipeline import build_dataset_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset caches.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["jsrt", "montgomery", "shenzhen", "nih_chestxray14"],
        help="Dataset key defined in configs/paths.yaml.",
    )
    parser.add_argument(
        "--subset",
        default="full",
        help="Optional subset label (e.g., pilot, site_a).",
    )
    parser.add_argument(
        "--config",
        default="configs/paths.yaml",
        help="Path to the workspace-level paths config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    paths_cfg = load_paths_config(config_path, validate=False)
    build_dataset_cache(dataset_key=args.dataset, subset=args.subset, paths_cfg=paths_cfg)


if __name__ == "__main__":
    main()
