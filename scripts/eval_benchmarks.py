#!/usr/bin/env python
"""Run benchmark suite comparing attribution shift scores against segmentation quality."""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable

import argparse
from pathlib import Path


from xfp.config import load_experiment_config, load_paths_config
from xfp.shift.benchmark import run_benchmark_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute predefined shift detection benchmarks.")
    parser.add_argument(
        "--experiment",
        default="jsrt_to_montgomery",
        help="Experiment key from configs/experiments.yaml.",
    )
    parser.add_argument(
        "--paths-config",
        default="configs/paths.yaml",
        help="Paths configuration YAML.",
    )
    parser.add_argument(
        "--experiments-config",
        default="configs/experiments.yaml",
        help="Experiment manifest YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to store benchmark tables and figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths_cfg = load_paths_config(Path(args.paths_config))
    exp_cfg = load_experiment_config(Path(args.experiments_config), args.experiment)
    run_benchmark_suite(exp_cfg=exp_cfg, paths_cfg=paths_cfg, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
