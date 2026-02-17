#!/usr/bin/env python
"""Generate attribution fingerprints for a given experiment."""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable

import argparse
from pathlib import Path


from xfp.config import load_experiment_config, load_paths_config
from xfp.fingerprint.runner import run_fingerprint_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attribution fingerprint experiment.")
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment key defined in configs/experiments.yaml.",
    )
    parser.add_argument(
        "--paths-config",
        default="configs/paths.yaml",
        help="Paths configuration YAML.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip validation of all paths/models in configs/paths.yaml.",
    )
    parser.add_argument(
        "--experiments-config",
        default="configs/experiments.yaml",
        help="Experiment manifest YAML.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for model inference.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        help="Override subset defined in experiment config.",
    )
    parser.add_argument(
        "--endpoint-mode",
        default="upper_bound_gt",
        choices=["upper_bound_gt", "predicted_mask", "mask_free"],
        help="Feature extraction endpoint mode. upper_bound_gt is analysis-only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths_cfg = load_paths_config(Path(args.paths_config), validate=not args.skip_validate)
    exp_cfg = load_experiment_config(Path(args.experiments_config), args.experiment)
    if args.subset:
        exp_cfg.subset = args.subset  # type: ignore[assignment]
    run_fingerprint_experiment(
        exp_cfg=exp_cfg,
        paths_cfg=paths_cfg,
        device=args.device,
        endpoint_mode=args.endpoint_mode,
    )


if __name__ == "__main__":
    main()
