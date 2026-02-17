#!/usr/bin/env python
"""CLI tool to generate synthetic counterfactual cohorts."""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import yaml



from xfp.config import load_paths_config
from xfp.counterfactuals.dataset import iterate_cache
from xfp.counterfactuals.perturbations import (
    dilate_mask,
    erode_mask,
    insert_gaussian_nodule,
    apply_intensity_offset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic counterfactual cohorts.")
    parser.add_argument("--experiment", required=True, help="Experiment key in configs/counterfactuals.yaml")
    parser.add_argument(
        "--counterfactual-config",
        default="configs/counterfactuals.yaml",
        help="YAML describing perturbation experiments.",
    )
    parser.add_argument(
        "--paths-config",
        default="configs/paths.yaml",
        help="Project paths configuration file.",
    )
    return parser.parse_args()


def load_counterfactual_config(path: Path, experiment_key: str) -> Dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = data.get("defaults", {})
    experiment = data.get("experiments", {}).get(experiment_key)
    if experiment is None:
        raise KeyError(f"Experiment '{experiment_key}' not found in {path}")

    merged = {**defaults, **experiment}
    merged.setdefault("output_root", "data/counterfactuals")
    return merged


def resolve_perturbation(spec: Dict[str, object]) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, object]]]:
    name = spec.get("name")
    params = spec.get("params", {}) or {}

    if name == "dilate":
        radius = int(params.get("radius", 4))

        def perturb(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
            new_mask = dilate_mask(mask, radius=radius)
            return image.copy(), new_mask, {"type": "dilate", "radius": radius}

        return perturb

    if name == "erode":
        radius = int(params.get("radius", 4))

        def perturb(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
            new_mask = erode_mask(mask, radius=radius)
            return image.copy(), new_mask, {"type": "erode", "radius": radius}

        return perturb

    if name == "gaussian_nodule":
        radius = int(params.get("radius", 12))
        intensity = float(params.get("intensity", 0.35))

        def perturb(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
            perturbed_image, new_mask, meta = insert_gaussian_nodule(image, mask, radius=radius, intensity=intensity)
            return perturbed_image, new_mask, meta

        return perturb

    if name == "intensity_offset":
        offset = float(params.get("offset", 0.1))

        def perturb(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
            perturbed_image = apply_intensity_offset(image, offset=offset)
            return perturbed_image, mask.copy(), {"type": "intensity_offset", "offset": offset}

        return perturb

    raise ValueError(f"Unsupported perturbation type: {name}")


def main() -> None:
    args = parse_args()
    counter_cfg = load_counterfactual_config(Path(args.counterfactual_config), args.experiment)
    paths_cfg = load_paths_config(Path(args.paths_config))

    dataset_key = counter_cfg["dataset_key"]
    subset = counter_cfg.get("subset", "full")
    perturb_spec = counter_cfg.get("perturbation", {})
    limit = counter_cfg.get("limit")

    cache_dir = paths_cfg.cache_root / dataset_key / subset
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory missing: {cache_dir}. Run scripts/prepare_data.py first.")

    perturb_fn = resolve_perturbation(perturb_spec)

    output_root = Path(counter_cfg.get("output_root", "data/counterfactuals")).expanduser()
    output_dir = output_root / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_records = []

    for sample in iterate_cache(cache_dir, perturb_fn, limit=int(limit) if limit else None):
        npz_path = output_dir / f"{sample.sample_id}.npz"
        np.savez_compressed(
            npz_path,
            original_image=sample.original_image,
            original_mask=sample.original_mask,
            perturbed_image=sample.perturbed_image,
            perturbed_mask=sample.perturbed_mask,
            dataset_key=sample.dataset_key,
            perturbation_metadata=sample.perturbation_metadata,
        )
        record = {
            "sample_id": sample.sample_id,
            "dataset_key": sample.dataset_key,
            **sample.perturbation_metadata,
        }
        metadata_records.append(record)

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata_records, indent=2), encoding="utf-8")
    print(f"[generate_counterfactuals] wrote {len(metadata_records)} samples to {output_dir}")


if __name__ == "__main__":
    main()
