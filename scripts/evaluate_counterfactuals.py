#!/usr/bin/env python
"""Evaluate attribution deltas for generated counterfactuals."""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch



from xfp.attribution import compute_integrated_gradients
from xfp.config import load_paths_config
from xfp.counterfactuals.metrics import compute_basic_metrics, metrics_as_dict
from xfp.models import load_unet_checkpoint

MODEL_KEY_MAP: Dict[str, str] = {
    "jsrt": "unet_jsrt_full",
    "montgomery": "unet_jsrt_full",
    "shenzhen": "unet_jsrt_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute attribution deltas for counterfactual samples.")
    parser.add_argument("--experiment", required=True, help="Counterfactual experiment folder name.")
    parser.add_argument("--paths-config", default="configs/paths.yaml", help="Project paths configuration file.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for attribution computation.")
    parser.add_argument("--ig-steps", type=int, default=8, help="Integrated Gradients steps (lower for speed, higher for fidelity).")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick runs.")
    return parser.parse_args()


def load_counterfactual_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def main() -> None:
    args = parse_args()
    paths_cfg = load_paths_config(Path(args.paths_config))

    experiment_dir = Path("data/counterfactuals") / args.experiment
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Counterfactual directory not found: {experiment_dir}")

    device = torch.device(args.device)

    npz_files = sorted(experiment_dir.glob("*.npz"))
    if args.limit is not None:
        npz_files = npz_files[: args.limit]

    rows = []
    model_cache: Dict[str, torch.nn.Module] = {}

    for npz_path in npz_files:
        sample = load_counterfactual_npz(npz_path)
        dataset_key = str(sample.get("dataset_key", args.experiment.split("_")[0]))
        model_key = MODEL_KEY_MAP.get(dataset_key, "unet_jsrt_full")
        model_path = paths_cfg.models.get(model_key)
        if model_path is None:
            raise KeyError(f"Model '{model_key}' not configured in paths.yaml")

        if model_key not in model_cache:
            model_cache[model_key] = load_unet_checkpoint(model_path, device=args.device)
        model = model_cache[model_key]

        original_image = sample["original_image"].astype(np.float32)
        perturbed_image = sample["perturbed_image"].astype(np.float32)
        mask = sample["original_mask"].astype(np.uint8)

        orig_tensor = torch.from_numpy(original_image).unsqueeze(0).unsqueeze(0).to(device)
        pert_tensor = torch.from_numpy(perturbed_image).unsqueeze(0).unsqueeze(0).to(device)

        orig_attr = compute_integrated_gradients(model, orig_tensor, n_steps=args.ig_steps).detach().cpu().numpy()[0, 0]
        pert_attr = compute_integrated_gradients(model, pert_tensor, n_steps=args.ig_steps).detach().cpu().numpy()[0, 0]

        metrics = compute_basic_metrics(orig_attr, pert_attr, mask)
        row = {
            "sample_id": npz_path.stem,
            "dataset_key": dataset_key,
            **metrics_as_dict(metrics),
        }
        rows.append(row)

    if not rows:
        print("[evaluate_counterfactuals] No samples processed.")
        return

    df = pd.DataFrame(rows)
    output_path = experiment_dir / "metrics.parquet"
    df.to_parquet(output_path, index=False)
    print(f"[evaluate_counterfactuals] wrote metrics for {len(rows)} samples to {output_path}")


if __name__ == "__main__":
    main()
