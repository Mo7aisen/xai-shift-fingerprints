#!/usr/bin/env python
"""Export qualitative triptych panels for counterfactual samples."""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable


import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from typing import Dict

import numpy as np
import torch



from xfp.attribution import compute_integrated_gradients  # noqa: E402
from xfp.config import load_paths_config  # noqa: E402
from xfp.counterfactuals.visualization import save_triptych  # noqa: E402
from xfp.models import load_unet_checkpoint  # noqa: E402

MODEL_KEY_MAP: Dict[str, str] = {
    "jsrt": "unet_jsrt_full",
    "montgomery": "unet_jsrt_full",
    "shenzhen": "unet_jsrt_full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export counterfactual triptych visualisations.")
    parser.add_argument("--experiment", required=True, help="Counterfactual experiment folder name.")
    parser.add_argument("--sample-id", default=None, help="Specific sample id to export (defaults to first available).")
    parser.add_argument("--paths-config", default="configs/paths.yaml", help="Paths configuration YAML.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for attribution.")
    parser.add_argument("--ig-steps", type=int, default=4, help="Integrated Gradients steps.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "counterfactuals",
        help="Directory to write triptych PNGs.",
    )
    return parser.parse_args()


def load_counterfactual_npz(experiment_dir: Path, sample_id: str | None) -> tuple[Path, Dict[str, np.ndarray]]:
    if sample_id:
        npz_path = experiment_dir / f"{sample_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Sample '{sample_id}' not found in {experiment_dir}")
        return npz_path, dict(np.load(npz_path, allow_pickle=True))

    npz_files = sorted(experiment_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No counterfactual samples found in {experiment_dir}")
    npz_path = npz_files[0]
    return npz_path, dict(np.load(npz_path, allow_pickle=True))


def compute_attribution(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    *,
    n_steps: int,
) -> np.ndarray:
    tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    attr = compute_integrated_gradients(model, tensor, n_steps=n_steps).detach().cpu().numpy()[0, 0]
    return attr


def main() -> None:
    args = parse_args()
    paths_cfg = load_paths_config(Path(args.paths_config))
    experiment_dir = Path("data/counterfactuals") / args.experiment
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    npz_path, sample = load_counterfactual_npz(experiment_dir, args.sample_id)
    dataset_key = str(sample.get("dataset_key", args.experiment.split("_")[0]))
    model_key = MODEL_KEY_MAP.get(dataset_key, "unet_jsrt_full")
    model_path = paths_cfg.models.get(model_key)
    if model_path is None:
        raise KeyError(f"Model '{model_key}' not defined in paths config.")

    device = torch.device(args.device)
    model = load_unet_checkpoint(model_path, device=args.device)

    original_image = sample["original_image"]
    perturbed_image = sample["perturbed_image"]

    orig_attr = compute_attribution(model, original_image, device, n_steps=args.ig_steps)
    pert_attr = compute_attribution(model, perturbed_image, device, n_steps=args.ig_steps)

    metadata_map: Dict[str, object] = {}
    raw_meta = sample.get("perturbation_metadata")
    if raw_meta is not None:
        if isinstance(raw_meta, np.ndarray):
            try:
                metadata_map = dict(raw_meta.item())
            except Exception:
                metadata_map = {}
        elif isinstance(raw_meta, dict):
            metadata_map = raw_meta

    metadata = {"experiment": args.experiment, "sample_id": npz_path.stem, **metadata_map}

    output_path = output_dir / f"{args.experiment}_{npz_path.stem}.png"
    save_triptych(
        original_image=original_image,
        perturbed_image=perturbed_image,
        original_attr=orig_attr,
        perturbed_attr=pert_attr,
        output_path=output_path,
        metadata=metadata,
    )

    print(f"[export_counterfactual_triptych] saved {output_path}")


if __name__ == "__main__":
    main()
