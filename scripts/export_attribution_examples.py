#!/usr/bin/env python
"""Export qualitative attribution overlays for selected samples."""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensures xfp is importable

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import sys


from xfp.config import ExperimentConfig, PathsConfig, load_experiment_config, load_paths_config
from xfp.fingerprint.runner import _resolve_checkpoint, _run_attribution_method, _slug  # type: ignore[attr-defined]
from xfp.models import load_unet_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export attribution overlays for inspection.")
    parser.add_argument("--experiment", required=True, help="Experiment key (configs/experiments.yaml).")
    parser.add_argument("--dataset", required=True, help="Dataset key to visualise (e.g., jsrt, montgomery, shenzhen).")
    parser.add_argument(
        "--metric",
        default="dice",
        help="Metric column used to rank samples (default: dice).",
    )
    parser.add_argument(
        "--method",
        default="integrated_gradients",
        help="Attribution method registered in the pipeline.",
    )
    parser.add_argument("--k", type=int, default=4, help="Number of top/bottom samples to export.")
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
        default="reports/overlays",
        help="Directory to store generated figures.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for model inference.")
    return parser.parse_args()


def _rank_samples(df: pd.DataFrame, metric: str, k: int) -> Tuple[pd.Series, pd.Series]:
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in fingerprint table.")
    ascending = df.sort_values(metric, ascending=True)
    descending = df.sort_values(metric, ascending=False)
    return ascending.head(k)["sample_id"], descending.head(k)["sample_id"]


def _load_npz(cache_root: Path, dataset: str, subset: str, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
    npz_path = cache_root / dataset / subset / f"{sample_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Cache file not found: {npz_path}")
    data = np.load(npz_path)
    return data["image"].astype(np.float32), data["mask"].astype(np.uint8)


def _prepare_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device=device)
    tensor.requires_grad_(True)
    return tensor


def _normalise_attr(attr: np.ndarray) -> np.ndarray:
    if attr.size == 0:
        return attr
    arr = np.abs(attr)
    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val
    return arr


def _render_overlay(image: np.ndarray, mask: np.ndarray, attribution: np.ndarray, title: str, output_path: Path) -> None:
    attribution = _normalise_attr(attribution)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(attribution, cmap="inferno", alpha=0.6)
    axes[2].set_title("Attribution Overlay")
    axes[2].axis("off")

    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    paths_cfg: PathsConfig = load_paths_config(Path(args.paths_config))
    exp_cfg: ExperimentConfig = load_experiment_config(Path(args.experiments_config), args.experiment)
    subset = exp_cfg.subset

    fingerprint_path = paths_cfg.fingerprints_root / args.experiment / f"{args.dataset}.parquet"
    if not fingerprint_path.exists():
        raise FileNotFoundError(f"Fingerprint table not found: {fingerprint_path}")

    fingerprint_df = pd.read_parquet(fingerprint_path)
    worst_ids, best_ids = _rank_samples(fingerprint_df, metric=args.metric, k=args.k)

    checkpoint = _resolve_checkpoint(args.dataset, exp_cfg.train_dataset, paths_cfg)
    model = load_unet_checkpoint(checkpoint, device=device)

    cache_root = paths_cfg.cache_root
    output_dir = Path(args.output_dir) / args.experiment / args.dataset / _slug(args.method)

    selections: List[Tuple[str, str]] = []
    for sid in worst_ids:
        selections.append((sid, "worst"))
    for sid in best_ids:
        selections.append((sid, "best"))

    for sample_id, label in selections:
        image, mask = _load_npz(cache_root, args.dataset, subset, sample_id)
        tensor = _prepare_tensor(image, device=device)
        attribution = _run_attribution_method(args.method, model, tensor, device=device)  # type: ignore[attr-defined]
        output_path = output_dir / f"{label}_{args.metric}_{sample_id}.png"
        title = f"{args.dataset} | {label} {args.metric}: {sample_id}"
        _render_overlay(image, mask, attribution, title, output_path)
        del tensor

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Saved overlays → {output_dir}")


if __name__ == "__main__":
    import sys

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    main()
