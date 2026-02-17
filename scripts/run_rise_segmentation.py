#!/usr/bin/env python3
"""RISE-style attribution for segmentation and fingerprint robustness checks.

Computes RISE saliency maps for a subset of JSRT and Shenzhen samples
and correlates attribution mass with IG mass from existing fingerprints.
"""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensure repo imports are available

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

from xfp.config import load_paths_config
from xfp.models.loader import load_unet_checkpoint
from xfp.fingerprint.metrics import compute_border_stats, compute_coverage_curve, compute_histogram_features


class CachedImageDataset(Dataset):
    def __init__(self, cache_dir: Path, max_samples: int | None = None):
        self.cache_dir = cache_dir
        self.files = sorted(cache_dir.glob("*.npz"))
        if max_samples:
            self.files = self.files[:max_samples]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data = np.load(path)
        image = data["image"].astype(np.float32)  # [H, W]
        mask = data["mask"].astype(np.float32)   # [H, W]
        return image, mask, path.stem


def _rise_saliency(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    n_masks: int,
    grid_size: int,
    p: float,
    mask_batch: int,
) -> np.ndarray:
    """Compute RISE saliency map for one image.

    image: [1, 1, H, W]
    target_mask: [1, 1, H, W]
    """
    device = image.device
    _, _, h, w = image.shape

    saliency = torch.zeros((1, 1, h, w), device=device)
    denom = torch.zeros((1, 1, h, w), device=device)

    remaining = n_masks
    with torch.no_grad():
        while remaining > 0:
            current_batch = min(mask_batch, remaining)
            remaining -= current_batch
            masks = (torch.rand(current_batch, 1, grid_size, grid_size, device=device) < p).float()
            masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
            masked_inputs = image * masks
            logits = model(masked_inputs)
            probs = torch.sigmoid(logits)
            # Score = mean probability within ground-truth mask
            score = (probs * target_mask).mean(dim=(1, 2, 3))
            saliency += (score.view(-1, 1, 1, 1) * masks).sum(dim=0, keepdim=True)
            denom += masks.sum(dim=0, keepdim=True)

    saliency = saliency / (denom + 1e-8)
    return saliency.squeeze().detach().cpu().numpy()


def _compute_metrics(saliency: np.ndarray, mask: np.ndarray) -> dict:
    abs_map = np.abs(saliency)
    border_stats = compute_border_stats(saliency, mask)
    coverage = compute_coverage_curve(saliency, mask=mask)
    hist = compute_histogram_features(saliency, bins=32, return_distribution=False)
    return {
        "attribution_abs_sum": float(abs_map.sum()),
        "border_abs_sum": float(border_stats.abs_sum),
        "hist_entropy": float(hist["hist_entropy"]),
        "coverage_auc": float(coverage["coverage_auc"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RISE segmentation robustness check.")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=repo_root / "configs/paths.yaml")
    parser.add_argument("--datasets", type=str, default="jsrt,shenzhen")
    parser.add_argument("--subset", type=str, default="full")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--n-masks", type=int, default=1000)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--mask-batch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-key", type=str, default="unet_jsrt_full")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "reports" / "rise")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(root / args.config, validate=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = cfg.models.get(args.model_key)
    if model_path is None:
        raise KeyError(f"Model key '{args.model_key}' not found in configs/paths.yaml")
    model = load_unet_checkpoint(model_path, device=str(device))
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []

    for ds in datasets:
        cache_dir = cfg.cache_root / ds / args.subset
        loader = DataLoader(
            CachedImageDataset(cache_dir, max_samples=args.max_samples),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        for image, mask, sample_id in loader:
            image = image.unsqueeze(1).to(device)
            mask = mask.unsqueeze(1).to(device)
            saliency = _rise_saliency(
                model,
                image,
                mask,
                n_masks=args.n_masks,
                grid_size=args.grid_size,
                p=args.p,
                mask_batch=args.mask_batch,
            )
            metrics = _compute_metrics(saliency, mask.squeeze().cpu().numpy())
            metrics_rows.append({"dataset": ds, "sample_id": sample_id[0], **metrics})

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = args.output_dir / "rise_fingerprints.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Correlate with IG attribution mass
    corr_rows = []
    for ds in datasets:
        fp_path = root / "data" / "fingerprints" / f"{ds}_baseline" / f"{ds}.parquet"
        if not fp_path.exists():
            continue
        fp_df = pd.read_parquet(fp_path, columns=["sample_id", "attribution_abs_sum"])
        merged = metrics_df[metrics_df["dataset"] == ds].merge(
            fp_df, on="sample_id", how="inner", suffixes=("_rise", "_ig")
        )
        if merged.empty:
            continue
        r, pval = pearsonr(merged["attribution_abs_sum_rise"], merged["attribution_abs_sum_ig"])
        corr_rows.append({"dataset": ds, "pearson_r": r, "p_value": pval, "n": len(merged)})

    corr_df = pd.DataFrame(corr_rows)
    corr_path = args.output_dir / "rise_ig_correlation.csv"
    corr_df.to_csv(corr_path, index=False)

    print(f"[INFO] Saved RISE fingerprints to {metrics_path}")
    print(f"[INFO] Saved RISE-IG correlation to {corr_path}")


if __name__ == "__main__":
    main()
