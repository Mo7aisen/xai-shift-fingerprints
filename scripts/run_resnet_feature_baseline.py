#!/usr/bin/env python3
"""ResNet-50 feature baseline for shift detection (AUC-ROC).

Extracts penultimate-layer features from cached images and computes
Mahalanobis OOD scores with JSRT+Montgomery as in-distribution.
"""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensure repo imports are available

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights

from xfp.config import load_paths_config
from xfp.utils.ood_eval import binary_ood_metrics_with_bootstrap


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
        image = data["image"].astype(np.float32)  # [H, W] in [0,1]
        sample_id = path.stem
        return image, sample_id


def _prepare_batch(images: torch.Tensor) -> torch.Tensor:
    # images: [B, H, W] -> [B, 3, 224, 224]
    images = images.unsqueeze(1)  # [B,1,H,W]
    images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def _extract_features(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    feats = []
    ids = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images, sample_ids = batch
            images = images.to(device)
            images = _prepare_batch(images)
            outputs = model(images).squeeze(-1).squeeze(-1)
            feats.append(outputs.detach().cpu().numpy())
            ids.extend(sample_ids)
    if not feats:
        return np.empty((0, 2048), dtype=np.float32), []
    return np.vstack(feats), ids


def _mahalanobis_scores(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    mu: np.ndarray,
    inv_cov: np.ndarray,
) -> tuple[list[float], list[str]]:
    scores = []
    ids = []
    mu_t = torch.from_numpy(mu).to(device=device, dtype=torch.float32)
    inv_t = torch.from_numpy(inv_cov).to(device=device, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images, sample_ids = batch
            images = images.to(device)
            images = _prepare_batch(images)
            feats = model(images).squeeze(-1).squeeze(-1)
            diff = feats - mu_t
            # score = diff^T * inv_cov * diff
            score = torch.einsum("bi,ij,bj->b", diff, inv_t, diff)
            scores.extend(score.detach().cpu().numpy().tolist())
            ids.extend(sample_ids)
    return scores, ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ResNet-50 feature baseline for shift detection.")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=repo_root / "configs/paths.yaml")
    parser.add_argument(
        "--datasets",
        type=str,
        default="jsrt,montgomery,shenzhen,nih_chestxray14",
        help="Comma-separated dataset keys.",
    )
    parser.add_argument(
        "--in-datasets",
        type=str,
        default="jsrt,montgomery",
        help="Comma-separated in-distribution dataset keys.",
    )
    parser.add_argument("--subset", type=str, default="full")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples per dataset (0=all)")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "results" / "baselines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(root / args.config, validate=False)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    max_samples = args.max_samples if args.max_samples > 0 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet50_Weights.IMAGENET1K_V1
    resnet = models.resnet50(weights=weights)
    feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "resnet_ood_scores.csv"
    auc_path = args.output_dir / "resnet_ood_auc.csv"

    # In-distribution set is configurable per experiment.
    in_dist = {d.strip() for d in args.in_datasets.split(",") if d.strip()}
    in_features = []

    for ds in datasets:
        if ds not in in_dist:
            continue
        cache_dir = cfg.cache_root / ds / args.subset
        loader = DataLoader(
            CachedImageDataset(cache_dir, max_samples=max_samples),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        feats, _ = _extract_features(loader, feature_extractor, device)
        in_features.append(feats)

    if not in_features:
        raise RuntimeError("No in-distribution features extracted (check cache paths).")

    in_features = np.vstack(in_features).astype(np.float64)
    mu = in_features.mean(axis=0)
    cov = np.cov(in_features, rowvar=False)
    inv_cov = np.linalg.pinv(cov)

    # Write scores
    with score_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample_id", "dataset", "score"])
        writer.writeheader()

        all_scores = []
        all_labels = []
        scores_by_dataset = {ds: [] for ds in datasets}

        for ds in datasets:
            cache_dir = cfg.cache_root / ds / args.subset
            loader = DataLoader(
                CachedImageDataset(cache_dir, max_samples=max_samples),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            scores, ids = _mahalanobis_scores(loader, feature_extractor, device, mu, inv_cov)
            label = 0 if ds in in_dist else 1
            for sample_id, score in zip(ids, scores):
                writer.writerow({"sample_id": sample_id, "dataset": ds, "score": score})
            all_scores.extend(scores)
            all_labels.extend([label] * len(scores))
            scores_by_dataset[ds].extend(scores)

    # AUC summary
    all_scores = np.asarray(all_scores, dtype=float)
    all_labels = np.asarray(all_labels, dtype=int)
    overall = binary_ood_metrics_with_bootstrap(all_labels, all_scores, n_boot=500, seed=42)

    with auc_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "scope",
                "auc",
                "ci_low",
                "ci_high",
                "aupr",
                "aupr_ci_low",
                "aupr_ci_high",
                "fpr95",
                "fpr95_ci_low",
                "fpr95_ci_high",
                "tpr_at_fpr05",
                "tpr_at_fpr05_ci_low",
                "tpr_at_fpr05_ci_high",
                "ece",
                "brier",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scope": "overall",
                "auc": overall["auc"],
                "ci_low": overall["auc_ci_low"],
                "ci_high": overall["auc_ci_high"],
                "aupr": overall["aupr"],
                "aupr_ci_low": overall["aupr_ci_low"],
                "aupr_ci_high": overall["aupr_ci_high"],
                "fpr95": overall["fpr95"],
                "fpr95_ci_low": overall["fpr95_ci_low"],
                "fpr95_ci_high": overall["fpr95_ci_high"],
                "tpr_at_fpr05": overall["tpr_at_fpr05"],
                "tpr_at_fpr05_ci_low": overall["tpr_at_fpr05_ci_low"],
                "tpr_at_fpr05_ci_high": overall["tpr_at_fpr05_ci_high"],
                "ece": overall["ece"],
                "brier": overall["brier"],
            }
        )

        # Per-out dataset AUCs
        in_scores = np.concatenate(
            [np.asarray(scores_by_dataset[ds]) for ds in in_dist if ds in scores_by_dataset]
        )
        for ds in datasets:
            if ds in in_dist:
                continue
            out_scores = np.asarray(scores_by_dataset[ds])
            if out_scores.size == 0 or in_scores.size == 0:
                continue
            ds_scores = np.concatenate([in_scores, out_scores])
            ds_labels = np.concatenate(
                [np.zeros(in_scores.size, dtype=int), np.ones(out_scores.size, dtype=int)]
            )
            ds_metrics = binary_ood_metrics_with_bootstrap(ds_labels, ds_scores, n_boot=500, seed=100)
            writer.writerow(
                {
                    "scope": f"{'+'.join(sorted(in_dist))} vs {ds}",
                    "auc": ds_metrics["auc"],
                    "ci_low": ds_metrics["auc_ci_low"],
                    "ci_high": ds_metrics["auc_ci_high"],
                    "aupr": ds_metrics["aupr"],
                    "aupr_ci_low": ds_metrics["aupr_ci_low"],
                    "aupr_ci_high": ds_metrics["aupr_ci_high"],
                    "fpr95": ds_metrics["fpr95"],
                    "fpr95_ci_low": ds_metrics["fpr95_ci_low"],
                    "fpr95_ci_high": ds_metrics["fpr95_ci_high"],
                    "tpr_at_fpr05": ds_metrics["tpr_at_fpr05"],
                    "tpr_at_fpr05_ci_low": ds_metrics["tpr_at_fpr05_ci_low"],
                    "tpr_at_fpr05_ci_high": ds_metrics["tpr_at_fpr05_ci_high"],
                    "ece": ds_metrics["ece"],
                    "brier": ds_metrics["brier"],
                }
            )

    print(f"[INFO] Saved scores to {score_path}")
    print(f"[INFO] Saved AUC summary to {auc_path}")


if __name__ == "__main__":
    main()
