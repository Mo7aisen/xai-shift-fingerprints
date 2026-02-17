#!/usr/bin/env python3
"""Energy/confidence-based OOD baseline using UNet predictions.

Computes mean pixel entropy and 1 - mean MSP as OOD scores.
"""

from __future__ import annotations

import _path_setup  # noqa: F401 - ensure repo imports are available

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from xfp.config import load_paths_config
from xfp.models.loader import load_unet_checkpoint


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
        image = data["image"].astype(np.float32)
        sample_id = path.stem
        return image, sample_id


def _bootstrap_auc(labels: np.ndarray, scores: np.ndarray, n_boot: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = labels.size
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        boot.append(roc_auc_score(labels[idx], scores[idx]))
    if not boot:
        return float("nan"), float("nan")
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return float(ci_low), float(ci_high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Energy-based OOD baseline for UNet.")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=repo_root / "configs/paths.yaml")
    parser.add_argument(
        "--datasets",
        type=str,
        default="jsrt,montgomery,shenzhen,nih_chestxray14",
        help="Comma-separated dataset keys.",
    )
    parser.add_argument("--subset", type=str, default="full")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples per dataset (0=all)")
    parser.add_argument("--model-key", type=str, default="unet_jsrt_full")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "results" / "baselines")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = load_paths_config(root / args.config, validate=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    max_samples = args.max_samples if args.max_samples > 0 else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = cfg.models.get(args.model_key)
    if model_path is None:
        raise KeyError(f"Model key '{args.model_key}' not found in configs/paths.yaml")

    model = load_unet_checkpoint(model_path, device=str(device))
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "energy_ood_scores.csv"
    auc_path = args.output_dir / "energy_ood_auc.csv"

    in_dist = {"jsrt", "montgomery"}
    all_labels = []
    entropy_scores = []
    msp_scores = []
    scores_by_dataset = {ds: {"entropy": [], "msp": []} for ds in datasets}

    with score_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["sample_id", "dataset", "entropy_score", "msp_score"],
        )
        writer.writeheader()

        for ds in datasets:
            cache_dir = cfg.cache_root / ds / args.subset
            loader = DataLoader(
                CachedImageDataset(cache_dir, max_samples=max_samples),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            label = 0 if ds in in_dist else 1
            with torch.no_grad():
                for images, sample_ids in loader:
                    images = images.unsqueeze(1).to(device)
                    logits = model(images)
                    probs = torch.sigmoid(logits)
                    # Binary entropy per pixel
                    entropy = -(
                        probs * torch.log(probs + 1e-8)
                        + (1 - probs) * torch.log(1 - probs + 1e-8)
                    )
                    entropy_score = entropy.mean(dim=(1, 2, 3))
                    # MSP proxy
                    msp = torch.maximum(probs, 1 - probs)
                    msp_score = 1.0 - msp.mean(dim=(1, 2, 3))

                    entropy_np = entropy_score.detach().cpu().numpy().tolist()
                    msp_np = msp_score.detach().cpu().numpy().tolist()

                    for sample_id, ent, msp_val in zip(sample_ids, entropy_np, msp_np):
                        writer.writerow(
                            {
                                "sample_id": sample_id,
                                "dataset": ds,
                                "entropy_score": ent,
                                "msp_score": msp_val,
                            }
                        )

                    entropy_scores.extend(entropy_np)
                    msp_scores.extend(msp_np)
                    all_labels.extend([label] * len(entropy_np))
                    scores_by_dataset[ds]["entropy"].extend(entropy_np)
                    scores_by_dataset[ds]["msp"].extend(msp_np)

    # AUC summary
    labels = np.asarray(all_labels, dtype=int)
    entropy_scores = np.asarray(entropy_scores, dtype=float)
    msp_scores = np.asarray(msp_scores, dtype=float)

    entropy_auc = roc_auc_score(labels, entropy_scores)
    msp_auc = roc_auc_score(labels, msp_scores)
    entropy_ci = _bootstrap_auc(labels, entropy_scores)
    msp_ci = _bootstrap_auc(labels, msp_scores)

    with auc_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["scope", "metric", "auc", "ci_low", "ci_high"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scope": "overall",
                "metric": "entropy",
                "auc": entropy_auc,
                "ci_low": entropy_ci[0],
                "ci_high": entropy_ci[1],
            }
        )
        writer.writerow(
            {
                "scope": "overall",
                "metric": "msp",
                "auc": msp_auc,
                "ci_low": msp_ci[0],
                "ci_high": msp_ci[1],
            }
        )

        in_entropy = np.concatenate(
            [np.asarray(scores_by_dataset[ds]["entropy"]) for ds in in_dist if ds in scores_by_dataset]
        )
        in_msp = np.concatenate(
            [np.asarray(scores_by_dataset[ds]["msp"]) for ds in in_dist if ds in scores_by_dataset]
        )

        for ds in datasets:
            if ds in in_dist:
                continue
            out_entropy = np.asarray(scores_by_dataset[ds]["entropy"])
            out_msp = np.asarray(scores_by_dataset[ds]["msp"])
            if out_entropy.size == 0:
                continue

            labels_ds = np.concatenate(
                [np.zeros(in_entropy.size, dtype=int), np.ones(out_entropy.size, dtype=int)]
            )
            scores_ds = np.concatenate([in_entropy, out_entropy])
            auc_ds = roc_auc_score(labels_ds, scores_ds)
            ci_ds = _bootstrap_auc(labels_ds, scores_ds)
            writer.writerow(
                {
                    "scope": f"jsrt+montgomery vs {ds}",
                    "metric": "entropy",
                    "auc": auc_ds,
                    "ci_low": ci_ds[0],
                    "ci_high": ci_ds[1],
                }
            )

            labels_ds = np.concatenate(
                [np.zeros(in_msp.size, dtype=int), np.ones(out_msp.size, dtype=int)]
            )
            scores_ds = np.concatenate([in_msp, out_msp])
            auc_ds = roc_auc_score(labels_ds, scores_ds)
            ci_ds = _bootstrap_auc(labels_ds, scores_ds)
            writer.writerow(
                {
                    "scope": f"jsrt+montgomery vs {ds}",
                    "metric": "msp",
                    "auc": auc_ds,
                    "ci_low": ci_ds[0],
                    "ci_high": ci_ds[1],
                }
            )

    print(f"[INFO] Saved scores to {score_path}")
    print(f"[INFO] Saved AUC summary to {auc_path}")


if __name__ == "__main__":
    main()
