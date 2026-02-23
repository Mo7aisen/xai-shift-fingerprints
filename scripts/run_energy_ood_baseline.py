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
from torch.utils.data import Dataset, DataLoader

from xfp.config import load_paths_config
from xfp.models.loader import load_unet_checkpoint
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
        image = data["image"].astype(np.float32)
        sample_id = path.stem
        return image, sample_id


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
    parser.add_argument(
        "--in-datasets",
        type=str,
        default="jsrt,montgomery",
        help="Comma-separated in-distribution dataset keys.",
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

    in_dist = {d.strip() for d in args.in_datasets.split(",") if d.strip()}
    all_labels = []
    entropy_scores = []
    msp_scores = []
    maxlogit_scores = []
    scores_by_dataset = {ds: {"entropy": [], "msp": [], "maxlogit": []} for ds in datasets}

    with score_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["sample_id", "dataset", "entropy_score", "msp_score", "maxlogit_score"],
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
                    # MaxLogit proxy for binary logits: lower abs(logit) => more OOD-like.
                    maxlogit_score = -torch.abs(logits).mean(dim=(1, 2, 3))

                    entropy_np = entropy_score.detach().cpu().numpy().tolist()
                    msp_np = msp_score.detach().cpu().numpy().tolist()
                    maxlogit_np = maxlogit_score.detach().cpu().numpy().tolist()

                    for sample_id, ent, msp_val, maxlogit_val in zip(sample_ids, entropy_np, msp_np, maxlogit_np):
                        writer.writerow(
                            {
                                "sample_id": sample_id,
                                "dataset": ds,
                                "entropy_score": ent,
                                "msp_score": msp_val,
                                "maxlogit_score": maxlogit_val,
                            }
                        )

                    entropy_scores.extend(entropy_np)
                    msp_scores.extend(msp_np)
                    maxlogit_scores.extend(maxlogit_np)
                    all_labels.extend([label] * len(entropy_np))
                    scores_by_dataset[ds]["entropy"].extend(entropy_np)
                    scores_by_dataset[ds]["msp"].extend(msp_np)
                    scores_by_dataset[ds]["maxlogit"].extend(maxlogit_np)

    # AUC summary
    labels = np.asarray(all_labels, dtype=int)
    entropy_scores = np.asarray(entropy_scores, dtype=float)
    msp_scores = np.asarray(msp_scores, dtype=float)
    maxlogit_scores = np.asarray(maxlogit_scores, dtype=float)

    entropy_metrics = binary_ood_metrics_with_bootstrap(labels, entropy_scores, n_boot=500, seed=42)
    msp_metrics = binary_ood_metrics_with_bootstrap(labels, msp_scores, n_boot=500, seed=43)
    maxlogit_metrics = binary_ood_metrics_with_bootstrap(labels, maxlogit_scores, n_boot=500, seed=44)

    with auc_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "scope",
                "metric",
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
                "metric": "entropy",
                "auc": entropy_metrics["auc"],
                "ci_low": entropy_metrics["auc_ci_low"],
                "ci_high": entropy_metrics["auc_ci_high"],
                "aupr": entropy_metrics["aupr"],
                "aupr_ci_low": entropy_metrics["aupr_ci_low"],
                "aupr_ci_high": entropy_metrics["aupr_ci_high"],
                "fpr95": entropy_metrics["fpr95"],
                "fpr95_ci_low": entropy_metrics["fpr95_ci_low"],
                "fpr95_ci_high": entropy_metrics["fpr95_ci_high"],
                "tpr_at_fpr05": entropy_metrics["tpr_at_fpr05"],
                "tpr_at_fpr05_ci_low": entropy_metrics["tpr_at_fpr05_ci_low"],
                "tpr_at_fpr05_ci_high": entropy_metrics["tpr_at_fpr05_ci_high"],
                "ece": entropy_metrics["ece"],
                "brier": entropy_metrics["brier"],
            }
        )
        writer.writerow(
            {
                "scope": "overall",
                "metric": "msp",
                "auc": msp_metrics["auc"],
                "ci_low": msp_metrics["auc_ci_low"],
                "ci_high": msp_metrics["auc_ci_high"],
                "aupr": msp_metrics["aupr"],
                "aupr_ci_low": msp_metrics["aupr_ci_low"],
                "aupr_ci_high": msp_metrics["aupr_ci_high"],
                "fpr95": msp_metrics["fpr95"],
                "fpr95_ci_low": msp_metrics["fpr95_ci_low"],
                "fpr95_ci_high": msp_metrics["fpr95_ci_high"],
                "tpr_at_fpr05": msp_metrics["tpr_at_fpr05"],
                "tpr_at_fpr05_ci_low": msp_metrics["tpr_at_fpr05_ci_low"],
                "tpr_at_fpr05_ci_high": msp_metrics["tpr_at_fpr05_ci_high"],
                "ece": msp_metrics["ece"],
                "brier": msp_metrics["brier"],
            }
        )
        writer.writerow(
            {
                "scope": "overall",
                "metric": "maxlogit",
                "auc": maxlogit_metrics["auc"],
                "ci_low": maxlogit_metrics["auc_ci_low"],
                "ci_high": maxlogit_metrics["auc_ci_high"],
                "aupr": maxlogit_metrics["aupr"],
                "aupr_ci_low": maxlogit_metrics["aupr_ci_low"],
                "aupr_ci_high": maxlogit_metrics["aupr_ci_high"],
                "fpr95": maxlogit_metrics["fpr95"],
                "fpr95_ci_low": maxlogit_metrics["fpr95_ci_low"],
                "fpr95_ci_high": maxlogit_metrics["fpr95_ci_high"],
                "tpr_at_fpr05": maxlogit_metrics["tpr_at_fpr05"],
                "tpr_at_fpr05_ci_low": maxlogit_metrics["tpr_at_fpr05_ci_low"],
                "tpr_at_fpr05_ci_high": maxlogit_metrics["tpr_at_fpr05_ci_high"],
                "ece": maxlogit_metrics["ece"],
                "brier": maxlogit_metrics["brier"],
            }
        )

        in_entropy = np.concatenate(
            [np.asarray(scores_by_dataset[ds]["entropy"]) for ds in in_dist if ds in scores_by_dataset]
        )
        in_msp = np.concatenate(
            [np.asarray(scores_by_dataset[ds]["msp"]) for ds in in_dist if ds in scores_by_dataset]
        )
        in_maxlogit = np.concatenate(
            [np.asarray(scores_by_dataset[ds]["maxlogit"]) for ds in in_dist if ds in scores_by_dataset]
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
            metrics_ds = binary_ood_metrics_with_bootstrap(labels_ds, scores_ds, n_boot=500, seed=100)
            writer.writerow(
                {
                    "scope": f"{'+'.join(sorted(in_dist))} vs {ds}",
                    "metric": "entropy",
                    "auc": metrics_ds["auc"],
                    "ci_low": metrics_ds["auc_ci_low"],
                    "ci_high": metrics_ds["auc_ci_high"],
                    "aupr": metrics_ds["aupr"],
                    "aupr_ci_low": metrics_ds["aupr_ci_low"],
                    "aupr_ci_high": metrics_ds["aupr_ci_high"],
                    "fpr95": metrics_ds["fpr95"],
                    "fpr95_ci_low": metrics_ds["fpr95_ci_low"],
                    "fpr95_ci_high": metrics_ds["fpr95_ci_high"],
                    "tpr_at_fpr05": metrics_ds["tpr_at_fpr05"],
                    "tpr_at_fpr05_ci_low": metrics_ds["tpr_at_fpr05_ci_low"],
                    "tpr_at_fpr05_ci_high": metrics_ds["tpr_at_fpr05_ci_high"],
                    "ece": metrics_ds["ece"],
                    "brier": metrics_ds["brier"],
                }
            )

            labels_ds = np.concatenate(
                [np.zeros(in_msp.size, dtype=int), np.ones(out_msp.size, dtype=int)]
            )
            scores_ds = np.concatenate([in_msp, out_msp])
            metrics_ds = binary_ood_metrics_with_bootstrap(labels_ds, scores_ds, n_boot=500, seed=101)
            writer.writerow(
                {
                    "scope": f"{'+'.join(sorted(in_dist))} vs {ds}",
                    "metric": "msp",
                    "auc": metrics_ds["auc"],
                    "ci_low": metrics_ds["auc_ci_low"],
                    "ci_high": metrics_ds["auc_ci_high"],
                    "aupr": metrics_ds["aupr"],
                    "aupr_ci_low": metrics_ds["aupr_ci_low"],
                    "aupr_ci_high": metrics_ds["aupr_ci_high"],
                    "fpr95": metrics_ds["fpr95"],
                    "fpr95_ci_low": metrics_ds["fpr95_ci_low"],
                    "fpr95_ci_high": metrics_ds["fpr95_ci_high"],
                    "tpr_at_fpr05": metrics_ds["tpr_at_fpr05"],
                    "tpr_at_fpr05_ci_low": metrics_ds["tpr_at_fpr05_ci_low"],
                    "tpr_at_fpr05_ci_high": metrics_ds["tpr_at_fpr05_ci_high"],
                    "ece": metrics_ds["ece"],
                    "brier": metrics_ds["brier"],
                }
            )

            out_maxlogit = np.asarray(scores_by_dataset[ds]["maxlogit"])
            labels_ds = np.concatenate(
                [np.zeros(in_maxlogit.size, dtype=int), np.ones(out_maxlogit.size, dtype=int)]
            )
            scores_ds = np.concatenate([in_maxlogit, out_maxlogit])
            metrics_ds = binary_ood_metrics_with_bootstrap(labels_ds, scores_ds, n_boot=500, seed=102)
            writer.writerow(
                {
                    "scope": f"{'+'.join(sorted(in_dist))} vs {ds}",
                    "metric": "maxlogit",
                    "auc": metrics_ds["auc"],
                    "ci_low": metrics_ds["auc_ci_low"],
                    "ci_high": metrics_ds["auc_ci_high"],
                    "aupr": metrics_ds["aupr"],
                    "aupr_ci_low": metrics_ds["aupr_ci_low"],
                    "aupr_ci_high": metrics_ds["aupr_ci_high"],
                    "fpr95": metrics_ds["fpr95"],
                    "fpr95_ci_low": metrics_ds["fpr95_ci_low"],
                    "fpr95_ci_high": metrics_ds["fpr95_ci_high"],
                    "tpr_at_fpr05": metrics_ds["tpr_at_fpr05"],
                    "tpr_at_fpr05_ci_low": metrics_ds["tpr_at_fpr05_ci_low"],
                    "tpr_at_fpr05_ci_high": metrics_ds["tpr_at_fpr05_ci_high"],
                    "ece": metrics_ds["ece"],
                    "brier": metrics_ds["brier"],
                }
            )

    print(f"[INFO] Saved scores to {score_path}")
    print(f"[INFO] Saved AUC summary to {auc_path}")


if __name__ == "__main__":
    main()
