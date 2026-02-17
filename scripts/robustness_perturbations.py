#!/usr/bin/env python3
"""Robustness checks via image perturbations and fingerprint re-analysis."""
import _path_setup  # noqa: F401 - ensures xfp is importable

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy import ndimage, stats

from xfp.config import load_experiment_config, load_paths_config, PathsConfig
from xfp.fingerprint.runner import run_fingerprint_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness checks using image perturbations.")
    parser.add_argument("--datasets", default="jsrt,montgomery,shenzhen", help="Comma-separated dataset keys.")
    parser.add_argument("--subset-size", type=int, default=50, help="Number of samples per dataset.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for inference.")
    parser.add_argument(
        "--output-dir",
        default="reports/robustness",
        help="Output directory for robustness artifacts.",
    )
    return parser.parse_args()


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    mean_diff = np.mean(x) - np.mean(y)
    pooled_std = np.sqrt((np.var(x, ddof=1) + np.var(y, ddof=1)) / 2)
    return mean_diff / pooled_std if pooled_std > 0 else 0.0


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1, 1)
    y = y.reshape(1, -1)
    greater = np.sum(x > y)
    less = np.sum(x < y)
    denom = x.size * y.size
    return (greater - less) / denom if denom else 0.0


def _compute_shift_stats(ref_df: pd.DataFrame, tgt_df: pd.DataFrame) -> dict:
    results = {}
    metrics = ["dice", "coverage_auc", "attribution_abs_sum", "border_abs_sum", "hist_entropy"]
    if "dice" in ref_df.columns and "dice" in tgt_df.columns:
        ref_df = ref_df.copy()
        tgt_df = tgt_df.copy()
        ref_df["iou"] = ref_df["dice"] / np.clip(2 - ref_df["dice"], 1e-6, None)
        tgt_df["iou"] = tgt_df["dice"] / np.clip(2 - tgt_df["dice"], 1e-6, None)
        metrics.append("iou")

    for metric in metrics:
        if metric not in ref_df.columns or metric not in tgt_df.columns:
            continue
        ref_vals = ref_df[metric].values
        tgt_vals = tgt_df[metric].values
        t_stat, t_pvalue = stats.ttest_ind(ref_vals, tgt_vals)
        results[f"{metric}_ttest_p"] = t_pvalue
        results[f"{metric}_cohens_d"] = _cohens_d(ref_vals, tgt_vals)
        results[f"{metric}_cliffs_delta"] = _cliffs_delta(ref_vals, tgt_vals)
    return results


def _percent_change(new: float, base: float) -> float:
    if not np.isfinite(base) or base == 0:
        return np.nan
    return (new - base) / abs(base) * 100.0


def _apply_intensity_shift(image: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(image + delta, 0.0, 1.0)


def _apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(image, sigma=sigma)


def _apply_salt_pepper(image: np.ndarray, amount: float, salt_vs_pepper: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng()
    noisy = image.copy()
    n_total = image.size
    n_salt = int(amount * n_total * salt_vs_pepper)
    n_pepper = int(amount * n_total * (1 - salt_vs_pepper))
    coords = rng.choice(n_total, size=n_salt + n_pepper, replace=False)
    flat = noisy.reshape(-1)
    flat[coords[:n_salt]] = 1.0
    flat[coords[n_salt:]] = 0.0
    return noisy


def _prepare_cache_subset(
    cache_root: Path,
    original_cache_root: Path,
    dataset_key: str,
    subset_name: str,
    rows: pd.DataFrame,
    perturb_fn,
    perturb_label: str,
) -> None:
    cache_dir = cache_root / dataset_key / subset_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    for row in rows.itertuples(index=False):
        src = original_cache_root / dataset_key / "full" / row.cache_file
        if not src.exists():
            continue
        data = np.load(src)
        image = data["image"].astype(np.float32)
        mask = data["mask"].astype(np.uint8)
        perturbed = perturb_fn(image)
        np.savez_compressed(cache_dir / row.cache_file, image=perturbed, mask=mask)

    rows.to_parquet(cache_dir / "metadata.parquet", index=False)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    paths_cfg = load_paths_config(Path("configs/paths.yaml"), validate=False)
    exp_keys = []
    if "montgomery" in datasets:
        exp_keys.append("jsrt_to_montgomery")
    if "shenzhen" in datasets:
        exp_keys.append("jsrt_to_shenzhen")

    perturbations = {
        "intensity_shift": lambda img: _apply_intensity_shift(img, delta=0.1),
        "gaussian_blur": lambda img: _apply_gaussian_blur(img, sigma=1.0),
        "salt_pepper": lambda img: _apply_salt_pepper(img, amount=0.01),
    }

    # Build sampled metadata per dataset
    sampled_rows = {}
    for dataset_key in datasets:
        metadata_path = paths_cfg.cache_root / dataset_key / "full" / "metadata.parquet"
        if not metadata_path.exists():
            print(f"[WARN] Missing metadata for {dataset_key}: {metadata_path}")
            continue
        metadata = pd.read_parquet(metadata_path)
        if len(metadata) == 0:
            continue
        sample = metadata.sample(
            n=min(args.subset_size, len(metadata)),
            random_state=args.seed,
        )
        sampled_rows[dataset_key] = sample

    summary_rows = []
    for perturb_label, perturb_fn in perturbations.items():
        subset_name = f"robustness_{perturb_label}"
        for dataset_key, sample in sampled_rows.items():
            _prepare_cache_subset(
                cache_root=cache_root,
                original_cache_root=paths_cfg.cache_root,
                dataset_key=dataset_key,
                subset_name=subset_name,
                rows=sample,
                perturb_fn=perturb_fn,
                perturb_label=perturb_label,
            )

        robust_paths = PathsConfig(
            datasets_root=paths_cfg.datasets_root,
            models_root=paths_cfg.models_root,
            fingerprints_root=output_dir / "fingerprints" / perturb_label,
            cache_root=cache_root,
            datasets=paths_cfg.datasets,
            models=paths_cfg.models,
        )
        robust_paths.fingerprints_root.mkdir(parents=True, exist_ok=True)

        for exp_key in exp_keys:
            exp_cfg = load_experiment_config(Path("configs/experiments.yaml"), exp_key)
            exp_cfg.subset = subset_name
            run_fingerprint_experiment(exp_cfg=exp_cfg, paths_cfg=robust_paths, device=args.device)

            perturbed_root = robust_paths.fingerprints_root / exp_cfg.key
            base_root = paths_cfg.fingerprints_root / exp_cfg.key

            ref_key = exp_cfg.train_dataset
            for tgt_key in exp_cfg.test_datasets:
                ref_base = pd.read_parquet(base_root / f"{ref_key}.parquet")
                tgt_base = pd.read_parquet(base_root / f"{tgt_key}.parquet")
                ref_pert = pd.read_parquet(perturbed_root / f"{ref_key}.parquet")
                tgt_pert = pd.read_parquet(perturbed_root / f"{tgt_key}.parquet")

                base_stats = _compute_shift_stats(ref_base, tgt_base)
                pert_stats = _compute_shift_stats(ref_pert, tgt_pert)

                for key, base_val in base_stats.items():
                    pert_val = pert_stats.get(key, np.nan)
                    summary_rows.append({
                        "perturbation": perturb_label,
                        "comparison": f"{ref_key}_to_{tgt_key}",
                        "metric": key,
                        "baseline": base_val,
                        "perturbed": pert_val,
                        "percent_change": _percent_change(pert_val, base_val),
                    })

                # Correlation robustness per dataset
                for dataset_key, df_base, df_pert in [
                    (ref_key, ref_base, ref_pert),
                    (tgt_key, tgt_base, tgt_pert),
                ]:
                    if "attribution_abs_sum" not in df_base.columns or "dice" not in df_base.columns:
                        continue
                    base_mask = np.isfinite(df_base["attribution_abs_sum"]) & np.isfinite(df_base["dice"])
                    pert_mask = np.isfinite(df_pert["attribution_abs_sum"]) & np.isfinite(df_pert["dice"])
                    if base_mask.sum() < 3 or pert_mask.sum() < 3:
                        continue
                    base_pearson, _ = stats.pearsonr(
                        df_base["attribution_abs_sum"][base_mask], df_base["dice"][base_mask]
                    )
                    base_spearman, _ = stats.spearmanr(
                        df_base["attribution_abs_sum"][base_mask], df_base["dice"][base_mask]
                    )
                    pert_pearson, _ = stats.pearsonr(
                        df_pert["attribution_abs_sum"][pert_mask], df_pert["dice"][pert_mask]
                    )
                    pert_spearman, _ = stats.spearmanr(
                        df_pert["attribution_abs_sum"][pert_mask], df_pert["dice"][pert_mask]
                    )
                    summary_rows.append({
                        "perturbation": perturb_label,
                        "comparison": dataset_key,
                        "metric": "pearson_r",
                        "baseline": base_pearson,
                        "perturbed": pert_pearson,
                        "percent_change": _percent_change(pert_pearson, base_pearson),
                    })
                    summary_rows.append({
                        "perturbation": perturb_label,
                        "comparison": dataset_key,
                        "metric": "spearman_r",
                        "baseline": base_spearman,
                        "perturbed": pert_spearman,
                        "percent_change": _percent_change(pert_spearman, base_spearman),
                    })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "robustness_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    summary_md = output_dir / "robustness_summary.md"
    summary_md.write_text(
        "# Robustness Summary\\n\\n"
        f"- Samples per dataset: {args.subset_size}\\n"
        f"- Perturbations: {', '.join(perturbations.keys())}\\n"
        "\\nSee `robustness_summary.csv` for full details.\\n"
    )

    print(f"[INFO] Saved robustness summary to: {summary_path}")
    print(f"[INFO] Saved robustness notes to: {summary_md}")


if __name__ == "__main__":
    main()
