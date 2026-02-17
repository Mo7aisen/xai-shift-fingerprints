"""Attribution fingerprint runner."""

from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from xfp.config import ExperimentConfig, PathsConfig
from xfp.attribution import compute_grad_cam, compute_integrated_gradients
from xfp.fingerprint.metrics import (
    compute_border_stats,
    compute_component_stats,
    compute_coverage_curve,
    compute_histogram_features,
)
from xfp.models import load_unet_checkpoint

MODEL_KEY_MAP = {
    "jsrt": "unet_jsrt_full",
    "montgomery": "unet_montgomery_full",
    # Provide a Shenzhen-specific model key when available; falls back to train model otherwise.
    "shenzhen": "unet_shenzhen_full",
    "nih_chestxray14": "unet_nih_full",
}

EXCLUDED_METADATA_COLUMNS = {
    "dataset_key",
    "subset",
    "sample_id",
    "cache_file",
    "source_image",
    "source_mask",
    "original_height",
    "original_width",
    "processed_height",
    "processed_width",
    "pad_left",
    "pad_top",
    "pad_right",
    "pad_bottom",
    "scale",
    "mask_coverage",
}

ENDPOINT_MODES = {"upper_bound_gt", "predicted_mask", "mask_free"}
MASK_DEPENDENT_FEATURES = [
    "border_abs_sum",
    "border_signed_sum",
    "border_pixel_count",
    "component_count",
    "component_mean_size",
    "component_median_size",
    "component_largest_size",
    "component_border_fraction",
    "component_border_mass_fraction",
]


@dataclass
class FingerprintResult:
    experiment_key: str
    fingerprint_path: Path
    dataset_tables: Dict[str, Path]
    summaries: Dict[str, Dict[str, float]]


def run_fingerprint_experiment(
    exp_cfg: ExperimentConfig,
    paths_cfg: PathsConfig,
    device: str = "cuda",
    endpoint_mode: str = "upper_bound_gt",
    seed: int | None = None,
) -> FingerprintResult:
    """Generate attribution fingerprints for configured datasets."""
    _validate_endpoint_mode(endpoint_mode)
    if seed is not None:
        _set_reproducibility_seed(seed)

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[xfp] Requested CUDA device but no GPU detected; falling back to CPU.")
        device = "cpu"
    else:
        print(
            f"[xfp] Running fingerprint experiment '{exp_cfg.key}' "
            f"on device: {device} (endpoint={endpoint_mode})"
        )

    # Keep historical output layout for upper-bound mode; isolate new deployment-oriented modes.
    if endpoint_mode == "upper_bound_gt":
        fingerprint_dir = paths_cfg.fingerprints_root / exp_cfg.key
    else:
        fingerprint_dir = paths_cfg.fingerprints_root / endpoint_mode / exp_cfg.key
    fingerprint_dir.mkdir(parents=True, exist_ok=True)

    datasets = list(dict.fromkeys([exp_cfg.train_dataset, *exp_cfg.test_datasets]))
    per_dataset_tables: Dict[str, Path] = {}
    summaries: Dict[str, Dict[str, float]] = {}

    for dataset_key in datasets:
        model_checkpoint = _resolve_checkpoint(
            dataset_key=dataset_key, train_dataset=exp_cfg.train_dataset, paths_cfg=paths_cfg
        )
        model = load_unet_checkpoint(model_checkpoint, device=device)

        df = _process_dataset(
            dataset_key=dataset_key,
            subset=exp_cfg.subset,
            cache_root=paths_cfg.cache_root,
            model=model,
            device=device,
            attribution_methods=exp_cfg.attribution_methods or ["integrated_gradients"],
            endpoint_mode=endpoint_mode,
        )
        output_path = fingerprint_dir / f"{dataset_key}.parquet"
        df.to_parquet(output_path, index=False)
        per_dataset_tables[dataset_key] = output_path
        summaries[dataset_key] = _summarise_dataset(df)

        # Clear GPU memory after each dataset when running on CUDA
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        del model

    fingerprint_path = fingerprint_dir / "summary.json"
    fingerprint_path.write_text(json.dumps(summaries, indent=2, default=float), encoding="utf-8")

    return FingerprintResult(
        experiment_key=exp_cfg.key,
        fingerprint_path=fingerprint_path,
        dataset_tables=per_dataset_tables,
        summaries=summaries,
    )


def _set_reproducibility_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_checkpoint(dataset_key: str, train_dataset: str, paths_cfg: PathsConfig) -> Path:
    """Resolve the most appropriate checkpoint for a dataset with sensible fallbacks.

    CRITICAL FIX (2025-10-31): Modified to enforce single-model cross-dataset evaluation.
    This ensures true shift detection by using the SAME model across all test datasets,
    rather than comparing different models (which was the original behavior).

    For deployment monitoring scenarios, we want to test how ONE trained model behaves
    on different data distributions, not compare multiple models trained on different datasets.
    """

    candidate_keys: list[str] = []

    # DISABLED: Do NOT use dataset-specific models (this caused two-model comparison)
    # # 1) Explicit dataset override in MODEL_KEY_MAP
    # dataset_model = MODEL_KEY_MAP.get(dataset_key)
    # if dataset_model:
    #     candidate_keys.append(dataset_model)

    # DISABLED: Do NOT allow direct dataset key override
    # # 2) Direct key match inside paths config (allows custom names)
    # if dataset_key in paths_cfg.models:
    #     candidate_keys.append(dataset_key)

    # ENABLED: Always use the training dataset model for ALL test datasets
    # 3) Use the training dataset model (single-model evaluation)
    train_model = MODEL_KEY_MAP.get(train_dataset)
    if train_model:
        candidate_keys.append(train_model)

    seen: set[str] = set()
    for model_key in candidate_keys:
        if model_key in seen:
            continue
        seen.add(model_key)
        checkpoint = paths_cfg.models.get(model_key)
        if checkpoint and checkpoint.exists():
            return checkpoint

    raise FileNotFoundError(
        f"No checkpoint available for dataset '{dataset_key}'. "
        "Update configs/paths.yaml with a valid model path or extend MODEL_KEY_MAP."
    )


def _process_dataset(
    *,
    dataset_key: str,
    subset: str,
    cache_root: Path,
    model: torch.nn.Module,
    device: str,
    attribution_methods: List[str],
    endpoint_mode: str,
) -> pd.DataFrame:
    cache_dir = cache_root / dataset_key / subset
    metadata_path = cache_dir / "metadata.parquet"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata missing for {dataset_key}:{subset}. Run scripts/prepare_data.py first."
        )

    metadata = pd.read_parquet(metadata_path)
    metadata_fields = [col for col in metadata.columns if col not in EXCLUDED_METADATA_COLUMNS]
    records: List[Dict[str, float]] = []
    missing_cache_files = 0

    for row in metadata.itertuples(index=False):
        cache_file = cache_dir / row.cache_file
        if not cache_file.exists():
            missing_cache_files += 1
            continue
        sample_metrics = _compute_sample_metrics(
            cache_file=cache_file,
            model=model,
            device=device,
            attribution_methods=attribution_methods,
            endpoint_mode=endpoint_mode,
        )
        sample_metrics.update({"dataset_key": dataset_key, "subset": subset, "sample_id": row.sample_id})
        if endpoint_mode == "upper_bound_gt" and hasattr(row, "mask_coverage"):
            sample_metrics["mask_coverage"] = float(row.mask_coverage)
        for field in metadata_fields:
            value = getattr(row, field)
            if isinstance(value, (np.floating, np.integer)):
                sample_metrics[field] = float(value)
            else:
                sample_metrics[field] = value
        records.append(sample_metrics)

    if missing_cache_files:
        warnings.warn(
            f"Skipped {missing_cache_files} missing cache files for {dataset_key}:{subset}.",
            UserWarning,
            stacklevel=2,
        )

    return pd.DataFrame.from_records(records)


def _compute_sample_metrics(
    *,
    cache_file: Path,
    model: torch.nn.Module,
    device: str,
    attribution_methods: List[str],
    endpoint_mode: str,
) -> Dict[str, float]:
    data = np.load(cache_file)
    image = data["image"].astype(np.float32)
    gt_mask = data["mask"].astype(np.uint8) if endpoint_mode == "upper_bound_gt" else None

    # CRITICAL FIX: Apply same normalization as training
    # Training used: Normalize(mean=[0.5], std=[0.5])
    # This transforms [0, 1] → [-1, 1]
    image_normalized = (image - 0.5) / 0.5

    tensor = torch.from_numpy(image_normalized).unsqueeze(0).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)

    prediction = (probs.detach().cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
    metrics: Dict[str, float] = {
        "_endpoint_mode": endpoint_mode,
        "_mask_source": _mask_source(endpoint_mode),
    }
    if gt_mask is not None:
        metrics["dice"] = _dice_score(prediction, gt_mask)
        metrics["gt_mask_coverage"] = float(gt_mask.mean())
    if endpoint_mode == "predicted_mask":
        metrics["predicted_mask_coverage"] = float(prediction.mean())

    # Process each attribution method and prefix features by method name
    # Note: First method (index 0) gets no prefix for backward compatibility
    # Subsequent methods are prefixed with their slugified name (e.g., "grad_cam_")
    for idx, method in enumerate(attribution_methods):
        attribution = _run_attribution_method(method, model, tensor, device=device)
        # Use method slug as prefix for all methods except the first (backward compat)
        # When only one method is used, features are unprefixed
        # When multiple methods: first unprefixed, rest prefixed by method name
        prefix = "" if idx == 0 else f"{_slug(method)}_"
        if endpoint_mode == "mask_free":
            metrics.update(_reduce_attribution_mask_free(attribution, prefix=prefix))
        else:
            mask = gt_mask if endpoint_mode == "upper_bound_gt" else prediction
            metrics.update(_reduce_attribution(attribution, mask, prefix=prefix))

    # Record which attribution methods were used for traceability
    metrics["_attribution_methods"] = ",".join(attribution_methods)

    # Final cleanup - clear all tensors and cache
    del tensor, logits, probs, prediction
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def _run_attribution_method(
    method: str,
    model: torch.nn.Module,
    tensor: torch.Tensor,
    *,
    device: str,
) -> np.ndarray:
    method = method.lower()

    if method in {"integrated_gradients", "ig"}:
        baseline_tensor = torch.full_like(tensor, -1.0)
        attribution_tensor = compute_integrated_gradients(
            model,
            tensor,
            n_steps=16,
            internal_batch_size=4,
            baseline=baseline_tensor,
        )
        attribution = attribution_tensor.detach().cpu().numpy()[0, 0]
        del attribution_tensor
        if tensor.is_cuda:
            torch.cuda.synchronize()
        return attribution

    if method in {"grad_cam", "gradcam"}:
        reducer = lambda probs: probs.mean()
        cam = compute_grad_cam(model, tensor, target_reducer=reducer)
        return cam.numpy()

    raise KeyError(f"Unsupported attribution method '{method}'.")


def _reduce_attribution(
    attribution_map: np.ndarray,
    mask: np.ndarray,
    *,
    prefix: str,
) -> Dict[str, float]:
    border_stats = compute_border_stats(attribution_map, mask)
    coverage = compute_coverage_curve(attribution_map, mask=mask)
    histogram = compute_histogram_features(
        attribution_map,
        bins=HISTOGRAM_BINS,
        return_distribution=True,
    )
    components = compute_component_stats(attribution_map, mask)

    def _name(key: str) -> str:
        return f"{prefix}{key}" if prefix else key

    abs_map = np.abs(attribution_map)
    mass = float(abs_map.sum())

    metrics = {
        _name("attribution_abs_mean"): float(abs_map.mean()),
        _name("attribution_abs_sum"): mass,
        _name("border_abs_sum"): border_stats.abs_sum,
        _name("border_signed_sum"): border_stats.signed_sum,
        _name("border_pixel_count"): float(border_stats.pixel_count),
    }
    if mass > 0:
        metrics[_name("attribution_abs_mean_log10")] = float(np.log10(abs_map.mean()))
        metrics[_name("attribution_abs_sum_log10")] = float(np.log10(mass))
    else:
        # Use a large negative value instead of -inf to avoid NaN propagation in statistics
        metrics[_name("attribution_abs_mean_log10")] = -30.0  # Represents ~0 attribution
        metrics[_name("attribution_abs_sum_log10")] = -30.0
    metrics.update({_name(k): float(v) for k, v in coverage.items()})
    metrics.update({_name(k): float(v) for k, v in histogram.items()})
    metrics.update({_name(k): float(v) for k, v in components.items()})

    # Validate all metrics are finite to prevent silent NaN/Inf propagation
    return _validate_metrics(metrics)


def _reduce_attribution_mask_free(
    attribution_map: np.ndarray,
    *,
    prefix: str,
) -> Dict[str, float]:
    """Reduce attribution map without any segmentation mask dependency."""
    coverage = compute_coverage_curve(attribution_map, mask=None)
    histogram = compute_histogram_features(
        attribution_map,
        bins=HISTOGRAM_BINS,
        return_distribution=True,
    )

    def _name(key: str) -> str:
        return f"{prefix}{key}" if prefix else key

    abs_map = np.abs(attribution_map)
    mass = float(abs_map.sum())
    metrics: Dict[str, float] = {
        _name("attribution_abs_mean"): float(abs_map.mean()),
        _name("attribution_abs_sum"): mass,
    }
    if mass > 0:
        metrics[_name("attribution_abs_mean_log10")] = float(np.log10(abs_map.mean()))
        metrics[_name("attribution_abs_sum_log10")] = float(np.log10(mass))
    else:
        metrics[_name("attribution_abs_mean_log10")] = -30.0
        metrics[_name("attribution_abs_sum_log10")] = -30.0

    metrics.update({_name(k): float(v) for k, v in coverage.items()})
    metrics.update({_name(k): float(v) for k, v in histogram.items()})
    for feature_key in MASK_DEPENDENT_FEATURES:
        metrics[_name(feature_key)] = 0.0

    return _validate_metrics(metrics)


def _validate_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Validate that all metric values are finite (no NaN or Inf).

    This prevents silent data quality issues from propagating to downstream analysis.

    Args:
        metrics: Dictionary of metric name to value

    Returns:
        The same metrics dict if all values are valid

    Raises:
        ValueError: If any metric value is NaN or Inf
    """
    invalid = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not np.isfinite(value):
            invalid.append(f"{key}={value}")

    if invalid:
        raise ValueError(
            f"Non-finite metric values detected (NaN or Inf). "
            f"This indicates a computation error. Invalid metrics: {', '.join(invalid)}"
        )

    return metrics


def _slug(method: str) -> str:
    return method.lower().replace(" ", "_")


def _mask_source(endpoint_mode: str) -> str:
    if endpoint_mode == "upper_bound_gt":
        return "ground_truth"
    if endpoint_mode == "predicted_mask":
        return "prediction"
    return "none"


def _validate_endpoint_mode(endpoint_mode: str) -> None:
    if endpoint_mode not in ENDPOINT_MODES:
        raise ValueError(
            f"Unsupported endpoint_mode '{endpoint_mode}'. "
            f"Expected one of: {sorted(ENDPOINT_MODES)}."
        )


def _dice_score(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Compute Dice similarity coefficient between prediction and target masks.

    Handles edge cases properly:
    - Both empty (union=0): Returns 1.0 (perfect agreement on empty)
    - One empty, one not: Returns 0.0 (complete disagreement)
    - Normal case: Standard Dice formula with epsilon for numerical stability

    Args:
        prediction: Binary prediction mask
        target: Binary ground truth mask
        eps: Small value for numerical stability in normal cases

    Returns:
        Dice score in [0, 1]
    """
    prediction = prediction.astype(np.float32)
    target = target.astype(np.float32)

    pred_sum = float(prediction.sum())
    target_sum = float(target.sum())
    union = pred_sum + target_sum

    # Handle edge cases explicitly
    if union == 0:
        # Both masks are empty - this is a perfect match
        return 1.0

    intersection = float(np.sum(prediction * target))
    return (2.0 * intersection + eps) / (union + eps)


def _summarise_dataset(df: pd.DataFrame) -> Dict[str, float]:
    summary: Dict[str, float] = {"count": float(len(df))}
    if df.empty:
        return summary

    for column in df.columns:
        if column in {"dataset_key", "subset", "sample_id"}:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        summary[f"{column}_mean"] = float(numeric.mean())
        summary[f"{column}_std"] = float(numeric.std(ddof=0))
        summary[f"{column}_median"] = float(numeric.median())
    return summary
HISTOGRAM_BINS = 32
