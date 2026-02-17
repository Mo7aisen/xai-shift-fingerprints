"""Configuration loading helpers.

This module provides type-safe configuration loading with validation.
All paths are validated at load time to catch configuration errors early.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _expand_path(raw_value: str, base_dir: Path) -> Path:
    expanded = Path(os.path.expandvars(raw_value)).expanduser()
    if not expanded.is_absolute():
        expanded = (base_dir / expanded).resolve()
    return expanded


def _model_override_env(name: str) -> str:
    return f"XFP_MODEL_{name}".upper()


@dataclass
class DatasetPaths:
    """Paths to dataset images and masks."""
    images: Path
    masks: Path


@dataclass
class PathsConfig:
    """Configuration for all file paths used in the pipeline.

    Attributes:
        datasets_root: Base directory for datasets
        models_root: Base directory for model checkpoints
        fingerprints_root: Output directory for fingerprint parquet files
        cache_root: Cache directory for preprocessed data
        datasets: Mapping of dataset names to their paths
        models: Mapping of model names to checkpoint paths
    """
    datasets_root: Path
    models_root: Path
    fingerprints_root: Path
    cache_root: Path
    datasets: Dict[str, DatasetPaths]
    models: Dict[str, Path]


@dataclass
class ExperimentConfig:
    key: str
    train_dataset: str
    test_datasets: List[str]
    subset: str
    attribution_methods: List[str]
    fingerprint_metrics: List[str]
    shift_metrics: List[str]
    notes: str | None = None


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_paths_config(path: Path, validate: bool = True) -> PathsConfig:
    """Load paths configuration from YAML file.

    Args:
        path: Path to the YAML configuration file
        validate: If True, validate that configured paths exist (default: True)

    Returns:
        PathsConfig with all paths resolved and expanded

    Raises:
        FileNotFoundError: If config file or any validated path doesn't exist
        TypeError: If configuration format is invalid
    """
    raw = load_yaml(path)
    paths = raw["paths"]
    base_dir = path.parent

    datasets_root = _expand_path(
        os.getenv("XFP_DATASETS_ROOT", paths["datasets_root"]),
        base_dir,
    )
    models_root = _expand_path(
        os.getenv("XFP_MODELS_ROOT", paths["models_root"]),
        base_dir,
    )
    fingerprints_root = _expand_path(
        os.getenv("XFP_FINGERPRINTS_ROOT", paths["fingerprints_root"]),
        base_dir,
    )
    cache_root = _expand_path(
        os.getenv("XFP_CACHE_ROOT", paths["cache_root"]),
        base_dir,
    )

    datasets_cfg = raw.get("datasets", {}) or {}
    datasets: Dict[str, DatasetPaths] = {}
    for name, cfg in datasets_cfg.items():
        if not isinstance(cfg, dict):
            raise TypeError(f"Dataset config for '{name}' must be a mapping.")
        images = _expand_path(str(cfg["images"]), datasets_root)
        masks = _expand_path(str(cfg["masks"]), datasets_root)
        datasets[name] = DatasetPaths(images=images, masks=masks)

    models_cfg = raw.get("models", {}) or {}
    models: Dict[str, Path] = {}
    for name, rel_path in models_cfg.items():
        override = os.getenv(_model_override_env(name))
        raw_path = override if override else str(rel_path)
        models[name] = _expand_path(raw_path, models_root)

    config = PathsConfig(
        datasets_root=datasets_root,
        models_root=models_root,
        fingerprints_root=fingerprints_root,
        cache_root=cache_root,
        datasets=datasets,
        models=models,
    )

    if validate:
        _validate_paths_config(config, config_file=path)

    return config


def _validate_paths_config(config: PathsConfig, config_file: Path) -> None:
    """Validate that configured paths exist, with helpful error messages.

    Args:
        config: The PathsConfig to validate
        config_file: Path to the config file (for error messages)

    Raises:
        FileNotFoundError: With detailed message about which path is missing
    """
    missing_paths: List[str] = []

    # Validate root directories (warnings only - they may be created)
    for name, root_path in [
        ("datasets_root", config.datasets_root),
        ("models_root", config.models_root),
    ]:
        if not root_path.exists():
            warnings.warn(
                f"Root directory '{name}' does not exist: {root_path}\n"
                f"Update {config_file} if this is incorrect.",
                UserWarning,
                stacklevel=3,
            )

    # Validate model checkpoints (errors - required for experiments)
    for model_name, model_path in config.models.items():
        if not model_path.exists():
            missing_paths.append(
                f"  Model '{model_name}': {model_path}"
            )

    # Validate dataset paths (warnings - some datasets may be optional)
    for dataset_name, dataset_paths in config.datasets.items():
        if not dataset_paths.images.exists():
            warnings.warn(
                f"Dataset '{dataset_name}' images directory not found: {dataset_paths.images}",
                UserWarning,
                stacklevel=3,
            )
        if not dataset_paths.masks.exists():
            warnings.warn(
                f"Dataset '{dataset_name}' masks directory not found: {dataset_paths.masks}",
                UserWarning,
                stacklevel=3,
            )

    if missing_paths:
        if os.getenv("XFP_ALLOW_MISSING_MODELS") == "1":
            warnings.warn(
                "Missing model checkpoints (continuing because XFP_ALLOW_MISSING_MODELS=1):\n"
                + "\n".join(missing_paths),
                UserWarning,
                stacklevel=3,
            )
            return
        raise FileNotFoundError(
            f"Model checkpoints not found. Please update {config_file}:\n"
            + "\n".join(missing_paths)
            + "\n\nEnsure model files exist at the specified paths."
        )


def load_experiment_config(path: Path, experiment_key: str) -> ExperimentConfig:
    raw = load_yaml(path)
    defaults = raw.get("defaults", {})
    experiment = raw["experiments"].get(experiment_key)
    if experiment is None:
        raise KeyError(f"Experiment '{experiment_key}' not found in {path}.")
    return ExperimentConfig(
        key=experiment_key,
        train_dataset=experiment["train_dataset"],
        test_datasets=experiment.get("test_datasets", []),
        subset=experiment.get("subset", "full"),
        attribution_methods=experiment.get(
            "attribution_methods", defaults.get("attribution_methods", [])
        ),
        fingerprint_metrics=experiment.get(
            "fingerprint_metrics", defaults.get("fingerprint_metrics", [])
        ),
        shift_metrics=experiment.get("shift_metrics", defaults.get("shift_metrics", [])),
        notes=experiment.get("notes"),
    )
