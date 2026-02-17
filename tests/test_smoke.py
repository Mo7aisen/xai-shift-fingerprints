"""Smoke tests for early pipeline scaffolding."""

from __future__ import annotations

from pathlib import Path
import textwrap

from xfp.config import load_experiment_config, load_paths_config


def test_configs_round_trip(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    models_root = tmp_path / "models"
    fingerprints_root = tmp_path / "fingerprints"
    cache_root = tmp_path / "cache"
    images_dir = datasets_root / "JSRT" / "images"
    masks_dir = datasets_root / "JSRT" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    models_root.mkdir(parents=True)
    (models_root / "jsrt_unet_baseline.pt").write_text("stub", encoding="utf-8")

    paths_yaml = textwrap.dedent(
        f"""
        paths:
          datasets_root: {datasets_root}
          models_root: {models_root}
          fingerprints_root: {fingerprints_root}
          cache_root: {cache_root}

        datasets:
          jsrt:
            images: JSRT/images
            masks: JSRT/masks

        models:
          unet_jsrt_full: jsrt_unet_baseline.pt
        """
    ).strip()

    experiments_yaml = textwrap.dedent(
        """
        defaults:
          attribution_methods: [integrated_gradients]
          fingerprint_metrics: []
          shift_metrics: []
        experiments:
          jsrt_baseline:
            train_dataset: jsrt
            test_datasets: []
            subset: full
        """
    ).strip()

    paths_path = tmp_path / "paths.yaml"
    experiments_path = tmp_path / "experiments.yaml"
    paths_path.write_text(paths_yaml, encoding="utf-8")
    experiments_path.write_text(experiments_yaml, encoding="utf-8")

    paths_cfg = load_paths_config(paths_path)
    exp_cfg = load_experiment_config(experiments_path, "jsrt_baseline")

    assert paths_cfg.datasets_root == datasets_root
    assert "jsrt" in paths_cfg.datasets
    assert paths_cfg.datasets["jsrt"].images.exists()
    assert paths_cfg.datasets["jsrt"].masks.exists()
    assert exp_cfg.train_dataset == "jsrt"
