"""Benchmark pipeline for attribution shift detection."""

from __future__ import annotations

import json
from pathlib import Path

from xfp.config import ExperimentConfig, PathsConfig
from xfp.fingerprint.runner import run_fingerprint_experiment
from xfp.shift.divergence import compute_shift_scores


def run_benchmark_suite(exp_cfg: ExperimentConfig, paths_cfg: PathsConfig, output_dir: Path) -> None:
    """Execute fingerprint generation and compute shift diagnostics."""

    output_dir = output_dir / exp_cfg.key
    output_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = run_fingerprint_experiment(exp_cfg=exp_cfg, paths_cfg=paths_cfg, device="cuda")

    # Persist per-dataset parquet tables alongside the report for reproducibility.
    for dataset, parquet_path in fingerprint.dataset_tables.items():
        target_path = output_dir / f"{dataset}.parquet"
        if parquet_path.exists():
            target_path.write_bytes(parquet_path.read_bytes())

    benchmark = {
        "experiment": exp_cfg.key,
        "train_dataset": exp_cfg.train_dataset,
        "test_datasets": [],
        "shift_metrics": {},
        "fingerprint_summary_path": str(fingerprint.fingerprint_path),
    }

    reference_path = fingerprint.dataset_tables[exp_cfg.train_dataset]
    for dataset in exp_cfg.test_datasets:
        target_path = fingerprint.dataset_tables.get(dataset)
        if target_path is None:
            continue
        shift_scores = compute_shift_scores(reference_path, target_path, metrics=exp_cfg.shift_metrics)
        benchmark["test_datasets"].append(dataset)
        benchmark["shift_metrics"][dataset] = shift_scores.scores

    report_path = output_dir / "benchmark_metrics.json"
    report_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
