"""Unit tests for shift metric calculations."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from xfp.shift.divergence import bootstrap_shift_scores, compute_shift_scores


def _make_dataframe(
    weights: List[List[float]],
    coverage: List[List[float]],
    components: List[List[float]],
) -> pd.DataFrame:
    bins = len(weights[0])
    hist_cols = {f"hist_bin_{idx:02d}": [row[idx] for row in weights] for idx in range(bins)}
    coverage_cols = {
        f"coverage_q_{i/10:.2f}": [row[i] for row in coverage]
        for i in range(len(coverage[0]))
    }
    component_cols = {
        name: [row[idx] for row in components]
        for idx, name in enumerate(
            [
                "component_count",
                "component_mean_size",
                "component_median_size",
                "component_largest_size",
                "component_border_fraction",
                "component_border_mass_fraction",
            ]
        )
    }
    return pd.DataFrame({**hist_cols, **coverage_cols, **component_cols})


def test_compute_shift_scores(tmp_path: Path) -> None:
    # Reference distribution skewed towards lower attribution bins.
    ref_weights = [
        [0.7, 0.2, 0.1, 0.0],
        [0.6, 0.25, 0.1, 0.05],
    ]
    ref_coverage = [
        np.linspace(0.0, 1.0, 11).tolist(),
        np.linspace(0.0, 1.0, 11).tolist(),
    ]
    ref_components = [
        [3, 120, 100, 200, 0.8, 0.7],
        [4, 110, 90, 210, 0.75, 0.65],
    ]

    # Target distribution emphasises higher attribution bins and fewer components.
    tgt_weights = [
        [0.2, 0.3, 0.3, 0.2],
        [0.1, 0.3, 0.4, 0.2],
    ]
    tgt_coverage = [
        (np.linspace(0.0, 1.0, 11) ** 0.5).tolist(),
        (np.linspace(0.0, 1.0, 11) ** 0.55).tolist(),
    ]
    tgt_components = [
        [2, 90, 80, 150, 0.4, 0.3],
        [2, 95, 85, 155, 0.42, 0.32],
    ]

    ref_df = _make_dataframe(ref_weights, ref_coverage, ref_components)
    tgt_df = _make_dataframe(tgt_weights, tgt_coverage, tgt_components)

    ref_path = tmp_path / "ref.parquet"
    tgt_path = tmp_path / "tgt.parquet"
    ref_df.to_parquet(ref_path, index=False)
    tgt_df.to_parquet(tgt_path, index=False)

    scores = compute_shift_scores(
        reference=ref_path,
        target=tgt_path,
        metrics=["emd", "kl_divergence", "graph_edit_distance"],
    )

    assert scores.scores["emd"] > 0.0
    assert scores.scores["kl_divergence"] > 0.0
    assert scores.scores["graph_edit_distance"] > 0.0

    bootstrap_df = bootstrap_shift_scores(
        reference=ref_path,
        target=tgt_path,
        metrics=["emd", "kl_divergence", "graph_edit_distance"],
        n_resamples=16,
        random_state=42,
    )

    assert list(bootstrap_df.columns) == ["emd", "kl_divergence", "graph_edit_distance"]
    assert len(bootstrap_df) == 16
    assert bootstrap_df["emd"].mean() > 0.0
    assert bootstrap_df["graph_edit_distance"].mean() > 0.0
