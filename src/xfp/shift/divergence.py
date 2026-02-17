"""Dataset shift metrics for attribution fingerprints."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.stats import entropy, wasserstein_distance

HIST_COLUMN_PREFIX = "hist_bin_"
COVERAGE_COLUMN_PREFIX = "coverage_q_"
COMPONENT_FEATURES = [
    "component_count",
    "component_mean_size",
    "component_median_size",
    "component_largest_size",
    "component_border_fraction",
    "component_border_mass_fraction",
]


def _columns_with_prefix(df: pd.DataFrame, prefix: str) -> list[str]:
    return sorted([col for col in df.columns if col.startswith(prefix)])


def _require_matching_prefix_columns(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    *,
    prefix: str,
    metric: str,
) -> list[str]:
    ref_cols = _columns_with_prefix(ref_df, prefix)
    tgt_cols = _columns_with_prefix(tgt_df, prefix)
    if not ref_cols or not tgt_cols:
        raise ValueError(
            f"Shift metric '{metric}' requires columns starting with '{prefix}', "
            f"but none were found in {'reference' if not ref_cols else 'target'} data."
        )
    if ref_cols != tgt_cols:
        ref_only = sorted(set(ref_cols) - set(tgt_cols))
        tgt_only = sorted(set(tgt_cols) - set(ref_cols))
        raise ValueError(
            f"Shift metric '{metric}' requires matching '{prefix}' columns. "
            f"Ref-only: {ref_only or 'none'}, Target-only: {tgt_only or 'none'}."
        )
    return ref_cols


def _shared_component_columns(ref_df: pd.DataFrame, tgt_df: pd.DataFrame) -> list[str]:
    ref_cols = [col for col in COMPONENT_FEATURES if col in ref_df.columns]
    tgt_cols = [col for col in COMPONENT_FEATURES if col in tgt_df.columns]
    shared = [col for col in COMPONENT_FEATURES if col in ref_df.columns and col in tgt_df.columns]
    if not shared:
        raise ValueError(
            "Shift metric 'graph_edit_distance' requires component feature columns "
            f"{COMPONENT_FEATURES}, but none were found in both tables."
        )
    missing_ref = [col for col in COMPONENT_FEATURES if col not in ref_df.columns]
    missing_tgt = [col for col in COMPONENT_FEATURES if col not in tgt_df.columns]
    if missing_ref or missing_tgt:
        warnings.warn(
            "Component feature columns missing for graph_edit_distance. "
            f"Missing in reference: {missing_ref or 'none'}, "
            f"missing in target: {missing_tgt or 'none'}. "
            "Using shared columns only.",
            UserWarning,
            stacklevel=3,
        )
    return shared


@dataclass
class ShiftScores:
    scores: Dict[str, float]

    def model_dump_json(self, indent: int = 2) -> str:
        return json.dumps({"scores": self.scores}, indent=indent)


def compute_shift_scores(reference: Path, target: Path, metrics: Iterable[str]) -> ShiftScores:
    """Compute requested shift metrics between two fingerprint tables on disk."""
    ref_df = _load_fingerprint_table(reference)
    tgt_df = _load_fingerprint_table(target)
    return compute_shift_scores_from_frames(ref_df, tgt_df, metrics)


def compute_shift_scores_from_frames(
    ref_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    metrics: Iterable[str],
) -> ShiftScores:
    """Compute shift metrics from in-memory fingerprint dataframes."""
    scores: Dict[str, float] = {}
    for metric in metrics:
        if metric == "emd":
            _require_matching_prefix_columns(ref_df, tgt_df, prefix=HIST_COLUMN_PREFIX, metric=metric)
            scores[metric] = _histogram_emd(ref_df, tgt_df)
        elif metric == "kl_divergence":
            _require_matching_prefix_columns(ref_df, tgt_df, prefix=COVERAGE_COLUMN_PREFIX, metric=metric)
            scores[metric] = _coverage_kl(ref_df, tgt_df)
        elif metric == "graph_edit_distance":
            scores[metric] = _component_distance(ref_df, tgt_df)
        else:
            raise KeyError(f"Unknown shift metric '{metric}'")
    return ShiftScores(scores=scores)


def bootstrap_shift_scores(
    reference: Path,
    target: Path,
    metrics: Iterable[str],
    *,
    n_resamples: int = 1000,
    random_state: int,
) -> pd.DataFrame:
    """Bootstrap divergence metrics by resampling reference and target fingerprints.

    IMPORTANT: random_state is required to ensure reproducible confidence intervals.
    For publication results, always use a fixed seed (e.g., 2025).

    Args:
        reference: Path to reference fingerprint parquet file
        target: Path to target fingerprint parquet file
        metrics: Iterable of metric names to compute
        n_resamples: Number of bootstrap iterations (default: 1000)
        random_state: Random seed for reproducibility (REQUIRED)

    Returns:
        DataFrame with one column per metric and ``n_resamples`` rows of bootstrap samples.

    Raises:
        ValueError: If no metrics requested or tables are empty
    """
    metric_list = list(metrics)
    if not metric_list:
        raise ValueError("No metrics requested for bootstrapping.")

    ref_df = _load_fingerprint_table(reference)
    tgt_df = _load_fingerprint_table(target)
    if ref_df.empty or tgt_df.empty:
        raise ValueError("Cannot bootstrap with empty fingerprint tables.")

    rng = np.random.default_rng(random_state)
    ref_indices = np.arange(len(ref_df))
    tgt_indices = np.arange(len(tgt_df))

    samples: Dict[str, List[float]] = {metric: [] for metric in metric_list}
    for _ in range(n_resamples):
        ref_draw = ref_df.iloc[rng.choice(ref_indices, size=len(ref_indices), replace=True)]
        tgt_draw = tgt_df.iloc[rng.choice(tgt_indices, size=len(tgt_indices), replace=True)]
        scores = compute_shift_scores_from_frames(ref_draw, tgt_draw, metric_list)
        for metric in metric_list:
            samples[metric].append(scores.scores[metric])

    return pd.DataFrame(samples)


def _load_fingerprint_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fingerprint table not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported fingerprint file format: {path.suffix}")


def _histogram_emd(ref_df: pd.DataFrame, tgt_df: pd.DataFrame) -> float:
    ref_hist = _mean_histogram(ref_df)
    tgt_hist = _mean_histogram(tgt_df)
    if ref_hist is None or tgt_hist is None:
        return 0.0
    bins = np.arange(ref_hist.shape[0], dtype=float)
    return float(
        wasserstein_distance(
            bins,
            bins,
            u_weights=ref_hist,
            v_weights=tgt_hist,
        )
    )


def _mean_histogram(df: pd.DataFrame) -> np.ndarray | None:
    hist_cols = sorted(
        (col for col in df.columns if col.startswith(HIST_COLUMN_PREFIX)),
        key=lambda name: int(name.split("_")[-1]),
    )
    if not hist_cols:
        return None
    hist = df[hist_cols].fillna(0.0).to_numpy(dtype=float)
    mean_hist = hist.mean(axis=0)
    total = mean_hist.sum()
    if total <= 0:
        return mean_hist
    return mean_hist / total


def _coverage_kl(ref_df: pd.DataFrame, tgt_df: pd.DataFrame) -> float:
    coverage_axis, ref_curve = _mean_coverage_curve(ref_df)
    _, tgt_curve = _mean_coverage_curve(tgt_df)
    ref_pdf = _cdf_to_pdf(ref_curve, coverage_axis)
    tgt_pdf = _cdf_to_pdf(tgt_curve, coverage_axis)
    ref_pdf = _normalise_prob(ref_pdf)
    tgt_pdf = _normalise_prob(tgt_pdf)
    eps = 1e-8
    ref_pdf = np.clip(ref_pdf, eps, None)
    tgt_pdf = np.clip(tgt_pdf, eps, None)
    kl_forward = float(entropy(ref_pdf, tgt_pdf))
    kl_reverse = float(entropy(tgt_pdf, ref_pdf))
    return 0.5 * (kl_forward + kl_reverse)


def _mean_coverage_curve(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    coverage_cols = [
        col for col in df.columns if col.startswith(COVERAGE_COLUMN_PREFIX)
    ]
    if not coverage_cols:
        axis = np.linspace(0.0, 1.0, 11)
        return axis, np.ones_like(axis)

    def _extract_fraction(name: str) -> float:
        return float(name.split("_")[-1])

    sorted_cols = sorted(coverage_cols, key=_extract_fraction)
    axis = np.array([_extract_fraction(col) for col in sorted_cols], dtype=float)
    coverage_df = df[sorted_cols].copy()
    coverage_df = coverage_df.ffill(axis=1).fillna(0.0)
    curve = coverage_df.mean(axis=0).to_numpy(dtype=float)
    return axis, curve


def _cdf_to_pdf(cdf: np.ndarray, axis: np.ndarray) -> np.ndarray:
    diffs = np.diff(cdf, prepend=cdf[0])
    width = np.diff(axis, prepend=axis[0])
    width[width == 0] = 1.0
    pdf = diffs / width
    return np.clip(pdf, 0.0, None)


def _normalise_prob(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    if total <= 0:
        return np.ones_like(values) / values.size
    return values / total


def _component_distance(ref_df: pd.DataFrame, tgt_df: pd.DataFrame) -> float:
    shared = _shared_component_columns(ref_df, tgt_df)
    ref_vec = ref_df[shared].mean(axis=0).to_numpy(dtype=float)
    tgt_vec = tgt_df[shared].mean(axis=0).to_numpy(dtype=float)
    return float(euclidean(ref_vec, tgt_vec))
