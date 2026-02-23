"""Shared OOD evaluation metrics used across gates and baseline scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import rankdata


ArrayLike = np.ndarray


@dataclass(frozen=True)
class MetricWithCI:
    point: float
    ci_low: float
    ci_high: float


def roc_auc_rank(y: ArrayLike, score: ArrayLike) -> float:
    """Rank-based AUROC implementation without sklearn dependency."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score, method="average")
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def average_precision(y: ArrayLike, score: ArrayLike) -> float:
    """Average precision / area under PR curve for binary labels."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    ap = 0.0
    prev_recall = 0.0
    for p, r, yi in zip(precision, recall, y_sorted):
        if yi == 1:
            ap += float(p) * float(r - prev_recall)
            prev_recall = float(r)
    return float(ap)


def _roc_curve_arrays(y: ArrayLike, score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    order = np.argsort(-score)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    return fpr, tpr


def fpr_at_tpr(y: ArrayLike, score: ArrayLike, target_tpr: float = 0.95) -> float:
    """Minimum FPR reaching at least target TPR."""
    fpr, tpr = _roc_curve_arrays(y, score)
    if fpr.size == 0:
        return float("nan")
    mask = tpr >= float(target_tpr)
    if not np.any(mask):
        return float("nan")
    return float(np.min(fpr[mask]))


def tpr_at_fpr(y: ArrayLike, score: ArrayLike, max_fpr: float = 0.05) -> float:
    """Maximum TPR while keeping FPR <= threshold."""
    fpr, tpr = _roc_curve_arrays(y, score)
    if fpr.size == 0:
        return float("nan")
    mask = fpr <= float(max_fpr)
    if not np.any(mask):
        return float(0.0)
    return float(np.max(tpr[mask]))


def _minmax_prob(score: ArrayLike) -> np.ndarray:
    score = np.asarray(score, dtype=float)
    smin = float(np.min(score))
    smax = float(np.max(score))
    if not np.isfinite(smin) or not np.isfinite(smax) or smax <= smin:
        return np.full_like(score, 0.5, dtype=float)
    return (score - smin) / (smax - smin)


def ece_from_scores(y: ArrayLike, score: ArrayLike, n_bins: int = 10) -> float:
    """Expected calibration error after min-max score scaling."""
    y = np.asarray(y, dtype=int)
    p = _minmax_prob(score)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not np.any(m):
            continue
        acc = float(np.mean(y[m]))
        conf = float(np.mean(p[m]))
        w = float(np.mean(m))
        ece += w * abs(acc - conf)
    return float(ece)


def brier_from_scores(y: ArrayLike, score: ArrayLike) -> float:
    """Brier score after min-max score scaling."""
    y = np.asarray(y, dtype=int)
    p = _minmax_prob(score)
    return float(np.mean((p - y) ** 2))


def binary_ood_metrics(y: ArrayLike, score: ArrayLike) -> dict[str, float]:
    """Compute reviewer-facing OOD metrics from labels and scalar scores."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    return {
        "auc": roc_auc_rank(y, score),
        "aupr": average_precision(y, score),
        "fpr95": fpr_at_tpr(y, score, target_tpr=0.95),
        "tpr_at_fpr05": tpr_at_fpr(y, score, max_fpr=0.05),
        "ece": ece_from_scores(y, score),
        "brier": brier_from_scores(y, score),
    }


def _stratified_boot_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if idx0.size == 0 or idx1.size == 0:
        return np.asarray([], dtype=int)
    b0 = rng.choice(idx0, size=idx0.size, replace=True)
    b1 = rng.choice(idx1, size=idx1.size, replace=True)
    return np.concatenate([b0, b1])


def bootstrap_metric_ci(
    y: ArrayLike,
    score: ArrayLike,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 500,
    seed: int = 42,
) -> MetricWithCI:
    """Stratified bootstrap CI for a scalar metric."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    point = float(metric_fn(y, score))
    rng = np.random.default_rng(seed)
    boot_vals: list[float] = []
    for _ in range(int(n_boot)):
        idx = _stratified_boot_indices(y, rng)
        if idx.size == 0:
            continue
        value = float(metric_fn(y[idx], score[idx]))
        if np.isfinite(value):
            boot_vals.append(value)
    if not boot_vals:
        return MetricWithCI(point=point, ci_low=float("nan"), ci_high=float("nan"))
    ci_low, ci_high = np.percentile(np.asarray(boot_vals, dtype=float), [2.5, 97.5])
    return MetricWithCI(point=point, ci_low=float(ci_low), ci_high=float(ci_high))


def binary_ood_metrics_with_bootstrap(
    y: ArrayLike,
    score: ArrayLike,
    *,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Compute binary OOD metrics plus bootstrap CIs for threshold-free and threshold metrics."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    point = binary_ood_metrics(y, score)

    auc_ci = bootstrap_metric_ci(y, score, roc_auc_rank, n_boot=n_boot, seed=seed)
    aupr_ci = bootstrap_metric_ci(y, score, average_precision, n_boot=n_boot, seed=seed + 1)
    fpr95_ci = bootstrap_metric_ci(
        y, score, lambda yy, ss: fpr_at_tpr(yy, ss, target_tpr=0.95), n_boot=n_boot, seed=seed + 2
    )
    tpr05_ci = bootstrap_metric_ci(
        y, score, lambda yy, ss: tpr_at_fpr(yy, ss, max_fpr=0.05), n_boot=n_boot, seed=seed + 3
    )

    return {
        **point,
        "auc_ci_low": auc_ci.ci_low,
        "auc_ci_high": auc_ci.ci_high,
        "aupr_ci_low": aupr_ci.ci_low,
        "aupr_ci_high": aupr_ci.ci_high,
        "fpr95_ci_low": fpr95_ci.ci_low,
        "fpr95_ci_high": fpr95_ci.ci_high,
        "tpr_at_fpr05_ci_low": tpr05_ci.ci_low,
        "tpr_at_fpr05_ci_high": tpr05_ci.ci_high,
    }

