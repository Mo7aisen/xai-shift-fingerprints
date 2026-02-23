"""Unit tests for shared OOD evaluation metrics."""

import numpy as np

from xfp.utils.ood_eval import (
    average_precision,
    binary_ood_metrics,
    binary_ood_metrics_with_bootstrap,
    fpr_at_tpr,
    roc_auc_rank,
    tpr_at_fpr,
)


def test_perfect_separation_metrics() -> None:
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    s = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0], dtype=float)

    assert np.isclose(roc_auc_rank(y, s), 1.0)
    assert np.isclose(average_precision(y, s), 1.0)
    assert np.isclose(fpr_at_tpr(y, s, target_tpr=0.95), 0.0)
    assert np.isclose(tpr_at_fpr(y, s, max_fpr=0.05), 1.0)

    m = binary_ood_metrics(y, s)
    assert np.isclose(m["auc"], 1.0)
    assert np.isclose(m["aupr"], 1.0)
    assert np.isclose(m["fpr95"], 0.0)
    assert np.isclose(m["tpr_at_fpr05"], 1.0)
    assert 0.0 <= m["ece"] <= 1.0
    assert 0.0 <= m["brier"] <= 1.0


def test_bootstrap_metrics_ci_fields_present_and_ordered() -> None:
    rng = np.random.default_rng(123)
    y = np.array([0] * 40 + [1] * 40, dtype=int)
    s = np.concatenate(
        [
            rng.normal(loc=0.0, scale=1.0, size=40),
            rng.normal(loc=1.0, scale=1.0, size=40),
        ]
    )

    res = binary_ood_metrics_with_bootstrap(y, s, n_boot=64, seed=7)

    for key in [
        "auc",
        "aupr",
        "fpr95",
        "tpr_at_fpr05",
        "ece",
        "brier",
        "auc_ci_low",
        "auc_ci_high",
        "aupr_ci_low",
        "aupr_ci_high",
        "fpr95_ci_low",
        "fpr95_ci_high",
        "tpr_at_fpr05_ci_low",
        "tpr_at_fpr05_ci_high",
    ]:
        assert key in res
        assert np.isfinite(res[key]) or np.isnan(res[key])

    assert res["auc_ci_low"] <= res["auc_ci_high"]
    assert res["aupr_ci_low"] <= res["aupr_ci_high"]
    assert res["fpr95_ci_low"] <= res["fpr95_ci_high"]
    assert res["tpr_at_fpr05_ci_low"] <= res["tpr_at_fpr05_ci_high"]

