from __future__ import annotations

import pytest

from tfm_dirty_values_ablation.analysis import (
    METRICS,
    _improvement,
    _normalize_metrics,
)


def test_improvement_sign_follows_metric_direction() -> None:
    # Higher is better: a larger candidate is an improvement.
    assert _improvement("roc_auc", 0.70, 0.80) == pytest.approx(0.10)
    assert _improvement("balanced_accuracy", 0.60, 0.55) == pytest.approx(-0.05)
    # Lower is better: a smaller candidate is an improvement.
    assert _improvement("log_loss", 0.80, 0.50) == pytest.approx(0.30)
    assert _improvement("ece", 0.10, 0.15) == pytest.approx(-0.05)


def test_improvement_is_antisymmetric() -> None:
    for metric in METRICS:
        forward = _improvement(metric, 0.4, 0.9)
        backward = _improvement(metric, 0.9, 0.4)
        assert forward == pytest.approx(-backward)


def test_normalize_metrics_applies_the_registered_ece_alias() -> None:
    payload = {
        "roc_auc": 0.72,
        "balanced_accuracy": 0.61,
        "log_loss": 0.75,
        "ece_10_equal_width": 0.1765,
    }
    normalized = _normalize_metrics(payload, "t1")
    assert normalized["ece"] == pytest.approx(0.1765)
    assert set(normalized) == set(METRICS)


def test_normalize_metrics_prefers_an_explicit_generic_key() -> None:
    payload = {
        "roc_auc": 0.5,
        "balanced_accuracy": 0.5,
        "log_loss": 0.5,
        "ece": 0.25,
        "ece_10_equal_width": 0.25,
    }
    assert _normalize_metrics(payload, "t2")["ece"] == pytest.approx(0.25)


def test_normalize_metrics_rejects_conflicting_ece_values() -> None:
    payload = {
        "roc_auc": 0.5,
        "balanced_accuracy": 0.5,
        "log_loss": 0.5,
        "ece": 0.10,
        "ece_10_equal_width": 0.20,
    }
    with pytest.raises(ValueError, match="Conflicting ECE"):
        _normalize_metrics(payload, "t3")


def test_normalize_metrics_rejects_a_missing_ece() -> None:
    payload = {"roc_auc": 0.5, "balanced_accuracy": 0.5, "log_loss": 0.5}
    with pytest.raises(ValueError, match="neither ece"):
        _normalize_metrics(payload, "t4")


def test_recovery_and_residual_decompose_the_phase2_damage() -> None:
    """recovery + residual must equal the uncleaned damage, per metric.

    This identity is what lets the ablation report a recovery fraction at all,
    so it is asserted directly rather than assumed.
    """
    clean, uncleaned, cleaned = 0.90, 0.70, 0.82
    for metric in METRICS:
        recovery = _improvement(metric, uncleaned, cleaned)
        residual = _improvement(metric, cleaned, clean)
        damage = _improvement(metric, uncleaned, clean)
        assert recovery + residual == pytest.approx(damage)
