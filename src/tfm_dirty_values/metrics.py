from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score


def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if y_true.ndim != 1 or probabilities.ndim != 1 or len(y_true) != len(probabilities):
        raise ValueError("ECE expects one-dimensional arrays of equal length")
    if bins < 2:
        raise ValueError("ECE requires at least two bins")
    if not len(y_true):
        raise ValueError("ECE is undefined for an empty sample")

    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1)
    predictions = (probabilities >= 0.5).astype(np.int64)
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    ece = 0.0
    for index in range(bins):
        selected = indices == index
        if not selected.any():
            continue
        bin_accuracy = np.mean(y_true[selected] == predictions[selected])
        bin_confidence = np.mean(confidence[selected])
        ece += selected.sum() / len(y_true) * abs(float(bin_accuracy - bin_confidence))
    return float(ece)


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities must be within [0, 1]")
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "log_loss": float(
            log_loss(
                y_true,
                np.column_stack([1.0 - probabilities, probabilities]),
                labels=[0, 1],
            )
        ),
        "ece_10_equal_width": expected_calibration_error(y_true, probabilities, bins=10),
    }

