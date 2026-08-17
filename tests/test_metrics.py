import numpy as np
import pytest

from tfm_dirty_values.metrics import classification_metrics, expected_calibration_error


def test_metrics_return_finite_binary_scores() -> None:
    metrics = classification_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.4, 0.6, 0.9]),
    )
    assert metrics["roc_auc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert all(np.isfinite(value) for value in metrics.values())


def test_ece_rejects_misaligned_input() -> None:
    with pytest.raises(ValueError, match="equal length"):
        expected_calibration_error(np.asarray([0, 1]), np.asarray([0.2]))

