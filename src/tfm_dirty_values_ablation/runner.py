"""Phase 3 task runner.

Subclasses the frozen Phase 2 ``ExperimentRunner`` and inserts exactly one step:
after corruption injection, optionally apply the fixed cleaner.

With ``cleaned=False`` this runner is behaviourally identical to Phase 2. That
property is what makes the preregistered paired-identity probe possible: running
a Phase 2 arm-B cell through this runner with ``cleaned=False`` must reproduce
the frozen Phase 2 metrics exactly.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from typing import Any

import numpy as np

from tfm_dirty_values.experiment import ExperimentRunner
from tfm_dirty_values.injection import inject_corruption
from tfm_dirty_values.metrics import classification_metrics
from tfm_dirty_values.models import predict_probabilities

from .cleaning import clean_partitions


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


class AblationExperimentRunner(ExperimentRunner):
    """Phase 2 pipeline plus an optional cleaning stage."""

    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        split = self.datasets.prepare_split(int(task["dataset_id"]), int(task["seed"]))
        parameters = task.get("parameters", {})
        if isinstance(parameters, str):
            parameters = json.loads(parameters)

        cleaned = bool(task.get("cleaned", False))
        arm = str(task.get("arm", "c" if cleaned else "b"))

        X_train, X_test, injection_metadata = inject_corruption(
            split.X_train,
            split.X_test,
            dataset_id=int(task["dataset_id"]),
            split_seed=int(task["seed"]),
            corruption=str(task["corruption"]),
            rate=float(task["rate"]),
            categorical_indices=split.categorical_indices,
            params=parameters,
        )

        dirty_train_sha256 = _array_sha256(X_train)
        dirty_test_sha256 = _array_sha256(X_test)

        cleaning_metadata: dict[str, Any] | None = None
        if cleaned:
            cleaning_started = time.perf_counter()
            X_train, X_test, cleaning_metadata = clean_partitions(
                X_train,
                X_test,
                categorical_indices=split.categorical_indices,
            )
            cleaning_metadata["cleaning_seconds"] = float(time.perf_counter() - cleaning_started)

        model_started = time.perf_counter()
        probabilities, model_metadata = predict_probabilities(
            str(task["model"]),
            X_train,
            split.y_train,
            X_test,
            categorical_indices=split.categorical_indices,
            seed=int(task["seed"]),
            config=self.config,
        )
        model_seconds = time.perf_counter() - model_started

        metrics = classification_metrics(split.y_test, probabilities)
        metrics.update(
            {
                "model_seconds": float(model_seconds),
                "arm": arm,
                "cleaned": cleaned,
                "injection": injection_metadata,
                "cleaning": cleaning_metadata,
                "model_provenance": model_metadata,
                "input_provenance": {
                    "clean_X_train_sha256": split.metadata["X_train_sha256"],
                    "clean_X_test_sha256": split.metadata["X_test_sha256"],
                    "dirty_X_train_sha256": dirty_train_sha256,
                    "dirty_X_test_sha256": dirty_test_sha256,
                    "model_X_train_sha256": _array_sha256(X_train),
                    "model_X_test_sha256": _array_sha256(X_test),
                    "train_indices_sha256": split.metadata["train_indices_sha256"],
                    "test_indices_sha256": split.metadata["test_indices_sha256"],
                },
                "host": {"platform": platform.platform(), "python": platform.python_version()},
            }
        )
        return metrics
