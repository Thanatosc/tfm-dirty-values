from __future__ import annotations

import hashlib
import json
import platform
import time
from typing import Any

import numpy as np

from .datasets import DatasetManager
from .injection import inject_corruption
from .metrics import classification_metrics
from .models import predict_probabilities


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


class ExperimentRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.datasets = DatasetManager(config)

    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        split = self.datasets.prepare_split(int(task["dataset_id"]), int(task["seed"]))
        parameters = task.get("parameters", {})
        if isinstance(parameters, str):
            parameters = json.loads(parameters)
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
                "injection": injection_metadata,
                "model_provenance": model_metadata,
                "input_provenance": {
                    "clean_X_train_sha256": split.metadata["X_train_sha256"],
                    "clean_X_test_sha256": split.metadata["X_test_sha256"],
                    "model_X_train_sha256": _array_sha256(X_train),
                    "model_X_test_sha256": _array_sha256(X_test),
                    "train_indices_sha256": split.metadata["train_indices_sha256"],
                    "test_indices_sha256": split.metadata["test_indices_sha256"],
                },
                "host": {"platform": platform.platform(), "python": platform.python_version()},
            }
        )
        return metrics
