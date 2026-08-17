from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..config import resolve_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_model_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    verified: dict[str, Any] = {}
    for model_name in config["models"]["enabled"]:
        settings = config["models"][model_name]
        record: dict[str, Any] = {"package": settings["package"]}
        if "checkpoint" in settings:
            path = resolve_path(config, settings["checkpoint"])
            if not path.is_file():
                raise FileNotFoundError(f"Missing {model_name} checkpoint: {path}")
            actual = _sha256_file(path)
            expected = str(settings["checkpoint_sha256"]).upper()
            if actual != expected:
                raise ValueError(f"{model_name} checkpoint hash mismatch: {actual} != {expected}")
            record.update({"path": str(path), "sha256": actual, "bytes": path.stat().st_size})
        verified[str(model_name)] = record
    return verified


def _positive_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities)
    if values.ndim == 1:
        return values.astype(np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"Expected binary probabilities, received shape {values.shape}")
    return values[:, 1].astype(np.float64)


def _tabicl_empty_feature_compatibility(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    empty_mask = np.isnan(X_train).all(axis=0)
    indices = np.flatnonzero(empty_mask)
    if not len(indices):
        return X_train, X_test, {"applied": False, "feature_indices": []}
    train_output = X_train.copy()
    test_output = X_test.copy()
    train_output[:, empty_mask] = 0.0
    test_output[:, empty_mask] = 0.0
    return train_output, test_output, {
        "applied": True,
        "reason": "tabicl_2_1_1_empty_context_feature_compatibility",
        "feature_indices": indices.astype(int).tolist(),
        "fill_value": 0.0,
    }


def predict_probabilities(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    categorical_indices: list[int],
    seed: int,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {"model": model_name, "seed": int(seed)}
    model: Any = None
    try:
        if model_name == "tabpfn":
            from tabpfn import TabPFNClassifier

            settings = config["models"]["tabpfn"]
            model = TabPFNClassifier(
                n_estimators=int(settings["n_estimators"]),
                categorical_features_indices=categorical_indices or None,
                model_path=resolve_path(config, settings["checkpoint"]),
                device=settings["device"],
                random_state=int(seed),
                fit_mode="low_memory",
                n_jobs=1,
            )
            metadata.update(
                {
                    "package": settings["package"],
                    "checkpoint_sha256": settings["checkpoint_sha256"],
                    "n_estimators": int(settings["n_estimators"]),
                    "device": settings["device"],
                }
            )
        elif model_name == "tabiclv2":
            from tabicl import TabICLClassifier

            settings = config["models"]["tabiclv2"]
            X_train, X_test, compatibility = _tabicl_empty_feature_compatibility(X_train, X_test)
            model = TabICLClassifier(
                n_estimators=int(settings["n_estimators"]),
                batch_size=int(settings["batch_size"]),
                model_path=resolve_path(config, settings["checkpoint"]),
                allow_auto_download=False,
                checkpoint_version=settings["checkpoint_version"],
                device=settings["device"],
                random_state=int(seed),
                verbose=False,
            )
            metadata.update(
                {
                    "package": settings["package"],
                    "checkpoint_sha256": settings["checkpoint_sha256"],
                    "checkpoint_version": settings["checkpoint_version"],
                    "n_estimators": int(settings["n_estimators"]),
                    "device": settings["device"],
                    "empty_feature_compatibility": compatibility,
                }
            )
        elif model_name == "xgboost":
            from xgboost import XGBClassifier

            settings = config["models"]["xgboost"]
            model = XGBClassifier(
                n_estimators=int(settings["n_estimators"]),
                tree_method=settings["tree_method"],
                n_jobs=int(settings["n_jobs"]),
                random_state=int(seed),
                eval_metric="logloss",
            )
            metadata.update({"package": settings["package"], "n_estimators": settings["n_estimators"]})
        elif model_name == "lightgbm":
            from lightgbm import LGBMClassifier

            settings = config["models"]["lightgbm"]
            model = LGBMClassifier(
                n_estimators=int(settings["n_estimators"]),
                n_jobs=int(settings["n_jobs"]),
                random_state=int(seed),
                verbosity=-1,
            )
            metadata.update({"package": settings["package"], "n_estimators": settings["n_estimators"]})
        elif model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.impute import SimpleImputer

            settings = config["models"]["random_forest"]
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            model = RandomForestClassifier(
                n_estimators=int(settings["n_estimators"]),
                n_jobs=int(settings["n_jobs"]),
                random_state=int(seed),
            )
            metadata.update(
                {
                    "package": settings["package"],
                    "n_estimators": settings["n_estimators"],
                    "native_missing_treatment": "context_fitted_median",
                }
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train, y_train)
        probabilities = _positive_probabilities(model.predict_proba(X_test))
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{model_name} returned non-finite probabilities")
        return probabilities, metadata
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

