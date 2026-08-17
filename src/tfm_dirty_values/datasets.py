from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from .config import canonical_json, resolve_path


@dataclass
class PreparedSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    categorical_indices: list[int]
    metadata: dict[str, Any]


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


class DatasetManager:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.raw_dir = resolve_path(config, config["paths"]["data_raw"])
        self.processed_dir = resolve_path(config, config["paths"]["data_processed"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _dataset_config(self, data_id: int) -> dict[str, Any]:
        for dataset in self.config["datasets"]:
            if int(dataset["data_id"]) == int(data_id):
                return dataset
        raise KeyError(f"Unknown dataset ID: {data_id}")

    def fetch_and_validate(self, data_id: int) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
        expected = self._dataset_config(data_id)
        bunch = fetch_openml(
            data_id=data_id,
            as_frame=True,
            data_home=self.raw_dir,
            parser="auto",
        )
        X = bunch.data.copy()
        y = pd.Series(bunch.target, name="target").copy()
        details = dict(getattr(bunch, "details", {}) or {})
        constraints = self.config["dataset_constraints"]
        native_missing = float(X.isna().to_numpy().mean())
        classes = sorted(y.dropna().astype(str).unique().tolist())
        metadata = {
            "data_id": int(data_id),
            "expected_name": expected["name"],
            "openml_name": details.get("name") or expected["name"],
            "version": details.get("version"),
            "license": details.get("licence") or details.get("license"),
            "rows": int(X.shape[0]),
            "features": int(X.shape[1]),
            "classes": classes,
            "native_missing_fraction": native_missing,
            "source_url": f"https://www.openml.org/d/{data_id}",
        }
        errors: list[str] = []
        rows = constraints["rows"]
        features = constraints["features"]
        if not int(rows["min"]) <= X.shape[0] <= int(rows["max"]):
            errors.append(f"rows={X.shape[0]} outside [{rows['min']}, {rows['max']}]")
        if not int(features["min"]) <= X.shape[1] <= int(features["max"]):
            errors.append(f"features={X.shape[1]} outside [{features['min']}, {features['max']}]")
        if len(classes) != int(constraints["classes"]):
            errors.append(f"target has {len(classes)} classes")
        missing_limit = float(constraints["native_missing_fraction"]["max_exclusive"])
        if native_missing >= missing_limit:
            errors.append(f"native missing fraction={native_missing:.6f} is not below {missing_limit}")
        if constraints["reusable_license_required"] and not metadata["license"]:
            errors.append("OpenML metadata has no reusable license declaration")
        if errors:
            raise ValueError(f"Dataset {data_id} failed preregistered validation: {'; '.join(errors)}")
        return X, y, metadata

    @staticmethod
    def _encode_predictors(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, Any]]:
        train_columns: list[np.ndarray] = []
        test_columns: list[np.ndarray] = []
        categorical_indices: list[int] = []
        encodings: dict[str, Any] = {}
        for index, column in enumerate(X_train.columns):
            train_series = X_train[column]
            test_series = X_test[column]
            is_categorical = (
                isinstance(train_series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(train_series.dtype)
                or pd.api.types.is_string_dtype(train_series.dtype)
                or pd.api.types.is_bool_dtype(train_series.dtype)
            )
            if is_categorical:
                categorical_indices.append(index)
                categories = sorted(train_series.dropna().astype(str).unique().tolist())
                mapping = {value: float(position) for position, value in enumerate(categories)}

                def encode(series: pd.Series) -> np.ndarray:
                    missing = series.isna().to_numpy()
                    values = series.astype(str).map(mapping).fillna(-1.0).to_numpy(dtype=np.float64)
                    values[missing] = np.nan
                    return values

                train_values = encode(train_series)
                test_values = encode(test_series)
                encodings[str(column)] = {
                    "type": "categorical",
                    "categories": categories,
                    "unknown_value": -1.0,
                }
            else:
                train_values = pd.to_numeric(train_series, errors="coerce").to_numpy(dtype=np.float64)
                test_values = pd.to_numeric(test_series, errors="coerce").to_numpy(dtype=np.float64)
                encodings[str(column)] = {"type": "numeric"}
            train_columns.append(train_values)
            test_columns.append(test_values)
        return (
            np.column_stack(train_columns),
            np.column_stack(test_columns),
            categorical_indices,
            encodings,
        )

    def prepare_split(self, data_id: int, seed: int) -> PreparedSplit:
        dataset_dir = self.processed_dir / str(data_id)
        array_path = dataset_dir / f"split-{seed}.npz"
        metadata_path = dataset_dir / f"split-{seed}.json"
        if array_path.exists() and metadata_path.exists():
            arrays = np.load(array_path, allow_pickle=False)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("config_sha256") != self.config["_config_sha256"]:
                raise ValueError(f"Cached split {data_id}/{seed} uses a different configuration")
            return PreparedSplit(
                X_train=arrays["X_train"],
                X_test=arrays["X_test"],
                y_train=arrays["y_train"],
                y_test=arrays["y_test"],
                categorical_indices=arrays["categorical_indices"].astype(int).tolist(),
                metadata=metadata,
            )

        X, y, dataset_metadata = self.fetch_and_validate(data_id)
        row_indices = np.arange(len(X))
        train_indices, test_indices = train_test_split(
            row_indices,
            test_size=float(self.config["split"]["test_size"]),
            random_state=int(seed),
            stratify=y.astype(str).to_numpy(),
        )
        X_train_frame = X.iloc[train_indices].reset_index(drop=True)
        X_test_frame = X.iloc[test_indices].reset_index(drop=True)
        y_train_raw = y.iloc[train_indices].astype(str).reset_index(drop=True)
        y_test_raw = y.iloc[test_indices].astype(str).reset_index(drop=True)
        classes = sorted(y_train_raw.unique().tolist())
        class_map = {value: index for index, value in enumerate(classes)}
        y_train = y_train_raw.map(class_map).to_numpy(dtype=np.int64)
        y_test = y_test_raw.map(class_map).to_numpy(dtype=np.int64)
        X_train, X_test, categorical_indices, encodings = self._encode_predictors(
            X_train_frame,
            X_test_frame,
        )

        metadata = {
            **dataset_metadata,
            "seed": int(seed),
            "test_size": float(self.config["split"]["test_size"]),
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "train_indices_sha256": _sha256_array(train_indices),
            "test_indices_sha256": _sha256_array(test_indices),
            "X_train_sha256": _sha256_array(X_train),
            "X_test_sha256": _sha256_array(X_test),
            "classes": classes,
            "class_map": class_map,
            "categorical_indices": categorical_indices,
            "categorical_count": len(categorical_indices),
            "encodings": encodings,
            "config_sha256": self.config["_config_sha256"],
        }
        dataset_dir.mkdir(parents=True, exist_ok=True)
        temporary = array_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                categorical_indices=np.asarray(categorical_indices, dtype=np.int64),
            )
        os.replace(temporary, array_path)
        _atomic_json(metadata_path, metadata)
        return PreparedSplit(X_train, X_test, y_train, y_test, categorical_indices, metadata)

    def prepare_all(self) -> list[dict[str, Any]]:
        metadata: list[dict[str, Any]] = []
        split_files: list[dict[str, Any]] = []
        for dataset in self.config["datasets"]:
            for seed in self.config["seeds"]:
                data_id = int(dataset["data_id"])
                split_seed = int(seed)
                metadata.append(self.prepare_split(data_id, split_seed).metadata)
                dataset_dir = self.processed_dir / str(data_id)
                array_path = dataset_dir / f"split-{split_seed}.npz"
                metadata_path = dataset_dir / f"split-{split_seed}.json"
                split_files.append(
                    {
                        "data_id": data_id,
                        "seed": split_seed,
                        "array_path": array_path.relative_to(Path(self.config["_project_root"])).as_posix(),
                        "array_sha256": _sha256_file(array_path),
                        "metadata_path": metadata_path.relative_to(
                            Path(self.config["_project_root"])
                        ).as_posix(),
                        "metadata_sha256": _sha256_file(metadata_path),
                    }
                )
        manifest_path = self.processed_dir / "manifest.json"
        manifest = {
            "config_sha256": self.config["_config_sha256"],
            "split_count": len(metadata),
            "splits": metadata,
            "manifest_sha256": hashlib.sha256(canonical_json(metadata).encode("ascii")).hexdigest(),
            "split_files": split_files,
            "split_file_manifest_sha256": hashlib.sha256(
                canonical_json(split_files).encode("ascii")
            ).hexdigest(),
        }
        _atomic_json(manifest_path, manifest)
        return metadata
