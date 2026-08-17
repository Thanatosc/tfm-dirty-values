from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


SUPPORTED_CORRUPTIONS = (
    "clean",
    "measurement_noise",
    "outlier",
    "quantization",
    "unit_error",
    "categorical_error",
)


def _stable_seed(*parts: Any) -> int:
    joined = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(joined.encode("utf-8")).digest()[:8], "little")


def _validate_inputs(X_train: np.ndarray, X_test: np.ndarray, rate: float) -> None:
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Corruption inputs must be two-dimensional")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Context and query must have the same feature count")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("Affected-cell rate must be within [0, 1]")


def _eligible_mask(
    X: np.ndarray,
    *,
    categorical_indices: list[int],
    categorical: bool,
) -> np.ndarray:
    selected_columns = np.zeros(X.shape[1], dtype=bool)
    category_set = set(int(value) for value in categorical_indices)
    for column in range(X.shape[1]):
        selected_columns[column] = (column in category_set) == categorical
    return np.isfinite(X) & selected_columns[np.newaxis, :]


def _exact_mask(
    eligible: np.ndarray,
    *,
    rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.flatnonzero(eligible.ravel())
    target = int(round(rate * len(candidates)))
    if rate > 0 and not len(candidates):
        raise ValueError("No observed eligible cells are available for this corruption")
    selected = rng.choice(candidates, size=target, replace=False) if target else np.asarray([], dtype=int)
    mask = np.zeros(eligible.size, dtype=bool)
    mask[selected] = True
    return mask.reshape(eligible.shape)


def _numeric_statistics(X_train: np.ndarray, categorical_indices: list[int]) -> dict[str, np.ndarray]:
    means = np.zeros(X_train.shape[1], dtype=np.float64)
    scales = np.ones(X_train.shape[1], dtype=np.float64)
    minimum = np.zeros(X_train.shape[1], dtype=np.float64)
    maximum = np.zeros(X_train.shape[1], dtype=np.float64)
    categorical = set(int(value) for value in categorical_indices)
    for column in range(X_train.shape[1]):
        if column in categorical:
            continue
        values = X_train[np.isfinite(X_train[:, column]), column]
        if not len(values):
            continue
        means[column] = float(np.mean(values))
        scale = float(np.std(values))
        scales[column] = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
        minimum[column] = float(np.min(values))
        maximum[column] = float(np.max(values))
    return {"means": means, "scales": scales, "minimum": minimum, "maximum": maximum}


def _category_statistics(
    X_train: np.ndarray,
    categorical_indices: list[int],
) -> dict[int, np.ndarray]:
    return {
        int(column): np.unique(X_train[np.isfinite(X_train[:, column]), column])
        for column in categorical_indices
    }


def _quantize(values: np.ndarray, columns: np.ndarray, stats: dict[str, np.ndarray], params: dict[str, Any]) -> np.ndarray:
    method = str(params.get("method", "equal_width_bins"))
    output = values.copy()
    if method == "equal_width_bins":
        bins = int(params["bins"])
        if bins < 2:
            raise ValueError("Quantization requires at least two bins")
        lower = stats["minimum"][columns]
        upper = stats["maximum"][columns]
        width = (upper - lower) / bins
        nonconstant = width > 1e-12
        indices = np.zeros(len(values), dtype=np.int64)
        indices[nonconstant] = np.floor(
            (values[nonconstant] - lower[nonconstant]) / width[nonconstant]
        ).astype(np.int64)
        indices = np.clip(indices, 0, bins - 1)
        output[nonconstant] = lower[nonconstant] + (indices[nonconstant] + 0.5) * width[nonconstant]
        return output
    if method == "significant_digits":
        digits = int(params["digits"])
        if digits < 1:
            raise ValueError("Significant-digit quantization requires digits >= 1")
        for index, value in enumerate(values):
            if value == 0.0:
                continue
            decimals = digits - 1 - int(np.floor(np.log10(abs(value))))
            output[index] = np.round(value, decimals=decimals)
        return output
    raise ValueError(f"Unknown quantization method: {method}")


def _corrupt_matrix(
    X: np.ndarray,
    *,
    corruption: str,
    rate: float,
    categorical_indices: list[int],
    params: dict[str, Any],
    numeric_stats: dict[str, np.ndarray],
    category_stats: dict[int, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    categorical = corruption == "categorical_error"
    eligible = _eligible_mask(
        X,
        categorical_indices=categorical_indices,
        categorical=categorical,
    )
    rng = np.random.default_rng(seed)
    mask = _exact_mask(eligible, rate=rate, rng=rng)
    rows, columns = np.nonzero(mask)
    output = np.asarray(X, dtype=np.float64).copy()
    original = output[rows, columns].copy()

    if corruption == "measurement_noise":
        sigma_ratio = float(params["sigma_ratio"])
        if sigma_ratio <= 0:
            raise ValueError("sigma_ratio must be positive")
        output[rows, columns] += rng.normal(size=len(rows)) * sigma_ratio * numeric_stats["scales"][columns]
    elif corruption == "outlier":
        k = float(params["k"])
        if k <= 0:
            raise ValueError("Outlier k must be positive")
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(rows))
        output[rows, columns] = numeric_stats["means"][columns] + signs * k * numeric_stats["scales"][columns]
    elif corruption == "quantization":
        output[rows, columns] = _quantize(original, columns, numeric_stats, params)
    elif corruption == "unit_error":
        factors = np.asarray(params.get("factors", [1000.0, 0.001]), dtype=np.float64)
        if not len(factors) or not np.isfinite(factors).all() or np.any(factors <= 0):
            raise ValueError("Unit-error factors must be finite positive values")
        output[rows, columns] *= rng.choice(factors, size=len(rows))
    elif corruption == "categorical_error":
        strategy = str(params.get("strategy", "mixed"))
        novel_fraction = float(params.get("novel_level_fraction", 0.5))
        if strategy not in {"swap", "novel", "mixed"}:
            raise ValueError(f"Unknown categorical-error strategy: {strategy}")
        if not 0.0 <= novel_fraction <= 1.0:
            raise ValueError("novel_level_fraction must be within [0, 1]")
        for index, (row, column) in enumerate(zip(rows, columns, strict=True)):
            categories = category_stats[int(column)]
            current = original[index]
            use_novel = strategy == "novel" or (
                strategy == "mixed" and rng.random() < novel_fraction
            )
            alternatives = categories[categories != current]
            if use_novel or not len(alternatives):
                base = float(np.max(categories) + 1.0) if len(categories) else 0.0
                output[row, column] = base
            else:
                output[row, column] = float(rng.choice(alternatives))
    else:
        raise ValueError(f"Unknown corruption: {corruption}")

    changed = ~np.isclose(output[rows, columns], original, equal_nan=True)
    metadata = {
        "seed": int(seed),
        "requested_rate": float(rate),
        "eligible_observed_cells": int(eligible.sum()),
        "selected_cells": int(mask.sum()),
        "realized_rate": float(mask.sum() / eligible.sum()) if eligible.any() else 0.0,
        "changed_cells": int(changed.sum()),
        "mask_sha256": hashlib.sha256(np.ascontiguousarray(mask).view(np.uint8)).hexdigest(),
    }
    return output, metadata


def inject_corruption(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    dataset_id: int,
    split_seed: int,
    corruption: str,
    rate: float,
    categorical_indices: list[int],
    params: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = np.asarray(X_train, dtype=np.float64)
    test = np.asarray(X_test, dtype=np.float64)
    _validate_inputs(train, test, rate)
    if corruption not in SUPPORTED_CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {corruption}")
    if corruption == "clean":
        if rate != 0.0:
            raise ValueError("Clean baseline must use rate=0")
        return train.copy(), test.copy(), {
            "corruption": "clean",
            "rate": 0.0,
            "fit_scope": "none",
            "context": None,
            "query": None,
        }

    parameters = dict(params or {})
    numeric_stats = _numeric_statistics(train, categorical_indices)
    category_stats = _category_statistics(train, categorical_indices)
    context_seed = _stable_seed("context", dataset_id, split_seed, corruption, rate, parameters)
    query_seed = _stable_seed("query", dataset_id, split_seed, corruption, rate, parameters)
    train_output, context_metadata = _corrupt_matrix(
        train,
        corruption=corruption,
        rate=rate,
        categorical_indices=categorical_indices,
        params=parameters,
        numeric_stats=numeric_stats,
        category_stats=category_stats,
        seed=context_seed,
    )
    test_output, query_metadata = _corrupt_matrix(
        test,
        corruption=corruption,
        rate=rate,
        categorical_indices=categorical_indices,
        params=parameters,
        numeric_stats=numeric_stats,
        category_stats=category_stats,
        seed=query_seed,
    )
    return train_output, test_output, {
        "corruption": corruption,
        "rate": float(rate),
        "parameters": parameters,
        "fit_scope": "clean_context_only",
        "categorical_indices": [int(value) for value in categorical_indices],
        "context": context_metadata,
        "query": query_metadata,
    }
