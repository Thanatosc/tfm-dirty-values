"""The single fixed cleaner used by the Phase 3 ablation.

Frozen by ``research/phase3-cleaning-ablation-preregistration.md``:

- Numeric columns are winsorized at Tukey fences, ``median +/- 1.5 * IQR``.
- Categorical query levels absent from the fitted context level set are mapped
  to the fitted context mode.

The cleaner is fitted only on the context partition it is given. It never sees
clean data when cleaning a corrupted task, never sees the query partition, and
never sees labels. The same pipeline runs for every corruption mechanism; there
is no mechanism-specific variant and no parameter search.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

CLEANER_ID = "tukey_winsorize_plus_unseen_level_to_mode_v1"

CLEANER_PARAMETERS: dict[str, Any] = {
    "numeric_method": "tukey_fence_winsorization",
    "iqr_multiplier": 1.5,
    "categorical_method": "unseen_level_to_context_mode",
}

_EPSILON = 1e-12


def _column_kinds(n_features: int, categorical_indices: list[int]) -> tuple[list[int], list[int]]:
    categorical = sorted({int(value) for value in categorical_indices})
    for column in categorical:
        if not 0 <= column < n_features:
            raise ValueError(f"Categorical index {column} is outside the feature range")
    numeric = [column for column in range(n_features) if column not in set(categorical)]
    return numeric, categorical


def _mode(values: np.ndarray) -> float | None:
    if not values.size:
        return None
    levels, counts = np.unique(values, return_counts=True)
    best = int(np.argmax(counts))
    # np.unique returns sorted levels, so ties resolve to the smallest level.
    return float(levels[best])


def fit_cleaner(
    X_context: np.ndarray,
    *,
    categorical_indices: list[int],
) -> dict[str, Any]:
    """Estimate cleaning parameters from a context partition only.

    ``X_context`` is the partition the cleaner is allowed to see. For a
    corrupted task this is the corrupted context, never the clean original.
    """
    context = np.asarray(X_context, dtype=np.float64)
    if context.ndim != 2:
        raise ValueError("Cleaner input must be two-dimensional")

    numeric, categorical = _column_kinds(context.shape[1], categorical_indices)
    multiplier = float(CLEANER_PARAMETERS["iqr_multiplier"])

    lower = np.full(context.shape[1], -np.inf, dtype=np.float64)
    upper = np.full(context.shape[1], np.inf, dtype=np.float64)
    fitted_numeric = 0
    for column in numeric:
        values = context[np.isfinite(context[:, column]), column]
        if values.size < 4:
            # Too few observations for a stable quartile estimate; leave the
            # column untouched rather than inventing bounds.
            continue
        q1, median, q3 = (float(x) for x in np.percentile(values, [25, 50, 75]))
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= _EPSILON:
            # Degenerate spread: winsorizing would collapse the column.
            continue
        lower[column] = median - multiplier * iqr
        upper[column] = median + multiplier * iqr
        fitted_numeric += 1

    levels: dict[int, np.ndarray] = {}
    modes: dict[int, float] = {}
    for column in categorical:
        values = context[np.isfinite(context[:, column]), column]
        levels[column] = np.unique(values)
        mode = _mode(values)
        if mode is not None:
            modes[column] = mode

    return {
        "cleaner_id": CLEANER_ID,
        "parameters": dict(CLEANER_PARAMETERS),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "lower": lower,
        "upper": upper,
        "levels": levels,
        "modes": modes,
        "fitted_numeric_columns": fitted_numeric,
        "fit_rows": int(context.shape[0]),
    }


def apply_cleaner(X: np.ndarray, fitted: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply fitted cleaning parameters to a partition.

    Returns the cleaned matrix and per-partition metadata describing how many
    cells were actually altered, which is recorded for auditability.
    """
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Cleaner input must be two-dimensional")
    if matrix.shape[1] != len(fitted["lower"]):
        raise ValueError("Feature count does not match the fitted cleaner")

    output = matrix.copy()
    lower = np.asarray(fitted["lower"], dtype=np.float64)
    upper = np.asarray(fitted["upper"], dtype=np.float64)

    observed = np.isfinite(output)
    numeric_columns = list(fitted["numeric_columns"])
    clipped_low = 0
    clipped_high = 0
    if numeric_columns:
        index = np.asarray(numeric_columns, dtype=int)
        block = output[:, index]
        seen = observed[:, index]
        low_bound = lower[index][np.newaxis, :]
        high_bound = upper[index][np.newaxis, :]
        below = seen & (block < low_bound)
        above = seen & (block > high_bound)
        clipped_low = int(below.sum())
        clipped_high = int(above.sum())
        block = np.where(below, np.broadcast_to(low_bound, block.shape), block)
        block = np.where(above, np.broadcast_to(high_bound, block.shape), block)
        output[:, index] = block

    remapped = 0
    for column in fitted["categorical_columns"]:
        mode = fitted["modes"].get(column)
        if mode is None:
            continue
        known = fitted["levels"][column]
        values = output[:, column]
        seen = np.isfinite(values)
        unknown = seen & ~np.isin(values, known)
        remapped += int(unknown.sum())
        output[unknown, column] = mode

    changed = int((~np.isclose(output, matrix, equal_nan=True)).sum())
    metadata = {
        "observed_cells": int(observed.sum()),
        "clipped_low_cells": clipped_low,
        "clipped_high_cells": clipped_high,
        "remapped_categorical_cells": remapped,
        "changed_cells": changed,
        "output_sha256": hashlib.sha256(
            np.ascontiguousarray(output).view(np.uint8)
        ).hexdigest(),
    }
    return output, metadata


def clean_partitions(
    X_context: np.ndarray,
    X_query: np.ndarray,
    *,
    categorical_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit on the context partition, then clean both partitions.

    This is the entry point used by the Phase 3 worker. Fitting on the context
    and applying to both partitions mirrors the dual-partition contamination
    design of Phase 2 and the deployment case where a practitioner estimates
    cleaning parameters from held dirty training data.
    """
    fitted = fit_cleaner(X_context, categorical_indices=categorical_indices)
    context_clean, context_meta = apply_cleaner(X_context, fitted)
    query_clean, query_meta = apply_cleaner(X_query, fitted)
    return context_clean, query_clean, {
        "cleaner_id": fitted["cleaner_id"],
        "parameters": fitted["parameters"],
        "fit_scope": "corrupted_context_only",
        "fitted_numeric_columns": fitted["fitted_numeric_columns"],
        "categorical_columns": list(fitted["categorical_columns"]),
        "context": context_meta,
        "query": query_meta,
    }
