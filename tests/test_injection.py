from __future__ import annotations

import numpy as np
import pytest

from tfm_dirty_values.injection import inject_corruption


@pytest.fixture
def arrays() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    train = rng.normal(size=(120, 8))
    test = rng.normal(size=(50, 8))
    train[:, 0] = rng.integers(0, 3, size=len(train))
    test[:, 0] = rng.integers(0, 3, size=len(test))
    train[:, 1] = rng.integers(0, 4, size=len(train))
    test[:, 1] = rng.integers(0, 4, size=len(test))
    train[0, 2] = np.nan
    test[0, 2] = np.nan
    return train, test


@pytest.mark.parametrize(
    ("corruption", "params", "categorical"),
    [
        ("measurement_noise", {"sigma_ratio": 0.5}, False),
        ("outlier", {"k": 10.0}, False),
        ("quantization", {"method": "equal_width_bins", "bins": 8}, False),
        ("unit_error", {"factors": [1000.0, 0.001]}, False),
        (
            "categorical_error",
            {"strategy": "mixed", "novel_level_fraction": 0.5},
            True,
        ),
    ],
)
def test_exact_eligible_rate_and_type_boundary(
    arrays: tuple[np.ndarray, np.ndarray],
    corruption: str,
    params: dict,
    categorical: bool,
) -> None:
    train, test = arrays
    train_original = train.copy()
    test_original = test.copy()
    train_out, test_out, metadata = inject_corruption(
        train,
        test,
        dataset_id=999,
        split_seed=1907,
        corruption=corruption,
        rate=0.25,
        categorical_indices=[0, 1],
        params=params,
    )
    eligible_train = np.isfinite(train[:, :2]).sum() if categorical else np.isfinite(train[:, 2:]).sum()
    eligible_test = np.isfinite(test[:, :2]).sum() if categorical else np.isfinite(test[:, 2:]).sum()
    assert metadata["context"]["selected_cells"] == round(0.25 * eligible_train)
    assert metadata["query"]["selected_cells"] == round(0.25 * eligible_test)
    assert np.array_equal(train, train_original, equal_nan=True)
    assert np.array_equal(test, test_original, equal_nan=True)
    if categorical:
        assert np.array_equal(train_out[:, 2:], train[:, 2:], equal_nan=True)
        assert np.array_equal(test_out[:, 2:], test[:, 2:], equal_nan=True)
    else:
        assert np.array_equal(train_out[:, :2], train[:, :2], equal_nan=True)
        assert np.array_equal(test_out[:, :2], test[:, :2], equal_nan=True)


def test_injection_is_deterministic_and_streams_are_distinct(
    arrays: tuple[np.ndarray, np.ndarray],
) -> None:
    train, test = arrays
    kwargs = dict(
        dataset_id=999,
        split_seed=2027,
        corruption="measurement_noise",
        rate=0.5,
        categorical_indices=[0, 1],
        params={"sigma_ratio": 0.5},
    )
    first = inject_corruption(train, test, **kwargs)
    second = inject_corruption(train, test, **kwargs)
    assert np.array_equal(first[0], second[0], equal_nan=True)
    assert np.array_equal(first[1], second[1], equal_nan=True)
    assert first[2] == second[2]
    assert first[2]["context"]["seed"] != first[2]["query"]["seed"]


def test_outliers_use_context_statistics_for_query() -> None:
    train = np.column_stack([np.zeros(20), np.arange(20, dtype=float)])
    test = np.column_stack([np.zeros(10), np.arange(1000, 1010, dtype=float)])
    _, test_out, metadata = inject_corruption(
        train,
        test,
        dataset_id=1,
        split_seed=1,
        corruption="outlier",
        rate=1.0,
        categorical_indices=[0],
        params={"k": 3.0},
    )
    expected = np.mean(train[:, 1]) + np.asarray([-1.0, 1.0]) * 3.0 * np.std(train[:, 1])
    assert set(np.round(test_out[:, 1], 12)).issubset(set(np.round(expected, 12)))
    assert metadata["fit_scope"] == "clean_context_only"


def test_clean_baseline_is_copy_only(arrays: tuple[np.ndarray, np.ndarray]) -> None:
    train, test = arrays
    train_out, test_out, metadata = inject_corruption(
        train,
        test,
        dataset_id=1,
        split_seed=1,
        corruption="clean",
        rate=0.0,
        categorical_indices=[0, 1],
    )
    assert np.array_equal(train_out, train, equal_nan=True)
    assert np.array_equal(test_out, test, equal_nan=True)
    assert train_out is not train
    assert test_out is not test
    assert metadata["fit_scope"] == "none"


def test_categorical_corruption_rejects_numeric_only_data() -> None:
    X = np.arange(20, dtype=float).reshape(10, 2)
    with pytest.raises(ValueError, match="No observed eligible cells"):
        inject_corruption(
            X,
            X,
            dataset_id=1,
            split_seed=1,
            corruption="categorical_error",
            rate=0.5,
            categorical_indices=[],
            params={"strategy": "mixed", "novel_level_fraction": 0.5},
        )

