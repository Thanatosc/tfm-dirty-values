from __future__ import annotations

import numpy as np
import pytest

from tfm_dirty_values.injection import inject_corruption
from tfm_dirty_values_ablation.cleaning import (
    CLEANER_ID,
    CLEANER_PARAMETERS,
    apply_cleaner,
    clean_partitions,
    fit_cleaner,
)


def _numeric_frame(rows: int = 200, columns: int = 4, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, columns)) * 3.0 + 7.0


def test_cleaner_identity_and_parameters_are_frozen() -> None:
    assert CLEANER_ID == "tukey_winsorize_plus_unseen_level_to_mode_v1"
    assert CLEANER_PARAMETERS == {
        "numeric_method": "tukey_fence_winsorization",
        "iqr_multiplier": 1.5,
        "categorical_method": "unseen_level_to_context_mode",
    }


def test_fitted_bounds_are_tukey_fences() -> None:
    context = _numeric_frame()
    fitted = fit_cleaner(context, categorical_indices=[])
    for column in range(context.shape[1]):
        q1, median, q3 = np.percentile(context[:, column], [25, 50, 75])
        iqr = q3 - q1
        assert fitted["lower"][column] == pytest.approx(median - 1.5 * iqr)
        assert fitted["upper"][column] == pytest.approx(median + 1.5 * iqr)


def test_winsorization_clips_outliers_into_the_fence() -> None:
    context = _numeric_frame()
    fitted = fit_cleaner(context, categorical_indices=[])
    query = context.copy()
    query[0, 0] = 1e6
    query[1, 0] = -1e6

    cleaned, metadata = apply_cleaner(query, fitted)

    assert cleaned[0, 0] == pytest.approx(fitted["upper"][0])
    assert cleaned[1, 0] == pytest.approx(fitted["lower"][0])
    assert metadata["clipped_high_cells"] >= 1
    assert metadata["clipped_low_cells"] >= 1


def test_cleaner_never_reads_the_query_partition() -> None:
    """Fitted parameters must be identical no matter what the query contains."""
    context = _numeric_frame()
    baseline = fit_cleaner(context, categorical_indices=[])

    hostile = _numeric_frame(seed=99) * 500.0
    _, _, metadata = clean_partitions(context, hostile, categorical_indices=[])

    refit = fit_cleaner(context, categorical_indices=[])
    assert np.array_equal(baseline["lower"], refit["lower"])
    assert np.array_equal(baseline["upper"], refit["upper"])
    assert metadata["fit_scope"] == "corrupted_context_only"


def test_cleaner_does_not_mutate_its_inputs() -> None:
    context = _numeric_frame()
    query = _numeric_frame(seed=5)
    context_before = context.copy()
    query_before = query.copy()

    clean_partitions(context, query, categorical_indices=[])

    assert np.array_equal(context, context_before)
    assert np.array_equal(query, query_before)


def test_degenerate_and_tiny_columns_are_left_untouched() -> None:
    context = np.column_stack(
        [
            np.full(50, 4.0),          # zero spread
            np.arange(50, dtype=float),  # healthy spread
        ]
    )
    fitted = fit_cleaner(context, categorical_indices=[])

    assert fitted["lower"][0] == -np.inf
    assert fitted["upper"][0] == np.inf
    assert np.isfinite(fitted["lower"][1])
    assert fitted["fitted_numeric_columns"] == 1

    tiny = np.array([[1.0], [2.0], [3.0]])
    tiny_fit = fit_cleaner(tiny, categorical_indices=[])
    assert tiny_fit["fitted_numeric_columns"] == 0

    cleaned, _ = apply_cleaner(context, fitted)
    assert np.array_equal(cleaned[:, 0], context[:, 0])


def test_unseen_categorical_level_maps_to_context_mode() -> None:
    context = np.column_stack(
        [
            np.asarray([0.0, 0.0, 0.0, 1.0, 2.0, 1.0]),  # mode is 0
            np.arange(6, dtype=float),
        ]
    )
    fitted = fit_cleaner(context, categorical_indices=[0])
    assert fitted["modes"][0] == pytest.approx(0.0)

    query = np.column_stack([np.asarray([99.0, 1.0, 2.0]), np.asarray([1.0, 2.0, 3.0])])
    cleaned, metadata = apply_cleaner(query, fitted)

    assert cleaned[0, 0] == pytest.approx(0.0)
    assert cleaned[1, 0] == pytest.approx(1.0)
    assert cleaned[2, 0] == pytest.approx(2.0)
    assert metadata["remapped_categorical_cells"] == 1


def test_known_categorical_levels_are_preserved() -> None:
    context = np.column_stack([np.asarray([0.0, 1.0, 1.0, 2.0]), np.arange(4, dtype=float)])
    fitted = fit_cleaner(context, categorical_indices=[0])
    query = np.column_stack([np.asarray([0.0, 1.0, 2.0]), np.arange(3, dtype=float)])

    cleaned, metadata = apply_cleaner(query, fitted)

    assert np.array_equal(cleaned[:, 0], query[:, 0])
    assert metadata["remapped_categorical_cells"] == 0


def test_nan_cells_survive_cleaning() -> None:
    context = _numeric_frame()
    fitted = fit_cleaner(context, categorical_indices=[])
    query = context.copy()
    query[3, 1] = np.nan

    cleaned, _ = apply_cleaner(query, fitted)

    assert np.isnan(cleaned[3, 1])


def test_cleaning_is_deterministic() -> None:
    context = _numeric_frame()
    query = _numeric_frame(seed=7)
    first_c, first_q, first_meta = clean_partitions(context, query, categorical_indices=[])
    second_c, second_q, second_meta = clean_partitions(context, query, categorical_indices=[])

    assert np.array_equal(first_c, second_c)
    assert np.array_equal(first_q, second_q)
    assert first_meta["context"]["output_sha256"] == second_meta["context"]["output_sha256"]
    assert first_meta["query"]["output_sha256"] == second_meta["query"]["output_sha256"]


def test_feature_count_mismatch_is_rejected() -> None:
    fitted = fit_cleaner(_numeric_frame(), categorical_indices=[])
    with pytest.raises(ValueError, match="Feature count"):
        apply_cleaner(np.zeros((10, 2)), fitted)


def test_one_dimensional_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        fit_cleaner(np.zeros(10), categorical_indices=[])


def _corrupt_then_clean(mechanism: str, rate: float, params: dict, seed: int = 3):
    rng = np.random.default_rng(seed)
    context = rng.normal(size=(300, 5)) * 2.0 + 1.0
    query = rng.normal(size=(120, 5)) * 2.0 + 1.0
    dirty_context, dirty_query, _ = inject_corruption(
        context,
        query,
        dataset_id=1,
        split_seed=1907,
        corruption=mechanism,
        rate=rate,
        categorical_indices=[],
        params=params,
    )
    cleaned_context, cleaned_query, metadata = clean_partitions(
        dirty_context, dirty_query, categorical_indices=[]
    )
    return context, query, dirty_context, dirty_query, cleaned_context, cleaned_query, metadata


def test_cleaning_reduces_outlier_damage_below_the_breakdown_point() -> None:
    """At rate 0.15 the Tukey fence still identifies injected extremes."""
    context, _, dirty_context, _, cleaned_context, _, metadata = _corrupt_then_clean(
        "outlier", 0.15, {"k": 10.0}
    )

    assert np.abs(cleaned_context).max() < np.abs(dirty_context).max()
    assert metadata["context"]["changed_cells"] > 0

    before = float(np.abs(dirty_context - context).mean())
    after = float(np.abs(cleaned_context - context).mean())
    assert after < before


def test_tukey_fence_breaks_down_when_half_the_cells_are_outliers() -> None:
    """Registered in preregistration Amendment 1.

    At rate 0.50 the injected outliers are half the observed cells, so they
    capture the median and inflate the IQR that define the fence. The fence
    widens to contain the contamination and clips nothing. This is the
    estimator's breakdown point, not evidence that the damage is unrepairable.
    """
    _, _, dirty_context, _, cleaned_context, _, metadata = _corrupt_then_clean(
        "outlier", 0.50, {"k": 10.0}
    )

    assert metadata["context"]["changed_cells"] == 0
    assert np.array_equal(cleaned_context, dirty_context)


def test_unit_error_is_recoverable_at_both_registered_rates() -> None:
    """Factor-1000 errors sit far outside the fence even at rate 0.50."""
    for rate in (0.15, 0.50):
        context, _, dirty_context, _, cleaned_context, _, metadata = _corrupt_then_clean(
            "unit_error", rate, {"factors": [1000.0, 0.001]}
        )
        assert metadata["context"]["changed_cells"] > 0
        before = float(np.abs(dirty_context - context).mean())
        after = float(np.abs(cleaned_context - context).mean())
        assert after < before, f"unit_error should recover at rate {rate}"


def test_quantization_damage_is_largely_untouched_by_winsorization() -> None:
    """Preregistered expectation: binning is not recoverable by clipping."""
    _, query, _, dirty_query, _, cleaned_query, metadata = _corrupt_then_clean(
        "quantization", 0.50, {"method": "equal_width_bins", "bins": 8}, seed=4
    )

    # Quantized values stay inside the data range, so almost nothing is clipped.
    fraction_changed = metadata["query"]["changed_cells"] / metadata["query"]["observed_cells"]
    assert fraction_changed < 0.10
    assert not np.allclose(cleaned_query, query)


def test_winsorizing_mild_in_distribution_noise_can_increase_distance() -> None:
    """Registered in preregistration Amendment 1.

    Measurement noise at sigma_ratio 0.50 stays largely inside the fence, so the
    cells the cleaner clips are genuine distribution tails rather than corrupted
    cells. Cleaning can therefore move the matrix further from the clean
    original. This is a real cost of applying a standard cleaner to data that
    does not need it, and it is why arm D of the ablation exists.
    """
    context, _, dirty_context, _, cleaned_context, _, _ = _corrupt_then_clean(
        "measurement_noise", 0.05, {"sigma_ratio": 0.50}
    )

    before = float(np.abs(dirty_context - context).mean())
    after = float(np.abs(cleaned_context - context).mean())
    assert after > before
