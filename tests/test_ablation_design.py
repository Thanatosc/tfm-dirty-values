from __future__ import annotations

import numpy as np
import pytest

from tfm_dirty_values.config import load_config
from tfm_dirty_values.injection import inject_corruption
from tfm_dirty_values_ablation.design import (
    PHASE3_STAGE,
    ablation_task_manifest,
    ablation_tasks,
    normalize_parameters,
    verify_expected_counts,
)


@pytest.fixture(scope="module")
def configs() -> tuple[dict, dict]:
    return (
        load_config("configs/cleaning_ablation.yaml"),
        load_config("configs/full_study.yaml"),
    )


def test_parameter_key_order_changes_the_corruption_seed() -> None:
    """Regression guard for the paired-identity bug.

    ``injection._stable_seed`` string-formats the parameters dict, so insertion
    order changes the RNG seed and therefore the corrupted array. Phase 2 ran
    with key-sorted parameters. If this property ever stops holding, the
    normalization in ``normalize_parameters`` can be removed; until then it is
    load-bearing and this test documents why.
    """
    rng = np.random.default_rng(0)
    context = rng.normal(size=(120, 4))
    query = rng.normal(size=(60, 4))

    yaml_order = {"method": "equal_width_bins", "bins": 8}
    sorted_order = {"bins": 8, "method": "equal_width_bins"}
    assert list(yaml_order) != list(sorted_order)

    def corrupt(params: dict) -> np.ndarray:
        train, _, _ = inject_corruption(
            context,
            query,
            dataset_id=1049,
            split_seed=4049,
            corruption="quantization",
            rate=0.50,
            categorical_indices=[],
            params=params,
        )
        return train

    assert not np.array_equal(corrupt(yaml_order), corrupt(sorted_order))
    assert np.array_equal(
        corrupt(normalize_parameters(yaml_order)),
        corrupt(sorted_order),
    )


def test_normalize_parameters_sorts_keys_and_preserves_values() -> None:
    normalized = normalize_parameters({"strategy": "mixed", "novel_level_fraction": 0.5})
    assert list(normalized) == ["novel_level_fraction", "strategy"]
    assert normalized == {"novel_level_fraction": 0.5, "strategy": "mixed"}

    single = normalize_parameters({"k": 10.0})
    assert list(single) == ["k"]


def test_all_generated_task_parameters_are_key_sorted(configs) -> None:
    config, phase2 = configs
    for task in ablation_tasks(config, phase2):
        keys = list(task.parameters)
        assert keys == sorted(keys), f"unsorted parameters in {task.task_id}"


def test_manifest_matches_preregistered_counts(configs) -> None:
    config, phase2 = configs
    manifest = ablation_task_manifest(config, phase2)

    assert manifest["stage"] == PHASE3_STAGE
    assert manifest["task_count"] == 2425
    assert manifest["arm_counts"] == {"c": 2150, "d": 275}
    assert verify_expected_counts(manifest, config) == []
    assert len(manifest["task_manifest_sha256"]) == 64


def test_manifest_is_deterministic(configs) -> None:
    config, phase2 = configs
    first = ablation_task_manifest(config, phase2)
    second = ablation_task_manifest(config, phase2)
    assert first["task_manifest_sha256"] == second["task_manifest_sha256"]


def test_arm_d_has_no_rate_dimension(configs) -> None:
    config, phase2 = configs
    arm_d = [task for task in ablation_tasks(config, phase2) if task.arm == "d"]

    assert len(arm_d) == 275
    assert {task.rate for task in arm_d} == {0.0}
    assert {task.corruption for task in arm_d} == {"clean"}
    assert all(task.cleaned for task in arm_d)


def test_arm_c_covers_both_registered_rates_and_respects_panels(configs) -> None:
    config, phase2 = configs
    arm_c = [task for task in ablation_tasks(config, phase2) if task.arm == "c"]
    panels = phase2["corruptions"]["panels"]

    assert {task.rate for task in arm_c} == {0.15, 0.50}
    assert all(task.cleaned for task in arm_c)

    for task in arm_c:
        allowed = {int(value) for value in panels[task.corruption]}
        assert task.dataset_id in allowed


def test_every_task_is_marked_cleaned(configs) -> None:
    """Phase 3 executes only the cleaned arms; B and A are reused from Phase 2."""
    config, phase2 = configs
    assert all(task.cleaned for task in ablation_tasks(config, phase2))


def test_task_identities_are_unique(configs) -> None:
    config, phase2 = configs
    tasks = ablation_tasks(config, phase2)
    assert len({task.task_id for task in tasks}) == len(tasks)


def test_phase3_task_ids_do_not_collide_with_phase2(configs) -> None:
    from tfm_dirty_values.design import full_study_tasks

    config, phase2 = configs
    phase3_ids = {task.task_id for task in ablation_tasks(config, phase2)}
    phase2_ids = {task.task_id for task in full_study_tasks(phase2)}
    assert not (phase3_ids & phase2_ids)


def test_ablation_writes_to_its_own_result_tree(configs) -> None:
    config, _ = configs
    for key in ("result_dir", "database", "outputs", "authorization"):
        assert "full_study" not in config["paths"][key]
    assert config["paths"]["phase2_database"].startswith("results/full_study")
