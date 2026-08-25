"""Phase 3 task panel generation.

Builds only the two arms that Phase 3 executes:

- Arm C: corrupted then cleaned, for each mechanism at each registered rate.
- Arm D: clean then cleaned, the cost control, with no rate dimension.

Arms A and B are reused from the frozen Phase 2 results and are never generated
here. Panels, datasets, seeds, models, and corruption parameters are inherited
from ``configs/full_study.yaml`` so the arms are genuinely paired.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from tfm_dirty_values.config import canonical_json

PHASE3_STAGE = 4


def normalize_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return parameters in key-sorted insertion order.

    This is load-bearing, not cosmetic. ``tfm_dirty_values.injection._stable_seed``
    derives the corruption RNG seed by string-formatting the parameters dict, so
    the dict's insertion order changes the seed, the affected-cell mask, and the
    corrupted array.

    Phase 2 executed with key-sorted parameters, because the task manifest was
    serialized through ``canonical_json`` (which sorts keys), persisted to SQLite
    as ``parameters_json``, and rebuilt with ``json.loads``. Reading the same
    values straight from YAML instead preserves YAML order and silently produces
    a different corruption.

    Phase 3 must reproduce Phase 2's corruption exactly for arm C to be paired
    with frozen arm B, so every Phase 3 code path routes parameters through here.
    Mechanisms with a single parameter key are unaffected, which is why the
    paired-identity probe initially passed for measurement noise, outliers, and
    unit errors while failing for quantization and categorical errors.
    """
    return {key: parameters[key] for key in sorted(parameters)}


@dataclass(frozen=True)
class AblationTask:
    task_id: str
    stage: int
    arm: str
    dataset_id: int
    dataset_name: str
    seed: int
    corruption: str
    rate: float
    model: str
    cleaned: bool
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make_task(condition: dict[str, Any]) -> AblationTask:
    identity = canonical_json({"stage": PHASE3_STAGE, **condition})
    task_id = hashlib.sha256(identity.encode("ascii")).hexdigest()[:24]
    return AblationTask(task_id=task_id, stage=PHASE3_STAGE, **condition)


def ablation_tasks(
    config: dict[str, Any],
    phase2_config: dict[str, Any],
) -> list[AblationTask]:
    models = [str(value) for value in phase2_config["models"]["enabled"]]
    rates = [float(value) for value in config["corruptions"]["affected_cell_rates"]]
    parameters = phase2_config["corruptions"]["parameters"]
    panels = {
        str(name): {int(value) for value in data_ids}
        for name, data_ids in phase2_config["corruptions"]["panels"].items()
    }
    mechanisms = [str(value) for value in config["corruptions"]["enabled"]]

    tasks: list[AblationTask] = []
    for dataset in phase2_config["datasets"]:
        data_id = int(dataset["data_id"])
        for seed in phase2_config["seeds"]:
            common = {
                "dataset_id": data_id,
                "dataset_name": str(dataset["name"]),
                "seed": int(seed),
            }

            # Arm D: clean input, cleaner applied. No rate dimension.
            for model in models:
                tasks.append(
                    _make_task(
                        {
                            **common,
                            "arm": "d",
                            "corruption": "clean",
                            "rate": 0.0,
                            "model": model,
                            "cleaned": True,
                            "parameters": {},
                        }
                    )
                )

            # Arm C: corrupted input, cleaner applied.
            for mechanism in mechanisms:
                if data_id not in panels[mechanism]:
                    continue
                for rate in rates:
                    for model in models:
                        tasks.append(
                            _make_task(
                                {
                                    **common,
                                    "arm": "c",
                                    "corruption": mechanism,
                                    "rate": rate,
                                    "model": model,
                                    "cleaned": True,
                                    "parameters": normalize_parameters(parameters[mechanism]),
                                }
                            )
                        )

    return sorted(
        tasks,
        key=lambda task: (
            task.stage,
            task.arm,
            task.dataset_id,
            task.seed,
            task.corruption,
            task.rate,
            task.model,
        ),
    )


def ablation_task_manifest(
    config: dict[str, Any],
    phase2_config: dict[str, Any],
) -> dict[str, Any]:
    tasks = [task.to_dict() for task in ablation_tasks(config, phase2_config)]
    arm_counts: dict[str, int] = {}
    for task in tasks:
        arm_counts[task["arm"]] = arm_counts.get(task["arm"], 0) + 1
    return {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "protocol_version": config["protocol_version"],
        "stage": PHASE3_STAGE,
        "config_sha256": config["_config_sha256"],
        "phase2_config_sha256": phase2_config["_config_sha256"],
        "task_count": len(tasks),
        "arm_counts": arm_counts,
        "task_manifest_sha256": hashlib.sha256(
            canonical_json(tasks).encode("ascii")
        ).hexdigest(),
        "tasks": tasks,
    }


def verify_expected_counts(manifest: dict[str, Any], config: dict[str, Any]) -> list[str]:
    """Return a list of mismatches against the preregistered counts."""
    expected = config["expected_counts"]
    errors: list[str] = []
    if manifest["task_count"] != int(expected["total_tasks"]):
        errors.append(
            f"Phase 3 task count {manifest['task_count']} does not match "
            f"preregistered {expected['total_tasks']}"
        )
    arm_counts = manifest["arm_counts"]
    if arm_counts.get("c", 0) != int(expected["arm_c_tasks"]):
        errors.append(
            f"Arm C count {arm_counts.get('c', 0)} does not match "
            f"preregistered {expected['arm_c_tasks']}"
        )
    if arm_counts.get("d", 0) != int(expected["arm_d_tasks"]):
        errors.append(
            f"Arm D count {arm_counts.get('d', 0)} does not match "
            f"preregistered {expected['arm_d_tasks']}"
        )
    task_ids = [task["task_id"] for task in manifest["tasks"]]
    if len(set(task_ids)) != len(task_ids):
        errors.append("Phase 3 task manifest contains duplicate task identities")
    return errors
