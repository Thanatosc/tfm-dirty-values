from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

from .config import canonical_json


@dataclass(frozen=True)
class Task:
    task_id: str
    stage: int
    dataset_id: int
    dataset_name: str
    seed: int
    corruption: str
    rate: float
    model: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make_task(stage: int, condition: dict[str, Any]) -> Task:
    identity = canonical_json({"stage": stage, **condition})
    task_id = hashlib.sha256(identity.encode("ascii")).hexdigest()[:24]
    return Task(task_id=task_id, stage=stage, **condition)


def kill_test_tasks(config: dict[str, Any], *, stress: bool = False) -> list[Task]:
    stage = 2 if stress else 1
    parameters_key = "stress_parameters" if stress else "parameters"
    parameters = config["corruptions"][parameters_key]
    models = [str(value) for value in config["models"]["enabled"]]
    tasks: list[Task] = []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            common = {
                "dataset_id": int(dataset["data_id"]),
                "dataset_name": str(dataset["name"]),
                "seed": int(seed),
            }
            if not stress:
                for model in models:
                    tasks.append(
                        _make_task(
                            stage,
                            {
                                **common,
                                "corruption": "clean",
                                "rate": 0.0,
                                "model": model,
                                "parameters": {},
                            },
                        )
                    )
            for corruption in config["corruptions"]["enabled"]:
                rates = (
                    [float(config["analysis"]["primary_rate"])]
                    if stress
                    else [float(value) for value in config["corruptions"]["affected_cell_rates"]]
                )
                for rate in rates:
                    for model in models:
                        tasks.append(
                            _make_task(
                                stage,
                                {
                                    **common,
                                    "corruption": str(corruption),
                                    "rate": rate,
                                    "model": model,
                                    "parameters": dict(parameters[corruption]),
                                },
                            )
                        )
    return sorted(
        tasks,
        key=lambda task: (
            task.stage,
            task.dataset_id,
            task.seed,
            task.corruption != "clean",
            task.corruption,
            task.rate,
            task.model,
        ),
    )


def full_study_tasks(config: dict[str, Any]) -> list[Task]:
    """Build the frozen Phase 2 task panel without accessing model outcomes."""
    stage = 3
    models = [str(value) for value in config["models"]["enabled"]]
    rates = [float(value) for value in config["corruptions"]["affected_cell_rates"]]
    parameters = config["corruptions"]["parameters"]
    panels = {
        str(name): {int(value) for value in data_ids}
        for name, data_ids in config["corruptions"]["panels"].items()
    }
    tasks: list[Task] = []
    for dataset in config["datasets"]:
        data_id = int(dataset["data_id"])
        for seed in config["seeds"]:
            common = {
                "dataset_id": data_id,
                "dataset_name": str(dataset["name"]),
                "seed": int(seed),
            }
            for model in models:
                tasks.append(
                    _make_task(
                        stage,
                        {
                            **common,
                            "corruption": "clean",
                            "rate": 0.0,
                            "model": model,
                            "parameters": {},
                        },
                    )
                )
            for corruption in config["corruptions"]["enabled"]:
                corruption = str(corruption)
                if data_id not in panels[corruption]:
                    continue
                for rate in rates:
                    for model in models:
                        tasks.append(
                            _make_task(
                                stage,
                                {
                                    **common,
                                    "corruption": corruption,
                                    "rate": rate,
                                    "model": model,
                                    "parameters": dict(parameters[corruption]),
                                },
                            )
                        )
    return sorted(
        tasks,
        key=lambda task: (
            task.stage,
            task.dataset_id,
            task.seed,
            task.corruption != "clean",
            task.corruption,
            task.rate,
            task.model,
        ),
    )
