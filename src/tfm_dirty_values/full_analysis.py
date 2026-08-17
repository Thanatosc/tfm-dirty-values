from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .design import full_study_tasks
from .state import RunStore


def exact_sign_flip_pvalue(values: Iterable[float]) -> float:
    sample = np.asarray(list(values), dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if not len(sample):
        raise ValueError("Sign-flip test requires at least one finite dataset aggregate")
    observed = abs(float(np.mean(sample)))
    exceedances = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(sample)):
        statistic = abs(float(np.mean(sample * np.asarray(signs))))
        exceedances += statistic >= observed - 1e-15
        total += 1
    return float(exceedances / total)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    count = len(values)
    order = sorted(range(count), key=values.__getitem__)
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    sample = np.asarray(list(values), dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if not len(sample):
        raise ValueError("Bootstrap requires at least one finite dataset aggregate")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sample), size=(int(resamples), len(sample)))
    means = sample[indices].mean(axis=1)
    return {
        "mean": float(np.mean(sample)),
        "median": float(np.median(sample)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
    }


def _bootstrap_seed(base: int, *parts: Any) -> int:
    identity = "|".join(str(part) for part in (base, *parts))
    return int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "little")


def _degradation(metric: str, clean: float, polluted: float) -> float:
    if metric in {"roc_auc", "balanced_accuracy"}:
        return float(clean - polluted)
    if metric in {"log_loss", "ece"}:
        return float(polluted - clean)
    raise KeyError(f"Unknown metric direction: {metric}")


def _test_record(
    values: list[float],
    *,
    config: dict[str, Any],
    identity: tuple[Any, ...],
) -> dict[str, Any]:
    bootstrap = bootstrap_mean_ci(
        values,
        resamples=int(config["analysis"]["bootstrap_resamples"]),
        seed=_bootstrap_seed(int(config["analysis"]["bootstrap_seed"]), *identity),
    )
    return {
        "n_datasets": len(values),
        "dataset_values": values,
        **bootstrap,
        "p_raw": exact_sign_flip_pvalue(values),
    }


def _apply_holm_by_panel(records: list[dict[str, Any]]) -> None:
    for panel in {str(record["multiplicity_panel"]) for record in records}:
        selected = [record for record in records if record["multiplicity_panel"] == panel]
        adjusted = holm_adjust(record["p_raw"] for record in selected)
        for record, value in zip(selected, adjusted, strict=True):
            record["p_holm"] = value


def evaluate_full_study(store: RunStore, config: dict[str, Any]) -> dict[str, Any]:
    expected_tasks = full_study_tasks(config)
    expected_ids = {task.task_id for task in expected_tasks}
    rows = store.rows(stage=3)
    actual_ids = {str(row["task_id"]) for row in rows}
    if actual_ids != expected_ids:
        raise ValueError("Persisted full-study tasks do not match the frozen manifest")
    incomplete = [row for row in rows if row["status"] != "completed"]
    if incomplete:
        raise ValueError(
            f"Confirmatory analysis requires all frozen tasks; {len(incomplete)} are incomplete"
        )

    metrics = ["roc_auc", "balanced_accuracy", "log_loss", "ece"]
    clean: dict[tuple[int, int, str], dict[str, float]] = {}
    polluted_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        payload = json.loads(row["metrics_json"])
        if row["corruption"] == "clean":
            clean[(int(row["dataset_id"]), int(row["seed"]), str(row["model"]))] = payload
        else:
            polluted_rows.append((row, payload))

    within_dataset: dict[tuple[int, str, str, float, str], list[float]] = defaultdict(list)
    for row, payload in polluted_rows:
        baseline = clean[(int(row["dataset_id"]), int(row["seed"]), str(row["model"]))]
        for metric in metrics:
            within_dataset[
                (
                    int(row["dataset_id"]),
                    str(row["model"]),
                    str(row["corruption"]),
                    float(row["rate"]),
                    metric,
                )
            ].append(_degradation(metric, float(baseline[metric]), float(payload[metric])))

    aggregates = {
        key: float(np.mean(values)) for key, values in within_dataset.items()
    }
    base_seed = int(config["analysis"]["bootstrap_seed"])
    summaries: list[dict[str, Any]] = []
    for model in config["models"]["enabled"]:
        for corruption in config["corruptions"]["enabled"]:
            data_ids = [int(value) for value in config["corruptions"]["panels"][corruption]]
            for rate in config["corruptions"]["affected_cell_rates"]:
                for metric in metrics:
                    values = [
                        aggregates[(data_id, str(model), str(corruption), float(rate), metric)]
                        for data_id in data_ids
                    ]
                    summaries.append(
                        {
                            "model": str(model),
                            "corruption": str(corruption),
                            "rate": float(rate),
                            "metric": metric,
                            "n_datasets": len(values),
                            **bootstrap_mean_ci(
                                values,
                                resamples=int(config["analysis"]["bootstrap_resamples"]),
                                seed=_bootstrap_seed(
                                    base_seed, "summary", model, corruption, rate, metric
                                ),
                            ),
                        }
                    )

    primary_rate = float(config["analysis"]["primary_rate"])
    tfms = [str(value) for value in config["models"]["tfm_class"]]
    trees = [str(value) for value in config["models"]["tree_class"]]
    all_models = tfms + trees
    rates = np.asarray(config["corruptions"]["affected_cell_rates"], dtype=np.float64)
    hypotheses: dict[str, list[dict[str, Any]]] = {"h1": [], "h2": [], "h3": []}
    for corruption in config["corruptions"]["enabled"]:
        corruption = str(corruption)
        data_ids = [int(value) for value in config["corruptions"]["panels"][corruption]]
        panel = "categorical_singleton" if corruption == "categorical_error" else "numeric_four"

        h1_values = [
            float(
                np.mean(
                    [aggregates[(data_id, model, corruption, primary_rate, "roc_auc")] for model in tfms]
                )
            )
            for data_id in data_ids
        ]
        h2_values = [
            float(
                np.mean(
                    [aggregates[(data_id, model, corruption, primary_rate, "roc_auc")] for model in tfms]
                )
                - np.mean(
                    [aggregates[(data_id, model, corruption, primary_rate, "roc_auc")] for model in trees]
                )
            )
            for data_id in data_ids
        ]
        h3_values: list[float] = []
        for data_id in data_ids:
            slopes = []
            for model in all_models:
                curve = np.asarray(
                    [aggregates[(data_id, model, corruption, float(rate), "roc_auc")] for rate in rates]
                )
                slopes.append(float(np.polyfit(rates, curve, deg=1)[0]))
            h3_values.append(float(np.mean(slopes)))

        for name, values in (("h1", h1_values), ("h2", h2_values), ("h3", h3_values)):
            record = {
                "corruption": corruption,
                "multiplicity_panel": panel,
                **_test_record(values, config=config, identity=(name, corruption)),
            }
            hypotheses[name].append(record)

    for records in hypotheses.values():
        _apply_holm_by_panel(records)

    return {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "stage": 3,
        "status": "ANALYZED",
        "task_count": len(rows),
        "primary_unit": "dataset",
        "seed_aggregation": config["analysis"]["seed_aggregation"],
        "hypotheses": hypotheses,
        "condition_summaries": summaries,
    }
