"""Phase 3 confirmatory analysis.

Reuses the Phase 2 inference functions unchanged: exact two-sided sign-flip tests
on dataset-level aggregates, Holm correction within a hypothesis family, and
dataset bootstrap intervals. The dataset is the independent unit and the five
seeds are averaged arithmetically first.

Arms A and B are read from the frozen Phase 2 database read-only. Arms C and D
come from the Phase 3 database. The registered ECE key alias is applied on read,
exactly as Phase 2's compatibility amendment specifies.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tfm_dirty_values.config import resolve_path
from tfm_dirty_values.full_analysis import (
    _bootstrap_seed,
    bootstrap_mean_ci,
    exact_sign_flip_pvalue,
    holm_adjust,
)

from .design import ablation_tasks

METRICS = ("roc_auc", "balanced_accuracy", "log_loss", "ece")

# Higher is better for these; lower is better for the rest.
_HIGHER_IS_BETTER = {"roc_auc", "balanced_accuracy"}


def _normalize_metrics(payload: dict[str, Any], task_id: str) -> dict[str, float]:
    """Apply the registered ECE alias and extract the four analysis metrics."""
    generic = payload.get("ece")
    stored = payload.get("ece_10_equal_width")
    if generic is not None and stored is not None and float(generic) != float(stored):
        raise ValueError(
            f"Conflicting ECE values for task {task_id}: ece={generic}, "
            f"ece_10_equal_width={stored}"
        )
    if generic is None:
        if stored is None:
            raise ValueError(f"Task {task_id} has neither ece nor ece_10_equal_width")
        generic = float(stored)
    return {
        "roc_auc": float(payload["roc_auc"]),
        "balanced_accuracy": float(payload["balanced_accuracy"]),
        "log_loss": float(payload["log_loss"]),
        "ece": float(generic),
    }


def _read_rows(database: Path, *, with_arm: bool) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        if with_arm:
            query = """
                SELECT r.task_id, r.dataset_id, r.seed, r.corruption, r.rate,
                       r.model, r.status, r.metrics_json, a.arm AS arm
                  FROM runs r JOIN ablation_arms a ON a.task_id = r.task_id
            """
        else:
            query = """
                SELECT task_id, dataset_id, seed, corruption, rate,
                       model, status, metrics_json, NULL AS arm
                  FROM runs
            """
        return [dict(row) for row in connection.execute(query)]
    finally:
        connection.close()


def _improvement(metric: str, reference: float, candidate: float) -> float:
    """Signed improvement of ``candidate`` over ``reference``.

    Positive always means candidate is better, whichever direction the metric
    runs. This keeps recovery and cost signs interpretable across metrics.
    """
    if metric in _HIGHER_IS_BETTER:
        return float(candidate - reference)
    return float(reference - candidate)


def _cell_means(
    per_seed: dict[tuple, list[float]],
) -> dict[tuple, float]:
    return {key: float(np.mean(values)) for key, values in per_seed.items()}


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


def evaluate_cleaning_ablation(
    config: dict[str, Any],
    phase2_config: dict[str, Any],
) -> dict[str, Any]:
    phase3_db = resolve_path(config, config["paths"]["database"])
    phase2_db = resolve_path(config, config["paths"]["phase2_database"])
    if not phase3_db.is_file():
        raise ValueError(f"Missing Phase 3 database: {phase3_db}")
    if not phase2_db.is_file():
        raise ValueError(f"Missing frozen Phase 2 database: {phase2_db}")

    expected = {task.task_id for task in ablation_tasks(config, phase2_config)}
    phase3_rows = _read_rows(phase3_db, with_arm=True)
    actual = {str(row["task_id"]) for row in phase3_rows}
    if actual != expected:
        raise ValueError("Persisted Phase 3 tasks do not match the frozen manifest")
    incomplete = [row for row in phase3_rows if row["status"] != "completed"]
    if incomplete:
        raise ValueError(
            f"Confirmatory analysis requires all frozen Phase 3 tasks; "
            f"{len(incomplete)} are incomplete"
        )

    phase2_rows = [row for row in _read_rows(phase2_db, with_arm=False) if row["status"] == "completed"]

    # arm A: clean uncleaned; arm B: corrupted uncleaned. Both from Phase 2.
    arm_a: dict[tuple[int, int, str], dict[str, float]] = {}
    arm_b: dict[tuple[int, int, str, str, float], dict[str, float]] = {}
    for row in phase2_rows:
        payload = _normalize_metrics(json.loads(row["metrics_json"]), str(row["task_id"]))
        key_base = (int(row["dataset_id"]), int(row["seed"]), str(row["model"]))
        if row["corruption"] == "clean":
            arm_a[key_base] = payload
        else:
            arm_b[(*key_base, str(row["corruption"]), float(row["rate"]))] = payload

    # arm C: corrupted cleaned; arm D: clean cleaned. Both from Phase 3.
    arm_c: dict[tuple[int, int, str, str, float], dict[str, float]] = {}
    arm_d: dict[tuple[int, int, str], dict[str, float]] = {}
    for row in phase3_rows:
        payload = _normalize_metrics(json.loads(row["metrics_json"]), str(row["task_id"]))
        key_base = (int(row["dataset_id"]), int(row["seed"]), str(row["model"]))
        if str(row["arm"]) == "d":
            arm_d[key_base] = payload
        else:
            arm_c[(*key_base, str(row["corruption"]), float(row["rate"]))] = payload

    rates = [float(value) for value in config["corruptions"]["affected_cell_rates"]]
    mechanisms = [str(value) for value in config["corruptions"]["enabled"]]
    panels = {
        str(name): [int(value) for value in ids]
        for name, ids in phase2_config["corruptions"]["panels"].items()
    }
    models = [str(value) for value in phase2_config["models"]["enabled"]]
    tfms = [str(value) for value in phase2_config["models"]["tfm_class"]]
    trees = [str(value) for value in phase2_config["models"]["tree_class"]]
    seeds = [int(value) for value in phase2_config["seeds"]]

    recovery_seed: dict[tuple, list[float]] = defaultdict(list)
    residual_seed: dict[tuple, list[float]] = defaultdict(list)
    damage_seed: dict[tuple, list[float]] = defaultdict(list)
    cost_seed: dict[tuple, list[float]] = defaultdict(list)

    for mechanism in mechanisms:
        for data_id in panels[mechanism]:
            for rate in rates:
                for model in models:
                    for seed in seeds:
                        base = (data_id, seed, model)
                        cell = (data_id, seed, model, mechanism, rate)
                        a = arm_a[base]
                        b = arm_b[cell]
                        c = arm_c[cell]
                        for metric in METRICS:
                            key = (data_id, model, mechanism, rate, metric)
                            # Recovery: cleaned corrupted vs uncleaned corrupted.
                            recovery_seed[key].append(_improvement(metric, b[metric], c[metric]))
                            # Residual: how far cleaned still sits from clean.
                            residual_seed[key].append(_improvement(metric, c[metric], a[metric]))
                            # Phase 2 damage, recomputed here for the fraction denominator.
                            damage_seed[key].append(_improvement(metric, b[metric], a[metric]))

    for data_id in {int(item["data_id"]) for item in phase2_config["datasets"]}:
        for model in models:
            for seed in seeds:
                base = (data_id, seed, model)
                for metric in METRICS:
                    cost_seed[(data_id, model, metric)].append(
                        _improvement(metric, arm_a[base][metric], arm_d[base][metric])
                    )

    recovery = _cell_means(recovery_seed)
    residual = _cell_means(residual_seed)
    damage = _cell_means(damage_seed)
    cost = _cell_means(cost_seed)

    min_denominator = float(config["analysis"]["recovery_fraction_min_denominator"])
    base_seed = int(config["analysis"]["bootstrap_seed"])

    # ---- Confirmatory H4: TFM recovery, one test per mechanism per rate ----
    h4_records: list[dict[str, Any]] = []
    for mechanism in mechanisms:
        for rate in rates:
            values = [
                float(np.mean([recovery[(data_id, model, mechanism, rate, "roc_auc")] for model in tfms]))
                for data_id in panels[mechanism]
            ]
            h4_records.append(
                {
                    "corruption": mechanism,
                    "rate": rate,
                    "family": "h4_tfm_recovery",
                    **_test_record(values, config=config, identity=("h4", mechanism, rate)),
                }
            )
    for record, adjusted in zip(
        h4_records, holm_adjust(item["p_raw"] for item in h4_records), strict=True
    ):
        record["p_holm"] = adjusted

    # ---- Confirmatory H5: cleaning cost on clean data ----
    all_data_ids = sorted({int(item["data_id"]) for item in phase2_config["datasets"]})
    h5_values = [
        float(np.mean([cost[(data_id, model, "roc_auc")] for model in tfms]))
        for data_id in all_data_ids
    ]
    h5_record = {
        "family": "h5_cleaning_cost_on_clean",
        **_test_record(h5_values, config=config, identity=("h5",)),
    }
    h5_record["p_holm"] = holm_adjust([h5_record["p_raw"]])[0]

    # ---- Descriptive: per mechanism, rate, model class, and metric ----
    descriptive: list[dict[str, Any]] = []
    for mechanism in mechanisms:
        for rate in rates:
            for label, group in (("tfm", tfms), ("tree", trees)):
                for metric in METRICS:
                    rec_values = [
                        float(np.mean([recovery[(d, m, mechanism, rate, metric)] for m in group]))
                        for d in panels[mechanism]
                    ]
                    res_values = [
                        float(np.mean([residual[(d, m, mechanism, rate, metric)] for m in group]))
                        for d in panels[mechanism]
                    ]
                    dmg_values = [
                        float(np.mean([damage[(d, m, mechanism, rate, metric)] for m in group]))
                        for d in panels[mechanism]
                    ]
                    fractions = [
                        rec / dmg
                        for rec, dmg in zip(rec_values, dmg_values, strict=True)
                        if dmg > min_denominator
                    ]
                    descriptive.append(
                        {
                            "corruption": mechanism,
                            "rate": rate,
                            "model_class": label,
                            "metric": metric,
                            "n_datasets": len(rec_values),
                            "recovery": bootstrap_mean_ci(
                                rec_values,
                                resamples=int(config["analysis"]["bootstrap_resamples"]),
                                seed=_bootstrap_seed(
                                    base_seed, "rec", mechanism, rate, label, metric
                                ),
                            ),
                            "residual_damage": bootstrap_mean_ci(
                                res_values,
                                resamples=int(config["analysis"]["bootstrap_resamples"]),
                                seed=_bootstrap_seed(
                                    base_seed, "res", mechanism, rate, label, metric
                                ),
                            ),
                            "phase2_damage": bootstrap_mean_ci(
                                dmg_values,
                                resamples=int(config["analysis"]["bootstrap_resamples"]),
                                seed=_bootstrap_seed(
                                    base_seed, "dmg", mechanism, rate, label, metric
                                ),
                            ),
                            "recovery_fraction_mean": (
                                float(np.mean(fractions)) if fractions else None
                            ),
                            "recovery_fraction_n": len(fractions),
                            "recovery_fraction_excluded": len(rec_values) - len(fractions),
                        }
                    )

    # ---- Descriptive: per-model cleaning cost on clean data ----
    cost_records: list[dict[str, Any]] = []
    for model in models:
        for metric in METRICS:
            values = [cost[(data_id, model, metric)] for data_id in all_data_ids]
            cost_records.append(
                {
                    "model": model,
                    "metric": metric,
                    "n_datasets": len(values),
                    **bootstrap_mean_ci(
                        values,
                        resamples=int(config["analysis"]["bootstrap_resamples"]),
                        seed=_bootstrap_seed(base_seed, "cost", model, metric),
                    ),
                }
            )

    # ---- Descriptive: cleaning-priority ordering by TFM ROC-AUC recovery ----
    ranking: list[dict[str, Any]] = []
    for rate in rates:
        ordered = sorted(
            (
                record
                for record in h4_records
                if record["rate"] == rate
            ),
            key=lambda item: item["mean"],
            reverse=True,
        )
        ranking.append(
            {
                "rate": rate,
                "order": [
                    {
                        "rank": index + 1,
                        "corruption": record["corruption"],
                        "mean_recovery": record["mean"],
                        "ci_low": record["ci_low"],
                        "ci_high": record["ci_high"],
                        "p_holm": record["p_holm"],
                    }
                    for index, record in enumerate(ordered)
                ],
            }
        )

    return {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "protocol_version": config["protocol_version"],
        "stage": 4,
        "status": "ANALYZED",
        "task_count": len(phase3_rows),
        "primary_unit": "dataset",
        "seed_aggregation": phase2_config["analysis"]["seed_aggregation"],
        "ece_alias_applied": True,
        "arms": {
            "a_clean_uncleaned": len(arm_a),
            "b_corrupted_uncleaned": len(arm_b),
            "c_corrupted_cleaned": len(arm_c),
            "d_clean_cleaned": len(arm_d),
        },
        "cleaner": config["cleaner"],
        "confirmatory": {
            "h4_tfm_recovery": h4_records,
            "h5_cleaning_cost_on_clean": h5_record,
        },
        "descriptive": descriptive,
        "cleaning_cost_by_model": cost_records,
        "cleaning_priority_ranking": ranking,
    }
