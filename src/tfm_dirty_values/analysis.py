from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .state import RunStore


def _completed_frame(store: RunStore) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in store.rows(status="completed"):
        metrics = json.loads(row["metrics_json"])
        records.append({**row, "roc_auc": float(metrics["roc_auc"])})
    return pd.DataFrame(records)


def degradation_frame(store: RunStore, *, stage: int) -> pd.DataFrame:
    frame = _completed_frame(store)
    if frame.empty:
        return pd.DataFrame()
    baselines = frame[frame["corruption"] == "clean"][
        ["dataset_id", "dataset_name", "seed", "model", "roc_auc"]
    ].rename(columns={"roc_auc": "clean_roc_auc"})
    polluted = frame[(frame["stage"] == stage) & (frame["corruption"] != "clean")].copy()
    merged = polluted.merge(
        baselines,
        on=["dataset_id", "dataset_name", "seed", "model"],
        how="left",
        validate="many_to_one",
    )
    if merged["clean_roc_auc"].isna().any():
        raise ValueError("Every polluted task requires a matching clean baseline")
    merged["degradation"] = merged["clean_roc_auc"] - merged["roc_auc"]
    return merged


def evaluate_kill_test(store: RunStore, config: dict[str, Any], *, stage: int) -> dict[str, Any]:
    frame = degradation_frame(store, stage=stage)
    if frame.empty:
        raise ValueError(f"No completed polluted tasks are available for stage {stage}")
    expected_corruptions = set(config["corruptions"]["enabled"])
    if set(frame["corruption"].unique()) != expected_corruptions:
        raise ValueError("Completed corruption panel is incomplete")

    dataset_summary = (
        frame.groupby(["dataset_id", "dataset_name", "corruption", "rate", "model"], as_index=False)[
            "degradation"
        ]
        .mean()
        .sort_values(["corruption", "rate", "model", "dataset_id"])
    )
    primary_rate = float(config["analysis"]["primary_rate"])
    primary_rows = frame[np.isclose(frame["rate"], primary_rate)].copy()
    expected_groups = {
        (int(dataset["data_id"]), str(corruption), str(model))
        for dataset in config["datasets"]
        for corruption in expected_corruptions
        for model in config["models"]["enabled"]
    }
    observed_groups = {
        (int(row.dataset_id), str(row.corruption), str(row.model))
        for row in primary_rows[["dataset_id", "corruption", "model"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    missing_groups = sorted(expected_groups - observed_groups)
    if missing_groups:
        raise ValueError(f"Primary panel is missing dataset/corruption/model groups: {missing_groups}")
    seed_counts = (
        primary_rows.groupby(["dataset_id", "corruption", "model"], as_index=False)["seed"]
        .nunique()
        .rename(columns={"seed": "completed_seeds"})
    )
    expected_seed_count = len(config["seeds"])
    incomplete_seed_groups = seed_counts[seed_counts["completed_seeds"] < expected_seed_count]
    primary = dataset_summary[np.isclose(dataset_summary["rate"], primary_rate)].copy()
    if primary.empty:
        raise ValueError(f"Primary rate {primary_rate} is missing")

    criterion_a: dict[str, Any] = {}
    tfm_threshold = float(config["analysis"]["pass_tfm_degradation_auc"])
    for corruption in sorted(expected_corruptions):
        values = primary[
            (primary["corruption"] == corruption) & (primary["model"] == "tabpfn")
        ]
        dataset_values = {
            str(int(row.dataset_id)): float(row.degradation)
            for row in values.itertuples(index=False)
        }
        mean_degradation = float(values["degradation"].mean()) if len(values) else float("nan")
        criterion_a[corruption] = {
            "mean_tfm_degradation": mean_degradation,
            "dataset_degradations": dataset_values,
            "qualifies": bool(len(values) and mean_degradation >= tfm_threshold),
        }

    criterion_b: dict[str, Any] = {}
    gap_threshold = float(config["analysis"]["pass_model_gap_auc"])
    for corruption in sorted(expected_corruptions):
        values = primary[primary["corruption"] == corruption].pivot(
            index="dataset_id",
            columns="model",
            values="degradation",
        )
        if not {"tabpfn", "xgboost"}.issubset(values.columns):
            criterion_b[corruption] = {"available": False, "qualifies": False}
            continue
        gaps = values["tabpfn"] - values["xgboost"]
        mean_gap = float(gaps.mean())
        signs = np.sign(gaps.to_numpy())
        same_direction = bool(len(signs) and np.all(signs == signs[0]) and signs[0] != 0)
        criterion_b[corruption] = {
            "available": True,
            "mean_gap": mean_gap,
            "dataset_gaps": {str(int(key)): float(value) for key, value in gaps.items()},
            "same_direction": same_direction,
            "qualifies": bool(abs(mean_gap) >= gap_threshold and same_direction),
        }

    combo_summary = (
        primary.groupby(["corruption", "model"], as_index=False)["degradation"].mean()
    )
    max_combo_degradation = float(combo_summary["degradation"].max())
    passes = any(value["qualifies"] for value in criterion_a.values()) or any(
        value["qualifies"] for value in criterion_b.values()
    )
    stress_threshold = float(config["analysis"]["stress_trigger_max_degradation_auc"])
    if passes:
        decision = "GO"
    elif max_combo_degradation < stress_threshold:
        decision = "STRESS_RETEST" if stage == 1 else "NO_GO"
    else:
        decision = "REVIEW"
    return {
        "stage": int(stage),
        "decision": decision,
        "primary_rate": primary_rate,
        "criterion_a": criterion_a,
        "criterion_b": criterion_b,
        "max_combo_degradation": max_combo_degradation,
        "expected_seeds_per_group": expected_seed_count,
        "incomplete_seed_groups": incomplete_seed_groups.to_dict(orient="records"),
        "combo_summary": combo_summary.to_dict(orient="records"),
        "dataset_summary": dataset_summary.to_dict(orient="records"),
    }
