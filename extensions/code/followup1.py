# Extracted first-extension analysis functions.
from common import *
import pandas as pd

SEEDS = {1907, 2027, 4049, 6029, 8089}
KEYS = ["experiment", "dataset_id", "seed", "corruption", "rate", "model"]
GROUPS = ["experiment", "corruption", "rate", "metric", "estimand"]
FAMILIES = {"tfm_mean": ["tabpfn", "tabiclv2"],
            "boosting_mean": ["xgboost", "lightgbm"],
            "tree_mean": ["xgboost", "lightgbm", "random_forest"]}
def task_id(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:24]


def improvement(metric, new, old):
    return (1 if metric in ["roc_auc", "balanced_accuracy"] else -1) * (new - old)


def decomposition(metric, old_clean, old_dirty, new_clean, new_dirty):
    clean = improvement(metric, new_clean, old_clean)
    dirty = improvement(metric, new_dirty, old_dirty)
    base_loss = improvement(metric, old_clean, old_dirty)
    new_loss = improvement(metric, new_clean, new_dirty)
    reduction = base_loss - new_loss
    if not np.isclose(reduction, dirty - clean, atol=1e-12, rtol=0):
        raise ValueError("Degradation decomposition failed")
    return dict(clean_change=clean, dirty_change=dirty,
                baseline_degradation=base_loss, intervention_degradation=new_loss,
                degradation_reduction=reduction)


def paired_effects(actual, full, legacy):
    archives = {"matched_uncleaned": {(condition(r), r["model"]): r for r in full},
                "matched_legacy": {(condition(r), r["model"]): r for r in legacy}}

    def control(task, arm):
        if task["model"] == "tabiclv2":
            return actual[task_id({**task, "experiment": arm})]["metrics"]
        return archives[arm][(condition(task), task["model"])]["metrics"]

    rows = []
    for result in actual.values():
        task = result["task"]
        experiment = task["experiment"]
        if experiment.startswith("matched_"):
            continue
        if experiment == "canonical_tukey":
            original, old = control(task, "matched_uncleaned"), control(task, "matched_legacy")
            for metric in METRICS:
                new_value = result["metrics"][metric]
                effects = {"canonical_vs_legacy": improvement(metric, new_value, old[metric]),
                           "canonical_vs_uncleaned": improvement(metric, new_value, original[metric]),
                           "legacy_vs_uncleaned": improvement(metric, old[metric], original[metric])}
                for estimand, value in effects.items():
                    rows.append(dict(**task, metric=metric, estimand=estimand, value=value))
        elif task["corruption"] != "clean":
            clean_task = {**task, "corruption": "clean", "rate": 0.0}
            base_clean, base_dirty = control(clean_task, "matched_uncleaned"), control(task, "matched_uncleaned")
            changed_clean = actual[task_id(clean_task)]["metrics"]
            for metric in METRICS:
                effects = decomposition(metric, base_clean[metric], base_dirty[metric],
                                        changed_clean[metric], result["metrics"][metric])
                for estimand, value in effects.items():
                    rows.append(dict(**task, metric=metric, estimand=estimand, value=value))
    return pd.DataFrame(rows)


def aggregate_seeds(effects):
    group = GROUPS + ["dataset_id", "model"]
    if effects.duplicated(group + ["seed"]).any():
        raise ValueError("Duplicate paired seed effect")
    for key, values in effects.groupby(group, sort=True):
        if set(values["seed"]) != SEEDS or len(values) != 5:
            raise ValueError(f"Incomplete five-seed pairing: {key}")
    return effects.groupby(group, as_index=False, sort=True)["value"].mean()


def add_families(datasets):
    index = GROUPS + ["dataset_id"]
    wide = datasets.pivot(index=index, columns="model", values="value")
    if set(wide.columns) != set(MODELS) or wide.isna().any().any():
        raise ValueError("Unpaired model family")
    extra = []
    for name, models in FAMILIES.items():
        table = wide[models].mean(axis=1).rename("value").reset_index()
        table["model"] = name
        extra.append(table)
    comparisons = [("tfm_minus_trees", MODELS[2:]),
                   ("tfm_minus_boosting", ["xgboost", "lightgbm"]),
                   ("tfm_minus_random_forest", ["random_forest"])]
    for name, models in comparisons:
        table = (wide[MODELS[:2]].mean(axis=1) - wide[models].mean(axis=1)).rename("value").reset_index()
        table["model"] = name
        extra.append(table)
    return pd.concat([datasets, *extra], ignore_index=True).sort_values(index + ["model"])


def summarize(datasets):
    group = GROUPS + ["model"]
    rows = []
    for key, part in datasets.groupby(group, sort=True):
        rows.append(dict(zip(group, key)) | bootstrap(part["value"].to_numpy(), ("followup-v1", *key)))
    return pd.DataFrame(rows)
