# Extracted analysis functions; no model loading or input reconstruction.
from common import *
import pandas as pd

SIGN = {"roc_auc": 1., "balanced_accuracy": 1., "log_loss": -1., "ece_10_equal_width": -1.}
CONDITION = ["cohort", "corruption", "level", "rate", "placement", "pattern"]
GROUP = CONDITION + ["representation", "contrast", "estimand", "metric", "model"]
def decomposition(clean_a, dirty_a, clean_b, dirty_b, sign=1.):
    clean = sign * (clean_b-clean_a)
    dirty = sign * (dirty_b-dirty_a)
    return {"clean_change": clean, "dirty_change": dirty, "degradation_reduction": dirty-clean}


def effect_rows(records):
    effects = []
    values = []
    def get(task):
        return records[task_id(task)]["metrics"]
    def emit(task, contrast, estimand, metric, value, representation=None):
        effects.append({**{k: task[k] for k in CONDITION},
            "representation": representation or task["representation"],
            "dataset_id": task["dataset_id"], "seed": task["seed"], "model": task["model"],
            "contrast": contrast, "estimand": estimand, "metric": metric, "value": float(value)})
    for record in records.values():
        task = record["task"]
        metric_values = record["metrics"]
        values.append({**task, **metric_values, "source": record["source"]})
        for metric in METRICS:
            emit(task, "absolute", "metric_value", metric, metric_values[metric])
        if task["corruption"] == "clean": continue
        clean = make_task(task["dataset_id"], task["seed"], task["model"], representation=task["representation"])
        clean_values = get(clean)
        for metric in METRICS:
            emit(task, "clean_to_dirty", "degradation", metric, SIGN[metric]*(clean_values[metric]-metric_values[metric]))
        if task["representation"] != "raw":
            alternatives = ["raw"] + (["affine"] if task["representation"] == "asinh" else [])
            for source_rep in alternatives:
                a = {**task, "representation": source_rep}
                if task_id(a) not in records: continue
                ca = {**clean, "representation": source_rep}
                for metric in METRICS:
                    dec = decomposition(get(ca)[metric], get(a)[metric], clean_values[metric],
                                        metric_values[metric], SIGN[metric])
                    for estimand, value in dec.items():
                        emit(task, source_rep + "_to_" + task["representation"], estimand, metric, value)
    # Predetermined pairwise severity and placement changes. The sign always
    # means the oriented performance of the named second condition minus first.
    for record in records.values():
        t = record["task"]
        if t["corruption"] == "clean" or t["cohort"] != "original" or t["pattern"] != "iid": continue
        if t["placement"] == "joint":
            for level in LEVELS[t["corruption"]]:
                if level >= t["level"]: continue
                a = {**t, "level": float(level)}
                if task_id(a) not in records: continue
                for metric in METRICS:
                    emit(t, f"level_{level:g}_to_{t['level']:g}", "performance_change", metric,
                         SIGN[metric]*(record["metrics"][metric]-get(a)[metric]))
        placements = ["context", "query", "joint"]
        for earlier in placements[:placements.index(t["placement"])]:
            a = {**t, "placement": earlier}
            if task_id(a) not in records: continue
            for metric in METRICS:
                emit(t, earlier + "_to_" + t["placement"], "performance_change", metric,
                     SIGN[metric]*(record["metrics"][metric]-get(a)[metric]))
    return pd.DataFrame(values), pd.DataFrame(effects)


def dataset_aggregate(seed_frame):
    group = GROUP + ["dataset_id"]
    counts = seed_frame.groupby(group, dropna=False).seed.agg(["count", "nunique"])
    if not ((counts["count"] == len(SEEDS)) & (counts["nunique"] == len(SEEDS))).all():
        raise ValueError("A paired estimand lacks exactly five distinct seeds")
    return seed_frame.groupby(group, dropna=False, as_index=False).value.mean()


def add_model_groups(dataset_frame):
    key = [c for c in GROUP if c != "model"] + ["dataset_id"]
    groups = []
    families = {"tfm_mean": ["tabpfn", "tabiclv2"], "boosting_mean": ["xgboost", "lightgbm"]}
    for name, models in families.items():
        x = dataset_frame[dataset_frame.model.isin(models)]
        if not (x.groupby(key, dropna=False).model.nunique() == len(models)).all():
            raise ValueError("Incomplete model family")
        y = x.groupby(key, dropna=False, as_index=False).value.mean()
        y["model"] = name
        groups.append(y)
    all_models = pd.concat([dataset_frame, *groups], ignore_index=True)
    for comparator in ["boosting_mean", "random_forest"]:
        a = all_models[all_models.model == "tfm_mean"].drop(columns="model")
        b = all_models[all_models.model == comparator].drop(columns="model")
        joined = a.merge(b, on=key, suffixes=("_tfm", "_comparator"), validate="one_to_one")
        if len(joined) != len(a): raise ValueError("Incomplete paired model-family contrast")
        joined["value"] = joined.pop("value_tfm")-joined.pop("value_comparator")
        joined["model"] = "tfm_minus_" + comparator
        groups.append(joined)
    return pd.concat([dataset_frame, *groups], ignore_index=True)


def summarize(dataset_frame):
    rows = []
    for keys, frame in dataset_frame.groupby(GROUP, sort=True, dropna=False):
        ordered = frame.sort_values("dataset_id")
        if ordered.dataset_id.duplicated().any(): raise ValueError("Dataset counted more than once")
        result = {**dict(zip(GROUP, keys)), **bootstrap(ordered.value.to_numpy(), ("strengthening-v1", keys))}
        result["min_dataset_value"] = float(ordered.value.min())
        result["max_dataset_value"] = float(ordered.value.max())
        rows.append(result)
    return pd.DataFrame(rows)
