# Functions extracted unchanged from the recorded study sources.
import hashlib
import json
import numpy as np

MODELS = ["tabpfn", "tabiclv2", "xgboost", "lightgbm", "random_forest"]
SEEDS = [1907, 2027, 4049, 6029, 8089]
METRICS = ["roc_auc", "balanced_accuracy", "log_loss", "ece_10_equal_width"]
NUMERIC = [31, 44, 802, 1049, 1487, 1504, 42193, 44158, 44160]
EXTERNAL = [735, 752, 833, 1479, 41156]
LEVELS = {"measurement_noise": [.1, .25, .5], "outlier": [3., 5., 10.],
          "quantization": [4., 8., 16.], "unit_error": [10., 100., 1000.],
          "categorical_error": [0., .5, 1.]}
def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def task_id(task):
    return identity(task)[:24]


def make_task(dataset, seed, model, mechanism="clean", level=0., rate=0.,
              placement="none", representation="raw", pattern="iid"):
    return dict(dataset_id=int(dataset), seed=int(seed), model=model,
                corruption=mechanism, level=float(level), rate=float(rate),
                placement=placement, representation=representation, pattern=pattern,
                cohort="external" if dataset in EXTERNAL else "original")
def bootstrap(values, key):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError(f"Invalid dataset estimates: {key}")
    seed = int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0,len(x),size=(10000,len(x)))].mean(axis=1)
    lo, hi = np.quantile(means,[.025,.975])
    return dict(n_datasets=len(x), mean=float(x.mean()), ci_low=float(lo), ci_high=float(hi),
        positive_datasets=int((x>0).sum()), negative_datasets=int((x<0).sum()))


def condition(r):
    return int(r["dataset_id"]), int(r["seed"]), str(r["corruption"]), float(r["rate"])
