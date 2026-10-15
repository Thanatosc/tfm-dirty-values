"""Design and context-only encodings for the third exploratory extension."""
from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"src"))
import component_inputs as previous
from tfm_dirty_values.metrics import classification_metrics

MODELS = previous.MODELS
TREES = ["xgboost", "lightgbm", "random_forest"]
METRICS = previous.METRICS
SEEDS = previous.SEEDS
PANELS = {"outlier": previous.NUMERIC, "category": previous.CATEGORICAL}
RATES = {"outlier": [0.05, 0.50], "category": [0.15, 0.50]}
VARIANTS = {
    "outlier": {"tabpfn": ["default", "outlier_off"],
        "tabiclv2": ["default", "outlier_off"],
        "xgboost": ["default", "exact"],
        "lightgbm": ["default", "fine_bins"],
        "random_forest": ["default", "leaf5"]},
    "category": {"xgboost": ["legacy", "ordinal", "onehot", "native"],
        "lightgbm": ["legacy", "ordinal", "onehot", "native"],
        "random_forest": ["legacy", "ordinal", "onehot", "ordinal_full", "onehot_full"]}}
def utc():
    return datetime.now(timezone.utc).isoformat()


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def identity(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return previous.array_sha(np.asarray(a))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.05 * (attempt+1), 0.5))


def build_jobs():
    jobs = []
    for experiment in ["outlier", "category"]:
        for dataset in PANELS[experiment]:
            for seed in SEEDS:
                for model, variants in VARIANTS[experiment].items():
                    for variant in variants:
                        for context_rate in [0.0] + RATES[experiment]:
                            jobs.append(dict(experiment=experiment, dataset_id=dataset, seed=seed,
                                model=model, variant=variant, context_rate=context_rate,
                                shard="gpu" if model in ["tabpfn", "tabiclv2"] else "cpu"))
    assert len(jobs) == 2715
    assert sum(j["shard"] == "gpu" for j in jobs) == 540
    return jobs


def job_id(job):
    return identity(job)[:24]


def query_rates(job):
    return [0.0] + (RATES[job["experiment"]] if job["context_rate"] == 0.0 else [job["context_rate"]])


def view_key(rate):
    return "clean" if rate == 0 else f"dirty_{round(rate*100):02d}"


def placement(context_rate, query_rate):
    if context_rate == 0 and query_rate == 0:
        return "clean"
    if context_rate == 0:
        return "query"
    if query_rate == 0:
        return "context"
    return "joint"


@lru_cache(maxsize=6)
def condition_arrays(experiment, dataset, seed, rate):
    if rate == 0:
        a = previous.load_arrays(dataset, seed)
        x, q = a["X_train"], a["X_test"]
        return x, q, np.zeros_like(x, dtype=bool), np.zeros_like(q, dtype=bool)
    corruption, level = ("outlier", 10.0) if experiment == "outlier" else ("categorical_error", 0.5)
    x, q, mx, mq, _ = previous.dirty_arrays(dataset, seed, corruption, level, rate, "joint", "iid")
    for a in [x, q, mx, mq]:
        a.setflags(write=False)
    return x, q, mx, mq


def inputs(job):
    a = previous.load_arrays(job["dataset_id"], job["seed"])
    cats = a["categorical_indices"].astype(int).tolist()
    x, _, mx, _ = condition_arrays(job["experiment"], job["dataset_id"], job["seed"], job["context_rate"])
    views = {}
    for rate in query_rates(job):
        _, q, _, mq = condition_arrays(job["experiment"], job["dataset_id"], job["seed"], rate)
        changed = np.isfinite(q) & np.isfinite(a["X_test"]) & (q != a["X_test"])
        views[view_key(rate)] = dict(rate=rate, X=q, mask=mq, changed_rows=changed.any(axis=1))
    return a, cats, x, mx, views


class ContextEncoding:
    """Fit only the supplied context, with one common missing/unseen policy.

    Matched ordinal/native/onehot arms map finite context levels to 0..K-1.
    Missing and query levels absent from that vocabulary both map to NaN.
    One-hot represents these by all-zero category blocks. The legacy bridge
    preserves the original finite codes (including query -1 and novel codes).
    """
    def __init__(self, cats, variant):
        self.cats = list(cats)
        self.variant = variant

    def fit(self, x):
        self.n_features = x.shape[1]
        self.levels = {c: np.unique(x[np.isfinite(x[:, c]), c]) for c in self.cats}
        self.is_onehot = self.variant in ["onehot", "onehot_full"]
        self.numeric = [c for c in range(self.n_features) if c not in self.cats]
        self.output_features = (len(self.numeric)+sum(len(v) for v in self.levels.values())
            if self.is_onehot else self.n_features)
        self.metadata = dict(variant=self.variant, source_features=self.n_features,
            output_features=self.output_features, vocabulary_sha256=identity({str(c):v.tolist() for c,v in self.levels.items()}),
            vocabulary_sizes={str(c):len(v) for c,v in self.levels.items()},
            vocabulary_fit="supplied_context_only", unknown_policy="legacy_preserved" if self.variant=="legacy" else "missing_and_unseen_to_NaN; onehot_all_zero")
        return self

    def transform(self, matrix):
        x = np.asarray(matrix, dtype=np.float64).copy()
        if self.variant == "legacy":
            return x, dict(unseen_cells=None, missing_cells=int(np.isnan(x[:, self.cats]).sum()))
        unseen = 0
        missing = int(np.isnan(x[:, self.cats]).sum())
        codes = {}
        for c, levels in self.levels.items():
            v = x[:, c]
            finite = np.isfinite(v)
            pos = np.searchsorted(levels, np.where(finite, v, np.inf))
            valid = finite & (pos < len(levels))
            if len(levels):
                valid &= levels[np.minimum(pos, len(levels)-1)] == v
            else:
                valid[:] = False
            unseen += int((finite & ~valid).sum())
            z = np.full(len(v), np.nan)
            z[valid] = pos[valid]
            x[:, c] = z
            codes[c] = z
        if self.is_onehot:
            result = np.zeros((len(x), self.output_features), dtype=np.float64)
            result[:, :len(self.numeric)] = x[:, self.numeric]
            offset = len(self.numeric)
            for c in self.cats:
                values = codes[c]
                observed = np.isfinite(values)
                result[np.flatnonzero(observed), offset+values[observed].astype(int)] = 1.0
                offset += len(self.levels[c])
            x = result
        return x, dict(unseen_cells=unseen, missing_cells=missing)


def encode_job(job, x, cats, views):
    if job["experiment"] == "outlier":
        return x.copy(), {k:{**v,"X":v["X"].copy()} for k,v in views.items()}, dict(variant="unchanged", output_features=x.shape[1])
    encoder = ContextEncoding(cats, job["variant"]).fit(x)
    tx, train_info = encoder.transform(x)
    result = {}
    for k,v in views.items():
        q, info = encoder.transform(v["X"])
        result[k] = {**v, "X":q, "encoding":info}
    return tx, result, {**encoder.metadata, "context":train_info}


def mean_or_none(x):
    x = np.asarray(x)
    return float(np.mean(x)) if x.size else None
