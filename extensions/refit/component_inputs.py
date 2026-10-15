"""Frozen corruption arithmetic with caller-supplied input locations."""
from __future__ import annotations
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import numpy as np
from tfm_dirty_values.config import load_config
from tfm_dirty_values.injection.corruptions import (
    _stable_seed, _eligible_mask, _exact_mask, _numeric_statistics,
    _category_statistics, _corrupt_matrix)
PROJECT=Path(__file__).resolve().parents[2]
MODELS=["tabpfn","tabiclv2","xgboost","lightgbm","random_forest"]
SEEDS=[1907,2027,4049,6029,8089]
METRICS=["roc_auc","balanced_accuracy","log_loss","ece_10_equal_width"]
NUMERIC=[31,44,802,1049,1487,1504,42193,44158,44160]
CATEGORICAL=[3,24,31,802,42193,44158,44160]

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()

@lru_cache(maxsize=1)
def config():
    result=load_config(PROJECT/"configs/full_study.yaml")
    if os.environ.get("TFM_MODEL_ROOT"):
        for model in ["tabpfn","tabiclv2"]:
            parts=Path(result["models"][model]["checkpoint"]).parts
            relative=Path(*parts[parts.index("models")+1:])
            result["models"][model]["checkpoint"]=str(Path(os.environ["TFM_MODEL_ROOT"])/relative)
    return result

@lru_cache(maxsize=12)
def load_arrays(dataset,seed):
    path=Path(os.environ["TFM_SPLITS_ROOT"])/str(dataset)/f"split-{seed}.npz"
    manifest=json.loads((PROJECT/"data/full-study-processed/manifest.json").read_text(encoding="utf-8"))
    expected=next(r["array_sha256"] for r in manifest["split_files"]
                  if r["data_id"]==dataset and r["seed"]==seed)
    if hashlib.sha256(path.read_bytes()).hexdigest().lower()!=expected.lower():
        raise ValueError("The provided split differs from the frozen input")
    with np.load(path,allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    for name in ["X_train","X_test"]:
        arrays[name]=np.asarray(arrays[name],dtype=np.float64)
        arrays[name].setflags(write=False)
    return arrays

def base_params(mechanism):
    # Archived parameters were serialized in key-sorted order, which affects RNG.
    return json.loads(json.dumps(config()["corruptions"]["parameters"][mechanism], sort_keys=True))


def parameters(mechanism, level):
    p = base_params(mechanism)
    if mechanism == "measurement_noise": p["sigma_ratio"] = float(level)
    elif mechanism == "outlier": p["k"] = float(level)
    elif mechanism == "quantization": p["bins"] = int(level)
    elif mechanism == "unit_error": p["factors"] = [float(level), 1. / float(level)]
    elif mechanism == "categorical_error": p["novel_level_fraction"] = float(level)
    return p


def coupled_category(matrix, cats, stats, mask, rng, fraction, secondary_seed):
    """Keep the original p=.5 random stream, plus independent alternate swaps.

    At p=.5 output is bit-identical to the archive. Other p values threshold the
    same uniforms; unused swap draws come from a separate deterministic stream.
    """
    output = np.asarray(matrix, dtype=np.float64).copy()
    fallback = np.random.default_rng(secondary_seed)
    rows, columns = np.nonzero(mask)
    for row, col in zip(rows, columns, strict=True):
        levels = stats[int(col)]
        alternatives = levels[levels != matrix[row, col]]
        u = rng.random()
        if u >= .5 and len(alternatives):
            alternative = float(rng.choice(alternatives))
        elif len(alternatives):
            alternative = float(fallback.choice(alternatives))
        else:
            alternative = float(np.max(levels) + 1.) if len(levels) else 0.
        novel = float(np.max(levels) + 1.) if len(levels) else 0.
        output[row, col] = novel if u < fraction or not len(alternatives) else alternative
    return output


def partition_corruption(matrix, mechanism, level, rate, dataset, split_seed, partition, cats, numeric_stats, category_stats, pattern="iid"):
    eligible = _eligible_mask(matrix, categorical_indices=cats, categorical=mechanism == "categorical_error")
    seed = _stable_seed(partition, dataset, split_seed, mechanism, rate, base_params(mechanism))
    rng = np.random.default_rng(seed)
    if pattern == "row_batch":
        # Synthetic cohort, not a temporal/production batch. Random row selection
        # is label independent; one multiplier is shared across its numeric cells.
        rng = np.random.default_rng(_stable_seed("batch-v1", partition, dataset, split_seed, rate))
        selected_rows = rng.choice(matrix.shape[0], size=int(round(rate * matrix.shape[0])), replace=False)
        mask = np.zeros_like(eligible)
        mask[selected_rows] = eligible[selected_rows]
        dirty = matrix.copy()
        dirty[mask] *= float(level)
    elif mechanism == "categorical_error":
        mask = _exact_mask(eligible, rate=rate, rng=rng)
        dirty = coupled_category(matrix, cats, category_stats, mask, rng, float(level),
                                 _stable_seed("category-unused-swap-v1", seed))
    else:
        mask = _exact_mask(eligible, rate=rate, rng=rng)
        dirty, _ = _corrupt_matrix(matrix, corruption=mechanism, rate=rate,
            categorical_indices=cats, params=parameters(mechanism, level),
            numeric_stats=numeric_stats, category_stats=category_stats, seed=seed)
    return dirty, mask


def describe_partition(clean, dirty, mask, cats, mechanism):
    eligible = _eligible_mask(clean, categorical_indices=cats, categorical=mechanism == "categorical_error")
    changed = np.isfinite(clean) & np.isfinite(dirty) & (dirty != clean)
    return dict(eligible_observed_cells=int(eligible.sum()), selected_cells=int(mask.sum()),
                changed_cells=int(changed.sum()), total_rows=len(clean),
                selected_rows=int(mask.any(axis=1).sum()), changed_rows=int(changed.any(axis=1).sum()),
                selected_cell_fraction=float(mask.sum()/eligible.sum()) if eligible.any() else 0.,
                changed_row_fraction=float(changed.any(axis=1).mean()), mask_sha256=array_sha(mask))


@lru_cache(maxsize=12)
def dirty_arrays(dataset, seed, mechanism, level, rate, placement, pattern):
    arrays = load_arrays(dataset, seed)
    cats = arrays["categorical_indices"].astype(int).tolist()
    clean = [arrays["X_train"], arrays["X_test"]]
    numeric_stats = _numeric_statistics(clean[0], cats)
    category_stats = _category_statistics(clean[0], cats)
    result = []
    descriptions = {}
    for partition, matrix in zip(["context", "query"], clean, strict=True):
        if mechanism == "clean" or placement not in ("joint", partition):
            dirty, mask = matrix.copy(), np.zeros(matrix.shape, dtype=bool)
        else:
            dirty, mask = partition_corruption(matrix, mechanism, level, rate, dataset, seed,
                partition, cats, numeric_stats, category_stats, pattern)
        assert np.array_equal(np.isfinite(matrix), np.isfinite(dirty)), "Missingness changed"
        if mechanism != "categorical_error":
            assert np.array_equal(matrix[:, cats], dirty[:, cats], equal_nan=True)
        descriptions[partition] = describe_partition(matrix, dirty, mask, cats, mechanism)
        result.extend([dirty, mask])
    dt, mt, dq, mq = result
    return dt, dq, mt, mq, descriptions
