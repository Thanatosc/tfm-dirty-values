"""Failure-preserving GPU prediction functions from amendment A1."""
from __future__ import annotations
import gc
import time
import numpy as np
import worker as original
from core import (HERE,METRICS,read,sha,identity,array_sha,job_id,inputs,
    encode_job,placement,classification_metrics,mean_or_none,utc)
from model_controls import FittedPipeline
AMENDMENT_ID="f27d063eb7d8ba537bd2e64fae650567fe31eb43d080eb78755f2490f3799329"

def probability_quality(raw):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 2 or not len(raw):
        raise ValueError("Unexpected binary probability array shape")
    invalid = (~np.isfinite(raw)) | (raw < 0) | (raw > 1)
    finite = raw[np.isfinite(raw)]
    return dict(status="numerical_failure" if invalid.any() else "valid",
                invalid_rows=int(invalid.any(axis=1).sum()),
                total_rows=len(raw), nonfinite_values=int((~np.isfinite(raw)).sum()),
                out_of_range_values=int(((raw < 0) | (raw > 1)).sum()),
                dtype=str(raw.dtype), finite_min=float(finite.min()) if finite.size else None,
                finite_max=float(finite.max()) if finite.size else None)


def score_or_undefined(y, raw):
    quality = probability_quality(raw)
    p = np.asarray(raw)[:, 1].astype(np.float64)
    scored = (classification_metrics(y, p) if quality["status"] == "valid"
              else {metric: None for metric in METRICS})
    return p, scored, quality


def execute_a1(job, study_identity, output_dir):
    if job["shard"] != "gpu":
        raise ValueError("A1 runner is limited to GPU jobs")
    started = time.monotonic()
    a, cats, x, mx, views = inputs(job)
    tx, tviews, encoding = encode_job(job, x, cats, views)
    input_hash = array_sha(x)
    pipe = None
    try:
        pipe = FittedPipeline(job, cats, tx, a["y_train"])
        predictions = {"y_query": a["y_test"].astype(np.int64)}
        view_records = {}
        if job["model"] == "tabiclv2":
            pipe.metadata["resolved_amp_fa3"] = list(pipe.model._resolve_amp_fa3())
        for key, view in tviews.items():
            pipe.stage = key
            raw = np.asarray(pipe.model.predict_proba(pipe.prepare_query(view["X"])))
            p, scored, quality = score_or_undefined(a["y_test"], raw)
            predictions["raw_"+key] = raw
            predictions["p_"+key] = p
            predictions["changed_"+key] = view["changed_rows"].astype(np.uint8)
            stats = dict(rate=view["rate"], placement=placement(job["context_rate"], view["rate"]),
                metrics=scored, prediction_quality=quality,
                probability_sha256=array_sha(p), raw_probability_sha256=array_sha(raw),
                original_query_sha256=array_sha(views[key]["X"]), model_query_sha256=array_sha(view["X"]),
                changed_rows=int(view["changed_rows"].sum()), total_rows=len(p), encoding=view.get("encoding"))
            if key != "clean":
                complete = quality["status"] == "valid" and view_records["clean"]["prediction_quality"]["status"] == "valid"
                if complete:
                    change = np.abs(p-predictions["p_clean"])
                    stats["probability_displacement"] = dict(mean_absolute=float(change.mean()),
                        corrupted_rows_mean=mean_or_none(change[view["changed_rows"]]),
                        unchanged_rows_mean=mean_or_none(change[~view["changed_rows"]]))
                else:
                    stats["probability_displacement"] = dict(mean_absolute=None, corrupted_rows_mean=None,
                        unchanged_rows_mean=None)
            if quality["status"] == "valid":
                check = original.archive_check(job, stats, input_hash, stats["original_query_sha256"])
                if check is not None:
                    stats["archival_default_check"] = check
            view_records[key] = stats
        if job["model"] == "tabiclv2" and not any(t["component"] == "soft_outlier" for t in pipe.trace):
            raise ValueError("TabICL observational hook did not execute")
        pred_path = output_dir / (job_id(job)+".npz")
        original.save_predictions(pred_path, predictions)
        record = dict(identity=study_identity, amendment_identity=AMENDMENT_ID,
            job_id=job_id(job), job=job, original_context_sha256=input_hash, model_context_sha256=array_sha(tx),
            context_mask_sha256=array_sha(mx), encoding=encoding, model_metadata=pipe.metadata,
            component_trace=pipe.trace, tree_diagnostics=None, views=view_records,
            predictions_file=pred_path.name, predictions_sha256=sha(pred_path),
            seconds=time.monotonic()-started, finished_at_utc=utc(), fresh_fit=True)
        record["record_sha256"] = identity(record)
        return record
    finally:
        if pipe is not None:
            pipe.close()
        del pipe
        gc.collect()
        import torch
        torch.cuda.empty_cache()
