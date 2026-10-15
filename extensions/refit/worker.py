"""Frozen fit/predict functions; archival reuse and process scheduling omitted."""
from __future__ import annotations
import gc
import os
import time
import numpy as np
from core import (HERE,previous,read,atomic,utc,sha,identity,array_sha,
    job_id,inputs,encode_job,placement,classification_metrics,mean_or_none)
from model_controls import FittedPipeline,routing_summary

def archive_check(*args):
    return None

def save_predictions(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f".{os.getpid()}.tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(temp, path)


def execute(job, study_identity, output_dir):
    started = time.monotonic()
    a,cats,x,mx,views = inputs(job)
    input_hash = array_sha(x)
    tx,tviews,encoding = encode_job(job,x,cats,views)
    pipe = None
    try:
        pipe = FittedPipeline(job,cats,tx,a["y_train"])
        predictions = {"y_query":a["y_test"].astype(np.int64)}
        view_records = {}
        clean_leaves = None
        tree_diag = None
        if job["experiment"]=="outlier" and job["shard"]=="cpu":
            tree_diag = pipe.tree_diagnostics(a["X_train"])
        for key,v in tviews.items():
            p = pipe.predict(v["X"],key)
            predictions["p_"+key] = p
            predictions["changed_"+key] = v["changed_rows"].astype(np.uint8)
            stats = dict(rate=v["rate"], placement=placement(job["context_rate"],v["rate"]),
                metrics=classification_metrics(a["y_test"],p), probability_sha256=array_sha(p),
                original_query_sha256=array_sha(views[key]["X"]), model_query_sha256=array_sha(v["X"]),
                changed_rows=int(v["changed_rows"].sum()), total_rows=len(p),
                encoding=v.get("encoding"))
            if key!="clean":
                change = np.abs(p-predictions["p_clean"])
                stats["probability_displacement"] = dict(mean_absolute=float(change.mean()),
                    corrupted_rows_mean=mean_or_none(change[v["changed_rows"]]),
                    unchanged_rows_mean=mean_or_none(change[~v["changed_rows"]]))
            if tree_diag is not None:
                leaves = pipe.leaves(v["X"])
                if key=="clean":
                    clean_leaves = leaves
                else:
                    routing,per_row = routing_summary(clean_leaves,leaves,v["changed_rows"])
                    stats["routing"] = routing
                    predictions["leaf_switch_"+key] = per_row
            check = archive_check(job,stats,input_hash,stats["original_query_sha256"])
            if check is not None:
                stats["archival_default_check"] = check
            view_records[key] = stats
        if job["model"]=="tabiclv2" and not any(t["component"]=="soft_outlier" for t in pipe.trace):
            raise ValueError("TabICL observational hook did not execute")
        pred_path = output_dir / (job_id(job)+".npz")
        save_predictions(pred_path,predictions)
        record = dict(identity=study_identity,job_id=job_id(job),job=job,
            original_context_sha256=input_hash,model_context_sha256=array_sha(tx),
            context_mask_sha256=array_sha(mx), encoding=encoding,
            model_metadata=pipe.metadata, component_trace=pipe.trace, tree_diagnostics=tree_diag,
            views=view_records,predictions_file=pred_path.name,predictions_sha256=sha(pred_path),
            seconds=time.monotonic()-started,finished_at_utc=utc(),fresh_fit=True)
        record["record_sha256"] = identity(record)
        return record
    finally:
        if pipe is not None:
            pipe.close()
        del pipe
        gc.collect()
        if job["shard"]=="gpu":
            import torch
            torch.cuda.empty_cache()
