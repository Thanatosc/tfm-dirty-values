"""Portable, complete-only reanalysis of saved row predictions.

Dependencies: numpy, pandas, scikit-learn. Does not import experimental model
packages, read original raw inputs, or run a prediction. Stored probabilities
are the independently checked boundary of this reconstruction.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
METRICS = ["roc_auc","balanced_accuracy","log_loss","ece_10_equal_width"]
SEEDS = {1907,2027,4049,6029,8089}
EFFECT_KEYS = ["experiment","model","variant","baseline","rate","placement","comparison","estimand","metric"]
DIAG_KEYS = ["experiment","model","variant","context_rate","query_rate","diagnostic"]

def canonical(obj):
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
def identity(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):
            h.update(chunk)
    return h.hexdigest()
def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()
def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")

def metrics(y,p):
    """Preserve the original metric definition, including probability-bin ECE."""
    y,p=np.asarray(y,dtype=np.int64),np.asarray(p,dtype=np.float64)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Invalid probabilities")
    predictions=(p>=.5).astype(np.int64)
    bin_ids=np.clip(np.digitize(p,np.linspace(0,1,11)[1:-1],right=False),0,9)
    confidence=np.maximum(p,1-p)
    ece=0.0
    for b in range(10):
        selected=bin_ids==b
        if selected.any():
            ece+=selected.sum()/len(y)*abs(float(np.mean(y[selected]==predictions[selected])-np.mean(confidence[selected])))
    return dict(roc_auc=float(roc_auc_score(y,p)),balanced_accuracy=float(balanced_accuracy_score(y,predictions)),
        log_loss=float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])),ece_10_equal_width=float(ece))

def oriented(value,metric):
    if value is None:
        return np.nan
    return -float(value) if metric in ["log_loss","ece_10_equal_width"] else float(value)

def bootstrap(values,key):
    values=np.asarray(values,dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite dataset values")
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(canonical(list(key)).encode()).digest()[:8],"little"))
    draws=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    low,high=np.quantile(draws,[.025,.975])
    return dict(n_datasets=len(values),mean=float(values.mean()),ci_low=float(low),ci_high=float(high),
        positive_datasets=int((values>0).sum()),negative_datasets=int((values<0).sum()),
        minimum=float(values.min()),maximum=float(values.max()))

def require_seed_group(group):
    if len(group)!=5 or set(group["seed"])!=SEEDS:
        raise ValueError("Missing or duplicated paired seeds")

def aggregate_effects(frame):
    dataset_rows=[]
    for key,group in frame.groupby(EFFECT_KEYS+["dataset_id"],dropna=False,sort=True):
        require_seed_group(group)
        valid=np.isfinite(group["value"])
        dataset_rows.append({**dict(zip(EFFECT_KEYS+["dataset_id"],key)),"n_seeds":5,
            "n_valid_seeds":int(valid.sum()),
            "value":float(group["value"].mean()) if valid.all() else np.nan})
    dataset=pd.DataFrame(dataset_rows)
    summary=[]
    for key,group in dataset.groupby(EFFECT_KEYS,dropna=False,sort=True):
        expected=9 if key[0]=="outlier" else 7
        if len(group)!=expected:
            raise ValueError(f"Incomplete dataset panel in {key}")
        ordered=group.sort_values("dataset_id")
        good=ordered[np.isfinite(ordered["value"])]
        complete=len(good)==expected
        stats=(bootstrap(ordered["value"].to_numpy(),key) if complete else
            dict(n_datasets=expected,mean=np.nan,ci_low=np.nan,ci_high=np.nan,
                 positive_datasets=np.nan,negative_datasets=np.nan,minimum=np.nan,maximum=np.nan))
        conditional=(bootstrap(good["value"].to_numpy(),tuple(key)+("available_complete_datasets",)) if len(good) else
            dict(n_datasets=0,mean=np.nan,ci_low=np.nan,ci_high=np.nan,
                 positive_datasets=0,negative_datasets=0,minimum=np.nan,maximum=np.nan))
        summary.append({**dict(zip(EFFECT_KEYS,key)),**stats,
            "full_panel_defined":complete,"n_valid_datasets":len(good),
            "undefined_dataset_ids":";".join(str(int(v)) for v in ordered.loc[~np.isfinite(ordered["value"]),"dataset_id"]),
            **{"available_complete_"+k:v for k,v in conditional.items()}})
    return dataset,pd.DataFrame(summary)

def build_effects(rows):
    """All four placement cells are selected at matching rates before pairing."""
    lookup={}
    combos={}
    for row in rows:
        key=tuple(row[k] for k in ["experiment","dataset_id","seed","model","variant","context_rate","query_rate"])
        if key in lookup:
            raise ValueError("Duplicate prediction view")
        lookup[key]=row
        group=key[:4]
        combos.setdefault(group,set()).add(row["variant"])
    result=[]
    for group,variants in sorted(combos.items()):
        experiment,dataset,seed,model=group
        rates=[.05,.5] if experiment=="outlier" else [.15,.5]
        def score(variant,cr,qr,metric):
            return oriented(lookup[group+(variant,cr,qr)][metric],metric)
        def add(variant,baseline,rate,place,comparison,estimand,metric,value):
            result.append(dict(experiment=experiment,dataset_id=dataset,seed=seed,model=model,
                variant=variant,baseline=baseline,rate=rate,placement=place,comparison=comparison,
                estimand=estimand,metric=metric,value=float(value)))
        for variant in sorted(variants):
            for rate in rates:
                for metric in METRICS:
                    clean=score(variant,0.,0.,metric)
                    losses={p:clean-score(variant,cr,qr,metric) for p,cr,qr in
                        [("context",rate,0.),("query",0.,rate),("joint",rate,rate)]}
                    for place,loss in losses.items():
                        add(variant,"none",rate,place,"clean_to_dirty","degradation",metric,loss)
                    add(variant,"none",rate,"joint","placement_interaction","interaction",metric,
                        losses["joint"]-losses["context"]-losses["query"])
        comparisons=[]
        if experiment=="outlier":
            comparisons=[(v,"default") for v in variants if v!="default"]
        else:
            comparisons=[("ordinal","legacy"),("onehot","ordinal")]
            if "native" in variants:
                comparisons.append(("native","ordinal"))
            if "onehot_full" in variants:
                comparisons.append(("onehot_full","ordinal_full"))
        for variant,base in comparisons:
            for rate in rates:
                for metric in METRICS:
                    clean_gain=score(variant,0,0,metric)-score(base,0,0,metric)
                    for place,cr,qr in [("context",rate,0.),("query",0.,rate),("joint",rate,rate)]:
                        dirty_gain=score(variant,cr,qr,metric)-score(base,cr,qr,metric)
                        for estimand,value in [("clean_gain",clean_gain),("dirty_gain",dirty_gain),("degradation_reduction",dirty_gain-clean_gain)]:
                            add(variant,base,rate,place,variant+"_vs_"+base,estimand,metric,value)
        if experiment=="category" and model=="random_forest":
            for rate in rates:
                for metric in METRICS:
                    def difference(cr,qr):
                        return ((score("onehot_full",cr,qr,metric)-score("ordinal_full",cr,qr,metric))-
                            (score("onehot",cr,qr,metric)-score("ordinal",cr,qr,metric)))
                    c=difference(0,0)
                    for place,cr,qr in [("context",rate,0.),("query",0.,rate),("joint",rate,rate)]:
                        d=difference(cr,qr)
                        for estimand,value in [("clean_gain",c),("dirty_gain",d),("degradation_reduction",d-c)]:
                            add("onehot_full","ordinal_full",rate,place,"rf_encoding_effect_full_minus_sqrt",estimand,metric,value)
    return pd.DataFrame(result)

def diagnostic_summaries(rows):
    frame=pd.DataFrame(rows)
    dataset_rows=[]
    for key,group in frame.groupby(DIAG_KEYS+["dataset_id"],dropna=False,sort=True):
        require_seed_group(group)
        good=group["value"].notna()
        dataset_rows.append({**dict(zip(DIAG_KEYS+["dataset_id"],key)),"n_valid_seeds":int(good.sum()),
            "value":float(group.loc[good,"value"].mean()) if good.any() else np.nan})
    dataset=pd.DataFrame(dataset_rows)
    summary=[]
    for key,group in dataset.groupby(DIAG_KEYS,dropna=False,sort=True):
        valid=group.dropna(subset=["value"]).sort_values("dataset_id")
        summary.append({**dict(zip(DIAG_KEYS,key)),"n_datasets_total":len(group),
            **(bootstrap(valid["value"].to_numpy(),key) if len(valid) else dict(n_datasets=0,mean=np.nan,ci_low=np.nan,ci_high=np.nan))})
    return frame,dataset,pd.DataFrame(summary)

def load_study_metadata(root):
    """Verify a local freeze or an explicitly path-sanitized public projection."""
    root=Path(root).resolve()
    snapshot_path=root/"public_snapshot.json"
    if snapshot_path.exists():
        snapshot=read(snapshot_path)
        if identity({k:v for k,v in snapshot.items() if k!="snapshot_identity"})!=snapshot["snapshot_identity"]:
            raise ValueError("Public snapshot identity mismatch")
        for relative,expected in snapshot["files"].items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root) or not path.is_file() or sha(path)!=expected:
                raise ValueError(f"Public file mismatch: {relative}")
        manifest=dict(identity=snapshot["original_study_identity"],expected_jobs=snapshot["expected_jobs"],
            expected_predictive_views=snapshot["expected_predictive_views"],
            frozen_files={"jobs.json":snapshot["files"]["jobs.json"]})
        amendment=dict(identity=snapshot["amendment_identity"],original_identity=manifest["identity"])
        return manifest,amendment
    manifest=read(root/"frozen_manifest.json")
    if identity({k:v for k,v in manifest.items() if k!="identity"})!=manifest["identity"]:
        raise ValueError("Freeze identity mismatch")
    amendment=read(root/"amendment_A1.json") if (root/"amendment_A1.json").exists() else None
    if amendment is not None:
        if (amendment["original_identity"]!=manifest["identity"] or
            identity({k:v for k,v in amendment.items() if k!="identity"})!=amendment["identity"]):
            raise ValueError("A1 manifest mismatch")
        for name,expected in amendment["files"].items():
            if sha(root/name)!=expected:
                raise ValueError(f"A1 source mismatch: {name}")
    return manifest,amendment


def analyze(root,output):
    root,output=Path(root),Path(output)
    manifest,amendment=load_study_metadata(root)
    job_hashes=[v for k,v in manifest["frozen_files"].items() if k.replace("\\","/").rsplit("/",1)[-1]=="jobs.json"]
    if len(job_hashes)!=1 or sha(root/"jobs.json")!=job_hashes[0]:
        raise ValueError("Job list differs from the frozen artifact")
    jobs=read(root/"jobs.json")
    if len(jobs)!=manifest["expected_jobs"] or len({identity(j) for j in jobs})!=len(jobs):
        raise ValueError("Job manifest incomplete or duplicated")
    expected_paths={root/"outputs"/j["shard"]/(identity(j)[:24]+".json") for j in jobs}
    missing=[p for p in expected_paths if not p.exists()]
    extra=set((root/"outputs").glob("*/*.json"))-expected_paths
    if missing or extra:
        raise ValueError(f"Incomplete experiment: {len(missing)} missing jobs; {len(extra)} extra")
    metrics_rows,diagnostics,encoding_rows,archive_rows,hashes=[],[],[],[],[]
    max_difference=0.
    for job in jobs:
        jid=identity(job)[:24]
        path=root/"outputs"/job["shard"]/(jid+".json")
        record=read(path)
        if record["job"]!=job or record["job_id"]!=jid or record["identity"]!=manifest["identity"]:
            raise ValueError("Job result identity mismatch")
        if identity({k:v for k,v in record.items() if k!="record_sha256"})!=record["record_sha256"]:
            raise ValueError("Job record content changed")
        if "amendment_identity" in record and (amendment is None or record["amendment_identity"]!=amendment["identity"]):
            raise ValueError("Unrecognized result amendment")
        prediction_path=path.parent/record["predictions_file"]
        if sha(prediction_path)!=record["predictions_sha256"]:
            raise ValueError("Prediction artifact changed")
        hashes.append([jid,record["record_sha256"],record["predictions_sha256"]])
        expected_rates=[0.,*( ([.05,.5] if job["experiment"]=="outlier" else [.15,.5]) if job["context_rate"]==0 else [job["context_rate"]])]
        if sorted(v["rate"] for v in record["views"].values())!=sorted(expected_rates):
            raise ValueError("Missing query views")
        def diagnostic(name,value,query_rate=-1.):
            diagnostics.append({**{k:job[k] for k in ["experiment","dataset_id","seed","model","variant","context_rate"]},
                "query_rate":query_rate,"diagnostic":name,"value":np.nan if value is None else float(value)})
        with np.load(prediction_path,allow_pickle=False) as saved:
            for key,view in record["views"].items():
                p=saved["p_"+key]
                if array_sha(p)!=view["probability_sha256"]:
                    raise ValueError("Probability digest mismatch")
                status="valid"
                invalid_rows=0
                if "prediction_quality" in view:
                    if "amendment_identity" not in record:
                        raise ValueError("Quality amendment missing")
                    raw=saved["raw_"+key]
                    if raw.shape!=(len(p),2) or array_sha(raw)!=view["raw_probability_sha256"]:
                        raise ValueError("Raw probability digest/shape mismatch")
                    if not np.array_equal(p,raw[:,1].astype(np.float64),equal_nan=True):
                        raise ValueError("Stored positive-class probabilities were altered")
                    bad=(~np.isfinite(raw))|(raw<0)|(raw>1)
                    invalid_rows=int(bad.any(axis=1).sum())
                    status="numerical_failure" if invalid_rows else "valid"
                    q=view["prediction_quality"]
                    if (q["status"]!=status or q["invalid_rows"]!=invalid_rows or q["total_rows"]!=len(p) or
                        q["nonfinite_values"]!=int((~np.isfinite(raw)).sum()) or
                        q["out_of_range_values"]!=int(((raw<0)|(raw>1)).sum())):
                        raise ValueError("Recorded numerical failure differs from raw predictions")
                if status=="valid":
                    recomputed=metrics(saved["y_query"],p)
                    delta=max(abs(recomputed[m]-view["metrics"][m]) for m in METRICS)
                    max_difference=max(max_difference,delta)
                    if delta>1e-12:
                        raise ValueError(f"Metric reconstruction differs: {jid}, {key}, {delta}")
                else:
                    if any(view["metrics"][m] is not None for m in METRICS):
                        raise ValueError("Numerical failure has a fabricated metric")
                    recomputed={m:None for m in METRICS}
                changed=saved["changed_"+key].astype(bool)
                if len(p)!=len(saved["y_query"]) or changed.shape!=p.shape or int(changed.sum())!=view["changed_rows"]:
                    raise ValueError("Query labels/mask/coverage mismatch")
                metrics_rows.append({**job,"job_id":jid,"query_rate":view["rate"],"placement":view["placement"],
                    **recomputed,"query_rows":len(p),"changed_rows":view["changed_rows"],
                    "prediction_status":status,"invalid_query_rows":invalid_rows})
                if view.get("encoding"):
                    count=view["encoding"]["unseen_cells"]
                    ncat=len(record["encoding"].get("vocabulary_sizes",{}))
                    if count is not None and ncat:
                        diagnostic("unseen_fraction_of_category_slots",count/(len(p)*ncat),view["rate"])
                if "routing" in view:
                    rows=saved["leaf_switch_"+key]
                    if abs(float(rows.mean())-view["routing"]["mean_tree_switch_fraction"])>1e-12:
                        raise ValueError("Saved leaf-routing mean changed")
                    changed=saved["changed_"+key].astype(bool)
                    if np.any(rows[~changed]!=0):
                        raise ValueError("Unchanged query routed differently")
                    for name,value in view["routing"].items():
                        if name not in ["comparison"]:
                            diagnostic("routing_"+name,value,view["rate"])
                for name,value in view.get("probability_displacement",{}).items():
                    diagnostic("probability_"+name,value,view["rate"])
                check=view.get("archival_default_check")
                if check:
                    archive_rows.append({**job,"query_rate":view["rate"],"exact":check["exact"],**check["differences"]})
        if record.get("tree_diagnostics"):
            for name,value in record["tree_diagnostics"].items():
                if name!="bounds_source":
                    diagnostic("tree_"+name,value)
        trace_groups={}
        for t in record["component_trace"]:
            k=(t["stage"],t["component"])
            trace_groups.setdefault(k,[]).append(t)
        for (stage,component),values in trace_groups.items():
            cells=sum(t["cells"] for t in values)
            qr=-1. if stage=="fit_context" else (0. if stage=="clean" else int(stage.split("_")[1])/100)
            diagnostic("trace_"+component+"_changed_fraction",sum(t["changed_cells"] for t in values)/cells,qr)
            if component=="standardization":
                diagnostic("trace_hard_clip_fraction",sum(t["hard_clip_cells"] for t in values)/cells,qr)
        if job["experiment"]=="category":
            encoding_rows.append({**job,"source_features":record["encoding"]["source_features"],
                "output_features":record["encoding"]["output_features"],
                "realized_features_per_split":record["model_metadata"].get("realized_features_per_split"),
                "vocabulary_sha256":record["encoding"]["vocabulary_sha256"]})
    if len(metrics_rows)!=manifest["expected_predictive_views"]:
        raise ValueError("Predictive view count differs")
    effects=build_effects(metrics_rows)
    dataset,summary=aggregate_effects(effects)
    diagnostic_seeds,diagnostic_datasets,diagnostic_summary=diagnostic_summaries(diagnostics)
    prediction_frame=pd.DataFrame(metrics_rows)
    failed=prediction_frame[prediction_frame.prediction_status=="numerical_failure"].copy()
    failure_keys=["experiment","model","variant","context_rate","query_rate","placement"]
    failure_summary=[]
    for key,group in prediction_frame.groupby(failure_keys,dropna=False,sort=True):
        by_dataset=[]
        for _,sub in group.groupby("dataset_id"):
            require_seed_group(sub)
            by_dataset.append(float((sub.invalid_query_rows/sub.query_rows).mean()))
        failure_summary.append({**dict(zip(failure_keys,key)),"planned_views":len(group),
            "failed_views":int((group.prediction_status!="valid").sum()),
            "datasets_with_any_failure":int(group.loc[group.prediction_status!="valid","dataset_id"].nunique()),
            "n_datasets":int(group.dataset_id.nunique()),
            "invalid_query_rows":int(group.invalid_query_rows.sum()),"query_rows":int(group.query_rows.sum()),
            "dataset_mean_invalid_row_fraction":float(np.mean(by_dataset))})
    output.mkdir(parents=True,exist_ok=True)
    tables={"prediction_metrics.csv":prediction_frame,"paired_seed_effects.csv":effects,
        "paired_dataset_effects.csv":dataset,"summary.csv":summary,
        "diagnostic_seed_values.csv":diagnostic_seeds,"diagnostic_dataset_values.csv":diagnostic_datasets,
        "diagnostic_summary.csv":diagnostic_summary,"encoding_dimensions.csv":pd.DataFrame(encoding_rows),
        "archival_default_comparisons.csv":pd.DataFrame(archive_rows),
        "numerical_failure_views.csv":failed,"numerical_failure_summary.csv":pd.DataFrame(failure_summary)}
    for name,frame in tables.items():
        frame.to_csv(output/name,index=False,float_format="%.17g",lineterminator="\n")
    verification=dict(status="VERIFIED_FROM_PERSISTED_PREDICTIONS",study_identity=manifest["identity"],
        amendment_identity=None if amendment is None else amendment["identity"],
        fit_jobs=len(jobs),predictive_views=len(metrics_rows),valid_predictive_views=len(metrics_rows)-len(failed),
        numerical_failure_views=len(failed),recomputed_metric_values=4*(len(metrics_rows)-len(failed)),
        undefined_metric_values=4*len(failed),
        maximum_metric_reconstruction_difference=max_difference,
        output_content_identity=identity(sorted(hashes)),analysis_script_sha256=sha(__file__),
        independent_unit="dataset; five paired seeds averaged before aggregation",
        expected_panels={"outlier":9,"category":7},confirmatory_tests=0,bootstrap_resamples=10000,
        archival_checks=len(archive_rows),archival_exact=sum(r["exact"] for r in archive_rows),
        tables={name:{"rows":len(frame),"sha256":sha(output/name)} for name,frame in tables.items()},
        missingness_policy="No row deletion or score imputation; five valid paired seeds required per dataset; full-panel means undefined if incomplete",
        boundary="Exact reconstruction from saved row probabilities, not an independent model rerun")
    write_json(output/"verification.json",verification)
    print(json.dumps({k:verification[k] for k in ["status","fit_jobs","predictive_views","recomputed_metric_values","maximum_metric_reconstruction_difference"]}),flush=True)
    return verification

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=HERE)
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    analyze(a.root,a.output or a.root/"analysis")

if __name__=="__main__":
    main()
