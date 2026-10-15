"""Check actual encoding inputs, component flags and the four precision-case fits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_components import read,sha,identity,array_sha,metrics,write_json

HERE=Path(__file__).resolve().parent
BUDGETS={"tabpfn":8,"tabiclv2":8,"xgboost":100,"lightgbm":100,"random_forest":500}

def audit(root):
    root=Path(root)
    records=[]
    for job in read(root/"jobs.json"):
        path=root/"outputs"/job["shard"]/(identity(job)[:24]+".json")
        r=read(path)
        if r["job"]!=job or not r["fresh_fit"]:
            raise ValueError("Incorrect job or reused fit")
        meta=r["model_metadata"]
        model,variant=job["model"],job["variant"]
        if meta["requested_estimators"]!=BUDGETS[model]:
            raise ValueError("Changed estimator budget")
        if model=="tabpfn":
            if not meta["outlier_active"] or any(flag!=(variant!="outlier_off") for flag in meta["outlier_active"]):
                raise ValueError("TabPFN outlier switch differs")
        elif model=="tabiclv2":
            expected="infinity" if variant=="outlier_off" else 4.
            if meta["outlier_threshold"]!=expected or not meta["standardization_and_100_clip_retained"]:
                raise ValueError("TabICL component settings differ")
            if variant=="outlier_off" and any(t["changed_cells"] for t in r["component_trace"] if t["component"]=="soft_outlier"):
                raise ValueError("Disabled soft step altered its input")
        elif model=="xgboost":
            if meta["tree_method"]!=("exact" if variant=="exact" else "hist"):
                raise ValueError("XGBoost split search differs")
            if variant=="native" and (not meta["native_categories"] or "c" not in meta["realized_feature_types"]):
                raise ValueError("Native XGBoost flags absent")
        elif model=="lightgbm":
            if meta["max_bin"]!=(1023 if variant=="fine_bins" else 255):
                raise ValueError("LightGBM bin setting differs")
            if variant=="native" and (not meta["native_categories"] or not meta["realized_categorical_columns"]):
                raise ValueError("Native LightGBM flags absent")
        elif model=="random_forest":
            if meta["min_samples_leaf"]!=(5 if variant=="leaf5" else 1):
                raise ValueError("RF leaf setting differs")
            if meta["max_features"]!=(1. if variant.endswith("_full") else "sqrt"):
                raise ValueError("RF feature sampling differs")
        records.append(r)
    groups={}
    for r in records:
        j=r["job"]
        key=(j["experiment"],j["dataset_id"],j["seed"],j["context_rate"])
        groups.setdefault(key,[]).append(r)
    category_groups=0
    native_pairs=0
    rf_pairs=0
    for key,rs in groups.items():
        if len({r["original_context_sha256"] for r in rs})!=1:
            raise ValueError("Matched conditions have different original context inputs")
        if key[0]!="category":
            continue
        category_groups+=1
        if len({r["encoding"]["vocabulary_sha256"] for r in rs})!=1:
            raise ValueError("Context vocabulary differs between encoding arms")
        lookup={(r["job"]["model"],r["job"]["variant"]):r for r in rs}
        for model in ["xgboost","lightgbm"]:
            a,b=lookup[(model,"ordinal")],lookup[(model,"native")]
            if a["model_context_sha256"]!=b["model_context_sha256"]:
                raise ValueError("Ordinal/native training arrays differ")
            if any(a["views"][v]["model_query_sha256"]!=b["views"][v]["model_query_sha256"] for v in a["views"]):
                raise ValueError("Ordinal/native query arrays differ")
            native_pairs+=1
        for representation in ["ordinal","onehot"]:
            a,b=lookup[("random_forest",representation)],lookup[("random_forest",representation+"_full")]
            if a["model_context_sha256"]!=b["model_context_sha256"] or any(
                a["views"][v]["model_query_sha256"]!=b["views"][v]["model_query_sha256"] for v in a["views"]):
                raise ValueError("RF feature-sampling control changed its input representation")
            rf_pairs+=1
    diagnosis=root/"diagnosis"/"precision_case"
    if not diagnosis.exists():
        diagnosis=root/"diagnostics"/"precision_case"
    cases=[]
    for variant in ["default","outlier_off"]:
        for precision in ["auto","fp32"]:
            stem=variant+"_"+precision
            record=read(diagnosis/(stem+".json"))
            prediction_path=diagnosis/(stem+".npz")
            if sha(prediction_path)!=record["predictions_sha256"]:
                raise ValueError("Precision-case prediction hash differs")
            with np.load(prediction_path,allow_pickle=False) as saved:
                for name,view in record["views"].items():
                    raw=saved["raw_"+name]
                    if array_sha(raw)!=view["probability_sha256"]:
                        raise ValueError("Precision-case array hash differs")
                    bad=(~np.isfinite(raw))|(raw<0)|(raw>1)
                    count=int(bad.any(axis=1).sum())
                    if count!=view["quality"]["invalid_rows"]:
                        raise ValueError("Precision-case numerical failure count differs")
                    if count:
                        if any(v is not None for v in view["metrics"].values()):
                            raise ValueError("Failed precision-case view was assigned a metric")
                    else:
                        actual=metrics(saved["y_query"],raw[:,1].astype(np.float64))
                        if max(abs(actual[m]-view["metrics"][m]) for m in actual)>1e-12:
                            raise ValueError("Precision-case metric differs")
                    cases.append(dict(variant=variant,precision=precision,view=name,invalid_rows=count,query_rows=len(raw)))
    report=dict(status="VERIFIED_SAVED_DESIGN_AND_DIAGNOSTICS",fit_jobs=len(records),
        category_context_groups=category_groups,identical_ordinal_native_input_pairs=native_pairs,
        identical_rf_feature_sampling_input_pairs=rf_pairs,precision_case_fits=4,precision_case_views=cases,
        boundary="Checks saved realized settings and input hashes; does not refit estimators")
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=HERE)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=audit(args.root)
    write_json(args.output or args.root/"saved_design_verification.json",result)
    print(json.dumps(result),flush=True)
