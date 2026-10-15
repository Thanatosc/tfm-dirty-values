# Extracted fingerprint-control analysis functions.
from common import *
from followup2 import SIGN, decomposition
import pandas as pd

GROUP=["fingerprint","representation","contrast","estimand","metric"]
CONTRASTS=[("raw","affine"),("affine","asinh"),("raw","asinh")]
def effect_rows(records,datasets=NUMERIC):
    rows=[]
    def metrics(dataset,seed,flag,rep,dirty):
        task=make_task(dataset,seed,"tabpfn","unit_error" if dirty else "clean",
            1000. if dirty else 0.,.5 if dirty else 0.,"joint" if dirty else "none",rep)
        return records[task_id({**task,"fingerprint_feature":flag})]["metrics"]
    def emit(dataset,seed,flag,rep,contrast,estimand,metric,value):
        rows.append(dict(dataset_id=dataset,seed=seed,fingerprint="on" if flag else "off",
            representation=rep,contrast=contrast,estimand=estimand,metric=metric,value=float(value)))
    for dataset in datasets:
        for seed in SEEDS:
            for flag in [True,False]:
                clean={rep:metrics(dataset,seed,flag,rep,False) for rep in ["raw","affine","asinh"]}
                dirty={rep:metrics(dataset,seed,flag,rep,True) for rep in ["raw","affine","asinh"]}
                for rep in clean:
                    for metric in METRICS:
                        emit(dataset,seed,flag,rep,"absolute","clean_value",metric,clean[rep][metric])
                        emit(dataset,seed,flag,rep,"absolute","dirty_value",metric,dirty[rep][metric])
                        emit(dataset,seed,flag,rep,"clean_to_dirty","degradation",metric,
                            SIGN[metric]*(clean[rep][metric]-dirty[rep][metric]))
                for a,b in CONTRASTS:
                    for metric in METRICS:
                        dec=decomposition(clean[a][metric],dirty[a][metric],clean[b][metric],dirty[b][metric],SIGN[metric])
                        for estimand,value in dec.items():
                            emit(dataset,seed,flag,b,a+"_to_"+b,estimand,metric,value)
    return pd.DataFrame(rows)


def dataset_aggregate(frame):
    keys=GROUP+["dataset_id"]
    actual=frame.groupby(keys,dropna=False).seed.agg(list)
    if any(sorted(seeds)!=sorted(SEEDS) for seeds in actual):
        raise ValueError("Expected each of the five paired seeds exactly once")
    return frame.groupby(keys,as_index=False,dropna=False).value.mean()


def add_flag_differences(frame):
    # Compare paired effects within each dataset. Do not subtract bootstrap CIs.
    subset=frame[frame.contrast!="absolute"]
    keys=[k for k in GROUP if k!="fingerprint"]+["dataset_id"]
    off=subset[subset.fingerprint=="off"].drop(columns="fingerprint")
    on=subset[subset.fingerprint=="on"].drop(columns="fingerprint")
    paired=off.merge(on,on=keys,suffixes=("_off","_on"),validate="one_to_one")
    if len(paired)!=len(off) or len(paired)!=len(on):raise ValueError("Missing paired fingerprint condition")
    paired["value"]=paired.pop("value_off")-paired.pop("value_on")
    paired["fingerprint"]="off_minus_on"
    return pd.concat([frame,paired],ignore_index=True)


def summarize(frame):
    rows=[]
    for keys,group in frame.groupby(GROUP,dropna=False,sort=True):
        ordered=group.sort_values("dataset_id")
        if ordered.dataset_id.duplicated().any():raise ValueError("Duplicate dataset estimate")
        row={**dict(zip(GROUP,keys)),**bootstrap(ordered.value.to_numpy(),("fingerprint-control-v1",keys))}
        row.update(min_dataset_value=float(ordered.value.min()),max_dataset_value=float(ordered.value.max()))
        rows.append(row)
    return pd.DataFrame(rows)
