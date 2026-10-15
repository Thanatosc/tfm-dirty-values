# Unchanged aggregation of recorded diagnostic measurements.
from common import *
import pandas as pd

KEYS = ["cohort", "dataset_id", "seed", "corruption", "level", "rate", "placement", "pattern"]
GROUP = [k for k in KEYS if k not in ["dataset_id", "seed"]] + ["partition"]
def aggregate(frame):
    values=[c for c in frame.columns if c not in KEYS + ["partition", "mask_sha256"]]
    datasets=frame.groupby(GROUP+["dataset_id"],dropna=False,as_index=False)[values].mean()
    rows=[]
    for keys, group in datasets.groupby(GROUP,dropna=False,sort=True):
        for metric in values:
            x=group.sort_values("dataset_id")[metric].dropna().to_numpy()
            estimate=bootstrap(x,("input-diagnostic-v1",keys,metric)) if len(x) else dict(
                n_datasets=0,mean=None,ci_low=None,ci_high=None,positive_datasets=0,negative_datasets=0)
            rows.append({**dict(zip(GROUP,keys)),"metric":metric,**estimate})
    return datasets,pd.DataFrame(rows)
