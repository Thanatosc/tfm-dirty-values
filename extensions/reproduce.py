"""Reproduce archived analyses. This file is copied to software/extensions/reproduce.py."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def verify_data(root):
    record=json.loads((root/"MANIFEST.json").read_text(encoding="utf-8"))
    for relative,value in record["files"].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or unsafe data path: "+relative)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=value["sha256"]:
            raise ValueError("Data digest mismatch: "+relative)
    return len(record["files"])

def compare_tables(actual_dir,reference_dir):
    checks=[]
    for reference in sorted(reference_dir.glob("*.csv")):
        actual=actual_dir/reference.name
        left=pd.read_csv(actual,float_precision="round_trip")
        right=pd.read_csv(reference,float_precision="round_trip")
        if list(left.columns)!=list(right.columns) or len(left)!=len(right):
            raise ValueError("Reconstructed schema differs: "+reference.name)
        maximum=0.
        for column in right.columns:
            a,b=left[column],right[column]
            if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
                x,y=a.to_numpy(float),b.to_numpy(float)
                if not np.allclose(x,y,rtol=0,atol=2e-14,equal_nan=True):
                    raise ValueError("Reconstructed values differ: "+reference.name+" / "+column)
                good=np.isfinite(x)&np.isfinite(y)
                if good.any():
                    maximum=max(maximum,float(np.max(np.abs(x[good]-y[good]))))
            elif not a.fillna("").astype(str).equals(b.fillna("").astype(str)):
                raise ValueError("Reconstructed identities differ: "+reference.name+" / "+column)
        checks.append(dict(table=reference.name,rows=len(left),maximum_difference=maximum))
    return checks

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--verify-only",action="store_true")
    p.add_argument("--figures",action="store_true")
    args=p.parse_args()
    root=args.data_root.resolve()
    files=verify_data(root)
    if args.verify_only:
        print(json.dumps(dict(status="VERIFIED_DATA_FILES",files=files)))
        return
    previous=module("previous_reproduce",HERE/"previous_reproduce.py")
    previous.HERE=root/"previous"
    old_argv=sys.argv
    try:
        sys.argv=["previous_reproduce.py","--output-dir","reproduced"]
        previous.main()
    finally:
        sys.argv=old_argv
    components=module("component_analysis",HERE/"analyze_components.py")
    result=components.analyze(root/"components",root/"components"/"reproduced")
    sys.path.insert(0,str(HERE))
    audit_module=module("audit_components",HERE/"audit_components.py")
    design_audit=audit_module.audit(root/"components")
    checks=compare_tables(root/"components"/"reproduced",root/"components"/"reference_analysis")
    if args.figures:
        figures=module("component_figures",HERE/"plot_components.py")
        figures.generate(root/"components",root/"components"/"figures_reproduced")
        old_figures=module("old_figures",HERE/"plot_previous.py")
        old_figures.HERE=root/"previous"
        old_figures.render(root/"previous"/"reproduced",root/"previous"/"figures_reproduced")
    report=dict(status="REPRODUCED",verified_data_files=files,component_table_checks=checks,
        design_verification=design_audit,
        component_study_identity=result["study_identity"],component_views=result["predictive_views"],
        component_numerical_failure_views=result["numerical_failure_views"],
        boundary="Earlier extensions: recorded-metric reconstruction. Component controls: saved-probability reconstruction. No model refit.")
    (root/"reproduction_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="component_table_checks"}),flush=True)

if __name__=="__main__":
    main()
