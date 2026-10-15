"""Refit component jobs against verified frozen splits and pinned checkpoints."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

for name in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]:
    os.environ[name]="4"
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"src"))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--splits-root",type=Path,required=True)
    p.add_argument("--model-root",type=Path)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--shard",choices=["cpu","gpu"],required=True)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--dataset",type=int)
    p.add_argument("--seed",type=int)
    p.add_argument("--model")
    p.add_argument("--variant")
    p.add_argument("--experiment",choices=["outlier","category"])
    p.add_argument("--context-rate",type=float)
    args=p.parse_args()
    output=args.output_root.resolve()
    if output==args.splits_root.resolve() or (output/"MANIFEST.json").exists():
        raise ValueError("Choose a separate output directory, not an archived input directory")
    os.environ["TFM_SPLITS_ROOT"]=str(args.splits_root.resolve())
    if args.model_root:
        os.environ["TFM_MODEL_ROOT"]=str(args.model_root.resolve())
    from core import read,sha,atomic,identity,build_jobs,job_id,previous
    versions={"numpy":"2.2.6","pandas":"2.3.3","scikit-learn":"1.6.1",
              "xgboost":"3.2.0","lightgbm":"4.7.0"}
    if args.shard=="gpu":
        versions.update({"torch":"2.10.0+cu128","tabpfn":"2.2.1","tabicl":"2.1.1"})
    for package,expected in versions.items():
        if importlib.metadata.version(package)!=expected:
            raise ValueError("Pinned package version required: "+package+"=="+expected)
    jobs=read(HERE.parent/"component_jobs.json")
    if jobs!=build_jobs():
        raise ValueError("Job specification differs from the frozen design")
    study_identity=read(HERE.parent/"component_design.json")["original_study_identity"]
    selection=[j for j in jobs if j["shard"]==args.shard and all(
        value is None or j[key]==value for key,value in [
            ("dataset_id",args.dataset),("seed",args.seed),("model",args.model),
            ("variant",args.variant),("experiment",args.experiment),("context_rate",args.context_rate)])]
    if not selection:
        raise ValueError("No frozen jobs match the selection")
    if args.shard=="gpu":
        from worker_gpu_a1 import execute_a1 as execute
        for name in sorted({j["model"] for j in selection}):
            settings=previous.config()["models"][name]
            checkpoint=Path(settings["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint=ROOT/checkpoint
            if sha(checkpoint).lower()!=settings["checkpoint_sha256"].lower():
                raise ValueError("Checkpoint identity differs: "+name)
    else:
        from worker import execute
    output.mkdir(parents=True,exist_ok=True)
    marker=output/"REFIT_RUN.json"
    run_identity=identity({"study_identity":study_identity,
        "code":{p.relative_to(HERE).as_posix():sha(p) for p in HERE.rglob("*")
                if p.is_file() and p.suffix in {".py",".json"}}})
    if marker.exists() and read(marker)["run_identity"]!=run_identity:
        raise ValueError("The existing output directory belongs to a different runner")
    if not marker.exists():
        if any(output.iterdir()):
            raise ValueError("Use an empty directory for a new refit")
        atomic(marker,{"run_identity":run_identity,"original_study_identity":study_identity,
            "kind":"model_reexecution","versions":versions})
        atomic(output/"jobs.json",jobs)
    for name,subdir in {
        "TEMP":"temp","TMP":"temp","TMPDIR":"temp","HF_HOME":"hf","TORCH_HOME":"torch",
        "XDG_CACHE_HOME":"xdg","CUDA_CACHE_PATH":"cuda","TRITON_CACHE_DIR":"triton",
        "NUMBA_CACHE_DIR":"numba","JOBLIB_TEMP_FOLDER":"joblib"}.items():
        cache=output/"runtime"/subdir
        cache.mkdir(parents=True,exist_ok=True)
        os.environ[name]=str(cache)
    folder=output/"outputs"/args.shard
    folder.mkdir(parents=True,exist_ok=True)
    new=0
    for job in selection:
        path=folder/(job_id(job)+".json")
        if path.exists():
            r=read(path)
            if (r["job"]!=job or identity({k:v for k,v in r.items() if k!="record_sha256"})!=r["record_sha256"]
                or sha(folder/r["predictions_file"])!=r["predictions_sha256"]):
                raise ValueError("A saved refit record differs")
            continue
        r=execute(job,study_identity,folder)
        r["reexecution"]=True
        r["runner_identity"]=run_identity
        r["record_sha256"]=identity({k:v for k,v in r.items() if k!="record_sha256"})
        atomic(path,r)
        new+=1
        print(json.dumps({"completed_new":new,"job":job}),flush=True)
        if args.limit and new>=args.limit:
            break
    completed=sum((output/"outputs"/j["shard"]/(job_id(j)+".json")).exists() for j in jobs)
    if completed==len(jobs):
        snapshot={"format_version":1,"original_study_identity":study_identity,
            "amendment_identity":"f27d063eb7d8ba537bd2e64fae650567fe31eb43d080eb78755f2490f3799329",
            "expected_jobs":2715,"expected_predictive_views":6335,
            "files":{p.relative_to(output).as_posix():sha(p)
                     for p in [output/"jobs.json",*sorted((output/"outputs").glob("*/*"))] if p.is_file()}}
        snapshot["snapshot_identity"]=identity(snapshot)
        atomic(output/"public_snapshot.json",snapshot)
    print(json.dumps({"status":"COMPLETE" if completed==len(jobs) else "PARTIAL_REFIT",
                      "completed_jobs":completed,"planned_jobs":len(jobs)}))

if __name__=="__main__":
    main()
