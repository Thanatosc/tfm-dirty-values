"""Recompute both exploratory extensions and fingerprint summaries from scalar records.

No OpenML download, model import, fitting, or prediction is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "code"))
import followup1
import followup2
import fingerprint
import diagnostics
from common import task_id


def read_records(label, count, identifier):
    records = {}
    source = HERE / "data" / label / "records.jsonl"
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            envelope = json.loads(line)
            item = envelope["record"]
            tid = identifier(item["task"])
            if envelope["source_filename"] != tid + ".json" or tid in records:
                raise ValueError(f"Duplicate or mismatched task in {label}: {tid}")
            if not all(np.isfinite(float(v)) for v in item["metrics"].values()):
                raise ValueError(f"Nonfinite metric in {label}: {tid}")
            records[tid] = item
    if len(records) != count:
        raise ValueError(f"Incomplete {label}: {len(records)} / {count}")
    return records


def compare(actual, label, name, keys):
    path = HERE / "reference_analysis" / label / (name + ".csv")
    expected = pd.read_csv(path, float_precision="round_trip")
    if set(actual.columns) != set(expected.columns) or len(actual) != len(expected):
        raise ValueError(f"Schema/count mismatch: {label}/{name}")
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError(f"Duplicate aggregate identity: {label}/{name}")
    left = actual.sort_values(keys).reset_index(drop=True)[expected.columns]
    right = expected.sort_values(keys).reset_index(drop=True)
    maximum = 0.0
    for col in expected.columns:
        if pd.api.types.is_numeric_dtype(right[col]):
            a, b = left[col].to_numpy(float), right[col].to_numpy(float)
            if not np.allclose(a, b, atol=2e-14, rtol=0, equal_nan=True):
                raise ValueError(f"Numerical mismatch: {label}/{name}/{col}")
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                maximum = max(maximum, float(np.max(np.abs(a[finite] - b[finite]))))
        elif not left[col].fillna("").equals(right[col].fillna("")):
            raise ValueError(f"Identity mismatch: {label}/{name}/{col}")
    return {"table": f"{label}/{name}.csv", "rows": len(actual),
            "max_absolute_difference": maximum, "absolute_tolerance": 2e-14}


def emit(frame, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / (name + ".csv"), index=False, float_format="%.17g")


def verify_package():
    path = HERE / "MANIFEST.json"
    if not path.exists():
        raise ValueError("Package MANIFEST.json is missing; integrity verification cannot proceed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        candidate = (HERE / relative).resolve()
        if not candidate.is_relative_to(HERE) or not candidate.is_file():
            raise ValueError(f"Missing or invalid packaged file: {relative}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError(f"Package hash mismatch: {relative}")
    return {"status": "VERIFIED", "files": len(manifest["files"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="reproduced")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--figures", action="store_true", help="Also redraw manuscript figures after analysis")
    args = parser.parse_args()
    out = (HERE / args.output_dir).resolve()
    if not out.is_relative_to(HERE) or out == HERE or out.parts[len(HERE.parts)] in {
        "data", "reference_analysis", "code", "recorded_code", "protocols", "assets", "provenance", "manuscript"}:
        raise ValueError("Use a dedicated generated-output directory inside this package")
    package_check = verify_package()
    if args.verify_only:
        print(json.dumps(package_check))
        return
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    full = json.loads((HERE / "data/original/full_study.json").read_text(encoding="utf-8"))
    legacy = json.loads((HERE / "data/original/cleaning_ablation.json").read_text(encoding="utf-8"))
    if len(full) != 4575 or len(legacy) != 2425:
        raise ValueError("Original control archive coverage mismatch")
    print("Recomputing first exploratory extension (4,195 records)", flush=True)
    records = read_records("followup1", 4195, followup1.task_id)
    seed = followup1.paired_effects(records, full, legacy)
    dataset = followup1.add_families(followup1.aggregate_seeds(seed))
    summary = followup1.summarize(dataset)
    for name, frame, keys in [
        ("paired_seed_effects", seed, followup1.KEYS + ["metric", "estimand"]),
        ("dataset_effects", dataset, followup1.GROUPS + ["dataset_id", "model"]),
        ("summary", summary, followup1.GROUPS + ["model"])]:
        checks.append(compare(frame, "followup1", name, keys))
        emit(frame, out / "followup1", name)
    print("Recomputing second exploratory extension (12,450 records)", flush=True)
    records = read_records("followup2", 12450, task_id)
    raw, seed = followup2.effect_rows(records)
    dataset = followup2.add_model_groups(followup2.dataset_aggregate(seed))
    summary = followup2.summarize(dataset)
    for name, frame, keys in [
        ("seed_effects", seed, followup2.GROUP + ["dataset_id", "seed"]),
        ("dataset_effects", dataset, followup2.GROUP + ["dataset_id"]),
        ("summary", summary, followup2.GROUP)]:
        checks.append(compare(frame, "followup2", name, keys))
        emit(frame, out / "followup2", name)
    emit(raw, out / "followup2", "task_metrics")
    print("Recomputing fingerprint control (540 records)", flush=True)
    records = read_records("fingerprint", 540, task_id)
    seed = fingerprint.effect_rows(records)
    dataset = fingerprint.add_flag_differences(fingerprint.dataset_aggregate(seed))
    summary = fingerprint.summarize(dataset)
    for name, frame, keys in [
        ("seed_effects", seed, fingerprint.GROUP + ["dataset_id", "seed"]),
        ("dataset_effects", dataset, fingerprint.GROUP + ["dataset_id"]),
        ("summary", summary, fingerprint.GROUP)]:
        checks.append(compare(frame, "fingerprint", name, keys))
        emit(frame, out / "fingerprint", name)
    print("Reaggregating recorded input and cleaner diagnostics", flush=True)
    for label in ["coverage", "categorical_repair"]:
        seeds = pd.read_csv(HERE / "reference_analysis/diagnostics_extension" / (label + "_seeds.csv"), float_precision="round_trip")
        counts = seeds.groupby(diagnostics.GROUP + ["dataset_id"], dropna=False).seed.agg(["count", "nunique"])
        if not ((counts["count"] == 5) & (counts["nunique"] == 5)).all():
            raise ValueError("Diagnostic condition is missing a paired seed")
        dataset, summary = diagnostics.aggregate(seeds)
        for suffix, frame, keys in [("datasets", dataset, diagnostics.GROUP + ["dataset_id"]),
                                    ("summary", summary, diagnostics.GROUP + ["metric"])]:
            name = label + "_" + suffix
            checks.append(compare(frame, "diagnostics_extension", name, keys))
            emit(frame, out / "diagnostics_extension", name)
    report = {"status": "REPRODUCED_FROM_RECORDED_METRICS", "package": package_check,
              "record_counts": {"original_controls": 7000, "followup1": 4195, "followup2": 12450, "fingerprint": 540},
              "checks": checks, "seconds": time.monotonic() - started,
              "model_predictions": 0, "confirmatory_tests_added": 0,
              "scope": "Portable scalar-metric reproduction and reaggregation of recorded diagnostics; not an independent prediction or raw-input reconstruction run."}
    (out / "reproduction_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.figures:
        from plot_figures import render
        render(out, out / "figures")
    print(json.dumps({"status": report["status"], "checks": len(checks), "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
