"""Portable metric replay and explicitly requested corrective refits; no network."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

import numpy as np

METRICS = ("roc_auc", "balanced_accuracy", "log_loss", "ece_10_equal_width")
PINS = {"numpy": "2.2.6", "pandas": "2.3.3", "scikit-learn": "1.6.1", "scipy": "1.15.3"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8).tobytes()).hexdigest()


def original_modules(root):
    archive = Path(__file__).resolve().parents[2] / "archives" / "tfm-dirty-values-v1.2.1.zip"
    assert digest(archive) == "4bfb4a9963f476dd4e1ffd121b5061d7219beecb33935d871be0546979dc827b"
    sys.path.insert(0, str(archive) + "/tfm-dirty-values/src")
    from tfm_dirty_values.metrics import classification_metrics
    from tfm_dirty_values.injection import inject_corruption
    return classification_metrics, inject_corruption


def replay(root):
    metric_function, _ = original_modules(root)
    directory = root / "correction"
    verification = read(directory / "verification-report.json")
    assert verification["status"] == "PASS" and verification["tasks"] == 915
    manifest = read(directory / "task-manifest.json")
    expected = read(directory / "expected-inputs.json")
    counts = {"verified_v1_reuse": 0, "corrective_refit": 0}
    replayed = 0
    for task in manifest["tasks"]:
        path = directory / "results" / (task["task_id"] + ".json")
        assert digest(path) == verification["source_record_manifest"][path.name]
        record = read(path)
        assert record["status"] == "completed" and record["task"] == task
        assert record["metrics"]["input_provenance"] == expected[task["task_id"]]["input_provenance"]
        assert record["metrics"]["injection"] == expected[task["task_id"]]["injection"]
        kind = record["source"]["kind"]
        counts[kind] += 1
        if kind == "verified_v1_reuse":
            original = {k: v for k, v in record.items() if k != "source"}
            serialized = json.dumps(original, sort_keys=True, ensure_ascii=False, indent=2)
            candidates = [serialized.encode("utf-8"), serialized.replace("\n", "\r\n").encode("utf-8")]
            assert record["source"]["sha256"] in [hashlib.sha256(x).hexdigest() for x in candidates]
        else:
            pred_path = (directory / record["predictions"]["file"]).resolve()
            assert pred_path.is_relative_to(directory.resolve())
            assert digest(pred_path) == record["predictions"]["sha256"]
            with np.load(pred_path, allow_pickle=False) as arrays:
                p, y = arrays["probabilities"], arrays["y_true"]
                assert len(p) == len(y) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                actual = metric_function(y, p)
            assert all(actual[k] == record["metrics"][k] for k in METRICS)
            replayed += 4
    assert counts == {"verified_v1_reuse": 595, "corrective_refit": 320} and replayed == 1280
    print(json.dumps({"status": "PASS", "records": 915, "retained_record_byte_identities": 595,
                      "prediction_vectors_replayed": 320, "metrics_replayed": replayed,
                      "maximum_metric_difference": 0., "model_refits": 0}), flush=True)


def reconstruct(task, input_root, expected, injector):
    folder = input_root / str(task["dataset_id"])
    with np.load(folder / f"split-{task['seed']}.npz", allow_pickle=False) as loaded:
        arrays = dict(loaded)
    meta = read(folder / f"split-{task['seed']}.json")
    cats = list(map(int, meta["categorical_indices"]))
    x, query, injection = injector(arrays["X_train"], arrays["X_test"], dataset_id=int(task["dataset_id"]),
        split_seed=int(task["seed"]), corruption=task["corruption"], rate=float(task["rate"]),
        categorical_indices=cats, params=dict(sorted(task["parameters"].items())))
    provenance = {"clean_X_train_sha256": array_digest(arrays["X_train"]), "clean_X_test_sha256": array_digest(arrays["X_test"]),
                  "model_X_train_sha256": array_digest(x), "model_X_test_sha256": array_digest(query),
                  "train_indices_sha256": meta["train_indices_sha256"], "test_indices_sha256": meta["test_indices_sha256"]}
    assert provenance == expected["input_provenance"] and injection == expected["injection"]
    assert np.array_equal(np.unique(arrays["y_train"]), [0, 1])
    return arrays, cats, x, query


def verify_inputs_or_refit(root, input_root, output):
    metric_function, injector = original_modules(root)
    manifest = read(root / "correction/task-manifest.json")
    expected = read(root / "correction/expected-inputs.json")
    assert len(manifest["tasks"]) == 915 and len(manifest["refit_ids"]) == 320
    # Complete all input checks before any model fitting is allowed.
    for task in manifest["tasks"]:
        reconstruct(task, input_root, expected[task["task_id"]], injector)
    if output is None:
        print(json.dumps({"status": "PASS", "archived_input_identity_checks": 915, "model_refits": 0}), flush=True)
        return
    assert metadata.version("catboost") == "1.2.10"
    assert not output.exists(), "Choose a new output directory; archived outcomes must not be replaced"
    output.mkdir(parents=True)
    sys.path.insert(0, str(root))
    import catboost_adapter
    ids = set(manifest["refit_ids"])
    count = 0
    for task in manifest["tasks"]:
        if task["task_id"] not in ids:
            continue
        arrays, cats, x, query = reconstruct(task, input_root, expected[task["task_id"]], injector)
        probabilities, model = catboost_adapter.fit_predict(x, arrays["y_train"], query, cats, task["seed"])
        assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
        metrics = metric_function(arrays["y_test"], probabilities)
        np.savez_compressed(output / (task["task_id"] + ".npz"), probabilities=probabilities, y_true=arrays["y_test"])
        record = {"task": task, "metrics": metrics, "model_provenance": model,
                  "scope": "independent corrective refit; not a replacement for an archived record"}
        (output / (task["task_id"] + ".json")).write_text(json.dumps(record, indent=2), encoding="utf-8")
        count += 1
        print(json.dumps({"corrective_refits_completed": count, "assigned": 320}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--verify-inputs", action="store_true")
    actions.add_argument("--refit", action="store_true")
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert sys.version_info[:3] == (3, 11, 9), "Use the recorded Python 3.11.9 runtime"
    assert {k: metadata.version(k) for k in PINS} == PINS, "Use the pinned metric dependencies"
    if args.refit or args.verify_inputs:
        assert args.input_root is not None, "Supply the restored archived split directory"
        assert not args.refit or args.output is not None, "Refits require a new output directory"
        verify_inputs_or_refit(args.root.resolve(), args.input_root.resolve(), args.output.resolve() if args.refit else None)
    else:
        replay(args.root.resolve())


if __name__ == "__main__":
    main()
