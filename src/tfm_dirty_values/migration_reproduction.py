from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .config import load_config, resolve_path
from .datasets import DatasetManager
from .metrics import classification_metrics
from .models import predict_probabilities


SOURCE_TASK_ID = "978fda0ee7280f41ec965fc0"
SOURCE_OUTPUT_SHA256 = "EEF40D23041E1DD2892761D1E6FDCE8C9A39CB2A56385C7CF8FB69B41A5B7C2F"
SOURCE_INJECTOR_SHA256 = "27AC0EC4D6B3963063C4021FC4622603B2BC10529C80C6AA1616C6AD2E629C89"
REPLAY_CONDITION = {
    "dataset_id": 44,
    "dataset_name": "spambase",
    "seed": 1907,
    "mechanism": "mcar",
    "rate": 0.30,
    "location": "query",
    "model": "tabiclv2",
    "treatment": "native",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _source_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    source_root = Path(config["source_pipeline"]["path"]).resolve()
    output = source_root / "results" / "kill_test" / "outputs" / f"{SOURCE_TASK_ID}.json"
    injector = source_root / "src" / "tfm_missingness" / "injection" / "missingness.py"
    return output, injector


def _validate_source_record(payload: dict[str, Any]) -> None:
    task = payload.get("task", {})
    for key, expected in REPLAY_CONDITION.items():
        actual = task.get(key)
        if isinstance(expected, float):
            matches = float(actual) == expected
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"Source task {key} mismatch: {actual!r} != {expected!r}")
    if task.get("task_id") != SOURCE_TASK_ID:
        raise ValueError(f"Unexpected source task ID: {task.get('task_id')}")
    metrics = payload.get("metrics", {})
    required = ("roc_auc", "injection", "model_provenance", "split_provenance")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"Source output is missing metric fields: {missing}")


def _load_source_record(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    output_path, _ = _source_paths(config)
    if not output_path.is_file():
        raise FileNotFoundError(f"Missing frozen source output: {output_path}")
    actual_hash = _sha256_file(output_path)
    if actual_hash != SOURCE_OUTPUT_SHA256:
        raise ValueError(f"Source output hash mismatch: {actual_hash} != {SOURCE_OUTPUT_SHA256}")
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    _validate_source_record(payload)
    return payload, output_path


def _load_source_injector(config: dict[str, Any]) -> tuple[ModuleType, Path]:
    _, injector_path = _source_paths(config)
    if not injector_path.is_file():
        raise FileNotFoundError(f"Missing frozen source injector: {injector_path}")
    actual_hash = _sha256_file(injector_path)
    if actual_hash != SOURCE_INJECTOR_SHA256:
        raise ValueError(
            f"Source injector hash mismatch: {actual_hash} != {SOURCE_INJECTOR_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("tfm_missingness_phase0_compat", injector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load source injector: {injector_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, injector_path


def _prepare_inputs(
    config: dict[str, Any],
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    source, source_output_path = _load_source_record(config)
    injector, injector_path = _load_source_injector(config)
    split = DatasetManager(config).prepare_split(
        int(REPLAY_CONDITION["dataset_id"]), int(REPLAY_CONDITION["seed"])
    )
    X_train, X_test, injection = injector.inject_missingness(
        split.X_train,
        split.X_test,
        dataset_id=int(REPLAY_CONDITION["dataset_id"]),
        split_seed=int(REPLAY_CONDITION["seed"]),
        mechanism=str(REPLAY_CONDITION["mechanism"]),
        rate=float(REPLAY_CONDITION["rate"]),
        location=str(REPLAY_CONDITION["location"]),
        categorical_indices=split.categorical_indices,
        config=config,
    )

    source_metrics = source["metrics"]
    source_split = source_metrics["split_provenance"]
    split_keys = (
        "train_indices_sha256",
        "test_indices_sha256",
        "X_train_sha256",
        "X_test_sha256",
    )
    split_comparisons = {
        key: {
            "source": source_split[key],
            "local": split.metadata[key],
            "match": source_split[key] == split.metadata[key],
        }
        for key in split_keys
    }
    source_mask = source_metrics["injection"]["query"]["mask_sha256"]
    local_mask = injection["query"]["mask_sha256"]

    checkpoint = resolve_path(config, config["models"]["tabiclv2"]["checkpoint"])
    checkpoint_hash = _sha256_file(checkpoint)
    expected_checkpoint_hash = str(
        config["models"]["tabiclv2"]["checkpoint_sha256"]
    ).upper()
    source_checkpoint_hash = str(
        source_metrics["model_provenance"]["checkpoint_sha256"]
    ).upper()
    installed_tabicl = importlib.metadata.version("tabicl")
    expected_tabicl = str(config["models"]["tabiclv2"]["package"]).split("==", 1)[1]

    checks = {
        "source_output_sha256": {
            "expected": SOURCE_OUTPUT_SHA256,
            "actual": _sha256_file(source_output_path),
            "match": True,
        },
        "source_injector_sha256": {
            "expected": SOURCE_INJECTOR_SHA256,
            "actual": _sha256_file(injector_path),
            "match": True,
        },
        "split": split_comparisons,
        "query_mask_sha256": {
            "source": source_mask,
            "local": local_mask,
            "match": source_mask == local_mask,
        },
        "clean_context_unchanged": {
            "expected": split.metadata["X_train_sha256"],
            "actual": _sha256_array(X_train),
            "match": split.metadata["X_train_sha256"] == _sha256_array(X_train),
        },
        "tabicl_checkpoint_sha256": {
            "configured": expected_checkpoint_hash,
            "source": source_checkpoint_hash,
            "actual": checkpoint_hash,
            "match": checkpoint_hash == expected_checkpoint_hash == source_checkpoint_hash,
        },
        "tabicl_package": {
            "expected": expected_tabicl,
            "installed": installed_tabicl,
            "match": installed_tabicl == expected_tabicl,
        },
    }
    ready = (
        all(item["match"] for item in split_comparisons.values())
        and checks["query_mask_sha256"]["match"]
        and checks["clean_context_unchanged"]["match"]
        and checks["tabicl_checkpoint_sha256"]["match"]
        and checks["tabicl_package"]["match"]
    )
    preflight = {
        "status": "READY" if ready else "BLOCKED",
        "outcome_producing": False,
        "condition": REPLAY_CONDITION,
        "source_task_id": SOURCE_TASK_ID,
        "checks": checks,
    }
    return split, X_train, X_test, injection, {"source": source, "preflight": preflight}


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    return _prepare_inputs(config)[4]["preflight"]


def run_reproduction(config: dict[str, Any]) -> dict[str, Any]:
    split, X_train, X_test, injection, context = _prepare_inputs(config)
    preflight_report = context["preflight"]
    if preflight_report["status"] != "READY":
        raise ValueError("Migration reproduction preflight is not READY")

    probabilities, model_provenance = predict_probabilities(
        str(REPLAY_CONDITION["model"]),
        X_train,
        split.y_train,
        X_test,
        categorical_indices=split.categorical_indices,
        seed=int(REPLAY_CONDITION["seed"]),
        config=config,
    )
    reproduced_metrics = classification_metrics(split.y_test, probabilities)
    source_metrics = context["source"]["metrics"]
    original_auc = float(source_metrics["roc_auc"])
    reproduced_auc = float(reproduced_metrics["roc_auc"])
    auc_difference = abs(original_auc - reproduced_auc)
    checkpoint_match = (
        str(model_provenance.get("checkpoint_sha256", "")).upper()
        == str(source_metrics["model_provenance"]["checkpoint_sha256"]).upper()
        and model_provenance.get("checkpoint_version")
        == source_metrics["model_provenance"].get("checkpoint_version")
    )
    passed = auc_difference == 0.0 and checkpoint_match
    return {
        "verification_status": "VERIFIED" if passed else "MISMATCH",
        "experiment_class": "deterministic_cross_project_same_seed_data_model_environment",
        "outcome_producing_dirty_values": False,
        "outcome_producing_migration_reproduction": True,
        "condition": REPLAY_CONDITION,
        "source_task_id": SOURCE_TASK_ID,
        "preflight": preflight_report,
        "reproduced": {
            "metrics": reproduced_metrics,
            "query_mask_sha256": injection["query"]["mask_sha256"],
            "probabilities_sha256": _sha256_array(probabilities),
            "model_provenance": model_provenance,
        },
        "comparison": {
            "roc_auc_original": original_auc,
            "roc_auc_reproduced": reproduced_auc,
            "roc_auc_absolute_difference": auc_difference,
            "auc_absolute_tolerance": 0.0,
            "checkpoint_match": checkpoint_match,
            "passed": passed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the isolated Phase 0 migration reproduction")
    parser.add_argument("--config", default="configs/kill_test.yaml")
    parser.add_argument("--output", default="artifacts/phase0/migration-reproduction.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-outcome-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.execute:
        if not args.confirm_outcome_run:
            raise SystemExit("Refusing model execution without --confirm-outcome-run")
        report = run_reproduction(config)
        output = resolve_path(config, args.output)
        _atomic_json(output, report)
        print(
            json.dumps(
                {
                    "output": str(output),
                    "verification_status": report["verification_status"],
                    "comparison": report["comparison"],
                },
                indent=2,
            )
        )
        if report["verification_status"] != "VERIFIED":
            raise SystemExit(1)
    else:
        if args.confirm_outcome_run:
            raise SystemExit("--confirm-outcome-run is valid only with --execute")
        print(json.dumps(preflight(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
