"""Recompute the Phase 3 cleaning-ablation analysis from the archived records.

Companion to ``reanalyze_archived_results.py``. That script reproduces the
degradation panel; this one reproduces the repair ablation, which needs two
databases because the uncleaned comparison arms live in the Phase 2 record.

Both databases are opened read-only. Nothing is written except the output file.

Typical use against the two published archives, unpacked side by side::

    python scripts/reanalyze_archived_ablation.py \
        --database        ../tfm-dirty-values-results/phase3/state-public.sqlite3 \
        --phase2-database ../tfm-dirty-values-results/phase2/state-public.sqlite3 \
        --task-manifest   ../tfm-dirty-values-results/phase3/task-manifest.json \
        --output          results/cleaning_ablation/analysis-public-rerun.json

The recomputed file is then compared against the archived
``results/cleaning_ablation/analysis.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfm_dirty_values.config import load_config  # noqa: E402
from tfm_dirty_values_ablation.analysis import evaluate_cleaning_ablation  # noqa: E402
from tfm_dirty_values_ablation.design import ablation_task_manifest  # noqa: E402

EXPECTED_TASK_COUNT = 2425
EXPECTED_ARMS = {"c": 2150, "d": 275}


def validate_manifest(config: dict, phase2_config: dict, manifest_path: Path) -> None:
    """Confirm the archived manifest still matches what the generator produces."""
    archived = json.loads(manifest_path.read_text(encoding="utf-8"))
    regenerated = ablation_task_manifest(config, phase2_config)

    if archived.get("task_count") != EXPECTED_TASK_COUNT:
        raise RuntimeError(
            f"Archived manifest holds {archived.get('task_count')} tasks, "
            f"expected {EXPECTED_TASK_COUNT}"
        )
    if archived.get("arm_counts") != EXPECTED_ARMS:
        raise RuntimeError(f"Archived arm counts {archived.get('arm_counts')} != {EXPECTED_ARMS}")
    if archived.get("task_manifest_sha256") != regenerated["task_manifest_sha256"]:
        raise RuntimeError(
            "Archived task-manifest identity does not match the regenerated panel; "
            "the configuration and the archive disagree"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cleaning_ablation.yaml")
    parser.add_argument("--phase2-config", default="configs/full_study.yaml")
    parser.add_argument("--database", required=True, help="Phase 3 public SQLite database")
    parser.add_argument("--phase2-database", required=True, help="Phase 2 public SQLite database")
    parser.add_argument("--task-manifest", default=None)
    parser.add_argument(
        "--output", default="results/cleaning_ablation/analysis-public-rerun.json"
    )
    parser.add_argument(
        "--compare-to",
        default="results/cleaning_ablation/analysis.json",
        help="archived analysis to compare against; pass an empty string to skip",
    )
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    phase2_config = load_config(ROOT / args.phase2_config)

    # Point the frozen contracts at the archived public databases without
    # editing either configuration file on disk.
    config["paths"] = dict(config["paths"])
    config["paths"]["database"] = str(Path(args.database).resolve())
    config["paths"]["phase2_database"] = str(Path(args.phase2_database).resolve())

    manifest_path = (
        Path(args.task_manifest).resolve()
        if args.task_manifest
        else ROOT / "artifacts/phase3/cleaning-ablation-task-manifest.json"
    )
    if manifest_path.is_file():
        validate_manifest(config, phase2_config, manifest_path)
        print(f"task manifest verified: {manifest_path}")
    else:
        print(f"task manifest not found, skipping identity check: {manifest_path}")

    report = evaluate_cleaning_ablation(config, phase2_config)
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    target = Path(args.output)
    if not target.is_absolute():
        target = ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, target)

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"status      : {report['status']}")
    print(f"tasks       : {report['task_count']}")
    print(f"arms        : {report['arms']}")
    print(f"written     : {target}")
    print(f"sha256      : {digest}")

    if args.compare_to:
        reference = Path(args.compare_to)
        if not reference.is_absolute():
            reference = ROOT / reference
        if reference.is_file():
            reference_digest = hashlib.sha256(reference.read_bytes()).hexdigest()
            match = reference_digest == digest
            print(f"archived    : {reference_digest}")
            print(f"byte-identical to the archived analysis: {match}")
            return 0 if match else 1
        print(f"archived analysis not found, comparison skipped: {reference}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
