"""Run the Phase 3 paired-identity probe and write its report.

This replays already-frozen Phase 2 arm-B cells with the cleaner switched off.
It produces no new research condition and writes no Phase 3 outcome.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfm_dirty_values.config import load_config, resolve_path  # noqa: E402
from tfm_dirty_values_ablation.paired_identity import run_paired_identity_probe  # noqa: E402


def main() -> int:
    config = load_config(ROOT / "configs" / "cleaning_ablation.yaml")
    phase2 = load_config(ROOT / "configs" / "full_study.yaml")

    report = run_paired_identity_probe(config, phase2)

    print(
        f"VERDICT: {report['verdict']}  "
        f"matched {report['matched_cell_count']}/{report['probe_cell_count']}  "
        f"tolerance {report['tolerance']}"
    )
    for entry in report["cells"]:
        cell = entry["cell"]
        tag = (
            f"{cell['dataset_id']}/{cell['seed']}/{cell['corruption']}"
            f"/{cell['rate']}/{cell['model']}"
        )
        print(("  OK   " if entry.get("match") else "  FAIL ") + tag)
        for key, diff in (entry.get("metrics") or {}).items():
            print(f"         {key}: abs_diff={diff['absolute_difference']}")
        for key, diff in (entry.get("array_hashes") or {}).items():
            print(f"         {key}: equal={diff['equal']}")
        if not entry.get("match") and entry.get("reason"):
            print(f"         reason={entry['reason']}")
    if report["errors"]:
        print("errors:")
        for error in report["errors"]:
            print(f"  - {error}")

    target = resolve_path(config, "artifacts/phase3/paired-identity-probe.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, target)
    print(f"report written: {target}")

    return 0 if report["verdict"] == "VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
