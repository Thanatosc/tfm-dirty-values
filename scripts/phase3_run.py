"""Phase 3 driver: authorize, initialize, run, report.

Each step is a separate flag so the gates stay inspectable. Invoked by
scripts/start_cleaning_ablation.ps1, which owns the human confirmation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfm_dirty_values.config import load_config  # noqa: E402
from tfm_dirty_values_ablation.authorization import (  # noqa: E402
    validate_authorization,
    write_authorization,
)
from tfm_dirty_values_ablation.worker import (  # noqa: E402
    completion_report,
    initialize,
    run_worker,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 cleaning ablation driver")
    parser.add_argument("--authorize", action="store_true", help="write the authorization artifact")
    parser.add_argument("--initialize", action="store_true", help="queue the frozen task panel")
    parser.add_argument("--run", action="store_true", help="run the serial worker")
    parser.add_argument("--report", action="store_true", help="write the completion report")
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "cleaning_ablation.yaml")
    phase2 = load_config(ROOT / "configs" / "full_study.yaml")

    if args.authorize:
        valid, reason, _ = validate_authorization(config, phase2)
        if valid:
            print("authorization: already present and valid")
        else:
            payload = write_authorization(
                config,
                phase2,
                command="scripts/start_cleaning_ablation.ps1 -ConfirmPreregisteredRun",
            )
            print(f"authorization: written at {payload['authorized_at']}")
            print(f"  task_count            : {payload['task_count']}")
            print(f"  manifest identity     : {payload['task_manifest_identity_sha256'][:16]}")
            print(f"  paired-identity       : {payload['paired_identity_verdict']}")
            print(f"  phase2 runtime bound  : {payload['phase2_runtime_code_identity_sha256'][:16]}")

    if args.initialize:
        result = initialize(config, phase2)
        print(
            f"initialize: inserted {result['inserted']} of {result['task_count']} "
            f"tasks, arms {result['arm_counts']}"
        )

    if args.run:
        summary = run_worker(config, phase2)
        print(
            f"worker: completed {summary['completed']}, failed {summary['failed']}, "
            f"wall {summary['wall_seconds'] / 60:.1f} min, "
            f"open_remain {summary['open_tasks_remain']}"
        )

    if args.run or args.report:
        report = completion_report(config)
        print(f"completion gate: {report['gate']}  counts {report['status_counts']}")
        if report["arm_coverage_errors"]:
            print(f"  arm coverage errors: {report['arm_coverage_errors']}")
        return 0 if report["gate"] == "COMPLETE_PENDING_VALIDATION" else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
