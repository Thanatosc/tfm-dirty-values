"""Freeze the Phase 3 task manifest. Produces no model outcome."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfm_dirty_values.config import load_config  # noqa: E402
from tfm_dirty_values_ablation.worker import freeze_task_manifest  # noqa: E402


def main() -> int:
    config = load_config(ROOT / "configs" / "cleaning_ablation.yaml")
    phase2 = load_config(ROOT / "configs" / "full_study.yaml")
    manifest = freeze_task_manifest(config, phase2)
    print(f"task_count : {manifest['task_count']}")
    print(f"arm_counts : {manifest['arm_counts']}")
    print(f"identity   : {manifest['task_manifest_sha256']}")
    print(f"written    : {config['paths']['task_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
