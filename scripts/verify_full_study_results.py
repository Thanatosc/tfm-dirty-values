from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tfm_dirty_values.config import canonical_json, load_config, resolve_path
from tfm_dirty_values.worker import create_store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify every full-study atomic output against the SQLite queue."
    )
    parser.add_argument("--config", default="configs/full_study.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    rows = create_store(config).rows(stage=3)
    output_dir = resolve_path(config, config["paths"]["outputs"])
    errors: list[str] = []
    hashes: list[tuple[str, str]] = []
    for row in rows:
        task_id = str(row["task_id"])
        path = output_dir / f"{task_id}.json"
        if not path.is_file():
            errors.append(f"missing:{task_id}")
            continue
        raw = path.read_bytes()
        hashes.append((task_id, hashlib.sha256(raw).hexdigest()))
        output = json.loads(raw)
        if output.get("task", {}).get("task_id") != task_id:
            errors.append(f"task_id:{task_id}")
        if row["status"] != "completed":
            errors.append(f"status:{task_id}:{row['status']}")
        if canonical_json(output.get("metrics")) != row["metrics_json"]:
            errors.append(f"metrics:{task_id}")

    output_count = len(list(Path(output_dir).glob("*.json")))
    manifest_identity = hashlib.sha256(
        canonical_json(sorted(hashes)).encode("ascii")
    ).hexdigest()
    report = {
        "rows": len(rows),
        "output_files": output_count,
        "matched": len(rows) - len(errors),
        "error_count": len(errors),
        "errors": errors[:20],
        "output_manifest_identity_sha256": manifest_identity,
    }
    print(json.dumps(report, indent=2))
    if errors or output_count != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
