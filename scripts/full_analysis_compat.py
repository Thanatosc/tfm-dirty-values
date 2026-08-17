from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from tfm_dirty_values.config import canonical_json, load_config
from tfm_dirty_values.full_analysis import evaluate_full_study
from tfm_dirty_values.full_authorization import require_full_authorized
from tfm_dirty_values.worker import create_store


class MetricAliasStore:
    """Read-only adapter for the preregistered ECE result-key mismatch."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self.alias_count = 0

    def rows(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        adapted: list[dict[str, Any]] = []
        for source_row in self._store.rows(*args, **kwargs):
            row = dict(source_row)
            raw_metrics = row.get("metrics_json")
            if raw_metrics is None:
                adapted.append(row)
                continue
            metrics = json.loads(str(raw_metrics))
            stored = metrics.get("ece_10_equal_width")
            generic = metrics.get("ece")
            if generic is not None and stored is not None and float(generic) != float(stored):
                raise ValueError(
                    f"Conflicting ECE values for task {row.get('task_id')}: "
                    f"ece={generic}, ece_10_equal_width={stored}"
                )
            if generic is None:
                if stored is None:
                    raise ValueError(
                        f"Task {row.get('task_id')} has neither ece nor ece_10_equal_width"
                    )
                metrics["ece"] = float(stored)
                row["metrics_json"] = canonical_json(metrics)
                self.alias_count += 1
            adapted.append(row)
        return adapted


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen full-study analysis with the registered ECE key alias."
    )
    parser.add_argument("--config", default="configs/full_study.yaml")
    parser.add_argument("--output", default="results/full_study/analysis.json")
    args = parser.parse_args()

    config = load_config(args.config)
    authorization = require_full_authorized(config)
    store = MetricAliasStore(create_store(config))
    report = evaluate_full_study(store, config)
    report["compatibility_amendment"] = {
        "type": "metric_key_alias",
        "source_key": "ece_10_equal_width",
        "analysis_key": "ece",
        "aliased_task_count": store.alias_count,
        "source_database_mutated": False,
        "authorized_runtime_code_identity_sha256": authorization[
            "runtime_code_identity_sha256"
        ],
    }
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    _atomic_write(Path(args.output), rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
