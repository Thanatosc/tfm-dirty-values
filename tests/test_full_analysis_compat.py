from __future__ import annotations

import json

import pytest

from scripts.full_analysis_compat import MetricAliasStore


class FakeStore:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def rows(self, *args: object, **kwargs: object) -> list[dict[str, object]]:
        return self._rows


def test_metric_alias_store_maps_ece_without_mutating_source() -> None:
    source = [{"task_id": "task-1", "metrics_json": '{"ece_10_equal_width":0.125}'}]
    adapter = MetricAliasStore(FakeStore(source))

    adapted = adapter.rows(stage=3)

    assert json.loads(str(adapted[0]["metrics_json"]))["ece"] == 0.125
    assert json.loads(str(source[0]["metrics_json"])) == {"ece_10_equal_width": 0.125}
    assert adapter.alias_count == 1


@pytest.mark.parametrize(
    "metrics",
    [
        {"roc_auc": 0.5},
        {"ece": 0.1, "ece_10_equal_width": 0.2},
    ],
)
def test_metric_alias_store_fails_closed(metrics: dict[str, float]) -> None:
    adapter = MetricAliasStore(
        FakeStore([{"task_id": "task-1", "metrics_json": json.dumps(metrics)}])
    )

    with pytest.raises(ValueError):
        adapter.rows(stage=3)
