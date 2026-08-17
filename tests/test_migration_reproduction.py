from copy import deepcopy

import pytest

from tfm_dirty_values.migration_reproduction import (
    REPLAY_CONDITION,
    SOURCE_TASK_ID,
    _validate_source_record,
)


def _source_record() -> dict:
    return {
        "task": {"task_id": SOURCE_TASK_ID, **REPLAY_CONDITION},
        "metrics": {
            "roc_auc": 0.9,
            "injection": {"query": {"mask_sha256": "mask"}},
            "model_provenance": {"checkpoint_sha256": "checkpoint"},
            "split_provenance": {"X_train_sha256": "train"},
        },
    }


def test_frozen_source_record_identity_is_accepted() -> None:
    _validate_source_record(_source_record())


@pytest.mark.parametrize("key,value", [("dataset_id", 31), ("rate", 0.5), ("model", "tabpfn")])
def test_source_record_identity_drift_is_rejected(key: str, value: object) -> None:
    record = deepcopy(_source_record())
    record["task"][key] = value
    with pytest.raises(ValueError, match=f"Source task {key} mismatch"):
        _validate_source_record(record)


def test_source_record_requires_provenance_fields() -> None:
    record = _source_record()
    del record["metrics"]["split_provenance"]
    with pytest.raises(ValueError, match="missing metric fields"):
        _validate_source_record(record)
