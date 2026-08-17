from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_RUNTIME_ROOT = PROJECT_ROOT / "artifacts" / "tmp" / "pytest-fixtures"


@pytest.fixture
def tmp_path() -> Path:
    TEST_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_RUNTIME_ROOT / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

