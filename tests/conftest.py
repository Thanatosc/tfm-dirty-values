from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
TEST_RUNTIME_ROOT = PROJECT_ROOT / "artifacts" / "tmp" / "pytest-fixtures"


@pytest.fixture
def tmp_path() -> Path:
    TEST_RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_RUNTIME_ROOT / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        target = path.resolve()
        if not target.is_relative_to(TEST_RUNTIME_ROOT.resolve()):
            raise ValueError("Test cleanup escaped the fixture directory")
        shutil.rmtree(target, ignore_errors=True)

