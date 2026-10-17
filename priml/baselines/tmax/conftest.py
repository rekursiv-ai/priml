"""Unpack the published TMax rollout record for the tests that read it from disk."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import gzip

import pytest


_CWD: Final = Path(__file__).resolve().parent


@pytest.fixture
def recorded_rollout(tmp_path: Path) -> Path:
    """Return a plain JSONL copy of the gzipped published record."""
    path = tmp_path / "recorded_rollout.jsonl"
    packed = _CWD / "testdata" / "recorded_rollout.jsonl.gz"
    path.write_bytes(gzip.decompress(packed.read_bytes()))
    return path
