"""Check the thin release entry point without starting a benchmark."""

from pathlib import Path

import runpy
import sys

import pytest


def test_release_entry_point_dispatches_help(monkeypatch: pytest.MonkeyPatch) -> None:
    script = Path(__file__).with_name("release.py")
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    with pytest.raises(SystemExit) as status:
        runpy.run_path(str(script), run_name="__main__")
    assert status.value.code == 0
