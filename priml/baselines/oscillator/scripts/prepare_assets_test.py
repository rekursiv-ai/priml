"""The polyglot launcher exits with the asset preparation status."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import runpy

import pytest

from priml.baselines.oscillator.scripts import asset_preparation


_CWD: Final = Path(__file__).resolve().parent


def test_main_guard_exits_with_preparation_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(asset_preparation, "main", lambda: 7)
    script = _CWD / "prepare_assets.py"
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(script), run_name="__main__")
    assert raised.value.code == 7


def test_import_does_not_run_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> int:
        raise AssertionError("main ran on import")

    monkeypatch.setattr(asset_preparation, "main", fail)
    script = _CWD / "prepare_assets.py"
    namespace = runpy.run_path(str(script), run_name="prepare_assets")
    assert namespace["main"] is fail


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
