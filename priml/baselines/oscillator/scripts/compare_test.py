"""Check the polyglot entry point runs the comparison CLI."""

from pathlib import Path
from typing import Final

import runpy

import pytest

from priml.baselines.oscillator.scripts import comparison


_THIS: Final = Path(__file__).resolve()
_CWD: Final = _THIS.parent


def _exit_code(argv: list[str] | None = None) -> int:
    del argv
    return 7


def test_running_the_entry_point_exits_with_the_comparison_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comparison, "main", _exit_code)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(
            str(_THIS.with_name("compare.py")),
            run_name="__main__",
        )
    assert raised.value.code == 7


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
