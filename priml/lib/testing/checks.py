"""Assertions and helpers a single test calls.

- ``assert_help_without_docstring``: a CLI's ``--help`` survives ``python -OO``.
- ``default_signals``: start a child with default signal handling, whatever the
  suite inherited. A process inherits an ignored signal across ``exec``, so a
  suite started as ``nohup pytest ... &`` hands its children SIGHUP and SIGINT
  ignored, and a non-interactive bash cannot trap a signal it was started with
  ignored. A test that signals such a child then measures how the suite was
  launched, not the code under test::

      subprocess.Popen([*default_signals(signal.SIGINT), "/bin/bash", script])
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import pytest


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    import signal


__all__ = ["assert_help_without_docstring", "default_signals"]


def assert_help_without_docstring(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    entry: Callable[[], object],
) -> None:
    """Assert ``--help`` exits 0 when ``python -OO`` has stripped ``__doc__``.

    Args:
      monkeypatch: The test's monkeypatch fixture.
      module: Module whose docstring feeds ``description=``.
      entry: Calls the entry point; it reads ``sys.argv``.

    """
    monkeypatch.setattr(module, "__doc__", None)
    monkeypatch.setattr(sys, "argv", [module.__name__, "--help"])
    with pytest.raises(SystemExit) as exit_info:
        entry()
    if exit_info.value.code != 0:
        raise ValueError("Expected exit_info.value.code == 0.")


def default_signals(*signums: signal.Signals) -> list[str]:
    """Return an argv prefix that runs the rest with ``signums`` at their default action.

    A short Python interpreter resets each signal and then ``exec``s the command
    in its own place, so the command keeps the pid, process group and session
    ``subprocess`` gave it.

    Args:
      signums: The signals the command must start with at their default action.

    Returns:
      prefix: Arguments to put before the command's own argv.

    """
    if not signums:
        raise ValueError("name at least one signal to reset")
    numbers = ", ".join(str(int(signum)) for signum in signums)
    return [
        sys.executable,
        "-c",
        (
            "import os, signal, sys\n"
            f"for number in ({numbers},):\n"
            "    signal.signal(number, signal.SIG_DFL)\n"
            "os.execvp(sys.argv[1], sys.argv[1:])\n"
        ),
    ]
