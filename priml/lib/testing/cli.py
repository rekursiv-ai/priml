"""Test helpers for command-line entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sys

import pytest


if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


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
