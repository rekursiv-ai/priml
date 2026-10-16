"""Whether this pytest run rewrites goldens instead of asserting against them.

Two flags, registered by priml's ``conftest.py`` (which the monorepo root's
calls, so every suite sees them):

- ``--regenerate-golden``: text goldens (config pprints, rendered text).
- ``--regenerate-b4b``: bit-for-bit tensor goldens.

Flags, not environment variables: pytest rejects a misspelled flag, and a flag
cannot linger in a shell to silently rewrite every later run's goldens.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import weakref


if TYPE_CHECKING:
    from collections.abc import Generator

    import pytest


__all__ = [
    "add_options",
    "b4b",
    "configure",
    "forced",
    "golden",
    "override",
    "suppressed",
]


@dataclass(slots=True, kw_only=True)
class _Flags:
    golden: bool = False
    b4b: bool = False


_FLAGS: Final = _Flags()
"""This run's flags, read from pytest's command line by :func:`configure`."""

_registered: weakref.WeakSet[pytest.Parser] = weakref.WeakSet()
"""Parsers already given the flags."""


def add_options(parser: pytest.Parser) -> None:
    """Register ``--regenerate-golden`` and ``--regenerate-b4b``, once per parser.

    Both the monorepo's root conftest and priml's call this: priml's so the
    export ships the flags, the root's so suites outside priml see them. A run
    of priml tests loads both, and pytest refuses a second ``addoption``.

    Args:
      parser: The pytest option parser.

    """
    if parser in _registered:
        return
    _registered.add(parser)
    parser.addoption(
        "--regenerate-golden",
        action="store_true",
        default=False,
        help="Rewrite text goldens instead of asserting against them.",
    )
    parser.addoption(
        "--regenerate-b4b",
        action="store_true",
        default=False,
        help="Rewrite bit-for-bit tensor goldens instead of asserting against them.",
    )


def configure(config: pytest.Config) -> None:
    """Read both flags from ``config`` for the rest of the run."""
    _FLAGS.golden = bool(config.getoption("--regenerate-golden"))
    _FLAGS.b4b = bool(config.getoption("--regenerate-b4b"))


def golden() -> bool:
    """Return whether text goldens are rewritten this run."""
    return _FLAGS.golden


def b4b() -> bool:
    """Return whether bit-for-bit tensor goldens are rewritten this run."""
    return _FLAGS.b4b


def override(
    monkeypatch: pytest.MonkeyPatch,
    *,
    golden: bool | None = None,
    b4b: bool | None = None,
) -> None:
    """Set the flags for one test; ``monkeypatch`` restores them after it.

    Args:
      monkeypatch: The test's monkeypatch fixture.
      golden: New text-golden flag; ``None`` keeps the current one.
      b4b: New tensor-golden flag; ``None`` keeps the current one.

    """
    if golden is not None:
        monkeypatch.setattr(_FLAGS, "golden", golden)
    if b4b is not None:
        monkeypatch.setattr(_FLAGS, "b4b", b4b)


@contextmanager
def suppressed() -> Generator[None]:
    """Turn both flags off inside the block, so a comparison cannot rewrite.

    Yields:
      nothing: No value; used as a context manager.

    """
    before = (_FLAGS.golden, _FLAGS.b4b)
    _FLAGS.golden = _FLAGS.b4b = False
    try:
        yield
    finally:
        _FLAGS.golden, _FLAGS.b4b = before


@contextmanager
def forced() -> Generator[None]:
    """Turn the tensor-golden flag on inside the block, then restore it.

    Yields:
      nothing: No value; used as a context manager.

    """
    before = _FLAGS.b4b
    _FLAGS.b4b = True
    try:
        yield
    finally:
        _FLAGS.b4b = before
