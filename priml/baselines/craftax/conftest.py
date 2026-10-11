"""Compile the game's Numba kernels once per session, before any test starts.

A kernel compiles on its first call unless Numba's cache holds it
(``game/jit.py``), so with a cold cache the first test to reach one paid the
compile inside its own 60 s timeout: ``ghosts/build_test.py``'s fixtures took
over 60 s on a 4-CPU GitHub runner. These hooks run ``kernel_cache.py`` in a
child process first. The child only fills the cache, so its compile time counts
against no test's timeout and its lines against no coverage report. Every test
process then loads the kernels from that cache.

Where it runs follows where pytest loads this file. A run whose arguments name a
path under this package loads it at startup in every process, and the xdist
controller, which collects no tests, compiles before any worker starts. Every
other process compiles once it has collected, before its first test: a serial
run, or each worker of a run that reaches this package only by collecting it,
as a bare ``pytest -n 2`` does.

Such a process compiles only if a test it selected may call a kernel
(:func:`_needs_kernels`), so an integration-tier run, whose tests here call
none, skips a compile of over two minutes on a GitHub runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import os
import subprocess
import sys

from treekle import from_plain

import pytest


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


_CWD: Final = Path(__file__).resolve().parent

_SEEN_AT_START: Final = pytest.StashKey[bool]()
"""Set once this conftest saw the session start; a worker's controller then compiled."""


class _PluginManager(Protocol):
    def has_plugin(self, name: str) -> bool: ...
    def get_plugin(self, name: str) -> object | None: ...


class _Config(Protocol):
    @property
    def stash(self) -> pytest.Stash: ...
    @property
    def pluginmanager(self) -> _PluginManager: ...
    @property
    def rootpath(self) -> Path: ...
    def getoption(self, name: str) -> object: ...


class _Item(Protocol):
    @property
    def path(self) -> Path: ...
    def get_closest_marker(self, name: str) -> pytest.Mark | None: ...


class _Session(Protocol):
    """What these hooks read of a ``pytest.Session``."""

    @property
    def config(self) -> _Config: ...
    @property
    def items(self) -> Sequence[_Item]: ...


def pytest_sessionstart(session: _Session) -> None:
    """Compile on the xdist controller, when it loaded this file at startup."""
    session.config.stash[_SEEN_AT_START] = True
    if session.config.pluginmanager.has_plugin("dsession"):
        _compile(session.config)


def pytest_collection_finish(session: _Session) -> None:
    """Compile once the tests are known, if one may call a kernel and no controller did."""
    if "PYTEST_XDIST_WORKER" in os.environ and session.config.stash.get(
        _SEEN_AT_START,
        False,
    ):
        return
    if _needs_kernels(session.items):
        _compile(session.config)


# A test skipped by a mark runs nothing. An integration test here drives a child
# process, node or a device, never a kernel in this process. Should one call a kernel
# the cache lacks, the kernel compiles at that call, as any kernel the child missed
# does: slower, inside the test's timeout, never wrong.
def _needs_kernels(items: Iterable[_Item]) -> bool:
    """Whether a selected test of this package may call a compiled kernel."""
    return any(
        item.path.is_relative_to(_CWD)
        and item.get_closest_marker("integration") is None
        and item.get_closest_marker("skip") is None
        for item in items
    )


def _compile(config: _Config) -> None:
    """Run ``compile_kernels`` in a child process; report a failure, not raise it."""
    if from_plain(config.getoption("collectonly"), bool):
        return
    # Without ``COVERAGE_PROCESS_*`` coverage's .pth cannot start in the child,
    # so the lines it runs are not counted as tested.
    ran = subprocess.run(
        [sys.executable, "-m", "priml.baselines.craftax.kernel_cache"],
        cwd=config.rootpath,
        env={
            name: value
            for name, value in os.environ.items()
            if not name.startswith("COVERAGE_PROCESS_")
        },
        capture_output=True,
        text=True,
        check=False,
    )
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if ran.returncode and isinstance(reporter, pytest.TerminalReporter):
        reporter.write_line(
            "craftax: compiling the kernels before the tests failed; each test "
            f"compiles what it calls.\n{ran.stderr[-2_000:]}",
            yellow=True,
        )
