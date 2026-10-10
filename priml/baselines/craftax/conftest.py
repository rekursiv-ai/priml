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
from typing import TYPE_CHECKING, Final

import os
import subprocess
import sys

import pytest

from priml.lib.codec import from_plain


if TYPE_CHECKING:
    from collections.abc import Iterable


_CWD: Final = Path(__file__).resolve().parent

_SEEN_AT_START: Final = pytest.StashKey[bool]()
"""Set once this conftest saw the session start; a worker's controller then compiled."""


def pytest_sessionstart(session: pytest.Session) -> None:
    """Compile on the xdist controller, when it loaded this file at startup."""
    session.config.stash[_SEEN_AT_START] = True
    if session.config.pluginmanager.has_plugin("dsession"):
        _compile(session.config)


def pytest_collection_finish(session: pytest.Session) -> None:
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
def _needs_kernels(items: Iterable[pytest.Item]) -> bool:
    """Whether a selected test of this package may call a compiled kernel."""
    return any(
        item.path.is_relative_to(_CWD)
        and item.get_closest_marker("integration") is None
        and item.get_closest_marker("skip") is None
        for item in items
    )


def _compile(config: pytest.Config) -> None:
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
