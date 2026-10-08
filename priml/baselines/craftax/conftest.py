"""Compile the game's Numba kernels once per session, before any test starts.

A kernel compiles on its first call unless Numba's cache holds it
(``game/jit.py``), so with a cold cache the first test to reach one paid the
compile inside its own 60 s timeout: ``ghosts/build_test.py``'s fixtures took
over 60 s on a 4-CPU GitHub runner. These hooks run ``kernel_cache.py`` in a
child process first. The child only fills the cache, so its compile time counts
against no test's timeout and its lines against no coverage report. Every test
process then loads the kernels from that cache.

Where it runs follows where pytest loads this file. A run whose arguments name a
path under this package loads it at startup in every process: the xdist
controller, or a serial run, compiles before any worker starts. A run that
reaches this package only by collecting it, as a bare ``pytest -n 2`` does,
loads it in each worker, and each compiles before its first test.
"""

from __future__ import annotations

from typing import Final

import os
import subprocess
import sys

import pytest

from priml.lib.codec import from_plain


_SEEN_AT_START: Final = pytest.StashKey[bool]()
"""Set once this conftest saw the session start: its kernels are compiled."""


def pytest_sessionstart(session: pytest.Session) -> None:
    """Compile on the xdist controller or a serial run that loaded this file at startup."""
    session.config.stash[_SEEN_AT_START] = True
    if "PYTEST_XDIST_WORKER" not in os.environ:
        _compile(session.config)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Compile in a process that loaded this file only while collecting."""
    if not session.config.stash.get(_SEEN_AT_START, False):
        _compile(session.config)


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
