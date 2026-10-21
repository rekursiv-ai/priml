"""Tests for ``priml.lib.testing.checks``."""

from __future__ import annotations

from functools import partial

import signal
import subprocess
import sys

import pytest

from priml.lib.testing import checks
from priml.lib.testing.checks import (
    assert_help_without_docstring,
    default_signals,
    smuggle,
)


@pytest.mark.cli_python_subprocess
@pytest.mark.parametrize(("prefixed", "seen"), [(True, "default"), (False, "ignored")])
def test_a_child_starts_with_the_signal_at_its_default_action(
    *,
    prefixed: bool,
    seen: str,
) -> None:
    # Unprefixed is the control: it proves the parent below really hands its
    # children SIGINT ignored, as a suite started with ``nohup ... &`` does.
    child = [
        *(default_signals(signal.SIGINT) if prefixed else []),
        sys.executable,
        "-c",
        (
            "import signal\n"
            "ignored = signal.getsignal(signal.SIGINT) is signal.SIG_IGN\n"
            "print('ignored' if ignored else 'default')\n"
        ),
    ]
    parent = (
        "import signal, subprocess, sys\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "subprocess.run(sys.argv[1:], check=True)\n"
    )

    completed = subprocess.run(  # noqa: S603 -- fixed interpreter and inline scripts.
        [sys.executable, "-c", parent, *child],
        capture_output=True,
        check=True,
        text=True,
    )

    assert completed.stdout.strip() == seen


@pytest.mark.cli_python_subprocess
def test_the_command_keeps_the_pid_it_was_started_with() -> None:
    # Callers signal the child's pid and process group, so the reset must exec
    # the command in place rather than run it as a grandchild.
    process = subprocess.Popen(  # noqa: S603 -- fixed interpreter and inline script.
        [
            *default_signals(signal.SIGHUP),
            sys.executable,
            "-c",
            "import os; print(os.getpid())",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    stdout, _ = process.communicate(timeout=60)

    assert int(stdout) == process.pid


def test_the_prefix_runs_this_interpreter() -> None:
    # What the prefix DOES is the subprocess tests above; this pins the shape
    # they rely on, in the tier that runs on every commit.
    assert default_signals(signal.SIGINT, signal.SIGHUP)[:2] == [sys.executable, "-c"]


def test_naming_no_signal_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one signal"):
        default_signals()


def test_help_that_exits_zero_without_a_docstring_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_help_without_docstring(monkeypatch, checks, partial(sys.exit, 0))
    assert checks.__doc__ is None


def test_help_that_fails_without_a_docstring_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match=r"^Expected exit_info\.value\.code == 0\.$"):
        assert_help_without_docstring(monkeypatch, checks, partial(sys.exit, 2))


def test_smuggle_returns_the_value_itself() -> None:
    value = object()
    assert smuggle(value, int) is value


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
