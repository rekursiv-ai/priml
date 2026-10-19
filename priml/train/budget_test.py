"""Cold-time, durable publication, exception and real-alarm budget contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json
import math
import os
import signal
import stat
import time

from treekle import from_plain, json_loads, to_plain

import pytest

from priml.train.budget import WallBudget


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from types import FrameType


class Clock:
    """Advance only at explicit workload or publication boundaries."""

    def __init__(self) -> None:
        self.now: float = 40

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def alarm_scope() -> Generator[None]:
    """Restore the process alarm even when a deliberately broken test fails."""
    previous = signal.getsignal(signal.SIGALRM)
    timer, interval = signal.getitimer(signal.ITIMER_REAL)
    before = time.perf_counter()
    signal.setitimer(signal.ITIMER_REAL, 0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        signal.setitimer(
            signal.ITIMER_REAL,
            max(1e-6, timer - (time.perf_counter() - before)) if timer else 0,
            interval,
        )


def test_config_is_printable_serializable_and_does_no_io(tmp_path: Path) -> None:
    config = WallBudget.Config()
    config.receipt_path = tmp_path / "literal-{experiment_name}.json"
    restored = json_loads(
        json.dumps(to_plain(config)),
        WallBudget.Config,
        allow_imports=True,
    )
    assert restored == config
    assert restored.clock is time.perf_counter
    assert "reserve_evaluation_sec" in config.pformat(hide_default_values=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("seconds", [0, -1, -math.inf, math.nan])
def test_reject_invalid_seconds(seconds: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        WallBudget.Config(seconds=seconds).make()


@pytest.mark.parametrize("reserve", [-1, math.inf, math.nan, 10, 11])
def test_reject_invalid_reserve(reserve: float) -> None:
    with pytest.raises(ValueError, match="reserve"):
        WallBudget.Config(seconds=10, reserve_evaluation_sec=reserve).make()


def test_cold_preparation_is_charged_before_entry(tmp_path: Path) -> None:
    clock = Clock()
    runtime = WallBudget.Config(
        seconds=10,
        reserve_evaluation_sec=2,
        clock=clock,
        receipt_path=tmp_path / "receipt.json",
    ).make()
    clock.now += 3
    with runtime as entered:
        assert entered is runtime
        assert runtime.started == 40
        assert runtime.elapsed == 3
        assert runtime.remaining == 7
        assert runtime.training_seconds() == 5
        runtime.metadata.update(score=0.75, qualified=False, rows=["a", "b"])
        clock.now += 2
    record = json_loads((tmp_path / "receipt.json").read_text(), dict[str, object])
    assert record == {
        "status": "complete",
        "budget_seconds": 10,
        "elapsed_seconds": 5,
        "metadata": {"score": 0.75, "qualified": False, "rows": ["a", "b"]},
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == ["receipt.json"]


def test_infinite_allowance_and_optional_receipt() -> None:
    clock = Clock()
    runtime = WallBudget.Config(clock=clock).make()
    clock.now += 1e10
    with runtime:
        assert runtime.remaining == math.inf
        assert runtime.training_seconds() == math.inf
        runtime.check()


def test_training_reserve_and_inclusive_deadline() -> None:
    clock = Clock()
    runtime = WallBudget.Config(
        seconds=10,
        reserve_evaluation_sec=2,
        clock=clock,
    ).make()
    clock.now += 8
    with pytest.raises(TimeoutError, match="training"):
        runtime.training_seconds()
    clock.now += 2
    runtime.check()
    clock.now += 0.25
    assert runtime.remaining == -0.25
    with pytest.raises(TimeoutError, match="exceeded"):
        runtime.check()


def test_expired_work_persists_failure(tmp_path: Path) -> None:
    clock = Clock()
    runtime = WallBudget.Config(
        seconds=2,
        clock=clock,
        receipt_path=tmp_path / "receipt.json",
    ).make()
    with pytest.raises(TimeoutError), runtime:
        clock.now += 3
    record = json_loads((tmp_path / "receipt.json").read_text(), dict[str, object])
    assert record["status"] == "failed"
    assert record["elapsed_seconds"] == 3
    assert "TimeoutError" in str(record["error"])


@pytest.mark.parametrize("publication", ["replace", "fsync"])
def test_publication_and_durable_seal_are_charged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication: str,
) -> None:
    clock = Clock()
    path = tmp_path / "receipt.json"
    replace = os.replace
    fsync = os.fsync

    def slow_replace(source: str | Path, destination: str | Path) -> None:
        replace(source, destination)
        clock.now += 3

    def slow_fsync(fd: int) -> None:
        fsync(fd)
        clock.now += 3

    monkeypatch.setattr(
        os,
        publication,
        slow_replace if publication == "replace" else slow_fsync,
    )
    runtime = WallBudget.Config(seconds=2, clock=clock, receipt_path=path).make()
    with pytest.raises(TimeoutError), runtime:
        pass
    record = json_loads(path.read_text(), dict[str, object])
    assert record["status"] == "failed"
    assert from_plain(record["elapsed_seconds"], float) >= 3
    assert "TimeoutError" in str(record["error"])
    assert sorted(item.name for item in tmp_path.iterdir()) == ["receipt.json"]


@pytest.mark.parametrize("error", [ValueError("workload"), KeyboardInterrupt()])
def test_workload_exception_identity_survives_expiry(
    tmp_path: Path,
    error: BaseException,
) -> None:
    clock = Clock()
    runtime = WallBudget.Config(
        seconds=2,
        clock=clock,
        receipt_path=tmp_path / "receipt.json",
    ).make()
    clock.now += 3
    with pytest.raises(type(error)) as caught, runtime:
        raise error
    assert caught.value is error
    record = json_loads((tmp_path / "receipt.json").read_text(), dict[str, object])
    assert record["status"] == "failed"
    assert record["error"] == f"{type(error).__name__}: {error}"


def test_receipt_failure_preserves_workload_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ValueError("workload")

    def unavailable(source: str | Path, destination: str | Path) -> None:
        del source, destination
        raise OSError("publication unavailable")

    monkeypatch.setattr(os, "replace", unavailable)
    runtime = WallBudget.Config(receipt_path=tmp_path / "receipt.json").make()
    with pytest.raises(ValueError, match="workload") as caught, runtime:
        raise error
    assert caught.value is error
    assert any("publication unavailable" in note for note in error.__notes__)
    assert list(tmp_path.iterdir()) == []


def test_failed_rewrite_removes_expired_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    original_replace = os.replace
    calls = 0

    def unavailable_rewrite(source: str | Path, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise OSError("failure receipt unavailable")
        original_replace(source, destination)
        clock.now += 3

    monkeypatch.setattr(os, "replace", unavailable_rewrite)
    runtime = WallBudget.Config(
        seconds=2,
        clock=clock,
        receipt_path=tmp_path / "receipt.json",
    ).make()
    with pytest.raises(TimeoutError) as caught, runtime:
        pass
    assert any("failure receipt unavailable" in note for note in caught.value.__notes__)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("alarm_scope")
def test_interruptible_deadline_interrupts_blocking_work(tmp_path: Path) -> None:
    runtime = WallBudget.Config(
        seconds=0.005,
        receipt_path=tmp_path / "receipt.json",
    ).make()
    before = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError), runtime, runtime.interruptible():
        time.sleep(0.05)
    assert signal.getsignal(signal.SIGALRM) is before
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)
    assert (
        json_loads((tmp_path / "receipt.json").read_text(), dict[str, object])["status"]
        == "failed"
    )


@pytest.mark.usefixtures("alarm_scope")
@pytest.mark.parametrize("fail", [False, True])
def test_interruptible_restores_caller_handler_and_elapsed_timer(fail: bool) -> None:
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    signal.setitimer(signal.ITIMER_REAL, 10, 2)
    runtime = WallBudget.Config(seconds=5).make()
    try:
        with runtime.interruptible():
            assert signal.getitimer(signal.ITIMER_REAL)[0] <= 5
            _workload(fail)
    except LookupError:
        assert fail
    assert signal.getsignal(signal.SIGALRM) == signal.SIG_IGN
    delay, interval = signal.getitimer(signal.ITIMER_REAL)
    assert 9 < delay < 10
    assert interval == 2


@pytest.mark.usefixtures("alarm_scope")
def test_earlier_caller_alarm_runs_with_original_frame_then_budget_still_expires() -> (
    None
):
    calls: list[tuple[int, FrameType | None]] = []

    def caller(signum: int, frame: FrameType | None) -> None:
        calls.append((signum, frame))

    signal.signal(signal.SIGALRM, caller)
    runtime = WallBudget.Config(seconds=0.025).make()
    signal.setitimer(signal.ITIMER_REAL, 0.003, 0.004)
    with pytest.raises(TimeoutError), runtime.interruptible():
        time.sleep(0.06)
    assert len(calls) >= 2
    assert all(
        signum == signal.SIGALRM and frame is not None for signum, frame in calls
    )
    assert signal.getsignal(signal.SIGALRM) is caller
    delay, interval = signal.getitimer(signal.ITIMER_REAL)
    assert 0 < delay <= 0.004
    assert interval == 0.004


@pytest.mark.usefixtures("alarm_scope")
def test_infinite_interruptible_context_leaves_caller_alarm_untouched() -> None:
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    signal.setitimer(signal.ITIMER_REAL, 10, 2)
    with WallBudget.Config().make().interruptible():
        assert signal.getsignal(signal.SIGALRM) == signal.SIG_IGN
        delay, interval = signal.getitimer(signal.ITIMER_REAL)
        assert 9 < delay <= 10
        assert interval == 2


def test_receipt_is_literal_and_file_then_directory_are_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original = os.fsync

    def observed_fsync(fd: int) -> None:
        events.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        original(fd)

    monkeypatch.setattr(os, "fsync", observed_fsync)
    path = tmp_path / "nested" / "{literal}.json"
    with WallBudget.Config(receipt_path=path).make():
        pass
    assert events == ["file", "directory"]
    record = json_loads(path.read_text(), dict[str, object])
    assert record["budget_seconds"] == math.inf
    assert record["status"] == "complete"


def test_accounting_context_is_single_use() -> None:
    runtime = WallBudget.Config().make()
    with (
        runtime,
        pytest.raises(
            RuntimeError,
            match=r"^A WallBudget context can only be entered once\.$",
        ),
        runtime,
    ):
        pass
    with (
        pytest.raises(
            RuntimeError,
            match=r"^A WallBudget context can only be entered once\.$",
        ),
        runtime,
    ):
        pass


@pytest.mark.usefixtures("alarm_scope")
def test_expired_interruptible_entry_preserves_existing_alarm() -> None:
    clock = Clock()
    runtime = WallBudget.Config(seconds=2, clock=clock).make()
    clock.now += 3
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    signal.setitimer(signal.ITIMER_REAL, 10, 2)
    with pytest.raises(TimeoutError), runtime.interruptible():
        pass
    assert signal.getsignal(signal.SIGALRM) == signal.SIG_IGN
    delay, interval = signal.getitimer(signal.ITIMER_REAL)
    assert 9 < delay <= 10
    assert interval == 2


def test_receipt_path_failure_does_not_replace_workload_error(tmp_path: Path) -> None:
    runtime = WallBudget.Config(receipt_path=tmp_path / ".." / "invalid.json").make()
    error = ValueError("workload")
    with pytest.raises(ValueError, match="workload") as caught, runtime:
        raise error
    assert caught.value is error
    assert any("publication failed" in note for note in error.__notes__)
    assert any("invalidation failed" in note for note in error.__notes__)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("alarm_scope")
def test_earlier_default_alarm_keeps_default_signal_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []

    def terminate(signum: int) -> None:
        observed.append(signum)
        assert signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
        raise LookupError("caller default action")

    monkeypatch.setattr(signal, "raise_signal", terminate)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    runtime = WallBudget.Config(seconds=1).make()
    signal.setitimer(signal.ITIMER_REAL, 0.003)
    with (
        pytest.raises(LookupError, match="caller default action"),
        runtime.interruptible(),
    ):
        time.sleep(0.02)
    assert observed == [signal.SIGALRM]
    assert signal.getsignal(signal.SIGALRM) == signal.SIG_DFL
    assert signal.getitimer(signal.ITIMER_REAL) == (0, 0)


def _workload(fail: bool) -> None:
    if fail:
        raise LookupError("body")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
