"""Tests for the checkpointable step timer.

The class is small, but what it measures is the number a run STOPS on, so
each case pins one property a caller would otherwise have to take on faith:
what a block charges, what survives a resume, and what deliberately does not.
"""

from __future__ import annotations

import time

import pytest

from priml.timer import CheckpointableStepTimer


def test_a_block_is_counted_and_charged() -> None:
    timer = CheckpointableStepTimer()
    with timer:
        time.sleep(0.001)
    assert timer.global_count == 1
    assert timer.local_count == 1
    assert timer.global_sec > 0.0
    assert timer.local_sec == timer.global_sec


def test_blocks_accumulate() -> None:
    timer = CheckpointableStepTimer()
    for _ in range(3):
        with timer:
            pass
    assert timer.global_count == 3
    assert timer.local_count == 3


def test_a_call_that_raised_is_still_counted() -> None:
    """It happened, and hiding it would misreport the very step being debugged."""
    timer = CheckpointableStepTimer()
    with pytest.raises(ValueError, match="boom"), timer:
        raise ValueError("boom")
    assert timer.global_count == 1


def test_a_nested_block_is_refused_rather_than_double_counted() -> None:
    """One timer measures one activity, so re-entering it is a wiring error.

    Silently allowed, the inner block would be charged twice and counted
    twice -- and the number that misreports is the budget the run stops on.
    """
    timer = CheckpointableStepTimer()
    with timer, pytest.raises(RuntimeError, match="already running"), timer:
        pass


def test_a_timer_is_saved_closed() -> None:
    """Its start is a reading from THIS process's clock, so it cannot travel."""
    timer = CheckpointableStepTimer()
    with timer:
        pass
    assert set(timer.state_dict()) == {"global_count", "global_sec"}


def test_a_resume_keeps_the_lifetime_totals_and_restarts_the_session_ones() -> None:
    """The pair separates this job's work from the whole run's.

    A warmup exclusion or a first-step branch reads the local pair; a schedule
    and a stop condition read the global one. Seconds especially cannot carry:
    they come from a monotonic clock whose origin is this process's own, so a
    figure restored from another one measures nothing.
    """
    source = CheckpointableStepTimer(
        global_count=5,
        local_count=5,
        global_sec=12.5,
        local_sec=12.5,
    )

    target = CheckpointableStepTimer()
    target.load_state_dict(source.state_dict())
    assert target.global_count == 5
    assert target.global_sec == pytest.approx(12.5)
    assert target.local_count == 0
    assert target.local_sec == 0.0


def test_a_restored_timer_keeps_counting_from_the_lifetime_total() -> None:
    """Resuming must extend the run's count, not restart it."""
    source = CheckpointableStepTimer(global_count=5, global_sec=12.5)
    target = CheckpointableStepTimer()
    target.load_state_dict(source.state_dict())
    with target:
        pass
    assert target.global_count == 6
    assert target.local_count == 1


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
