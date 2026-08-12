"""Tests for the distributed WorkerPool lifecycle (INF-007)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock, patch

import functools
import queue as queue_mod
import socket
import tempfile
import time

import pytest

from priml.distributed.testing import WorkerPool


if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

    from torch.distributed.device_mesh import DeviceMesh


def _pool(world_size: int) -> WorkerPool:
    """Build a WorkerPool whose mesh has ``world_size`` ranks."""
    return WorkerPool(WorkerPool.Config(mesh_dims={"r": world_size}))


def test_enter_kills_started_children_when_a_later_start_fails() -> None:
    started: list[MagicMock] = []

    def _make_process(*_args: Any, **_kwargs: Any) -> MagicMock:
        proc = MagicMock()
        # Fail every third spawn; each attempt's first two must be killed on
        # cleanup before the next retry starts a fresh batch.
        if len(started) % 3 >= 2:
            proc.start.side_effect = OSError("cannot fork")
        started.append(proc)
        return proc

    # The pool builds its children through a spawn context
    # (``tm.get_context("spawn")``), so patch that context's Process/Queue
    # rather than the top-level ``tm.Process``/``tm.Queue``.
    fake_ctx = MagicMock()
    fake_ctx.Process = _make_process
    fake_ctx.Queue = MagicMock()
    with (
        patch(
            "priml.distributed.testing.tm.get_context",
            return_value=fake_ctx,
        ),
        # A persistent spawn failure exhausts the retries and surfaces as a
        # rendezvous error chaining the underlying OSError.
        pytest.raises(RuntimeError, match="failed to rendezvous"),
    ):
        _pool(5).__enter__()

    # The two started children of the first (failed) attempt were cleaned up.
    assert started[0].kill.called
    assert started[1].kill.called


def test_terminate_force_kills_wedged_child_after_join_timeout() -> None:
    wedged = MagicMock()
    wedged.is_alive.return_value = True
    healthy = MagicMock()
    healthy.is_alive.return_value = False

    pool = _pool(2)
    pool.queue = MagicMock()
    pool.processes = cast("list[BaseProcess]", [healthy, wedged])

    pool.terminate()

    # Bounded join, then force-kill the still-alive child.
    assert wedged.join.call_args_list[0].kwargs["timeout"] > 0
    wedged.kill.assert_called_once()
    healthy.kill.assert_not_called()


def _sentinel_worker(result_dir_str: str, mesh: DeviceMesh) -> None:
    """Sleep, then write a per-rank sentinel file.

    The sleep makes a fire-and-forget dispatch observably lose the race: the
    sentinel would be absent the instant ``pool(fn)`` returned.
    """
    time.sleep(0.01)
    (Path(result_dir_str) / f"rank_{mesh.get_rank()}").write_text("done")


@pytest.mark.integration
def test_call_blocks_until_worker_finishes() -> None:
    """``pool(fn)`` returns only after the worker has finished ``fn``."""
    with (
        tempfile.TemporaryDirectory() as tmp,
        WorkerPool.Config(mesh_dims={"r": 1}).make() as pool,
    ):
        pool(functools.partial(_sentinel_worker, tmp))
        # The file must already exist: the call blocked on the worker's ack,
        # so a fire-and-forget dispatch would see an empty dir here.
        written = sorted(p.name for p in Path(tmp).iterdir() if p.is_file())
        assert written == ["rank_0"], written


def test_call_retries_then_raises_when_worker_keeps_dying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead reused worker triggers respawn+retry; persistent death then raises.

    A transient gloo peer-close / OOM kill must not fail the test on the first
    dispatch -- the pool respawns and re-dispatches. Only when every respawn
    still loses a worker (here ``_respawn`` is stubbed to a no-op so the same
    dead mock persists) does dispatch give up, after ``_DISPATCH_ATTEMPTS`` tries.
    """
    proc = MagicMock()
    proc.exitcode = 1
    proc.is_alive.return_value = False
    pool = _pool(1)
    pool.queue = MagicMock()
    pool.ack_queue = MagicMock()
    pool.ack_queue.get.side_effect = queue_mod.Empty
    pool.processes = cast("list[BaseProcess]", [proc])
    monkeypatch.setattr(WorkerPool, "_READY_POLL_SEC", 0.0)
    respawns: list[int] = []

    def _record_respawn(_self: WorkerPool) -> None:
        respawns.append(1)

    monkeypatch.setattr(WorkerPool, "_respawn", _record_respawn)

    with pytest.raises(RuntimeError, match="failed dispatch"):
        pool(functools.partial(_sentinel_worker, "unused"))

    # Respawned once per failed attempt except the last (which raises).
    assert len(respawns) == WorkerPool._DISPATCH_ATTEMPTS - 1


def test_call_retries_then_raises_on_persistent_ack_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live-but-unacked worker (CPU-starved past the deadline) also respawns.

    Under a loaded CI runner rank 0 can miss the ack deadline while still alive,
    so ``_await_dispatch_ack`` raises ``TimeoutError`` rather than
    ``_WorkerDiedError``. That must also trigger respawn-and-retry, not fail the
    dispatch outright; only persistent timeouts give up after
    ``_DISPATCH_ATTEMPTS`` tries.
    """
    proc = MagicMock()
    proc.exitcode = None  # alive: never produces a _WorkerDiedError
    proc.is_alive.return_value = True
    pool = _pool(1)
    pool.queue = MagicMock()
    pool.ack_queue = MagicMock()
    pool.ack_queue.get.side_effect = queue_mod.Empty  # never acks -> timeout
    pool.processes = cast("list[BaseProcess]", [proc])
    # Make the ack deadline elapse immediately so the test does not sleep.
    monkeypatch.setattr(WorkerPool, "_RENDEZVOUS_TIMEOUT", timedelta(0))
    monkeypatch.setattr(WorkerPool, "_READY_POLL_SEC", 0.0)

    def _noop_kill(_self: WorkerPool, _procs: list[BaseProcess]) -> None:
        return None

    monkeypatch.setattr(WorkerPool, "_kill_all", _noop_kill)
    respawns: list[int] = []

    def _record_respawn(_self: WorkerPool) -> None:
        respawns.append(1)

    monkeypatch.setattr(WorkerPool, "_respawn", _record_respawn)

    with pytest.raises(RuntimeError, match="failed dispatch"):
        pool(functools.partial(_sentinel_worker, "unused"))

    assert len(respawns) == WorkerPool._DISPATCH_ATTEMPTS - 1


@pytest.mark.integration
def test_enter_recovers_from_a_port_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A port stolen in the find_free_port TOCTOU window is re-picked, not hung.

    Reproduces the intermittent-timeout root cause deterministically: the first
    candidate port is already bound by a live socket. Without the in-parent bind
    check, rank 0's TCPStore would refuse to listen and every client rank would
    burn the full rendezvous timeout retrying (read by pytest as a bare
    ``Timeout (>60s)``). With it the parent detects the collision locally in
    microseconds, picks another port, and the pool comes up. A real dispatch
    afterwards proves the recovered pool is fully functional, not just built.
    """
    # A real, OS-assigned port held open for the whole test, so the parent's
    # bind probe must reject it and pick another.
    dead = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    dead.bind(("127.0.0.1", 0))
    dead_port = dead.getsockname()[1]
    assert isinstance(dead_port, int)
    real_find_free_port = WorkerPool.find_free_port

    ports: list[int] = []

    def _colliding_then_free() -> int:
        port = dead_port if not ports else real_find_free_port()
        ports.append(port)
        return port

    monkeypatch.setattr(
        WorkerPool, "find_free_port", staticmethod(_colliding_then_free)
    )

    try:
        with (
            tempfile.TemporaryDirectory() as tmp,
            WorkerPool.Config(mesh_dims={"r": 2}).make() as pool,
        ):
            pool(functools.partial(_sentinel_worker, tmp))
            written = sorted(p.name for p in Path(tmp).iterdir() if p.is_file())
            assert written == ["rank_0", "rank_1"], written
    finally:
        dead.close()

    # The first candidate was the collided port; the probe rejected it and the
    # next pick was a genuinely free one.
    assert ports[0] == dead_port
    assert len(ports) >= 2


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
