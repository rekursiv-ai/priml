"""Tests for the distributed WorkerPool lifecycle (INF-007)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock, patch

import functools
import os
import pickle
import queue as queue_mod
import re
import socket
import tempfile
import time

from torch import (
    Tensor,
    multiprocessing as tm,
)

import pytest
import torch
import torch.distributed as dist

from priml.distributed import testing
from priml.distributed.testing import PoolWorker, WorkerPool, do_something


if TYPE_CHECKING:
    from multiprocessing.process import BaseProcess

    from torch import multiprocessing as tm
    from torch.distributed.device_mesh import DeviceMesh


def _pool(world_size: int) -> WorkerPool:
    """Build a WorkerPool whose mesh has ``world_size`` ranks."""
    return WorkerPool(WorkerPool.Config(mesh_dims={"r": world_size}))


def test_call_reports_exact_message_when_dispatch_attempts_are_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(WorkerPool, "_DISPATCH_ATTEMPTS", 0)
    pool = _pool(1)
    pool.queue = cast("tm.Queue[bytes | None]", MagicMock())
    pool.ack_queue = cast("tm.Queue[bool]", MagicMock())
    pool.processes = []

    with pytest.raises(
        RuntimeError,
        match=(
            r"^worker pool \{'r': 1\} failed dispatch across 0 attempts "
            r"\(last: NoneType\)$"
        ),
    ) as error:
        pool(do_something)

    assert error.value.__cause__ is None


def test_respawn_reports_exact_message_when_rendezvous_attempts_are_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(WorkerPool, "_RENDEZVOUS_ATTEMPTS", 0)

    with pytest.raises(
        RuntimeError,
        match=r"^worker pool \{'r': 1\} failed to rendezvous$",
    ) as error:
        _pool(1)._respawn()

    assert error.value.__cause__ is None


def test_worker_pool_starts_with_no_process_or_queue() -> None:
    pool = _pool(2)

    assert pool.mesh_dims == {"r": 2}
    assert pool.processes is None
    assert pool.queue is None
    assert pool.ack_queue is None


def test_enter_kills_started_children_when_a_later_start_fails() -> None:
    started: list[MagicMock] = []
    start_errors: list[OSError] = []

    def _make_process(*_args: object, **_kwargs: object) -> MagicMock:
        proc = MagicMock()
        # Fail every third spawn; each attempt's first two must be killed on
        # cleanup before the next retry starts a fresh batch.
        if len(started) % 3 >= 2:
            error = OSError("cannot fork")
            start_errors.append(error)
            proc.start.side_effect = error
        started.append(proc)
        return proc

    # The pool builds its children through a spawn context
    # (``tm.get_context("spawn")``), so patch that context's Process/Queue
    # rather than the top-level ``tm.Process``/``tm.Queue``.
    fake_ctx = MagicMock()
    fake_ctx.Process = _make_process
    fake_ctx.Queue = MagicMock()
    pool = _pool(5)
    with (
        patch(
            "priml.distributed.testing.tm.get_context",
            return_value=fake_ctx,
        ),
        # A persistent spawn failure exhausts the retries and surfaces as a
        # rendezvous error chaining the underlying OSError.
        pytest.raises(
            RuntimeError,
            match=r"^worker pool \{'r': 5\} failed to rendezvous$",
        ) as error,
    ):
        pool.__enter__()

    assert error.value.__cause__ is start_errors[-1]
    assert len(start_errors) == WorkerPool._RENDEZVOUS_ATTEMPTS
    assert len(started) == 3 * WorkerPool._RENDEZVOUS_ATTEMPTS
    for attempt in range(WorkerPool._RENDEZVOUS_ATTEMPTS):
        assert all(started[3 * attempt + rank].kill.called for rank in range(2))
        assert not started[3 * attempt + 2].kill.called
    assert pool.processes is None
    assert pool.queue is None
    assert pool.ack_queue is None


def test_terminate_force_kills_wedged_child_after_join_timeout() -> None:
    wedged = MagicMock()
    wedged.is_alive.return_value = True
    healthy = MagicMock()
    healthy.is_alive.return_value = False

    pool = _pool(2)
    pool.queue = cast("tm.Queue[bytes | None]", MagicMock())
    pool.processes = [cast(PoolWorker, healthy), cast(PoolWorker, wedged)]

    pool.terminate()

    # Bounded join, then force-kill the still-alive child.
    timeout = wedged.join.call_args_list[0].kwargs["timeout"]
    assert isinstance(timeout, (int, float))
    assert timeout > 0
    wedged.kill.assert_called_once()
    healthy.kill.assert_not_called()


# The sleep makes a fire-and-forget dispatch observably lose the race: the sentinel
# would be absent the instant ``pool(fn)`` returned.
def _sentinel_worker(result_dir_str: str, mesh: DeviceMesh) -> None:
    """Sleep, then write a per-rank sentinel file."""
    time.sleep(0.01)
    (Path(result_dir_str) / f"rank_{mesh.get_rank()}").write_text("done")


@pytest.mark.compute_distributed
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
    pool.queue = cast("tm.Queue[bytes | None]", MagicMock())
    pool.ack_queue = cast("tm.Queue[bool]", MagicMock())
    assert pool.ack_queue is not None
    cast(MagicMock, pool.ack_queue).get.side_effect = queue_mod.Empty
    pool.processes = [cast(PoolWorker, proc)]
    monkeypatch.setattr(WorkerPool, "_READY_POLL_SEC", 0.0)
    respawns: list[int] = []

    def _record_respawn(self: WorkerPool) -> None:
        del self
        respawns.append(1)

    monkeypatch.setattr(WorkerPool, "_respawn", _record_respawn)

    with pytest.raises(
        RuntimeError,
        match=r"^worker pool \{'r': 1\} failed dispatch across 3 attempts \(last: _WorkerDiedError\)$",
    ):
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
    proc.exitcode = None  # Alive: never produces a _WorkerDiedError.
    proc.is_alive.return_value = True
    pool = _pool(1)
    pool.queue = cast("tm.Queue[bytes | None]", MagicMock())
    pool.ack_queue = cast("tm.Queue[bool]", MagicMock())
    cast(
        MagicMock,
        pool.ack_queue,
    ).get.side_effect = queue_mod.Empty  # Never acks -> timeout.
    pool.processes = [cast(PoolWorker, proc)]
    # Make the ack deadline elapse immediately so the test does not sleep.
    monkeypatch.setattr(WorkerPool, "_RENDEZVOUS_TIMEOUT", timedelta(0))
    monkeypatch.setattr(WorkerPool, "_READY_POLL_SEC", 0.0)

    def _noop_kill(self: WorkerPool, procs: list[BaseProcess]) -> None:
        del self, procs

    monkeypatch.setattr(WorkerPool, "_kill_all", _noop_kill)
    respawns: list[int] = []

    def _record_respawn(self: WorkerPool) -> None:
        del self
        respawns.append(1)

    monkeypatch.setattr(WorkerPool, "_respawn", _record_respawn)

    with pytest.raises(
        RuntimeError,
        match=r"^worker pool \{'r': 1\} failed dispatch across 3 attempts \(last: TimeoutError\)$",
    ):
        pool(functools.partial(_sentinel_worker, "unused"))

    assert len(respawns) == WorkerPool._DISPATCH_ATTEMPTS - 1


@pytest.mark.compute_distributed
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
    dead_port = int(dead.getsockname()[1])  # pyright: ignore[reportAny] -- socket.getsockname is Any in typeshed; int() pins the port.
    real_find_free_port = WorkerPool.find_free_port

    ports: list[int] = []

    def _colliding_then_free() -> int:
        port = real_find_free_port() if ports else dead_port
        ports.append(port)
        return port

    monkeypatch.setattr(
        WorkerPool,
        "find_free_port",
        staticmethod(_colliding_then_free),
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


def test_enter_spawns_every_rank_and_exit_tears_them_down() -> None:
    started: list[MagicMock] = []
    process_specs: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    def make_process(
        *,
        target: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> MagicMock:
        process_specs.append((target, args, kwargs))
        rank = args[0]
        ready = kwargs["ready_queue"]
        assert isinstance(rank, int)
        assert isinstance(ready, queue_mod.Queue)
        ready_ranks = cast("queue_mod.Queue[int]", ready)
        proc = MagicMock()
        proc.exitcode = None
        proc.is_alive.return_value = False
        proc.start.side_effect = lambda: ready_ranks.put(rank)
        started.append(proc)
        return proc

    ctx = MagicMock()
    ctx.Process = make_process
    ctx.Queue = queue_mod.Queue
    pool = _pool(3)
    with (
        patch(
            "priml.distributed.testing.tm.get_context",
            return_value=ctx,
        ) as get_context,
        patch.object(WorkerPool, "_pick_bindable_port", return_value=23_456),
        patch.object(WorkerPool, "_await_ready", autospec=True) as await_ready,
    ):
        entered = pool.__enter__()
        assert entered is pool
        assert pool.processes is not None
        assert [id(p) for p in pool.processes] == [id(p) for p in started]
        queue = pool.queue
        ack_queue = pool.ack_queue
        assert queue is not None
        assert ack_queue is not None
        assert pool.__exit__(None, None, None) is False

    get_context.assert_called_once_with("spawn")
    await_ready.assert_called_once()
    assert await_ready.call_args.args[0] is pool
    assert await_ready.call_args.args[1] == started
    assert await_ready.call_args.args[2] is process_specs[0][2]["ready_queue"]
    assert await_ready.call_args.args[3] == 3
    assert len(process_specs) == 3
    for rank, (target, args, kwargs) in enumerate(process_specs):
        assert target == WorkerPool.worker
        assert args[:3] == (rank, {"r": 3}, 23_456)
        assert args[3] is queue
        assert args[4] is ack_queue
        assert kwargs["ready_queue"] is not None
    assert queue.get_nowait() is None
    assert pool.processes is None
    assert pool.queue is None
    assert pool.ack_queue is None
    for proc in started:
        proc.join.assert_called_once_with(timeout=WorkerPool._JOIN_TIMEOUT_SEC)


def test_call_returns_once_rank_zero_acks(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = MagicMock()
    proc.exitcode = None
    pool = _pool(1)
    commands: queue_mod.Queue[bytes | None] = queue_mod.Queue()
    acks: queue_mod.Queue[bool] = queue_mod.Queue()
    acks.put(True)
    pool.queue = cast("tm.Queue[bytes | None]", commands)
    pool.ack_queue = cast("tm.Queue[bool]", acks)
    pool.processes = [cast(PoolWorker, proc)]
    respawns: list[int] = []

    def record_respawn(self: WorkerPool) -> None:
        del self
        respawns.append(1)

    monkeypatch.setattr(WorkerPool, "_respawn", record_respawn)

    pool(functools.partial(_sentinel_worker, "unused"))

    dispatched = commands.get_nowait()
    assert dispatched is not None
    assert isinstance(pickle.loads(dispatched), functools.partial)
    assert respawns == []
    proc.kill.assert_not_called()


def test_rendezvous_times_out_when_no_rank_acks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alive = MagicMock()
    alive.exitcode = None
    monkeypatch.setattr(WorkerPool, "_RENDEZVOUS_TIMEOUT", timedelta(0))
    ready: queue_mod.Queue[int] = queue_mod.Queue()

    with pytest.raises(
        TimeoutError,
        match=r"^rendezvous timed out: 0/2 ranks ready$",
    ):
        _pool(2)._await_ready(
            [cast(PoolWorker, alive)],
            cast("tm.Queue[int]", ready),
            2,
        )


def test_rendezvous_fails_fast_when_a_rank_dies_before_acking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead rank surfaces at the next poll, not after the whole timeout."""
    dead = MagicMock()
    dead.exitcode = 1
    monkeypatch.setattr(WorkerPool, "_READY_POLL_SEC", 0.0)
    ready: queue_mod.Queue[int] = queue_mod.Queue()
    started = time.monotonic()

    with pytest.raises(
        RuntimeError,
        match=r"^a worker exited before rendezvous \(port collision\?\)$",
    ):
        _pool(1)._await_ready(
            [cast(PoolWorker, dead)],
            cast("tm.Queue[int]", ready),
            1,
        )

    assert time.monotonic() - started < WorkerPool._RENDEZVOUS_TIMEOUT.total_seconds()


def test_await_dispatch_ack_deadline_and_timeout_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _pool(1)
    ack_queue = MagicMock()
    ack_queue.get.side_effect = queue_mod.Empty
    pool.ack_queue = cast("tm.Queue[bool]", ack_queue)
    killed: list[list[PoolWorker]] = []

    def record_kill(self: WorkerPool, processes: list[PoolWorker]) -> None:
        del self
        killed.append(processes)

    monkeypatch.setattr(WorkerPool, "_kill_all", record_kill)
    monkeypatch.setattr(WorkerPool, "_READY_POLL_SEC", 0.1)
    ticks = iter((0.0, 0.0, 30.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    alive = MagicMock()
    alive.exitcode = None
    processes = [cast(PoolWorker, alive)]

    with pytest.raises(
        TimeoutError,
        match=r"^worker dispatch timed out waiting for rank 0 ack$",
    ):
        pool._await_dispatch_ack(processes)

    ack_queue.get.assert_called_once_with(timeout=0.1)
    assert killed == [processes]


def test_await_dispatch_ack_accepts_ack_inside_final_second(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _pool(1)
    ack_queue = MagicMock()
    ack_queue.get.return_value = True
    pool.ack_queue = cast("tm.Queue[bool]", ack_queue)
    monkeypatch.setattr(WorkerPool, "_RENDEZVOUS_TIMEOUT", timedelta(seconds=1))
    ticks = iter((0.0, 0.5))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))

    pool._await_dispatch_ack([])

    ack_queue.get.assert_called_once_with(timeout=0.1)


def test_await_ready_times_out_at_deadline_without_polling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _pool(1)
    monkeypatch.setattr(WorkerPool, "_RENDEZVOUS_TIMEOUT", timedelta(0))
    ticks = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    ready = MagicMock()
    ready.get.side_effect = queue_mod.Empty
    alive = MagicMock()
    alive.exitcode = None

    with pytest.raises(TimeoutError, match=r"^rendezvous timed out: 0/1 ranks ready$"):
        pool._await_ready([cast(PoolWorker, alive)], cast("tm.Queue[int]", ready), 1)

    ready.get.assert_not_called()


def test_await_ready_accepts_ack_with_less_than_one_second_remaining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(WorkerPool, "_RENDEZVOUS_TIMEOUT", timedelta(seconds=1))
    ticks = iter((0.0, 0.5))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    ready: queue_mod.Queue[int] = queue_mod.Queue()
    ready.put(0)

    _pool(1)._await_ready([], cast("tm.Queue[int]", ready), 1)

    assert ready.empty()


def test_await_ready_counts_each_rank_ack() -> None:
    ready: queue_mod.Queue[int] = queue_mod.Queue()
    ready.put(0)
    ready.put(1)
    ready.put(2)

    _pool(3)._await_ready([], cast("tm.Queue[int]", ready), 3)

    assert ready.empty()


def test_await_dispatch_ack_reports_dead_worker_and_kills_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dead = MagicMock()
    dead.exitcode = 1
    pool = _pool(2)
    pool.ack_queue = cast("tm.Queue[bool]", MagicMock())
    cast(MagicMock, pool.ack_queue).get.side_effect = queue_mod.Empty
    killed: list[list[PoolWorker]] = []

    def record_kill(self: WorkerPool, processes: list[PoolWorker]) -> None:
        del self
        killed.append(processes)

    monkeypatch.setattr(WorkerPool, "_kill_all", record_kill)
    monkeypatch.setattr(WorkerPool, "_READY_POLL_SEC", 0.0)
    processes = [cast(PoolWorker, dead)]

    with pytest.raises(RuntimeError, match=r"^worker exited before dispatch ack$"):
        pool._await_dispatch_ack(processes)

    assert killed == [processes]


def test_respawn_tears_down_the_degraded_pool_before_spawning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = MagicMock()
    stale.is_alive.return_value = False
    pool = _pool(1)
    pool.processes = [cast(PoolWorker, stale)]
    pool.queue = cast("tm.Queue[bytes | None]", MagicMock())
    order: list[str] = []

    def join(timeout: float | None = None) -> None:
        del timeout
        order.append("join")

    stale.join.side_effect = join

    def _spawn(self: WorkerPool) -> None:
        del self
        order.append("spawn")

    monkeypatch.setattr(WorkerPool, "_spawn_once", _spawn)

    pool._respawn()

    assert order == ["join", "spawn"]


@pytest.mark.parametrize(
    ("cuda_available", "expected_backend"),
    [(False, "gloo"), (True, "cpu:gloo,cuda:nccl")],
)
def test_worker_configures_cpu_mesh_and_acknowledges_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    expected_backend: str,
) -> None:
    mesh = MagicMock()
    mesh.get_rank.return_value = 0
    mesh_constructor = MagicMock(return_value=mesh)
    monkeypatch.setattr(testing, "DeviceMesh", mesh_constructor)
    dispatched = pickle.dumps(functools.partial(_sentinel_worker, str(tmp_path)))
    commands: queue_mod.Queue[bytes | None] = queue_mod.Queue()
    commands.put(dispatched)
    commands.put(None)
    ready: queue_mod.Queue[int] = queue_mod.Queue()
    acks: queue_mod.Queue[bool] = queue_mod.Queue()
    fake_dist = MagicMock()
    monkeypatch.setattr(testing, "td", fake_dist)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setenv("MASTER_ADDR", "before")
    monkeypatch.setenv("MASTER_PORT", "before")

    WorkerPool.worker(
        0,
        {"dp": 2, "tp": 3},
        23_456,
        cast("tm.Queue[bytes | None]", commands),
        cast("tm.Queue[bool]", acks),
        ready_queue=cast("tm.Queue[int]", ready),
    )

    assert (tmp_path / "rank_0").read_text() == "done"
    assert mesh_constructor.call_args.args == ("cpu",)
    mesh_kwargs = mesh_constructor.call_args.kwargs
    mesh_value = mesh_kwargs["mesh"]
    assert isinstance(mesh_value, Tensor)
    assert torch.equal(mesh_value, torch.tensor([[0, 1, 2], [3, 4, 5]]))
    assert mesh_kwargs["mesh_dim_names"] == ("dp", "tp")
    assert ready.get_nowait() == 0
    assert acks.get_nowait() is True
    assert fake_dist.init_process_group.call_args.kwargs == {
        "backend": expected_backend,
        "rank": 0,
        "world_size": 6,
        "timeout": WorkerPool._RENDEZVOUS_TIMEOUT,
    }
    assert fake_dist.broadcast_object_list.call_args_list[0].kwargs == {"src": 0}
    fake_dist.barrier.assert_called_once_with()
    fake_dist.destroy_process_group.assert_called_once_with()
    assert os.environ["MASTER_ADDR"] == "localhost"
    assert os.environ["MASTER_PORT"] == "23456"


def test_worker_runs_dispatched_fns_until_the_stop_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker body, run in-process at world size 1.

    It rendezvouses, acks readiness, runs each pickled fn against its mesh,
    acks the completion, and tears the group down on ``None``. CUDA is
    reported absent so the single-process group is gloo-only rather than
    constructing an NCCL backend inside the test process.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("MASTER_ADDR", "")
    monkeypatch.setenv("MASTER_PORT", "")
    commands: queue_mod.Queue[bytes | None] = queue_mod.Queue()
    commands.put(pickle.dumps(functools.partial(_sentinel_worker, str(tmp_path))))
    commands.put(None)
    acks: queue_mod.Queue[bool] = queue_mod.Queue()
    ready: queue_mod.Queue[int] = queue_mod.Queue()

    WorkerPool.worker(
        0,
        {"r": 1},
        WorkerPool.find_free_port(),
        cast("tm.Queue[bytes | None]", commands),
        cast("tm.Queue[bool]", acks),
        ready_queue=cast("tm.Queue[int]", ready),
    )

    assert ready.get_nowait() == 0
    assert acks.get_nowait() is True
    assert (tmp_path / "rank_0").read_text() == "done"
    assert not dist.is_initialized()


def test_worker_nonzero_rank_receives_commands_without_reading_queue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = MagicMock()
    mesh.get_rank.return_value = 1
    monkeypatch.setattr(testing, "DeviceMesh", MagicMock(return_value=mesh))
    dispatched = pickle.dumps(functools.partial(_sentinel_worker, str(tmp_path)))
    broadcasts = iter((dispatched, None))
    fake_dist = MagicMock()

    def broadcast(values: list[bytes | None], src: int) -> None:
        if src == 0:
            values[0] = next(broadcasts)

    fake_dist.broadcast_object_list.side_effect = broadcast
    monkeypatch.setattr(testing, "td", fake_dist)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    command_queue = MagicMock()
    command_queue.get.side_effect = AssertionError("nonzero rank read its queue")
    ready: queue_mod.Queue[int] = queue_mod.Queue()
    acks: queue_mod.Queue[bool] = queue_mod.Queue()

    WorkerPool.worker(
        1,
        {"r": 2},
        23_456,
        cast("tm.Queue[bytes | None]", command_queue),
        cast("tm.Queue[bool]", acks),
        ready_queue=cast("tm.Queue[int]", ready),
    )

    assert (tmp_path / "rank_1").read_text() == "done"
    command_queue.get.assert_not_called()
    assert ready.get_nowait() == 1
    assert acks.empty()


class _TpGroup:
    """Stands in for the process group a mesh dimension hands back."""


class _TpMesh:
    """The ``tp`` sub-mesh ``do_something`` reads."""

    def __init__(self, group: _TpGroup) -> None:
        self._group = group

    def get_group(self) -> _TpGroup:
        return self._group

    def size(self) -> int:
        return 3


class _FullMesh:
    """A 2x3 ``("dp", "tp")`` mesh as seen from ``rank``."""

    ndim = 2
    shape = (2, 3)
    mesh_dim_names = ("dp", "tp")
    device_type = "cpu"

    def __init__(self, rank: int, tp: _TpMesh) -> None:
        self._rank = rank
        self._tp = tp

    def size(self) -> int:
        return 6

    def get_rank(self) -> int:
        return self._rank

    def get_coordinate(self) -> tuple[int, int]:
        return divmod(self._rank, 3)

    def get_local_rank(self, mesh_dim: str) -> int:
        return self.get_coordinate()[0 if mesh_dim == "dp" else 1]

    def __getitem__(self, name: str) -> _TpMesh:
        assert name == "tp"
        return self._tp


def test_do_something_gathers_its_rank_across_the_tp_group(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The smoke fn all-gathers a per-rank tensor over the tp group, not world."""
    group = _TpGroup()
    gathered: list[tuple[list[Tensor], Tensor, object]] = []

    def all_gather(outputs: list[Tensor], value: Tensor, group: object) -> None:
        gathered.append((outputs, value, group))
        for i, out in enumerate(outputs):
            out.copy_(value + i)

    fake_dist = MagicMock()
    fake_dist.get_rank.return_value = 4
    fake_dist.all_gather.side_effect = all_gather
    monkeypatch.setattr(testing, "td", fake_dist)

    do_something(cast("DeviceMesh", _FullMesh(4, _TpMesh(group))))

    ((outputs, value, used_group),) = gathered
    assert used_group is group
    assert torch.equal(value, torch.full((4,), 4.0, dtype=torch.float32))
    assert len(outputs) == 3
    assert torch.equal(
        torch.stack(outputs),
        torch.tensor(
            [
                [4.0, 4.0, 4.0, 4.0],
                [5.0, 5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0, 6.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert all(output.dtype == torch.float32 for output in outputs)
    assert capsys.readouterr().out == (
        "mesh.get_rank()=4 mesh.get_coordinate()=(1, 1) "
        "mesh.get_local_rank(mesh_dim='dp')=1 "
        "mesh.get_local_rank(mesh_dim='tp')=1\n"
        "Rank 4 all_gather received: "
        "[tensor([4., 4., 4., 4.]), tensor([5., 5., 5., 5.]), "
        "tensor([6., 6., 6., 6.])]\n"
    )


def test_bindable_port_selection_retries_a_stolen_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    held_port = cast(tuple[str, int], holder.getsockname())[1]
    free_port = WorkerPool.find_free_port()
    candidates = iter((held_port, free_port))
    monkeypatch.setattr(
        WorkerPool,
        "find_free_port",
        staticmethod(lambda: next(candidates)),
    )

    try:
        assert _pool(1)._pick_bindable_port() == free_port
    finally:
        holder.close()


def test_port_selection_sees_a_holder_on_a_non_loopback_address() -> None:
    """A port held on ANY local address must not be offered as free.

    ``TCPStore`` listens on the wildcard address even when handed
    ``MASTER_ADDR=localhost``, so a port bound by another process on a
    non-loopback local address (``127.0.1.1`` here, but on CI any interface
    the runner holds) collides with rank 0's listen. Selecting on
    ``127.0.0.1`` cannot see such a holder and hands back a doomed port,
    which surfaced as an intermittent CI
    ``DistNetworkError ... EADDRINUSE`` during ``init_process_group``.
    """
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.bind(("127.0.1.1", 0))
        holder.listen(1)
    except OSError as error:  # pragma: no cover -- host lacks a second local addr
        holder.close()
        pytest.skip(f"no non-loopback local address to hold: {error}")
    held_port = int(holder.getsockname()[1])  # pyright: ignore[reportAny] -- socket.getsockname is Any in typeshed; int() pins the port.
    try:
        # The pool must reject a port it cannot actually listen on. Feeding the
        # held port as the first candidate makes the probe the only thing that
        # can catch it.
        pool = _pool(2)
        with (
            patch.object(WorkerPool, "find_free_port", staticmethod(lambda: held_port)),
            pytest.raises(RuntimeError, match="could not secure a bindable"),
        ):
            pool._pick_bindable_port()
    finally:
        holder.close()


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("rank", 1, "Expected td.get_rank() == mesh.get_rank()."),
        ("ndim", 1, "Expected mesh.ndim == 2."),
        ("shape", (3, 2), "Expected mesh.shape == (2, 3)."),
        ("size", 5, "Expected mesh.size() == 6."),
        (
            "mesh_dim_names",
            ("tp", "dp"),
            'Expected mesh.mesh_dim_names == ("dp", "tp").',
        ),
        ("device_type", "cuda", 'Expected mesh.device_type == "cpu".'),
    ],
)
def test_do_something_reports_exact_mesh_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    expected: str,
) -> None:
    mesh = MagicMock()
    mesh.get_rank.return_value = 0
    mesh.ndim = 2
    mesh.shape = (2, 3)
    mesh.size.return_value = 6
    mesh.mesh_dim_names = ("dp", "tp")
    mesh.device_type = "cpu"
    fake_dist = MagicMock()
    fake_dist.get_rank.return_value = 0
    monkeypatch.setattr(testing, "td", fake_dist)
    if field == "rank":
        fake_dist.get_rank.return_value = value
    elif field == "size":
        mesh.size.return_value = value
    else:
        setattr(mesh, field, value)

    with pytest.raises(ValueError, match=re.escape(expected)) as error:
        do_something(cast("DeviceMesh", mesh))

    assert error.value.args == (expected,)


def test_call_reports_exact_error_when_queue_is_missing() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("Expected self.queue is not None."),
    ) as error:
        _pool(1)(do_something)

    assert error.value.args == ("Expected self.queue is not None.",)


def test_call_reports_exact_error_when_ack_queue_is_missing() -> None:
    pool = _pool(1)
    pool.queue = cast("tm.Queue[bytes | None]", MagicMock())

    with pytest.raises(
        ValueError,
        match=re.escape("Expected self.ack_queue is not None."),
    ) as error:
        pool(do_something)

    assert error.value.args == ("Expected self.ack_queue is not None.",)


def test_call_reports_exact_error_when_processes_are_missing() -> None:
    pool = _pool(1)
    pool.queue = cast("tm.Queue[bytes | None]", MagicMock())
    pool.ack_queue = cast("tm.Queue[bool]", MagicMock())

    with pytest.raises(
        ValueError,
        match=re.escape("Expected self.processes is not None."),
    ) as error:
        pool(do_something)

    assert error.value.args == ("Expected self.processes is not None.",)


def test_await_dispatch_ack_reports_exact_error_when_ack_queue_is_missing() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("Expected self.ack_queue is not None."),
    ) as error:
        _pool(1)._await_dispatch_ack([])

    assert error.value.args == ("Expected self.ack_queue is not None.",)


def test_terminate_reports_exact_error_when_queue_is_missing() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape("Expected self.queue is not None."),
    ) as error:
        _pool(1).terminate()

    assert error.value.args == ("Expected self.queue is not None.",)


def test_terminate_reports_exact_error_when_processes_are_missing() -> None:
    pool = _pool(1)
    pool.queue = cast("tm.Queue[bytes | None]", MagicMock())

    with pytest.raises(
        ValueError,
        match=re.escape("Expected self.processes is not None."),
    ) as error:
        pool.terminate()

    assert error.value.args == ("Expected self.processes is not None.",)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
