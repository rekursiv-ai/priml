from __future__ import annotations

# ruff: noqa: INP001 (Implicit namespace package.)
# ruff: noqa: S301 (Suspicious pickle usage.)
# ruff: noqa: T201 (Print.)
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportImplicitStringConcatenation=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportAttributeAccessIssue=false
from collections.abc import Callable, Mapping
from dataclasses import field
from datetime import timedelta
from multiprocessing.process import BaseProcess
from types import TracebackType
from typing import Any, Self, cast

import math
import os
import pickle
import queue as queue_mod
import socket
import time

from configgle import Fig
from torch import (
    distributed as td,
    multiprocessing as tm,
)
from torch.distributed.device_mesh import DeviceMesh

import torch


class _WorkerDiedError(RuntimeError):
    """A pool worker exited before acking -- a recoverable, transient failure.

    Distinguishes a dead worker (gloo peer-close, OOM kill) from a wedged-but-
    alive worker (a dispatch timeout), so only the former triggers a respawn-
    and-retry rather than failing the test outright.
    """


class WorkerPool:
    __slots__ = ("ack_queue", "mesh_dims", "processes", "queue")

    class Config(Fig["WorkerPool"]):
        mesh_dims: dict[str, int] = field(default_factory=dict)
        """Mesh dimension name to size; their product is the world size."""

    def __init__(self, config: Config):
        self.mesh_dims: dict[str, int] = config.mesh_dims
        self.processes = self.queue = self.ack_queue = None

    def __enter__(self) -> Self:
        # ``find_free_port`` closes the socket before the workers bind it, so
        # ANY process on this host -- a concurrent pool, or anything else
        # holding a port on any local address -- can claim it in the gap,
        # leaving this pool's ranks unable to rendezvous. Rather than race,
        # spawn against a fresh port and require every worker to ack readiness
        # within a bound; a collided/wedged attempt is torn down and retried
        # (the shared spawn-with-retry loop lives in ``_respawn``).
        self._respawn()
        return self

    def _spawn_once(self) -> None:
        """Spawn the worker ranks once and block until all report ready.

        Raises if any rank dies or fails to ack readiness within
        ``_RENDEZVOUS_TIMEOUT`` (a port collision or stalled rendezvous), after
        tearing down whatever it started so the caller can retry cleanly.
        """
        world_size = math.prod(self.mesh_dims.values())
        port = self._pick_bindable_port()
        # Spawn (not the platform-default fork) so each worker starts in a fresh
        # interpreter. A forked child inherits the pytest parent's process image,
        # which is unsafe here in two ways: (1) if an earlier test initialized
        # CUDA in the parent, the child inherits a half-initialized CUDA context
        # and every torch.cuda call raises "CUDA error: initialization error";
        # (2) torch's multi-threaded intra-op pool guards a mutex that fork()
        # copies locked (the owning thread is absent in the child), deadlocking
        # any child that runs a torch op. Spawn sidesteps both by construction.
        ctx = tm.get_context("spawn")
        queue: tm.Queue[Any] = ctx.Queue()
        ack_queue: tm.Queue[Any] = ctx.Queue()
        ready_queue: tm.Queue[Any] = ctx.Queue()
        processes: list[BaseProcess] = []
        # ``__exit__`` does not run when ``__enter__`` raises (PEP 343), so a
        # failure mid-spawn must kill the children already started here.
        try:
            for rank in range(world_size):
                p: BaseProcess = ctx.Process(
                    target=type(self).worker,
                    args=(
                        rank,
                        self.mesh_dims,
                        port,
                        queue,
                        ack_queue,
                    ),
                    kwargs={"ready_queue": ready_queue},
                )
                p.start()
                processes.append(p)
            # Block until every rank has completed init_process_group. A rank
            # that loses the port race dies (rank 0 raises EADDRINUSE) or stalls
            # in the rendezvous; either way its ack never arrives. Poll the
            # readiness acks against a deadline while watching for dead workers,
            # so a collision fails in milliseconds (dead process) rather than
            # waiting out the whole timeout.
            self._await_ready(processes, ready_queue, world_size)
        except BaseException:
            self._kill_all(processes)
            raise
        self.queue = queue
        self.ack_queue = ack_queue
        self.processes = processes

    def _await_ready(
        self,
        processes: list[BaseProcess],
        ready_queue: tm.Queue[Any],
        world_size: int,
    ) -> None:
        """Block until all ``world_size`` ranks ack readiness, else raise.

        Polls the readiness queue against ``_RENDEZVOUS_TIMEOUT`` while watching
        the worker processes: if any exits before acking (a port collision makes
        rank 0 raise ``EADDRINUSE``), this raises at once rather than waiting out
        the full timeout, so the caller's retry fires near-instantly.
        """
        deadline = time.monotonic() + self._RENDEZVOUS_TIMEOUT.total_seconds()
        seen = 0
        while seen < world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                msg = f"rendezvous timed out: {seen}/{world_size} ranks ready"
                raise TimeoutError(msg)
            try:
                ready_queue.get(timeout=min(remaining, self._READY_POLL_SEC))
                seen += 1
            except queue_mod.Empty:
                # No ack yet; a dead worker means the group will never form.
                if any(p.exitcode is not None for p in processes):
                    msg = "a worker exited before rendezvous (port collision?)"
                    raise RuntimeError(msg) from None

    _READY_POLL_SEC = 0.1

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        del exc_type, exc_val, exc_tb
        self.terminate()
        self.processes = self.queue = self.ack_queue = None
        return False

    def __call__(self, fn: Callable[[DeviceMesh], None]) -> None:
        """Run ``fn`` on every worker and block until all ranks finish it.

        Dispatch is synchronous: this returns only after every rank has
        completed ``fn`` and rank 0 has acknowledged, so result files the
        workers write are readable the instant it returns -- the guarantee a
        reused pool relies on, since it has no teardown to synchronize against.

        If a warm-pool worker has died since the last dispatch (a transient gloo
        ``Connection closed by peer`` or an OOM kill on a starved CI runner makes
        a rank exit before acking), the whole pool is respawned and ``fn`` is
        re-dispatched. ``fn`` is required to be idempotent (workers write result
        files), so a clean re-run is safe; only ``_DISPATCH_ATTEMPTS`` respawns
        are tried before the failure surfaces.
        """
        last_exc: BaseException | None = None
        for attempt in range(self._DISPATCH_ATTEMPTS):
            assert self.queue is not None
            assert self.ack_queue is not None
            assert self.processes is not None
            self.queue.put(pickle.dumps(fn))
            try:
                self._await_dispatch_ack(self.processes)
                return
            except (_WorkerDiedError, TimeoutError) as exc:
                # The pool is now unusable: either a worker died (gloo peer-close
                # / OOM -> _WorkerDiedError) or rank 0 never acked in time
                # (TimeoutError -- under a loaded CI runner the worker can be
                # CPU-starved past the deadline while still alive). Both leave the
                # queues in an unknown state and ``_await_dispatch_ack`` has
                # already killed the workers, so rebuild the pool and retry -- but
                # not after the final attempt, whose failure surfaces below.
                last_exc = exc
                if attempt < self._DISPATCH_ATTEMPTS - 1:
                    self._respawn()
        raise RuntimeError(
            f"worker pool {self.mesh_dims} failed dispatch across "
            f"{self._DISPATCH_ATTEMPTS} attempts (last: {type(last_exc).__name__})"
        ) from last_exc

    _DISPATCH_ATTEMPTS = 3

    def _respawn(self) -> None:
        """Tear down the current (degraded) pool and spawn a fresh one."""
        if self.processes is not None:
            self._kill_all(self.processes)
        self.processes = self.queue = self.ack_queue = None
        last_exc: BaseException | None = None
        for _ in range(self._RENDEZVOUS_ATTEMPTS):
            try:
                self._spawn_once()
                return
            except BaseException as exc:  # noqa: BLE001 -- retry any spawn failure
                last_exc = exc
        msg = f"worker pool {self.mesh_dims} failed to rendezvous"
        raise RuntimeError(msg) from last_exc

    def _await_dispatch_ack(self, processes: list[BaseProcess]) -> None:
        """Wait for rank 0's command ack; fail if the warm pool died."""
        assert self.ack_queue is not None
        deadline = time.monotonic() + self._RENDEZVOUS_TIMEOUT.total_seconds()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill_all(processes)
                msg = "worker dispatch timed out waiting for rank 0 ack"
                raise TimeoutError(msg)
            try:
                self.ack_queue.get(timeout=min(remaining, self._READY_POLL_SEC))
                return
            except queue_mod.Empty:
                if any(p.exitcode is not None for p in processes):
                    self._kill_all(processes)
                    msg = "worker exited before dispatch ack"
                    raise _WorkerDiedError(msg) from None

    def terminate(self) -> None:
        assert self.queue is not None
        assert self.processes is not None
        self.queue.put(None)
        self._kill_all(self.processes)

    @classmethod
    def _kill_all(cls, processes: list[BaseProcess]) -> None:
        """Join each child with a bounded timeout, killing any that hang.

        A wedged worker (stuck in ``init_process_group`` or a user ``fn``)
        would block an unbounded ``join`` forever, so each child is given a
        grace period and then force-killed.
        """
        for p in processes:
            p.join(timeout=cls._JOIN_TIMEOUT_SEC)
            if p.is_alive():
                p.kill()
                p.join()

    _JOIN_TIMEOUT_SEC = 10.0

    # Cap the gloo rendezvous/collective wait well under pytest's per-test
    # timeout so a collision or wedged rank fails fast and visibly.
    _RENDEZVOUS_TIMEOUT = timedelta(seconds=30)

    # Fresh-port spawn attempts before giving up; covers transient port
    # collisions between concurrent pools without masking a real defect.
    _RENDEZVOUS_ATTEMPTS = 3

    def _pick_bindable_port(self) -> int:
        """Return a port the parent just confirmed it can bind on the wildcard.

        ``find_free_port`` has a TOCTOU window: a concurrent pool can claim the
        port between selection and the workers' bind. Re-binding it here, in the
        parent, immediately before spawn turns that race into an instant local
        ``EADDRINUSE`` (rank 0's TCPStore would otherwise refuse and every
        client rank would burn the full rendezvous timeout retrying). On a
        collision we simply pick again; an OS-free port binds in microseconds,
        so the common path costs nothing.
        """
        for _ in range(self._PORT_PICK_ATTEMPTS):
            port = type(self).find_free_port()
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                # "" is INADDR_ANY -- the address TCPStore listens on. See
                # ``find_free_port``; probing 127.0.0.1 here would re-admit
                # exactly the ports this method exists to reject.
                probe.bind(("", port))
            except OSError:
                probe.close()
                continue  # stolen in the TOCTOU window; pick another
            # Close only after confirming the bind; the window to the workers'
            # bind is now as small as possible (next statements spawn them).
            probe.close()
            return port
        msg = f"could not secure a bindable rendezvous port for {self.mesh_dims}"
        raise RuntimeError(msg)

    _PORT_PICK_ATTEMPTS = 20

    @classmethod
    def find_free_port(cls) -> int:
        """Return an OS-assigned free port for the worker rendezvous.

        Bound on ``""`` (INADDR_ANY, the wildcard), matching where ``TCPStore``
        actually listens: handed ``MASTER_ADDR=localhost`` it still binds
        ``*:port``. Probing ``127.0.0.1`` tests a strictly weaker condition --
        a port held by any process on a NON-loopback local address is invisible
        to a loopback probe yet collides with the wildcard listen, so rank 0
        dies with ``EADDRINUSE`` on a port just certified free. Measured with
        400 ports held on a secondary local address, a loopback probe returned
        a doomed port 118 times in 4000; the wildcard probe, 0.

        The socket is closed before the caller uses it, so a concurrent process
        could claim the port in the interim (a TOCTOU window). Callers that need
        the port to survive to a later bind go through ``_pick_bindable_port``,
        which re-verifies and re-picks on a collision.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            # getsockname() is `Any`; AF_INET always yields (host, port).
            port = s.getsockname()[1]
            assert isinstance(port, int)
            return port

    @classmethod
    def worker(
        cls,
        rank: int,
        mesh_dims: dict[str, int],
        port: int,
        command_queue: tm.Queue[Any],
        ack_queue: tm.Queue[Any],
        *,
        ready_queue: tm.Queue[Any],
    ) -> None:
        mesh_dim_sizes = tuple(mesh_dims.values())
        world_size = math.prod(mesh_dim_sizes)

        # setup fake process group
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        # The combined "cpu:gloo,cuda:nccl" backend eagerly constructs the NCCL
        # backend even when no CUDA tensor is ever used, which raises on a
        # GPU-less host. Select gloo-only there so CPU multirank tests run.
        backend = "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
        # gloo's default op timeout is 30 minutes; a starved or port-collided
        # rendezvous would then block the dispatching parent on ack_queue.get()
        # for half an hour, long past pytest's per-test timeout (the failure
        # reads as a bare "Timeout (>60s)" with no cause). Bound it so a wedged
        # rank instead raises here, is caught by the dispatched fn, and surfaces
        # as a readable FAIL:... result.
        td.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=cls._RENDEZVOUS_TIMEOUT,
        )
        # Signal the parent that this rank's rendezvous succeeded; the parent
        # blocks on these acks to distinguish a formed group from a collision.
        ready_queue.put(rank)

        device_mesh: DeviceMesh = DeviceMesh(
            "cpu",
            mesh=torch.arange(world_size).reshape(mesh_dim_sizes).tolist(),
            mesh_dim_names=tuple(mesh_dims.keys()),
        )

        while True:
            fn_list: list[bytes | None] = [command_queue.get() if rank == 0 else None]
            td.broadcast_object_list(fn_list, src=0)
            fn_pickled = fn_list[0]
            if fn_pickled is None:
                break
            fn = cast(Callable[[DeviceMesh], None], pickle.loads(fn_pickled))
            fn(device_mesh)
            # A barrier before the ack guarantees every rank has finished fn
            # (and flushed any result files) before the dispatching call
            # returns, so a caller reading those files -- e.g. a reused warm
            # pool with no teardown between tests -- never races the workers.
            td.barrier()
            if rank == 0:
                ack_queue.put(True)

        td.destroy_process_group()


# Maps ``mesh_dims`` to a reusable pool. The key must preserve insertion order
# because the worker builds the mesh from ``tuple(mesh_dims.values())`` and
# ``.keys()``, so ``{"dp": 2}`` and ``{"dp": 1, "tp": 2}`` are distinct meshes
# (and distinct pools) despite both spanning two ranks.
type WarmPoolGetter = Callable[[Mapping[str, int]], WorkerPool]


def do_something(mesh: DeviceMesh) -> None:
    rank = td.get_rank()

    assert td.get_rank() == mesh.get_rank()
    assert mesh.ndim == 2
    assert mesh.shape == (2, 3)
    assert mesh.size() == 6
    assert mesh.mesh_dim_names == ("dp", "tp")
    assert mesh.device_type == "cpu"

    tp_mesh: DeviceMesh = mesh["tp"]
    tp_group = tp_mesh.get_group()
    tp_size = tp_mesh.size()

    print(
        f"{mesh.get_rank()=} "
        f"{mesh.get_coordinate()=} "
        f"{mesh.get_local_rank(mesh_dim='dp')=} "
        f"{mesh.get_local_rank(mesh_dim='tp')=}",
    )

    input_tensor = torch.tensor([rank] * 4, dtype=torch.float32)
    output_tensors = [torch.empty_like(input_tensor) for _ in range(tp_size)]

    td.all_gather(output_tensors, input_tensor, group=tp_group)

    print(f"Rank {rank} all_gather received: {output_tensors}")


if __name__ == "__main__":
    with WorkerPool.Config(mesh_dims={"dp": 2, "tp": 3}).make() as q:
        q(do_something)
