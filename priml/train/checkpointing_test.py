"""Tests for the Checkpointer (storage I/O + integration with policy)."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import functools
import shutil
import tempfile

from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

import pytest
import torch
import torch.distributed as dist

from priml.distributed.testing import WorkerPool
from priml.train.checkpointing import (
    AsyncLocalStateDictStorer,
    Checkpointer,
    SyncLocalStateDictStorer,
    _read_checkpoint,
)


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from priml.distributed.testing import WarmPoolGetter


@pytest.fixture
def temp_checkpoint_dir() -> Generator[Path, None, None]:
    """Create temporary checkpoint directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def single_rank_group(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Initialize a 1-rank gloo process group for sharded-checkpoint tests."""
    try:
        port = WorkerPool.find_free_port()
    except OSError as error:
        raise pytest.skip.Exception(
            f"Gloo requires a permitted loopback socket: {error}"
        ) from error
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(port))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    dist.init_process_group(
        backend="gloo",
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


class _DictTarget:
    """In-test ``Checkpointable``: serializes/restores a plain state dict.

    ``state_dict`` supplies both the save blob and the load/resharding template
    (the same dict reference, mutated in place by a DCP load). ``load_state_dict``
    captures the restored blob in ``loaded`` so a test can assert on it.
    """

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state
        self.loaded: dict[str, Any] | None = None

    def state_dict(self) -> dict[str, Any]:
        return self._state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.loaded = state_dict


def _save(ckpt: Checkpointer, step: int, state: dict[str, Any]) -> None:
    """Force-save ``state`` at ``step`` (unconditional, to set up a checkpoint)."""
    ckpt.save(_DictTarget(state), step)


def _load(
    checkpoint_dir: Path,
    *,
    resume_step: int = -1,
    into: dict[str, Any] | None = None,
    **config: Any,
) -> dict[str, Any]:
    """Resume the checkpoint selected by ``resume_step``; return the restored state.

    Loading a specific step is a per-config concern now, so a fresh Checkpointer
    is built with ``resume_step`` set. ``guard=False`` (no overwrite guard for a
    pure load assertion); ``max_steps`` large so it never gates.
    """
    ckpt = Checkpointer(
        Checkpointer.Config(
            working_dir=checkpoint_dir, resume_step=resume_step, **config
        ),
    )
    target = _DictTarget({} if into is None else into)
    assert ckpt.load(target, max_steps=1e9, guard=False)
    assert target.loaded is not None
    return target.loaded


# -- basics ----------------------------------------------------------------


def test_save_logs_size_and_duration(
    temp_checkpoint_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A successful save reports path, on-disk size, and duration (telemetry)."""
    ckpt = Checkpointer(
        Checkpointer.Config(working_dir=temp_checkpoint_dir, save_every=1),
    )
    with caplog.at_level("INFO"):
        ckpt.save(_DictTarget({"x": torch.zeros(256)}), 1)
    assert any(
        "Saved checkpoint" in r.message and "MB" in r.message for r in caplog.records
    ), "save telemetry (size/duration) missing"


def test_init_validates_and_sets_fields(temp_checkpoint_dir: Path) -> None:
    ckpt = Checkpointer(
        Checkpointer.Config(
            working_dir=temp_checkpoint_dir, save_every=100, keep_last_n=3
        ),
    )
    assert ckpt.checkpoint_dir == temp_checkpoint_dir
    assert ckpt.save_every == 100
    assert ckpt.keep_last_n == 3


def test_default_working_dir_is_opinionated() -> None:
    assert Checkpointer.Config().working_dir == "/checkpoints"


def test_owner_resolves_checkpoint_working_dir() -> None:
    config = Checkpointer.Config()
    config.base_dir = "/scratch/runs/study/exp001"

    checkpointing = config.make()

    expected = Path("/scratch/runs/study/exp001/checkpoints")
    assert checkpointing.checkpoint_dir == expected


def test_explicit_working_dir_is_preserved(tmp_path: Path) -> None:
    checkpointing = Checkpointer.Config(working_dir=tmp_path / "checkpoints").make()

    assert checkpointing.checkpoint_dir == tmp_path / "checkpoints"


def test_maybe_save_follows_cadence(temp_checkpoint_dir: Path) -> None:
    ckpt = Checkpointer(
        Checkpointer.Config(working_dir=temp_checkpoint_dir, save_every=100),
    )
    t = _DictTarget({"step": 0})
    assert not ckpt.maybe_save(t, 0)  # step 0 is never saved
    assert not ckpt.maybe_save(t, 50)
    assert ckpt.maybe_save(t, 100)
    assert not ckpt.maybe_save(t, 150)
    assert ckpt.available_steps() == [100]


def test_save_and_load_roundtrip(temp_checkpoint_dir: Path) -> None:
    ckpt = Checkpointer(Checkpointer.Config(working_dir=temp_checkpoint_dir))
    state = {"model": "dummy_state", "step": 1000}
    _save(ckpt, 1000, state)
    assert ckpt.available_steps() == [1000]
    assert _load(temp_checkpoint_dir, resume_step=1000) == state


def test_load_returns_false_when_empty(temp_checkpoint_dir: Path) -> None:
    ckpt = Checkpointer(Checkpointer.Config(working_dir=temp_checkpoint_dir))
    assert ckpt.load(_DictTarget({}), max_steps=1e9, guard=False) is False


def test_save_prunes_to_keep_last_n(temp_checkpoint_dir: Path) -> None:
    ckpt = Checkpointer(
        Checkpointer.Config(
            working_dir=temp_checkpoint_dir, save_every=100, keep_last_n=2
        ),
    )
    for step in (100, 200, 300, 400):
        _save(ckpt, step, {"step": step})
    assert ckpt.available_steps() == [300, 400]


def test_keep_every_exempts_archival_checkpoints(temp_checkpoint_dir: Path) -> None:
    ckpt = Checkpointer(
        Checkpointer.Config(
            working_dir=temp_checkpoint_dir,
            save_every=100,
            keep_last_n=2,
            keep_every=300,
        ),
    )
    for step in range(100, 1001, 100):
        _save(ckpt, step, {"step": step})
    # Archival multiples of 300 survive alongside the rolling last-2 window.
    assert ckpt.available_steps() == [300, 600, 900, 1000]


def test_keep_every_zero_keeps_prior_behavior(temp_checkpoint_dir: Path) -> None:
    ckpt = Checkpointer(
        Checkpointer.Config(
            working_dir=temp_checkpoint_dir, save_every=100, keep_last_n=2
        ),
    )
    for step in (100, 200, 300, 400):
        _save(ckpt, step, {"step": step})
    assert ckpt.available_steps() == [300, 400]


def test_rejects_keep_every_off_cadence(temp_checkpoint_dir: Path) -> None:
    for keep_every in (-1, 150):
        with pytest.raises(ValueError, match="keep_every"):
            Checkpointer(
                Checkpointer.Config(
                    working_dir=temp_checkpoint_dir,
                    save_every=100,
                    keep_every=keep_every,
                ),
            )


def test_load_latest_returns_highest(temp_checkpoint_dir: Path) -> None:
    ckpt = Checkpointer(
        Checkpointer.Config(working_dir=temp_checkpoint_dir, save_every=100),
    )
    for step in (100, 200, 300):
        _save(ckpt, step, {"step": step})
    assert _load(temp_checkpoint_dir)["step"] == 300


def test_load_negative_index(temp_checkpoint_dir: Path) -> None:
    ckpt = Checkpointer(
        Checkpointer.Config(working_dir=temp_checkpoint_dir, save_every=100),
    )
    for step in (100, 200, 300, 400):
        _save(ckpt, step, {"step": step})
    assert _load(temp_checkpoint_dir, resume_step=-1)["step"] == 400
    assert _load(temp_checkpoint_dir, resume_step=-2)["step"] == 300
    assert _load(temp_checkpoint_dir, resume_step=-3)["step"] == 200


def test_load_specific_step(temp_checkpoint_dir: Path) -> None:
    ckpt = Checkpointer(
        Checkpointer.Config(working_dir=temp_checkpoint_dir, save_every=100),
    )
    for step in (100, 200, 300):
        _save(ckpt, step, {"step": step})
    assert _load(temp_checkpoint_dir, resume_step=200)["step"] == 200


def test_load_out_of_range_negative_index(temp_checkpoint_dir: Path) -> None:
    ckpt = Checkpointer(
        Checkpointer.Config(working_dir=temp_checkpoint_dir, save_every=100),
    )
    for step in (100, 200):
        _save(ckpt, step, {"step": step})
    out_of_range = Checkpointer(
        Checkpointer.Config(working_dir=temp_checkpoint_dir, resume_step=-3),
    )
    assert out_of_range.load(_DictTarget({}), max_steps=1e9, guard=False) is False


def test_available_steps_ignores_malformed_names(temp_checkpoint_dir: Path) -> None:
    """Malformed ``step_*.pt`` names must not crash int-parse."""
    ckpt = Checkpointer(Checkpointer.Config(working_dir=temp_checkpoint_dir))
    temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (temp_checkpoint_dir / "step_100.pt").write_bytes(b"x")
    (temp_checkpoint_dir / "step_latest.pt").write_bytes(b"x")  # malformed
    assert ckpt.available_steps() == [100]


def test_load_uses_weights_only(temp_checkpoint_dir: Path) -> None:
    """Load must use ``weights_only=True`` (no arbitrary code exec)."""
    ckpt = Checkpointer(Checkpointer.Config(working_dir=temp_checkpoint_dir))
    _save(ckpt, 0, {"step": torch.tensor([1, 2, 3])})

    seen: list[str] = []
    orig = torch.load

    def spy(path: Any, **kwargs: Any) -> Any:
        seen.append(f"weights_only={kwargs.get('weights_only')}")
        return orig(path, **kwargs)

    torch.load = spy  # ty: ignore[invalid-assignment]
    try:
        ckpt.load(_DictTarget({}), max_steps=1e9, guard=False)
    finally:
        torch.load = orig
    assert seen == ["weights_only=True"]


def test_rejects_nonpositive_save_every(temp_checkpoint_dir: Path) -> None:
    for save_every in (0, -1):
        with pytest.raises(ValueError, match="save_every"):
            Checkpointer(
                Checkpointer.Config(
                    working_dir=temp_checkpoint_dir, save_every=save_every
                ),
            )


def test_rejects_invalid_keep_last_n(temp_checkpoint_dir: Path) -> None:
    for keep_last_n in (-2, 0):
        with pytest.raises(ValueError, match="keep_last_n"):
            Checkpointer(
                Checkpointer.Config(
                    working_dir=temp_checkpoint_dir, keep_last_n=keep_last_n
                ),
            )


# -- completeness: resume + collisions skip crashed partials ---------------


def _incomplete_shard(checkpoint_dir: Path, step: int) -> None:
    """Create a shard directory without its ``.metadata`` completeness marker."""
    (checkpoint_dir / f"step_{step}.pt").mkdir(parents=True, exist_ok=True)


def test_prune_ignores_incomplete_checkpoints(temp_checkpoint_dir: Path) -> None:
    """Retention counts and deletes only complete checkpoints.

    A partial (crashed, or an in-flight async write) must not count toward
    keep_last_n nor be deleted out from under its writer. With keep_last_n=2 and
    two complete + one incomplete checkpoint, both complete ones survive and the
    partial is left untouched.
    """
    ckpt = Checkpointer(
        Checkpointer.Config(
            working_dir=temp_checkpoint_dir, save_every=10, keep_last_n=2
        ),
    )
    _save(ckpt, 10, {"step": 10})
    _incomplete_shard(temp_checkpoint_dir, 20)  # partial; must not be counted
    _save(ckpt, 30, {"step": 30})

    # Saving 30 prunes complete checkpoints to the newest 2 -> {10, 30} both kept
    # (the partial 20 is neither counted nor deleted).
    assert ckpt.available_steps() == [10, 30]
    assert (temp_checkpoint_dir / "step_20.pt").exists()  # partial left intact


def test_load_falls_back_past_incomplete_latest(temp_checkpoint_dir: Path) -> None:
    """A crashed latest shard is skipped; resume gets the last good checkpoint."""
    ckpt = Checkpointer(
        Checkpointer.Config(working_dir=temp_checkpoint_dir, save_every=10),
    )
    _save(ckpt, 10, {"step": 10})
    _save(ckpt, 20, {"step": 20})
    _incomplete_shard(temp_checkpoint_dir, 30)  # crashed, no .metadata

    assert ckpt.available_steps() == [10, 20]  # 30 is incomplete
    assert _load(temp_checkpoint_dir, save_every=10)["step"] == 20


def test_load_guard_detects_overlap(temp_checkpoint_dir: Path) -> None:
    _save(
        Checkpointer(
            Checkpointer.Config(working_dir=temp_checkpoint_dir, save_every=10),
        ),
        10,
        {"step": 10},
    )
    _save(
        Checkpointer(
            Checkpointer.Config(working_dir=temp_checkpoint_dir, save_every=10),
        ),
        20,
        {"step": 20},
    )
    # Fresh run (resume=False, start_step=0): future saves would overwrite 10, 20.
    fresh = Checkpointer(
        Checkpointer.Config(
            working_dir=temp_checkpoint_dir, save_every=10, resume=False
        ),
    )
    with pytest.raises(RuntimeError, match="would overwrite existing"):
        fresh.load(_DictTarget({}), max_steps=20, guard=True)
    # Resuming from step 20: only steps > 20 are checked, none exist -> no raise.
    resumed = Checkpointer(
        Checkpointer.Config(
            working_dir=temp_checkpoint_dir, save_every=10, resume_step=20
        ),
    )
    assert resumed.load(_DictTarget({}), max_steps=100, guard=True)


def test_load_guard_ignores_incomplete(temp_checkpoint_dir: Path) -> None:
    """A crashed partial is overwritable, not work worth protecting."""
    ckpt = Checkpointer(
        Checkpointer.Config(
            working_dir=temp_checkpoint_dir, save_every=10, resume=False
        ),
    )
    _incomplete_shard(temp_checkpoint_dir, 20)
    # No complete checkpoint collides, so the guard does not halt the run.
    assert ckpt.load(_DictTarget({}), max_steps=100, guard=True) is False


def test_load_guard_finite_with_inf_max_steps(temp_checkpoint_dir: Path) -> None:
    _save(
        Checkpointer(
            Checkpointer.Config(working_dir=temp_checkpoint_dir, save_every=10),
        ),
        10,
        {"step": 10},
    )
    ckpt = Checkpointer(
        Checkpointer.Config(
            working_dir=temp_checkpoint_dir, save_every=10, resume=False
        ),
    )
    with pytest.raises(RuntimeError, match="would overwrite existing"):
        ckpt.load(_DictTarget({}), max_steps=float("inf"), guard=True)


# -- distributed / sharded capability --------------------------------------


def test_dtensor_state_saves_complete_directory(
    temp_checkpoint_dir: Path,
    single_rank_group: None,
) -> None:
    """DTensor-bearing state writes a complete distributed-checkpoint directory.

    Completeness marker (``.metadata``) is written last, distinguishing a
    finished checkpoint from a crashed partial.
    """
    del single_rank_group
    ckpt = Checkpointer(
        Checkpointer.Config(working_dir=temp_checkpoint_dir, save_every=1),
    )
    fake = nn.Linear(4, 4)
    # Pin a CPU mesh: the gloo group is CPU, and an auto mesh resolves to MPS
    # on Apple silicon, where torch's DeviceMesh setup calls the missing
    # torch.mps.is_initialized and raises.
    fully_shard(fake, mesh=init_device_mesh("cpu", (1,)))
    _save(ckpt, 0, {"model": fake.state_dict()})

    ckpt_dir = temp_checkpoint_dir / "step_00000000.pt"
    assert ckpt_dir.is_dir(), "DTensor save must create a directory"
    assert (ckpt_dir / ".metadata").exists(), "completeness marker missing"
    assert any(f.suffix == ".distcp" for f in ckpt_dir.iterdir()), "no shard data"
    assert ckpt.available_steps() == [0]


def test_distributed_save_barriers(
    temp_checkpoint_dir: Path,
    single_rank_group: None,
) -> None:
    """A distributed save must barrier so ranks stay in lockstep."""
    del single_rank_group
    ckpt = Checkpointer(
        Checkpointer.Config(working_dir=temp_checkpoint_dir, save_every=1),
    )
    fake = nn.Linear(4, 4)
    # CPU mesh (see test_dtensor_state_saves_complete_directory): gloo is CPU
    # and an auto mesh hits the MPS DeviceMesh-setup crash on Apple silicon.
    fully_shard(fake, mesh=init_device_mesh("cpu", (1,)))

    calls: list[str] = []
    orig = dist.barrier

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append("barrier")
        return orig(*args, **kwargs)

    dist.barrier = spy  # ty: ignore[invalid-assignment]
    try:
        _save(ckpt, 0, {"model": fake.state_dict()})
    finally:
        dist.barrier = orig
    assert calls, "distributed save must call dist.barrier()"


def test_partial_shard_not_loadable(
    temp_checkpoint_dir: Path,
    single_rank_group: None,
) -> None:
    """A directory missing its completeness marker must not be returned."""
    del single_rank_group
    ckpt = Checkpointer(
        Checkpointer.Config(working_dir=temp_checkpoint_dir, save_every=1),
    )
    (temp_checkpoint_dir / "step_0.pt").mkdir(parents=True)  # no .metadata
    assert ckpt.load(_DictTarget({}), max_steps=1e9, guard=False) is False
    assert ckpt.available_steps() == []  # incomplete -> not available


# -- DTensor-safe checkpointing (round-trip / resharding) ------------------


# -- async storage ---------------------------------------------------------


def test_async_storage_roundtrip(
    temp_checkpoint_dir: Path,
    single_rank_group: None,
) -> None:
    """AsyncLocalStateDictStorer save/load returns identical state, once flushed."""
    del single_rank_group
    storage = AsyncLocalStateDictStorer()
    path = temp_checkpoint_dir / "step_0.pt"
    storage.write(path, {"x": torch.tensor([1, 2, 3])})
    # The in-flight write reports incomplete without blocking (hot path free).
    assert not storage.is_complete(path)
    # flush() awaits the write; afterwards the checkpoint is durable.
    storage.flush()
    assert storage.is_complete(path)
    loaded = storage.read(path, {"x": torch.zeros(3, dtype=torch.long)})
    assert torch.equal(loaded["x"], torch.tensor([1, 2, 3]))


def test_async_reads_a_sync_written_plain_file(
    temp_checkpoint_dir: Path,
    single_rank_group: None,
) -> None:
    """Toggling async_save on must still load a prior sync run's ``.pt`` file.

    Sync ``SyncLocalStateDictStorer`` writes plain state as a ``.pt`` file; async always
    writes DCP dirs. ``AsyncLocalStateDictStorer.read`` must dispatch on the on-disk format,
    not assume its own -- else a sync->async resume fails to load.
    """
    del single_rank_group
    path = temp_checkpoint_dir / "step_0.pt"
    SyncLocalStateDictStorer().write(path, {"x": torch.tensor([7, 8, 9])})
    assert path.is_file(), "sync plain write should be a .pt file, not a dir"

    loaded = AsyncLocalStateDictStorer().read(
        path, {"x": torch.zeros(3, dtype=torch.long)}
    )
    # A plain ``.pt`` load maps storages onto the current device (CUDA when
    # present), so compare values on CPU rather than assuming the saved device.
    assert torch.equal(loaded["x"].cpu(), torch.tensor([7, 8, 9]))


@pytest.mark.gpu_torch_cuda
def test_plain_file_read_maps_to_current_device(
    temp_checkpoint_dir: Path,
    single_rank_group: None,
) -> None:
    """A ``.pt`` checkpoint's tensors land on THIS rank's current device.

    Regression: without an explicit ``map_location``, tensors deserialize
    onto their SAVED device (rank 0's ``cuda:0``), so on a multi-GPU resume
    every non-zero rank materialized -- and its allocator permanently
    cached -- a full checkpoint copy on GPU 0, starving rank 0 into an OOM
    at its first resumed train step (arc2_drm_7m, 2026-07-06).
    """
    del single_rank_group
    if torch.cuda.device_count() < 2:
        pytest.skip("needs >= 2 CUDA devices to observe cross-device mapping")
    path = temp_checkpoint_dir / "step_0.pt"
    with torch.cuda.device(0):
        SyncLocalStateDictStorer().write(path, {"x": torch.ones(4, device="cuda")})
    with torch.cuda.device(1):
        loaded = SyncLocalStateDictStorer().read(
            path, {"x": torch.zeros(4, device="cuda")}
        )
    assert loaded["x"].device == torch.device("cuda", 1)


def test_plain_read_preserves_cpu_rng_state(
    temp_checkpoint_dir: Path,
) -> None:
    """A ``.pt`` load must keep the torch RNG blob a CPU ``ByteTensor``.

    ``torch.get_rng_state()`` returns a CPU ``uint8`` tensor; ``torch.set_rng_state``
    rejects anything else. ``_read_checkpoint`` maps model/optimizer storages onto
    the current device, but that remap must not touch the RNG state -- moving it to
    CUDA makes the restored state unusable (``RNG state must be a torch.ByteTensor``).
    """
    path = temp_checkpoint_dir / "step_0.pt"
    rng = torch.get_rng_state()
    torch.save({"rng": {"torch": rng}, "weight": torch.zeros(3)}, path)

    loaded = _read_checkpoint(path, {"rng": {"torch": torch.get_rng_state()}})

    restored = loaded["rng"]["torch"]
    assert restored.device.type == "cpu", (
        f"RNG state must stay on CPU, got {restored.device}"
    )
    # The real contract: the restored state must be accepted by torch.
    torch.set_rng_state(restored)


def test_async_storage_runs_after_write_callback_post_durability(
    temp_checkpoint_dir: Path,
    single_rank_group: None,
) -> None:
    """Write's after_write callback runs only once the bytes are durable.

    This is how retention rides the async pipeline: the prune callback fires on
    the background thread after the checkpoint lands, so it sees its own write
    complete -- no trailing-by-one, no separate end-of-run reconcile.
    """
    del single_rank_group
    storage = AsyncLocalStateDictStorer()
    path = temp_checkpoint_dir / "step_0.pt"
    saw_complete: list[bool] = []
    storage.write(
        path,
        {"x": torch.tensor([1])},
        after_write=lambda: saw_complete.append(storage.is_complete(path)),
    )
    storage.flush()
    assert saw_complete == [True], "after_write ran before the write was durable"


def test_async_storage_state_safe_to_mutate_after_write(
    temp_checkpoint_dir: Path,
    single_rank_group: None,
) -> None:
    """The contract: ``write`` captures state synchronously (staged to CPU).

    Mutating the source tensors after ``write`` returns must not corrupt the
    checkpoint -- the async stager has already snapshotted.
    """
    del single_rank_group
    storage = AsyncLocalStateDictStorer()
    path = temp_checkpoint_dir / "step_0.pt"
    src = torch.tensor([1, 2, 3])
    storage.write(path, {"x": src})
    src.add_(100)  # mutate after write returns
    loaded = storage.read(path, {"x": torch.zeros(3, dtype=torch.long)})
    assert torch.equal(loaded["x"], torch.tensor([1, 2, 3])), "stale snapshot"
    storage.flush()


def test_async_checkpointer_end_to_end(
    temp_checkpoint_dir: Path,
    single_rank_group: None,
) -> None:
    """A Checkpointer with async_save persists and resumes through the policy."""
    del single_rank_group
    ckpt = Checkpointer(
        Checkpointer.Config(
            working_dir=temp_checkpoint_dir,
            save_every=10,
            storer=AsyncLocalStateDictStorer.Config(),
        ),
    )
    ckpt.save(_DictTarget({"step": torch.tensor([10])}), 10)
    # load() flushes the in-flight async write, then restores.
    target = _DictTarget({"step": torch.zeros(1, dtype=torch.long)})
    assert ckpt.load(target, max_steps=1e9, guard=False)
    assert target.loaded is not None
    assert int(target.loaded["step"][0]) == 10
    assert ckpt.available_steps() == [10]  # now flushed -> visible


def test_async_retention_enforced_after_flush_no_close(
    temp_checkpoint_dir: Path,
    single_rank_group: None,
) -> None:
    """Retention rides the async pipeline -- flush alone enforces keep_last_n.

    No ``close``: prune runs as each write's after_write callback, so after the
    final flush only the newest keep_last_n complete checkpoints remain.
    """
    del single_rank_group
    ckpt = Checkpointer(
        Checkpointer.Config(
            working_dir=temp_checkpoint_dir,
            save_every=10,
            keep_last_n=2,
            storer=AsyncLocalStateDictStorer.Config(),
        ),
    )
    for step in (10, 20, 30, 40):
        ckpt.save(_DictTarget({"step": torch.tensor([step])}), step)
    ckpt.close()
    assert ckpt.available_steps() == [30, 40]


def test_storer_defaults_to_sync() -> None:
    """The default storer is synchronous local disk -- async is opt-in."""
    storer = Checkpointer.Config(working_dir="/scratch/checkpoints").make().storage
    assert isinstance(storer, SyncLocalStateDictStorer)


def test_async_is_complete_is_pure_no_join(
    temp_checkpoint_dir: Path,
    single_rank_group: None,
) -> None:
    """is_complete must not join the in-flight write (keeps async overlap).

    After ``write`` returns, the background write is still pending; querying
    completeness of OTHER paths (as retention/collision do) must not block on
    it. Asserts the pending future is unresolved after a prune-style scan, so
    the disk write genuinely overlaps subsequent work.
    """
    del single_rank_group
    storage = AsyncLocalStateDictStorer()
    (temp_checkpoint_dir / "step_0.pt").mkdir()
    (temp_checkpoint_dir / "step_0.pt" / ".metadata").write_bytes(b"x")  # prior, done

    storage.write(temp_checkpoint_dir / "step_10.pt", {"x": torch.tensor([10])})
    # Inspect every checkpoint's completeness (what _prune/_list does)...
    _ = storage.is_complete(temp_checkpoint_dir / "step_0.pt")
    _ = storage.is_complete(temp_checkpoint_dir / "step_10.pt")
    # ...and the in-flight write must still be unjoined (overlap preserved).
    assert storage.has_pending_write(), "is_complete joined the write -> no overlap"
    storage.flush()
    assert not storage.has_pending_write()


def _async_multisave_worker(result_dir: str, mesh: DeviceMesh) -> None:
    """Two async saves of one sharded model across ranks, then load back.

    Mirrors a real loop: shard ONCE, then checkpoint repeatedly. (Re-sharding a
    fresh FSDP model per step is not how training works and wedges the group.)
    """
    rank = mesh.get_rank()
    try:
        model, optimizer = _shard_and_step(mesh)
        ckpt = Checkpointer(
            Checkpointer.Config(
                working_dir=Path(result_dir) / "ck",
                save_every=1,
                keep_last_n=1,
                storer=AsyncLocalStateDictStorer.Config(),
            ),
        )
        save_target = _DictTarget(
            {"model": model.state_dict(), "opt": optimizer.state_dict()}
        )
        # Force-save each step: exercises the collective broadcast in save()
        # interleaved with the async storer's barriers (regression: must not
        # desync the gloo group).
        for step in (1, 2):
            ckpt.save(save_target, step)
        ckpt.close()
        # Retention kept the newest complete checkpoint.
        steps = ckpt.available_steps()
        load_target = _DictTarget(
            {"model": model.state_dict(), "opt": optimizer.state_dict()}
        )
        merged = ckpt.load(load_target, max_steps=1e9, guard=False)
        ok = steps == [2] and merged
        (Path(result_dir) / f"rank_{rank}").write_text("ok" if ok else f"FAIL:{steps}")
    except Exception as e:  # noqa: BLE001  -- surface worker error to parent
        (Path(result_dir) / f"rank_{rank}").write_text(f"FAIL:{e!r}")


@pytest.mark.compute_distributed
def test_async_multirank_does_not_deadlock(warm_pools: WarmPoolGetter) -> None:
    """Async save+prune+load across 2 ranks must not desync barriers.

    Prune runs rank-0-only; any collective (barrier/join) reached only on rank 0
    deadlocks the others. This drives repeated async saves with retention and a
    final load across a real 2-rank group.
    """
    pool = warm_pools({"dp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(functools.partial(_async_multisave_worker, tmp))
        results = {p.name: p.read_text() for p in Path(tmp).iterdir() if p.is_file()}
    assert results == {"rank_0": "ok", "rank_1": "ok"}, results


class _Tiny(nn.Module):
    """Minimal shardable model: one linear, recursive-ownership reset."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 2)

    def reset_parameters(self) -> None:
        self.fc.reset_parameters()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _full_tensor(t: torch.Tensor) -> torch.Tensor:
    """Gather a (possibly sharded) tensor to its full, replicated value."""
    return t.full_tensor() if isinstance(t, DTensor) else t


def _shard_and_step(mesh: DeviceMesh) -> tuple[nn.Module, torch.optim.Optimizer]:
    """Fully-shard a fresh model over ``mesh`` and run one AdamW step."""
    torch.manual_seed(0)
    model = _Tiny()
    fully_shard(model, mesh=mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    model(torch.randn(1, 2)).sum().backward()
    optimizer.step()
    return model, optimizer


def _resume_worker(result_dir: str, mesh: DeviceMesh) -> None:
    """Save+reload model and optimizer DTensors; compare full param + opt state."""
    rank = mesh.get_rank()
    try:
        model, optimizer = _shard_and_step(mesh)
        orig_weight = _full_tensor(next(model.parameters())).clone()
        opt_state = optimizer.state_dict()["state"]
        orig_exp_avg = _full_tensor(opt_state[0]["exp_avg"]).clone()
        ckpt = Checkpointer(
            Checkpointer.Config(working_dir=Path(result_dir) / "ck"),
        )
        ckpt.save(
            _DictTarget({"model": model.state_dict(), "opt": optimizer.state_dict()}),
            0,
        )

        reload_model, reload_opt = _shard_and_step(mesh)
        target = _DictTarget(
            {"model": reload_model.state_dict(), "opt": reload_opt.state_dict()}
        )
        loaded = ckpt.load(target, max_steps=1e9, guard=False)
        merged = target.loaded
        if merged is not None:
            reload_model.load_state_dict(merged["model"])
            reload_opt.load_state_dict(merged["opt"])
        weight = _full_tensor(next(reload_model.parameters()))
        exp_avg = _full_tensor(reload_opt.state_dict()["state"][0]["exp_avg"])
        ok = (
            loaded
            and torch.equal(weight, orig_weight)
            and torch.equal(exp_avg, orig_exp_avg)
        )
        (Path(result_dir) / f"rank_{rank}").write_text(
            "ok" if ok else f"FAIL loaded={loaded}",
        )
    except Exception as e:  # noqa: BLE001  -- surface worker error to parent
        (Path(result_dir) / f"rank_{rank}").write_text(f"FAIL:{e!r}")


@pytest.mark.compute_distributed
def test_sharded_resume_remaps_or_reshards(warm_pools: WarmPoolGetter) -> None:
    """Sharded resume must restore model AND optimizer DTensors correctly."""
    pool = warm_pools({"dp": 2})
    with tempfile.TemporaryDirectory() as tmp:
        pool(functools.partial(_resume_worker, tmp))
        results = {p.name: p.read_text() for p in Path(tmp).iterdir() if p.is_file()}
    assert results == {"rank_0": "ok", "rank_1": "ok"}, results


def _world2_save_worker(ckpt_dir: str, result_dir: str, mesh: DeviceMesh) -> None:
    """Save a sharded checkpoint at world=2 and record the full parameter."""
    rank = mesh.get_rank()
    try:
        model, _ = _shard_and_step(mesh)
        ckpt = Checkpointer(Checkpointer.Config(working_dir=Path(ckpt_dir)))
        ckpt.save(_DictTarget({"model": model.state_dict()}), 0)
        full = _full_tensor(next(model.parameters())).clone()
        if rank == 0:
            torch.save(full, Path(result_dir) / "full.pt")
        (Path(result_dir) / f"save_rank_{rank}").write_text("ok")
    except Exception as e:  # noqa: BLE001  -- surface worker error to parent
        (Path(result_dir) / f"save_rank_{rank}").write_text(f"FAIL:{e!r}")


def _world1_load_worker(ckpt_dir: str, result_dir: str, mesh: DeviceMesh) -> None:
    """Load a world=2 checkpoint at world=1; DCP must reshard to full tensor."""
    rank = mesh.get_rank()
    try:
        model, _ = _shard_and_step(mesh)
        ckpt = Checkpointer(Checkpointer.Config(working_dir=Path(ckpt_dir)))
        target = _DictTarget({"model": model.state_dict()})
        loaded = ckpt.load(target, max_steps=1e9, guard=False)
        merged = target.loaded
        if merged is not None:
            model.load_state_dict(merged["model"])
        full = _full_tensor(next(model.parameters()))
        ref = torch.load(Path(result_dir) / "full.pt")
        ok = loaded and torch.equal(full, ref)
        (Path(result_dir) / f"load_rank_{rank}").write_text(
            "ok" if ok else f"FAIL loaded={loaded}",
        )
    except Exception as e:  # noqa: BLE001  -- surface worker error to parent
        (Path(result_dir) / f"load_rank_{rank}").write_text(f"FAIL:{e!r}")


@pytest.mark.compute_distributed
def test_world_size_change_resume(warm_pools: WarmPoolGetter) -> None:
    """A world=2 checkpoint must reshard correctly when loaded at world=1."""
    save_pool = warm_pools({"dp": 2})
    load_pool = warm_pools({"dp": 1})
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = str(Path(tmp) / "ck")
        res = Path(tmp) / "res"
        res.mkdir()
        save_pool(functools.partial(_world2_save_worker, ckpt_dir, str(res)))
        load_pool(functools.partial(_world1_load_worker, ckpt_dir, str(res)))
        results = {
            p.name: p.read_text()
            for p in res.iterdir()
            if p.is_file() and not p.name.endswith(".pt")
        }
    assert results == {
        "save_rank_0": "ok",
        "save_rank_1": "ok",
        "load_rank_0": "ok",
    }, results


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
