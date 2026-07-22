"""Checkpointing: save/load/retention of training state on a step cadence.

One ``Checkpointer`` class. The plain-vs-distributed choice is not a subclass --
it is decided per save by the state's shape inside ``StateDictStorer``:

- Plain state -> a single ``.pt`` file, written via temp file + atomic rename
  (a present file is therefore always complete).
- DTensor-bearing state -> a distributed-checkpoint *directory* written
  collectively across ranks, whose ``.metadata`` marker (written last) signals
  completeness and whose load reshards to the current placement.

``StateDictStorer`` owns byte I/O and the single notion of completeness. The
``Checkpointer`` owns the directory, the step<->path naming, and one directory
scan producing the ``Checkpoint`` inventory that every selection/retention/
collision decision reads -- so no consumer re-derives "what exists" with a
divergent definition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import TYPE_CHECKING, Any, Protocol, Self, cast, override

import logging
import re
import shutil
import time

from configgle import Fig, Makeable
from torch.distributed.checkpoint import state_dict_loader, state_dict_saver
from torch.distributed.tensor import DTensor

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from priml.custom_types import CheckpointableProtocol
from priml.lib.userdirs import resolve_working_dir
from priml.runtime import is_rank_zero, runtime_output_path


if TYPE_CHECKING:
    from collections.abc import Iterable


logger = logging.getLogger(__name__)

type StateDict = dict[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Checkpoint:
    """A checkpoint found on disk (internal scan record): step, path, complete."""

    step: int
    path: Path
    complete: bool
    """A plain ``.pt`` file (atomic rename) is always complete; a shard dir is
    complete once its ``.metadata`` marker is present."""


class StateDictStorer(Protocol):
    """The byte-I/O seam: write, read, and judge completeness at a path.

    The boundary an alternative backend (async, remote, mmap) implements to plug
    into ``Checkpointer`` without touching its step/cadence/retention policy.
    Keys are fully-resolved paths chosen by the ``Checkpointer``; the backend
    never parses them. ``SyncLocalStateDictStorer`` is the default.

    Contract (the invariants that keep ``Checkpointer`` correct across sync,
    async, and distributed backends):

    - **``write`` returning does not mean durable.** It means the in-memory
      ``state`` is captured (safe for the caller to mutate). The bytes may still
      be landing on a background thread. ``after_write`` runs once they are
      durable; ``flush`` blocks until all pending writes (and their
      ``after_write``) finish.
    - **``is_complete`` is a pure, non-blocking disk read.** It reports an
      in-flight write as incomplete on its own (a partial checkpoint has no
      completeness marker), and never blocks or joins. This lets retention and
      collision scans inspect the whole inventory without stalling the hot path
      on a background write. ``Checkpointer`` only ever counts, loads, prunes, or
      guards checkpoints ``is_complete`` reports true, so a still-writing or
      crashed-partial checkpoint is never trusted or deleted.
    - **Synchronization happens only in ``flush`` (and a backend's own
      ``write``/``read``), collectively.** ``Checkpointer`` calls ``write`` /
      ``read`` / ``flush`` on every rank in lockstep, so a backend's barriers are
      always reached in step. ``is_complete`` must never barrier (it runs in
      rank-divergent scan loops).
    """

    def write(
        self,
        path: Path,
        state_dict: StateDict,
        after_write: Callable[[], None] = lambda: None,
    ) -> None:
        """Capture ``state_dict`` and begin persisting it at ``path``.

        On return, ``state_dict`` is captured (safe to mutate) but the checkpoint
        is durable only once ``is_complete(path)`` is true -- immediately for a
        synchronous backend, later for an async one. ``after_write`` runs once
        the bytes are durable (synchronously for a sync backend; on the
        background thread, after the write lands, for an async one) -- this is
        how retention rides the write pipeline. May be collective; called in
        lockstep on all ranks.
        """
        ...

    def read(self, path: Path, into: StateDict) -> StateDict:
        """Load the checkpoint at ``path``; ``into`` is the resharding template.

        Returns the restored state (callers use the return, not ``into``).
        ``path`` is assumed complete (the caller selected it via ``is_complete``).
        May be collective; called in lockstep on all ranks.
        """
        ...

    def is_complete(self, path: Path) -> bool:
        """Whether ``path`` is a finished, durable checkpoint -- not a partial.

        The completion signal retention/collision decisions gate on. A backend
        with a still-running write to ``path`` reports it incomplete *without
        blocking* (so the hot path stays clear); ``flush`` forces those writes
        to land. True only once all ranks' bytes are durable.
        """
        ...

    def flush(self) -> None:
        """Block until all in-flight writes (and their ``after_write``) finish.

        A synchronous backend is a no-op. The single "finish pending writes"
        operation -- called before ``load`` (so a resume sees a just-issued
        write) and once at end of run. Must be reached on every rank in lockstep.
        Collective: a barrier follows so ranks stay in step.
        """
        ...


class SyncLocalStateDictStorer:
    """Local-disk ``StateDictStorer``: writes/reads both formats, judges completeness.

    The format is chosen by the *state* on write (``_has_dtensor``) and by the
    *path* on read (``is_dir``), so one directory may legitimately hold both a
    ``.pt`` file and a ``.pt`` directory across saves.
    """

    class Config(Fig["SyncLocalStateDictStorer"]):
        """Synchronous local storer configuration (empty)."""

    def __init__(self, _config: Config | None = None) -> None:
        pass

    def write(
        self,
        path: Path,
        state_dict: StateDict,
        after_write: Callable[[], None] = lambda: None,
    ) -> None:
        """Persist ``state_dict`` at ``path`` durably, then run ``after_write``.

        DTensor-bearing state is written collectively as a distributed
        checkpoint directory (``.metadata`` last, ending on a barrier so ranks
        stay in lockstep). Plain state is written by rank 0 as a temp file then
        atomically renamed, so a present file is never partial. ``after_write``
        (retention) runs on every rank once the write is durable; it self-guards
        rank-0-only work.
        """
        start = time.perf_counter()
        if _has_dtensor(state_dict):
            path.mkdir(parents=True, exist_ok=True)
            state_dict_saver.save(state_dict, checkpoint_id=str(path))
            if dist.is_initialized():
                dist.barrier()
            if is_rank_zero():
                logger.info(
                    "Saved distributed checkpoint -> %s (%.2fs).",
                    path,
                    time.perf_counter() - start,
                )
        elif dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = path.with_suffix(path.suffix + ".tmp")
            torch.save(state_dict, temp_path)
            temp_path.rename(path)
            if dist.is_initialized():
                dist.barrier()
            logger.info(
                "Saved checkpoint -> %s (%.1f MB, %.2fs).",
                path,
                path.stat().st_size / 1024**2,
                time.perf_counter() - start,
            )
        after_write()

    def read(self, path: Path, into: StateDict) -> StateDict:
        """Load the checkpoint at ``path`` (DCP dir reshards; ``.pt`` file loads)."""
        return _read_checkpoint(path, into)

    def is_complete(self, path: Path) -> bool:
        """Whether ``path`` is a finished checkpoint, not a crashed partial."""
        return _is_complete(path)

    def flush(self) -> None:
        """No-op: synchronous writes are durable on return."""


class AsyncLocalStateDictStorer:
    """``StateDictStorer`` that stages to CPU synchronously then writes off the hot path.

    Backed by ``torch.distributed.checkpoint.async_save``: ``write`` de-stages
    the state dict to CPU (the snapshot -- so the caller may mutate its tensors
    immediately) and returns while a background thread performs the disk write.

    The synchronization model keeps async genuinely asynchronous AND
    multi-rank-safe:

    - ``is_complete`` is a **pure disk read** -- no join, no barrier. DCP writes
      the ``.metadata`` marker last, so an in-flight write's directory reports
      incomplete on its own; retention/collision scans (which call
      ``is_complete`` over the whole inventory) therefore never block on the
      background write, preserving overlap with subsequent training.
    - The join is performed only at explicit, **all-rank** call sites: the start
      of the next ``write`` (one save in flight at a time), ``read`` (a resume
      must see a just-issued write), and ``flush``. Because none is reached on a
      rank-divergent path, the collective barrier inside ``_join`` never desyncs.
    - ``after_write`` (retention) runs at the join, *after* the write is durable,
      on the main thread in lockstep -- so prune sees its own write complete (no
      trailing) and never deletes files from a background thread.

    Same DCP on-disk format as ``SyncLocalStateDictStorer`` -- loads still reshard across a
    world-size change. ``flush`` must be reached once after the final save (the
    trainer does this in cleanup) to drain the last write and its retention.
    """

    class Config(Fig["AsyncLocalStateDictStorer"]):
        """Asynchronous local storer configuration (empty)."""

    def __init__(self, _config: Config | None = None) -> None:
        self._pending: Future[Any] | None = None
        self._after_write: Callable[[], None] = lambda: None
        self._pending_path: Path | None = None
        self._pending_start: float = 0.0

    def write(
        self,
        path: Path,
        state_dict: StateDict,
        after_write: Callable[[], None] = lambda: None,
    ) -> None:
        """Stage ``state_dict`` to CPU synchronously; write in the background.

        Joins any prior in-flight write first (running its ``after_write``), then
        de-stages and launches this one. ``state_dict`` is safe to mutate on
        return; durability and ``after_write`` complete at the next join
        (``write``/``read``/``flush``).
        """
        self._join()
        path.mkdir(parents=True, exist_ok=True)
        # async_save is present at runtime but absent from torch's experimental
        # stubs -- suppressions confined here. It returns either a bare Future or
        # an AsyncSaveResponse; join on the upload (disk-write) future either way.
        self._pending_start = time.perf_counter()
        response: object = dcp.async_save(state_dict, checkpoint_id=str(path))  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        upload: object = getattr(response, "upload_completion", response)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        self._pending = cast("Future[Any]", upload)
        self._pending_path = path
        self._after_write = after_write

    def read(self, path: Path, into: StateDict) -> StateDict:
        """Join any pending write, then load (DCP dir reshards; ``.pt`` file loads).

        Dispatches on the on-disk format just like the sync backend, so toggling
        ``async_save`` on for a resume of a sync-written run loads correctly.
        """
        self._join()
        return _read_checkpoint(path, into)

    def is_complete(self, path: Path) -> bool:
        """Pure disk check -- ``.metadata`` (dir) or file existence. Never joins.

        An in-flight write reports incomplete because DCP writes ``.metadata``
        last; no special-casing or blocking is needed.
        """
        return _is_complete(path)

    def flush(self) -> None:
        """Block until the in-flight write is durable, then run its retention."""
        self._join()

    def has_pending_write(self) -> bool:
        """Whether a background write is still in flight (for tests/diagnostics)."""
        return self._pending is not None

    def _join(self) -> None:
        """Await the in-flight write, run its ``after_write``, barrier -- lockstep.

        Called only from all-rank entry points (``write``, ``read``, ``flush``),
        so the barrier is reached in lockstep. ``after_write`` (retention) runs
        after the write is durable and before the barrier.
        """
        if self._pending is not None:
            self._pending.result()
            self._pending = None
            # Telemetry at completion: the async write is durable only now, so
            # duration (dispatch -> join) and on-disk size are measured here, to
            # match the synchronous backend's save logging as closely as async
            # allows.
            if is_rank_zero() and self._pending_path is not None:
                logger.info(
                    "Saved async checkpoint -> %s (%.1f MB, %.2fs).",
                    self._pending_path,
                    _dir_size_mb(self._pending_path),
                    time.perf_counter() - self._pending_start,
                )
            self._pending_path = None
            after_write, self._after_write = self._after_write, lambda: None
            after_write()
        if dist.is_initialized():
            dist.barrier()


class Checkpointer:
    """The stepped disk engine: save cadence, resume, overwrite-guard, retention.

    Driven per training step against a *target* (any ``CheckpointableProtocol``,
    in practice the ``TrainLoop``) passed to each call -- the checkpointer holds
    no run state, only its own config + storer. ``maybe_save`` saves on cadence;
    ``save`` forces an end-of-run save; ``load`` resumes (per config) and guards
    against overwriting existing checkpoints; ``close`` drains a final async
    write. The training loop does no disk work itself -- it calls these.
    """

    class Config(Fig["Checkpointer"]):
        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""
        working_dir: Path | str = "/checkpoints"
        """Logical checkpoint directory."""
        filename: str = "step_{step:08d}.pt"
        """Checkpoint filename template containing one decimal ``{step}`` field."""
        save_every: int = 1000
        """Save cadence: a checkpoint is written at every multiple of this step."""
        keep_last_n: int = -1
        """Retain at most this many newest checkpoints (-1 = keep all)."""
        keep_every: int = 0
        """Also retain every checkpoint whose step is a multiple of this (0 =
        off). Archival retention on top of ``keep_last_n``: checkpoints on
        this interval are never pruned, so a long run keeps periodic
        trajectory snapshots (ensembling, transfer sources, analysis) while
        the rolling window stays small. Must be a multiple of ``save_every``
        so the archival steps actually land on the save cadence."""
        storer: Makeable[StateDictStorer] = field(
            default_factory=SyncLocalStateDictStorer.Config,
        )
        """Backend that persists/restores the state dict. Defaults to synchronous
        local-disk; use ``AsyncLocalStateDictStorer.Config`` to overlap the disk
        write with training (one save in flight, a CPU snapshot per save)."""
        resume: bool = True
        """Resume from an existing checkpoint on ``load`` when one is present.

        Preemption-restart is the common case, so this defaults on. With
        ``resume_step=-1`` (latest): load the largest checkpoint if any exist,
        else start fresh. With ``resume_step>=0``: load exactly that step or
        raise. ``resume=False`` never loads (always starts fresh)."""
        resume_step: int = -1
        """Which checkpoint ``load`` restores: -1 = latest, >=0 = that exact step."""
        allow_checkpoint_overwrite: bool = False
        """Permit saves that would overwrite an existing checkpoint. Off by
        default: ``load`` halts the run at startup if a future save on the
        cadence would land on a checkpoint already on disk (a fresh run reusing a
        name, or a rewind-resume clobbering newer checkpoints). Set True to
        deliberately re-mint, e.g. re-running a training section."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config):
        if config.save_every <= 0:
            raise ValueError(f"save_every must be positive, got {config.save_every}")
        if config.keep_last_n < -1 or config.keep_last_n == 0:
            raise ValueError(
                "keep_last_n must be -1 (keep all) or a positive count, got "
                f"{config.keep_last_n}",
            )
        if config.keep_every < 0 or (
            config.keep_every > 0 and config.keep_every % config.save_every != 0
        ):
            raise ValueError(
                f"keep_every must be 0 (off) or a positive multiple of "
                f"save_every ({config.save_every}) so archival steps land on "
                f"the save cadence, got {config.keep_every}",
            )
        try:
            filename_fields = list(Formatter().parse(config.filename))
            rendered_filename = config.filename.format(step=0)
        except (IndexError, KeyError, ValueError) as error:
            raise ValueError(
                f"invalid checkpoint filename template: {config.filename!r}"
            ) from error
        prefix: list[str] = []
        suffix: list[str] = []
        found_step = False
        for literal, field_name, format_spec, conversion in filename_fields:
            (suffix if found_step else prefix).append(literal)
            if field_name is None:
                continue
            if (
                found_step
                or field_name != "step"
                or conversion is not None
                or format_spec is None
                or re.fullmatch(r"(?:0?\d+)?d?", format_spec) is None
            ):
                raise ValueError(
                    "filename must contain exactly one decimal {step} field, got "
                    f"{config.filename!r}"
                )
            found_step = True
        if not found_step:
            raise ValueError(
                "filename must contain exactly one decimal {step} field, got "
                f"{config.filename!r}"
            )
        if Path(rendered_filename).name != rendered_filename:
            raise ValueError(
                f"checkpoint filename must not contain a directory: {config.filename!r}"
            )
        self.checkpoint_dir = Path(config.working_dir)
        self.filename = config.filename
        self._filename_pattern = re.compile(
            rf"{re.escape(''.join(prefix))}(?P<step>\d+){re.escape(''.join(suffix))}"
        )
        self.save_every = config.save_every
        self.keep_last_n = config.keep_last_n
        self.keep_every = config.keep_every
        self.resume = config.resume
        self.resume_step = config.resume_step
        self.allow_checkpoint_overwrite = config.allow_checkpoint_overwrite
        self.storage: StateDictStorer = config.storer.make()

    def maybe_save(self, target: CheckpointableProtocol, step: int) -> bool:
        """Save ``target`` at ``step`` iff it is on the cadence; return whether saved.

        The cadence decision and the write are one call (no should/do gap). Step
        0 is never saved. Pruning rides the write's ``after_write`` callback.
        """
        if step == 0 or step % self.save_every != 0:
            return False
        self._write(target, step)
        return True

    def save(self, target: CheckpointableProtocol, step: int) -> None:
        """Force-save ``target`` at ``step`` unless that step already exists.

        For the end-of-training save at an off-cadence step. The exists-check is
        collective (rank 0's verdict broadcast) so ranks never disagree and
        strand each other at the save barrier.

        Drains any pending async write first: its background barrier must
        complete before this method's collective broadcast, or the two
        collectives interleave and desync the ranks.
        """
        self.storage.flush()
        exists = step in self.available_steps()
        if _agreed_across_ranks(exists):
            return
        self._write(target, step)

    def load(
        self,
        target: CheckpointableProtocol,
        *,
        max_steps: float,
        guard: bool = True,
    ) -> bool:
        """Resume ``target`` per config, then guard overwrites; return whether loaded.

        Atomic over one inventory read (resume selection and the collision guard
        share it -- no time-of-check/use gap):

        - If ``resume`` is on, restore the checkpoint chosen by ``resume_step``
          (-1 = latest, with fallback past a crashed-latest; >=0 = that exact
          step, or raise if absent). ``resume=False`` or an empty dir starts
          fresh.
        - Then, unless ``allow_checkpoint_overwrite`` (or ``guard=False``, e.g.
          an eval-only run that writes nothing), halt if a future save on the
          cadence in ``(start_step, max_steps]`` would overwrite a checkpoint
          already on disk -- caught up front, not thousands of steps in.

        Returns True iff a checkpoint was restored (so the loop's start step is
        the resumed one, already set inside ``target``).
        """
        self.storage.flush()  # a just-issued async write must be visible to resume
        inventory = [c for c in self._list() if c.complete]
        resumed_step = self._resume(target, inventory) if self.resume else None
        if guard and not self.allow_checkpoint_overwrite:
            self._guard_overwrite(inventory, resumed_step or 0, max_steps)
        return resumed_step is not None

    def close(self) -> None:
        """Finish any pending async write and its retention. Call once at run end.

        A no-op for the synchronous backend; for the async backend it drains the
        last in-flight write and runs its prune, enforcing ``keep_last_n``.
        """
        self.storage.flush()

    def _resume(
        self,
        target: CheckpointableProtocol,
        inventory: list[_Checkpoint],
    ) -> int | None:
        """Restore the checkpoint selected by ``resume_step``; return its step.

        ``resume_step>=0`` requires that exact step (raises if absent);
        ``resume_step<0`` reverse-indexes the complete checkpoints, returning
        None on an empty dir (start fresh).
        """
        if self.resume_step >= 0:
            chosen = next((c for c in inventory if c.step == self.resume_step), None)
            if chosen is None:
                raise RuntimeError(
                    f"resume_step={self.resume_step} requested but not found in "
                    f"{self.checkpoint_dir} (available={[c.step for c in inventory]}). "
                    "Verify the checkpoint location and step.",
                )
        else:
            try:
                chosen = inventory[self.resume_step]
            except IndexError:
                logger.info(
                    "No checkpoint to resume under %s; starting fresh from step 0.",
                    self.checkpoint_dir,
                )
                return None
        logger.info("Resuming from checkpoint %s.", chosen.path)
        blob = self.storage.read(chosen.path, target.state_dict())
        target.load_state_dict(blob)
        return chosen.step

    def _guard_overwrite(
        self,
        inventory: list[_Checkpoint],
        start_step: int,
        max_steps: float,
    ) -> None:
        """Halt if a future save in ``(start_step, max_steps]`` would overwrite.

        Predicts the full collision set up front from existing complete
        checkpoints (incomplete partials are not protected). Uses ``max_steps``
        as the trajectory bound, so an early-stopping run may be refused over a
        collision it would not reach -- a deliberate bias toward refusing rather
        than silently overwriting.
        """
        collisions = [
            c.step
            for c in inventory
            if start_step < c.step <= max_steps and c.step % self.save_every == 0
        ]
        if not collisions:
            return
        raise RuntimeError(
            f"a future save would overwrite existing checkpoints at steps "
            f"{collisions} in {self.checkpoint_dir} (start_step={start_step}, "
            f"max_steps={max_steps}). Resume to continue the run, change the run "
            "name / checkpoint location, or set allow_checkpoint_overwrite=True "
            "to deliberately re-mint over them.",
        )

    def _write(self, target: CheckpointableProtocol, step: int) -> None:
        """Serialize ``target`` and write it at ``step``; retention rides the write."""
        self.storage.write(
            runtime_output_path(self._path(step)),
            target.state_dict(),
            after_write=self._prune,
        )

    def available_steps(self) -> list[int]:
        """Ascending steps of all complete checkpoints on disk (for diagnostics)."""
        return sorted(c.step for c in self._list() if c.complete)

    def _path(self, step: int) -> Path:
        """The on-disk path for ``step`` (a file and a shard dir share this stem)."""
        return self.checkpoint_dir / self.filename.format(step=step)

    def _list(self) -> list[_Checkpoint]:
        """Scan the directory once into checkpoint records (the only scan)."""
        if not self.checkpoint_dir.exists():
            return []
        out: list[_Checkpoint] = []
        for entry in self.checkpoint_dir.iterdir():
            step = self._parse_step(entry.name)
            if step is None:
                continue  # malformed (e.g. ``step_latest.pt``) or a temp file
            out.append(
                _Checkpoint(
                    step=step, path=entry, complete=self.storage.is_complete(entry)
                ),
            )
        out.sort(key=lambda c: c.step)
        return out

    def _parse_step(self, name: str) -> int | None:
        """Decode a checkpoint step from ``name``; None if it does not match.

        None covers a non-matching prefix/suffix, a temp file (``.pt.tmp``), and
        a matching shape whose middle is not an integer (e.g. ``step_latest.pt``).
        """
        match = self._filename_pattern.fullmatch(name)
        return int(match.group("step")) if match is not None else None

    def _prune(self) -> None:
        """Delete complete checkpoints beyond ``keep_last_n``, oldest first.

        Runs as ``storage.write``'s ``after_write`` on every rank, so it
        self-guards to rank 0 (deletion is rank-0 file I/O; no other rank reads
        an aged-out checkpoint mid-run, so no barrier is needed). Counts and
        deletes only *complete* checkpoints: a partial is either crashed or an
        in-flight write, never a retention candidate. Checkpoints on the
        ``keep_every`` archival interval are exempt.
        """
        if not is_rank_zero() or self.keep_last_n < 0:
            return
        complete = [c for c in self._list() if c.complete]
        for doomed in complete[: max(0, len(complete) - self.keep_last_n)]:
            if self.keep_every > 0 and doomed.step % self.keep_every == 0:
                continue  # Archival snapshot: retained forever.
            try:
                if doomed.path.is_dir():  # a shard checkpoint
                    shutil.rmtree(doomed.path)
                else:  # a plain .pt file
                    doomed.path.unlink()
                logger.info(
                    "Purged checkpoint %s (keep_last_n=%d).",
                    doomed.path,
                    self.keep_last_n,
                )
            except OSError as e:
                logger.warning("Failed to delete checkpoint %s: %s", doomed.path, e)


def _dir_size_mb(path: Path) -> float:
    """On-disk size in MB of a checkpoint -- a single file or a shard directory."""
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        total = path.stat().st_size
    return total / 1024**2


def _agreed_across_ranks(verdict: bool) -> bool:
    """Broadcast rank 0's boolean ``verdict`` so every rank takes the same branch.

    A per-rank checkpoint-existence check can disagree (a shard becomes visible
    on one rank before another), which would let one rank skip a save while the
    others enter the save barrier -- a deadlock. Rank 0's view wins.
    """
    if not dist.is_initialized():
        return verdict
    shared = [verdict]
    dist.broadcast_object_list(shared, src=0)
    return shared[0]


def _read_checkpoint(path: Path, into: StateDict) -> StateDict:
    """Load a checkpoint, dispatching on its on-disk format (backend-agnostic).

    A DCP *directory* is loaded in place into ``into``, resharding each tensor to
    this rank's current placement (so a load survives a world-size change), and
    returned. A plain ``.pt`` *file* is ``torch.load``ed and the fresh dict
    returned. Shared by every ``StateDictStorer`` because the format is a property of
    what is on disk, not of which backend wrote it -- so a dir written by one
    backend loads through another. ``path`` is assumed complete.

    The ``.pt`` load maps storages to this rank's current device. Without a
    ``map_location``, tensors deserialize onto their saved device.

    The RNG blob is exempt: ``torch.get_rng_state()`` is a CPU ``uint8`` tensor
    and ``torch.set_rng_state`` rejects anything that is not a CPU
    ``ByteTensor``. Mapping it onto CUDA alongside the model/optimizer tensors
    would make the restored RNG state unusable, so its subtree is pulled back to
    CPU after the load.
    """
    if path.is_dir():
        state_dict_loader.load(into, checkpoint_id=str(path))
        return into
    if torch.cuda.is_available():
        map_location = torch.device("cuda", torch.cuda.current_device())
    else:
        map_location = torch.device("cpu")
    blob = cast(
        "StateDict",
        torch.load(path, weights_only=True, map_location=map_location),
    )
    if "rng" in blob:
        blob["rng"] = _to_cpu(blob["rng"])
    return blob


def _to_cpu(obj: object) -> object:
    """Recursively move every tensor in ``obj`` to CPU, preserving structure.

    Used to exempt the RNG blob from ``_read_checkpoint``'s device remap: torch's
    RNG states (``torch.get_rng_state()`` and each entry of
    ``torch.cuda.get_rng_state_all()``) are CPU ``ByteTensor``s, and
    ``set_rng_state`` / ``set_rng_state_all`` reject anything on another device.
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, Mapping):
        mapping = cast("Mapping[object, object]", obj)
        return {k: _to_cpu(v) for k, v in mapping.items()}
    if isinstance(obj, list):
        items = cast("list[object]", obj)
        return [_to_cpu(v) for v in items]
    if isinstance(obj, tuple):
        elems = cast("tuple[object, ...]", obj)
        return tuple(_to_cpu(v) for v in elems)
    return obj


def _is_complete(path: Path) -> bool:
    """Whether ``path`` is a finished checkpoint, not a crashed partial.

    A plain file is complete by existence (atomic rename). A DCP directory is
    complete once its ``.metadata`` marker -- written last -- is present. Pure
    disk read; shared by every ``StateDictStorer`` (completeness is a disk property).
    """
    if path.is_dir():
        return (path / ".metadata").exists()
    return path.is_file()


def _has_dtensor(obj: object) -> bool:
    """Whether ``obj`` contains a ``DTensor`` anywhere in its structure.

    DTensor-bearing state (sharded model/optimizer under FSDP/HSDP) cannot be
    persisted by a plain ``torch.save`` of one rank -- that stores a single
    shard mislabeled as the whole tensor -- so it is routed to distributed
    checkpointing instead.
    """
    if isinstance(obj, DTensor):
        return True
    if isinstance(obj, Mapping):
        return any(_has_dtensor(v) for v in cast("Iterable[object]", obj.values()))
    if isinstance(obj, (list, tuple, set)):
        return any(_has_dtensor(v) for v in cast("Iterable[object]", obj))
    return False
