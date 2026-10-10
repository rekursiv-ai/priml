"""Signal dumps: every evaluated view's label-free signals, written beside a run.

:class:`~priml.baselines.arcagi.metric.CanonicalPassK` collects, per view,
the halt confidence, mean log-prob, stability, and per-step trajectory it was
handed; :class:`SignalDumpTracker` gathers every rank's rows and writes them as
one ``.npz`` per evaluation, for offline reranking.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Self, cast, override

import logging
import re

from configgle import Fig

import numpy as np

from priml.baselines.arcagi.augmentation import ArcSpec
from priml.lib.codec import from_plain
from priml.paths import resolve_working_dir
from priml.runtime import is_rank_zero


if TYPE_CHECKING:
    from numpy.typing import NDArray

    import torch.distributed as dist
else:
    from wrapt import lazy_import

    dist = lazy_import("torch.distributed")


logger = logging.getLogger(__name__)

type SignalRow = tuple[str, str, str, float, float, float, int, int]
"""``(task, input hash, prediction hash, halt logit, log-prob, stability, rows, cols)``."""

type StepRow = tuple[int, int, tuple[float, ...], tuple[int, ...]]
"""``(converge step, changes, per-step halt logits, per-step correctness)``."""


class SignalDumpPayload(NamedTuple):
    """Per-rank signal-dump data :meth:`CanonicalPassK.compute` emits in ``extras``."""

    rows: list[SignalRow]
    grids: dict[str, NDArray[np.uint8]]
    steps: list[StepRow]
    pass_ks: tuple[int, ...]


class SignalDumpTracker:
    """Tracker writing :class:`CanonicalPassK`'s signal dump from eval ``extras``."""

    class Config(Fig["SignalDumpTracker"]):
        """Signal-dump destination."""

        base_dir: Path | str | None = None
        """Run root supplied during parent finalization."""

        working_dir: Path | str = "/signals_{global_step}.npz"
        """Logical destination; ``{global_step}`` is formatted at write time."""

        spec: ArcSpec = field(default_factory=ArcSpec)
        """Prepared dataset's packed-grid geometry."""

        keep_last_n: int = -1
        """Newest step-stamped dumps kept after each write; ``-1`` keeps all.

        Rotation deletes files an offline analysis may still want: copy a dump
        out of the run directory to pin it, or archive with ``keep_every``."""

        keep_every: int = 0
        """Never delete a dump whose step is a multiple of this; ``0`` is off."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.keep_last_n < -1 or config.keep_last_n == 0:
            raise ValueError(
                "keep_last_n must be -1 (keep all) or positive; got "
                f"{config.keep_last_n}.",
            )
        if config.keep_every < 0:
            raise ValueError(f"keep_every must be >= 0; got {config.keep_every}.")
        self.config = config

    def log_metrics(
        self,
        metrics: Mapping[str, object],
        step: int,
        *,
        prefix: str = "",
    ) -> None:
        """Write the dump carried in ``metrics["extras"]`` for evaluation metrics.

        Args:
          metrics: Evaluation payload; ``extras["signal_dump"]`` holds the dump.
          step: Global step, formatted into ``{global_step}``.
          prefix: Only ``"eval/"`` writes.

        """
        if prefix != "eval/":
            return
        extras = metrics.get("extras")
        if extras is None:
            if not _any_rank(False):
                return
            payload = SignalDumpPayload(rows=[], grids={}, steps=[], pass_ks=())
        else:
            try:
                extras_map = from_plain(extras, dict[str, object])
            except TypeError as err:
                raise TypeError(
                    "SignalDumpTracker expected metrics['extras'] to be "
                    f"a mapping, got {type(extras).__name__}.",
                ) from err
            found = extras_map.get("signal_dump")
            if not isinstance(found, SignalDumpPayload):
                raise TypeError(
                    "SignalDumpTracker expected extras['signal_dump'] to be "
                    f"SignalDumpPayload, got {type(found).__name__}.",
                )
            if not _any_rank(True):
                return
            payload = found
        write_signal_dump(
            payload=payload,
            dump_signals_path=str(self.config.working_dir),
            global_step=step,
            spec=self.config.spec,
        )
        self._prune()

    def _prune(self) -> None:
        """Delete the oldest step-stamped dumps beyond ``keep_last_n``, on rank 0."""
        template = str(self.config.working_dir)
        if self.config.keep_last_n == -1 or "{global_step}" not in template:
            return
        if not is_rank_zero():
            return
        path = Path(template).expanduser()
        pattern = re.compile(
            re.escape(path.name).replace(re.escape("{global_step}"), r"(\d+)"),
        )
        dumps = sorted(
            (int(match.group(1)), candidate)
            for candidate in path.parent.glob(path.name.replace("{global_step}", "*"))
            if (match := pattern.fullmatch(candidate.name))
        )
        every = self.config.keep_every
        for step, doomed in dumps[: max(0, len(dumps) - self.config.keep_last_n)]:
            if every > 0 and step % every == 0:
                continue
            doomed.unlink()
            logger.info("Deleted signal dump %s (keep_last_n rotation).", doomed)

    def log_images(self, key: str, images: list[object], step: int) -> None:
        """Ignore images; this tracker writes only the signal dump."""
        del key, images, step

    def log_notes(self, notes: str) -> None:
        """Ignore run notes; a signal dump has none."""
        del notes

    def close(self) -> None:
        """Release nothing; every write is synchronous."""


def write_signal_dump(
    *,
    payload: SignalDumpPayload,
    dump_signals_path: str | Path,
    global_step: int,
    spec: ArcSpec,
) -> None:
    """Gather every rank's dump and write it as one ``.npz`` from rank 0.

    Args:
      payload: This rank's rows, unique prediction grids, and step rows.
      dump_signals_path: A ``Path`` is literal; a ``str`` formats ``{global_step}``.
      global_step: Step substituted into a ``str`` path.
      spec: Dataset-owned packed-grid geometry.

    """
    if isinstance(dump_signals_path, Path):
        out_path = str(dump_signals_path)
    else:
        out_path = dump_signals_path.replace("{global_step}", str(global_step))
        if "{" in out_path or "}" in out_path:
            # A surviving placeholder means finalize never filled the run context.
            if is_rank_zero():
                logger.info(
                    "[eval] signal dump skipped: unresolved path %r",
                    dump_signals_path,
                )
            return
    rows = _gather_list(payload.rows)
    grids = _gather_grids(payload.grids)
    steps = _gather_list(payload.steps)
    if not is_rank_zero():
        return
    n = len(rows)
    group_ids = np.empty(n, dtype=np.int32)
    pred_ids = np.empty(n, dtype=np.int32)
    q_halt = np.empty(n, dtype=np.float32)
    logprob = np.empty(n, dtype=np.float32)
    stability = np.empty(n, dtype=np.float32)
    n_rows = np.empty(n, dtype=np.uint8)
    n_cols = np.empty(n, dtype=np.uint8)
    group_index: dict[tuple[str, str], int] = {}
    pred_index: dict[str, int] = {}
    group_table: list[str] = []
    pred_table: list[str] = []
    for j, (name, input_h, pred_h, qh, lp, st, nr, nc) in enumerate(rows):
        gid = group_index.get((name, input_h))
        if gid is None:
            gid = group_index[name, input_h] = len(group_table)
            group_table.append(f"{name}\t{input_h}")
        pid = pred_index.get(pred_h)
        if pid is None:
            pid = pred_index[pred_h] = len(pred_table)
            pred_table.append(pred_h)
        group_ids[j] = gid
        pred_ids[j] = pid
        q_halt[j] = qh
        logprob[j] = lp
        stability[j] = st
        n_rows[j] = nr
        n_cols[j] = nc
    n_pred = len(pred_table)
    pred_grids = np.zeros((n_pred, spec.max_grid, spec.max_grid), dtype=np.uint8)
    pred_n_rows = np.zeros(n_pred, dtype=np.uint8)
    pred_n_cols = np.zeros(n_pred, dtype=np.uint8)
    for pid, pred_h in enumerate(pred_table):
        grid = grids.get(pred_h)
        if grid is None:
            continue
        gr, gc = _shape(grid)
        pred_grids[pid, :gr, :gc] = grid
        pred_n_rows[pid] = gr
        pred_n_cols[pid] = gc
    arrays: dict[str, NDArray[np.generic]] = {
        "group_id": group_ids,
        "pred_id": pred_ids,
        "q_halt": q_halt,
        "logprob": logprob,
        "stability": stability,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "group_table": np.array(group_table),
        "pred_table": np.array(pred_table),
        "pred_grids": pred_grids,
        "pred_n_rows": pred_n_rows,
        "pred_n_cols": pred_n_cols,
        "pass_ks": np.array(list(payload.pass_ks)),
    }
    if steps:
        arrays["converge_step"] = np.array([s[0] for s in steps], dtype=np.uint8)
        # A 30x30 grid can change 900 cells per step; uint8 would wrap.
        arrays["n_changes"] = np.array([s[1] for s in steps], dtype=np.uint16)
        arrays["q_halt_steps"] = np.array([s[2] for s in steps], dtype=np.float32)
        arrays["correct_step"] = np.array([s[3] for s in steps], dtype=np.uint8)
    path = Path(out_path).expanduser()
    if path.exists():
        logger.warning("[eval] overwriting existing signal dump at %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The stub's ``**kwds: ArrayLike`` collides with ``allow_pickle`` under unpacking.
    cast(Callable[..., object], np.savez_compressed)(path, **arrays)
    logger.info(
        "[eval] wrote signal dump (%d rows, %d groups, %d preds, %d bytes) to %s",
        n,
        len(group_table),
        n_pred,
        path.stat().st_size,
        path,
    )


def _shape(grid: NDArray[np.uint8]) -> tuple[int, int]:
    rows, cols = cast(tuple[int, int], grid.shape)
    return rows, cols


def _any_rank(has_payload: bool) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return has_payload
    gathered: list[bool | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, has_payload)
    return any(bool(value) for value in gathered)


def _gather_list[T](items: list[T]) -> list[T]:
    if not dist.is_available() or not dist.is_initialized():
        return items
    gathered: list[list[T] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, items)
    return [item for part in gathered if part is not None for item in part]


def _gather_grids(grids: dict[str, NDArray[np.uint8]]) -> dict[str, NDArray[np.uint8]]:
    if not dist.is_available() or not dist.is_initialized():
        return grids
    gathered: list[dict[str, NDArray[np.uint8]] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, grids)
    merged: dict[str, NDArray[np.uint8]] = {}
    for part in gathered:
        if part is not None:
            merged.update(part)
    return merged
