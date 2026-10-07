"""Prepared ARC2 trees with reference Philox task sampling."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypedDict, cast, override

from configgle import Fig

import numpy as np
import torch
import torch.distributed as dist

from priml.baselines.arcagi1.augmentation import ArcAugmentation, ArcSpec
from priml.baselines.arcagi1.data import ArcData
from priml.lib.custom_json import convert
from priml.paths import resolve_working_dir
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from typing import Protocol

    from numpy.typing import NDArray

    class _PreparedArcSplit(Protocol):
        inputs: torch.Tensor | NDArray[np.generic]
        labels: torch.Tensor | NDArray[np.generic]
        groups: NDArray[np.int64]
        puzzles: NDArray[np.int64]
        identifiers: NDArray[np.int64]
        ignore_label_id: int
        batch_size: int
        device: torch.device
        seed: int

    class _ArcDataSource(Protocol):
        def train_dataloader(self) -> _PreparedArcSplit: ...

        def eval_dataloader(self) -> _PreparedArcSplit: ...


class ArcBatches:
    """A resident prepared split, sampled by task or traversed in row order."""

    def __init__(
        self,
        data: _ArcDataSource,
        *,
        train: bool,
        epochs_per_iter: int,
        passes: int = 0,
    ) -> None:
        prepared = data.train_dataloader() if train else data.eval_dataloader()
        if not isinstance(prepared.inputs, torch.Tensor) or not isinstance(
            prepared.labels,
            torch.Tensor,
        ):
            raise TypeError("ARC2 requires device-resident prepared data")
        self.inputs = prepared.inputs
        self.labels = prepared.labels
        self.groups = cast(list[int], prepared.groups.tolist())
        self.puzzles = cast(list[int], prepared.puzzles.tolist())
        self.identifiers = torch.from_numpy(prepared.identifiers).to(prepared.device)
        self.ignore = prepared.ignore_label_id
        self.batch_size = prepared.batch_size
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        self.global_batch_size = self.batch_size * self.num_replicas
        self.seed = prepared.seed
        self.train = train
        self.epochs_per_iter = epochs_per_iter
        self.passes = passes
        self.active_pass: int | None = None
        self.next_batch = 0

    def __len__(self) -> int:
        """Return full-width training batches or padded evaluation batches."""
        if self.train:
            pass_index = (
                self.active_pass if self.active_pass is not None else self.passes + 1
            )
            return sum(1 for _ in self._plan(pass_index)) - self.next_batch
        return (self.puzzles[-1] + self.global_batch_size - 1) // self.global_batch_size

    def __iter__(self) -> Iterator[dict[str, object]]:
        """Yield source-ordered rank slices of complete global plans."""
        local = slice(self.rank * self.batch_size, (self.rank + 1) * self.batch_size)
        if not self.train:
            for start in range(0, self.puzzles[-1], self.global_batch_size):
                rows = list(
                    range(start, min(start + self.global_batch_size, self.puzzles[-1])),
                )[local]
                puzzle_ids = cast(
                    list[int],
                    (np.searchsorted(self.puzzles, rows, side="right") - 1).tolist(),
                )
                yield self._batch(rows, puzzle_ids)
            return
        if self.active_pass is None:
            self.passes += 1
            self.active_pass = self.passes
        for index, (rows, puzzles) in enumerate(self._plan(self.active_pass)):
            if index < self.next_batch:
                continue
            self.next_batch = index + 1
            yield self._batch(rows[local], puzzles[local])
        self.active_pass = None
        self.next_batch = 0

    def _plan(self, pass_index: int) -> Iterator[tuple[list[int], list[int]]]:
        """Replay the reference task draws without mutating loader state."""
        rng = np.random.Generator(np.random.Philox(seed=self.seed + pass_index))
        order = np.concatenate(
            [
                rng.permutation(len(self.groups) - 1)
                for _ in range(self.epochs_per_iter)
            ],
        )
        tasks = cast(list[int], order.tolist())
        rows: list[int] = []
        puzzles: list[int] = []
        for task in tasks:
            puzzle = int(rng.integers(self.groups[task], self.groups[task + 1]))
            start, stop = self.puzzles[puzzle : puzzle + 2]
            take = min(stop - start, self.global_batch_size - len(rows))
            rows.extend(
                cast(
                    list[int],
                    (start + rng.choice(stop - start, take, replace=False)).tolist(),
                ),
            )
            puzzles.extend([puzzle] * take)
            if len(rows) == self.global_batch_size:
                yield rows, puzzles
                rows = []
                puzzles = []
        # The reference drops an incomplete GLOBAL batch, before rank slicing.

    def _batch(self, rows: list[int], puzzles: list[int]) -> dict[str, object]:
        """Gather and pad a source-compatible batch."""
        media = self.inputs[rows]
        labels = self.labels[rows]
        labels = torch.where(
            labels == self.ignore,
            torch.full_like(labels, -100),
            labels,
        )
        identifiers = self.identifiers[puzzles].to(torch.int64)
        valid = len(rows)
        pad = self.batch_size - valid
        if pad:
            media = torch.cat([media, media.new_zeros(pad, media.shape[1])])
            labels = torch.cat([labels, labels.new_full((pad, labels.shape[1]), -100)])
            identifiers = torch.cat([identifiers, identifiers.new_zeros(pad)])
        return {
            "media": media,
            "label": labels,
            "puzzle_identifiers": identifiers,
            "valid_count": valid,
            "spatial_tags": torch.tensor([1, 0, 0], device=media.device).expand(
                self.batch_size,
                3,
            ),
        }


class Arc2Data:
    """ARC2 prepared-data boundary; constructing it never stages ARC1 data."""

    class Config(Fig["Arc2Data"]):
        """Prepared-data reader and reference iteration grouping."""

        augmentation: ArcAugmentation.Config = field(
            default_factory=ArcAugmentation.Config,
        )
        """Offline recipe used by the prepared-data reader."""

        spec: ArcSpec = field(default_factory=ArcSpec)
        """Dataset-owned packed-grid and token vocabulary configuration."""

        base_dir: Path | str | None = None
        """Resource root supplied by the training loop."""

        working_dir: Path | str = "/datasets/arc2concept-aug-1000"
        """Shared corpus resolved beneath the training loop's resource root."""

        batch_size: int = 256
        """Training examples per rank."""

        eval_batch_size: int | None = None
        """Evaluation examples per rank; None reuses the training width."""

        device: torch.device | str | None = None
        """Device holding the resident arrays."""

        seed: int = 0
        """Shared Philox seed before the per-pass increment."""

        num_tasks: int | None = None
        """Optional training-task prefix for installation checks."""

        num_eval_tasks: int | None = None
        """Optional evaluation-task prefix for installation checks."""

        epochs_per_iter: int = 4
        """Independently shuffled task passes concatenated before batching."""

        @override
        def finalize(self) -> Self:
            self.augmentation.spec = self.spec
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        prepared = ArcData.Config()
        prepared.spec = config.spec
        prepared.augmentation = config.augmentation
        prepared.working_dir = config.working_dir
        prepared.batch_size = config.batch_size
        prepared.eval_batch_size = config.eval_batch_size
        prepared.device = config.device
        prepared.seed = config.seed
        prepared.num_tasks = config.num_tasks
        prepared.num_eval_tasks = config.num_eval_tasks
        self.prepared = prepared.make()
        self.timer_epoch = CheckpointableStepTimer()
        self.passes = 0
        self.live: ArcBatches | None = None
        self._eval: ArcBatches | None = None
        self.active_pass: int | None = None
        self.next_batch = 0

    def train_dataloader(self) -> ArcBatches:
        """Build a sampled stream continuing the previous iteration count.

        Returns:
          stream: Rank-local batches from the shared global plan.

        """
        if self.live is None:
            self.live = ArcBatches(
                self.prepared,
                train=True,
                epochs_per_iter=self.config.epochs_per_iter,
                passes=self.passes,
            )
            self.live.active_pass = self.active_pass
            self.live.next_batch = self.next_batch
        return self.live

    def eval_dataloader(self) -> ArcBatches:
        """Visit every evaluation row, padding the last batch.

        Returns:
          stream: Ordered evaluation batches over the split loaded at the
            first call; later evaluations reuse it instead of re-reading disk.

        """
        if self._eval is None:
            self._eval = ArcBatches(self.prepared, train=False, epochs_per_iter=1)
        return self._eval

    class StateDict(TypedDict):
        """Iteration seed and epoch timer for checkpoint replay."""

        passes: int
        active_pass: int | None
        next_batch: int
        timer_epoch: CheckpointableStepTimer.StateDict

    def state_dict(self) -> StateDict:
        """Capture the next reference shuffle seed and epoch timer.

        Returns:
          state: Global-plan cursor and accumulated epoch timing.

        """
        return {
            "passes": self.live.passes if self.live is not None else self.passes,
            "active_pass": self.live.active_pass
            if self.live is not None
            else self.active_pass,
            "next_batch": self.live.next_batch
            if self.live is not None
            else self.next_batch,
            "timer_epoch": self.timer_epoch.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the reference shuffle counter.

        Args:
          state_dict: Saved global-plan cursor and epoch timer.

        """
        self.passes = convert(state_dict["passes"], int)
        active_pass = state_dict["active_pass"]
        self.active_pass = None if active_pass is None else convert(active_pass, int)
        self.next_batch = convert(state_dict["next_batch"], int)
        if self.live is not None:
            self.live.passes = self.passes
            self.live.active_pass = self.active_pass
            self.live.next_batch = self.next_batch
        self.timer_epoch.load_state_dict(
            convert(state_dict["timer_epoch"], dict[str, object]),
        )
