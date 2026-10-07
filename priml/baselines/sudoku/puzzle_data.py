"""Sudoku puzzle data as the TRM trainer reads it.

Data on disk is a flattened cross-product of (instance, augmentation) pairs,
written by ``scripts/prepare_data.py``:
  all__inputs.npy            : [n_samples, seq_len]
  all__labels.npy            : [n_samples, seq_len]
  all__group_indices.npy     : [n_instances+1] - instance boundaries
  dataset.json               : {"vocab_size": int, "seq_len": int}

Tokens: 0=pad, 1=blank cell, 2-10=digits 1-9 (vocab_size 11).

Batch contract (what the trainer consumes) -- every yielded batch is a dict:
  "media"              : int32 [batch_size, seq_len] input tokens
  "label"              : int32 [batch_size, seq_len] solution tokens
  "valid_count"        : int; number of real rows. The final batch of an
                         epoch is zero-padded up to batch_size so shapes
                         stay constant; padded rows are all-zero.
  "puzzle_identifiers" : int32 [batch_size], all zeros. Sudoku has no
                         per-puzzle identity; the constant id feeds the
                         model's single-entry puzzle embedding.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    NotRequired,
    Self,
    TypedDict,
    cast,
    override,
)

import functools
import itertools
import math

from configgle import Fig
from torch import Tensor

import numpy as np
import torch

from priml.baselines.sudoku.puzzle_spec import SudokuSpec
from priml.lib.codec import from_plain, loads
from priml.paths import resolve_working_dir
from priml.runtime import get_device


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from numpy.typing import NDArray


def augment_sudoku(
    inputs: Tensor,
    labels: Tensor,
    *,
    spec: SudokuSpec,
    digits_only: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Random dihedral symmetry + digit permutation for sudoku.

    Args:
        inputs: [B, 81] token ids.
        labels: [B, 81] token ids.
        spec: Dataset-owned grid geometry and token vocabulary.
        digits_only: If True, permute the digit tokens 2-10 only, keeping the
            blank marker (token 1) fixed -- the TRM-reference shuffle. False
            (legacy, bit-identical to prior runs) permutes tokens 1-9: the
            blank marker is mixed into the permutation and token 10 (digit 9)
            is a fixed point.
        generator: Optional dedicated RNG for the permutation and dihedral
            draws. None (legacy, bit-identical to prior runs) uses the global
            torch RNG. A seeded generator makes the augmentation stream
            reproducible and independent of ambient RNG state.

    Returns:
        inputs_aug: [B, 81] augmented input tokens.
        labels_aug: [B, 81] labels under the same permutation and symmetry.

    """
    n = math.isqrt(inputs.shape[1])
    if inputs.shape[1] != n * n:
        raise ValueError("Expected inputs.shape[1] == n * n.")
    if spec.grid_shape != (n, n):
        raise ValueError("Expected spec.grid_shape == (n, n).")
    B = inputs.shape[0]
    device = inputs.device

    # Random value permutation over n consecutive tokens. Legacy starts at 1
    # (an off-by-one that permutes the blank marker with digits 1-8 and pins
    # token 10); digits_only starts at 2, the actual digit tokens.
    lo = 2 if digits_only else 1
    perms = (
        torch.arange(spec.vocab_size, device=device, dtype=torch.long)
        .expand(B, -1)
        .clone()
    )
    rand_vals = torch.rand(B, n, device=device, generator=generator)
    perms[:, lo : n + lo] = rand_vals.argsort(dim=1) + lo

    inputs_aug = torch.gather(perms, 1, inputs.long()).to(inputs.dtype)
    labels_aug = torch.gather(perms, 1, labels.long()).to(labels.dtype)

    # Random dihedral symmetry (8 symmetries of the square).
    dihedral = _build_dihedral_indices(n, device)
    choice = torch.randint(0, 8, (B,), device=device, generator=generator)
    selected = dihedral[choice]
    inputs_aug = torch.gather(inputs_aug, 1, selected)
    labels_aug = torch.gather(labels_aug, 1, selected)

    return inputs_aug, labels_aug


@functools.cache
def _build_dihedral_indices(n: int, device: torch.device) -> Tensor:
    """8 dihedral symmetries of an n x n grid as index permutations."""
    base = torch.arange(n * n, device=device).reshape(n, n)
    symmetries: list[Tensor] = []
    for k in range(4):
        rotated = torch.rot90(base, k)
        symmetries.append(rotated.reshape(-1))
        symmetries.append(rotated.flip(1).reshape(-1))
    return torch.stack(symmetries)


class PuzzleSplit(TypedDict):
    """One split as loaded from disk; token 0 is always PAD."""

    inputs: Tensor
    labels: Tensor
    group_indices: Tensor
    vocab_size: int
    seq_len: int


class PuzzleBatch(TypedDict):
    """The batch contract in the module docstring."""

    media: Tensor
    label: Tensor
    valid_count: int
    puzzle_identifiers: Tensor


def load_puzzle_dataset(
    dataset_dir: Path,
    split: str,
    max_samples: int | None = None,
) -> PuzzleSplit:
    """Load puzzle data from disk.

    Args:
        dataset_dir: Dataset root containing train/ and test/ splits.
        split: Split subdirectory name ("train" or "test").
        max_samples: Hard cap on samples loaded; None loads everything.

    Returns:
        data: Dict with keys inputs, labels, group_indices, vocab_size,
            seq_len. Token 0 is always PAD across all puzzle types.

    """
    data_path = Path(dataset_dir).expanduser() / split
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_path}. Build it first with"
            " scripts/prepare_data.py.",
        )

    metadata_path = data_path / "dataset.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    metadata = from_plain(loads(metadata_path.read_text()), _Metadata)

    group_path = data_path / "all__group_indices.npy"
    if not group_path.exists():
        raise FileNotFoundError(f"Required file not found: {group_path}")

    group_indices = torch.from_numpy(cast("NDArray[np.int32]", np.load(group_path)))
    if max_samples is not None and max_samples < 0:
        raise ValueError(f"max_samples must be non-negative, got {max_samples}.")
    inputs_mmap = cast(
        "NDArray[np.integer]",
        np.load(data_path / "all__inputs.npy", mmap_mode="r"),
    )
    labels_mmap = cast(
        "NDArray[np.integer]",
        np.load(data_path / "all__labels.npy", mmap_mode="r"),
    )
    available = min(len(inputs_mmap), len(labels_mmap))
    n = min(max_samples, available) if max_samples is not None else None
    inputs = torch.from_numpy(np.array(inputs_mmap[:n])).to(torch.int32)
    labels = torch.from_numpy(np.array(labels_mmap[:n])).to(torch.int32)
    if n is not None:
        group_indices = torch.cat(
            [
                group_indices[group_indices < n],
                torch.tensor([n], dtype=group_indices.dtype),
            ],
        )
    return {
        "inputs": inputs,
        "labels": labels,
        "group_indices": group_indices,
        "vocab_size": metadata["vocab_size"],
        "seq_len": metadata["seq_len"],
    }


class _Metadata(TypedDict):
    """The prepared split's ``dataset.json``."""

    vocab_size: int
    seq_len: int


class PuzzleDataset:
    """Device-cached sudoku dataset yielding fixed-size batches.

    See the module docstring for the on-disk layout and the batch contract.
    Training batches optionally get on-the-fly augmentation (dihedral-8 +
    digit permutation) on top of the pre-augmented disk data; eval batches
    are never augmented. Two augmentation RNG modes are supported (see
    :attr:`Config.augment_seed`): the ambient global torch RNG (legacy,
    checkpointed via the trainer's "rng" state) and a dedicated generator
    that deliberately restarts from its seed on every fresh
    :meth:`train_dataloader` (the segmented-repro semantics).
    """

    class Config(Fig["PuzzleDataset"]):
        """Configuration for device-cached sudoku data loading."""

        spec: SudokuSpec = field(default_factory=SudokuSpec)
        """Puzzle geometry and token vocabulary."""

        base_dir: Path | str | None = None
        """Resource root; ``None`` resolves beneath ``/opt/scratch``. The
        trainer may inject an explicit root here."""

        working_dir: Path | str = "/datasets/sudoku-extreme"
        """Dataset root containing train/ and test/ splits. A ``str`` is a
        logical location resolved beneath ``base_dir`` at finalize; a ``Path``
        is literal and kept as given."""

        batch_size: int = 192
        """Samples per training batch (the reproduction recipe uses 384, set
        at the experiment level)."""

        eval_batch_size: int | None = None
        """Samples per eval batch; None falls back to batch_size."""

        device: torch.device | str | None = None
        """Device data loads onto; ``None`` is the loop's."""

        augment: bool = False
        """Apply on-the-fly dihedral + digit-permutation augmentation per
        TRAIN batch (eval is never augmented). The disk data is already
        pre-augmented 1000x; this adds per-batch variety on top."""

        augment_digits_only: bool = False
        """If True, ``augment`` permutes the digit tokens 2-10 only, keeping
        the blank marker (token 1) fixed -- the TRM-reference shuffle. False
        (legacy) permutes tokens 1-9 (blank mixed in, token 10 pinned); the
        early-ladder measured numbers depend on this branch."""

        augment_seed: int | None = None
        """Seed for a dedicated augmentation RNG stream. None (default) draws
        from the ambient global torch RNG -- the legacy mode, reproduced via
        the trainer's checkpointed "rng" state. When set, each
        ``train_dataloader()`` call creates a fresh generator that RESTARTS
        from this seed (and advances across epochs within one loader), so a
        fixed segment/resume schedule reproduces exactly."""

        seed: int | None = None
        """Train-shuffle seed. None (default) uses the global torch RNG; an
        int drives the per-epoch shuffle from a dedicated ``torch.Generator``
        seeded ``seed + epoch`` (deterministic, reproducible epoch order)."""

        num_instances: int | None = None
        """Cap on training instances loaded; None uses all instances."""

        eval_num_instances: int | None = None
        """Cap on eval instances loaded; None uses all instances."""

        eval_instance_indices: tuple[int, ...] = ()
        """Explicit strictly-ascending eval instance subset (global test-split
        indices) for survivor-only reruns. When set, eval iterates EXACTLY
        these instances in ascending order, so arrival position ``i`` maps to
        global index ``eval_instance_indices[i]`` and downstream metrics join
        positionally against the frozen list. Applied after the
        ``eval_num_instances`` prefix cap; every index must lie inside the
        loaded eval split. Training is untouched."""

        max_samples: int | None = None
        """Hard cap on total samples loaded from disk; None loads everything."""

        @override
        def finalize(self) -> Self:
            if isinstance(self.working_dir, str):
                self.working_dir = resolve_working_dir(
                    "/opt/scratch" if self.base_dir is None else self.base_dir,
                    self.working_dir,
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.eval_instance_indices:
            indices = config.eval_instance_indices
            if any(b <= a for a, b in itertools.pairwise(indices)):
                raise ValueError(
                    f"eval_instance_indices must be strictly ascending, got {indices}.",
                )
            if indices[0] < 0:
                raise ValueError(
                    f"eval_instance_indices must be non-negative, got {indices}.",
                )
        self.config = config
        self.dataset_dir = Path(config.working_dir)
        self.batch_size = config.batch_size
        self.eval_batch_size = config.eval_batch_size or config.batch_size
        # Completed train epochs, persisted across checkpoint resume so the
        # restored run continues the per-epoch shuffle sequence instead of
        # replaying the seeded epoch 0.
        self._train_epochs = 0
        self._active_train_iter: _PuzzleBatchIterator | None = None

    def train_dataloader(self) -> _PuzzleBatchIterator:
        """Build the (re-iterable) training batch stream.

        Returns:
          loader: Shuffled, augmented batch iterator over the train split.

        """
        # Single-writer invariant: snapshot any prior live iterator's epoch
        # before building a fresh one, so re-creation continues the sequence.
        if self._active_train_iter is not None:
            self._train_epochs = self._active_train_iter.epoch
        iter_obj = _PuzzleBatchIterator(
            dataset_dir=self.dataset_dir,
            device=self.config.device,
            batch_size=self.batch_size,
            num_instances=self.config.num_instances,
            max_samples=self.config.max_samples,
            seed=self.config.seed,
            epoch_offset=self._train_epochs,
            augment=self.config.augment,
            augment_digits_only=self.config.augment_digits_only,
            augment_seed=self.config.augment_seed,
            spec=self.config.spec,
        )
        self._active_train_iter = iter_obj
        return iter_obj

    def eval_dataloader(self) -> _PuzzleBatchIterator:
        """Build the eval batch stream: disk order, never augmented.

        Returns:
          loader: Batch iterator over the eval split in disk order.

        """
        iterator = _PuzzleBatchIterator(
            dataset_dir=self.dataset_dir,
            device=self.config.device,
            batch_size=self.eval_batch_size,
            train=False,
            shuffle=False,
            num_instances=self.config.eval_num_instances,
            max_samples=self.config.max_samples,
            spec=self.config.spec,
        )
        if self.config.eval_instance_indices:
            return _subset_eval_iterator(iterator, self.config.eval_instance_indices)
        return iterator

    class StateDict(TypedDict):
        """The completed-epoch counter; tolerated absent on load."""

        train_epochs: NotRequired[int]

    def state_dict(self) -> StateDict:
        """Snapshot resume state (the completed-epoch counter).

        Returns:
          state: Mapping with the completed-epoch counter.

        """
        # The live iterator's counter advances when an epoch's iteration
        # BEGINS (the generator body runs its shuffle, then increments,
        # before the first batch is yielded) -- so a mid-epoch checkpoint
        # resumes with the NEXT epoch's shuffle rather than replaying the
        # aborted one. Ported behavior; resume parity depends on it.
        epochs = (
            self._active_train_iter.epoch
            if self._active_train_iter is not None
            else self._train_epochs
        )
        return {"train_epochs": epochs}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore resume state produced by :meth:`state_dict`."""
        state = cast(PuzzleDataset.StateDict, state_dict)
        if "train_epochs" in state:
            self._train_epochs = state["train_epochs"]
            if self._active_train_iter is not None:
                self._active_train_iter.epoch = self._train_epochs


# Ported survivor-rerun semantics: the iterator's tensors and instance bounds are re-
# sliced so it yields exactly ``indices`` in ascending order (arrival position ``i`` =
# ``indices[i]``); eval iterates unshuffled, so the position->global-index mapping is
# stable across runs.
def _subset_eval_iterator(
    it: _PuzzleBatchIterator,
    indices: tuple[int, ...],
) -> _PuzzleBatchIterator:
    """Restrict a built eval iterator to an explicit instance subset."""
    if indices[-1] >= it.n_instances:
        raise ValueError(
            f"eval_instance_indices max {indices[-1]} is outside the eval "
            f"split ({it.n_instances} instances).",
        )
    idx = torch.tensor(indices, dtype=torch.int64)
    starts = it.instance_bounds[:-1].cpu()
    sizes = (it.instance_bounds[1:] - it.instance_bounds[:-1]).cpu()
    rows = torch.cat(
        [torch.arange(int(starts[i]), int(starts[i]) + int(sizes[i])) for i in indices],
    ).to(it.inputs.device)
    it.inputs = it.inputs[rows]
    it.labels = it.labels[rows]
    new_sizes = sizes[idx]
    it.instance_bounds = torch.cat(
        [torch.zeros(1, dtype=torch.int64), new_sizes.cumsum(0)],
    ).to(it.instance_bounds.device)
    it.n_instances = len(indices)
    return it


class _PuzzleBatchIterator:
    """Iterates over device-resident puzzle data in batches.

    Yields the batch contract documented in the module docstring. The last
    batch of an epoch is zero-padded to batch_size with valid_count <
    batch_size. Augmentation (train only) is applied to the full padded
    batch -- the RNG draw shapes are [batch_size, ...] regardless of
    valid_count, and padded rows stay all-zero because token 0 is a fixed
    point of every permutation.
    """

    def __init__(
        self,
        dataset_dir: Path,
        device: torch.device | str | None,
        batch_size: int,
        *,
        train: bool = True,
        shuffle: bool = True,
        num_instances: int | None = None,
        max_samples: int | None = None,
        seed: int | None = None,
        epoch_offset: int = 0,
        augment: bool = False,
        augment_digits_only: bool = False,
        augment_seed: int | None = None,
        spec: SudokuSpec,
    ):
        if epoch_offset < 0:
            raise ValueError(f"epoch_offset must be non-negative, got {epoch_offset}.")
        split = "train" if train else "test"
        data = load_puzzle_dataset(dataset_dir, split, max_samples=max_samples)
        instance_bounds = data["group_indices"]
        self.spec = spec
        if data["vocab_size"] != self.spec.vocab_size:
            raise ValueError("Prepared vocabulary does not match dataset spec.")
        if data["seq_len"] != math.prod(self.spec.grid_shape):
            raise ValueError("Prepared grid does not match dataset spec.")
        self.vocab_size = self.spec.vocab_size
        self.seq_len = data["seq_len"]

        if num_instances is not None:
            instance_bounds = instance_bounds[: num_instances + 1]
            max_samples = int(instance_bounds[-1])

        self.device = get_device(device)
        self.inputs = data["inputs"][:max_samples].to(self.device)
        self.labels = data["labels"][:max_samples].to(self.device)
        self.instance_bounds = instance_bounds.to(self.device)
        self.n_instances = len(instance_bounds) - 1
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.augment = augment and train
        self.augment_digits_only = augment_digits_only
        # Dedicated augmentation stream: created fresh per iterator (i.e. per
        # train_dataloader() call), so it deliberately RESTARTS from the seed
        # on every new loader/process while advancing across epochs within
        # one loader -- the segmented-repro semantics. None falls back to the
        # ambient global torch RNG.
        self._augment_generator: torch.Generator | None = None
        if self.augment and augment_seed is not None:
            self._augment_generator = torch.Generator(device=self.device)
            self._augment_generator.manual_seed(augment_seed)
        self.epoch: int = epoch_offset
        """Epochs begun; folded into the seeded generator so each epoch
        reshuffles differently yet reproducibly. Starts at ``epoch_offset`` so
        a restored run continues the sequence instead of replaying it."""

    # When ``seed`` is set, returns a dedicated ``torch.Generator`` seeded by ``seed``
    # folded with the epoch index, so a given epoch's shuffle is deterministic and
    # independent of ambient RNG state. When ``seed`` is None, returns None and the
    # caller falls back to the global torch RNG (the original single-process path, bit-
    # for-bit unchanged).
    def _shuffle_generator(self, device: torch.device) -> torch.Generator | None:
        """Per-epoch generator for seeded shuffles, or None for the global RNG."""
        if self.seed is None:
            return None
        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed + self.epoch)
        return gen

    def __iter__(self) -> Iterator[PuzzleBatch]:
        indices = self._order()
        self.epoch += 1
        n_samples = len(indices)

        for i in range(0, n_samples, self.batch_size):
            batch_idx = indices[i : i + self.batch_size]
            valid = len(batch_idx)
            inputs = self.inputs[batch_idx]
            labels = self.labels[batch_idx]
            if valid < self.batch_size:
                pad = self.batch_size - valid
                inputs = torch.cat([inputs, inputs.new_zeros(pad, inputs.shape[1])])
                labels = torch.cat([labels, labels.new_zeros(pad, labels.shape[1])])
            if self.augment:
                # Augment AFTER padding: the RNG draws are shaped by the full
                # batch_size (load-bearing for stream reproducibility), and
                # zero pad rows pass through unchanged.
                inputs, labels = augment_sudoku(
                    inputs,
                    labels,
                    spec=self.spec,
                    digits_only=self.augment_digits_only,
                    generator=self._augment_generator,
                )
            yield {
                "media": inputs,
                "label": labels,
                "valid_count": valid,
                "puzzle_identifiers": torch.zeros(
                    inputs.shape[0],
                    dtype=torch.int32,
                    device=inputs.device,
                ),
            }

    def __len__(self) -> int:
        total = int(self.instance_bounds[-1] - self.instance_bounds[0])
        return math.ceil(total / self.batch_size)

    def _order(self) -> Tensor:
        """Return this epoch's sample order: disk order, or the two-level shuffle."""
        device = self.instance_bounds.device
        if not self.shuffle:
            return torch.arange(
                int(self.instance_bounds[0]),
                int(self.instance_bounds[-1]),
                device=device,
            )
        gen = self._shuffle_generator(device)
        starts = self.instance_bounds[:-1]
        # One host sync for every bound, not two per instance.
        bounds = zip(
            from_plain(starts.tolist(), list[int]),
            from_plain((self.instance_bounds[1:] - starts).tolist(), list[int]),
            strict=True,
        )
        # Two-level shuffle with a pinned draw order (per-instance randperms,
        # then one global randperm over all samples). This is the internal
        # bundle-grouped shuffle specialized to bundle size 1 -- the only
        # size any ported experiment uses -- with an identical RNG draw
        # sequence and identical resulting order.
        indices = torch.cat(
            [
                torch.arange(s, s + n, device=device)[
                    torch.randperm(n, device=device, generator=gen)
                ]
                for s, n in bounds
            ],
        )
        return indices[torch.randperm(len(indices), device=device, generator=gen)]
