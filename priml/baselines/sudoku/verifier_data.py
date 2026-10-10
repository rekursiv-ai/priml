"""On-device verifier training data: the generated stream over the harvest corpus.

Harvest shard contract (produced by
:class:`~priml.baselines.sudoku.harvest.Harvest`): files ``shard-*.npz``
under ``harvest_dir``, each with fields ``original`` / ``candidate``
(``[N, 81]`` token rows), ``flat_view_id`` (``[N]`` row index into the train
split; invariant ``original == train_inputs[flat_view_id]``), and
``base_group_id`` (``[N]`` in ``[0, 8)``; 8 base puzzle groups per shard,
globalized as ``base_group_id + 8 * shard_index`` in sorted-filename order).
The real corpus carries >= 800 global groups; groups ``[0, 700)`` train the
verifier (the default ``train_group_end``; scale it down for smaller corpora).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, TypedDict, override

import json

from configgle import Fig
from torch import Tensor
from treekle import ReadError, from_plain, loads

import numpy as np
import torch

from priml.baselines.sudoku.puzzle_data import (
    dihedral_indices,
    load_puzzle_dataset,
)
from priml.baselines.sudoku.runner import file_sha256, open_npz
from priml.paths import resolve_working_dir
from priml.runtime import get_device


if TYPE_CHECKING:
    from collections.abc import Iterator


class VerifierData:
    """On-device verifier data: the generated train stream.

    Train batches mix, on the fly and on device:

    - generated positives: train-split solutions under fresh
      validity-preserving augmentation (digit permutation + dihedral
      symmetry);
    - transposition-swap negatives: digit-histogram-preserving cell swaps
      (kills count-based shortcuts);
    - Hamming-k corruption negatives: k non-given cells of a solution
      replaced with different uniform digits (any single-cell change to a
      complete solution breaks a row, column, AND box);
    - optional clue-contradiction negatives: a rule-valid solution from a
      sibling row presented under this row's puzzle;
    - real solver candidates from the harvest corpus (see the module
      docstring for the shard contract), restricted to the train groups.

    Labels are exactly ``candidate == solution`` -- no Sudoku rule predicate
    anywhere. Every batch dict carries ``puzzle`` / ``candidate`` /
    ``label`` / ``cell_label`` / ``cell_mask`` tensors.
    """

    class Config(Fig["VerifierData"]):
        """Data-axis knobs of the verifier candidate batch."""

        base_dir: Path | str | None = None
        """Resource root; ``None`` resolves at construction beneath
        ``/opt/scratch``."""

        working_dir: Path | str = "/datasets/sudoku-extreme"
        """Sudoku dataset root with train/ and test/ splits; resolved beneath
        ``base_dir``."""

        harvest_dir: str | Path = "/runs/harvest/harvest"
        """Directory of ``shard-*.npz`` real solver-candidate shards (the
        harvest corpus; module docstring documents the contract). A relative
        logical path resolves beneath ``base_dir``; the default is where
        :class:`~Harvest` writes at its default ``experiment_name="harvest"``."""

        batch_size: int = 512
        """Training rows per batch."""

        steps_per_epoch: int = 250
        """Generated train batches per epoch (stream is infinite)."""

        real_fraction: float = 0.3
        """Batch fraction drawn from real harvest solver candidates."""

        positive_fraction: float = 0.3
        """Batch fraction of clean solutions (generated positives)."""

        swap_fraction: float = 0.15
        """Batch fraction of transposition-swap negatives."""

        contradiction_fraction: float = 0.0
        """Batch fraction of sibling-row clue-contradiction negatives,
        taken out of the Hamming-corruption remainder."""

        hamming_choices: tuple[int, ...] = (1, 1, 1, 2, 2, 3, 4, 8, 16)
        """Corruption sizes sampled uniformly (repeats skew the mix)."""

        swap_choices: tuple[int, ...] = (1, 1, 2, 3)
        """Swap-pair counts sampled uniformly."""

        train_group_end: int = 700
        """Harvest base groups [0, end) train the verifier. The default
        assumes the full >= 800-group harvest; scale it down for smaller
        corpora."""

        device: torch.device | str | None = None
        """Device for cached data and generation."""

        seed: int = 0
        """Train-stream seed (folded with the epoch index)."""

        @override
        def finalize(self) -> Self:
            base = self.base_dir if self.base_dir is not None else Path("/opt/scratch")
            if isinstance(self.working_dir, str):
                self.working_dir = resolve_working_dir(base, self.working_dir)
            if isinstance(self.harvest_dir, str):
                self.harvest_dir = resolve_working_dir(base, self.harvest_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = get_device(config.device)
        if config.steps_per_epoch < 1:
            raise ValueError(
                f"steps_per_epoch must be >= 1, got {config.steps_per_epoch}.",
            )
        fractions = (
            config.real_fraction,
            config.positive_fraction,
            config.swap_fraction,
            config.contradiction_fraction,
        )
        if min(fractions) < 0 or sum(fractions) > 1:
            raise ValueError(f"batch fractions {fractions} must be in [0, 1].")
        if config.train_group_end < 0:
            raise ValueError(
                f"train_group_end must be >= 0, got {config.train_group_end}.",
            )

        # ``finalize`` already resolved every logical path beneath ``base_dir``;
        # read the finalized values verbatim. Re-resolving here would rebase an
        # already-absolute path a second time (``base / base / ...``).
        self.dataset_dir = Path(config.working_dir)
        self.harvest_dir = Path(config.harvest_dir)

        train = load_puzzle_dataset(self.dataset_dir, "train")
        self.train_inputs = train["inputs"].to(self.device)
        self.train_labels = train["labels"].to(self.device)

        harvest = _load_harvest(self.harvest_dir, device=self.device)
        self.harvest_puzzle, self.harvest_candidate, flat_view, groups = harvest
        if int(flat_view.max()) >= len(self.train_labels):
            raise ValueError("harvest flat_view_id exceeds the train file.")
        if not torch.equal(self.harvest_puzzle, self.train_inputs[flat_view].long()):
            raise ValueError(
                "harvest original rows must equal train inputs[flat_view_id]; "
                "the harvest/train-file pairing contract is broken.",
            )
        self.harvest_solution = self.train_labels[flat_view].long()
        self._harvest_train_rows = (groups < config.train_group_end).nonzero()[:, 0]
        if config.real_fraction > 0 and not len(self._harvest_train_rows):
            raise ValueError(
                "real_fraction > 0 but no harvest rows fall in the train "
                f"groups [0, {config.train_group_end}).",
            )

        self._epoch = 0

    def train_dataloader(self) -> _VerifierTrainIterator:
        """Seeded generated-batch stream; epoch index folds into the seed."""
        iterator = _VerifierTrainIterator(self, epoch=self._epoch)
        self._epoch += 1
        return iterator

    def generate_batch(self, generator: torch.Generator) -> dict[str, Tensor]:
        """Generate one mixed training batch on device.

        Args:
          generator: Dedicated RNG stream for every draw in the batch.

        Returns:
          batch: Dict of ``puzzle`` / ``candidate`` / ``label`` /
            ``cell_label`` / ``cell_mask`` tensors; rows are
            positives, then swaps, then Hamming negatives, then real harvest
            candidates.

        """
        cfg = self.config
        n_real, n_positive, n_swap, n_contradiction = _batch_partition_counts(
            cfg.batch_size,
            (
                cfg.real_fraction,
                cfg.positive_fraction,
                cfg.swap_fraction,
                cfg.contradiction_fraction,
            ),
        )
        n_hamming = cfg.batch_size - n_real - n_positive - n_swap - n_contradiction

        rows = torch.randint(
            len(self.train_inputs),
            (n_positive + n_swap + n_hamming,),
            device=self.device,
            generator=generator,
        )
        puzzle = self.train_inputs[rows].long()
        solution = self.train_labels[rows].long()
        candidate = solution.clone()
        given = puzzle >= 2

        swap_slice = slice(n_positive, n_positive + n_swap)
        pairs = _choices(cfg.swap_choices, n_swap, generator=generator)
        candidate[swap_slice] = _swap_cells(
            candidate[swap_slice],
            given[swap_slice],
            pairs=pairs,
            generator=generator,
        )
        hamming_slice = slice(n_positive + n_swap, None)
        ks = _choices(cfg.hamming_choices, n_hamming, generator=generator)
        candidate[hamming_slice] = _corrupt_hamming(
            candidate[hamming_slice],
            given[hamming_slice],
            k=ks,
            generator=generator,
        )

        parts_puzzle = [puzzle]
        parts_candidate = [candidate]
        parts_solution = [solution]
        if n_contradiction:
            contradiction = self._contradiction_rows(
                n_contradiction,
                generator=generator,
            )
            parts_puzzle.append(contradiction[0])
            parts_candidate.append(contradiction[1])
            parts_solution.append(contradiction[2])
        if n_real:
            real = self._harvest_train_rows[
                torch.randint(
                    len(self._harvest_train_rows),
                    (n_real,),
                    device=self.device,
                    generator=generator,
                )
            ]
            parts_puzzle.append(self.harvest_puzzle[real])
            parts_candidate.append(self.harvest_candidate[real])
            parts_solution.append(self.harvest_solution[real])

        puzzle = torch.cat(parts_puzzle)
        candidate = torch.cat(parts_candidate)
        solution = torch.cat(parts_solution)
        puzzle, candidate, solution = _augment_consistent(
            puzzle,
            candidate,
            solution,
            generator=generator,
        )
        return _finish_batch(puzzle, candidate, solution)

    def _contradiction_rows(
        self,
        count: int,
        *,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return sibling solutions under unrelated puzzle givens."""
        rows = torch.randint(
            len(self.train_inputs),
            (count,),
            device=self.device,
            generator=generator,
        )
        siblings = torch.randint(
            len(self.train_inputs),
            (count,),
            device=self.device,
            generator=generator,
        )
        return (
            self.train_inputs[rows].long(),
            self.train_labels[siblings].long(),
            self.train_labels[rows].long(),
        )


class _VerifierTrainIterator:
    """One epoch of generated batches under a seeded generator."""

    def __init__(self, dataset: VerifierData, *, epoch: int) -> None:
        self._dataset = dataset
        self._epoch = epoch

    def __len__(self) -> int:
        return self._dataset.config.steps_per_epoch

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        generator = torch.Generator(device=self._dataset.device)
        generator.manual_seed(self._dataset.config.seed + self._epoch)
        for _ in range(self._dataset.config.steps_per_epoch):
            yield self._dataset.generate_batch(generator)


def _finish_batch(
    puzzle: Tensor,
    candidate: Tensor,
    solution: Tensor,
) -> dict[str, Tensor]:
    """Attach exact labels: ``label = (candidate == solution)`` per row."""
    cell_label = candidate != solution
    return {
        "puzzle": puzzle,
        "candidate": candidate,
        "label": (~cell_label.any(dim=-1)).float(),
        "cell_label": cell_label.float(),
        "cell_mask": torch.ones_like(cell_label),
    }


class _HarvestShardEntry(TypedDict):
    """One shard's line in the harvest manifest."""

    file: str
    rows: int
    sha256: str


class _HarvestManifest(TypedDict):
    """The fields of ``manifest.json`` the loader binds shards to."""

    schema_version: Literal[1]
    groups_per_shard: int
    shards: list[_HarvestShardEntry]
    shard_count: int
    row_count: int


def _load_harvest(
    harvest_dir: Path,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load the manifest-bound harvest shards and globalize their groups."""
    manifest_path = harvest_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"harvest manifest not found: {manifest_path}.")
    try:
        manifest = from_plain(loads(manifest_path.read_text()), _HarvestManifest)
    except (json.JSONDecodeError, ReadError) as error:
        raise ValueError(
            f"invalid harvest manifest {manifest_path}: {error}",
        ) from error
    groups_per_shard = manifest["groups_per_shard"]
    entries = manifest["shards"]
    declared_names = [entry["file"] for entry in entries]
    if any(Path(name).name != name for name in declared_names):
        raise ValueError(
            f"harvest manifest {manifest_path} has an invalid shard name.",
        )
    actual_names = sorted(path.name for path in harvest_dir.glob("shard-*.npz"))
    if actual_names != sorted(declared_names):
        raise ValueError(
            f"harvest manifest shard set mismatch: declared={declared_names}, "
            f"actual={actual_names}.",
        )
    if declared_names != actual_names:
        raise ValueError(
            "harvest manifest shards are not in canonical shard order: "
            f"declared={declared_names}, canonical={actual_names}.",
        )
    originals: list[Tensor] = []
    candidates: list[Tensor] = []
    flat_views: list[Tensor] = []
    groups: list[Tensor] = []
    row_count = 0
    for index, entry in enumerate(entries):
        shard = harvest_dir / entry["file"]
        if file_sha256(shard) != entry["sha256"]:
            raise ValueError(f"harvest shard digest mismatch: {shard}.")
        with open_npz(shard) as data:
            base_groups = np.array(data["base_group_id"], dtype=np.int64)
            if base_groups.size and (
                int(base_groups.min()) < 0 or int(base_groups.max()) >= groups_per_shard
            ):
                raise ValueError(
                    f"shard {shard.name}: base_group_id must lie in "
                    f"[0, {groups_per_shard}).",
                )
            originals.append(torch.from_numpy(np.array(data["original"])))
            candidates.append(torch.from_numpy(np.array(data["candidate"])))
            flat_views.append(torch.from_numpy(np.array(data["flat_view_id"])))
            groups.append(torch.from_numpy(base_groups) + groups_per_shard * index)
            rows = len(data["original"])
        if entry["rows"] != rows:
            raise ValueError(f"harvest shard row count mismatch: {shard}.")
        row_count += rows
    if manifest["shard_count"] != len(entries) or manifest["row_count"] != row_count:
        raise ValueError(f"harvest manifest aggregate count mismatch: {manifest_path}.")
    return (
        torch.cat(originals).to(device).long(),
        torch.cat(candidates).to(device).long(),
        torch.cat(flat_views).to(device).long(),
        torch.cat(groups).to(device),
    )


def _batch_partition_counts(
    batch_size: int,
    fractions: tuple[float, ...],
) -> tuple[int, ...]:
    """Round mixture sizes, trimming only pathological small-batch overflow."""
    counts = [round(batch_size * fraction) for fraction in fractions]
    overflow = max(0, sum(counts) - batch_size)
    for index in range(len(counts) - 1, -1, -1):
        removed = min(counts[index], overflow)
        counts[index] -= removed
        overflow -= removed
        if not overflow:
            break
    return tuple(counts)


def _choices(
    options: tuple[int, ...],
    count: int,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Sample ``count`` values uniformly from ``options``."""
    table = torch.tensor(options, device=generator.device)
    idx = torch.randint(
        len(options),
        (count,),
        device=generator.device,
        generator=generator,
    )
    return table[idx]


# The replacement is a uniform nonzero offset in digit space, so every corrupted cell is
# guaranteed to change (a Hamming-k corruption has exactly k wrong cells, keeping the
# negative strata pure).
def _corrupt_hamming(
    candidate: Tensor,
    given: Tensor,
    *,
    k: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Replace k random non-given cells per row with DIFFERENT digit tokens."""
    if not len(candidate):
        return candidate
    b, n = candidate.shape
    noise = (
        torch.rand(b, n, device=candidate.device, generator=generator)
        + given.float() * 10
    )
    order = noise.argsort(dim=1)
    take = torch.arange(n, device=candidate.device).expand(b, n) < k.unsqueeze(1)
    chosen = torch.zeros_like(given)
    chosen.scatter_(1, order, take)
    offsets = torch.randint(
        1,
        9,
        (b, n),
        device=candidate.device,
        generator=generator,
        dtype=candidate.dtype,
    )
    digits = ((candidate - 2 + offsets) % 9) + 2
    return torch.where(chosen & ~given, digits, candidate)


def _swap_cells(
    candidate: Tensor,
    given: Tensor,
    *,
    pairs: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Swap the digits of random non-given cell pairs (histogram-preserving)."""
    if not len(candidate):
        return candidate
    b, n = candidate.shape
    max_pairs = int(pairs.max()) if len(pairs) else 0
    if max_pairs == 0:
        return candidate
    noise = (
        torch.rand(b, n, device=candidate.device, generator=generator)
        + given.float() * 10
    )
    order = noise.argsort(dim=1)
    first = order[:, 0 : 2 * max_pairs : 2]
    second = order[:, 1 : 2 * max_pairs : 2]
    active = torch.arange(max_pairs, device=candidate.device).expand(
        b,
        max_pairs,
    ) < pairs.unsqueeze(1)
    out = candidate.clone()
    a = out.gather(1, first)
    c = out.gather(1, second)
    out.scatter_(1, first, torch.where(active, c, a))
    out.scatter_(1, second, torch.where(active, a, c))
    return out


# Both transforms preserve Sudoku validity and clue consistency, so every label -- grid
# and per-cell -- is invariant.
def _augment_consistent(
    puzzle: Tensor,
    candidate: Tensor,
    solution: Tensor,
    *,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """One shared digit permutation + dihedral symmetry across the triple."""
    b = puzzle.shape[0]
    device = puzzle.device
    perms = torch.arange(11, device=device).expand(b, -1).clone()
    perms[:, 2:11] = (
        torch.rand(b, 9, device=device, generator=generator).argsort(dim=1) + 2
    )
    dihedral = dihedral_indices(9, device)
    choice = torch.randint(0, 8, (b,), device=device, generator=generator)
    selected = dihedral[choice]

    def apply(tokens: Tensor) -> Tensor:
        permuted = torch.gather(perms, 1, tokens.long())
        return torch.gather(permuted, 1, selected)

    return apply(puzzle), apply(candidate), apply(solution)
