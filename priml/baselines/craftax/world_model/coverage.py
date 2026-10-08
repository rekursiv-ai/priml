"""Coverage counters per shard and the report that decides when the corpus is enough.

Counters come from the token frames, stored or, in a replay shard, replayed
(``snapshots.replay_episodes``), and the meta summaries, one set per closed
shard, cached beside the index and summed across shards. Values and transition
signatures count the training split only; reach counts both splits.

A **transition signature** is the floor, the executed action, the eight values
of the cell the player faces, the mob class (none, melee, passive, or
ranged) of each of the four cells beside the player, the reward, the terminal
flag, and the sign of the change of each inventory field (aux 0–21: materials,
tools, enchantments, bow, and potions) from ``o_t`` to ``o_{t+1}``. A terminal
decision has no next frame, so its inventory signs are all zero. Signatures are
exact: two int64 words, never a hash.

The **report** applies the design's three sufficiency checks:

1. reach: every floor is reached by enough training and validation episodes;
2. values: no cell-field value, auxiliary value, or achievement is rare, that
   is, seen at least once but fewer than ``min_count`` times on a floor;
3. missing mass: on every visited floor, the Good–Turing unseen mass
   (signatures seen once over transitions) is below ``max_unseen_mass``, and
   the last ``increment_decisions`` of the corpus added new signatures for
   under ``max_new_rate`` of their transitions.

Achievements come from the summary key ``achievements``, the sorted list of
achievement indices 0–66 unlocked by the episode; a summary of either split
without one fails the count. The list carries no floor, so achievements are
counted over the whole corpus.

A shard is counted in whole-episode chunks, decoded episode by episode, in a
process pool; threads do not help, because the per-episode work holds the GIL
too often. The counters are cached under ``index.shard_key``, since they read
all three files: actions and split from the records, tokens from the frames,
and achievements from the summaries.

Four workers is the default; 4 through 8 tie within noise. Cold counts of a
synthetic 10M-decision shard (2,466 episodes, 40 chunks of 250k decisions,
nearly every transition signature distinct) on an 18-core Mac took 5.2-8.4 s
with 4 workers at load average 5-12, under the 10 s budget, and 5.8-6.4 s
with 8. At load average 10-18 the same counts took 17.6-17.8 s with 2 workers,
13.2-14.1 s with 3, 12.0-16.2 s (median 13.8 s, 5 runs) with 4, and
17.0-22.3 s with 8. Serially, decoding took 7.6 s and counting 17.2 s; with 4
workers, pool start-up took 2.3 s and the final merge 2.3 s.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

import dataclasses

from torch import Tensor

import torch

from priml.baselines.craftax.game.state import (
    NUM_ACHIEVEMENTS,
    OBS_TILE_CHANNELS,
)
from priml.baselines.craftax.world_model.archive import (
    Episode,
    ManifestLine,
)
from priml.baselines.craftax.world_model.index import (
    FLOOR_AUX,
    FLOORS,
    load_tensors,
    map_chunks,
    save_tensors,
    shard_key,
)
from priml.baselines.craftax.world_model.schema import (
    craftax_schema,
    number_id,
)
from priml.baselines.craftax.world_model.snapshots import replay_episodes
from priml.lib.codec import PlainTree, ReadError, from_plain


INVENTORY_AUX: Final = 22
"""Leading auxiliary fields whose change signs enter a signature."""

AUX_VALUES: Final = 261
"""Auxiliary token values ``0…260`` counted per field."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Coverage:
    """Mergeable coverage counters of one or more shards.

    Attributes:
      decisions: Training decisions per floor, int64 ``[9]``.
      cells: Training count of each cell-field value per floor, int64
        ``[9, 8, 64]``.
      aux: Training count of each auxiliary value per floor, int64
        ``[9, 51, 261]``.
      achievements: Training episodes that unlocked each achievement, int64
        ``[67]``.
      reach: Episodes per split (0 training, 1 validation) that reached each
        floor, int64 ``[2, 9]``.
      signatures: Distinct training transition signatures, sorted, int64
        ``[M, 2]``.
      signature_counts: Transitions with each signature, int64 ``[M]``.

    """

    decisions: Tensor
    cells: Tensor
    aux: Tensor
    achievements: Tensor
    reach: Tensor
    signatures: Tensor
    signature_counts: Tensor


def transition_signatures(episode: Episode) -> Tensor:
    """Return the transition signature of every decision of one episode.

    Args:
      episode: A complete episode.

    Returns:
      signatures: Two int64 words per decision, ``[T, 2]``. Word 0 holds the
        floor in bits 53–56, so ``signatures[:, 0] >> 53`` is the floor.

    """
    aux = episode.aux.long()
    decisions = len(aux)
    # Beside the player at slot 49 (r4c5): left, right, up, down, the order of
    # the facing flags in aux 31–34.
    beside = torch.tensor([48, 50, 38, 60])
    facing = aux[:, 31:35].argmax(1)
    faced = episode.cells[torch.arange(decisions), beside[facing]].long()
    widths = torch.tensor(
        [(f.size - 1).bit_length() for f in craftax_schema().cell_fields],
    )
    shifts = torch.cumsum(widths, 0) - widths
    faced_code = (faced << shifts).sum(1)
    near = episode.cells[:, beside].long()
    mob = torch.where(near[..., 3] > 0, 1, 0)
    mob = torch.where((mob == 0) & (near[..., 4] > 0), 2, mob)
    mob = torch.where((mob == 0) & (near[..., 5] > 0), 3, mob)
    mob_code = (mob << torch.tensor([0, 2, 4, 6])).sum(1)
    change = torch.zeros(decisions, INVENTORY_AUX, dtype=torch.int64)
    change[:-1] = torch.sign(aux[1:, :INVENTORY_AUX] - aux[:-1, :INVENTORY_AUX])
    inventory_code = ((change + 1) << 2 * torch.arange(INVENTORY_AUX)).sum(1)
    high = (
        (aux[:, FLOOR_AUX] << 53)
        | (episode.actions.long() << 47)
        | ((episode.reward.long() + 1) << 39)
        | (episode.done.long() << 38)
        | (mob_code << 30)
        | faced_code
    )
    return torch.stack([high, inventory_code], dim=1)


def count_shard(episodes: Sequence[Episode]) -> Coverage:
    """Count one shard's coverage.

    Args:
      episodes: The shard's episodes.

    Returns:
      coverage: Its counters.

    """
    decisions = torch.zeros(FLOORS, dtype=torch.int64)
    cells = torch.zeros(FLOORS, OBS_TILE_CHANNELS, 64, dtype=torch.int64)
    aux = torch.zeros(FLOORS * 51 * AUX_VALUES, dtype=torch.int64)
    achievements = torch.zeros(NUM_ACHIEVEMENTS, dtype=torch.int64)
    reach = torch.zeros(2, FLOORS, dtype=torch.int64)
    signatures = [torch.empty(0, 2, dtype=torch.int64)]
    for episode in episodes:
        unlocked = _achievements(episode.summary)
        floor = episode.aux[:, FLOOR_AUX].long()
        visits = torch.bincount(floor, minlength=FLOORS)
        reach[episode.receipt.split] += visits.clamp(max=1)
        if episode.receipt.split:
            continue
        decisions += visits
        _count_cells(cells, episode.cells, floor)
        fields = floor[:, None].int() * 51 + torch.arange(51, dtype=torch.int32)
        keys = fields * AUX_VALUES + episode.aux
        aux += torch.bincount(keys.flatten(), minlength=len(aux))
        achievements[torch.tensor(unlocked, dtype=torch.int64)] += 1
        signatures.append(transition_signatures(episode))
    rows = torch.cat(signatures)
    unique, counts = _unique_rows(rows, torch.ones(len(rows), dtype=torch.int64))
    return Coverage(
        decisions=decisions,
        cells=cells,
        aux=aux.reshape(FLOORS, 51, AUX_VALUES),
        achievements=achievements,
        reach=reach,
        signatures=unique,
        signature_counts=counts,
    )


def merge(parts: Sequence[Coverage]) -> Coverage:
    """Sum the counters of several shards.

    Args:
      parts: Counters to merge; at least one.

    Returns:
      coverage: Their sum.

    """
    unique, counts = _unique_rows(
        torch.cat([p.signatures for p in parts]),
        torch.cat([p.signature_counts for p in parts]),
    )
    return Coverage(
        decisions=torch.stack([p.decisions for p in parts]).sum(0),
        cells=torch.stack([p.cells for p in parts]).sum(0),
        aux=torch.stack([p.aux for p in parts]).sum(0),
        achievements=torch.stack([p.achievements for p in parts]).sum(0),
        reach=torch.stack([p.reach for p in parts]).sum(0),
        signatures=unique,
        signature_counts=counts,
    )


def load_coverage(
    directory: Path,
    line: ManifestLine,
    *,
    coverage_dir: Path,
    workers: int = 4,
    chunk_decisions: int = 250_000,
) -> Coverage:
    """Return a shard's counters from the cache, counting and caching them if absent.

    Args:
      directory: The shard's worker directory.
      line: The shard's manifest line.
      coverage_dir: Directory of cached counters, created if absent.
      workers: Processes that count it; see ``index.map_chunks``. Four is the
        measured fastest; see the module docstring.
      chunk_decisions: Decisions each process decodes at once.

    Returns:
      coverage: The shard's counters.

    """
    path = coverage_dir / f"coverage-v1-{shard_key(line)}.pt"
    if path.exists():
        return Coverage(**load_tensors(path))
    coverage = merge(
        map_chunks(
            directory,
            line,
            count_shard,
            workers=workers,
            chunk_decisions=chunk_decisions,
            read=replay_episodes,
        ),
    )
    save_tensors(
        {f.name: getattr(coverage, f.name) for f in dataclasses.fields(coverage)},
        path,
    )
    return coverage


def report(
    parts: Sequence[Coverage],
    *,
    min_train_reach: int = 5_000,
    min_val_reach: int = 500,
    min_count: int = 100,
    max_unseen_mass: float = 0.01,
    max_new_rate: float = 0.001,
    increment_decisions: int = 100_000_000,
) -> dict[str, PlainTree]:
    """Build the coverage report of a corpus.

    Args:
      parts: Per-shard counters in corpus order; the trailing shards holding at
        least ``increment_decisions`` training decisions form the increment.
      min_train_reach: Training episodes required to reach every floor.
      min_val_reach: Validation episodes required to reach every floor.
      min_count: Occurrences below which an observed value is rare.
      max_unseen_mass: Unseen mass every visited floor must stay below.
      max_new_rate: New-signature rate every visited floor must stay below.
      increment_decisions: Training decisions in the increment.

    Returns:
      report: JSON-ready report with per-floor detail, ``checks``, and
        ``sufficient``.

    """
    total = merge(parts)
    new_rates = _new_rates(parts, increment_decisions=increment_decisions)
    floors: list[PlainTree] = []
    rare_values = 0
    mass_passes = True
    for floor in range(FLOORS):
        counts = total.signature_counts[(total.signatures[:, 0] >> 53) == floor]
        transitions = int(counts.sum())
        singletons = int((counts == 1).sum())
        unseen = singletons / transitions if transitions else None
        rare: dict[str, PlainTree] = {}
        never: dict[str, PlainTree] = {}
        for name, seen in _value_counts(total, floor) if transitions else ():
            rare[name] = _ints(((seen > 0) & (seen < min_count)).nonzero()[:, 0])
            never[name] = _ints((seen == 0).nonzero()[:, 0])
            rare_values += int(((seen > 0) & (seen < min_count)).sum())
        if unseen is not None:
            rate = new_rates[floor]
            mass_passes &= unseen < max_unseen_mass
            mass_passes &= rate is not None and rate < max_new_rate
        floors.append(
            {
                "floor": floor,
                "transitions": transitions,
                "signatures": len(counts),
                "singletons": singletons,
                "unseen_mass": unseen,
                "new_rate": new_rates[floor],
                "rare": rare,
                "never": never,
            },
        )
    unlocked = total.achievements
    rare_achievements = ((unlocked > 0) & (unlocked < min_count)).nonzero()[:, 0]
    checks = {
        "reach": bool(
            (total.reach[0] >= min_train_reach).all()
            and (total.reach[1] >= min_val_reach).all(),
        ),
        "values": rare_values == 0 and len(rare_achievements) == 0,
        "missing_mass": mass_passes,
    }
    return {
        "increment_decisions": increment_decisions,
        "reach": {"train": _ints(total.reach[0]), "val": _ints(total.reach[1])},
        "achievements": {
            "counts": _ints(unlocked),
            "rare": _ints(rare_achievements),
            "never": _ints((unlocked == 0).nonzero()[:, 0]),
        },
        "floors": floors,
        "checks": checks,
        "sufficient": all(checks.values()),
    }


def _value_counts(total: Coverage, floor: int) -> list[tuple[str, Tensor]]:
    """Return each cell field's and aux field's counts over its valid values."""
    schema = craftax_schema()
    fields = [
        (field.name, total.cells[floor, row, : field.valid])
        for row, field in enumerate(schema.cell_fields)
    ]
    fields += [
        (name, total.aux[floor, row, low - number_id(0) : high - number_id(0) + 1])
        for row, (name, (low, high)) in enumerate(
            zip(schema.scalar_names, schema.scalar_ranges, strict=True),
        )
    ]
    return fields


# ``None`` on every floor when no part precedes the increment.
def _new_rates(
    parts: Sequence[Coverage],
    *,
    increment_decisions: int,
) -> list[float | None]:
    """Return per floor the increment's share of transitions with new signatures."""
    decisions = torch.stack([p.decisions.sum() for p in parts])
    trailing = torch.cumsum(decisions.flip(0), 0).flip(0)
    split = int((trailing >= increment_decisions).sum()) - 1
    if split <= 0:
        return [None] * FLOORS
    prefix = merge(parts[:split])
    increment = merge(parts[split:])
    _, inverse = torch.cat([prefix.signatures, increment.signatures]).unique(
        dim=0,
        return_inverse=True,
    )
    known = len(prefix.signatures)
    seen = torch.isin(inverse[known:], inverse[:known])
    floor = increment.signatures[:, 0] >> 53
    weights = increment.signature_counts.double()
    new = torch.bincount(floor[~seen], weights[~seen], minlength=FLOORS)
    transitions = torch.bincount(floor, weights, minlength=FLOORS)
    return [float(rate) for rate in new / transitions.clamp(min=1)]


def _achievements(summary: Mapping[str, object]) -> list[int]:
    """Return a summary's unlocked achievements, a sorted list of indices 0–66."""
    try:
        items = from_plain(summary.get("achievements"), list[object])
        unlocked = from_plain(items, list[int])
    except (ReadError, TypeError) as error:
        raise ValueError(f"Episode summary lacks achievements: {summary}.") from error
    if (
        len(unlocked) != len(items)
        or unlocked != sorted(set(unlocked))
        or any(a < 0 or a >= NUM_ACHIEVEMENTS for a in unlocked)
    ):
        raise ValueError(
            f"Episode summary achievements {items} are not a sorted list of "
            f"distinct indices 0-{NUM_ACHIEVEMENTS - 1}.",
        )
    return unlocked


def _count_cells(counts: Tensor, cells: Tensor, floor: Tensor) -> None:
    """Add one episode's cell-field value counts into ``counts [9, 8, 64]``."""
    # Each pair of adjacent fields is read as one little-endian int16, low byte
    # first: values under 64 keep it non-negative and under 2**14. Half as many
    # bincounts run twice as fast as one uint8 bincount per field, which is
    # itself about 7x faster than one bincount of materialized int64 keys.
    pairs = cells.contiguous().view(torch.int16).permute(2, 0, 1).contiguous()
    bounds = (floor[1:] != floor[:-1]).nonzero()[:, 0] + 1
    starts = [0, *_ints(bounds)]
    ends = [*_ints(bounds), len(floor)]
    for start, end in zip(starts, ends, strict=True):
        row = counts[int(floor[start])]
        for pair in range(OBS_TILE_CHANNELS // 2):
            run = pairs[pair, start:end].flatten()
            grid = torch.bincount(run, minlength=1 << 14)[: 1 << 14].view(64, 256)
            row[2 * pair] += grid.sum(0)[:64]
            row[2 * pair + 1] += grid.sum(1)


# The rows match ``torch.unique(rows, dim=0)``, but come from three one-dimensional
# uniques, which are several times faster than a row unique.
def _unique_rows(rows: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
    """Return the sorted distinct rows of ``[N, 2]`` int64 and their summed weights."""
    high, high_id = rows[:, 0].unique(return_inverse=True)
    low, low_id = rows[:, 1].unique(return_inverse=True)
    pair, inverse = (high_id * len(low) + low_id).unique(return_inverse=True)
    sums = torch.zeros(len(pair), dtype=torch.int64).index_add_(0, inverse, weights)
    return torch.stack([high[pair // len(low)], low[pair % len(low)]], dim=1), sums


def _ints(tensor: Tensor) -> list[int]:
    """Return an integer tensor's values as a list of ints."""
    return from_plain(tensor.tolist(), list[int])
