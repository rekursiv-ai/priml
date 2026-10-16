"""A tier's episode sets, its quiet-stretch timelines, and its wins' time maps.

Every tier offers two sets, and a third when it has wins, each ordered so that
its first ``n`` episodes are an unbiased sample of it:

- ``all``: the main capture's episodes, by capture ordinal.
- ``short``: every episode, of any capture of the tier, that ends (dies or
  wins) within ``cap`` decisions; then the wins longer than that, shortest
  first, until wins make up at least their natural share, the share of wins
  among the episodes that ended in the uncapped captures (or every win, if
  there are fewer). The set is ordered by ordinal, then by capture.
- ``wins``: every win of every capture of the tier, by ordinal, then capture,
  but for a win pinned by its sampling seed, which comes first and plays
  unbroken. Every other win carries a time map
  (``keep_runs``): the decisions playback shows, its progress with a little
  context and a few decisions of each idle stretch, at most ``TimeRule.steps``
  of them, so wins of any length play side by side aligned by what they
  achieve rather than by the clock.

A tier with a pinned win may also offer ``unbroken`` (``unbroken_set``): the
pinned win and a few others short enough to play every decision, spread over
their lengths, for a figure that shows whole games.

A timeline counts, per decision, the shown episodes that are active at it (a
first visit to a tile, a map change, an achievement, a floor change, or the
episode's end; ``extract.py``) and keeps every decision but the middles of long
quiet runs (``layout.QuietRule``), so playback can skip what nobody does. A
decision at which a shown episode dies or wins is never quiet, so skipping
never jumps over a death or a win.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import dataclasses

import numpy as np

from priml.baselines.craftax.ghosts.extract import (
    ACHIEVED,
    ENDED,
    FLOOR_CHANGED,
    FOUGHT,
    HELD,
    MAP_CHANGE,
    NEW_TILE,
    TIMELINE,
    Ghost,
)
from priml.baselines.craftax.ghosts.layout import Composition, TimeRule
from priml.baselines.craftax.lib.arrays import ints


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from priml.baselines.craftax.ghosts.layout import QuietRule


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Pool:
    """The episodes one capture played of a tier.

    Attributes:
      root: The capture root.
      capped: Whether the capture cut episodes at a decision cap, which makes
        its ended episodes the short ones rather than a sample of the tier.
      provenance: The capture's provenance.
      ghosts: Its episodes, by ordinal.

    """

    root: str
    capped: bool
    provenance: dict[str, str]
    ghosts: tuple[Ghost, ...]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Chosen:
    """A set's episodes, each with the index of its pool, and what it holds."""

    episodes: tuple[tuple[int, Ghost], ...]
    composition: Composition


def all_set(pools: Sequence[Pool], *, count: int) -> Chosen:
    """Return the ``all`` set: the main capture's first ``count`` episodes.

    Args:
      pools: The tier's captures; the first is the main one, uncapped.
      count: Episodes the set holds at most.

    Returns:
      chosen: The set.

    """
    episodes = tuple((0, ghost) for ghost in pools[0].ghosts[:count])
    return Chosen(
        episodes=episodes,
        composition=_composition(
            episodes,
            run=len(pools[0].ghosts),
            qualified=len(pools[0].ghosts),
            added_wins=0,
            share=_natural_win_share(pools),
        ),
    )


def short_set(pools: Sequence[Pool], *, cap: int, count: int) -> Chosen:
    """Return the ``short`` set: episodes ending within ``cap``, and enough wins.

    Args:
      pools: The tier's captures.
      cap: Decisions within which an episode must end to qualify.
      count: Episodes the set holds at most: the first by ordinal.

    Returns:
      chosen: The set.

    """
    candidates = [(i, ghost) for i, pool in enumerate(pools) for ghost in pool.ghosts]
    ended = [
        (i, ghost)
        for i, ghost in candidates
        if ghost.outcome in {"death", "win"} and ghost.decisions <= cap
    ]
    deaths = sum(ghost.outcome == "death" for _, ghost in ended)
    wins = len(ended) - deaths
    long_wins = sorted(
        (
            (i, ghost)
            for i, ghost in candidates
            if ghost.outcome == "win" and ghost.decisions > cap
        ),
        key=lambda entry: (entry[1].decisions, entry[1].ordinal, entry[0]),
    )
    added = long_wins[: max(0, _wins_needed(pools, deaths=deaths) - wins)]
    qualified = sorted(ended + added, key=lambda entry: (entry[1].ordinal, entry[0]))
    episodes = tuple(qualified[:count])
    return Chosen(
        episodes=episodes,
        composition=_composition(
            episodes,
            run=len(candidates),
            qualified=len(qualified),
            added_wins=len(added),
            share=_natural_win_share(pools),
        ),
    )


def wins_set(pools: Sequence[Pool], *, unbroken: int | None = None) -> Chosen:
    """Return the ``wins`` set: every win of every capture, by ordinal, then capture.

    The win sampled with seed ``unbroken``, when given, moves to the front, so
    every count of the set shows it: it is the win playback follows, and it
    plays unbroken, every decision shown.

    Args:
      pools: The tier's captures.
      unbroken: The sampling seed of the win that plays unbroken, if any.

    Returns:
      chosen: The set, empty when the tier never wins.

    Raises:
      ValueError: No win, or more than one, has the seed ``unbroken``.

    """
    candidates = [(i, ghost) for i, pool in enumerate(pools) for ghost in pool.ghosts]
    wins = sorted(
        ((i, ghost) for i, ghost in candidates if ghost.outcome == "win"),
        key=lambda entry: (entry[1].ordinal, entry[0]),
    )
    if unbroken is not None:
        pinned = [k for k, (_, g) in enumerate(wins) if g.sampling_seed == unbroken]
        if len(pinned) != 1:
            raise ValueError(
                f"Expected one win sampled with seed {unbroken} to play unbroken; "
                f"found {len(pinned)}.",
            )
        wins.insert(0, wins.pop(pinned[0]))
    return Chosen(
        episodes=tuple(wins),
        composition=_composition(
            wins,
            run=len(candidates),
            qualified=len(wins),
            added_wins=0,
            share=_natural_win_share(pools),
        ),
    )


def unbroken_set(
    pools: Sequence[Pool],
    *,
    pinned: int,
    count: int,
    steps: int,
) -> Chosen:
    """Return the ``unbroken`` set: the pinned win and wins spread over their lengths.

    Of every win of every capture but the pinned one, those of at most
    ``steps`` decisions, sorted by decisions, then ordinal, then capture, give
    ``count - 1`` at evenly spaced ranks ``round(k (n - 1) / (count - 2))``
    for ``k = 0 .. count - 2``: the shortest, the longest and evenly between,
    so the wins end all along a timeline of ``steps``. With fewer than
    ``count - 1`` that short, it takes them all and the shortest of the
    longer wins, which a time map then compresses. The pinned win comes
    first, the others by ordinal, then capture.

    Args:
      pools: The tier's captures.
      pinned: The sampling seed of the pinned win.
      count: Episodes the set holds, at least 3; fewer when the tier has
        fewer wins.
      steps: Decisions a win of the set may play unbroken.

    Returns:
      chosen: The set.

    Raises:
      ValueError: No win, or more than one, has the seed ``pinned``, or
        ``count`` is under 3.

    """
    if count < 3:
        raise ValueError(f"An unbroken set needs at least 3 episodes, not {count}.")
    first, *wins = wins_set(pools, unbroken=pinned).episodes
    ranked = sorted(
        wins,
        key=lambda entry: (entry[1].decisions, entry[1].ordinal, entry[0]),
    )
    fits = [entry for entry in ranked if entry[1].decisions <= steps]
    if len(fits) >= count - 1:
        last = len(fits) - 1
        picked = [fits[round(k * last / (count - 2))] for k in range(count - 1)]
    else:
        picked = ranked[: count - 1]
    others = sorted(picked, key=lambda entry: (entry[1].ordinal, entry[0]))
    episodes = (first, *others)
    return Chosen(
        episodes=episodes,
        composition=_composition(
            episodes,
            run=sum(len(pool.ghosts) for pool in pools),
            qualified=len(fits) + (first[1].decisions <= steps),
            added_wins=max(0, count - 1 - len(fits)),
            share=_natural_win_share(pools),
        ),
    )


PROGRESS: Final = (
    NEW_TILE | MAP_CHANGE | ACHIEVED | FLOOR_CHANGED | ENDED | HELD | FOUGHT
)
"""The activity flags that make a decision progress, which a time map keeps."""

MUST_KEEP: Final = FLOOR_CHANGED | ENDED
"""The flags a time map keeps whatever its budget: floor changes and the end."""


def keep_runs(
    active: bytes,
    *,
    rule: TimeRule,
    forced: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.int64], int]:
    """Return the decisions a win's playback shows, as monotone kept runs.

    Progress decisions (``PROGRESS``) are kept with ``context`` decisions on
    each side; every other stretch keeps ``idle`` decisions spread evenly
    across it, its first among them. Decision 0, floor changes and the last
    decision are always kept. The first level of ``rule.levels`` (context,
    idle) that keeps at most ``rule.steps`` decisions is used; if none does,
    progress itself is thinned evenly to fit, the always-kept decisions first.

    Args:
      active: Per decision, the episode's activity flags (``Ghost.active``).
      rule: The budget and the levels to try.
      forced: Decisions kept as the always-kept ones are, if any: a win's
        sleeps, in the view that shows them.

    Returns:
      runs: int64 ``[n, 2]``, the kept ``[start, stop)`` decision ranges in
        order: decision ``runs`` covers in order is display step 0, 1, ....
      level: The index of the level used in ``rule.levels``, or
        ``len(rule.levels)`` when progress was thinned.

    """
    flags = np.frombuffer(active, np.uint8)
    must = np.not_equal(flags & MUST_KEEP, 0)
    must[0] = must[-1] = True
    if forced is not None:
        must[forced] = True
    progress = np.not_equal(flags & PROGRESS, 0)
    for level, (context, idle) in enumerate(rule.levels):
        kept = _dilate(progress, context) | must
        kept |= _spread(~kept, idle)
        if np.count_nonzero(kept) <= rule.steps:
            return _runs(kept), level
    kept = must.copy()
    spare = rule.steps - np.count_nonzero(must)
    candidates = np.flatnonzero(progress & ~must)
    if spare > 0 and len(candidates):
        kept[
            candidates[
                np.linspace(0, len(candidates) - 1, min(spare, len(candidates)))
                .round()
                .astype(np.int64)
            ]
        ] = True
    return _runs(kept), len(rule.levels)


def activity(
    ghosts: Sequence[Ghost],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Return per decision the episodes active at it, still live, and dying or winning.

    Args:
      ghosts: The shown episodes.

    Returns:
      active: int64 ``[D]``, ``D`` the longest episode's decisions.
      live: int64 ``[D]``, the episodes whose decisions reach past each.
      decisive: int64 ``[D]``, the episodes whose death or win is that decision.

    """
    longest = max(ghost.decisions for ghost in ghosts)
    active = np.zeros(longest, np.int64)
    live = np.zeros(longest + 1, np.int64)
    decisive = np.zeros(longest, np.int64)
    for ghost in ghosts:
        active[: ghost.decisions] += (
            np.frombuffer(ghost.active, np.uint8) & TIMELINE
        ) != 0
        live[ghost.decisions] -= 1
        decisive[ghost.decisions - 1] += ghost.outcome in {"death", "win"}
    live[0] += len(ghosts)
    return active, np.cumsum(live)[:longest], decisive


def quiet_segments(
    active: NDArray[np.int64],
    *,
    live: NDArray[np.int64],
    decisive: NDArray[np.int64],
    rule: QuietRule,
) -> NDArray[np.int64]:
    """Return the decisions playback keeps when it skips quiet stretches.

    Args:
      active: int ``[D]``, the shown episodes active at each decision.
      live: int ``[D]``, the shown episodes not yet ended at each decision.
      decisive: int ``[D]``, the shown episodes dying or winning at each
        decision, which is then never quiet.
      rule: When a decision is quiet and how a quiet run collapses.

    Returns:
      segments: int64 ``[n, 2]``, the kept ``[start, stop)`` ranges in order:
        everything but the middle of each run of at least ``rule.min_run``
        quiet decisions, of which its first and last ``rule.keep`` stay.

    """
    quiet = (active < np.maximum(1, -(-live // rule.per_live))) & np.equal(decisive, 0)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]])))
    starts, stops = edges[::2], edges[1::2]
    long = stops - starts >= rule.min_run
    gaps = np.stack([starts[long] + rule.keep, stops[long] - rule.keep], axis=1)
    bounds = np.concatenate([[0], gaps.ravel(), [len(active)]])
    return bounds.reshape(-1, 2).astype(np.int64)


def _dilate(mask: NDArray[np.bool], width: int) -> NDArray[np.bool]:
    """Return ``mask`` with every set element's ``width`` neighbours each side set too."""
    if not width:
        return mask.copy()
    hits = np.convolve(
        mask.astype(np.int64),
        np.ones(2 * width + 1, np.int64),
        mode="same",
    )
    return hits > 0


def _spread(gaps: NDArray[np.bool], idle: int) -> NDArray[np.bool]:
    """Return, of each run of set elements in ``gaps``, ``idle`` spread evenly over it."""
    kept = np.zeros_like(gaps)
    if not idle:
        return kept
    edges = np.flatnonzero(np.diff(np.concatenate([[0], gaps.astype(np.int8), [0]])))
    for start, stop in zip(ints(edges[::2]), ints(edges[1::2]), strict=True):
        take = min(idle, stop - start)
        kept[start + (np.arange(take) * (stop - start)) // take] = True
    return kept


def _runs(kept: NDArray[np.bool]) -> NDArray[np.int64]:
    """Return the ``[start, stop)`` runs of set elements of ``kept``, in order."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], kept.astype(np.int8), [0]])))
    return edges.reshape(-1, 2).astype(np.int64)


def _natural_win_share(pools: Sequence[Pool]) -> float:
    """Return the share of wins among the episodes the uncapped captures saw end."""
    wins, ended = _uncapped_ends(pools)
    return wins / ended if ended else 0.0


def _wins_needed(pools: Sequence[Pool], *, deaths: int) -> int:
    """Return the wins beside ``deaths`` that make up the natural share, at least."""
    wins, ended = _uncapped_ends(pools)
    if wins == ended:
        return sum(ghost.outcome == "win" for pool in pools for ghost in pool.ghosts)
    return -(-wins * deaths // (ended - wins))


def _uncapped_ends(pools: Sequence[Pool]) -> tuple[int, int]:
    """Return the wins, and the deaths and wins, of the uncapped captures."""
    outcomes = [
        ghost.outcome for pool in pools if not pool.capped for ghost in pool.ghosts
    ]
    wins = outcomes.count("win")
    return wins, wins + outcomes.count("death")


def _composition(
    episodes: Sequence[tuple[int, Ghost]],
    *,
    run: int,
    qualified: int,
    added_wins: int,
    share: float,
) -> Composition:
    """Return what a set holds."""
    by_outcome = {
        outcome: [ghost.decisions for _, ghost in episodes if ghost.outcome == outcome]
        for outcome in ("death", "win", "timeout", "truncated")
    }
    return Composition(
        run=run,
        qualified=qualified,
        deaths=len(by_outcome["death"]),
        wins=len(by_outcome["win"]),
        timeouts=len(by_outcome["timeout"]),
        truncated=len(by_outcome["truncated"]),
        added_wins=added_wins,
        natural_win_share=share,
        death_decisions=_span(by_outcome["death"]),
        win_decisions=_span(by_outcome["win"]),
    )


def _span(decisions: Sequence[int]) -> tuple[int, ...]:
    """Return ``(shortest, longest)``, or nothing for no episodes."""
    return (min(decisions), max(decisions)) if decisions else ()
