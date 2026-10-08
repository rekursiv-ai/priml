#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Write the ghost site's fixture: three real episodes of one world, built and decoded.

Episode 0 walks from the spawn to the surface's down ladder by the shortest
passable path, striking any creature in its way, descends, wanders 40 uniformly
random legal decisions, walks back to the up ladder and ascends, then plays
random legal actions until it ends. Episodes 1 and 2 play random legal actions
throughout (``replay.record``). Each is recorded with its state hashes, as
capture records it, and built by ``build.py`` into a site of one group per set
and one creature window per group, a 160-decision short set and a quiet rule
strict enough (quiet unless every live episode is active) for its timelines to
collapse runs, so every kind of file, both kinds of floor change, collapsed
runs and a short set smaller than its count occur in as few files as the
format allows. A second tier, ``fixture-wins``, is the same three episodes
won at their first wooden sword (``FIXTURE_WIN``), which two of them make, so
its ``wins`` set and time maps occur too, under a 120-decision budget: the
118-decision win is pinned to play unbroken and the 122-decision one is
compressed. OUT/site is that site; OUT/expected.json holds what
``layout.py`` decodes from it, which the page's decoder must reproduce: per
episode, the floor, row, column and facing before every decision and at the
end, the interaction targets, the end, the final maps, the creature samples
and, for a time-mapped set, the decision each display step shows; per
timeline, its activity and kept segments. The world is the platform's: its
libm generates it.

Examples:
  priml/baselines/craftax/ghosts/fixture.py priml/baselines/craftax/ghosts/testdata/fixture

'''
# fmt: on

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import dataclasses
import gzip
import hashlib
import itertools
import json

import numpy as np

from priml.baselines.craftax.game.rules import PLAYER_BLOCKED
from priml.baselines.craftax.game.state import (
    ATN_DIM,
    MAP_SIZE,
    OBS_SIZE,
    Achievement,
    Action,
    env_state,
    env_stats,
    new_stats,
)
from priml.baselines.craftax.game.step import Rules, observe_numba, play_numba
from priml.baselines.craftax.ghosts.build import TierGhosts, write_site
from priml.baselines.craftax.ghosts.extract import extract
from priml.baselines.craftax.ghosts.layout import (
    STEPS,
    GroupFile,
    Manifest,
    QuietRule,
    TimeRule,
    World,
    decode_path,
    decode_samples,
    displayed,
    displayed_sleep,
    interaction_targets,
    maps_before,
    read_events,
    read_keeps,
    read_sleep,
    read_timeline,
    read_window,
    read_world,
)
from priml.baselines.craftax.ghosts.sets import Pool
from priml.baselines.craftax.lib.arrays import ints
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import Receipt, Record
from priml.baselines.craftax.world_model.capture.seeds import splitmix64
from priml.lib.codec import from_plain, loads


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    import torch

    from priml.baselines.craftax.game.state import Array1, Array2, EnvState
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


_UNREACHED: Final = MAP_SIZE * MAP_SIZE
"""A walking distance no tile reaches: the tile has no path to the target."""

FIXTURE_WIN: Final = Achievement.MAKE_WOOD_SWORD
"""What ``fixture-wins`` counts as a win: episodes 0 and 2 make one, 1 does not."""

FIXTURE_TIME_RULE: Final = TimeRule(steps=120, levels=((1, 2), (0, 1), (0, 0)))
"""The fixture's time maps: a budget the unbroken win fits and the other exceeds."""

FIXTURE_UNBROKEN: Final = 2
"""The sampling seed of the ``fixture-wins`` win that plays unbroken: episode 2's."""


def main() -> int:
    """Write the fixture.

    Returns:
      status: 0 once it is written.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    flags.out.mkdir(parents=True)
    write_fixture(flags.out, world_seed=flags.world)
    return 0


def write_fixture(out: Path, *, world_seed: int) -> None:
    """Record, build and decode the fixture into ``out``: ``site/`` and ``expected.json``.

    Args:
      out: An existing directory without ``site/``.
      world_seed: The shared world.

    """
    records = fixture_records(world_seed=world_seed)
    pools = [
        Pool(
            root="fixture.py",
            capped=False,
            provenance={"policy": "ladder trip, then uniform-legal; uniform-legal"},
            ghosts=tuple(
                extract(r, ordinal=i, win=win.value) for i, r in enumerate(records)
            ),
        )
        for win in (Achievement.DEFEAT_NECROMANCER, FIXTURE_WIN)
    ]
    write_site(
        out / "site",
        tiers=[
            TierGhosts(name=name, arm=0, pools=(pool,), unbroken=unbroken)
            for name, pool, unbroken in zip(
                ("fixture", "fixture-wins"),
                pools,
                (None, FIXTURE_UNBROKEN),
                strict=True,
            )
        ],
        counts=(3,),
        short_decisions=160,
        quiet=QuietRule(per_live=1, min_run=16, keep=4),
        time_rule=FIXTURE_TIME_RULE,
    )
    (out / "expected.json").write_text(
        json.dumps(expected(out / "site"), separators=(",", ":")) + "\n",
    )


def fixture_records(*, world_seed: int) -> list[Record]:
    """Return the fixture's three episodes: the ladder trip, then two random ones.

    Args:
      world_seed: The shared world.

    Returns:
      records: Each with its state hashes, as capture records them.

    """
    randoms = [
        replay.record(world_seed=world_seed, sampling_seed=seed, max_decisions=20_000)
        for seed in (1, 2)
    ]
    return [
        ladder_trip(world_seed=world_seed, sampling_seed=0),
        *(
            Record(
                receipt=episode.receipt,
                actions=episode.actions,
                hashes=episode.hashes,
            )
            for episode in randoms
        ),
    ]


def ladder_trip(
    *,
    world_seed: int,
    sampling_seed: int,
    wander: int = 40,
    max_decisions: int = 20_000,
) -> Record:
    """Record the scripted episode: down the surface's ladder, back up, then random play.

    Args:
      world_seed: The world.
      sampling_seed: Seed of the random legal actions (SplitMix64).
      wander: Random decisions on floor 1 before walking back.
      max_decisions: Longest episode to accept.

    Returns:
      record: The episode with its state hashes, as capture records them.

    Raises:
      ValueError: A ladder is out of reach, or the episode did not end.

    """
    states, rng = replay.reset_world(world_seed)
    state, stats, rules = env_state(states, 0), new_stats(1), Rules()
    observation = np.zeros(OBS_SIZE, np.float32)
    mask = np.zeros(ATN_DIM, np.uint8)
    observe_numba(state, observation, mask, rules)
    trip = _Trip(stream=sampling_seed, wander=wander)
    actions: list[int] = []
    hashes: list[int] = []
    for t in range(max_decisions):
        if t % replay.HASH_STRIDE == 0:
            hashes.append(int(replay.fnv1a_numba(states.view(np.uint8))))
        actions.append(trip.act(state, mask))
        if play_numba(state, rng, env_stats(stats, 0), actions[-1], rules)[1]:
            hashes.append(int(replay.fnv1a_numba(states.view(np.uint8))))
            return Record(
                receipt=Receipt(
                    world_seed=world_seed,
                    sampling_seed=sampling_seed,
                    initial_state_hash=hashes[0],
                    arm=0,
                    split=0,
                ),
                actions=torch.tensor(actions, dtype=torch.uint8),
                hashes=torch.from_numpy(np.array(hashes, np.uint64).view(np.int64)),
            )
        observe_numba(state, observation, mask, rules)
    raise ValueError(f"The ladder trip did not end within {max_decisions} decisions.")


def expected(site: Path) -> dict[str, object]:
    """Return what the reference decoder reads from a site, the page's expected values.

    Args:
      site: A site data directory.

    Returns:
      expected: The world seed; per episode of every set of every tier, its
        decoded path, interaction targets, end, final maps and creature
        samples; per timeline, its activity and kept segments. An episode
        that an earlier entry already decodes the same (one ghost in two
        sets or two tiers) is written as a reference to it and the fields
        that differ (``shared``); ``resolved`` expands it.

    """
    manifest = from_plain(loads((site / "manifest.json").read_text()), Manifest)
    world = read_world(gzip.decompress((site / "world.bin.gz").read_bytes()))
    episodes: list[dict[str, object]] = []
    timelines: list[dict[str, object]] = []
    for tier in manifest.tiers:
        for episode_set in tier.sets:
            for group in episode_set.groups:
                episodes += _group_expected(
                    site / group.path,
                    world=world,
                    manifest=manifest,
                    windows=group.windows,
                )
            for timeline in episode_set.timelines:
                active, segments = read_timeline(
                    gzip.decompress((site / timeline.path).read_bytes()),
                )
                timelines.append(
                    {
                        "tier": tier.name,
                        "set": episode_set.name,
                        "count": timeline.count,
                        "activity": active.tolist(),
                        "segments": segments.tolist(),
                    },
                )
    return {
        "world_seed": manifest.world_seed,
        "episodes": shared(episodes),
        "timelines": timelines,
    }


_PLACE: Final = ("tier", "set", "index")
"""The fields that place an episode entry, which no other entry shares."""


def shared(episodes: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return ``episodes`` with each repeat written as a reference to its first entry.

    An entry whose ``path`` an earlier full entry has is the same ghost (the
    path is every state it passes through): unless that entry has a field
    this one lacks, it becomes its place, ``same_as`` (the earlier entry's
    position) and only the fields whose values differ from that entry's or
    that it lacks.

    Args:
      episodes: Full entries, in order.

    Returns:
      episodes: The same entries, the repeats as references.

    """
    first: dict[str, int] = {}
    out: list[dict[str, object]] = []
    for position, entry in enumerate(episodes):
        key = json.dumps(entry["path"])
        base = episodes[first[key]] if key in first else None
        if base is None or set(base) - set(entry):
            first.setdefault(key, position)
            out.append(entry)
            continue
        out.append(
            {
                **{name: entry[name] for name in _PLACE},
                "same_as": first[key],
                **{
                    name: value
                    for name, value in entry.items()
                    if name not in _PLACE and base.get(name) != value
                },
            },
        )
    return out


def resolved(episodes: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return ``shared``'s entries expanded back to full entries.

    Args:
      episodes: Entries as ``expected.json`` holds them.

    Returns:
      episodes: Each a full entry, a reference filled from the entry it names.

    """
    out: list[dict[str, object]] = []
    for entry in episodes:
        if "same_as" not in entry:
            out.append(entry)
            continue
        base = out[from_plain(entry["same_as"], int)]
        out.append(
            {
                **{name: value for name, value in base.items() if name not in _PLACE},
                **{name: value for name, value in entry.items() if name != "same_as"},
            },
        )
    return out


class Flags(Protocol):
    """Parsed command-line flags."""

    out: Path
    world: int


@dataclasses.dataclass(slots=True, kw_only=True)
class _Trip:
    """The ladder trip's policy and how far along it is."""

    stream: int
    wander: int
    phase: int = 0
    wandered: int = 0

    def act(self, state: EnvState, mask: NDArray[np.uint8]) -> int:
        """Return the next action: down, wander, back up, then random play.

        Args:
          state: The world before the decision.
          mask: Its legal actions.

        Returns:
          action: The action to play.

        """
        floor = int(state.player_level)
        if self.phase == 0 and floor == 1:
            self.phase = 1
        if self.phase == 1 and self.wandered == self.wander:
            self.phase = 2
        if self.phase == 2 and floor != 1:
            self.phase = 3
        if self.phase == 0:
            return _toward(state, ladder=state.down_ladders[0], take=Action.DESCEND)
        if self.phase == 2:
            return _toward(state, ladder=state.up_ladders[1], take=Action.ASCEND)
        self.wandered += self.phase == 1
        legal = np.flatnonzero(mask)
        self.stream, draw = splitmix64(self.stream)
        return legal.item(draw % len(legal))


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("out", type=Path, help="New fixture directory.")
    parser.add_argument("--world", type=int, default=1, help="The shared world's seed.")


def _toward(state: EnvState, *, ladder: Array1[int], take: Action) -> int:
    """Return the action that takes ``ladder`` or steps on the shortest path to it."""
    floor = int(state.player_level)
    row, col = (int(v) for v in state.player_position)
    target = (int(ladder[0]), int(ladder[1]))
    if (row, col) == target:
        return int(take)
    distance = _distances(state.map[floor], target=target)
    action, (dr, dc) = min(
        STEPS.items(),
        key=lambda step: _distance_at(distance, row + step[1][0], col + step[1][1]),
    )
    if _distance_at(distance, row + dr, col + dc) == _UNREACHED:
        raise ValueError(f"Floor {floor}'s ladder is out of reach of ({row}, {col}).")
    occupied = int(state.mob_bits[floor, row + dr]) >> (col + dc) & 1
    return int(
        Action.DO if occupied and state.player_direction == action else action,
    )


def _distances(blocks: Array2[int], *, target: tuple[int, int]) -> NDArray[np.int64]:
    """Return each tile's walking distance to ``target``, ``_UNREACHED`` where none."""
    distance = np.full((MAP_SIZE, MAP_SIZE), _UNREACHED, np.int64)
    distance[target] = 0
    queue = deque([target])
    while queue:
        row, col = queue.popleft()
        for dr, dc in STEPS.values():
            r, c = row + dr, col + dc
            if (
                _distance_at(distance, r, c) == _UNREACHED
                and 0 <= r < MAP_SIZE
                and 0 <= c < MAP_SIZE
                and not PLAYER_BLOCKED[blocks[r, c]]
            ):
                distance[r, c] = distance[row, col] + 1
                queue.append((r, c))
    return distance


def _distance_at(distance: NDArray[np.int64], row: int, col: int) -> int:
    """Return ``distance`` at a tile, ``_UNREACHED`` off the map."""
    if min(row, col) < 0 or max(row, col) >= MAP_SIZE:
        return _UNREACHED
    return distance.item(row, col)


def _group_expected(
    directory: Path,
    *,
    world: World,
    manifest: Manifest,
    windows: int,
) -> list[dict[str, object]]:
    """Return the expected values of one group's episodes."""
    group = from_plain(loads((directory / "episodes.json").read_text()), GroupFile)
    players = gzip.decompress((directory / "players.bin.gz").read_bytes())
    events = read_events(gzip.decompress((directory / "events.bin.gz").read_bytes()))
    runs = [
        read_window(
            gzip.decompress((directory / f"creatures-w{j}.bin.gz").read_bytes()),
        )
        for j in range(windows)
    ]
    keeps_file = directory / "keeps.bin.gz"
    keeps = (
        read_keeps(gzip.decompress(keeps_file.read_bytes()))
        if keeps_file.exists()
        else []
    )
    sleep_keeps = (
        read_keeps(gzip.decompress((directory / "keeps-sleep.bin.gz").read_bytes()))
        if keeps
        else []
    )
    sleeps = (
        read_sleep(
            gzip.decompress((directory / "sleep.bin.gz").read_bytes()),
            stride=manifest.sleep_stride,
        )
        if keeps
        else []
    )
    episodes: list[dict[str, object]] = []
    for i, entry in enumerate(group.episodes):
        own = players[entry.players : entry.players + entry.decisions]
        escapes = _rows(events.escapes, entry.escapes)
        path = decode_path(own, escapes=escapes, world=world, start=manifest.start)
        block, item = maps_before(
            world,
            map_events=_rows(events.map, entry.map),
            decision=entry.decisions,
        )
        changed = np.argwhere(
            np.logical_or(
                np.not_equal(block, world.block),
                np.not_equal(item, world.item),
            ),
        )
        samples = [sample for window in runs for sample in decode_samples(window[i])]
        shown = (
            {
                "displayed": displayed(keeps[i]).tolist(),
                "displayed_sleep": displayed_sleep(
                    sleep_keeps[i],
                    sleeps[i].sleeps,
                    stride=manifest.sleep_stride,
                ).tolist(),
                "sleeps": sleeps[i].sleeps.tolist(),
                "sleep_creatures": [
                    decode_samples(sleeps[i].creatures[a:b])[0]
                    for a, b in itertools.pairwise(ints(sleeps[i].samples))
                ],
            }
            if keeps
            else {}
        )
        episodes.append(
            {
                "tier": group.tier,
                "set": group.set,
                "index": entry.index,
                "decisions": entry.decisions,
                "path": path.tolist(),
                "interactions": interaction_targets(own, path=path),
                "end": list(entry.end),
                "outcome": entry.outcome,
                "achievement_return": entry.achievement_return,
                "final_changes": [
                    [*map(int, tile), int(block[tuple(tile)]), int(item[tuple(tile)])]
                    for tile in changed
                ],
                "final_map_sha256": hashlib.sha256(
                    block.tobytes() + item.tobytes(),
                ).hexdigest(),
                "creatures": [
                    [k * manifest.creature_stride, sample]
                    for k, sample in enumerate(samples)
                ],
                **shown,
            },
        )
    return episodes


def _rows(table: np.ndarray, span: Sequence[int]) -> np.ndarray:
    """Return one episode's rows of a group table: ``span`` is its start and count."""
    return table[span[0] : span[0] + span[1]]


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
