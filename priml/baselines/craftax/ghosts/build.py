#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Build the ghost site's data directory from captured episodes of one shared world.

Each tier is one capture arm. Its episodes are read from both splits of worker
0 of each capture: ROOT, the main capture, and any --extra (more uncapped
episodes) and --capped (episodes cut at a decision cap) captures. Every episode
is replayed on the port's game from its record, checked against every state
hash it carries (``replay.verify``) and against the last one again after the
extraction replay, and becomes a ghost: one player byte per decision, its map
changes, its achievements, its creatures every 4th decision, its activity, and
how it ended. A ghost ends at its first DEFEAT_NECROMANCER, as an evaluation that
stops at the boss would. An episode that a capture's summary says ended
otherwise fails the build.

Every tier has two sets (``sets.py``): ``all``, the main capture's episodes by
ordinal, and ``short``, the episodes that end within --short-decisions plus
enough longer wins for wins' natural share; a tier that wins has a third,
``wins``, every win of its captures, each with a time map that compresses it to
at most --win-steps decisions. Each set is written in nested groups (the first
100, 250, 500 and 1,000 episodes, then the rest of ``wins``), with a
quiet-stretch timeline per count, or a time map file per group for ``wins``. The site holds the shared world once and a manifest of
provenance, statistics and every file's SHA-256. FORMAT.md beside this file
specifies the files. Replay needs the capture's libm, so build on the
capture's node: world generation and daylight call it.

Examples:
  priml/baselines/craftax/ghosts/build.py /opt/scratch/artifacts/craftax/ghosts/capture/w15 /opt/scratch/artifacts/craftax/ghosts/site-data --tier early=0 --tier medium=1 --tier high=2
  priml/baselines/craftax/ghosts/build.py ROOT OUT --tier boss=3 --extra EXTRA --capped SHORT
  priml/baselines/craftax/ghosts/build.py PILOT OUT --tier high=2 --world 15 --counts 16,32,64

'''
# fmt: on

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import dataclasses
import gzip
import hashlib
import itertools
import json
import subprocess

from priml.baselines.craftax.game import jit
from priml.baselines.craftax.game.jit import package_digest, platform_key
from priml.baselines.craftax.game.state import (
    ACHIEVEMENT_REWARD_MAP,
    NUM_LEVELS,
    env_state,
)
from priml.baselines.craftax.ghosts.extract import (
    SLEEP_STRIDE,
    Ghost,
    extract,
)
from priml.baselines.craftax.ghosts.layout import (
    FORMAT,
    EpisodeEntry,
    EpisodeSet,
    Events,
    Group,
    GroupFile,
    Manifest,
    QuietRule,
    Sleeps,
    Source,
    Stats,
    Tier,
    Timeline,
    TimeMap,
    TimeRule,
    World,
    encode_events,
    encode_keeps,
    encode_sleep,
    encode_timeline,
    encode_window,
    world_bytes,
)
from priml.baselines.craftax.ghosts.sets import (
    Chosen,
    Pool,
    activity,
    all_set,
    keep_runs,
    quiet_segments,
    short_set,
    unbroken_set,
    wins_set,
)
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    EpisodeSummary,
    Record,
    read_manifest,
    read_records,
    read_summaries,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.capture.worker import (
    shard_directory,
)
from priml.lib.codec import from_plain
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    import numpy as np
else:
    from wrapt import lazy_import

    np = lazy_import("numpy")


_CWD: Final = Path(__file__).resolve().parent

_RULES: Final = {
    "all": "The main capture's episodes, by capture ordinal.",
    "short": (
        "Every episode that dies or wins within short_decisions, then longer "
        "wins, shortest first, until wins reach their share among the uncapped "
        "captures' ended episodes; by ordinal, then capture."
    ),
    "wins": (
        "Every win of the tier's captures, by ordinal, then capture, but for the "
        "win pinned to play unbroken, first (time_map.unbroken); each other win "
        "shows its progress and a few decisions of each idle stretch (time_map)."
    ),
    "unbroken": (
        "The pinned win, then wins of at most time_map.rule.steps decisions at "
        "evenly spaced ranks of length, by ordinal, then capture; each plays "
        "every decision (time_map.whole)."
    ),
}


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class TierGhosts:
    """One tier's captures, the main one first.

    Attributes:
      name: The tier's name, its directory in the site.
      arm: The capture arm that played it.
      pools: Each capture's ghosts; the first is the main, uncapped capture.
      unbroken: The sampling seed of the win the ``wins`` set follows and plays
        unbroken, every decision shown; it must fit ``TimeRule.steps``.

    """

    name: str
    arm: int
    pools: tuple[Pool, ...]
    unbroken: int | None = None


def main() -> int:
    """Build a site data directory and list its files.

    Returns:
      status: 0 once the site is written.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    counts = tuple(int(count) for count in flags.counts.split(","))
    read = partial(read_pool, workers=flags.workers, worlds=flags.world)
    unbroken = {
        name: int(seed) for name, seed in (spec.split("=") for spec in flags.unbroken)
    }
    if stray := set(unbroken) - {spec.split("=")[0] for spec in flags.tier}:
        parser.error(f"--unbroken names tiers not built: {sorted(stray)}.")
    whole_wins: tuple[int, int] | None = None
    if flags.unbroken_set:
        count, steps = flags.unbroken_set.split(":")
        whole_wins = (int(count), int(steps))
    tiers: list[TierGhosts] = []
    for spec in flags.tier:
        name, arm = spec.split("=")
        pools = [read(flags.root, arm=int(arm), capped=False)]
        pools += [read(root, arm=int(arm), capped=False) for root in flags.extra]
        pools += [read(root, arm=int(arm), capped=True) for root in flags.capped]
        tiers.append(
            TierGhosts(
                name=name,
                arm=int(arm),
                pools=tuple(pools),
                unbroken=unbroken.get(name),
            ),
        )
    manifest = write_site(
        flags.out,
        tiers=tiers,
        counts=counts,
        short_decisions=flags.short_decisions,
        time_rule=TimeRule(steps=flags.win_steps),
        whole_wins=whole_wins,
    )
    for path, size in manifest.sizes.items():
        print(f"{size:>12,} {path}")  # noqa: T201 -- CLI output: what the build wrote.
    return 0


def read_pool(
    root: Path,
    *,
    arm: int,
    capped: bool,
    workers: int,
    worlds: Sequence[int] = (),
    stride: int = 4,
) -> Pool:
    """Replay every episode one capture played of an arm into ghosts, by ordinal.

    Args:
      root: The capture root, holding ``{train,val}/arm{arm}/w0/``.
      arm: The capture arm.
      capped: Whether the capture cut its episodes at a decision cap.
      workers: Episodes replayed at once, on threads.
      worlds: Only episodes of these world seeds, as of a capture over
        several worlds; empty takes every episode.
      stride: Decisions between creature samples.

    Returns:
      pool: Its ghosts, each checked against its capture summary.

    Raises:
      ValueError: The arm holds no episode, shards of several provenances, or
        an episode that does not replay or disagrees with its summary.

    """
    entries: list[tuple[EpisodeSummary, Record]] = []
    provenances: list[dict[str, str]] = []
    for split in (TRAIN, VALIDATION):
        directory = shard_directory(root, split=split, arm=arm, worker=0)
        for line in read_manifest(directory):
            summaries = read_summaries(directory, line)
            records = read_records(directory, line, summaries=summaries)
            entries += [
                (summary, record)
                for summary, record in zip(summaries, records, strict=True)
                if not worlds or record.receipt.world_seed in worlds
            ]
            provenances.append(line.provenance)
    if not entries or any(p != provenances[0] for p in provenances):
        raise ValueError(
            f"Arm {arm} under {root} holds {len(entries)} episodes of "
            f"{len({json.dumps(p, sort_keys=True) for p in provenances})} "
            "provenances; a capture is one provenance's episodes.",
        )
    entries.sort(key=lambda entry: _ordinal(entry[0]))
    with ThreadPoolExecutor(workers) as pool:
        ghosts = tuple(pool.map(partial(_checked_ghost, stride=stride), entries))
    return Pool(root=str(root), capped=capped, provenance=provenances[0], ghosts=ghosts)


def write_site(
    out: Path,
    *,
    tiers: Sequence[TierGhosts],
    counts: Sequence[int] = (100, 250, 500, 1000),
    short_decisions: int = 10_000,
    stride: int = 4,
    window: int = 8192,
    quiet: QuietRule = QuietRule(),  # noqa: B008 -- A frozen dataclass, never mutated.
    time_rule: TimeRule = TimeRule(),  # noqa: B008 -- A frozen dataclass, never mutated.
    whole_wins: tuple[int, int] | None = None,
) -> Manifest:
    """Write the site data directory: the world, each tier's sets, the manifest.

    Args:
      out: A new directory.
      tiers: The tiers; each main capture holds at least ``counts[-1]`` ghosts,
        and every ghost starts in one world.
      counts: Episodes in a set's first group, its first two, and so on; a
        set with fewer episodes ends at its own size.
      short_decisions: Decisions within which a ``short`` episode ends.
      stride: Decisions between the ghosts' creature samples.
      window: Decisions per creature window file, a multiple of ``stride``.
      quiet: When a decision is quiet, for the timelines.
      time_rule: How each win of a ``wins`` set is compressed.
      whole_wins: ``(count, steps)``: give each tier with a pinned win an
        ``unbroken`` set of ``count`` wins that play every decision, as long
        as they fit ``steps`` (``sets.unbroken_set``).

    Returns:
      manifest: As written to ``manifest.json``.

    Raises:
      ValueError: The counts, window, rule or tiers do not fit together.
      FileExistsError: ``out`` exists.

    """
    seeds = {
        ghost.world_seed
        for tier in tiers
        for pool in tier.pools
        for ghost in pool.ghosts
    }
    if (
        len(seeds) != 1
        or list(counts) != sorted(set(counts))
        or counts[0] <= 0
        or window % stride
        or quiet.min_run <= 2 * quiet.keep
        or any(len(tier.pools[0].ghosts) < counts[-1] for tier in tiers)
        or any(tier.pools[0].capped for tier in tiers)
    ):
        raise ValueError(
            f"Expected one world (found {len(seeds)}), increasing counts, a "
            f"window that is a multiple of {stride}, a quiet run longer than "
            f"its kept ends, and {counts[-1]} ghosts in each uncapped main capture.",
        )
    (world_seed,) = seeds
    out = validated_output_path(out)
    out.mkdir(parents=True)
    states, _ = replay.reset_world(world_seed)
    state = env_state(states, 0)
    files = {
        "world.bin.gz": _gzip(
            world_bytes(
                World(
                    block=np.array(state.map, dtype=np.uint8),
                    item=np.array(state.item_map, dtype=np.uint8),
                    light=np.array(state.light_map, dtype=np.uint8),
                    down_ladders=np.array(state.down_ladders, dtype=np.int64),
                    up_ladders=np.array(state.up_ladders, dtype=np.int64),
                ),
            ),
        ),
    }
    layout = _Layout(counts=tuple(counts), stride=stride, window=window, quiet=quiet)
    entries = [
        Tier(
            name=tier.name,
            arm=tier.arm,
            sources=tuple(
                Source(
                    root=pool.root,
                    capped=pool.capped,
                    episodes=len(pool.ghosts),
                    provenance=pool.provenance,
                )
                for pool in tier.pools
            ),
            sets=_write_sets(
                files,
                tier=tier,
                layout=layout,
                short_decisions=short_decisions,
                time_rule=time_rule,
                whole_wins=whole_wins,
            ),
        )
        for tier in tiers
    ]
    for path, data in files.items():
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        (out / path).write_bytes(data)
    manifest = Manifest(
        format=FORMAT,
        git_commit=_git_commit(),
        game_package_digest=package_digest(Path(jit.__file__).parent),
        platform=platform_key(),
        world_seed=world_seed,
        start=(
            int(state.player_level),
            int(state.player_position[0]),
            int(state.player_position[1]),
            int(state.player_direction),
        ),
        creature_stride=stride,
        sleep_stride=SLEEP_STRIDE,
        window_decisions=window,
        counts=tuple(counts),
        short_decisions=short_decisions,
        quiet=quiet,
        achievement_rewards=tuple(
            int(ACHIEVEMENT_REWARD_MAP.item(i))
            for i in range(len(ACHIEVEMENT_REWARD_MAP))
        ),
        tiers=tuple(entries),
        files={path: hashlib.sha256(data).hexdigest() for path, data in files.items()},
        sizes={path: len(data) for path, data in files.items()},
    )
    (out / "manifest.json").write_text(
        json.dumps(dataclasses.asdict(manifest), indent=1) + "\n",
    )
    return manifest


class Flags(Protocol):
    """Parsed command-line flags."""

    root: Path
    out: Path
    tier: list[str]
    extra: list[Path]
    capped: list[Path]
    counts: str
    short_decisions: int
    win_steps: int
    unbroken: list[str]
    unbroken_set: str
    workers: int
    world: list[int]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _Layout:
    """The site-wide settings a set's files are written by."""

    counts: tuple[int, ...]
    stride: int
    window: int
    quiet: QuietRule


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "root",
        type=Path,
        help="Main capture root of the shared world.",
    )
    parser.add_argument("out", type=Path, help="New site data directory.")
    parser.add_argument(
        "--tier",
        action="append",
        required=True,
        help="NAME=ARM: a tier and the capture arm that played it; repeatable.",
    )
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help="Another uncapped capture root, for the short set; repeatable.",
    )
    parser.add_argument(
        "--capped",
        type=Path,
        action="append",
        default=[],
        help="A capture root cut at a decision cap, for the short set; repeatable.",
    )
    parser.add_argument(
        "--counts",
        default="100,250,500,1000",
        help="Episodes in each nested group, increasing.",
    )
    parser.add_argument(
        "--short-decisions",
        type=int,
        default=10_000,
        help="Decisions within which a short-set episode ends.",
    )
    parser.add_argument(
        "--win-steps",
        type=int,
        default=TimeRule().steps,
        help="Decisions a win's time map keeps at most, in the wins set.",
    )
    parser.add_argument(
        "--unbroken",
        action="append",
        default=[],
        help="TIER=SEED: the win of the tier's wins set, by sampling seed, that "
        "playback follows and shows unbroken; repeatable.",
    )
    parser.add_argument(
        "--unbroken-set",
        default="",
        help="COUNT:STEPS: give each tier with an --unbroken win an unbroken set of "
        "COUNT wins that play every decision, up to STEPS of them.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Replay threads.")
    parser.add_argument(
        "--world",
        type=int,
        action="append",
        default=[],
        help="Take only this world's episodes, of a capture over several; repeatable.",
    )


def _ordinal(summary: EpisodeSummary) -> int:
    """Return an episode's capture ordinal."""
    return from_plain(summary.summary["episode"], int)


def _checked_ghost(entry: tuple[EpisodeSummary, Record], *, stride: int) -> Ghost:
    """Extract one captured episode and check its end against its capture summary."""
    summary, record = entry
    ghost = extract(record, ordinal=_ordinal(summary), stride=stride)
    line = summary.summary
    if ghost.outcome != "win" and (
        from_plain(line["death"], int) != (ghost.outcome == "death")
        or from_plain(line["timeout"], int) != (ghost.outcome == "timeout")
        or from_plain(line["return"], float) != ghost.achievement_return
    ):
        raise ValueError(
            f"Episode {ghost.ordinal} ends {ghost.outcome} with return "
            f"{ghost.achievement_return}; its summary says {dict(line)}.",
        )
    return ghost


def _write_sets(
    files: dict[str, bytes],
    *,
    tier: TierGhosts,
    layout: _Layout,
    short_decisions: int,
    time_rule: TimeRule,
    whole_wins: tuple[int, int] | None,
) -> tuple[EpisodeSet, ...]:
    """Add a tier's sets' files to ``files``; return their manifest entries."""
    count = layout.counts[-1]
    sets = [
        _write_set(
            files,
            tier=tier.name,
            name="all",
            chosen=all_set(tier.pools, count=count),
            layout=layout,
        ),
        _write_set(
            files,
            tier=tier.name,
            name="short",
            chosen=short_set(tier.pools, cap=short_decisions, count=count),
            layout=layout,
        ),
    ]
    wins = wins_set(tier.pools, unbroken=tier.unbroken)
    if wins.episodes:
        sets.append(
            _write_set(
                files,
                tier=tier.name,
                name="wins",
                chosen=wins,
                layout=layout,
                time_rule=time_rule,
                unbroken=tier.unbroken is not None,
            ),
        )
    if whole_wins and tier.unbroken is not None:
        count, steps = whole_wins
        sets.append(
            _write_set(
                files,
                tier=tier.name,
                name="unbroken",
                chosen=unbroken_set(
                    tier.pools,
                    pinned=tier.unbroken,
                    count=count,
                    steps=steps,
                ),
                layout=layout,
                time_rule=dataclasses.replace(time_rule, steps=steps),
                unbroken=True,
                whole=True,
            ),
        )
    return tuple(sets)


# A set with ``time_rule`` gets time maps instead of timelines; with ``unbroken``, its
# first episode plays unbroken.
def _write_set(
    files: dict[str, bytes],
    *,
    tier: str,
    name: str,
    chosen: Chosen,
    layout: _Layout,
    time_rule: TimeRule | None = None,
    unbroken: bool = False,
    whole: bool = False,
) -> EpisodeSet:
    """Add one set's group files, and its timelines or time maps, to ``files``."""
    ghosts = [ghost for _, ghost in chosen.episodes]
    counts = tuple(count for count in layout.counts if count < len(ghosts))
    counts += (len(ghosts),) if ghosts else ()
    keeps, time_map = (
        _time_maps(ghosts, rule=time_rule, unbroken=unbroken, whole=whole)
        if time_rule
        else ([], None)
    )
    sleep_keeps, sleep_map = (
        _time_maps(ghosts, rule=time_rule, unbroken=unbroken, whole=whole, sleep=True)
        if time_rule
        else ([], None)
    )
    groups = [
        _write_group(
            files,
            path=f"{tier}/{name}/g{index}",
            group=GroupFile(
                tier=tier,
                set=name,
                group=index,
                first=first,
                episodes=_entries(chosen.episodes[first:stop], first=first),
            ),
            ghosts=ghosts[first:stop],
            layout=layout,
            keeps=keeps[first:stop],
            sleep_keeps=sleep_keeps[first:stop],
        )
        for index, (first, stop) in enumerate(itertools.pairwise((0, *counts)))
    ]
    return EpisodeSet(
        name=name,
        rule=_RULES[name],
        counts=counts,
        episodes=len(ghosts),
        decisions=sum(ghost.decisions for ghost in ghosts),
        max_decisions=max((ghost.decisions for ghost in ghosts), default=0),
        composition=chosen.composition,
        groups=tuple(groups),
        stats=tuple(_stats(ghosts[:count]) for count in counts),
        timelines=()
        if time_rule
        else tuple(
            _write_timeline(
                files,
                path=f"{tier}/{name}/timeline-n{count}.bin.gz",
                ghosts=ghosts[:count],
                quiet=layout.quiet,
            )
            for count in counts
        ),
        time_map=time_map,
        sleep_map=sleep_map,
    )


# With ``unbroken``, the first win plays unbroken, one run of all its decisions; with
# ``whole``, so does every win that fits ``rule.steps``; every other win is compressed
# (``sets.keep_runs``). With ``sleep``, the maps of the view that shows sleep: every
# sleep decision is kept, and each display step count adds the sleeps' samples, a
# step each.
def _time_maps(
    ghosts: Sequence[Ghost],
    *,
    rule: TimeRule,
    unbroken: bool,
    whole: bool = False,
    sleep: bool = False,
) -> tuple[list[NDArray[np.int64]], TimeMap]:
    """Return each win's kept runs and the set's time maps in numbers."""
    if unbroken and ghosts[0].decisions > rule.steps:
        raise ValueError(
            f"The unbroken win has {ghosts[0].decisions} decisions, more than the "
            f"{rule.steps} a time map keeps; it cannot play unbroken.",
        )
    keeps: list[NDArray[np.int64]] = []
    compressed: list[int] = []
    for k, ghost in enumerate(ghosts):
        if (unbroken and k == 0) or (whole and ghost.decisions <= rule.steps):
            keeps.append(np.array([[0, ghost.decisions]], np.int64))
            continue
        forced = ghost.sleeps[:, 0] if sleep else None
        runs, level = keep_runs(ghost.active, rule=rule, forced=forced)
        keeps.append(runs)
        compressed.append(level)
    samples = [len(ghost.sleep_samples) - 1 if sleep else 0 for ghost in ghosts]
    steps = np.array(
        [
            int(np.sum(runs[:, 1] - runs[:, 0])) + n
            for runs, n in zip(keeps, samples, strict=True)
        ],
        dtype=np.int64,
    )
    levels = np.bincount(compressed, minlength=len(rule.levels) + 1)
    return keeps, TimeMap(
        rule=rule,
        steps=int(np.max(steps)),
        kept=int(np.sum(steps)),
        shortest=int(np.min(steps)),
        median=int(np.median(steps)),
        levels=tuple(int(n) for n in levels),
        unbroken=0 if unbroken else None,
        whole=len(ghosts) - len(compressed),
        samples=sum(samples),
    )


# A time-mapped group also gets the maps of the view that shows sleep (``sleep_keeps``)
# and its episodes' sleeps.
def _write_group(
    files: dict[str, bytes],
    *,
    path: str,
    group: GroupFile,
    ghosts: Sequence[Ghost],
    layout: _Layout,
    keeps: Sequence[NDArray[np.int64]],
    sleep_keeps: Sequence[NDArray[np.int64]] = (),
) -> Group:
    """Add one group's files, with its time maps when ``keeps`` has them, to ``files``."""
    windows = -(-max(ghost.decisions for ghost in ghosts) // layout.window)
    files[f"{path}/episodes.json"] = (
        json.dumps(dataclasses.asdict(group), separators=(",", ":")) + "\n"
    ).encode()
    files[f"{path}/players.bin.gz"] = _gzip(b"".join(g.players for g in ghosts))
    files[f"{path}/events.bin.gz"] = _gzip(
        encode_events(
            Events(
                map=np.concatenate([g.events.map for g in ghosts]),
                achievements=np.concatenate([g.events.achievements for g in ghosts]),
                escapes=np.concatenate([g.events.escapes for g in ghosts]),
            ),
        ),
    )
    if keeps:
        files[f"{path}/keeps.bin.gz"] = _gzip(encode_keeps(keeps))
        files[f"{path}/keeps-sleep.bin.gz"] = _gzip(encode_keeps(sleep_keeps))
        files[f"{path}/sleep.bin.gz"] = _gzip(
            encode_sleep(
                [
                    Sleeps(
                        sleeps=g.sleeps,
                        samples=g.sleep_samples,
                        creatures=g.sleep_creatures,
                        changes=g.sleep_changes,
                    )
                    for g in ghosts
                ],
            ),
        )
    per_window = layout.window // layout.stride
    for j in range(windows):
        files[f"{path}/creatures-w{j}.bin.gz"] = _gzip(
            encode_window(
                [
                    _window_run(g, first=j * per_window, stop=(j + 1) * per_window)
                    for g in ghosts
                ],
            ),
        )
    return Group(
        path=path,
        first=group.first,
        count=len(ghosts),
        decisions=sum(ghost.decisions for ghost in ghosts),
        windows=windows,
    )


def _write_timeline(
    files: dict[str, bytes],
    *,
    path: str,
    ghosts: Sequence[Ghost],
    quiet: QuietRule,
) -> Timeline:
    """Add the timeline of a set's first ghosts to ``files``; return its manifest entry."""
    active, live, decisive = activity(ghosts)
    segments = quiet_segments(active, live=live, decisive=decisive, rule=quiet)
    files[path] = _gzip(encode_timeline(active, segments=segments))
    return Timeline(
        count=len(ghosts),
        path=path,
        decisions=len(active),
        kept=int((segments[:, 1] - segments[:, 0]).sum()),
        segments=len(segments),
    )


def _entries(
    episodes: Sequence[tuple[int, Ghost]],
    *,
    first: int,
) -> tuple[EpisodeEntry, ...]:
    """Return a group's ``episodes.json`` lines, with each ghost's offsets in its files."""
    entries: list[EpisodeEntry] = []
    players = 0
    starts = np.zeros(3, np.int64)
    for i, (source, ghost) in enumerate(episodes):
        sizes = (
            len(ghost.events.map),
            len(ghost.events.achievements),
            len(ghost.events.escapes),
        )
        entries.append(
            EpisodeEntry(
                index=first + i,
                source=source,
                ordinal=ghost.ordinal,
                sampling_seed=str(ghost.sampling_seed),
                decisions=ghost.decisions,
                outcome=ghost.outcome,
                end=ghost.end,
                achievement_return=ghost.achievement_return,
                floor_first=ghost.floor_first,
                players=players,
                map=(starts.item(0), sizes[0]),
                achievements=(starts.item(1), sizes[1]),
                escapes=(starts.item(2), sizes[2]),
            ),
        )
        players += ghost.decisions
        starts += sizes
    return tuple(entries)


def _window_run(ghost: Ghost, *, first: int, stop: int) -> bytes:
    """Return ``ghost``'s creature samples ``first`` to ``stop``, as far as it has them."""
    last = len(ghost.samples) - 1
    return ghost.creatures[
        ghost.samples[min(first, last)] : ghost.samples[min(stop, last)]
    ]


def _stats(ghosts: Sequence[Ghost]) -> Stats:
    """Return the statistics of a set's first ghosts."""
    deaths = np.zeros(NUM_LEVELS, np.int64)
    for ghost in ghosts:
        deaths[ghost.end[0]] += ghost.outcome == "death"
    return Stats(
        count=len(ghosts),
        mean_return=float(np.mean([ghost.achievement_return for ghost in ghosts])),
        mean_decisions=float(np.mean([ghost.decisions for ghost in ghosts])),
        reached=tuple(
            sum(ghost.floor_first[floor] >= 0 for ghost in ghosts)
            for floor in range(NUM_LEVELS)
        ),
        deaths=tuple(int(d) for d in deaths),
        timeouts=sum(ghost.outcome == "timeout" for ghost in ghosts),
        wins=sum(ghost.outcome == "win" for ghost in ghosts),
        escapes=sum(len(ghost.events.escapes) for ghost in ghosts),
    )


def _gzip(data: bytes) -> bytes:
    """Return ``data`` gzipped at level 9 with no timestamp, so a rebuild is byte-identical."""
    packed = gzip.compress(data, compresslevel=9, mtime=0)
    # Byte 9 of the header names the OS: Python 3.12 keeps zlib's (3 on Linux),
    # 3.13+ writes 255, "unknown". Pinned to 255, a site built on 3.12 has the
    # digests of the same site built on 3.14, and the committed fixture rebuilds.
    return packed[:9] + b"\xff" + packed[10:]


def _git_commit() -> str:
    """Return the commit of the checkout this module runs from."""
    return subprocess.run(  # noqa: S603 -- A fixed argv: no shell and no outside text.
        ["git", "-C", str(_CWD), "rev-parse", "HEAD"],  # noqa: S607 -- git from PATH, as every checkout tool here runs it.
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
