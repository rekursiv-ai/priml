#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Describe what a trajectory archive's training signal holds, per split and arm.

Each SOURCE is an archive root, whose published {train,val}/arm*/w* workers
are all read, or a frozen corpus file. The summary line of every episode gives
its length, the share of decisions in long and timeout episodes, deaths, floor
reach and decisions per floor, and starting states shared by several episodes.
With --frame-shards K, the token frames of K shards drawn at random per split
and arm (a replay shard's replayed) also give idle measures, as shares of
their decisions:

  unchanged  the next frame equals this one, the light level aside;
  repeated   the frame equals one of the previous 64 of its episode;
  idle       the decision lies in a run of at least 128 repeated frames;
  noop       the action is NOOP (0);
  sleeping   the player is asleep.

The report is one JSON object with a section per split and arm.

Examples:
  priml/baselines/craftax/world_model/scripts/data_stats.py /opt/scratch/datasets/craftax/world-model/archive-v1 --output /opt/scratch/artifacts/craftax/world-model/data-lane/stats-b.json
  priml/baselines/craftax/world_model/scripts/data_stats.py /opt/scratch/datasets/craftax/world-model/archive-v1/corpora/base.json --frame-shards 4 --workers 32 --output /opt/scratch/artifacts/craftax/world-model/data-lane/base-stats.json

'''
# fmt: on

from __future__ import annotations

from concurrent import futures
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import collections
import dataclasses
import hashlib
import json

from torch import Tensor

import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    EpisodeSummary,
    ManifestLine,
    read_corpus,
    read_manifest,
    read_summaries,
)
from priml.baselines.craftax.world_model.index import FLOORS
from priml.baselines.craftax.world_model.snapshots import replay_episodes
from priml.lib.codec import from_plain
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Sequence

    from priml.lib.codec import PlainTree


@dataclasses.dataclass(slots=True, kw_only=True)
class IdleCounts:
    """Decisions of one or more episodes, and how many are idle by each measure."""

    decisions: int = 0
    unchanged: int = 0
    repeated: int = 0
    idle: int = 0
    noop: int = 0
    sleeping: int = 0

    def add(self, other: IdleCounts) -> None:
        """Add ``other``'s counts to these."""
        for field in dataclasses.fields(self):
            name = field.name
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def shares(self) -> dict[str, PlainTree]:
        """Return the decision count and each measure's share of it."""
        total = max(self.decisions, 1)
        result: dict[str, PlainTree] = {"decisions": self.decisions}
        for field in dataclasses.fields(self)[1:]:
            result[field.name] = getattr(self, field.name) / total
        return result


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the report is written.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    output = validated_output_path(flags.output)
    groups = discover(flags.sources)
    report: dict[str, dict[str, PlainTree]] = {}
    for name, entries in groups.items():
        summaries = [s for entry in entries for s in read_summaries(*entry)]
        report[name] = summary_stats(summaries) | {"shards": len(entries)}
    if flags.frame_shards:
        chosen = {
            name: _draw(entries, count=flags.frame_shards, seed=flags.seed)
            for name, entries in groups.items()
        }
        frames = frame_stats(chosen, workers=flags.workers)
        for name, stats in frames.items():
            report[name]["frames"] = stats
    sources = [str(source) for source in flags.sources]
    result = {"sources": sources, "seed": flags.seed, "groups": report}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1) + "\n")
    print(f"Wrote {output}: {', '.join(groups)}.")
    return 0


def discover(sources: Sequence[Path]) -> dict[str, list[tuple[Path, ManifestLine]]]:
    """Return the published shards of ``sources``, grouped as ``split/armA``.

    Args:
      sources: Archive roots or corpus files (``.json``).

    Returns:
      groups: Each split and arm's shards with their directories, by name; an
        archive root's directories are absolute.

    """
    groups: dict[str, list[tuple[Path, ManifestLine]]] = {}
    for source in sources:
        if source.suffix == ".json":
            entries = read_corpus(source)
        else:
            # Absolute, as a corpus names them: a shard's directory seeds its draw
            # (``_draw``) and is written into derived corpora.
            manifests = sorted(source.absolute().glob("*/arm*/w*/MANIFEST.jsonl"))
            entries = [
                (m.parent, line) for m in manifests for line in read_manifest(m.parent)
            ]
        for directory, line in entries:
            name = f"{directory.parent.parent.name}/{directory.parent.name}"
            groups.setdefault(name, []).append((directory, line))
    return dict(sorted(groups.items()))


def summary_stats(summaries: Sequence[EpisodeSummary]) -> dict[str, PlainTree]:
    """Return the episode-level statistics of one group of summary lines.

    Args:
      summaries: Each episode's ``.meta.jsonl`` line.

    Returns:
      stats: ``episodes``, ``decisions``, ``length`` quantiles, ``buckets`` by
        episode length, ``timeouts``, ``deaths``, ``return_mean``, ``floors``
        (reach rate and decisions per floor), and ``repeated_starts``, the
        episodes whose initial state hash another episode shares.

    """
    length = torch.tensor([s.decisions for s in summaries], dtype=torch.float64)
    records = [s.summary for s in summaries]
    timeout = torch.tensor([from_plain(r.get("timeout"), int) for r in records]) > 0
    death = torch.tensor([from_plain(r.get("death"), int) for r in records]) > 0
    floors = torch.tensor(
        [
            [
                [from_plain(f.get("reached"), int), from_plain(f.get("decisions"), int)]
                for f in from_plain(r.get("floors"), list[dict[str, object]])
            ]
            for r in records
        ],
        dtype=torch.float64,
    ).reshape(len(records), FLOORS, 2)
    decisions = length.sum()
    # Hashes are unsigned 64-bit, beyond int64 tensors.
    starts = collections.Counter(s.receipt.initial_state_hash for s in summaries)
    ranked = length.sort().values
    median = int(torch.searchsorted(ranked.cumsum(0), decisions / 2))
    return {
        "episodes": len(summaries),
        "decisions": int(decisions),
        "length": {
            "mean": float(length.mean()),
            **{f"p{q}": float(length.quantile(q / 100)) for q in (10, 50, 90, 99)},
            "max": int(ranked[-1]),
            "decision_p50": int(ranked[median]),
        },
        "buckets": _buckets(length),
        "timeouts": {
            "episodes": int(timeout.sum()),
            "share_episodes": float(timeout.double().mean()),
            "share_decisions": float(length[timeout].sum() / decisions),
        },
        "deaths": {
            "episodes": int(death.sum()),
            "share_episodes": float(death.double().mean()),
        },
        "return_mean": float(
            torch.tensor(
                [float(from_plain(r.get("return"), int)) for r in records],
            ).mean(),
        ),
        "floors": {
            str(k): {
                "episodes": int(floors[:, k, 0].sum()),
                "reach": float(floors[:, k, 0].mean()),
                "decisions": int(floors[:, k, 1].sum()),
                "share": float(floors[:, k, 1].sum() / decisions),
            }
            for k in range(FLOORS)
        },
        "repeated_starts": sum(n for n in starts.values() if n > 1),
    }


def idle_counts(episode: Episode) -> IdleCounts:
    """Count one episode's decisions and its idle decisions by each measure.

    Frames compare exactly on their cells and every auxiliary token but the
    light level, which follows the clock.

    Args:
      episode: A decoded episode.

    Returns:
      counts: Its decisions and how many are unchanged, repeated, idle, NOOPs,
        and asleep.

    """
    frame = frame_ids(episode)
    repeated = repeats(frame)
    return IdleCounts(
        decisions=len(frame),
        unchanged=int((frame[1:] == frame[:-1]).sum()),
        repeated=int(repeated.sum()),
        idle=_long_runs(repeated, minimum=128),
        noop=int((episode.actions == 0).sum()),
        sleeping=int((episode.aux[:, 44] == 1).sum()),
    )


def frame_ids(episode: Episode) -> Tensor:
    """Return an id per frame, equal exactly where frames are, light aside.

    Args:
      episode: A decoded episode.

    Returns:
      ids: int64 ``[T]``, equal for two frames exactly when their cells and
        auxiliary tokens but the light level are.

    """
    decisions = len(episode.actions)
    aux = torch.cat([episode.aux[:, :43], episode.aux[:, 44:]], dim=1)
    rows = torch.cat(
        [episode.cells.reshape(decisions, -1), aux.contiguous().view(torch.uint8)],
        dim=1,
    )
    return rows.unique(dim=0, return_inverse=True)[1]


def repeats(frame: Tensor, *, window: int = 64) -> Tensor:
    """Return whether each frame id occurs among the ``window`` ids before it."""
    padded = torch.cat([torch.full((window,), -1), frame])
    return (padded.unfold(0, window, 1)[: len(frame)] == frame[:, None]).any(dim=1)


def frame_stats(
    groups: dict[str, list[tuple[Path, ManifestLine]]],
    *,
    workers: int,
) -> dict[str, PlainTree]:
    """Return each group's idle shares, overall and by episode-length bucket.

    Args:
      groups: The shards to decode, by group name.
      workers: Processes decoding shards at once; 1 decodes in this process.

    Returns:
      stats: Per group, ``IdleCounts.shares`` plus ``by_bucket``.

    """
    jobs = [(name, entry) for name, entries in groups.items() for entry in entries]
    if workers == 1:
        parts = [_shard_counts(entry) for _, entry in jobs]
    else:
        with futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=torch.set_num_threads,
            initargs=(1,),
        ) as pool:
            parts = list(pool.map(_shard_counts, [entry for _, entry in jobs]))
    totals: dict[str, dict[str, IdleCounts]] = {name: {} for name in groups}
    for (name, _), part in zip(jobs, parts, strict=True):
        for label, counts in part.items():
            totals[name].setdefault(label, IdleCounts()).add(counts)
    result: dict[str, PlainTree] = {}
    for name, buckets in totals.items():
        overall = IdleCounts()
        for counts in buckets.values():
            overall.add(counts)
        by_bucket: dict[str, PlainTree] = {
            label: counts.shares() for label, counts in sorted(buckets.items())
        }
        result[name] = overall.shares() | {"by_bucket": by_bucket}
    return result


class Flags(Protocol):
    """Parsed command-line flags."""

    sources: list[Path]
    output: Path
    frame_shards: int
    seed: int
    workers: int


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="Archive roots or corpus files.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Report JSON.")
    parser.add_argument(
        "--frame-shards",
        type=int,
        default=0,
        help="Shards per split and arm whose frames are decoded; default 0, none.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the shard draw; default 0.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Decoding processes; default 8.",
    )


def _bucket(decisions: int) -> str:
    """Return the length bucket of an episode of ``decisions`` decisions."""
    edges = (
        (64_000, ">=64k"),
        (16_000, "16k-64k"),
        (4_000, "4k-16k"),
        (1_000, "1k-4k"),
    )
    return next((label for edge, label in edges if decisions >= edge), "<1k")


def _buckets(length: Tensor) -> dict[str, PlainTree]:
    """Return episodes, decisions, and decision share per length bucket."""
    episodes: collections.Counter[str] = collections.Counter()
    decisions: collections.Counter[str] = collections.Counter()
    for count in (int(n) for n in length):
        episodes[_bucket(count)] += 1
        decisions[_bucket(count)] += count
    total = float(length.sum())
    return {
        label: {
            "episodes": episodes[label],
            "decisions": decisions[label],
            "share": decisions[label] / total,
        }
        for label in sorted(episodes)
    }


def _long_runs(flags: Tensor, *, minimum: int) -> int:
    """Return how many ``flags`` lie in runs of at least ``minimum`` Trues."""
    edges = torch.diff(torch.cat([torch.tensor([0]), flags.int(), torch.tensor([0])]))
    starts, ends = torch.nonzero(edges == 1)[:, 0], torch.nonzero(edges == -1)[:, 0]
    runs = ends - starts
    return int(runs[runs >= minimum].sum())


def _draw(
    entries: Sequence[tuple[Path, ManifestLine]],
    *,
    count: int,
    seed: int,
) -> list[tuple[Path, ManifestLine]]:
    """Return ``count`` of ``entries`` in a seeded random order, as freezes draw."""
    ranked = sorted(
        entries,
        key=lambda e: hashlib.sha256(f"{seed}/{e[0]}/{e[1].shard}".encode()).digest(),
    )
    return ranked[:count]


def _shard_counts(entry: tuple[Path, ManifestLine]) -> dict[str, IdleCounts]:
    """Return one shard's idle counts per episode-length bucket."""
    counts: dict[str, IdleCounts] = {}
    for summary in read_summaries(*entry):
        (episode,) = replay_episodes(*entry, summaries=[summary])
        bucket = counts.setdefault(_bucket(len(episode.actions)), IdleCounts())
        bucket.add(idle_counts(episode))
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
