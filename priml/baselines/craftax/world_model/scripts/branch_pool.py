#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Write a branch pool: late-floor states of one arm's archived episodes to branch from.

Every training episode of --arm in the replay shards the --corpus files name,
or the training worker manifests of --arm under a --root list (each shard
once), gives branch points on --floors (capture/branches.py select_points):
its first decision on such a floor, an entry, and then every --spacing
decisions of its time on the floor, over all its visits, times on the floor;
--window FLOOR=DECISIONS keeps a floor's points within its first that many
decisions there. The pool is those points shuffled with --seed, --copies
times over (each copy shuffled again), so a capture that stops partway has
branched from a uniform sample of them, and a point in a later copy branches
again under another sampling seed.

A capture worker with this pool starts branch n from point n (capture/
branches.py), so the pool is fixed once capture begins. Its parents' shard
directories are absolute paths on the node that holds them: write it there.
Prints the point counts by floor and kind as JSON.

Example:
  priml/baselines/craftax/world_model/scripts/branch_pool.py \
    --corpus /opt/scratch/datasets/craftax/world-model/archive-v1/corpora/base.json \
    --arm 0 --spacing 512 --copies 4 --seed 1 \
    --output /opt/scratch/datasets/craftax/world-model/archive-v2-branch/pools/arm0-w1.jsonl

'''
# fmt: on

from pathlib import Path
from typing import Protocol, cast

import argparse
import collections
import json

import numpy as np

from priml.baselines.craftax.lib.arrays import ints
from priml.baselines.craftax.world_model.archive import (
    ManifestLine,
    read_corpus,
    read_manifest,
)
from priml.baselines.craftax.world_model.capture.branches import (
    BranchPoint,
    select_points,
    write_pool,
)


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the pool is written.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    if not flags.corpus and not flags.root:
        parser.error("Name a --corpus or a --root.")
    entries: dict[tuple[Path, str], tuple[Path, ManifestLine]] = {}
    listed = [pair for corpus in flags.corpus for pair in read_corpus(corpus)]
    for root in flags.root:
        pattern = f"train/arm{flags.arm}/*/MANIFEST.jsonl"
        for manifest in sorted(root.absolute().glob(pattern)):
            listed += [
                (manifest.parent, line) for line in read_manifest(manifest.parent)
            ]
    for directory, line in listed:
        entries.setdefault((directory, line.shard), (directory, line))
    points = select_points(
        list(entries.values()),
        arm=flags.arm,
        floors=set(flags.floors),
        spacing=flags.spacing,
        windows=dict(flags.window),
    )
    generator = np.random.default_rng(flags.seed)
    pool: list[BranchPoint] = []
    for _ in range(flags.copies):
        pool += [points[i] for i in ints(generator.permutation(len(points)))]
    flags.output.parent.mkdir(parents=True, exist_ok=True)
    write_pool(flags.output, pool)
    counts = collections.Counter(f"{p.floor}/{p.kind}" for p in points)
    print(
        json.dumps(
            {
                "points": len(points),
                "copies": flags.copies,
                "by_floor_and_kind": dict(sorted(counts.items())),
            },
        ),
    )
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    corpus: list[Path]
    root: list[Path]
    arm: int
    floors: list[int]
    spacing: int
    window: list[tuple[int, int]]
    copies: int
    seed: int
    output: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--corpus",
        type=Path,
        action="append",
        default=[],
        help="Corpus of parent replay shards; repeatable.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=[],
        help="Archive root whose worker manifests list parent shards; repeatable.",
    )
    parser.add_argument("--arm", type=int, required=True, help="Arm of the parents.")
    parser.add_argument(
        "--floors",
        type=int,
        nargs="+",
        default=[5, 6, 7],
        help="Floors to branch on: 5 Troll Mines, 6 Fire, 7 Ice, 8 Graveyard.",
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=1_024,
        help="Decisions between time points.",
    )
    parser.add_argument(
        "--window",
        type=_window,
        action="append",
        default=[],
        metavar="FLOOR=DECISIONS",
        help="Take a floor's time points within this many decisions of its entry.",
    )
    parser.add_argument(
        "--copies",
        type=int,
        default=1,
        help="Shuffled copies of the points.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    parser.add_argument("--output", type=Path, required=True, help="Pool to write.")


def _window(text: str) -> tuple[int, int]:
    """Parse ``FLOOR=DECISIONS``."""
    floor, _, decisions = text.partition("=")
    return int(floor), int(decisions)


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
