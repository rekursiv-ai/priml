#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Freeze a corpus of whole published shards in the behaviour-mixture shares.

Arm A gets SHARES[A] / sum(SHARES) of --decisions training decisions, and a
nineteenth of that in validation decisions: capture makes every twentieth
episode a validation episode, on validation worlds. Each arm draws whole shards
of its split uniformly at random, without replacement, until it has its
decisions, so the last shard may overshoot. A worker fills its shards in the
order episodes end, so its first shards hold its shortest episodes; the draw
gives every published episode of an arm the same chance of inclusion, where
taking shards in manifest order would favour early deaths. --seed fixes the
draw: the same published shards, arguments, and seed freeze the same corpus.
An arm with a positive share and too few published decisions in either split
fails the freeze; pass --shares with 0 for that arm to freeze without it.
--all instead of --decisions freezes every published shard of both splits of
every arm with a positive share: the archive's own mix, with no draw. That is
not the capture budgets' mix: a worker finishes the episodes in flight when
its budget is met (capture/worker.py), so an arm of long episodes overshoots
most; the end-to-end check's archive holds 59/2/22/17 for budgets of
70/10/10/10, and --decisions draws 72/9/9/9 out of it.
--combine instead joins frozen corpora whole, such as two dataset versions, in
their own mixes: a worker may then appear under several roots, but only as
captures of different seed generations (distinct rollout seeds), and no shard
twice.

The corpus is ROOTS[0]/corpora/NAME.json; each shard in it keeps its own
directory, so one corpus may span archive roots. Freezing refuses an existing
corpus, a root holding HALT.json, and a worker published under two roots,
whose world seeds would repeat. Name the roots as training will read them.

Examples:
  priml/baselines/craftax/world_model/scripts/freeze_corpus.py /opt/scratch/datasets/craftax/world-model/archive-v1 --corpus base --decisions 50_000_000
  priml/baselines/craftax/world_model/scripts/freeze_corpus.py /opt/scratch/datasets/craftax/world-model/archive-v1 --corpus base --decisions 50_000_000 --shares 80,10,10,0
  priml/baselines/craftax/world_model/scripts/freeze_corpus.py /opt/scratch/datasets/craftax/world-model/archive-v1 --corpus every-shard --all
  priml/baselines/craftax/world_model/scripts/freeze_corpus.py /opt/scratch/datasets/craftax/world-model/archive-v2 --corpus v1v2 --combine /opt/scratch/datasets/craftax/world-model/archive-v1/corpora/every-shard.json /opt/scratch/datasets/craftax/world-model/archive-v2/corpora/v2-fresh.json

'''
# fmt: on

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import argparse
import hashlib

from priml.baselines.craftax.world_model.archive import (
    ManifestLine,
    read_corpus,
    read_manifest,
    write_corpus,
)
from priml.baselines.craftax.world_model.capture.control import (
    check_halt,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.capture.worker import (
    shard_directory,
)


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the corpus is written.

    Raises:
      FileExistsError: The corpus is already frozen.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    roots = [root.absolute() for root in flags.roots]
    path = roots[0] / "corpora" / f"{flags.corpus}.json"
    if path.exists():
        raise FileExistsError(f"Corpus {path} is already frozen.")
    if flags.combine is not None:
        for root in roots:
            check_halt(root)
        selection = combine_corpora(flags.combine)
    else:
        selection = select_shards(
            roots,
            decisions=flags.decisions,
            shares=flags.shares,
            seed=flags.seed,
        )
    write_corpus(path, entries=[e for group in selection.values() for e in group])
    for split in (TRAIN, VALIDATION):
        counts = [
            sum(line.decisions for _, line in selection[split, arm]) for arm in range(4)
        ]
        for arm, count in enumerate(counts):
            # A corpus of branches alone has no validation split.
            share = f" ({count / sum(counts):.1%})" if sum(counts) else ""
            print(
                f"{_split_name(split)} arm {arm}: {len(selection[split, arm])} shards, "
                f"{count:,} decisions{share}.",
            )
    print(f"Corpus: {path}, drawn with seed {flags.seed}.")
    return 0


def combine_corpora(
    paths: Sequence[Path],
) -> dict[tuple[int, int], list[tuple[Path, ManifestLine]]]:
    """Return every shard of frozen corpora, by split and arm, refusing repeats.

    One worker may appear under several roots only as captures of different
    seed generations, which its manifest lines' rollout seeds tell apart
    (``capture/seeds.py``); two captures of one worker and generation would
    repeat world seeds.

    Args:
      paths: Corpus files, e.g. v1's and v2's fresh and branch corpora.

    Returns:
      selection: Each ``(split, arm)``'s shards and their directories, training
        arms 0-3 then validation arms 0-3, each in corpus order.

    Raises:
      ValueError: A shard is named twice, or two directories hold captures of
        one worker and generation.

    """
    selection: dict[tuple[int, int], list[tuple[Path, ManifestLine]]] = {
        (split, arm): [] for split in (TRAIN, VALIDATION) for arm in range(4)
    }
    owners: dict[tuple[str, ...], Path] = {}
    named: set[tuple[Path, str]] = set()
    for directory, line in (entry for path in paths for entry in read_corpus(path)):
        if (directory, line.shard) in named:
            raise ValueError(f"{directory / line.shard} is named twice.")
        named.add((directory, line.shard))
        split, arm, worker = directory.parts[-3:]
        capture = (split, arm, worker, line.provenance.get("rollout_seed", ""))
        owner = owners.setdefault(capture, directory)
        if owner != directory:
            raise ValueError(
                f"{owner} and {directory} are two captures of one worker; their "
                "world seeds would repeat.",
            )
        side = VALIDATION if split == _split_name(VALIDATION) else TRAIN
        selection[side, int(arm.removeprefix("arm"))].append((directory, line))
    return selection


def select_shards(
    roots: Sequence[Path],
    *,
    decisions: int | None,
    shares: Sequence[int],
    seed: int,
) -> dict[tuple[int, int], list[tuple[Path, ManifestLine]]]:
    """Draw each arm's whole shards of each split at random in the mixture shares.

    Args:
      roots: Archive roots holding ``{train,val}/arm{arm}/w{worker}/``.
      decisions: Training decisions of the corpus, before the last shards'
        overshoot; None takes every published shard of each split instead.
      shares: Relative share of each of the four arms; 0 leaves an arm out.
      seed: Seed of the draw.

    Returns:
      selection: Each ``(split, arm)``'s shards and their directories, training
        arms 0-3 then validation arms 0-3.

    Raises:
      ValueError: ``shares`` does not weight four arms, ``decisions`` is not
        positive, a worker is published under two roots, or an arm with a
        positive share has too few published decisions in a split.
      CaptureHaltedError: A root holds ``HALT.json``.

    """
    if (
        len(shares) != 4
        or min(shares) < 0
        or sum(shares) <= 0
        or (decisions is not None and decisions <= 0)
    ):
        raise ValueError(
            f"Expected four nonnegative shares with a positive sum and positive "
            f"decisions, not shares {tuple(shares)} and decisions {decisions}.",
        )
    for root in roots:
        check_halt(root)
    selection: dict[tuple[int, int], list[tuple[Path, ManifestLine]]] = {}
    for split in (TRAIN, VALIDATION):
        for arm, share in enumerate(shares):
            published = _published(roots, split=split, arm=arm, seed=seed)
            if decisions is None:
                selection[split, arm] = published if share else []
                continue
            # Rounded up, so every arm with a positive share gets a shard.
            target = -(-decisions * share // sum(shares))
            # Every twentieth episode a worker starts is a validation episode.
            target = -(-target // 19) if split == VALIDATION else target
            taken = _take(published, target=target)
            have = sum(line.decisions for _, line in taken)
            if have < target:
                raise ValueError(
                    f"Arm {arm} has {have:,} published {_split_name(split)} "
                    f"decisions; the corpus needs {target:,}.",
                )
            selection[split, arm] = taken
    return selection


class Flags(Protocol):
    """Parsed command-line flags."""

    roots: list[Path]
    corpus: str
    decisions: int | None
    all: bool
    combine: list[Path] | None
    shares: tuple[int, ...]
    seed: int


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("roots", type=Path, nargs="+", help="Archive roots.")
    parser.add_argument("--corpus", required=True, help="Corpus name, e.g. base.")
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--decisions", type=int, help="Training decisions.")
    size.add_argument(
        "--all",
        action="store_true",
        help="Every published shard of both splits, in the archive's own mix.",
    )
    size.add_argument(
        "--combine",
        type=Path,
        nargs="+",
        metavar="CORPUS",
        help="Every shard of these frozen corpora, e.g. two dataset versions.",
    )
    parser.add_argument(
        "--shares",
        type=_shares,
        default=(70, 10, 10, 10),
        help="Relative shares of arms 0-3, comma-separated; default 70,10,10,10.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the shard draw; default 0.",
    )


def _shares(text: str) -> tuple[int, ...]:
    """Parse comma-separated integer shares."""
    return tuple(int(share) for share in text.split(","))


def _split_name(split: int) -> str:
    """Return a split's directory name."""
    return "val" if split == VALIDATION else "train"


def _published(
    roots: Sequence[Path],
    *,
    split: int,
    arm: int,
    seed: int,
) -> list[tuple[Path, ManifestLine]]:
    """Return an arm's published shards of one split in a seeded random order."""
    ranked: dict[bytes, tuple[Path, ManifestLine]] = {}
    for worker in range(4):
        manifests = {
            directory: read_manifest(directory)
            for directory in (
                shard_directory(root, split=split, arm=arm, worker=worker)
                for root in roots
            )
        }
        published = [directory for directory, lines in manifests.items() if lines]
        if len(published) > 1:
            raise ValueError(
                f"{' and '.join(map(str, published))} are two captures of one "
                "worker; their world seeds would repeat.",
            )
        for directory in published:
            for line in manifests[directory]:
                # Manifest order is episode-end order, so a prefix of it would
                # favour early deaths.
                name = f"{seed}/{_split_name(split)}/arm{arm}/w{worker}/{line.shard}"
                ranked[hashlib.sha256(name.encode()).digest()] = (directory, line)
    return [ranked[key] for key in sorted(ranked)]


def _take(
    shards: Sequence[tuple[Path, ManifestLine]],
    *,
    target: int,
) -> list[tuple[Path, ManifestLine]]:
    """Return the fewest leading shards holding ``target`` decisions, or all."""
    taken: list[tuple[Path, ManifestLine]] = []
    total = 0
    for directory, line in shards:
        if total >= target:
            break
        taken.append((directory, line))
        total += line.decisions
    return taken


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
