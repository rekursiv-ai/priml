#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Derive a corpus of whole or capped episodes drawn at random from archives.

The training episodes of every SOURCE, a frozen corpus or an archive root
whose published workers are all read, are pooled per arm. Arm A gets
SHARES[A] / sum(SHARES) of --decisions training decisions: its episodes,
each cut to its first --max-decisions decisions when that is positive, are
taken in a seeded random order, each ranked by itself, until the arm has its
share. So every episode of an arm has the same chance of inclusion, where a
corpus of whole shards gives an arm that needs one shard the episodes of
whichever shard it draws, and capture fills a shard in the order episodes end.
The chosen episodes, a replay shard's replayed, are written, in that order, as
new frame shards of about --shard-decisions decisions under OUT/train/arm{A}/w0/,
each published with a manifest line. OUT/corpora/NAME.json lists them with the
validation shards of --validation (default the SOURCEs), unchanged, so corpora
derived by different rules are scored on the same validation decisions. The
same sources and --seed rank episodes the same way, so a smaller corpus is a
prefix of a larger one arm by arm: nested corpora for a data-scaling test.

A cut episode keeps the actions, frames, rewards, and state hashes of its
prefix; it has no terminal decision, so the loader never scores one for it,
and its summary records the original length as "capped_from". Its record is
truncated (archive.py): its last hash is of the state after its last
decision, taken by replaying the prefix unless the cap is a multiple of 256,
where the episode already holds it, so the cut episode replays alone.
Each source episode appears at most once; a branch shares its parent's world
seed, so a source holding branches repeats worlds.

Examples:
  priml/baselines/craftax/world_model/scripts/data_derive.py /opt/scratch/datasets/craftax/world-model/archive-v1/corpora/data-200m.json /opt/scratch/datasets/craftax/world-model/derived-v1/base-cap16k --name base-cap16k --decisions 50_000_000 --max-decisions 16_384 --validation /opt/scratch/datasets/craftax/world-model/archive-v1/corpora/base.json

'''
# fmt: on

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import argparse
import dataclasses
import hashlib

import numpy as np
import torch

from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    Episode,
    EpisodeSummary,
    ManifestLine,
    read_manifest,
    read_summaries,
    write_corpus,
    write_shard,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.capture.worker import (
    shard_directory,
)
from priml.baselines.craftax.world_model.index import FLOOR_AUX, FLOORS
from priml.baselines.craftax.world_model.scripts.data_stats import (
    discover,
)
from priml.baselines.craftax.world_model.snapshots import replay_episodes
from priml.lib.codec import from_plain


type _Pick = tuple[tuple[Path, ManifestLine], EpisodeSummary]
"""A source episode: its shard's corpus entry and its summary line."""


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the corpus is written.

    Raises:
      FileExistsError: The corpus is already written.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    path = flags.out / "corpora" / f"{flags.name}.json"
    if path.exists():
        raise FileExistsError(f"Corpus {path} is already written.")
    entries = derive(
        flags.sources,
        # The corpus names the derived shards by it: absolute, they read from
        # any working directory.
        flags.out.absolute(),
        decisions=flags.decisions,
        shares=flags.shares,
        max_decisions=flags.max_decisions,
        seed=flags.seed,
        shard_decisions=flags.shard_decisions,
        validation=flags.validation,
    )
    write_corpus(path, entries=entries)
    for directory, _ in entries:
        print(directory)
    print(f"Corpus: {path}, {len(entries)} shards, drawn with seed {flags.seed}.")
    return 0


def derive(
    sources: Sequence[Path],
    out: Path,
    *,
    decisions: int,
    shares: Sequence[int],
    max_decisions: int,
    seed: int,
    shard_decisions: int,
    validation: Path | None = None,
) -> list[tuple[Path, ManifestLine]]:
    """Write each arm's drawn episodes as shards; return the corpus entries.

    Args:
      sources: Frozen corpora or archive roots whose training episodes are drawn.
      out: Root of the derived shards, ``out/train/arm{A}/w0/``.
      decisions: Training decisions of the derived corpus, before the last
        episode's overshoot per arm.
      shares: Relative share of each of the four arms; 0 leaves an arm out.
      max_decisions: Decisions kept of each episode; 0 keeps whole episodes.
      seed: Seed of the draw.
      shard_decisions: A derived shard closes at the first episode boundary at
        or after this many decisions.
      validation: Corpus whose validation shards the derived corpus keeps;
        ``None`` keeps those of ``sources``.

    Returns:
      entries: The derived training shards, then the validation shards.

    Raises:
      ValueError: ``shares`` does not weight four arms, an arm is short of its
        share, or ``out`` already holds a derived arm.

    """
    if len(shares) != 4 or min(shares) < 0 or sum(shares) <= 0:
        raise ValueError(f"Expected four nonnegative shares, not {tuple(shares)}.")
    pools = _pools(_entries(sources))
    entries: list[tuple[Path, ManifestLine]] = []
    for arm, share in enumerate(shares):
        if not share:
            continue
        target = -(-decisions * share // sum(shares))
        pool = pools.get(arm, [])
        chosen = _draw(pool, target=target, cap=max_decisions, seed=seed)
        directory = shard_directory(out, split=TRAIN, arm=arm, worker=0)
        if read_manifest(directory):
            raise ValueError(f"{directory} already holds derived shards.")
        directory.mkdir(parents=True, exist_ok=True)
        origin = {
            "derived_from": ",".join(str(source) for source in sources),
            "max_decisions": str(max_decisions),
        }
        entries += _write(
            directory,
            chosen,
            cap=max_decisions,
            shard=shard_decisions,
            origin=origin,
        )
    entries += [
        entry
        for entry in _entries([validation] if validation else sources)
        if read_summaries(*entry)[0].receipt.split == VALIDATION
    ]
    return entries


class Flags(Protocol):
    """Parsed command-line flags."""

    sources: list[Path]
    out: Path
    name: str
    decisions: int
    shares: tuple[int, ...]
    max_decisions: int
    seed: int
    shard_decisions: int
    validation: Path | None


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="Frozen corpora or archive roots to draw from.",
    )
    parser.add_argument("out", type=Path, help="Root of the derived shards.")
    parser.add_argument("--name", required=True, help="Corpus name.")
    parser.add_argument(
        "--decisions",
        type=int,
        required=True,
        help="Training decisions.",
    )
    parser.add_argument(
        "--shares",
        type=lambda text: tuple(int(share) for share in text.split(",")),
        default=(70, 10, 10, 10),
        help="Relative shares of arms 0-3, comma-separated; default 70,10,10,10.",
    )
    parser.add_argument(
        "--max-decisions",
        type=int,
        default=0,
        help="Decisions kept of each episode; default 0, whole episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the draw; default 0.",
    )
    parser.add_argument(
        "--shard-decisions",
        type=int,
        default=2_000_000,
        help="Decisions per derived shard; default 2,000,000.",
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=None,
        help="Corpus whose validation shards are kept; default the SOURCEs'.",
    )


def _entries(sources: Sequence[Path]) -> list[tuple[Path, ManifestLine]]:
    """Return the published shards of ``sources``, each once however many list it."""
    unique = {
        (directory.resolve(), line.shard): (directory, line)
        for group in discover(sources).values()
        for directory, line in group
    }
    return list(unique.values())


def _pools(entries: Sequence[tuple[Path, ManifestLine]]) -> dict[int, list[_Pick]]:
    """Return every training episode of ``entries`` by arm, in corpus order."""
    pools: dict[int, list[_Pick]] = {}
    for entry in entries:
        for summary in read_summaries(*entry):
            if summary.receipt.split == TRAIN:
                pools.setdefault(summary.receipt.arm, []).append((entry, summary))
    return pools


def _draw(pool: Sequence[_Pick], *, target: int, cap: int, seed: int) -> list[_Pick]:
    """Return episodes of ``pool`` in a seeded random order until ``target``."""
    ranked = sorted(pool, key=lambda pick: _rank(pick[1], seed=seed))
    chosen: list[_Pick] = []
    total = 0
    for pick in ranked:
        if total >= target:
            break
        chosen.append(pick)
        total += min(pick[1].decisions, cap or pick[1].decisions)
    if total < target:
        raise ValueError(f"The source holds {total:,} decisions; {target:,} needed.")
    return chosen


def _rank(summary: EpisodeSummary, *, seed: int) -> bytes:
    """Return the hash an episode is ranked by in the draw of ``seed``."""
    receipt = summary.receipt
    # A fresh episode's world seed is its own, so its key hashes that alone, as
    # every corpus derived so far was drawn: they stay reproducible and nest with
    # later draws. A branch shares its parent's world seed; keyed by it alone, a
    # family of branches would tie and be drawn as one block.
    key = f"{seed}/{receipt.world_seed}"
    if "branch" in summary.summary:
        key += f"/{receipt.sampling_seed}"
    return hashlib.sha256(key.encode()).digest()


def _write(
    directory: Path,
    chosen: Sequence[_Pick],
    *,
    cap: int,
    shard: int,
    origin: dict[str, str],
) -> list[tuple[Path, ManifestLine]]:
    """Publish ``chosen`` in order as shards of about ``shard`` decisions."""
    entries: list[tuple[Path, ManifestLine]] = []
    episodes: list[Episode] = []
    provenance: dict[str, str] = {}
    for position, ((source, line), summary) in enumerate(chosen):
        (episode,) = replay_episodes(source, line, summaries=[summary])
        episodes.append(_cut(episode, cap=cap))
        provenance = line.provenance | origin
        size = sum(len(e.actions) for e in episodes)
        if size >= shard or position == len(chosen) - 1:
            index = len(entries)
            written = write_shard(
                directory,
                index=index,
                episodes=episodes,
                provenance=provenance,
            )
            entries.append((directory, written))
            episodes = []
    return entries


def _cut(episode: Episode, *, cap: int) -> Episode:
    """Return ``episode``'s first ``cap`` decisions, or all of it."""
    decisions = len(episode.actions)
    if not cap or decisions <= cap:
        return episode
    return dataclasses.replace(
        episode,
        actions=episode.actions[:cap],
        # Hashes before decisions 0, 256, ... of the prefix, then the state after
        # its last decision, as a truncated record ends.
        hashes=torch.cat(
            [episode.hashes[: (cap + 255) // 256], _hash_at(episode, decision=cap)],
        ),
        cells=episode.cells[:cap],
        aux=episode.aux[:cap],
        reward=episode.reward[:cap],
        done=episode.done[:cap],
        summary={
            **episode.summary,
            "capped_from": decisions,
            "floors": _kept_floors(episode, cap=cap),
        },
        truncated=True,
    )


def _hash_at(episode: Episode, *, decision: int) -> torch.Tensor:
    """Return the state hash before ``decision`` of ``episode``, int64 ``[1]``."""
    if decision % replay.HASH_STRIDE == 0:
        index = decision // replay.HASH_STRIDE
        return episode.hashes[index : index + 1]
    # ``origin`` replays to the decision, checking every hash on the way, and
    # XORs the state with its world's reset, a branch's too; XORed back, it is
    # the state.
    reset = replay.save(*replay.reset_world(episode.receipt.world_seed))
    origin = replay.origin(episode, decision=decision)
    states, _ = replay.load(replay.xor_bytes(origin, right=reset))
    digest = np.array([replay.fnv1a_numba(states.view(np.uint8))], dtype=np.uint64)
    return torch.from_numpy(digest.view(np.int64))


# Kills, clears, and descents keep the whole episode's counts: the frames do not say
# when they happened.
def _kept_floors(episode: Episode, *, cap: int) -> list[dict[str, object]]:
    """Return the summary's floor records with reach and decisions of the prefix."""
    kept = episode.aux[:cap, FLOOR_AUX].long().bincount(minlength=FLOORS)
    return [
        {**record, "reached": int(count > 0), "decisions": int(count)}
        for record, count in zip(
            from_plain(
                episode.summary.get("floors"),
                list[dict[str, object]],
                default=[],
            ),
            kept,
            strict=True,
        )
    ]


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
