#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Rank a capture archive's games and save one as an exact game for the viewer.

ARCHIVE is a capture root (capture/worker.py writes it), or any directory below
one: every worker directory under it with a MANIFEST.jsonl is read. Its
complete episodes are the ones neither truncated nor branched from another's
state; their mean achievement return is the policy's average, which a saved
game shows beside its own score.

rank prints the average and the best complete episode: the highest achievement
return, then the fewest decisions. An episode is named by its worker directory
below ARCHIVE, its shard and its index there, as
val/arm0/w0/shard-000000/17.

save replays one episode (rank's best unless --episode names one) on the
port's game from its world's reset, checking every state hash its record
holds, and writes OUTPUT, a new directory, as an exact game bundle
(viewer/exact.py); --limit keeps its first N decisions. Replay must run on
the libm the capture ran on (glibc on the cluster): world generation and the
daylight curve go through it. viewer/games.mjs build turns bundles into one
page.

panel replays one episode as save does and writes OUTPUT as a policy-view
bundle (viewer/policy_view.py): each frame cut to the window and HUD the policy
saw, for the panel whose script viewer/games.mjs panel writes; and, unless
--sleep-stride is 0, a frame every --sleep-stride ticks (4) of each sleep,
played a tick at a time and checked against its collapsed step.

Examples:
  priml/baselines/craftax/world_model/scripts/games.py rank /opt/scratch/artifacts/craftax/games/exp007-s74/capture
  priml/baselines/craftax/world_model/scripts/games.py save /opt/scratch/artifacts/craftax/games/exp007-s74/capture /opt/scratch/artifacts/craftax/games/exp007-s74/best --title "exp103, seed 74"
  priml/baselines/craftax/world_model/scripts/games.py panel /opt/scratch/artifacts/craftax/policy-view/capture/w15 /opt/scratch/artifacts/craftax/policy-view/boss-short-935 --episode val/arm3/w0/shard-000000/34 --limit 3822 --sleep-stride 4

'''
# fmt: on

from pathlib import Path
from typing import Protocol, cast

import argparse
import dataclasses
import json

from priml.baselines.craftax.world_model.archive import (
    Record,
    read_manifest,
    read_records,
    read_summaries,
)
from priml.baselines.craftax.world_model.viewer import policy_view
from priml.baselines.craftax.world_model.viewer.exact import (
    replay_episode,
    write_bundle,
)
from priml.lib.codec import from_plain


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Candidate:
    """One complete episode of an archive.

    Attributes:
      episode: Its name: worker directory, shard and index.
      score: Its achievement return.
      decisions: Its decisions.

    """

    episode: str
    score: float
    decisions: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Ranking:
    """An archive's complete episodes: how many, their mean return and the best.

    Attributes:
      complete: Complete episodes.
      mean_score: Their mean achievement return.
      best: The highest return, then the fewest decisions.

    """

    complete: int
    mean_score: float
    best: Candidate


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the ranking is printed or the bundle written.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    archive = flags.archive.absolute()
    ranking = rank(archive)
    if flags.command == "rank":
        print(json.dumps(dataclasses.asdict(ranking), indent=2))
        return 0
    episode = flags.episode or ranking.best.episode
    record = read_episode(archive, episode)
    sleep_stride = (flags.sleep_stride or None) if flags.command == "panel" else None
    replayed = replay_episode(record, limit=flags.limit, sleep_stride=sleep_stride)
    if flags.command == "panel":
        panel = policy_view.write_bundle(
            replayed,
            flags.output,
            record=record,
            archive=archive,
            episode=episode,
            sleep_stride=sleep_stride,
        )
        print(
            f"{flags.output}: {panel.decisions:,} decisions, {panel.score} points, {panel.sleep_frames:,} sleep frames.",
        )
        return 0
    manifest = write_bundle(
        replayed,
        flags.output,
        title=flags.title or episode,
        description=flags.description,
        end_label=flags.end_label,
        provenance=(
            f"Episode {episode} of {archive} (world seed "
            f"{record.receipt.world_seed}), replayed on the port's game; every "
            "recorded state hash matched. The average is the archive's "
            f"{ranking.complete} complete episodes'."
        ),
        mean_score=ranking.mean_score,
    )
    print(f"{flags.output}: {manifest.actions:,} actions, {manifest.score} points.")
    return 0


def rank(archive: Path) -> Ranking:
    """Return an archive's complete episodes' mean return and its best one.

    Args:
      archive: A capture root, or a directory below one.

    Returns:
      ranking: The count, the mean achievement return and the best episode.

    Raises:
      ValueError: The archive holds no complete episode.

    """
    candidates: list[Candidate] = []
    for manifest in sorted(archive.rglob("MANIFEST.jsonl")):
        directory = manifest.parent
        worker = directory.relative_to(archive).as_posix()
        for line in read_manifest(directory):
            for index, entry in enumerate(read_summaries(directory, line)):
                facts = entry.summary
                if "truncated" in facts or "branch" in facts:
                    continue
                candidates.append(
                    Candidate(
                        episode=f"{worker}/{line.shard}/{index}",
                        score=from_plain(facts["return"], float),
                        decisions=entry.decisions,
                    ),
                )
    if not candidates:
        raise ValueError(f"No complete episode in {archive}.")
    return Ranking(
        complete=len(candidates),
        mean_score=sum(c.score for c in candidates) / len(candidates),
        best=max(candidates, key=lambda c: (c.score, -c.decisions)),
    )


def read_episode(archive: Path, episode: str) -> Record:
    """Return the record of the episode ``rank`` names ``episode``.

    Args:
      archive: The capture root ``episode`` is named below.
      episode: Worker directory, shard and index, as ``val/arm0/w0/shard-000000/17``.

    Returns:
      record: Its receipt, actions and hashes.

    Raises:
      ValueError: The worker directory publishes no such shard.

    """
    worker, shard, index = episode.rsplit("/", 2)
    directory = archive / worker
    lines = [line for line in read_manifest(directory) if line.shard == shard]
    if not lines:
        raise ValueError(f"{directory} publishes no shard {shard}.")
    summary = read_summaries(directory, lines[0])[int(index)]
    return read_records(directory, lines[0], summaries=[summary])[0]


class Flags(Protocol):
    """Parsed command-line flags."""

    command: str
    archive: Path
    output: Path
    episode: str
    limit: int | None
    sleep_stride: int
    title: str
    description: str
    end_label: str


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the three commands and their flags on ``parser``."""
    commands = parser.add_subparsers(dest="command", required=True)
    ranked = commands.add_parser("rank", help="Print the average and the best episode.")
    ranked.add_argument("archive", type=Path, help="Capture root.")
    saved = commands.add_parser("save", help="Write one episode as an exact game.")
    panel = commands.add_parser("panel", help="Write one as a policy-view bundle.")
    for replayed in (saved, panel):
        replayed.add_argument("archive", type=Path, help="Capture root.")
        replayed.add_argument("output", type=Path, help="New bundle directory.")
        replayed.add_argument("--episode", default="", help="Episode; rank's best.")
        replayed.add_argument("--limit", type=int, help="Keep the first N decisions.")
    saved.add_argument("--title", default="", help="Picker title; the episode's name.")
    saved.add_argument("--description", default="", help="Page description.")
    saved.add_argument("--end-label", default="", help="Name of the final frame.")
    panel.add_argument(
        "--sleep-stride",
        type=int,
        default=4,
        help="Ticks between each sleep's frames; 0 takes none.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
