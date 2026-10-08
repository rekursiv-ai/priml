#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Report how each ghost-overlay tier's episodes end: lengths, outcomes and wins.

Each --tier is NAME:ARM, an arm under every ROOT. An episode ends at its death,
at the timeout, or on the decision the necromancer falls, which capture does
not record: capture plays on past it, as replay does. That decision is found
by replaying the episode's record from its first decision on the last floor,
checking every state hash on the way, so run this under the capture node's
libm. An episode ends within --within decisions when it dies or wins within
that many.

Prints per tier and root the count of each outcome (a win that later died or
timed out counts once, as a win), how many end within --within, and quantiles
of the decisions to each episode's end and to each win. --output gets the
same, and every episode's ending, as JSON.

Examples:
  priml/baselines/craftax/ghosts/endings.py /opt/scratch/artifacts/craftax/ghosts/capture/w15 --tier boss:3 --output /opt/scratch/artifacts/craftax/ghosts/capture/w15-endings.json

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

import argparse
import dataclasses
import json

from priml.baselines.craftax.game.state import (
    NUM_LEVELS,
    Achievement,
    env_state,
)
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import (
    read_manifest,
    read_records,
    read_summaries,
)
from priml.lib.codec import from_plain
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Sequence

    from priml.baselines.craftax.world_model.archive import (
        EpisodeSummary,
        Record,
    )


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
    report: list[dict[str, object]] = []
    for spec in flags.tier:
        name, _, arm = spec.partition(":")
        if not arm.isdigit():
            parser.error(f"--tier {spec}: expected NAME:ARM.")
        for root in flags.roots:
            ends = episode_endings(root, arm=int(arm))
            counts = summary(ends, within=flags.within)
            print(json.dumps({"tier": name, "root": str(root), **counts}))  # noqa: T201 -- CLI result.
            report.append(
                {
                    "tier": name,
                    "root": str(root),
                    **counts,
                    "episodes": [dataclasses.asdict(end) for end in ends],
                },
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n")
    return 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Ending:
    """How one episode ends.

    Attributes:
      ordinal: The episode's ordinal.
      decisions: Its recorded decisions.
      outcome: Whether it defeated the necromancer, else died or timed out.
      end: The decisions to its win, or else all of them.

    """

    ordinal: int
    decisions: int
    outcome: Literal["win", "death", "timeout"]
    end: int


def unlocked_at(
    record: replay.Replayable,
    *,
    achievement: int,
    after: int,
) -> int:
    """Return the decisions an episode plays until it unlocks an achievement.

    Args:
      record: The episode's record; every state hash met is checked.
      achievement: The achievement, which the state before decision ``after``
        lacks.
      after: A decision at or before the unlock, such as the first decision on
        the floor where it happens.

    Returns:
      decisions: The least ``d`` whose state before decision ``d`` holds the
        achievement; the episode's length when only its final state does.

    """
    length = len(record.actions)
    low, step = after, 1
    high = low + step
    while high < length and not _holds(record, achievement=achievement, decision=high):
        low, step = high, 2 * step
        high = low + step
    high = min(high, length)
    while high - low > 1:
        middle = (low + high) // 2
        if _holds(record, achievement=achievement, decision=middle):
            high = middle
        else:
            low = middle
    return high


def episode_endings(root: Path, *, arm: int) -> list[Ending]:
    """Return how each episode of an arm under ``root`` ends, in ordinal order.

    Args:
      root: Capture root holding ``{train,val}/arm{arm}/w{worker}/``.
      arm: The tier's capture arm.

    Returns:
      endings: One per episode.

    """
    ends: list[Ending] = []
    for manifest in sorted(root.glob(f"*/arm{arm}/w*/MANIFEST.jsonl")):
        for line in read_manifest(manifest.parent):
            summaries = read_summaries(manifest.parent, line)
            records = read_records(manifest.parent, line, summaries=summaries)
            ends += map(_ending, summaries, records)
    return sorted(ends, key=lambda end: end.ordinal)


def summary(ends: Sequence[Ending], *, within: int) -> dict[str, object]:
    """Return the counts and quantiles of a tier's endings.

    Args:
      ends: Endings from :func:`episode_endings`.
      within: Decisions within which an episode that dies or wins ends.

    Returns:
      summary: The count of each outcome, ``ended_within``, and quantiles
        (min, p10, p25, p50, p75, p90, max) of the decisions to the end of
        every episode, ``end_decisions``, and to every win, ``win_decisions``.

    """
    outcomes = [end.outcome for end in ends]
    return {
        **{o: outcomes.count(o) for o in ("death", "timeout", "win")},
        "ended_within": sum(
            end.end <= within and end.outcome != "timeout" for end in ends
        ),
        "end_decisions": _quantiles([end.end for end in ends]),
        "win_decisions": _quantiles(
            [end.end for end in ends if end.outcome == "win"],
        ),
    }


class Flags(Protocol):
    """Parsed command-line flags."""

    roots: list[Path]
    tier: list[str]
    within: int
    output: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("roots", nargs="+", type=Path, help="Capture roots.")
    parser.add_argument(
        "--tier",
        action="append",
        required=True,
        metavar="NAME:ARM",
        help="A tier and its arm; repeatable.",
    )
    parser.add_argument(
        "--within",
        type=int,
        default=10_000,
        help="Decisions within which a death or win counts as ending.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Report JSON.")


def _ending(summary: EpisodeSummary, record: Record) -> Ending:
    """Return how one published episode ends."""
    line = summary.summary
    won = Achievement.DEFEAT_NECROMANCER in from_plain(line["achievements"], list[int])
    died = bool(from_plain(line["death"], int))
    if summary.floors is None:
        raise ValueError("Expected summary.floors is not None.")
    return Ending(
        ordinal=from_plain(line["episode"], int),
        decisions=summary.decisions,
        outcome="win" if won else "death" if died else "timeout",
        # The necromancer lives on the last floor: no win precedes reaching it.
        end=unlocked_at(
            record,
            achievement=Achievement.DEFEAT_NECROMANCER,
            after=min(d for d, f in summary.floors.changes if f == NUM_LEVELS - 1),
        )
        if won
        else summary.decisions,
    )


def _holds(record: replay.Replayable, *, achievement: int, decision: int) -> bool:
    """Return whether the state before ``decision`` holds ``achievement``."""
    start = replay.save(*replay.reset_world(record.receipt.world_seed))
    state = replay.xor_bytes(replay.origin(record, decision=decision), right=start)
    states, _ = replay.load(state)
    return bool(env_state(states, 0).achievements[achievement])


def _quantiles(values: Sequence[int]) -> dict[str, int]:
    """Return the lower quantiles of ``values``, numpy's ``"lower"``; empty for none."""
    if not values:
        return {}
    ordered = sorted(values)
    points = {"min": 0.0, "p10": 0.1, "p25": 0.25, "p50": 0.5}
    points |= {"p75": 0.75, "p90": 0.9, "max": 1.0}
    return {name: ordered[int(p * (len(ordered) - 1))] for name, p in points.items()}


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
