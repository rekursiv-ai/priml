#!/bin/sh
# ruff: noqa: EXE003, D300, D205, T201 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Choose the one world every ghost-overlay tier plays, from the tiers' pilots.

Each --tier is NAME:ARM:METRICS: a tier's name, its capture arm under ROOT, and
the metrics.json of its policy's final evaluation (about 10,000 episodes over
the pool worlds). A tier's outcomes on each world its arm played -- episodes,
mean achievement return and decisions, the share of episodes that stood on each
floor, deaths, timeouts and necromancer wins -- are read from the published
summaries alone.

A world's distance from a tier's evaluation is the relative error of its mean
return plus the earth mover's distance between the two deepest-floor
distributions relative to the evaluation's mean deepest floor: the sum over
floors 1-8 of the differences in reach, over the sum of the evaluation's
reach. The chosen world has the least distance summed over the tiers, among
the worlds every tier played; ties go to the lower seed.

Prints a table per tier and the ranking, and writes them with the choice to
--output as JSON. On the capture root of one world it reports that world's
outcomes per tier.

Examples:
  priml/baselines/craftax/ghosts/select_world.py /opt/scratch/artifacts/craftax/ghosts/capture/pilot --tier early:0:/opt/scratch/runs/craftax/exp001/metrics.json --tier medium:1:/opt/scratch/runs/craftax/exp102/metrics.json --tier high:2:/opt/scratch/runs/craftax/exp103/metrics.json --output /opt/scratch/artifacts/craftax/ghosts/capture/world.json

'''
# fmt: on

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import dataclasses
import json
import operator

from priml.baselines.craftax.game.state import Achievement
from priml.baselines.craftax.metric import FLOOR_NAMES
from priml.baselines.craftax.world_model.archive import (
    EpisodeSummary,
    read_manifest,
    read_summaries,
)
from priml.lib.codec import from_plain, loads
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the choice is written.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    output = validated_output_path(flags.output)
    tiers = [_tier(parser, spec) for spec in flags.tier]
    played = {name: outcomes(flags.root, arm=arm) for name, arm, _ in tiers}
    evaluations = {name: evaluation(path) for name, _, path in tiers}
    ranking = rank(played, evaluations=evaluations)
    for name, arm, _ in tiers:
        print(_table(name, arm=arm, played=played[name], evaluated=evaluations[name]))
    print("| world | distance |\n| ---: | ---: |")
    print("\n".join(f"| {seed} | {total:.4f} |" for seed, total in ranking))
    print(f"\nWorld {ranking[0][0]}. Written to {output}.")
    record = {
        "root": str(flags.root),
        "world": ranking[0][0],
        "ranking": [{"world": seed, "distance": total} for seed, total in ranking],
        "tiers": {
            name: {
                "arm": arm,
                "metrics": str(path),
                "evaluation": dataclasses.asdict(evaluations[name]),
                "worlds": {
                    str(seed): {
                        **dataclasses.asdict(world),
                        "distance": distance(world, evaluations[name]),
                    }
                    for seed, world in played[name].items()
                },
            }
            for name, arm, path in tiers
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=1) + "\n")
    return 0


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Outcomes:
    """A tier's outcomes over the episodes it played on one world.

    Attributes:
      episodes: Episode count.
      mean_return: Mean achievement return.
      mean_decisions: Mean decisions per episode.
      reach: Share of the episodes that stood on each floor, ``[9]``.
      deaths: Episodes that ended in death.
      timeouts: Episodes that ended at the timeout.
      wins: Episodes that defeated the necromancer.

    """

    episodes: int
    mean_return: float
    mean_decisions: float
    reach: tuple[float, ...]
    deaths: int
    timeouts: int
    wins: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Evaluation:
    """What the world choice compares of a tier's final evaluation.

    Attributes:
      mean_return: Mean achievement return.
      reach: Share of the episodes that reached each floor, ``[9]``.

    """

    mean_return: float
    reach: tuple[float, ...]


def outcomes(root: Path, *, arm: int) -> dict[int, Outcomes]:
    """Return an arm's outcomes on each world it played, from its published summaries.

    Args:
      root: Archive root holding ``{train,val}/arm{arm}/w{worker}/``.
      arm: The tier's capture arm.

    Returns:
      outcomes: Keyed by world seed, in seed order.

    """
    played: dict[int, list[EpisodeSummary]] = defaultdict(list)
    for manifest in sorted(root.glob(f"*/arm{arm}/w*/MANIFEST.jsonl")):
        for line in read_manifest(manifest.parent):
            for summary in read_summaries(manifest.parent, line):
                played[summary.receipt.world_seed].append(summary)
    return {seed: _outcomes(played[seed]) for seed in sorted(played)}


def evaluation(path: Path) -> Evaluation:
    """Return what the world choice compares of a final evaluation's ``metrics.json``."""
    metrics = from_plain(loads(path.read_text()), dict[str, float])
    return Evaluation(
        mean_return=metrics["eval/craftax_episode_return"],
        reach=tuple(metrics[f"eval/craftax_{name}"] for name in FLOOR_NAMES),
    )


def distance(world: Outcomes, evaluated: Evaluation) -> float:
    """Return how far a tier's outcomes on a world lie from its evaluation's.

    Args:
      world: The tier's outcomes on the world.
      evaluated: The tier's final evaluation.

    Returns:
      distance: The relative error of the mean return, plus the earth mover's
        distance between the deepest-floor distributions over the
        evaluation's mean deepest floor. Floor ``k`` is reached only through
        floor ``k - 1``, so reach is the deepest floor's survival function, and
        the earth mover's distance is the summed difference in reach.

    """
    floors = sum(
        abs(here - there)
        for here, there in zip(world.reach[1:], evaluated.reach[1:], strict=True)
    )
    error = abs(world.mean_return - evaluated.mean_return) / evaluated.mean_return
    return error + floors / sum(evaluated.reach[1:])


def rank(
    played: Mapping[str, Mapping[int, Outcomes]],
    *,
    evaluations: Mapping[str, Evaluation],
) -> list[tuple[int, float]]:
    """Rank the worlds every tier played by their distance summed over the tiers.

    Args:
      played: Each tier's outcomes per world.
      evaluations: Each tier's final evaluation.

    Returns:
      ranking: ``(world, distance)`` nearest first; ties in seed order.

    """
    tiers = list(played.values())
    totals = [
        (seed, sum(distance(played[n][seed], evaluations[n]) for n in played))
        for seed in sorted(tiers[0])
        if all(seed in tier for tier in tiers)
    ]
    return sorted(totals, key=operator.itemgetter(1))


class Flags(Protocol):
    """Parsed command-line flags."""

    root: Path
    tier: list[str]
    output: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("root", type=Path, help="Capture root of the pilots.")
    parser.add_argument(
        "--tier",
        action="append",
        required=True,
        metavar="NAME:ARM:METRICS",
        help="A tier, its arm and its evaluation's metrics.json; repeatable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON of the outcomes, the ranking and the chosen world.",
    )


def _tier(parser: argparse.ArgumentParser, spec: str) -> tuple[str, int, Path]:
    """Parse one NAME:ARM:METRICS spec, or stop with a usage error."""
    parts = spec.split(":", 2)
    if len(parts) != 3 or not parts[1].isdigit():
        parser.error(f"--tier {spec}: expected NAME:ARM:METRICS.")
    name, arm, metrics = parts
    return name, int(arm), Path(metrics)


def _outcomes(summaries: Sequence[EpisodeSummary]) -> Outcomes:
    """Return the outcomes of one tier's episodes on one world."""
    count = len(summaries)
    lines = [summary.summary for summary in summaries]
    floors = [from_plain(line["floors"], list[dict[str, object]]) for line in lines]
    reached = [[from_plain(floor["reached"], int) for floor in line] for line in floors]
    return Outcomes(
        episodes=count,
        mean_return=sum(from_plain(line["return"], float) for line in lines) / count,
        mean_decisions=sum(summary.decisions for summary in summaries) / count,
        reach=tuple(sum(floor) / count for floor in zip(*reached, strict=True)),
        deaths=sum(from_plain(line["death"], int) for line in lines),
        timeouts=sum(from_plain(line["timeout"], int) for line in lines),
        wins=sum(
            Achievement.DEFEAT_NECROMANCER
            in from_plain(line["achievements"], list[int])
            for line in lines
        ),
    )


def _table(
    name: str,
    *,
    arm: int,
    played: Mapping[int, Outcomes],
    evaluated: Evaluation,
) -> str:
    """Return one tier's outcomes per world as a Markdown table."""
    reach = "/".join(f"{100 * share:.1f}" for share in evaluated.reach)
    rows = [
        (
            f"\n{name} (arm {arm}): evaluation return {evaluated.mean_return:.2f}, "
            f"reach % {reach}\n"
        ),
        (
            "| world | episodes | return | decisions | deaths | timeouts | wins "
            "| reach % floors 0-8 | distance |"
        ),
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    rows += [
        f"| {seed} | {world.episodes} | {world.mean_return:.2f} "
        f"| {world.mean_decisions:.0f} | {world.deaths} | {world.timeouts} "
        f"| {world.wins} | {'/'.join(f'{100 * share:.1f}' for share in world.reach)} "
        f"| {distance(world, evaluated):.4f} |"
        for seed, world in played.items()
    ]
    return "\n".join(rows)


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
