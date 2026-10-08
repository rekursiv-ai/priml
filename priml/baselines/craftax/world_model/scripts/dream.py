#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Generate episodes from a trained world model and compare them with real ones.

Every engine row begins a new world at start and runs open-loop for --decisions
decisions, its actions sampled from the model's own action head, so the model
plays as the behaviour policy it learned inside the world it learned. Each
row's first episode, cut at --decisions, is summarized beside --reference real
episodes from their world's reset, drawn uniformly at random from the corpus's
split and arms and cut at the same horizon (episodes.py lists the statistics,
and where each side departs from the game's rules); a replay shard's episodes
are replayed. The model is built from the experiment factory that trained it
and takes the checkpoint's weights; on CUDA it runs in bfloat16. A corpus with
no such episode is refused before anything is generated.

OUTPUT, a new directory, receives report.json (both summaries, their distances,
both sides' rule departures, and the run's settings and speed), samples.pt (the
first --bundles generated and real episodes as token tensors), and
bundles/generated-ROW, viewer bundles of the first --bundles rows
(viewer/games.mjs build renders them).

Examples:
  priml/baselines/craftax/world_model/scripts/dream.py /opt/scratch/runs/craftax-world-model/exp001/checkpoints/step_00001525.pt --rows 256 --decisions 4000 --output /opt/scratch/artifacts/craftax/world-model/dreams-s0

'''
# fmt: on

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import argparse
import dataclasses
import json
import time

from torch import Tensor

import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    read_corpus,
    read_summaries,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
)
from priml.baselines.craftax.world_model.checkpoint import (
    load_world_model,
)
from priml.baselines.craftax.world_model.dream import Rollout, dream
from priml.baselines.craftax.world_model.engine import Engine
from priml.baselines.craftax.world_model.episodes import (
    Episodes,
    archived_episodes,
    compare,
    departures,
    first_episodes,
)
from priml.baselines.craftax.world_model.experiments import (
    WorldModelLoop,
)
from priml.baselines.craftax.world_model.snapshots import replay_episodes
from priml.baselines.craftax.world_model.viewer.bundle import (
    stream_of,
    write_bundle,
)
from priml.paths import validated_output_path


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the report, samples, and bundles are written.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    output = validated_output_path(flags.output)
    device = torch.device(flags.device)
    model, config = load_world_model(
        flags.experiment,
        flags.checkpoint,
        overrides=flags.override,
    )
    assert isinstance(config, WorldModelLoop.Config)
    corpus = flags.corpus or Path(config.dataset.corpus)
    real = sample_real(
        corpus,
        split=VALIDATION if flags.split == "val" else TRAIN,
        arms=flags.arms,
        count=flags.reference,
        seed=flags.seed,
    )
    if not real:
        raise ValueError(
            f"{corpus} holds no {flags.split} episode of arms {flags.arms} from "
            f"its world's reset to compare with; --reference is {flags.reference}.",
        )
    output.mkdir(parents=True)
    # bfloat16 halves the KV cache, 252 MB a row at 8,192 positions.
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    engine = Engine(
        model.to(device=device, dtype=dtype),
        rows=flags.rows,
        t_max=config.dataset.t_g,
        generator=torch.Generator(device).manual_seed(flags.seed),
    )
    clock = time.monotonic()
    rollout = dream(engine, decisions=flags.decisions)
    seconds = time.monotonic() - clock
    rollout = Rollout(
        **{
            f.name: cast("Tensor", getattr(rollout, f.name)).cpu()
            for f in dataclasses.fields(rollout)
        },
    )
    generated = first_episodes(rollout)
    reference = archived_episodes(real, horizon=flags.decisions)
    report = compare(generated, reference)
    report["departures"] = {
        "generated": departures(generated),
        "real": departures(reference),
    }
    report["run"] = {
        "checkpoint": str(flags.checkpoint),
        "experiment": flags.experiment,
        "overrides": list(flags.override),
        "corpus": str(corpus),
        "split": flags.split,
        "arms": list(flags.arms),
        "seed": flags.seed,
        "device": str(device),
        "dtype": str(dtype),
        "t_max": config.dataset.t_g,
        "decisions_generated": flags.rows * flags.decisions,
        "seconds": seconds,
        "decisions_per_second": flags.rows * flags.decisions / seconds,
    }
    (output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    torch.save(
        {
            "generated": _first(generated, count=flags.bundles),
            "real": _first(reference, count=flags.bundles),
        },
        output / "samples.pt",
    )
    for row in range(min(flags.bundles, flags.rows)):
        write_bundle(
            stream_of(rollout, row=row),
            output / "bundles" / f"generated-{row}",
            title=f"Generated row {row}",
            provenance=f"{flags.checkpoint}, seed {flags.seed}, action head sampled.",
        )
    print(
        f"{flags.rows * flags.decisions:,} decisions in {seconds:.0f} s; "
        f"distance {report['distance']}; report {output / 'report.json'}.",
    )
    return 0


def sample_real(
    corpus: Path,
    *,
    split: int,
    arms: Sequence[int],
    count: int,
    seed: int,
) -> list[Episode]:
    """Draw up to ``count`` episodes of a corpus uniformly at random, without replacement.

    Only episodes from their world's reset are eligible, as the generated ones
    begin there: a branch, which starts mid-game, is skipped.

    Args:
      corpus: Corpus file naming published shards.
      split: ``TRAIN`` or ``VALIDATION``; only episodes of this split are drawn.
      arms: Behaviour-mixture arms to draw from.
      count: Episodes to draw; every eligible one when there are fewer.
      seed: Seed of the draw.

    Returns:
      episodes: The drawn episodes, in draw order.

    """
    candidates = [
        (directory, line, summary)
        for directory, line in read_corpus(corpus)
        for summary in read_summaries(directory, line)
        if summary.receipt.split == split and summary.receipt.arm in arms
    ]
    order = torch.randperm(
        len(candidates),
        generator=torch.Generator().manual_seed(seed),
    )
    episodes: list[Episode] = []
    for index in order:
        if len(episodes) == count:
            break
        directory, line, summary = candidates[int(index)]
        (episode,) = replay_episodes(directory, line, summaries=[summary])
        if not episode.origin:
            episodes.append(episode)
    return episodes


class Flags(Protocol):
    """Parsed command-line flags."""

    checkpoint: Path
    experiment: str
    override: list[str]
    corpus: Path | None
    split: str
    arms: tuple[int, ...]
    rows: int
    decisions: int
    reference: int
    bundles: int
    seed: int
    device: str
    output: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("checkpoint", type=Path, help="TrainLoop checkpoint (.pt).")
    parser.add_argument(
        "--experiment",
        default="priml.baselines.craftax.world_model.experiments.exp001",
        help="Dotted path of the experiment factory that trained the checkpoint.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Config override the run was launched with; repeatable.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Corpus file of the real episodes; default the run's.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Split of the real episodes; default val.",
    )
    parser.add_argument(
        "--arms",
        type=lambda text: tuple(int(arm) for arm in text.split(",")),
        default=(0, 1, 2, 3),
        help="Comma-separated arms of the real episodes; default 0,1,2,3.",
    )
    parser.add_argument("--rows", type=int, default=256, help="Generated episodes.")
    parser.add_argument(
        "--decisions",
        type=int,
        default=4096,
        help="Decisions per row, the horizon.",
    )
    parser.add_argument(
        "--reference",
        type=int,
        default=256,
        help="Real episodes to draw.",
    )
    parser.add_argument(
        "--bundles",
        type=int,
        default=8,
        help="Episodes kept as samples and bundles.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--output", type=Path, required=True, help="New directory.")


def _first(episodes: Episodes, *, count: int) -> dict[str, Tensor]:
    """Return copies of the first ``count`` episodes' tensors by field name."""
    # ``torch.save`` of a view writes its whole storage: every row, at the
    # parent's row stride.
    return {
        field.name: cast("Tensor", getattr(episodes, field.name))[:count].clone()
        for field in dataclasses.fields(episodes)
    }


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
