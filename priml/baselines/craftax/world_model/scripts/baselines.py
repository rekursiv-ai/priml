#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Score a trained world model against the trivial validation baselines.

The design's third base criterion: next-frame cell accuracy above copying the
current frame, and action, reward, and done NLL below their empirical
frequencies (baselines.py). The model is rebuilt from the experiment factory
that trained it, with the --override values the run was launched with, and
scored on the experiment's own validation micro-batches -- rank 0's, so for a
run trained on several ranks a share of the windows its periodic evaluation
read -- with the kernels and autocast it trained under on CUDA (scoring.py).
OUTPUT receives the report and the run's settings as JSON.

Examples:
  priml/baselines/craftax/world_model/scripts/baselines.py /opt/scratch/runs/craftax-world-model/exp001/checkpoints/step_00001525.pt --override dataset.sampler_seed=0 --output /opt/scratch/artifacts/craftax/world-model/baselines-s0.json

'''
# fmt: on

from pathlib import Path
from typing import Protocol, cast

import argparse
import json
import time

from torch import Tensor

import torch

from priml.baselines.craftax.world_model.baselines import report, tally
from priml.baselines.craftax.world_model.batch import PackedBatch
from priml.baselines.craftax.world_model.experiments import (
    WorldModelLoop,
)
from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.scoring import (
    autocast,
    load_trained,
)
from priml.lib.codec import PlainTree
from priml.paths import validated_output_path


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
    flags = cast("Flags", parser.parse_args())
    output = validated_output_path(flags.output)
    device = torch.device(flags.device)
    model, config = load_trained(
        flags.experiment,
        flags.checkpoint,
        overrides=flags.override,
        device=device,
    )
    result, run = evaluate(model, config, device=device)
    result["run"] = {
        "checkpoint": str(flags.checkpoint),
        "experiment": flags.experiment,
        "overrides": list(flags.override),
        **run,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1) + "\n")
    print(
        f"beats {result['beats']}; model {result['model_nll']}; "
        f"empirical {result['empirical_nll']}; cells {result['cell_accuracy']}.",
    )
    return 0


def evaluate(
    model: WorldModel,
    config: WorldModelLoop.Config,
    *,
    device: torch.device,
) -> tuple[dict[str, PlainTree], dict[str, PlainTree]]:
    """Score ``model`` against the baselines on ``config``'s validation micro-batches.

    Args:
      model: The trained model, on ``device``.
      config: Its experiment's finalized config; its dataset serves the batches.
      device: Where the model scores.

    Returns:
      report: ``baselines.report`` of every validation micro-batch.
      run: The corpus, the batch count, and the seconds scoring took.

    """
    config.dataset.device = device
    dataset = config.dataset.make()
    clock = time.monotonic()
    tallies: list[dict[str, Tensor]] = []
    with autocast(config, device):
        for batch in dataset.eval_dataloader():
            media = batch["media"]
            assert isinstance(media, PackedBatch)
            tallies.append(tally(model, media))
    run: dict[str, PlainTree] = {
        "corpus": str(config.dataset.corpus),
        "batches": len(tallies),
        "seconds": time.monotonic() - clock,
    }
    return report(tallies), run


class Flags(Protocol):
    """Parsed command-line flags."""

    checkpoint: Path
    experiment: str
    override: list[str]
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
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument("--output", type=Path, required=True, help="Report JSON.")


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
