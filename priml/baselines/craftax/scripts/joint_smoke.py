#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Run a few epochs of a joint sole world-model arm on CUDA, then one evaluation.

The arm is one of the experiments that train their world model with the
policy, exp112 and exp113, at its own geometry and recipe -- practice,
self-imitation, the 20-layer world model of its checkpoint, FlashAttention 4
and compiled kernels in the actor and in the replay -- driven epoch by epoch
through its train step, as the training loop drives it, with no tracker.
``--sliding`` swaps the arm's refill for exact 512-decision windows
(``Sliding``), its practice rows resuming their donors' histories
(``DonorHistory``). With
two slots, the experiment's, the rollout runs one epoch ahead of the learner;
with one, each epoch learns from a rollout of its own starting weights, so
``joint/feature_gap`` measures the actor against the learner at the same
weights, after a rebuild from the second epoch on. ``--episodes`` then plays
one evaluation of that many episodes beside training's state, as a run's
periodic evaluation does; 0 plays none.

OUTPUT is one JSON report: the build's seconds, each epoch's timings
(``epoch_seconds``, ``learner_seconds``, ``rollout_seconds``,
``rebuild_seconds``), transitions per second, ``joint/feature_gap`` and the
feature's telemetry, the evaluation's rollouts and seconds, and the
process's peak CUDA memory over training and then over the evaluation.

Examples:
  priml/baselines/craftax/scripts/joint_smoke.py /opt/scratch/artifacts/craftax/sole-wm-joint/smoke-exp112.json --experiment exp112 --episodes 1024

'''
# fmt: on

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import json
import time

from priml.baselines.craftax.experiments import (
    CraftaxTrainLoop,
    exp112,
    exp113,
)
from priml.baselines.craftax.train_step import CraftaxTrainStep
from priml.baselines.craftax.world_model.feature import (
    DonorHistory,
    Sliding,
    WorldModelFeature,
)
from priml.paths import validated_output_path


if TYPE_CHECKING:
    import torch

    from priml.lib.codec import PlainTree
else:
    from wrapt import lazy_import

    # ~1050 ms; only the run reads it, not the config the test builds.
    torch = lazy_import("torch")


ARMS: Final[dict[str, Callable[[], CraftaxTrainLoop]]] = {
    "exp112": exp112,
    "exp113": exp113,
}
"""The experiments that train their world model with the policy."""


def main() -> int:
    """Run the epochs and the evaluation, and write the report.

    Returns:
      status: 0.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    output = validated_output_path(flags.output)
    device = torch.device("cuda")
    config = smoke_config(
        flags.experiment,
        slots=flags.slots,
        episodes=flags.episodes,
        sliding=flags.sliding,
    )
    started = time.perf_counter()
    step = config.make()
    built = time.perf_counter() - started
    epochs: list[dict[str, float]] = []
    report: dict[str, PlainTree] = {
        "experiment": flags.experiment,
        "sliding": flags.sliding,
        "slots": flags.slots,
        "device": torch.cuda.get_device_name(device),
        "build_seconds": built,
    }
    try:
        for _ in range(flags.epochs):
            epochs.append(_epoch_metrics(step))
            print(json.dumps(epochs[-1]), flush=True)
        report["training_peak_allocated_gib"] = _gib(torch.cuda.max_memory_allocated())
        report["training_peak_reserved_gib"] = _gib(torch.cuda.max_memory_reserved())
        if flags.episodes:
            started = time.perf_counter()
            evaluator = step.make_evaluator()
            try:
                played = evaluator.play()
            finally:
                evaluator.close()
            report["evaluation_seconds"] = time.perf_counter() - started
            report["evaluation_rollouts"] = played.rollouts
            report["peak_allocated_gib"] = _gib(torch.cuda.max_memory_allocated())
            report["peak_reserved_gib"] = _gib(torch.cuda.max_memory_reserved())
    finally:
        step.close()
    report["epochs"] = cast("PlainTree", epochs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


def smoke_config(
    experiment: str,
    *,
    slots: int,
    episodes: int,
    sliding: bool = False,
) -> CraftaxTrainStep.Config:
    """Return a joint arm's train step, placed on CUDA, its slots and evaluation set.

    Args:
      experiment: One of :data:`ARMS`.
      slots: Rollout slots: the experiment's 2 runs the rollout one epoch
        ahead, 1 learns each epoch from a rollout of its starting weights.
      episodes: Episodes the evaluation plays; 0 keeps the experiment's.
      sliding: Read exact 512-decision windows, practice rows resuming their
        donors' histories, in place of the experiment's refill.

    Returns:
      config: The step's config, unfinalized.

    """
    loop = ARMS[experiment]()
    step = loop.step
    # The training loop's runtime places the step and roots its inputs; driven
    # alone, it says so here.
    step.parallelism.device = "cuda"
    step.base_dir = loop.base_dir
    step.rollout.num_slots = slots
    if episodes:
        step.evaluation.num_episodes = episodes
    if sliding:
        feature = step.feature
        assert isinstance(feature, WorldModelFeature.Config)
        feature.history = Sliding.Config()
        feature.practice = DonorHistory.Config()
        # A block of 32 steps, not 128, plans fewer rows that will not slide yet.
        feature.hook_interval = 32
    return step


class Flags(Protocol):
    """Parsed command-line flags."""

    output: Path
    experiment: str
    sliding: bool
    slots: int
    epochs: int
    episodes: int


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("output", type=Path, help="Where to write the JSON report.")
    parser.add_argument("--experiment", choices=tuple(ARMS), default="exp112")
    parser.add_argument(
        "--sliding",
        action="store_true",
        help="Read exact 512-decision windows with donor histories, not refill.",
    )
    parser.add_argument("--slots", type=int, choices=(1, 2), default=2)
    parser.add_argument("--epochs", type=int, default=3, help="Epochs to train.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Episodes of the evaluation after training; 0 plays none.",
    )


def _epoch_metrics(step: CraftaxTrainStep) -> dict[str, float]:
    """Train one epoch; return its timings, rate, feature gap and telemetry."""
    metrics = step.train_step().get("metrics", {})
    return {
        name: float(value)
        for name, value in metrics.items()
        if name.endswith("_seconds")
        or name.startswith(("joint/", "feature/"))
        or name in {"transitions_per_second", "total_loss"}
    }


def _gib(count: int) -> float:
    """Return a byte count in GiB."""
    return count / 2**30


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
