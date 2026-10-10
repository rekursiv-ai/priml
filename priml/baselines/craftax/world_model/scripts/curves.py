#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Compare world-model runs' per-update training objectives with a reference run.

Every run logs train/loss, nanochat's debiased EMA of each update's mean
objective (WorldModelTrainStep.Config.loss_smoothing, 0.9). Consecutive values
give back each update's own mean exactly, so runs compare update by update
instead of through the smoothing's lag. Each candidate is reported against the
reference: the mean and last-100 absolute difference, the mean signed
difference, the mean signed difference per 40-update window, and the means
over the last 50 updates. Measure the bands to judge them by: a same-seed
repeat of the reference gives the nondeterminism band, and another
dataset.sampler_seed the data-order band. A speed change keeps the loss when
its differences stay inside the repeat band.

Runs must log every update from the first (num_steps_log=1) and must not
resume: a gap, or the EMA restarting at a resume, would misattribute updates.

training_report reads one run's whole history for report.py's stability
criteria: finite losses and gradient norms, loss spikes and whether they
recover, and whether each modality's validation bits per byte fell.

Examples:
  priml/baselines/craftax/world_model/scripts/curves.py RUN0 RUN1 RUN2
  priml/baselines/craftax/world_model/scripts/curves.py RUN0 RUN1 --output /opt/scratch/artifacts/craftax/world-model/curves.json

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import dataclasses
import itertools
import json
import math
import statistics

from priml.baselines.craftax.world_model.metric import MODALITIES
from priml.lib.codec import PlainTree, from_plain, to_plain
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import wandb
else:
    from wrapt import lazy_import

    wandb = lazy_import("wandb")


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class CurveDifference:
    """A candidate's per-update objective minus the reference's.

    Attributes:
      updates: Updates both runs logged.
      mean_abs: Mean absolute difference over them.
      last100_mean_abs: Mean absolute difference over the last 100.
      mean_signed: Mean signed difference; negative when the candidate is lower.
      window40_signed: Mean signed difference per window of 40 updates.
      tail50_reference: The reference's mean objective over the last 50.
      tail50_candidate: The candidate's mean objective over the last 50.

    """

    updates: int
    mean_abs: float
    last100_mean_abs: float
    mean_signed: float
    window40_signed: list[float]
    tail50_reference: float
    tail50_candidate: float


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Spike:
    """A logged loss far above the losses before it.

    Attributes:
      step: The update that logged it.
      loss: Its logged loss.
      baseline: The median of the ``SpikeRule.history`` losses before it.
      recovered: Whether a loss within ``SpikeRule.recovery`` of the baseline
        follows within ``SpikeRule.horizon`` updates, or none is logged after.

    """

    step: int
    loss: float
    baseline: float
    recovered: bool


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class SpikeRule:
    """What counts as a loss spike and as its recovery.

    Attributes:
      history: Logged losses before a step whose median is its baseline.
      ratio: A loss above this multiple of its baseline is a spike.
      horizon: Logged losses after a spike searched for its recovery.
      recovery: A loss at most this multiple of the baseline recovers it.

    """

    history: int = 20
    ratio: float = 1.3
    horizon: int = 50
    recovery: float = 1.05


def update_objectives(ema: Mapping[int, float], *, beta: float) -> dict[int, float]:
    """Return each update's mean objective from the logged debiased EMA.

    The step logs ``m_n / (1 - beta**n)`` with ``m_n = beta·m_{n-1} + (1-beta)·x_n``
    and ``m_0 = 0``, so ``x_n = (m_n - beta·m_{n-1}) / (1 - beta)``.

    Args:
      ema: ``train/loss`` per update, keyed 1, 2, ... without a gap.
      beta: The EMA coefficient.

    Returns:
      objectives: Per update, the mean objective the EMA absorbed.

    Raises:
      ValueError: The updates do not run 1, 2, ... without a gap.

    """
    steps = sorted(ema)
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError(
            f"Updates must run 1..n without a gap; got {steps[:3]}..{steps[-3:]}.",
        )
    objectives: dict[int, float] = {}
    previous = 0.0
    for step in steps:
        current = ema[step] * (1 - beta**step)
        objectives[step] = (current - beta * previous) / (1 - beta)
        previous = current
    return objectives


def compare(
    reference: Mapping[int, float],
    candidate: Mapping[int, float],
) -> CurveDifference:
    """Return the candidate's differences from the reference over shared updates.

    Args:
      reference: Per-update objective of the reference run.
      candidate: Per-update objective of the run compared with it.

    Returns:
      difference: Candidate minus reference, summarized.

    """
    steps = sorted(set(reference) & set(candidate))
    diffs = [candidate[s] - reference[s] for s in steps]
    late = diffs[-100:]
    tail = steps[-50:]
    return CurveDifference(
        updates=len(steps),
        mean_abs=sum(abs(d) for d in diffs) / len(diffs),
        last100_mean_abs=sum(abs(d) for d in late) / len(late),
        mean_signed=sum(diffs) / len(diffs),
        window40_signed=[
            sum(diffs[i : i + 40]) / len(diffs[i : i + 40])
            for i in range(0, len(diffs), 40)
        ],
        tail50_reference=sum(reference[s] for s in tail) / len(tail),
        tail50_candidate=sum(candidate[s] for s in tail) / len(tail),
    )


def spikes(
    steps: Sequence[int],
    losses: Sequence[float],
    *,
    rule: SpikeRule,
) -> list[Spike]:
    """Return every logged loss above ``rule.ratio`` times the median before it.

    Args:
      steps: The update of each logged loss, in order.
      losses: The logged losses.
      rule: The spike and recovery thresholds.

    Returns:
      spikes: Each spike, in order.

    """
    found: list[Spike] = []
    for i in range(rule.history, len(losses)):
        baseline = statistics.median(losses[i - rule.history : i])
        if losses[i] > rule.ratio * baseline:
            later = losses[i + 1 : i + 1 + rule.horizon]
            found.append(
                Spike(
                    step=steps[i],
                    loss=losses[i],
                    baseline=baseline,
                    recovered=not later
                    or any(v <= rule.recovery * baseline for v in later),
                ),
            )
    return found


def training_report(
    rows: Sequence[Mapping[str, object]],
    *,
    rule: SpikeRule,
) -> dict[str, PlainTree]:
    """Return one run's training stability and validation trend from its history.

    Args:
      rows: The run's logged rows, in order: ``train/*`` series, and ``val/*``
        at each evaluation, keyed by series name with ``_step``.
      rule: The spike and recovery thresholds.

    Returns:
      report: ``loss_finite`` and ``grad_norm_finite``; ``spikes`` and
        ``unrecovered_spikes``; how many updates logged each stability series;
        the first, last, and largest gradient norm, the last clip fraction,
        and the largest attention logit bound; ``eval_steps``; per modality
        the first and last validation bits per byte, whether it fell, and
        whether it never rose; ``final``, every ``val/*`` series' last value;
        and ``timing``.

    """
    train = [r for r in rows if r.get("train/loss") is not None]
    steps = [from_plain(r.get("_step"), int, default=i) for i, r in enumerate(train)]
    loss = [from_plain(r["train/loss"], float) for r in train]
    series = {
        name: [
            from_plain(r[f"train/{name}"], float)
            for r in train
            if r.get(f"train/{name}") is not None
        ]
        for name in ("grad_norm", "clip_fraction", "max_attention_logit")
    }
    grad = series["grad_norm"]
    evals = [r for r in rows if r.get("val/bpb") is not None]
    modalities: dict[str, PlainTree] = {}
    for name in MODALITIES:
        values = [
            from_plain(r[f"val/bpb/{name}"], float)
            for r in evals
            if r.get(f"val/bpb/{name}") is not None
        ]
        if values:
            modalities[name] = {
                "bpb_first": values[0],
                "bpb_last": values[-1],
                "decreased": values[-1] < values[0],
                "monotone": all(b <= a for a, b in itertools.pairwise(values)),
            }
    found = spikes(steps, loss, rule=rule)
    final = {
        key: from_plain(value, float)
        for r in evals
        for key, value in r.items()
        if key.startswith("val/") and isinstance(value, int | float)
    }
    return {
        "last_step": steps[-1] if steps else None,
        "loss_finite": all(math.isfinite(v) for v in loss),
        "grad_norm_finite": all(math.isfinite(v) for v in grad),
        "logged": {name: len(values) for name, values in series.items()},
        "loss_first_last": [loss[0], loss[-1]] if loss else None,
        "grad_norm_first_last_max": [grad[0], grad[-1], max(grad)] if grad else None,
        "clip_fraction_last": (series["clip_fraction"] or [None])[-1],
        "max_attention_logit_max": max(series["max_attention_logit"], default=None),
        "spikes": [to_plain(s) for s in found],
        "unrecovered_spikes": sum(not s.recovered for s in found),
        "eval_steps": [from_plain(r.get("_step"), int, default=0) for r in evals],
        "modalities": modalities,
        "final": dict(final),
        "timing": timing(rows),
    }


def timing(rows: Sequence[Mapping[str, object]]) -> dict[str, PlainTree]:
    """Return a run's evaluation time, update time, and throughput from its history.

    Args:
      rows: The run's logged rows, in order.

    Returns:
      timing: Each evaluation's ``eval/time`` and their sum; the median and 95th
        percentile of ``train/dt``; the means of ``train/mfu`` and
        ``train/tok_per_sec``; the peak ``train/gpu_mem_allocated_gb``; and the
        last ``_runtime``, each None where the run logged none.

    """
    evals = _series(rows, "eval/time")
    dt = sorted(_series(rows, "train/dt"))
    mfu, rate = _series(rows, "train/mfu"), _series(rows, "train/tok_per_sec")
    runtime = _series(rows, "_runtime")
    return {
        "eval_seconds": list(evals),
        "eval_total_seconds": sum(evals),
        "dt_median_seconds": statistics.median(dt) if dt else None,
        "dt_p95_seconds": dt[int(0.95 * len(dt))] if dt else None,
        "mfu_mean": statistics.fmean(mfu) if mfu else None,
        "tok_per_sec_mean": statistics.fmean(rate) if rate else None,
        "gpu_mem_max_gb": max(
            _series(rows, "train/gpu_mem_allocated_gb"),
            default=None,
        ),
        "runtime_seconds": runtime[-1] if runtime else None,
    }


def run_history(project: str, run: str) -> list[dict[str, object]]:
    """Return every logged row of a W&B run, in order.

    Args:
      project: W&B ``entity/project``, or a project of the default entity.
      run: The run's id.

    Returns:
      rows: Each logged row, keyed by series name.

    """
    history = (
        wandb.Api(timeout=60).run(f"{project}/{run}").scan_history(page_size=2_000)
    )
    return [from_plain(row, dict[str, object]) for row in history]


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the comparison is printed.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    reference = update_objectives(
        _logged_loss(flags.project, flags.reference),
        beta=flags.beta,
    )
    report = {
        run: to_plain(
            compare(
                reference,
                update_objectives(_logged_loss(flags.project, run), beta=flags.beta),
            ),
        )
        for run in flags.candidates
    }
    text = json.dumps({"reference": flags.reference, "candidates": report}, indent=1)
    if flags.output:
        output = validated_output_path(flags.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    print(text)
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    reference: str
    candidates: list[str]
    project: str
    beta: float
    output: Path | None


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("reference", help="W&B run id of the reference run.")
    parser.add_argument("candidates", nargs="+", help="W&B run ids to compare.")
    parser.add_argument(
        "--project",
        default="craftax-world-model",
        help="W&B project, as entity/project or under the default entity.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
        help="The runs' loss_smoothing.",
    )
    parser.add_argument("--output", type=Path, help="Also write the report here.")


def _series(rows: Sequence[Mapping[str, object]], key: str) -> list[float]:
    """Return every number one series logged, in order."""
    return [
        from_plain(r[key], float) for r in rows if isinstance(r.get(key), int | float)
    ]


def _logged_loss(project: str, run: str) -> dict[int, float]:
    """Return a W&B run's ``train/loss`` per logged step."""
    rows = (
        wandb.Api().run(f"{project}/{run}").scan_history(keys=["_step", "train/loss"])
    )
    return {
        from_plain(cast("object", row["_step"]), int): from_plain(
            cast("object", row["train/loss"]),
            float,
        )
        for row in rows
        if row.get("train/loss") is not None
    }


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
