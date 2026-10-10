#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Run the design's five engine correctness checks on a trained world-model checkpoint.

engine_test.py runs these checks on tiny random models; this runs them on a
trained model's weights and a real validation segment, the first --decisions
decisions of the corpus's first validation episode longer than that (a replay
shard's episodes are replayed):

1. teacher-forced equivalence: the engine's per-slot log-probabilities on the
   segment against the training forward's target_terms, exact up to
   --tolerance in float32 on SDPA; and, reported only, how far the sampling
   configuration (a bfloat16 engine) is from the training numerics (the
   experiment's own kernels under its autocast);
2. prefill then step equals step only (float32);
3. resetting one row leaves the other rows unchanged (bfloat16, exact);
4. Gumbel-max frequencies match the softmax of the trained action head
   (chi-square over --draws draws, bins expected below 5 merged);
5. a fixed seed reproduces identical tokens, and a saved session stream
   reloads to identical tensors.

OUTPUT is a JSON file with each check's measurements and verdict, and
all_passed.

Examples:
  priml/baselines/craftax/world_model/scripts/engine_checks.py /opt/scratch/runs/craftax-world-model/exp001/checkpoints/step_00001525.pt /opt/scratch/datasets/craftax/world-model/archive-v1/corpora/base.json /opt/scratch/artifacts/craftax/world-model/engine-checks.json

'''
# fmt: on

from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import copy
import dataclasses
import json
import tempfile

from torch import Tensor

import torch

from priml.baselines.craftax.world_model.archive import (
    read_corpus,
    read_summaries,
)
from priml.baselines.craftax.world_model.batch import (
    Segment,
    pack_windows,
)
from priml.baselines.craftax.world_model.capture.seeds import VALIDATION
from priml.baselines.craftax.world_model.checkpoint import (
    load_world_model,
)
from priml.baselines.craftax.world_model.engine import (
    Control,
    Engine,
    Prefix,
)
from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.scoring import (
    autocast,
    load_trained,
)
from priml.baselines.craftax.world_model.session import Session, Stream
from priml.baselines.craftax.world_model.snapshots import replay_episodes
from priml.lib.codec import PlainTree
from priml.math.probability import gumbel_max
from priml.paths import validated_output_path


if TYPE_CHECKING:
    from scipy import stats
else:
    from wrapt import lazy_import

    stats = lazy_import("scipy", "stats")  # ~200 ms; only check_gumbel needs it.


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Checked:
    """One trained model in three forms, and the segment the checks run it on.

    Attributes:
      sdpa: The model in float32 on SDPA, on the device.
      trained: The model with the experiment's own kernels, float32, on the device.
      sampler: The model in bfloat16, as dreams sample it, on the device.
      autocast: Enters the experiment's training autocast on the device.
      segment: The real validation segment.
      t_max: Engine context in global positions.

    """

    sdpa: WorldModel
    trained: WorldModel
    sampler: WorldModel
    autocast: AbstractContextManager[object]
    segment: Segment
    t_max: int


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the report is written, whatever the checks found.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    output = validated_output_path(flags.output)
    device = torch.device(flags.device)
    sdpa, _ = load_world_model(
        flags.experiment,
        flags.checkpoint,
        overrides=flags.override,
    )
    trained, config = load_trained(
        flags.experiment,
        flags.checkpoint,
        overrides=flags.override,
        device=device,
    )
    sdpa = sdpa.to(device)
    checked = Checked(
        sdpa=sdpa,
        trained=trained,
        sampler=copy.deepcopy(sdpa).to(dtype=torch.bfloat16),
        autocast=autocast(config, device),
        segment=real_segment(flags.corpus, decisions=flags.decisions),
        t_max=flags.t_max or config.dataset.t_g,
    )
    results: dict[str, PlainTree] = {
        "checkpoint": str(flags.checkpoint),
        "corpus": str(flags.corpus),
        "decisions": flags.decisions,
    }
    results |= run_checks(checked, tolerance=flags.tolerance, draws=flags.draws)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=1) + "\n")
    print(f"all_passed {results['all_passed']}; wrote {output}.")
    return 0


def run_checks(
    checked: Checked,
    *,
    tolerance: float,
    draws: int,
) -> dict[str, PlainTree]:
    """Run the five checks in order.

    Args:
      checked: The model's forms and the segment.
      tolerance: Largest float32 log-probability difference that passes.
      draws: Gumbel-max draws from the action head.

    Returns:
      results: Each check's entry, then ``all_passed``.

    """
    results = check_teacher_forced(checked, tolerance=tolerance)
    results |= check_prefill(checked, tolerance=tolerance)
    results |= check_reset(checked.sampler, t_max=checked.t_max)
    results |= check_gumbel(
        checked.sampler,
        segment=checked.segment,
        t_max=checked.t_max,
        draws=draws,
    )
    results |= check_seed_and_session(checked.sampler, t_max=checked.t_max)
    results["all_passed"] = all(
        value.get("passed", True)
        for value in results.values()
        if isinstance(value, dict)
    )
    return results


def real_segment(corpus: Path, *, decisions: int) -> Segment:
    """Return the first decisions of the corpus's first long-enough validation episode.

    Args:
      corpus: Corpus file naming published shards.
      decisions: Decisions kept; the episode must have more than one past them.

    Returns:
      segment: Its first ``decisions`` decisions and the frame after them.

    Raises:
      ValueError: If no validation episode is long enough.

    """
    for directory, line in read_corpus(corpus):
        for summary in read_summaries(directory, line):
            if (
                summary.receipt.split == VALIDATION
                and summary.decisions > decisions + 1
            ):
                (episode,) = replay_episodes(directory, line, summaries=[summary])
                return Segment(
                    cells=episode.cells[: decisions + 1],
                    aux=episode.aux[: decisions + 1],
                    actions=episode.actions[:decisions],
                    reward=episode.reward[:decisions],
                    done=episode.done[:decisions],
                    starts_episode=not episode.origin,
                )
    raise ValueError(f"No validation episode of {corpus} is long enough.")


def check_teacher_forced(checked: Checked, *, tolerance: float) -> dict[str, PlainTree]:
    """Hold the engine's log-probabilities to the training forward's on the segment.

    Args:
      checked: The models and the segment.
      tolerance: Largest absolute log-probability difference that passes in float32.

    Returns:
      results: ``1_teacher_forced_fp32``, judged, and
        ``1_teacher_forced_bf16_vs_training_numerics``, reported.

    """
    segment = checked.segment
    expected, expected_action = training_logp(
        checked.sdpa,
        segment,
        context=nullcontext(),
    )
    got, got_action = engine_logp(checked.sdpa, segment, t_max=checked.t_max)
    diff = (got - expected).abs()
    adiff = (got_action - expected_action).abs()
    fp32: dict[str, PlainTree] = {
        "max_abs_slot_logp_diff": float(diff.max()),
        "mean_abs_slot_logp_diff": float(diff.mean()),
        "max_abs_action_logp_diff": float(adiff.max()),
        "jobs": int(got.shape[0]),
        "tolerance": tolerance,
        "passed": bool(diff.max() < tolerance and adiff.max() < tolerance),
    }
    expected, expected_action = training_logp(
        checked.trained,
        segment,
        context=checked.autocast,
    )
    got, got_action = engine_logp(checked.sampler, segment, t_max=checked.t_max)
    diff = (got - expected).abs()
    bf16: dict[str, PlainTree] = {
        "max_abs_slot_logp_diff": float(diff.max()),
        "mean_abs_slot_logp_diff": float(diff.mean()),
        "p99_abs_slot_logp_diff": float(diff.flatten().quantile(0.99)),
        "max_abs_action_logp_diff": float((got_action - expected_action).abs().max()),
        "mean_total_logp_per_decision_engine": float(got[1:].sum(-1).mean()),
        "mean_total_logp_per_decision_training": float(expected[1:].sum(-1).mean()),
    }
    return {
        "1_teacher_forced_fp32": fp32,
        "1_teacher_forced_bf16_vs_training_numerics": bf16,
    }


def check_prefill(checked: Checked, *, tolerance: float) -> dict[str, PlainTree]:
    """Require a prefilled row to decide as a row that stepped through the segment.

    Args:
      checked: The float32 model and the segment.
      tolerance: Largest absolute log-probability difference that passes.

    Returns:
      results: ``2_prefill_then_step_equals_step_only_fp32``.

    """
    model, segment, t_max = checked.sdpa, checked.segment, checked.t_max
    device = model.start.device
    stepped = Engine(model, rows=1, t_max=t_max, generator=torch.Generator(device))
    stepped.start(forced_control(stepped, segment, 0))
    for step in range(len(segment.actions)):
        stepped.decide(forced_control(stepped, segment, step))
    prefilled = Engine(model, rows=1, t_max=t_max, generator=torch.Generator(device))
    prefilled.prefill(torch.tensor([0]), prefix_of(segment))
    length_equal = bool(torch.equal(prefilled.state.length, stepped.state.length))
    frame_equal = bool(torch.equal(prefilled.state.cells, stepped.state.cells))
    same = 0
    max_logp = 0.0
    for seed in range(8):
        stepped.generator.manual_seed(seed)
        prefilled.generator.manual_seed(seed)
        a = _undone_decision(stepped)
        b = _undone_decision(prefilled)
        same += int(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
        max_logp = max(max_logp, float((a[2] - b[2]).abs().max()))
    return {
        "2_prefill_then_step_equals_step_only_fp32": {
            "length_equal": length_equal,
            "frame_equal": frame_equal,
            "free_decisions_identical_tokens": f"{same}/8",
            "max_abs_logp_diff": max_logp,
            "passed": length_equal
            and frame_equal
            and same == 8
            and max_logp < tolerance,
        },
    }


def check_reset(model: WorldModel, *, t_max: int) -> dict[str, PlainTree]:
    """Require a reset of row 0 to change row 0's samples and leave row 1's exact.

    Args:
      model: The sampling model.
      t_max: Engine context in global positions.

    Returns:
      results: ``3_reset_one_row_leaves_others_bf16``.

    """
    device = model.start.device
    runs: list[list[tuple[Tensor, Tensor, Tensor, Tensor]]] = []
    for reset in (False, True):
        engine = Engine(
            model,
            rows=2,
            t_max=t_max,
            generator=torch.Generator(device).manual_seed(3),
        )
        steps: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        for i in range(3):
            begun, decision = engine.step(no_done(engine))
            steps.append(
                (
                    begun.started.clone(),
                    decision.action.clone(),
                    decision.job.tokens.clone(),
                    decision.job.logp.clone(),
                ),
            )
            if reset and i == 0:
                engine.reset(torch.tensor([True, False]))
        runs.append(steps)
    row1_equal = all(
        torch.equal(a[1][1], b[1][1])
        and torch.equal(a[2][1], b[2][1])
        and torch.equal(a[3][1], b[3][1])
        for a, b in zip(runs[0], runs[1], strict=True)
    )
    row0_changed = not torch.equal(runs[0][1][3][0], runs[1][1][3][0])
    started = bool(runs[1][1][0][0])
    return {
        "3_reset_one_row_leaves_others_bf16": {
            "row1_identical": row1_equal,
            "row0_changed": row0_changed,
            "reset_row_started": started,
            "passed": row1_equal and row0_changed and started,
        },
    }


def check_gumbel(
    model: WorldModel,
    *,
    segment: Segment,
    t_max: int,
    draws: int,
) -> dict[str, PlainTree]:
    """Compare Gumbel-max draws with the action head's softmax after the segment.

    Args:
      model: The sampling model.
      segment: Prefilled before the action head is read.
      t_max: Engine context in global positions.
      draws: Draws from the head's logits.

    Returns:
      results: ``4_gumbel_max_matches_softmax_action_head``: the chi-square
        statistic, its degrees of freedom and p-value, and the head's entropy;
        it passes above p = 0.001.

    """
    device = model.start.device
    engine = Engine(
        model,
        rows=1,
        t_max=t_max,
        generator=torch.Generator(device).manual_seed(0),
    )
    engine.prefill(torch.tensor([0]), prefix_of(segment))
    logits = engine.action_logits()[0]
    samples = gumbel_max(
        logits.expand(draws, -1),
        generator=torch.Generator(device).manual_seed(4),
    )
    counts = torch.bincount(samples, minlength=logits.numel()).double().cpu()
    expected = logits.softmax(-1).double().cpu() * draws
    big = expected >= 5
    observed = torch.cat([counts[big], counts[~big].sum()[None]])
    wanted = torch.cat([expected[big], expected[~big].sum()[None]])
    keep = wanted > 0
    chi = float(((observed - wanted)[keep] ** 2 / wanted[keep]).sum())
    degrees = int(keep.sum()) - 1
    p = float(stats.chi2.sf(chi, degrees))
    return {
        "4_gumbel_max_matches_softmax_action_head": {
            "chi_square": chi,
            "degrees_of_freedom": degrees,
            "p_value": p,
            "draws": draws,
            "entropy_nats": float(-(logits.softmax(-1) * logits.log_softmax(-1)).sum()),
            "passed": p > 0.001,
        },
    }


def check_seed_and_session(model: WorldModel, *, t_max: int) -> dict[str, PlainTree]:
    """Require one seed to give one token stream, and a saved session to reload as saved.

    Args:
      model: The sampling model.
      t_max: Engine context in global positions.

    Returns:
      results: ``5_fixed_seed_and_session_reload``.

    """
    device = model.start.device
    tokens: list[Tensor] = []
    for _ in range(2):
        engine = Engine(
            model,
            rows=2,
            t_max=t_max,
            generator=torch.Generator(device).manual_seed(5),
        )
        run: list[Tensor] = []
        for _ in range(4):
            _, decision = engine.step(engine.control())
            run.append(
                torch.cat(
                    [
                        decision.action[:, None].float(),
                        decision.job.tokens.flatten(1).float(),
                        decision.job.logp,
                    ],
                    -1,
                ).clone(),
            )
        tokens.append(torch.stack(run))
    engine = Engine(
        model,
        rows=1,
        t_max=t_max,
        generator=torch.Generator(device).manual_seed(6),
    )
    session = Session(engine, row=0)
    session.prefill()
    session.autoplay(5)
    stream = session.stream()
    with tempfile.TemporaryDirectory(prefix="engine-checks-") as scratch:
        path = Path(scratch) / "session.pt"
        stream.save(path)
        loaded = Stream.load(path)
    reloaded = all(
        torch.equal(
            torch.as_tensor(getattr(stream, f.name)),
            torch.as_tensor(getattr(loaded, f.name)),
        )
        for f in dataclasses.fields(stream)
    )
    identical = bool(torch.equal(tokens[0], tokens[1]))
    return {
        "5_fixed_seed_and_session_reload": {
            "same_seed_identical": identical,
            "session_reload_identical": reloaded,
            "session_decisions": len(stream.action),
            "passed": identical and reloaded,
        },
    }


def training_logp(
    model: WorldModel,
    segment: Segment,
    *,
    context: AbstractContextManager[object],
) -> tuple[Tensor, Tensor]:
    """Return the training forward's log-probabilities on one segment.

    Args:
      model: The model, on its device.
      segment: The segment, one episode from its start.
      context: The autocast to run under.

    Returns:
      slots: Per job, each local slot's log-probability, float32 ``[J, L]``; a
        start job's reward and done, never scored, read 0 as the engine's do.
      action: Each executed action's log-probability, float32 ``[N]``.

    """
    # One window of whole 128-position blocks: start, then an obs and an act
    # position per decision.
    t_g = 128 * -(-(2 * len(segment.actions) + 2) // 128)
    batch = pack_windows([[segment]], t_g=t_g, s_max=1).to(model.start.device)
    with torch.no_grad(), context:
        logits = model.logits(batch)
        terms = model.target_terms(batch, logits)
    slots = torch.cat(
        [
            -terms["reward"][0].nll[:, None],
            -terms["done"][0].nll[:, None],
            -terms["board"][0].nll,
            -terms["hud"][0].nll,
        ],
        dim=-1,
    ).float()
    scored = terms["action"][1][0]
    action = -terms["action"][0].nll[0][scored].float()
    slots[0, :2] = 0.0
    return slots, action


def engine_logp(
    model: WorldModel,
    segment: Segment,
    *,
    t_max: int,
) -> tuple[Tensor, Tensor]:
    """Return the engine's log-probabilities of one segment's forced tokens.

    Args:
      model: The model, on its device.
      segment: The segment, one episode from its start.
      t_max: Engine context in global positions.

    Returns:
      slots: Per job, start job first, each local slot's log-probability,
        float32 ``[J, L]``.
      action: Each forced action's log-probability, float32 ``[N]``.

    """
    engine = Engine(
        model,
        rows=1,
        t_max=t_max,
        generator=torch.Generator(model.start.device).manual_seed(0),
    )
    begun = engine.start(forced_control(engine, segment, 0))
    jobs = [begun.job.logp[0]]
    actions: list[Tensor] = []
    for step in range(len(segment.actions)):
        decision = engine.decide(forced_control(engine, segment, step))
        jobs.append(decision.job.logp[0])
        actions.append(decision.action_logp[0])
    return torch.stack(jobs).float(), torch.stack(actions).float()


def forced_control(engine: Engine, segment: Segment, step: int) -> Control:
    """Return a control forcing every row to ``segment``'s decision ``step``.

    The start job is forced to the segment's first frame, the action to its
    ``step``-th action, and the act job to that decision's reward, done and
    next frame (the last frame where the segment has none after it).

    Args:
      engine: The engine the control is for.
      segment: The recorded decisions.
      step: The decision to force.

    Returns:
      control: Every row active, every slot forced.

    """
    control = engine.control()
    control.start_forced[:] = True
    control.start_tokens[:] = engine.job_tokens(
        reward=torch.tensor(0),
        done=torch.tensor(data=False),
        cells=segment.cells[0],
        aux=segment.aux[0],
    )
    control.action_forced[:] = True
    control.action[:] = int(segment.actions[step])
    following = min(step + 1, len(segment.cells) - 1)
    control.job_forced[:] = True
    control.job_tokens[:] = engine.job_tokens(
        reward=segment.reward[step],
        done=segment.done[step],
        cells=segment.cells[following],
        aux=segment.aux[following],
    )
    return control


def prefix_of(segment: Segment) -> Prefix:
    """Return a one-row prefill of every decision of ``segment``.

    Args:
      segment: The recorded decisions and their frames.

    Returns:
      prefix: One row of them, starting an episode where the segment does.

    """
    return Prefix(
        cells=segment.cells[None],
        aux=segment.aux[None],
        actions=segment.actions[None],
        starts_episode=segment.starts_episode,
    )


def no_done(engine: Engine) -> Control:
    """Return a free control that forces ``done = false`` so episodes never end.

    Args:
      engine: The engine the control is for.

    Returns:
      control: Every row active and free, but for its act job's done slot.

    """
    control = engine.control()
    control.job_forced[:, 1] = True
    control.job_tokens[:] = engine.job_tokens(
        reward=torch.tensor(0),
        done=torch.tensor(data=False),
        cells=torch.zeros(engine.model.schema.cell_slots, 8, dtype=torch.long),
        aux=torch.zeros(len(engine.model.schema.scalar_ranges), dtype=torch.long),
    )
    return control


class Flags(Protocol):
    """Parsed command-line flags."""

    checkpoint: Path
    corpus: Path
    output: Path
    experiment: str
    override: list[str]
    t_max: int
    decisions: int
    tolerance: float
    draws: int
    device: str


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("checkpoint", type=Path, help="TrainLoop checkpoint (.pt).")
    parser.add_argument("corpus", type=Path, help="Corpus file of the segment.")
    parser.add_argument("output", type=Path, help="Report JSON.")
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
        "--t-max",
        type=int,
        default=0,
        help="Engine context in global positions; default the run's window.",
    )
    parser.add_argument(
        "--decisions",
        type=int,
        default=48,
        help="Decisions of the real segment; default 48.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2e-3,
        help="Largest float32 log-probability difference that passes; default 2e-3.",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=50_000,
        help="Gumbel-max draws from the action head; default 50,000.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")


def _undone_decision(engine: Engine) -> tuple[Tensor, Tensor, Tensor]:
    """Take one free decision that cannot end the episode, then restore the engine."""
    saved = {
        f.name: cast("Tensor", getattr(engine.state, f.name)).clone()
        for f in dataclasses.fields(engine.state)
    }
    decision = engine.decide(no_done(engine))
    out = (
        decision.action.clone(),
        decision.job.tokens.clone(),
        decision.job.logp.clone(),
    )
    for name, value in saved.items():
        cast("Tensor", getattr(engine.state, name)).copy_(value)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
