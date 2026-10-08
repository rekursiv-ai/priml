#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Score a world-model checkpoint's fidelity on fixed validation data, on one GPU.

A fast accuracy suite, so tuning optimizes more than validation bpb. The model
is rebuilt from the experiment factory that trained it, with the --override
values the run was launched with, and every measurement reads the validation
split of that experiment's corpus, a replay shard's frames replayed:

1. Teacher-forced NLL per modality on fixed spans: SPANS validation episodes
   drawn in proportion to their decisions, TARGETS decisions scored in each,
   after as much real history as the model's window holds. The targets depend
   only on the corpus and the seed, never on the window or the GPU count, so
   8K and 16K models are scored on the same decisions.
2. Open-loop dreams: ROWS new worlds of DECISIONS decisions under the model's
   own actions. Rule violations per 1,000 decisions: HUD values over their
   maxima, floor changes without or skipping a ladder, invalid cells, illegal
   actions, inconsistent moves, rewards over the episode's budget, health
   rises no action allows, and living frames at zero health. The same counts
   on REAL_EPISODES real episodes from their start validate every check: their
   rates must be about zero.
3. Continuations: ROWS real prefixes of PREFIX decisions continued for
   CONTINUATION decisions under the recorded actions. Cell mismatch at 1, 8,
   32, and 128 decisions against a frozen-frame baseline, time to the first
   divergence, and the food, drink, energy, and health dynamics against the
   real continuation. Step k compares the windows whose real continuation
   holds frame k, a set fixed by the corpus and the seed, for the model and
   the frozen frame alike; a continuation that ended earlier scores the worst
   value at every frame it lacks. Health losses and deaths that follow a
   need at zero are counted for the model and the real continuations.

OUTPUT, a new file, receives one JSON document with every number and its 95%
bootstrap interval; with --wandb the same numbers go to W&B run fidelity-TAG,
group fidelity. On CUDA the model scores teacher-forced with the kernels and
autocast it trained under (scoring.py), and dreams in bfloat16 with CUDA
graphs.

Examples:
  priml/baselines/craftax/world_model/scripts/fidelity.py /opt/scratch/runs/craftax-world-model/exp014/checkpoints/step_00006103.pt --experiment priml.baselines.craftax.world_model.experiments.exp014 --output /opt/scratch/artifacts/craftax/world-model/fidelity/exp014.json

'''
# fmt: on

from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Final, Protocol, cast

import argparse
import dataclasses
import functools
import hashlib
import itertools
import json
import math
import os
import socket
import sys
import time

from torch import Tensor
from torch.nn import functional

import torch

from priml.baselines.craftax.world_model.archive import (
    Episode,
    read_corpus,
    read_summaries,
)
from priml.baselines.craftax.world_model.batch import (
    PackedBatch,
    Segment,
    pack_windows,
)
from priml.baselines.craftax.world_model.capture.seeds import VALIDATION
from priml.baselines.craftax.world_model.dream import Rollout
from priml.baselines.craftax.world_model.engine import Engine
from priml.baselines.craftax.world_model.index import FLOOR_AUX
from priml.baselines.craftax.world_model.metric import (
    HEAD_BYTES,
    MODALITIES,
    SCALAR_BYTES,
)
from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.schema import (
    FrameSchema,
    craftax_schema,
    number_id,
)
from priml.baselines.craftax.world_model.scoring import (
    autocast,
    load_trained,
)
from priml.baselines.craftax.world_model.scripts.dream_eval import (
    CHECKS,
    Source,
    Step,
    Window,
    choose_windows,
    divergence,
    episode_features,
    episodes_of,
    ratio_interval,
    read_source,
    rollout,
    sampling_engine,
    truncate,
)
from priml.lib.codec import PlainTree, from_plain, to_plain
from priml.paths import validated_output_path
from priml.train.tracker import WandbTracker


SCHEMA: Final = "craftax-world-model-fidelity/v1"
"""Schema name of the report."""

FIDELITY_CHECKS: Final = (
    "floor_change_invalid",
    "reward_over_budget",
    "health_rise_over_bound",
    "alive_at_zero_health",
)
"""Rules checked here beside ``dream_eval.CHECKS``; ``counts`` defines them."""

DYNAMICS: Final = (
    *("food_down", "drink_down", "energy_down"),
    *("health_down", "health_down_at_zero_need"),
    *("deaths", "deaths_at_zero_need", "reward"),
)
"""Events counted beside the violations: need and health decrements, the health
decrements and deaths that follow a need at zero, deaths, and positive reward."""

COUNTS: Final = (*CHECKS, *FIDELITY_CHECKS, *DYNAMICS)
"""Every entry of ``counts``, reported per 1,000 decisions."""

HEADLINE: Final = (
    "over_maximum",
    "floor_change_without_ladder",
    "floor_change_invalid",
    "invalid_frame",
    "invalid_cell",
    "illegal_action",
    "move_inconsistent",
    "reward_over_budget",
    "health_rise_over_bound",
    "alive_at_zero_health",
)
"""The rule violations the summary reports for dreams and for real episodes."""

HUD_DYNAMICS: Final = ("food", "drink", "energy", "health")
"""Auxiliary fields whose continuation dynamics are compared with the real ones."""

SPAN_COLUMNS: Final = (
    *MODALITIES,
    "decisions",
    "frames",
    "cells",
    "model_correct",
    "copy_correct",
)
"""Leading columns of ``score_span``; one NLL column per auxiliary field follows."""

REWARD_BUDGET: Final = 234
"""Most positive reward an episode can earn: each of the 67 achievements once,
226 points, plus 8 armour levels, which only increase."""

REGENERATION: Final = 20
"""Health tokens one game tick can restore: one HP on the 0.05-HP grid."""

RED_POTION: Final = 180
"""Health tokens a red potion's tick can restore: its 8 HP plus regeneration."""

CONTINUATION_CHECKPOINTS: Final = (1, 8, 32, 128, 256, 512, 1_024)
"""Decisions after a prefix at which continuations are compared."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Settings:
    """How much of each measurement to run; the defaults fit 15 minutes on an H200.

    Attributes:
      spans: Teacher-forced spans.
      targets: Target decisions per span.
      rows: Engine rows: dreams, and continuation windows.
      decisions: Decisions per dream.
      real_episodes: Real episodes whose first ``decisions`` validate the checks.
      prefix: Real decisions prefilled before each continuation.
      continuation: Decisions continued under the recorded actions.
      seed: Seed of every choice of data, of sampling, and of the intervals.
      resamples: Bootstrap resamples per interval.

    """

    spans: int = 128
    targets: int = 1_024
    rows: int = 64
    decisions: int = 1_000
    real_episodes: int = 128
    prefix: int = 512
    continuation: int = 256
    seed: int = 0
    resamples: int = 1_000


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Span:
    """Target decisions of one validation episode, scored teacher-forced.

    Attributes:
      source: The episode.
      start: First target decision.
      targets: Target decisions, all inside the episode.

    """

    source: Source
    start: int
    targets: int


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once the report is written and logged.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    output = validated_output_path(flags.output)
    if output.exists():
        raise FileExistsError(f"Output {output} exists; choose a new file.")
    tag = flags.tag or f"{flags.checkpoint.parent.parent.name}-{flags.checkpoint.stem}"
    device = torch.device(flags.device)
    clock = time.monotonic()
    model, config = load_trained(
        flags.experiment,
        flags.checkpoint,
        overrides=flags.override,
        device=device,
    )
    corpus = Path(config.dataset.corpus)
    sources = validation_sources(corpus)
    provenance: dict[str, PlainTree] = {
        "tag": tag,
        "checkpoint": str(flags.checkpoint),
        "checkpoint_sha256": _sha256(flags.checkpoint),
        "experiment": flags.experiment,
        "overrides": list(flags.override),
        "corpus": str(corpus),
        "validation_episodes": len(sources),
        "t_g": config.dataset.t_g,
        "s_max": config.dataset.s_max,
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "host": socket.gethostname(),
        "job": os.environ.get("SLURM_JOB_ID", ""),
        "command": list(sys.argv),
        "load_seconds": time.monotonic() - clock,
    }
    settings = Settings(
        **{
            f.name: from_plain(cast("object", getattr(flags, f.name)), int)
            for f in dataclasses.fields(Settings)
        },
    )
    measured = measure(
        model,
        sources=sources,
        t_g=config.dataset.t_g,
        s_max=config.dataset.s_max,
        settings=settings,
        precision=autocast(config, device),
    )
    report: dict[str, PlainTree] = {
        "schema": SCHEMA,
        "provenance": provenance,
        **measured,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n")
    if flags.wandb != "disabled":
        _log_wandb(report, tag=tag, mode=flags.wandb, directory=output.parent)
    summary = from_plain(report["summary"], dict[str, object])
    lines = [f"{key} {value}" for key, value in summary.items()]
    print("\n".join([*lines, f"report {output}"]))
    return 0


def validation_sources(corpus: Path) -> list[Source]:
    """Return every validation episode a corpus names, in corpus order.

    Args:
      corpus: Corpus file naming published shards of both splits.

    Returns:
      sources: Each validation episode's shard, position, and summary.

    """
    return [
        Source(directory=directory, line=line, index=index, summary=summary)
        for directory, line in read_corpus(corpus)
        for index, summary in enumerate(read_summaries(directory, line))
        if summary.receipt.split == VALIDATION
    ]


def measure(
    model: WorldModel,
    *,
    sources: Sequence[Source],
    t_g: int,
    s_max: int,
    settings: Settings,
    precision: AbstractContextManager[object],
) -> dict[str, PlainTree]:
    """Run the three measurements; on CUDA the model is then cast to bfloat16.

    Args:
      model: The trained model, float32, on its device, in evaluation mode.
      sources: Validation episodes.
      t_g: The model's training window in global positions, also the engine's.
      s_max: Episode segments per window.
      settings: How much of each measurement to run.
      precision: The autocast teacher-forced scoring runs under, as training
        scored, or ``nullcontext()``.

    Returns:
      report: ``summary``, ``settings``, ``teacher_forced``, ``dreams``,
        ``continuations``, and ``seconds``, JSON-ready.

    Raises:
      ValueError: If no validation episode is long enough to continue, checked
        before anything is scored.

    """
    windows = choose_windows(
        sources,
        count=settings.rows,
        prefix=settings.prefix,
        decisions=settings.continuation,
        generator=torch.Generator().manual_seed(settings.seed),
    )
    if not windows:
        raise ValueError(
            f"No validation episode is long enough to continue: {settings.prefix} "
            f"+ {settings.continuation} decisions.",
        )
    seconds: dict[str, PlainTree] = {}
    clock = time.monotonic()
    spans = choose_spans(
        sources,
        count=settings.spans,
        targets=settings.targets,
        generator=torch.Generator().manual_seed(settings.seed),
    )
    with precision:
        teacher = teacher_forced(
            model,
            spans,
            t_g=t_g,
            s_max=s_max,
            resamples=settings.resamples,
            generator=torch.Generator().manual_seed(settings.seed),
        )
    seconds["teacher_forced"] = time.monotonic() - clock
    if model.start.device.type == "cuda":
        # A bfloat16 engine scores and dreams as the float32 one does, 5.5 times
        # faster (measured).
        model.to(torch.bfloat16)
    clock = time.monotonic()
    engine, step = sampling_engine(
        model,
        rows=settings.rows,
        t_max=t_g,
        seed=settings.seed,
    )
    dreams = _dreams(engine, step, sources=sources, settings=settings)
    seconds["dreams"] = time.monotonic() - clock
    clock = time.monotonic()
    continuations = _continuations(engine, step, windows, settings=settings)
    seconds["continuations"] = time.monotonic() - clock
    seconds["measured"] = sum(from_plain(value, float) for value in seconds.values())
    report: dict[str, PlainTree] = {
        "settings": to_plain(settings),
        "teacher_forced": teacher,
        "dreams": dreams,
        "continuations": continuations,
        "seconds": seconds,
    }
    checkpoints = [k for k in CONTINUATION_CHECKPOINTS if k <= settings.continuation]
    return {"summary": _summary(report, checkpoints=checkpoints), **report}


def choose_spans(
    sources: Sequence[Source],
    *,
    count: int,
    targets: int,
    generator: torch.Generator,
) -> list[Span]:
    """Draw spans with replacement, each episode in proportion to its decisions.

    The span's first target is then uniform among the starts that keep
    ``targets`` decisions inside the episode; a shorter episode is scored
    whole. So decisions are not equally likely: in an episode longer than
    ``targets``, a decision is covered in proportion to the starts whose span
    holds it, and its first and last decisions (a reset, a death) only by the
    one span that starts or ends there.

    Args:
      sources: Validation episodes.
      count: Spans to draw.
      targets: Target decisions per span.
      generator: Source of every draw.

    Returns:
      spans: The spans, in draw order.

    """
    lengths = torch.tensor([s.summary.decisions for s in sources], dtype=torch.float64)
    picks = torch.multinomial(lengths, count, replacement=True, generator=generator)
    spans: list[Span] = []
    for pick in picks:
        source = sources[int(pick)]
        length = source.summary.decisions
        kept = min(targets, length)
        start = int(torch.randint(length - kept + 1, (), generator=generator))
        spans.append(Span(source=source, start=start, targets=kept))
    return spans


@torch.no_grad()
def teacher_forced(
    model: WorldModel,
    spans: Sequence[Span],
    *,
    t_g: int,
    s_max: int,
    resamples: int,
    generator: torch.Generator,
) -> dict[str, PlainTree]:
    """Score every span; report NLL per modality with intervals over spans.

    Runs under the caller's autocast.

    Args:
      model: The model, on its device.
      spans: Target decisions to score.
      t_g: Global positions per window.
      s_max: Episode segments per window.
      resamples: Bootstrap resamples over spans.
      generator: Source of the resamples.

    Returns:
      nll: ``nats_per_decision`` and ``bpb`` with intervals; per modality its
        nats per decision and bits per byte, bytes counted as ``metric``
        counts them; per auxiliary field its nats per frame; and the
        next-frame cell accuracy of the model's per-field argmax and of
        copying the current frame; ``per_span`` lists each span's episode,
        its length, the floor at the first target, and its nats per decision,
        overall and per modality.

    """
    rows: list[Tensor] = []
    records: list[PlainTree] = []
    ordered = sorted(spans, key=lambda span: span.source.name)
    # Spans of one episode decode it once; a floor-8 timeout holds ~100k frames.
    for _, group in itertools.groupby(ordered, key=lambda span: span.source.name):
        members = list(group)
        episode = read_source(members[0].source)
        for s in members:
            sums = score_span(
                model,
                episode,
                start=s.start,
                targets=s.targets,
                t_g=t_g,
                s_max=s_max,
            )
            rows.append(sums)
            records.append(
                {
                    "episode": s.source.name,
                    "episode_decisions": len(episode.actions),
                    "start": s.start,
                    "targets": s.targets,
                    "floor": int(episode.aux[s.start, FLOOR_AUX]),
                    "nats_per_decision": float(sums[: len(MODALITIES)].sum())
                    / s.targets,
                    "modalities": {
                        name: float(sums[i]) / s.targets
                        for i, name in enumerate(MODALITIES)
                    },
                },
            )
    report = _nll_report(
        torch.stack(rows),
        schema=model.schema,
        resamples=resamples,
        generator=generator,
    )
    return {**report, "per_span": records}


@torch.no_grad()
def score_span(
    model: WorldModel,
    episode: Episode,
    *,
    start: int,
    targets: int,
    t_g: int,
    s_max: int,
) -> Tensor:
    """Sum the model's NLL of ``targets`` decisions after the history its window holds.

    One window holds the target decisions and up to ``(t_g - 1) // 2 - targets``
    decisions before them, from the episode's start when they all fit. A target
    decision scores its action, reward, and done, and its next frame unless the
    decision ended the episode.

    Args:
      model: The model, on its device, under the caller's autocast.
      episode: A complete archived episode.
      start: First target decision.
      targets: Target decisions, all inside the episode.
      t_g: Global positions per window.
      s_max: Episode segments per window.

    Returns:
      sums: float64 ``SPAN_COLUMNS``, then each auxiliary field's summed NLL.

    Raises:
      ValueError: If the targets alone do not fit the window.

    """
    history = min(start, (t_g - 1) // 2 - targets)
    if history < 0:
        raise ValueError(f"{targets} targets do not fit a {t_g}-position window.")
    segment = _cut(episode, first=start - history, stop=start + targets)
    device = model.start.device
    batch = pack_windows([[segment]], t_g=t_g, s_max=s_max).to(device)
    logits = model.logits(batch)
    terms = model.target_terms(batch, logits)
    # One window: a job's flat position is its position in the window, and a
    # start job sits at position 0, before every target.
    first_target = int(segment.starts_episode) + 2 * history
    target = torch.arange(t_g, device=device) >= first_target
    job = target[batch.job_at.long()]
    action, action_scored = terms["action"]
    sums = [action.nll[action_scored & target].double().sum()]
    for name in MODALITIES[1:]:
        value, scored = terms[name]
        keep = scored & job.view(-1, *[1] * (scored.ndim - 1))
        sums.append(torch.where(keep, value.nll, 0.0).double().sum())
    hud, hud_scored = terms["hud"]
    fields = torch.where(hud_scored & job[:, None], hud.nll, 0.0).double().sum(0)
    has_next = job & (batch.job_next >= 0)
    accuracy = _cell_accuracy(model, batch, logits.local, jobs=has_next)
    head = torch.stack([*sums, job.sum(), has_next.sum(), *accuracy]).double()
    return torch.cat([head, fields]).cpu()


def counts(segment: Segment) -> Tensor:
    """Count every entry of ``COUNTS`` in one observed episode.

    ``dream_eval.CHECKS`` come from ``episode_features``. The other checks read
    each transition, a decision that did not end the episode and whose next
    frame is observed:

    - ``floor_change_invalid``: the floor changes other than by one down with
      ``descend`` or one up with ``ascend``;
    - ``reward_over_budget``: a positive reward that lifts the positive return,
      counted from the segment's first decision, above ``REWARD_BUDGET``;
    - ``health_rise_over_bound``: health rises more than one game tick
      allows: ``REGENERATION``, or ``RED_POTION`` when drinking a potion. Sleep
      and rest repeat ticks within one decision, so they have no bound;
    - ``alive_at_zero_health``: the next frame shows zero health. Health at or
      below zero ends an episode, but a float32 health just above zero also
      rounds to zero, which 1,801 whole real validation episodes showed
      0.002-0.016 times per 1,000 decisions.

    Health drops have no bound to check: one tick can land 3 melee hits of up
    to 9 HP and 3 projectiles of up to 10 HP, 1.5 times as hard against the
    boss, beside a potion's 3 HP and starvation's 1, more than the 13 HP a
    player can hold. A drop to zero that does not end the episode is
    ``alive_at_zero_health``.

    Of the ``DYNAMICS``, ``health_down_at_zero_need`` counts health drops from
    a frame showing food, drink, or energy at zero, and
    ``deaths_at_zero_need`` counts deaths whose decision observed one.

    Args:
      segment: The observed decisions and frames, in the Craftax schema.

    Returns:
      counts: float64 ``[len(COUNTS)]``; ``reward`` sums the positive rewards.

    """
    fields = craftax_schema().scalar_names
    aux, actions = segment.aux.long(), segment.actions.long()
    reward, done = segment.reward.long(), segment.done
    needs = [fields.index(name) for name in ("food", "drink", "energy")]
    at_zero = (aux[: len(actions), needs] == 0).any(-1)
    steps = torch.arange(min(len(actions), len(aux) - 1))
    steps = steps[~done[steps]]
    act = actions[steps]
    v = {name: (aux[steps, i], aux[steps + 1, i]) for i, name in enumerate(fields)}
    floor = v["floor"][1] - v["floor"][0]
    # Actions: 6 sleep, 17 rest, 18 descend, 19 ascend, 29-34 potions.
    ladder = ((floor == 1) & (act == 18)) | ((floor == -1) & (act == 19))
    repeats = torch.isin(act, torch.tensor([6, 17]))
    potion = torch.isin(act, torch.arange(29, 35))
    rise = v["health"][1] - v["health"][0]
    earned = reward.clamp(min=0).cumsum(0)
    died = done & (reward == -1)
    extra = [
        ((floor != 0) & ~ladder).sum(),
        ((reward > 0) & (earned > REWARD_BUDGET)).sum(),
        (~repeats & (rise > torch.where(potion, RED_POTION, REGENERATION))).sum(),
        (v["health"][1] == 0).sum(),
        *((v[need][1] < v[need][0]).sum() for need in ("food", "drink", "energy")),
        (rise < 0).sum(),
        ((rise < 0) & at_zero[steps]).sum(),
        died.sum(),
        (died & at_zero).sum(),
        reward.clamp(min=0).sum(),
    ]
    checks = episode_features(segment, horizons=(1,)).checks[:, 0]
    return torch.cat([checks, torch.stack(extra).double()])


def segments(rollout: Rollout, *, row: int) -> list[Segment]:
    """Split one rollout row into its episodes; the last may be cut off.

    Args:
      rollout: A rollout; frame ``d + 1`` of a decision ``d`` that ended an
        episode is the next episode's first frame.
      row: The row.

    Returns:
      segments: Each episode's decisions and frames, in order; a terminal
        decision keeps no frame after it.

    """
    done = rollout.done[row]
    bounds = [0, *(int(i) + 1 for i in done.nonzero()[:, 0])]
    if bounds[-1] < len(done):
        bounds.append(len(done))
    parts: list[Segment] = []
    for first, stop in itertools.pairwise(bounds):
        frames = stop if bool(done[stop - 1]) else stop + 1
        parts.append(
            Segment(
                cells=rollout.cells[row, first:frames],
                aux=rollout.aux[row, first:frames],
                actions=rollout.action[row, first:stop],
                reward=rollout.reward[row, first:stop],
                done=done[first:stop],
                starts_episode=bool(rollout.starts[row, first]),
            ),
        )
    return parts


def rates(
    counted: Tensor,
    decisions: Tensor,
    *,
    names: Sequence[str],
    resamples: int,
    generator: torch.Generator,
) -> dict[str, PlainTree]:
    """Return each count per 1,000 decisions, pooled over units, with its interval.

    Args:
      counted: Counts per unit (row or episode) and name, ``[N, len(names)]``.
      decisions: Decisions per unit, ``[N]``.
      names: Name of each column.
      resamples: Bootstrap resamples over units.
      generator: Source of the resamples.

    Returns:
      rates: Per name, ``ratio_interval``'s entry with ``value``, ``low``, and
        ``high`` per 1,000 decisions.

    """
    return {
        name: _per_thousand(
            ratio_interval(
                counted[:, i],
                decisions,
                resamples=resamples,
                generator=generator,
            )[0],
        )
        for i, name in enumerate(names)
    }


def frozen(segment: Segment) -> Segment:
    """Return ``segment`` with every frame replaced by its first: nothing ever changes.

    Args:
      segment: A real continuation.

    Returns:
      frozen: Its decisions, each observing and producing its first frame, the
        baseline a model must beat.

    """
    frames = len(segment.cells)
    return dataclasses.replace(
        segment,
        cells=segment.cells[:1].expand(frames, -1, -1).clone(),
        aux=segment.aux[:1].expand(frames, -1).clone(),
    )


def hud_dynamics(
    dreamed: Sequence[Segment],
    real: Sequence[Segment],
    *,
    checkpoints: Sequence[int],
    resamples: int,
    generator: torch.Generator,
) -> dict[str, PlainTree]:
    """Compare the ``HUD_DYNAMICS`` fields of continuations with the real ones.

    Each pair shares its frame 0. Errors and first divergences are read at
    every frame the real continuation holds, as ``divergence`` reads boards: a
    continuation that ended earlier misses each later frame, scored as the
    value the field can hold that lies farthest from the real one. Decrements
    and increments are counted over the transitions both hold.

    Args:
      dreamed: Generated continuations, cut at their first ``done``.
      real: The real continuations under the same actions.
      checkpoints: Decisions after frame 0 at which values are compared.
      resamples: Bootstrap resamples over pairs.
      generator: Source of the resamples.

    Returns:
      dynamics: Per field, ``abs_error`` at each checkpoint of the model and of
        the frozen first frame; ``first_divergence``, the median first frame
        at which each differs from the real value or is missing, and the
        share that ever does; and ``decrements_per_1000`` and
        ``increments_per_1000`` decisions of the model, of the real episode,
        and their paired ``difference``.

    """
    schema = craftax_schema()
    common = [min(len(x.aux), len(y.aux)) for x, y in zip(dreamed, real, strict=True)]
    steps = torch.tensor([n - 1 for n in common], dtype=torch.float64)
    mean = functools.partial(_mean, resamples=resamples, generator=generator)
    result: dict[str, PlainTree] = {}
    for name in HUD_DYNAMICS:
        index = schema.scalar_names.index(name)
        low, high = (bound - number_id(0) for bound in schema.scalar_ranges[index])
        pairs = [
            (x.aux[:, index].long(), y.aux[:, index].long())
            for x, y in zip(dreamed, real, strict=True)
        ]
        errors: dict[str, PlainTree] = {}
        for k in checkpoints:
            kept = [(x, y) for x, y in pairs if len(y) > k]
            errors[str(k)] = {
                "model": mean(
                    [
                        (x[k] - y[k]).abs()
                        if len(x) > k
                        else torch.stack([y[k] - low, high - y[k]]).amax()
                        for x, y in kept
                    ],
                ),
                "frozen": mean([(y[0] - y[k]).abs() for _, y in kept]),
            }
        field: dict[str, PlainTree] = {
            "abs_error": errors,
            "first_divergence": {
                "model": _median_first([_first_change(x, y) for x, y in pairs]),
                "frozen": _median_first(
                    [_first_change(y[:1].expand_as(y), y) for _, y in pairs],
                ),
            },
        }
        for kind, sign in (("decrements", -1), ("increments", 1)):
            model, truth = (
                torch.stack(
                    [
                        (sign * values[:n].diff() > 0).sum()
                        for values, n in zip(side, common, strict=True)
                    ],
                )
                for side in ([x for x, _ in pairs], [y for _, y in pairs])
            )
            field[f"{kind}_per_1000"] = rates(
                torch.stack([model, truth, model - truth], dim=-1).double(),
                steps,
                names=("model", "real", "difference"),
                resamples=resamples,
                generator=generator,
            )
        result[name] = field
    return result


def flatten(tree: object, prefix: str = "") -> dict[str, float]:
    """Return every number of a JSON tree under its slash-joined path.

    Lists, strings, None, and booleans are skipped: W&B charts scalars.

    Args:
      tree: A JSON value.
      prefix: Path of ``tree``.

    Returns:
      numbers: Path to value.

    """
    if isinstance(tree, int | float) and not isinstance(tree, bool):
        return {prefix.removesuffix("/"): float(tree)}
    flat: dict[str, float] = {}
    # A list, string, or None coerces to no members.
    if isinstance(tree, dict):
        typed_tree = cast(dict[object, object], tree)
        mapping: dict[str, object] = {
            str(key): value for key, value in typed_tree.items()
        }
        for key, value in from_plain(mapping, dict[str, object]).items():
            flat |= flatten(value, f"{prefix}{key}/")
    return flat


class Flags(Protocol):
    """Parsed command-line flags."""

    checkpoint: Path
    experiment: str
    override: list[str]
    output: Path
    tag: str
    device: str
    wandb: str
    spans: int
    targets: int
    rows: int
    decisions: int
    real_episodes: int
    prefix: int
    continuation: int
    seed: int
    resamples: int


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    defaults = Settings()
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
    parser.add_argument("--output", type=Path, required=True, help="New report JSON.")
    parser.add_argument(
        "--tag",
        default="",
        help="Checkpoint name in the report and W&B; default RUN-STEP of its path.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument(
        "--wandb",
        choices=("online", "offline", "disabled"),
        default="disabled",
        help="W&B mode of run fidelity-TAG; off by default.",
    )
    helps = {
        "spans": "Teacher-forced spans.",
        "targets": "Target decisions per span.",
        "rows": "Engine rows: dreams, and continuation windows.",
        "decisions": "Decisions per dream.",
        "real_episodes": "Real episodes that validate the rule checks.",
        "prefix": "Real decisions prefilled before a continuation.",
        "continuation": "Decisions continued under the recorded actions.",
        "seed": "Seed of every draw.",
        "resamples": "Bootstrap resamples per interval.",
    }
    for name, text in helps.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=int,
            default=getattr(defaults, name),
            help=f"{text} Default %(default)s.",
        )


def _cut(episode: Episode, *, first: int, stop: int) -> Segment:
    """Return decisions ``[first, stop)`` of an episode and the frame after them."""
    frames = stop + 1 if stop < len(episode.actions) else stop
    return Segment(
        cells=episode.cells[first:frames],
        aux=episode.aux[first:frames],
        actions=episode.actions[first:stop],
        reward=episode.reward[first:stop],
        done=episode.done[first:stop],
        starts_episode=first == 0 and not episode.origin,
    )


def _cell_accuracy(
    model: WorldModel,
    batch: PackedBatch,
    local: Tensor,
    *,
    jobs: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return scored cells, and those the per-field argmax and a copy get right."""
    scored = jobs & (batch.job_memory >= 0)
    target = batch.cells[batch.job_next[scored].long()].long()
    current = batch.cells[batch.job_memory[scored].long()].long()
    prefix, cells = len(model.schema.prefix_ranges), model.schema.cell_slots
    board = local[scored, prefix : prefix + cells]
    fields = functional.pad(board, (0, 1), value=float("-inf"))[..., model.cell_index]
    return (
        scored.sum() * cells,
        (fields.argmax(-1) == target).all(-1).sum(),
        (current == target).all(-1).sum(),
    )


def _nll_report(
    sums: Tensor,
    *,
    schema: FrameSchema,
    resamples: int,
    generator: torch.Generator,
) -> dict[str, PlainTree]:
    """Return ``teacher_forced``'s report of per-span sums ``[spans, columns]``."""
    column = {name: sums[:, i] for i, name in enumerate(SPAN_COLUMNS)}
    decisions, frames = column["decisions"], column["frames"]
    size = {name: decisions * width for name, width in HEAD_BYTES.items()}
    size["board"] = frames * schema.cell_slots * len(schema.cell_fields)
    size["hud"] = frames * SCALAR_BYTES * len(schema.scalar_ranges)
    nats = torch.stack([column[name] for name in MODALITIES]).sum(0)
    total = torch.stack([size[name] for name in MODALITIES]).sum(0)
    bits = math.log(2)
    interval = functools.partial(_interval, resamples=resamples, generator=generator)
    hud = sums[:, len(SPAN_COLUMNS) :].sum(0) / frames.sum()
    return {
        "spans": len(sums),
        "decisions": int(decisions.sum()),
        "frames": int(frames.sum()),
        "nats_per_decision": interval(nats, decisions),
        "bpb": interval(nats / bits, total),
        "modalities": {
            name: {
                "nats_per_decision": interval(column[name], decisions),
                "bpb": float(column[name].sum() / (bits * size[name].sum())),
            }
            for name in MODALITIES
        },
        "hud_fields": {
            name: float(value)
            for name, value in zip(schema.scalar_names, hud, strict=True)
        },
        "cell_accuracy": {
            side: interval(column[f"{side}_correct"], column["cells"])
            for side in ("model", "copy")
        },
    }


# ``per_row`` keeps each dream row's counts, so two reports' rates can be compared with
# an interval on their difference.
def _dreams(
    engine: Engine,
    step: Step,
    *,
    sources: Sequence[Source],
    settings: Settings,
) -> dict[str, PlainTree]:
    """Count ``COUNTS`` in free-running dreams and in real episodes from their start."""
    decisions = settings.decisions
    free = _cpu(rollout(engine, step, decisions=decisions))
    dreamed = torch.stack(
        [
            torch.stack([counts(s) for s in segments(free, row=row)]).sum(0)
            for row in range(engine.rows)
        ],
    )
    generator = torch.Generator().manual_seed(settings.seed)
    # Corpus order groups episodes by shard, so its first episodes would all
    # come from one worker.
    order = torch.randperm(len(sources), generator=generator)
    chosen = [sources[int(i)] for i in order[: settings.real_episodes]]
    real = [truncate(read_source(s), decisions=decisions) for s in chosen]
    real_counts = torch.stack([counts(s) for s in real])
    real_decisions = torch.tensor([len(s.actions) for s in real], dtype=torch.float64)
    measured = functools.partial(
        rates,
        names=COUNTS,
        resamples=settings.resamples,
        generator=generator,
    )
    rows = torch.full((engine.rows,), float(decisions), dtype=torch.float64)
    return {
        "rows": engine.rows,
        "decisions": decisions,
        "episodes_started": int(free.starts.sum()),
        "real_episodes": len(real),
        "real_names": [s.name for s in chosen],
        "real_decisions": int(real_decisions.sum()),
        "dream": measured(dreamed, rows),
        "real": measured(real_counts, real_decisions),
        "totals": {
            side: {name: float(v) for name, v in zip(COUNTS, sums, strict=True)}
            for side, sums in (
                ("dream", dreamed.sum(0)),
                ("real", real_counts.sum(0)),
            )
        },
        "per_row": {
            name: [float(v) for v in row]
            for name, row in zip(COUNTS, dreamed.T, strict=True)
        },
    }


def _continuations(
    engine: Engine,
    step: Step,
    windows: Sequence[Window],
    *,
    settings: Settings,
) -> dict[str, PlainTree]:
    """Continue every window under its recorded actions; compare with the real one."""
    decisions, rows = settings.continuation, engine.rows
    # Rows without a window start new worlds; decisions after a window's real
    # end are never compared, so they repeat noop.
    actions = torch.full((rows, decisions), -1, dtype=torch.long)
    for row, w in enumerate(windows):
        actions[row] = 0
        actions[row, : len(w.real.actions)] = w.real.actions.long()
    prefixes = [w.prefix for w in windows] + [None] * (rows - len(windows))
    run = _cpu(
        rollout(engine, step, decisions=decisions, prefixes=prefixes, actions=actions),
    )
    dreamed = episodes_of(run, rows=range(len(windows)))
    truth = [w.real for w in windows]
    checkpoints = [k for k in CONTINUATION_CHECKPOINTS if k <= decisions]
    compare = functools.partial(
        divergence,
        checkpoints=checkpoints,
        resamples=settings.resamples,
        seed=settings.seed,
    )
    model, still = compare(dreamed, truth), compare([frozen(t) for t in truth], truth)
    generator = torch.Generator().manual_seed(settings.seed)
    measured = functools.partial(
        rates,
        names=COUNTS,
        resamples=settings.resamples,
        generator=generator,
    )
    return {
        "windows": [
            {"name": w.name, "kind": w.kind, "real_decisions": len(w.real.actions)}
            for w in windows
        ],
        "prefix": settings.prefix,
        "decisions": decisions,
        "model": model,
        "frozen": still,
        "first_divergence": {
            "model": _first_divergence(model),
            "frozen": _first_divergence(still),
        },
        "hud": hud_dynamics(
            dreamed,
            truth,
            checkpoints=checkpoints,
            resamples=settings.resamples,
            generator=generator,
        ),
        # The real continuations validate the checks mid-episode, on every floor.
        **{
            side: measured(
                torch.stack([counts(s) for s in observed]),
                torch.tensor([len(s.actions) for s in observed], dtype=torch.float64),
            )
            for side, observed in (("violations", dreamed), ("real_violations", truth))
        },
    }


def _first_divergence(result: Mapping[str, PlainTree]) -> dict[str, PlainTree]:
    """Return ``_median_first`` of ``divergence``'s first divergent frames per kind."""
    firsts = from_plain(result["first_divergence"], dict[str, object])
    return {
        kind: _median_first(
            [
                None if v is None else from_plain(v, int)
                for v in from_plain(firsts[kind], list[object])
            ],
        )
        for kind in ("frame", "board", "hud")
    }


def _first_change(a: Tensor, b: Tensor) -> int | None:
    """Return the first index of ``b`` at which ``a`` differs or ends, or None."""
    shown = min(len(a), len(b))
    hits = (a[:shown] != b[:shown]).nonzero()
    if len(hits):
        return int(hits[0, 0])
    return shown if shown < len(b) else None


def _median_first(values: Sequence[int | None]) -> dict[str, PlainTree]:
    """Return the median first divergence, never diverging last, and the share that do."""
    ranked = sorted(math.inf if v is None else v for v in values)
    middle = ranked[(len(ranked) - 1) // 2] if ranked else math.inf
    return {
        "median": None if math.isinf(middle) else int(middle),
        "diverged": sum(v is not None for v in values) / max(len(values), 1),
    }


def _summary(
    report: Mapping[str, PlainTree],
    *,
    checkpoints: Sequence[int],
) -> dict[str, PlainTree]:
    """Return the headline numbers of a report under flat names."""
    tf, cont = ("teacher_forced",), ("continuations",)
    horizon = str(max(k for k in checkpoints if k <= 128))
    paths: dict[str, tuple[str, ...]] = {
        "nll/nats_per_decision": (*tf, "nats_per_decision", "value"),
        "nll/bpb": (*tf, "bpb", "value"),
        **{
            f"nll/{name}": (*tf, "modalities", name, "nats_per_decision", "value")
            for name in MODALITIES
        },
        **{
            f"cell_accuracy/{side}": (*tf, "cell_accuracy", side, "value")
            for side in ("model", "copy")
        },
        **{
            f"{side}/{name}": ("dreams", side, name, "value")
            for side in ("dream", "real")
            for name in HEADLINE
        },
        **{
            f"continuation/board_mismatch/{k}/{side}": (
                *cont,
                side,
                "board_mismatch",
                str(k),
                "value",
            )
            for side in ("model", "frozen")
            for k in checkpoints
        },
        # Both sides count the same windows at each checkpoint.
        **{
            f"continuation/board_mismatch/{k}/n": (
                *cont,
                "model",
                "board_mismatch",
                str(k),
                "n",
            )
            for k in checkpoints
        },
        **{
            f"continuation/ended/{k}": (*cont, "model", "ended", str(k))
            for k in checkpoints
        },
        **{
            f"continuation/{name}/{side}": (*cont, key, name, stat)
            for side, key in (("model", "violations"), ("real", "real_violations"))
            for name, stat in (
                ("health_down", "value"),
                ("health_down_at_zero_need", "value"),
                ("deaths", "numerator"),
                ("deaths_at_zero_need", "numerator"),
            )
        },
        **{
            f"continuation/first_divergence/board/{side}": (
                *cont,
                "first_divergence",
                side,
                "board",
                "median",
            )
            for side in ("model", "frozen")
        },
        **{
            f"continuation/terminals/{side}": (
                *cont,
                "model",
                "done",
                f"{side}_terminals",
            )
            for side in ("dream", "real")
        },
        **{
            f"continuation/{name}/{'_'.join(path)}/{side}": (
                *cont,
                "hud",
                name,
                *path,
                side,
                key,
            )
            for name in HUD_DYNAMICS
            for path, key, sides in (
                (("decrements_per_1000",), "value", ("model", "real")),
                (("increments_per_1000",), "value", ("model", "real")),
                (("first_divergence",), "median", ("model", "frozen")),
                (("abs_error", horizon), "value", ("model", "frozen")),
            )
            for side in sides
        },
        "seconds/measured": ("seconds", "measured"),
    }
    return {name: _at(report, *path) for name, path in paths.items()}


def _at(tree: object, *keys: str) -> float | None:
    """Return the number at a path of nested objects, None where one is absent."""
    node: object = tree
    for key in keys:
        if node is None:
            return None
        node = from_plain(node, dict[str, object])[key]
    return None if node is None else from_plain(node, float)


def _interval(
    num: Tensor,
    den: Tensor,
    *,
    resamples: int,
    generator: torch.Generator,
) -> PlainTree:
    """Return ``ratio_interval``'s entry alone."""
    return ratio_interval(num, den, resamples=resamples, generator=generator)[0]


def _per_thousand(entry: dict[str, PlainTree]) -> dict[str, PlainTree]:
    """Return a rate entry with its estimate and interval scaled to 1,000 units."""
    scaled = {
        key: None if entry[key] is None else 1_000 * from_plain(entry[key], float)
        for key in ("value", "low", "high")
    }
    return entry | scaled


def _mean(
    values: Sequence[Tensor],
    *,
    resamples: int,
    generator: torch.Generator,
) -> PlainTree:
    """Return the mean of one value per unit with its bootstrap interval."""
    num = torch.stack(list(values)).double() if values else torch.zeros(0)
    return _interval(
        num,
        torch.ones_like(num),
        resamples=resamples,
        generator=generator,
    )


def _cpu(rollout: Rollout) -> Rollout:
    """Return ``rollout`` with every tensor on the CPU."""
    return Rollout(
        **{
            f.name: cast("Tensor", getattr(rollout, f.name)).cpu()
            for f in dataclasses.fields(rollout)
        },
    )


def _log_wandb(
    report: Mapping[str, PlainTree],
    *,
    tag: str,
    mode: str,
    directory: Path,
) -> None:
    """Log every number of the report to W&B run ``fidelity-TAG``, group fidelity."""
    config = WandbTracker.Config(
        project="craftax-world-model",
        name=f"fidelity-{tag}",
        group="fidelity",
        mode=mode,
        working_dir=str(directory / "wandb"),
        capture_console=False,
        run_config=from_plain(report["provenance"], dict[str, object]),
    )
    tracker = config.make()
    tracker.log_metrics(flatten(report), step=0)
    tracker.close()


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 hex digest."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
