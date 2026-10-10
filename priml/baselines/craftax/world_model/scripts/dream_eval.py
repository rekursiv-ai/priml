#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Generate episodes from a trained world model and compare them with real validation episodes.

Three measurements, all on one engine of ROWS rows at temperature 1, CUDA-graphed
on a GPU (GraphedStep):

1. Free-running dreams: every row starts a new world and plays DECISIONS decisions
   with the model's own action head. Each row's first episode is compared with as
   many real validation episodes of the same arm, truncated to the same decision
   count, so both are right-censored at the same horizon.
2. Prefix continuations: ROWS / 2 held-out validation windows (a quarter placed so
   the real death falls inside the continuation) are prefilled with PREFIX real
   decisions and continued open-loop for CONTINUATION decisions, with the model's
   actions, with the recorded actions (twice, to separate sampling noise from
   error), and fully teacher-forced (the likelihood sanity check).
3. Statistics with bootstrap confidence intervals over episodes and, for
   distributions, a null from permuting episode labels whose normal
   approximation gives the p-value; every p-value is Holm-corrected together.

Outputs under OUTPUT: stats.json, episodes.json (decoded frames of a dozen
episodes), and viewer model bundles under bundles/ for viewer/games.mjs build.

Examples:
  priml/baselines/craftax/world_model/scripts/dream_eval.py /opt/scratch/runs/craftax-world-model/exp001/checkpoints/step_00001525.pt --archive /opt/scratch/datasets/craftax/world-model/archive-v1 --output /opt/scratch/artifacts/craftax/world-model/dream-eval

'''
# fmt: on

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, Protocol, cast

import argparse
import base64
import collections
import dataclasses
import functools
import hashlib
import json
import math
import socket
import time

from torch import Tensor

import torch

from priml.baselines.craftax.game.state import (
    ATN_DIM,
    Action,
    BlockType,
    ItemType,
)
from priml.baselines.craftax.world_model.archive import (
    Episode,
    EpisodeSummary,
    ManifestLine,
    read_manifest,
    read_summaries,
)
from priml.baselines.craftax.world_model.batch import Segment
from priml.baselines.craftax.world_model.checkpoint import (
    load_world_model,
)
from priml.baselines.craftax.world_model.dream import (
    Rollout,
    invalid_cells,
)
from priml.baselines.craftax.world_model.engine import (
    Control,
    Decision,
    Engine,
    GraphedStep,
    Outcome,
    Prefix,
    StartResult,
)
from priml.baselines.craftax.world_model.experiments import (
    WorldModelLoop,
)
from priml.baselines.craftax.world_model.index import FLOOR_AUX, FLOORS
from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.rules import legal_actions
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.session import Mark
from priml.baselines.craftax.world_model.snapshots import replay_episodes
from priml.baselines.craftax.world_model.viewer.bundle import (
    stream_of,
    write_bundle,
)
from priml.lib.codec import PlainTree, from_plain
from priml.math.stats import holm, total_variation
from priml.paths import validated_output_path


type Step = Callable[[Control], tuple[StartResult, Decision]]

SCHEMA: Final = "craftax-dream-eval/v1"
"""Schema name of ``stats.json``."""

FIELDS: Final = craftax_schema().scalar_names
"""Auxiliary field names in frame order."""

ACTIONS: Final = ATN_DIM
"""The game's actions."""

AUX_VALUES: Final = 261
"""Auxiliary token values 0..260 counted per field."""

HEALTH_DELTAS: Final = 521
"""Health changes -260..260 in 0.05-HP steps."""

EVENTS: Final = (
    *("collect_wood", "collect_stone", "collect_coal", "collect_iron"),
    *("collect_diamond", "collect_sapphire", "collect_ruby", "collect_sapling"),
    *("eat", "drink", "take_damage", "gain_arrows", "gain_torches", "place_torch"),
    *("shoot_arrow", "drink_potion", "read_book", "sleep", "gain_xp"),
    *("wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe"),
    *("wood_sword", "stone_sword", "iron_sword", "diamond_sword", "find_bow"),
    *("iron_armour", "diamond_armour", "enchant_sword", "enchant_bow"),
    *("enchant_armour", "learn_fireball", "learn_iceball"),
    *("level_up_dexterity", "level_up_strength", "level_up_intelligence"),
)
"""Frame-derived achievements: what the auxiliary fields show happened."""

CHECKS: Final = (
    *("invalid_frame", "invalid_cell", "illegal_action", "over_maximum"),
    *("move_inconsistent", "facing_wrong", "terminal_without_death"),
    *("floor_change_without_ladder", "xp_without_descend", "pickaxe_without_craft"),
    *("sword_without_craft", "armour_without_craft", "attribute_without_level_up"),
    *("spell_without_book", "potion_without_drink", "material_without_do"),
    "inventory_change_on_move",
)
"""Validity checks, each a count of violations over a count of opportunities."""

ACTION_NAMES: Final = tuple(action.name.lower() for action in Action)
"""The game's action names, by action."""

BLOCK_NAMES: Final = tuple(block.name.lower() for block in BlockType)
"""The game's block names, by block value."""

ITEM_NAMES: Final = ("unseen", *(item.name.lower() for item in ItemType))
"""Item names by cell value: the observation writes ``item + 1``, 0 where unseen."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Teacher:
    """Real act jobs to force in chosen rows of a rollout.

    Attributes:
      rows: Rows whose every act job is forced, bool ``[B]``.
      outcome: Each decision's reward, done, and next frame, ``[B, D, ...]``.

    """

    rows: Tensor
    outcome: Outcome


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Source:
    """One archived episode: where it is, its summary line, and how to read it.

    Attributes:
      directory: The worker directory of its shard.
      line: The shard's manifest line.
      index: Its position in the shard.
      summary: Its ``.meta.jsonl`` line.

    """

    directory: Path
    line: ManifestLine
    index: int
    summary: EpisodeSummary

    @property
    def name(self) -> str:
        """Return ``split/arm/worker/shard#index``."""
        parts = self.directory.parts[-3:]
        return f"{'/'.join(parts)}/{self.line.shard}#{self.index}"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Window:
    """A real prefix and the real decisions that followed it.

    Attributes:
      name: The episode's ``Source.name`` and the anchor decision.
      kind: ``uniform`` or ``pre_death``.
      prefix: One row of real decisions to prefill.
      real: The real continuation; frame 0 is the prefix's last frame.

    """

    name: str
    kind: str
    prefix: Prefix
    real: Segment


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Features:
    """Statistics of one observed episode; ``compare`` compares samples of them.

    Attributes:
      length: Decisions observed.
      died: Whether the last observed decision ended the episode with reward -1.
      reward: Reward summed over the observed decisions.
      floor_first: Decision at which each floor is first shown, -1 if never, long [9].
      returns: Reward summed over the first ``h`` decisions per horizon, float64 [H].
      actions: Decisions per floor and action, float64 ``[9, 43]``.
      aux: Frames per auxiliary field and value, float64 ``[51, 261]``.
      events: Decision at which each of ``EVENTS`` first shows, -1 if never, long.
      checks: Violations and opportunities of each of ``CHECKS``, float64 ``[C, 2]``.
      health_delta: Transitions per health change -260..260, float64 ``[521]``.

    """

    length: int
    died: bool
    reward: float
    floor_first: Tensor
    returns: Tensor
    actions: Tensor
    aux: Tensor
    events: Tensor
    checks: Tensor
    health_delta: Tensor


def main() -> int:
    """Run the program; return the process exit code.

    Returns:
      status: 0 once every output is written.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args())
    output = validated_output_path(flags.output)
    if output.exists():
        raise FileExistsError(f"Output {output} exists; choose a new directory.")
    device = torch.device(flags.device)
    dtypes = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = flags.dtype or ("bfloat16" if device.type == "cuda" else "float32")
    clock = time.monotonic()
    # The engine decodes with its own SDPA, so the experiment's attention
    # kernels, which ``load_world_model`` builds as SDPA, are never called.
    model, config = load_world_model(
        flags.experiment,
        flags.checkpoint,
        overrides=flags.override,
    )
    model = model.to(device=device, dtype=dtypes[dtype])
    provenance: dict[str, PlainTree] = {
        "checkpoint": str(flags.checkpoint),
        "checkpoint_sha256": _sha256(flags.checkpoint),
        "experiment": flags.experiment,
        "overrides": list(flags.override),
        "dtype": dtype,
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "host": socket.gethostname(),
        "load_seconds": time.monotonic() - clock,
    }
    assert isinstance(config, WorldModelLoop.Config)
    stats = evaluate(
        model,
        archive=flags.archive,
        output=output,
        rows=flags.rows,
        t_max=flags.t_max or config.dataset.t_g,
        decisions=flags.decisions,
        prefix=flags.prefix,
        continuation=flags.continuation,
        horizons=flags.horizons,
        seed=flags.seed,
        resamples=flags.resamples,
        permutations=flags.permutations,
        arm=flags.arm,
        provenance=provenance,
    )
    comparison = from_plain(stats["comparison"], dict[str, object])
    print(f"Distinguishable from real: {comparison['distinguishable']}")
    return 0


@torch.no_grad()
def rollout(
    engine: Engine,
    step: Step,
    *,
    decisions: int,
    prefixes: Sequence[Prefix | None] = (),
    actions: Tensor | None = None,
    teacher: Teacher | None = None,
) -> Rollout:
    """Generate ``decisions`` decisions per row through ``step``, as ``dream`` does.

    ``step`` runs a row's start job before its decision, the order
    ``GraphedStep`` captures, so the frame a decision observes is recorded before
    the step and replaced where the step began an episode. With ``Engine.step`` it
    draws the same noise as ``dream`` and returns the same rollout.

    Args:
      engine: The engine; every row is restarted.
      step: ``Engine.step`` or a ``GraphedStep`` of ``engine``.
      decisions: Decisions per row.
      prefixes: One entry per row: a one-row ``Prefix``, or None for a new world;
        empty starts every row in a new world.
      actions: Actions ``[B, decisions]`` to force; negative entries are
        sampled from the action head. None samples every action.
      teacher: Rows whose act jobs are forced to real outcomes.

    Returns:
      rollout: Tokens, log-probabilities, and invalid-cell flags.

    Raises:
      ValueError: If ``decisions`` is not positive or ``prefixes`` is neither
        empty nor one per row.

    """
    if decisions < 1:
        raise ValueError(f"decisions={decisions} must be positive.")
    if prefixes and len(prefixes) != engine.rows:
        raise ValueError(f"Got {len(prefixes)} prefixes for {engine.rows} rows.")
    state = engine.state
    engine.reset(torch.ones(engine.rows, dtype=torch.bool))
    for row, prefix in enumerate(prefixes):
        if prefix is not None:
            engine.prefill(torch.tensor([row]), prefix)
    frame_logp = torch.zeros(
        engine.rows,
        engine.model.schema.frame_slots,
        device=state.length.device,
    )
    record: dict[str, list[Tensor]] = collections.defaultdict(list)
    room = 0
    for index in range(decisions):
        if room == 0:
            room = engine.ensure_room()
        control = _control(engine, index, actions=actions, teacher=teacher)
        cells, aux = state.cells.clone(), state.aux.clone()
        begun, decision = step(control)
        started = begun.started.clone()
        first = engine.outcome(begun.job.tokens)
        record["cells"].append(
            torch.where(started[:, None, None], first.cells, cells).to(torch.uint8),
        )
        record["aux"].append(
            torch.where(started[:, None], first.aux, aux).to(torch.int16),
        )
        record["frame_logp"].append(
            torch.where(started[:, None], begun.job.logp[:, 2:], frame_logp),
        )
        record["starts"].append(started)
        outcome = engine.outcome(decision.job.tokens)
        record["action"].append(decision.action.to(torch.uint8))
        record["reward"].append(outcome.reward.to(torch.int16))
        record["done"].append(outcome.done)
        record["action_logp"].append(decision.action_logp.clone())
        record["reward_logp"].append(decision.job.logp[:, 0].clone())
        record["done_logp"].append(decision.job.logp[:, 1].clone())
        frame_logp = decision.job.logp[:, 2:].clone()
        room -= 1
    control = engine.control()
    begun = (
        engine.start(control)
        if engine.starting(control)
        else engine.skip_start(control)
    )
    record["cells"].append(state.cells.to(torch.uint8))
    record["aux"].append(state.aux.to(torch.int16))
    record["frame_logp"].append(
        torch.where(begun.started[:, None], begun.job.logp[:, 2:], frame_logp),
    )
    record["starts"].append(begun.started.clone())
    stacked = {name: torch.stack(values, dim=1) for name, values in record.items()}
    return Rollout(
        invalid=invalid_cells(stacked["cells"], schema=engine.model.schema),
        **stacked,
    )


def val_sources(archive: Path, *, arm: int) -> list[Source]:
    """Return every published validation episode of one arm, in shard order.

    Args:
      archive: Archive root holding ``val/arm{arm}/w*/``.
      arm: Behaviour arm.

    Returns:
      sources: Each episode's shard, position, and summary.

    """
    return [
        Source(directory=directory, line=line, index=index, summary=summary)
        for directory in sorted((archive / "val" / f"arm{arm}").glob("w*"))
        for line in read_manifest(directory)
        for index, summary in enumerate(read_summaries(directory, line))
    ]


def read_source(source: Source) -> Episode:
    """Read one archived episode, replaying its frames from a replay shard.

    Args:
      source: The episode.

    Returns:
      episode: The episode with every frame.

    """
    (episode,) = replay_episodes(
        source.directory,
        source.line,
        summaries=[source.summary],
    )
    return episode


def truncate(episode: Episode, *, decisions: int) -> Segment:
    """Return an episode's first ``decisions`` decisions and the frame after them.

    Args:
      episode: A complete episode.
      decisions: Decisions to keep.

    Returns:
      segment: Its first decisions; the frame after the last one is kept unless
        that decision ended the episode.

    """
    kept = min(decisions, len(episode.actions))
    frames = kept + 1 if kept < len(episode.actions) else kept
    return Segment(
        cells=episode.cells[:frames].clone(),
        aux=episode.aux[:frames].clone(),
        actions=episode.actions[:kept].clone(),
        reward=episode.reward[:kept].clone(),
        done=episode.done[:kept].clone(),
        starts_episode=not episode.origin,
    )


def window(
    episode: Episode,
    *,
    anchor: int,
    prefix: int,
    decisions: int,
    kind: str,
    name: str,
) -> Window:
    """Cut a real prefix and up to ``decisions`` real decisions after it.

    Args:
      episode: A complete episode.
      anchor: First decision of the prefix.
      prefix: Decisions to prefill.
      decisions: Decisions of the continuation, fewer if the episode ends.
      kind: Why this window was chosen.
      name: The episode's name.

    Returns:
      window: The prefix and the real continuation.

    """
    start = anchor + prefix
    end = min(start + decisions, len(episode.actions))
    frames = end + 1 if end < len(episode.actions) else end
    return Window(
        name=f"{name}@{anchor}",
        kind=kind,
        prefix=Prefix(
            cells=episode.cells[None, anchor : start + 1].clone(),
            aux=episode.aux[None, anchor : start + 1].clone(),
            actions=episode.actions[None, anchor:start].clone(),
            starts_episode=anchor == 0 and not episode.origin,
        ),
        real=Segment(
            cells=episode.cells[start:frames].clone(),
            aux=episode.aux[start:frames].clone(),
            actions=episode.actions[start:end].clone(),
            reward=episode.reward[start:end].clone(),
            done=episode.done[start:end].clone(),
            starts_episode=False,
        ),
    )


def choose_windows(
    sources: Sequence[Source],
    *,
    count: int,
    prefix: int,
    decisions: int,
    generator: torch.Generator,
) -> list[Window]:
    """Choose windows: a quarter end in the real death, the rest are uniform.

    A uniform window starts uniformly where ``prefix + decisions`` decisions and
    the frame after them fit. A pre-death window is cut from an episode that
    died, placing the death uniformly among continuation decisions
    ``decisions // 16`` onward.

    Args:
      sources: Validation episodes to choose from, without replacement.
      count: Windows to return.
      prefix: Prefilled decisions.
      decisions: Continuation decisions.
      generator: Source of every choice.

    Returns:
      windows: Pre-death windows first, then uniform ones.

    """
    deaths = [
        s
        for s in sources
        if from_plain(s.summary.summary.get("death"), int, default=0)
        and s.summary.decisions > prefix + decisions // 16
    ]
    long = [s for s in sources if s.summary.decisions > prefix + decisions]
    chosen: list[Window] = []
    for pool, kind, wanted in (
        (deaths, "pre_death", count // 4),
        (long, "uniform", count - count // 4),
    ):
        taken = _names(chosen)
        picked = [s for s in _shuffled(pool, generator) if s.name not in taken]
        for source in picked[:wanted]:
            length = source.summary.decisions
            if kind == "uniform":
                anchor = _uniform(0, length - prefix - decisions - 1, generator)
            else:
                offset = _uniform(
                    decisions // 16,
                    min(decisions, length - prefix) - 1,
                    generator,
                )
                anchor = length - 1 - prefix - offset
            chosen.append(
                window(
                    read_source(source),
                    anchor=anchor,
                    prefix=prefix,
                    decisions=decisions,
                    kind=kind,
                    name=source.name,
                ),
            )
    return chosen


def episodes_of(
    rollout: Rollout,
    *,
    rows: Sequence[int] | None = None,
) -> list[Segment]:
    """Return each row's first episode: every decision up to its first ``done``.

    Args:
      rollout: A rollout; a prefixed row's frame 0 is real.
      rows: Rows to take; None takes every row.

    Returns:
      segments: One per row; the frame after the last decision is kept unless
        that decision ended the episode.

    """
    decisions = rollout.done.shape[1]
    segments: list[Segment] = []
    for row in range(len(rollout.done)) if rows is None else rows:
        ends = rollout.done[row].nonzero()
        kept = int(ends[0, 0]) + 1 if len(ends) else decisions
        frames = kept if len(ends) else kept + 1
        segments.append(
            Segment(
                cells=rollout.cells[row, :frames],
                aux=rollout.aux[row, :frames],
                actions=rollout.action[row, :kept],
                reward=rollout.reward[row, :kept],
                done=rollout.done[row, :kept],
                starts_episode=bool(rollout.starts[row, 0]),
            ),
        )
    return segments


def episode_features(segment: Segment, *, horizons: Sequence[int]) -> Features:
    """Measure one observed episode.

    A transition is a decision that did not end the episode and whose next
    frame is in the segment; the frame-derived achievements, the movement and
    facing checks, and the HUD rules read transitions.

    Args:
      segment: The observed decisions and frames.
      horizons: Decision counts to sum rewards over.

    Returns:
      features: Its statistics.

    """
    cells, aux = segment.cells.long(), segment.aux.long()
    actions, reward, done = segment.actions.long(), segment.reward.long(), segment.done
    decisions, frames = len(actions), len(cells)
    floor = aux[:, FLOOR_AUX]
    shown = _first(floor[:, None] == torch.arange(FLOORS))
    steps = torch.arange(min(decisions, frames - 1))
    steps = steps[~done[steps].bool()]
    delta = aux[steps + 1] - aux[steps]
    ended = decisions > 0 and bool(done[-1])
    died = ended and int(reward[-1]) == -1
    return Features(
        length=decisions,
        died=died,
        reward=float(reward.sum()),
        floor_first=torch.where(shown >= 0, (shown - 1).clamp(min=0), -1),
        returns=torch.stack([reward[:h].sum() for h in horizons]).double(),
        actions=torch.bincount(
            floor[:decisions] * ACTIONS + actions,
            minlength=FLOORS * ACTIONS,
        )
        .view(FLOORS, ACTIONS)
        .double(),
        aux=torch.bincount(
            (torch.arange(len(FIELDS)) * AUX_VALUES + aux).flatten(),
            minlength=len(FIELDS) * AUX_VALUES,
        )
        .view(len(FIELDS), AUX_VALUES)
        .double(),
        events=_events(aux, delta=delta, steps=steps),
        checks=_checks(segment, steps=steps, delta=delta, ended=ended, died=died),
        health_delta=torch.bincount(
            delta[:, FIELDS.index("health")] + HEALTH_DELTAS // 2,
            minlength=HEALTH_DELTAS,
        ).double(),
    )


def compare(
    real: Sequence[Features],
    dream: Sequence[Features],
    *,
    horizons: Sequence[int],
    resamples: int,
    permutations: int,
    seed: int,
) -> dict[str, PlainTree]:
    """Compare real and generated episodes on matched horizons.

    Rates and means carry 95% percentile-bootstrap intervals over episodes and
    a normal-approximation p-value of the difference from the bootstrap
    spread. Distributions carry the total-variation distance and a null of it
    from permuting episode labels; their p-value is the normal approximation
    of that null (``permutation_tv``), the permutation p-value reported
    beside it. Every p-value is Holm-corrected together, and an entry is
    ``distinguishable`` when its corrected p-value is below 0.05.

    Args:
      real: Features of real episodes.
      dream: Features of generated episodes, observed as long as the real ones.
      horizons: Decision counts at which rates are read.
      resamples: Bootstrap resamples.
      permutations: Label permutations.
      seed: Seed of both.

    Returns:
      comparison: JSON-ready statistics; ``distinguishable`` lists the names of
        the entries that differ.

    """
    r, d = _stack(real), _stack(dream)
    tests = _Tests(
        real=r,
        dream=d,
        resamples=resamples,
        permutations=permutations,
        generator=torch.Generator().manual_seed(seed),
    )
    result: dict[str, PlainTree] = {
        "episodes": {"real": len(real), "dream": len(dream)},
        "decisions": {
            "real": float(r["length"].sum()),
            "dream": float(d["length"].sum()),
        },
        "horizons": list(horizons),
        "length_quantiles": {
            "real": _quantiles(r["length"]),
            "dream": _quantiles(d["length"]),
        },
        "died_by": {
            str(h): tests.rate(
                f"died_by/{h}",
                lambda s, h=h: s["died"] * (s["length"] <= h),
            )
            for h in horizons
        },
        "reach_by": {
            f"floor{k}": {
                str(h): tests.rate(
                    f"reach_by/floor{k}/{h}",
                    lambda s, k=k, h=h: _hit(s["floor_first"][:, k], h),
                )
                for h in horizons
            }
            for k in range(1, FLOORS)
        },
        "return_by": {
            str(h): tests.rate(f"return_by/{h}", lambda s, i=i: s["returns"][:, i])
            for i, h in enumerate(horizons)
        },
        "reward_per_decision": tests.ratio(
            "reward_per_decision",
            lambda s: (s["reward"], s["length"]),
        ),
        "actions": {
            "names": list(ACTION_NAMES),
            "all": tests.spread("actions", lambda s: s["actions"].sum(1)),
            "floors": {
                f"floor{k}": tests.spread(
                    f"actions/floor{k}",
                    lambda s, k=k: s["actions"][:, k],
                )
                for k in range(FLOORS)
                if r["actions"][:, k].sum() > 0 and d["actions"][:, k].sum() > 0
            },
        },
        "aux": {name: tests.aux(index, name) for index, name in enumerate(FIELDS)},
        "events": {
            name: {
                str(h): tests.rate(
                    f"events/{name}/{h}",
                    lambda s, i=i, h=h: _hit(s["events"][:, i], h),
                )
                for h in horizons
            }
            for i, name in enumerate(EVENTS)
        },
        "checks": {
            name: tests.ratio(
                f"checks/{name}",
                lambda s, i=i: (s["checks"][:, i, 0], s["checks"][:, i, 1]),
            )
            for i, name in enumerate(CHECKS)
        },
    }
    delta = tests.spread("health_delta", lambda s: s["health_delta"])
    unseen = (d["health_delta"].sum(0) > 0) & (r["health_delta"].sum(0) == 0)
    delta["dream_only_deltas"] = {
        str(int(i) - HEALTH_DELTAS // 2): float(d["health_delta"].sum(0)[i])
        for i in unseen.nonzero()[:, 0]
    }
    result["health_delta"] = delta
    result["tests"] = len(tests.entries)
    result["distinguishable"] = _holm(tests.entries)
    return result


def permutation_tv(
    real: Tensor,
    dream: Tensor,
    *,
    permutations: int,
    generator: torch.Generator,
) -> dict[str, PlainTree]:
    """Compare two pooled distributions of per-episode counts.

    Args:
      real: Counts per real episode and category, ``[E_real, K]``.
      dream: Counts per generated episode, ``[E_dream, K]``.
      permutations: Random relabelings of episodes.
      generator: Source of the relabelings.

    Returns:
      entry: ``tv`` between the pooled distributions, the relabeled null's
        ``null_mean`` and ``null_std``, ``p_value`` (normal approximation of
        the null, one-sided), ``p_permutation``, both distributions, and counts.

    """
    counts = torch.cat([real, dream]).double()
    episodes, split = len(counts), len(real)
    tv = float(_tv(real.sum(0).double(), dream.sum(0).double()))
    ranks = torch.rand(permutations, episodes, generator=generator).argsort(-1)
    first = torch.zeros(permutations, episodes, dtype=torch.float64)
    first.scatter_(1, ranks[:, :split], 1.0)
    null = _tv(first @ counts, (1 - first) @ counts)
    mean, std = float(null.mean()), float(null.std())
    if std > 0:
        p = 0.5 * math.erfc((tv - mean) / (std * 2**0.5))
    else:
        p = 1.0 if tv <= mean + 1e-12 else 0.0
    return {
        "tv": tv,
        "null_mean": mean,
        "null_std": std,
        "p_value": p,
        "p_permutation": float((1 + (null >= tv - 1e-12).sum()) / (1 + permutations)),
        "count": {"real": float(real.sum()), "dream": float(dream.sum())},
        "n": {"real": len(real), "dream": len(dream)},
        "real": _distribution(real.sum(0)),
        "dream": _distribution(dream.sum(0)),
    }


def ratio_interval(
    num: Tensor,
    den: Tensor,
    *,
    resamples: int,
    generator: torch.Generator,
) -> tuple[dict[str, PlainTree], Tensor]:
    """Return a pooled ratio with its 95% percentile-bootstrap interval over units.

    Args:
      num: Numerator per unit (episode, row, or window), ``[N]``.
      den: Denominator per unit, ``[N]``.
      resamples: Bootstrap resamples of the units.
      generator: Source of the resamples.

    Returns:
      entry: ``value`` (the pooled ``sum(num) / sum(den)``), ``low``, ``high``,
        ``n``, ``numerator``, and ``denominator``; the estimate and interval
        are None when nothing is counted.
      draws: The ratio of each resample, float64 ``[resamples]``, NaN when
        nothing is counted.

    """
    num, den = num.double(), den.double()
    if not len(num) or den.sum() <= 0:
        return {
            "value": None,
            "low": None,
            "high": None,
            "n": len(num),
            "numerator": float(num.sum()),
            "denominator": float(den.sum()),
        }, torch.full((resamples,), math.nan, dtype=torch.float64)
    weights = _resample(len(num), resamples, generator)
    draws = (weights @ num) / (weights @ den)
    low, high = from_plain(
        draws.nanquantile(torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist(),
        list[float],
    )
    return {
        "value": float(num.sum() / den.sum()),
        "low": low,
        "high": high,
        "n": len(num),
        "numerator": float(num.sum()),
        "denominator": float(den.sum()),
    }, draws


def compare_ratio(
    real_num: Tensor,
    real_den: Tensor,
    dream_num: Tensor,
    dream_den: Tensor,
    *,
    resamples: int,
    generator: torch.Generator,
) -> dict[str, PlainTree]:
    """Compare the pooled ratios of two independent samples of units.

    Each sample's units are resampled on their own; the difference's interval
    is the 2.5th-97.5th percentile of the paired resamples' differences.

    Args:
      real_num: Numerator per unit of the first sample, ``[N]``.
      real_den: Denominator per unit of the first sample, ``[N]``.
      dream_num: Numerator per unit of the second sample, ``[M]``.
      dream_den: Denominator per unit of the second sample, ``[M]``.
      resamples: Bootstrap resamples of each sample.
      generator: Source of the resamples.

    Returns:
      entry: ``real`` and ``dream``, each sample's ``ratio_interval`` entry;
        ``diff``, the second ratio minus the first with its ``low`` and
        ``high``; and ``p_value``, a normal approximation from the resamples'
        spread; both None when a sample counts nothing.

    """
    real, real_draws = ratio_interval(
        real_num,
        real_den,
        resamples=resamples,
        generator=generator,
    )
    dream, dream_draws = ratio_interval(
        dream_num,
        dream_den,
        resamples=resamples,
        generator=generator,
    )
    entry: dict[str, PlainTree] = {
        "real": real,
        "dream": dream,
        "diff": None,
        "p_value": None,
    }
    if real_den.sum() <= 0 or dream_den.sum() <= 0:
        return entry
    draws = dream_draws - real_draws
    draws = draws[~draws.isnan()]
    value = float(dream_num.sum() / dream_den.sum() - real_num.sum() / real_den.sum())
    spread = float(draws.std()) if len(draws) > 1 else 0.0
    low, high = from_plain(
        draws.quantile(torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist(),
        list[float],
    )
    entry["diff"] = {"value": value, "low": low, "high": high}
    if value == 0:
        entry["p_value"] = 1.0
    else:
        entry["p_value"] = (
            math.erfc(abs(value) / (spread * 2**0.5)) if spread > 0 else 0.0
        )
    return entry


def sampling_engine(
    model: WorldModel,
    *,
    rows: int,
    t_max: int,
    seed: int,
) -> tuple[Engine, Step]:
    """Return an engine and its step: CUDA-graphed on a GPU, eager on the CPU.

    Args:
      model: The model to decode, on its device and dtype.
      rows: Engine rows.
      t_max: Engine context in global positions.
      seed: Seed of every sample; on a GPU it seeds the device's default
        generator, the only one ``GraphedStep`` accepts.

    Returns:
      engine: The engine.
      step: ``GraphedStep(engine)`` on a GPU, else ``engine.step``.

    """
    device = model.start.device
    if device.type != "cuda":
        engine = Engine(
            model,
            rows=rows,
            t_max=t_max,
            generator=torch.Generator(device).manual_seed(seed),
        )
        return engine, engine.step
    # Not ``torch.cuda.manual_seed``, which seeds the current device's generator.
    generator = torch.cuda.default_generators[device.index or 0].manual_seed(seed)
    engine = Engine(model, rows=rows, t_max=t_max, generator=generator)
    return engine, GraphedStep(engine)


def divergence(
    dreamed: Sequence[Segment],
    real: Sequence[Segment],
    *,
    checkpoints: Sequence[int],
    resamples: int,
    seed: int,
) -> dict[str, PlainTree]:
    """Compare continuations pair by pair from their shared first frame.

    Frame ``k`` is the frame after ``k`` decisions. A pair counts at checkpoint
    ``k`` when its ``real`` segment holds frame ``k``, so which pairs count
    depends on the reference alone. A ``dreamed`` segment that ended earlier
    differs in every cell and field of each frame it lacks: dropping the pair
    instead would score every generator on its own surviving subset, and
    reward it for ending early.

    Args:
      dreamed: Generated continuations, cut at their first ``done``.
      real: The matching real (or other generated) continuations.
      checkpoints: Frame indexes to report.
      resamples: Bootstrap resamples over pairs.
      seed: Bootstrap seed.

    Returns:
      divergence: Per-checkpoint exact-frame, no-divergence, board, and HUD
        mismatch rates, and in ``per_pair`` each pair's value, None where the
        checkpoint does not count it; ``ended``, the pairs counted at each
        checkpoint whose ``dreamed`` segment had ended; each pair's first
        divergent frame; reward, done, and action agreement.

    """
    generator = torch.Generator().manual_seed(seed)
    pairs = [_pair(x, y) for x, y in zip(dreamed, real, strict=True)]
    result: dict[str, PlainTree] = {
        "pairs": len(pairs),
        "checkpoints": list(checkpoints),
    }
    metrics: dict[str, Callable[[dict[str, Tensor], int], Tensor]] = {
        "frame_exact": lambda p, k: ~p["frame_diff"][k - 1],
        "no_divergence": lambda p, k: ~p["frame_diff"][:k].any(),
        "board_mismatch": lambda p, k: p["board_frac"][k - 1],
        "hud_mismatch": lambda p, k: p["hud_diff"][k - 1].double().mean(),
    }
    per_pair: dict[str, PlainTree] = {}
    for name, metric in metrics.items():
        table: dict[str, PlainTree] = {}
        each: dict[str, PlainTree] = {}
        for k in checkpoints:
            held = [
                float(metric(p, k)) if len(p["frame_diff"]) >= k else None
                for p in pairs
            ]
            values = [v for v in held if v is not None]
            table[str(k)] = (
                ratio_interval(
                    torch.tensor(values, dtype=torch.float64),
                    torch.ones(len(values)),
                    resamples=resamples,
                    generator=generator,
                )[0]
                if values
                else None
            )
            each[str(k)] = list(held)
        result[name] = table
        per_pair[name] = each
    result["per_pair"] = per_pair
    result["ended"] = {
        str(k): sum(len(p["frame_diff"]) >= k > int(p["shown"]) for p in pairs)
        for k in checkpoints
    }
    result["hud_field_mismatch"] = {
        str(k): from_plain(torch.stack(rows).double().mean(0).tolist(), list[float])
        if rows
        else None
        for k in checkpoints
        for rows in [[p["hud_diff"][k - 1] for p in pairs if len(p["frame_diff"]) >= k]]
    }
    result["board_field_mismatch"] = {
        str(k): from_plain(torch.stack(rows).mean(0).tolist(), list[float])
        if rows
        else None
        for k in checkpoints
        for rows in [
            [p["field_frac"][k - 1] for p in pairs if len(p["frame_diff"]) >= k],
        ]
    }
    result["first_divergence"] = {
        kind: [_first_true(p[f"{kind}_diff"]) for p in pairs]
        for kind in ("frame", "board", "hud")
    }
    compared = torch.tensor(
        [len(p["reward_equal"]) for p in pairs],
        dtype=torch.float64,
    )
    matched = torch.tensor([float(p["reward_equal"].sum()) for p in pairs])
    nonzero = {
        k: sum(int(p[k]) for p in pairs)
        for k in ("nonzero_dream", "nonzero_real", "matched_nonzero")
    }
    result["reward"] = {
        "compared": int(compared.sum()),
        "agreement": ratio_interval(
            matched,
            compared,
            resamples=resamples,
            generator=generator,
        )[0],
        **nonzero,
        "precision": nonzero["matched_nonzero"] / nonzero["nonzero_dream"]
        if nonzero["nonzero_dream"]
        else None,
        "recall": nonzero["matched_nonzero"] / nonzero["nonzero_real"]
        if nonzero["nonzero_real"]
        else None,
    }
    result["done"] = _done_agreement(pairs)
    agreed = torch.tensor([float(p["action_equal"].sum()) for p in pairs])
    result["actions"] = {
        "agreement": ratio_interval(
            agreed,
            compared,
            resamples=resamples,
            generator=generator,
        )[0],
        "first_disagreement": [_first_true(~p["action_equal"]) for p in pairs],
    }
    return result


def teacher_nll(
    rollout: Rollout,
    real: Sequence[Segment],
    *,
    resamples: int,
    seed: int,
) -> dict[str, PlainTree]:
    """Score teacher-forced rows: the model's NLL of each real decision.

    Decision ``d`` scores its action, reward, and done, and its next frame
    unless it ended the episode. Bytes follow the metric's canonical record,
    so ``bpb`` is comparable with training's ``val/bpb``.

    Args:
      rollout: Rows teacher-forced to ``real``, one per real continuation.
      real: The real continuations.
      resamples: Bootstrap resamples over rows.
      seed: Bootstrap seed.

    Returns:
      nll: ``nats_per_decision`` with its interval, ``bpb``, and each modality's
        nats per decision and bits per byte.

    """
    cells = craftax_schema().cell_slots
    parts: dict[str, list[Tensor]] = collections.defaultdict(list)
    for row, segment in enumerate(real):
        n = len(segment.actions)
        scored = ~rollout.starts[row, 1 : n + 1]
        frames = -rollout.frame_logp[row, 1 : n + 1].double() * scored[:, None]
        parts["action"].append(-rollout.action_logp[row, :n].double().sum())
        parts["reward"].append(-rollout.reward_logp[row, :n].double().sum())
        parts["done"].append(-rollout.done_logp[row, :n].double().sum())
        parts["board"].append(frames[:, :cells].sum())
        parts["hud"].append(frames[:, cells:].sum())
        parts["decisions"].append(torch.tensor(float(n), dtype=torch.float64))
        parts["frames"].append(scored.double().sum())
    sums = {name: torch.stack(values) for name, values in parts.items()}
    decisions = sums.pop("decisions")
    frames = sums.pop("frames")
    total = torch.stack(list(sums.values())).sum(0)
    size = {"action": decisions, "reward": 2 * decisions, "done": decisions}
    size |= {"board": 792 * frames, "hud": 102 * frames}
    generator = torch.Generator().manual_seed(seed)
    bits = math.log(2)
    return {
        "rows": len(real),
        "decisions": int(decisions.sum()),
        "frames": int(frames.sum()),
        "nats_per_decision": ratio_interval(
            total,
            decisions,
            resamples=resamples,
            generator=generator,
        )[0],
        "bpb": float(total.sum() / (bits * sum(s.sum() for s in size.values()))),
        "modalities": {
            name: {
                "nats_per_decision": float(value.sum() / decisions.sum()),
                "bpb": float(value.sum() / (bits * size[name].sum())),
            }
            for name, value in sums.items()
        },
    }


def episode_record(
    segment: Segment,
    *,
    name: str,
    source: str,
    max_frames: int,
) -> dict[str, PlainTree]:
    """Return one episode's decoded frames, JSON-ready, for a page that draws boards.

    Args:
      segment: The episode.
      name: Its label.
      source: ``model`` or ``real``.
      max_frames: Frames kept: the first half and the last half when longer.

    Returns:
      record: Every decision's action, reward, and done, and the kept frames:
        ``frame_index``, ``cells`` (base64 of 792 cell values each, 99 cells
        row-major, 8 fields innermost), and ``aux`` (51 values each).

    """
    frames = len(segment.cells)
    head = min(frames, max_frames // 2 + max_frames % 2)
    tail = min(frames - head, max_frames // 2)
    keep = [*range(head), *range(frames - tail, frames)]
    return {
        "name": name,
        "source": source,
        "starts_episode": segment.starts_episode,
        "frames": frames,
        "decisions": len(segment.actions),
        "frame_index": keep,
        "cells": [
            base64.b64encode(
                segment.cells[i].to(torch.uint8).numpy().tobytes(),
            ).decode()
            for i in keep
        ],
        "aux": [from_plain(segment.aux[i].tolist(), list[int]) for i in keep],
        "action": from_plain(segment.actions.tolist(), list[int]),
        "reward": from_plain(segment.reward.tolist(), list[int]),
        "done": from_plain(segment.done.tolist(), list[bool]),
    }


def evaluate(
    model: WorldModel,
    *,
    archive: Path,
    output: Path,
    rows: int,
    t_max: int,
    decisions: int,
    prefix: int,
    continuation: int,
    horizons: Sequence[int],
    seed: int,
    resamples: int,
    permutations: int,
    arm: int = 3,
    max_frames: int = 1_200,
    provenance: Mapping[str, PlainTree] | None = None,
) -> dict[str, PlainTree]:
    """Run every measurement and write ``stats.json``, ``episodes.json``, and bundles.

    Args:
      model: The trained model, on its device and dtype, in evaluation mode.
      archive: Archive root holding ``val/arm{arm}/w*/``.
      output: New directory for every output.
      rows: Engine rows; half as many windows are continued.
      t_max: Engine context in global positions.
      decisions: Free-running decisions per row, and the real horizon.
      prefix: Prefilled real decisions per window.
      continuation: Decisions continued per window.
      horizons: Decision counts at which free-running rates are compared.
      seed: Seed of the sampler, the episode choices, and the statistics.
      resamples: Bootstrap resamples.
      permutations: Permutation-test relabelings.
      arm: Behaviour arm whose validation episodes are real.
      max_frames: Frames kept per episode in ``episodes.json``.
      provenance: Extra ``provenance`` entries.

    Returns:
      stats: What ``stats.json`` holds.

    Raises:
      ValueError: If the arm's validation episodes do not give ``rows / 2``
        windows, checked before anything is generated or written.

    """
    sources = val_sources(archive, arm=arm)
    picker = torch.Generator().manual_seed(seed)
    chosen = _shuffled(sources, picker)[:rows]
    windows = choose_windows(
        sources,
        count=rows // 2,
        prefix=prefix,
        decisions=continuation,
        generator=picker,
    )
    if not windows or 2 * len(windows) != rows:
        raise ValueError(
            f"Continuations need rows / 2 = {rows / 2:g} windows; arm {arm}'s "
            f"validation episodes give {len(windows)} of {prefix} + {continuation} "
            "decisions.",
        )
    output.mkdir(parents=True)
    engine, step = sampling_engine(model, rows=rows, t_max=t_max, seed=seed)
    seconds: dict[str, PlainTree] = {}
    clock = time.monotonic()
    free = _cpu(rollout(engine, step, decisions=decisions))
    seconds["free_running"] = time.monotonic() - clock
    real = [truncate(read_source(s), decisions=decisions) for s in chosen]
    dreams = episodes_of(free)
    real_features = [episode_features(s, horizons=horizons) for s in real]
    dream_features = [episode_features(s, horizons=horizons) for s in dreams]
    comparison = compare(
        real_features,
        dream_features,
        horizons=horizons,
        resamples=resamples,
        permutations=permutations,
        seed=seed,
    )
    clock = time.monotonic()
    first, second = _continuations(engine, step, windows, decisions=continuation)
    seconds["continuations"] = time.monotonic() - clock
    half = len(windows)
    truth = [w.real for w in windows]
    model_rows = episodes_of(first, rows=range(half))
    real_rows = episodes_of(first, rows=range(half, 2 * half))
    resampled = episodes_of(second, rows=range(half, 2 * half))
    checkpoints = [k for k in (1, 8, 32, 128, 256, 512, 1024) if k <= continuation]
    measure = functools.partial(
        divergence,
        checkpoints=checkpoints,
        resamples=resamples,
        seed=seed,
    )
    continued: dict[str, PlainTree] = {
        "windows": [
            {"name": w.name, "kind": w.kind, "real_decisions": len(w.real.actions)}
            for w in windows
        ],
        "prefix": prefix,
        "decisions": continuation,
        "model_actions": measure(model_rows, truth),
        "real_actions": measure(real_rows, truth),
        "real_actions_resample": measure(resampled, truth),
        "real_actions_sample_to_sample": measure(resampled, real_rows),
        "teacher_forced": teacher_nll(
            _rows(second, range(half)),
            truth,
            resamples=resamples,
            seed=seed,
        ),
    }
    bundles = _write_bundles(
        output / "bundles",
        free=free,
        features=dream_features,
        first=first,
        windows=windows,
        firsts=[
            _first_true(_pair(x, y)["frame_diff"])
            for x, y in zip(real_rows, truth, strict=True)
        ],
        origin=str((provenance or {}).get("checkpoint", "World model")),
    )
    records = [
        episode_record(segment, name=name, source=source, max_frames=max_frames)
        for name, source, segment in _showcase(
            dreams,
            bundles=bundles,
            first=first,
            windows=windows,
            generator=picker,
        )
    ]
    (output / "episodes.json").write_text(
        json.dumps(_episodes_document(records)) + "\n",
    )
    stats: dict[str, PlainTree] = {
        "schema": SCHEMA,
        "provenance": {
            **(provenance or {}),
            "archive": str(archive),
            "arm": arm,
            "seed": seed,
            "rows": rows,
            "t_max": t_max,
            "temperature": 1.0,
            "graphed": isinstance(step, GraphedStep),
            "seconds": seconds,
            "decisions_per_second": {
                "free_running": rows * decisions / _seconds(seconds, "free_running"),
                "continuations": 2
                * rows
                * continuation
                / _seconds(seconds, "continuations"),
            },
        },
        "free_running": _free_summary(free),
        "real_episodes": [s.name for s in chosen],
        "comparison": comparison,
        "continuation": continued,
        "bundles": bundles,
        "episodes_file": "episodes.json",
    }
    (output / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    return stats


class Flags(Protocol):
    """Parsed command-line flags."""

    checkpoint: Path
    experiment: str
    override: list[str]
    archive: Path
    output: Path
    arm: int
    rows: int
    t_max: int
    decisions: int
    prefix: int
    continuation: int
    horizons: list[int]
    seed: int
    resamples: int
    permutations: int
    device: str
    dtype: str


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
    parser.add_argument("--archive", type=Path, required=True, help="Archive root.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory.",
    )
    parser.add_argument(
        "--arm",
        type=int,
        default=3,
        help="Behaviour arm of the real episodes.",
    )
    parser.add_argument("--rows", type=int, default=256, help="Engine rows.")
    parser.add_argument(
        "--t-max",
        type=int,
        default=0,
        help="Engine context positions; default the run's window.",
    )
    parser.add_argument(
        "--decisions",
        type=int,
        default=8_000,
        help="Free-running decisions per row.",
    )
    parser.add_argument(
        "--prefix",
        type=int,
        default=256,
        help="Prefilled real decisions.",
    )
    parser.add_argument(
        "--continuation",
        type=int,
        default=1_024,
        help="Open-loop decisions per window.",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[250, 500, 1_000, 2_000, 4_000, 8_000],
        help="Decision counts at which rates are compared.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed of every draw.")
    parser.add_argument(
        "--resamples",
        type=int,
        default=2_000,
        help="Bootstrap resamples.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1_000,
        help="Permutation-test relabelings.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device.")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="",
        help="Model dtype; default bfloat16 on CUDA, float32 elsewhere.",
    )


def _control(
    engine: Engine,
    index: int,
    *,
    actions: Tensor | None,
    teacher: Teacher | None,
) -> Control:
    """Return decision ``index``'s control: forced actions and forced act jobs."""
    control = engine.control()
    device = control.action.device
    if actions is not None:
        column = actions[:, index].to(device)
        control.action_forced.copy_(column >= 0)
        control.action.copy_(column.clamp(min=0))
    if teacher is not None:
        outcome = teacher.outcome
        tokens = engine.job_tokens(
            reward=outcome.reward[:, index],
            done=outcome.done[:, index],
            cells=outcome.cells[:, index],
            aux=outcome.aux[:, index],
        )
        control.job_tokens.copy_(tokens.to(device))
        control.job_forced.copy_(
            teacher.rows.to(device)[:, None].expand_as(control.job_forced),
        )
    return control


# The first rollout's rows are the windows under the model's actions, then under the
# recorded actions; the second's are the windows teacher-forced, then under the recorded
# actions again with fresh noise.
def _continuations(
    engine: Engine,
    step: Step,
    windows: Sequence[Window],
    *,
    decisions: int,
) -> tuple[Rollout, Rollout]:
    """Continue every window twice per rollout."""
    half = len(windows)
    prefixes = [w.prefix for w in windows] * 2
    recorded = torch.zeros(half, decisions, dtype=torch.long)
    outcome = {
        "reward": torch.zeros(half, decisions, dtype=torch.long),
        "done": torch.zeros(half, decisions, dtype=torch.bool),
    }
    frames = torch.zeros(
        half,
        decisions,
        *windows[0].real.cells.shape[1:],
        dtype=torch.uint8,
    )
    scalars = torch.zeros(
        half,
        decisions,
        windows[0].real.aux.shape[-1],
        dtype=torch.int16,
    )
    for row, w in enumerate(windows):
        n = len(w.real.actions)
        recorded[row, :n] = w.real.actions.long()
        outcome["reward"][row, :n] = w.real.reward.long()
        outcome["done"][row, :n] = w.real.done
        # A decision's next frame; after the real terminal, or past the real
        # end, the last real frame stands in, unscored.
        following = torch.arange(1, decisions + 1).clamp(max=len(w.real.cells) - 1)
        frames[row] = w.real.cells[following]
        scalars[row] = w.real.aux[following]
    sampled = torch.full((half, decisions), -1)
    first = _cpu(
        rollout(
            engine,
            step,
            decisions=decisions,
            prefixes=prefixes,
            actions=torch.cat([sampled, recorded]),
        ),
    )
    both = {name: torch.cat([value, value]) for name, value in outcome.items()}
    teacher = Teacher(
        rows=torch.arange(2 * half) < half,
        outcome=Outcome(
            reward=both["reward"],
            done=both["done"],
            cells=torch.cat([frames, frames]),
            aux=torch.cat([scalars, scalars]),
        ),
    )
    second = _cpu(
        rollout(
            engine,
            step,
            decisions=decisions,
            prefixes=prefixes,
            actions=torch.cat([recorded, recorded]),
            teacher=teacher,
        ),
    )
    return first, second


def _cpu(rollout: Rollout) -> Rollout:
    """Return ``rollout`` with every tensor on the CPU."""
    return Rollout(
        **{
            f.name: cast("Tensor", getattr(rollout, f.name)).cpu()
            for f in dataclasses.fields(rollout)
        },
    )


def _rows(rollout: Rollout, rows: Sequence[int]) -> Rollout:
    """Return some rows of a rollout."""
    index = torch.tensor(list(rows))
    return Rollout(
        **{
            f.name: cast("Tensor", getattr(rollout, f.name))[index]
            for f in dataclasses.fields(rollout)
        },
    )


def _slice(rollout: Rollout, *, row: int, stop: int) -> Rollout:
    """Return one row's decisions ``[0, stop)`` and frames ``[0, stop]``."""
    frames = {"cells", "aux", "starts", "frame_logp", "invalid"}
    return Rollout(
        **{
            f.name: cast("Tensor", getattr(rollout, f.name))[
                row : row + 1,
                : stop + (f.name in frames),
            ]
            for f in dataclasses.fields(rollout)
        },
    )


def _shuffled[T](items: Sequence[T], generator: torch.Generator) -> list[T]:
    """Return ``items`` in a random order."""
    return [items[int(i)] for i in torch.randperm(len(items), generator=generator)]


def _names(windows: Sequence[Window]) -> set[str]:
    """Return the episode names of ``windows``."""
    return {w.name.rsplit("@", 1)[0] for w in windows}


def _uniform(low: int, high: int, generator: torch.Generator) -> int:
    """Return a uniform integer in ``[low, high]``."""
    return int(torch.randint(low, max(high, low) + 1, (), generator=generator))


def _first(mask: Tensor) -> Tensor:
    """Return each column's first True row of ``mask [N, K]``, or -1."""
    if not len(mask):
        return torch.full(mask.shape[1:], -1)
    return torch.where(mask.any(0), mask.int().argmax(0), -1)


def _fields(*names: str) -> list[int]:
    """Return the auxiliary indexes of ``names``."""
    return [FIELDS.index(name) for name in names]


def _event_specs() -> dict[str, tuple[str, list[int], int]]:
    """Return each event's kind (``state``, ``up``, ``down``), fields, and threshold."""
    materials = (
        "wood",
        "stone",
        "coal",
        "iron",
        "diamond",
        "sapphire",
        "ruby",
        "sapling",
    )
    specs = {f"collect_{m}": ("up", _fields(m), 0) for m in materials}
    specs |= {
        "eat": ("up", _fields("food"), 0),
        "drink": ("up", _fields("drink"), 0),
        "take_damage": ("down", _fields("health"), 0),
        "gain_arrows": ("up", _fields("arrows"), 0),
        "gain_torches": ("up", _fields("torches"), 0),
        "place_torch": ("down", _fields("torches"), 0),
        "shoot_arrow": ("down", _fields("arrows"), 0),
        "drink_potion": (
            "down",
            [i for i, n in enumerate(FIELDS) if n.startswith("potion_")],
            0,
        ),
        "read_book": ("down", _fields("books"), 0),
        "sleep": ("state", _fields("sleeping"), 1),
        "gain_xp": ("state", _fields("xp"), 1),
        "find_bow": ("state", _fields("bow"), 1),
        "enchant_sword": ("state", _fields("sword_enchantment"), 1),
        "enchant_bow": ("state", _fields("bow_enchantment"), 1),
        "learn_fireball": ("state", _fields("learned_fireball"), 1),
        "learn_iceball": ("state", _fields("learned_iceball"), 1),
        "iron_armour": ("state", _fields(*(f"armour_{i}" for i in range(4))), 1),
        "diamond_armour": ("state", _fields(*(f"armour_{i}" for i in range(4))), 2),
        "enchant_armour": (
            "state",
            _fields(*(f"armour_enchantment_{i}" for i in range(4))),
            1,
        ),
    }
    for level, material in enumerate(("wood", "stone", "iron", "diamond"), start=1):
        specs[f"{material}_pickaxe"] = ("state", _fields("pickaxe"), level)
        specs[f"{material}_sword"] = ("state", _fields("sword"), level)
    for attribute in ("dexterity", "strength", "intelligence"):
        specs[f"level_up_{attribute}"] = ("state", _fields(attribute), 2)
    return specs


def _events(aux: Tensor, *, delta: Tensor, steps: Tensor) -> Tensor:
    """Return the decision at which each of ``EVENTS`` first shows, or -1."""
    specs = _event_specs()
    firsts: list[int] = []
    for name in EVENTS:
        kind, fields, threshold = specs[name]
        if kind == "state":
            shown = int(_first((aux[:, fields].amax(-1) >= threshold)[:, None])[0])
            firsts.append(max(shown - 1, 0) if shown >= 0 else -1)
            continue
        change = delta[:, fields]
        moved = int(
            _first(((change > 0) if kind == "up" else (change < 0)).any(-1)[:, None])[
                0
            ],
        )
        firsts.append(int(steps[moved]) if moved >= 0 else -1)
    return torch.tensor(firsts, dtype=torch.long)


def _checks(
    segment: Segment,
    *,
    steps: Tensor,
    delta: Tensor,
    ended: bool,
    died: bool,
) -> Tensor:
    """Return ``[violations, opportunities]`` of each of ``CHECKS``, float64 ``[C, 2]``."""
    cells, aux = segment.cells.long(), segment.aux.long()
    actions = segment.actions.long()
    decisions, frames = len(actions), len(cells)
    v = {name: aux[:, i] for i, name in enumerate(FIELDS)}
    invalid = invalid_cells(cells, schema=craftax_schema())
    legal = legal_actions(cells[:decisions], aux[:decisions])
    illegal = ~legal.gather(1, actions[:, None])[:, 0]
    needs = torch.stack([v["food"], v["drink"], v["energy"]], -1)
    over = (
        (v["health"] > 20 * (8 + v["strength"]))
        | (needs > (7 + 2 * v["dexterity"])[:, None]).any(-1)
        | (v["mana"] > 6 + 3 * v["intelligence"])
    )
    act = actions[steps]
    moves = (act >= 1) & (act <= 4)
    inconsistent = _move_inconsistent(cells[steps], cells[steps + 1], act) & moves
    facing = aux[steps + 1][:, 31:35].gather(1, (act.clamp(1, 4) - 1)[:, None])[:, 0]
    counts = {
        "invalid_frame": (invalid.any(-1).sum(), frames),
        "invalid_cell": (invalid.sum(), invalid.numel()),
        "illegal_action": (illegal.sum(), decisions),
        "over_maximum": (over.sum(), frames),
        "move_inconsistent": (inconsistent.sum(), moves.sum()),
        "facing_wrong": ((moves & (facing != 1)).sum(), moves.sum()),
        "terminal_without_death": (int(ended and not died), int(ended)),
    }
    counts |= {
        name: (violated.sum(), happened.sum())
        for name, (happened, violated) in _hud_rules(delta, act).items()
    }
    return torch.tensor(
        [[float(a), float(b)] for a, b in (counts[n] for n in CHECKS)],
        dtype=torch.float64,
    )


# A change needs an action of the game that makes it: floors change only by ladder,
# XP only from entering a floor, tools, armour, attributes, and spells only from their
# crafting, level-up, or reading action, potions go only by drinking that potion,
# materials come only from ``do``, and noop and moves never change the inventory.
# Chests, opened by ``do``, also hold diamond tools: 29 of 1,576 tool upgrades in 160
# real arm-3 episodes came from one.
def _hud_rules(delta: Tensor, act: Tensor) -> dict[str, tuple[Tensor, Tensor]]:
    """Return each HUD rule's transitions that changed a field and those that broke it."""
    up, down = delta > 0, delta < 0

    def made_by(changed: Tensor, allowed: Sequence[int]) -> tuple[Tensor, Tensor]:
        return changed, changed & ~torch.isin(act, torch.tensor(allowed))

    attributes = up[:, _fields("dexterity", "strength", "intelligence")]
    potions = down[:, [i for i, n in enumerate(FIELDS) if n.startswith("potion_")]]
    materials = (
        "wood",
        "stone",
        "coal",
        "iron",
        "diamond",
        "sapphire",
        "ruby",
        "sapling",
    )
    still = act <= 4
    return {
        "floor_change_without_ladder": made_by(delta[:, FLOOR_AUX] != 0, (18, 19)),
        "xp_without_descend": made_by(up[:, FIELDS.index("xp")], (18,)),
        "pickaxe_without_craft": made_by(
            up[:, FIELDS.index("pickaxe")],
            (5, 11, 12, 13, 20),
        ),
        "sword_without_craft": made_by(
            up[:, FIELDS.index("sword")],
            (5, 14, 15, 16, 21),
        ),
        "armour_without_craft": made_by(
            up[:, _fields(*(f"armour_{i}" for i in range(4)))].any(-1),
            (22, 23),
        ),
        "attribute_without_level_up": (
            attributes.any(-1),
            (attributes & (act[:, None] != torch.tensor([39, 40, 41]))).any(-1),
        ),
        "spell_without_book": made_by(
            up[:, _fields("learned_fireball", "learned_iceball")].any(-1),
            (35,),
        ),
        "potion_without_drink": (
            potions.any(-1),
            (potions & (act[:, None] != torch.arange(29, 35))).any(-1),
        ),
        "material_without_do": made_by(up[:, _fields(*materials)].any(-1), (5,)),
        "inventory_change_on_move": (still, still & (delta[:, :22] != 0).any(-1)),
    }


# Terrain is the block and item fields, compared only where both frames show the cell. A
# move left shows the old columns one to the right, and so on; a blocked move leaves the
# view in place.
def _move_inconsistent(before: Tensor, after: Tensor, act: Tensor) -> Tensor:
    """Return transitions whose terrain fits neither staying nor the move's shift."""
    shape = (len(act), 9, 11, 8)
    old, new = before.view(shape), after.view(shape)
    stayed = _terrain_differs(new, old)
    moved = torch.ones_like(stayed)
    everything = slice(None)
    shifts = {
        1: ((everything, slice(1, None)), (everything, slice(None, -1))),
        2: ((everything, slice(None, -1)), (everything, slice(1, None))),
        3: ((slice(1, None), everything), (slice(None, -1), everything)),
        4: ((slice(None, -1), everything), (slice(1, None), everything)),
    }
    for direction, (to, source) in shifts.items():
        rows = act == direction
        moved[rows] = _terrain_differs(
            new[rows][:, to[0], to[1]],
            old[rows][:, source[0], source[1]],
        )
    return stayed & moved


def _terrain_differs(a: Tensor, b: Tensor) -> Tensor:
    """Return whether any cell seen in both ``[N, R, C, 8]`` views differs in terrain."""
    seen = (a[..., 2] == 1) & (b[..., 2] == 1)
    return ((a[..., :2] != b[..., :2]).any(-1) & seen).flatten(1).any(1)


def _stack(features: Sequence[Features]) -> dict[str, Tensor]:
    """Stack one sample's features, one row per episode."""
    scalars = {
        "length": [float(f.length) for f in features],
        "died": [float(f.died) for f in features],
        "reward": [f.reward for f in features],
    }
    stacked = {
        name: torch.tensor(values, dtype=torch.float64)
        for name, values in scalars.items()
    }
    for name in (
        "floor_first",
        "returns",
        "actions",
        "aux",
        "events",
        "checks",
        "health_delta",
    ):
        stacked[name] = torch.stack([getattr(f, name) for f in features])
    return stacked


def _hit(first: Tensor, horizon: int) -> Tensor:
    """Return whether an event first shown at ``first`` happened before ``horizon``."""
    return ((first >= 0) & (first < horizon)).double()


@dataclasses.dataclass(slots=True, kw_only=True)
class _Tests:
    """The comparisons of one real and one generated sample, kept for Holm's correction.

    Each method reads a statistic from both samples' stacked features.
    """

    real: dict[str, Tensor]
    dream: dict[str, Tensor]
    resamples: int
    permutations: int
    generator: torch.Generator
    entries: list[tuple[str, dict[str, PlainTree]]] = dataclasses.field(
        default_factory=list,
    )

    def ratio(
        self,
        name: str,
        parts: Callable[[dict[str, Tensor]], tuple[Tensor, Tensor]],
    ) -> dict[str, PlainTree]:
        """Compare a pooled ratio: a numerator and a denominator per episode.

        Args:
          name: The test's name.
          parts: Reads the numerator and denominator per episode of a sample.

        Returns:
          entry: Both ratios, their difference, and its p-value.

        """
        entry = compare_ratio(
            *parts(self.real),
            *parts(self.dream),
            resamples=self.resamples,
            generator=self.generator,
        )
        self.entries.append((name, entry))
        return entry

    def rate(
        self,
        name: str,
        values: Callable[[dict[str, Tensor]], Tensor],
    ) -> dict[str, PlainTree]:
        """Compare the mean of a value per episode.

        Args:
          name: The test's name.
          values: Reads the value per episode of a sample.

        Returns:
          entry: Both means, their difference, and its p-value.

        """
        return self.ratio(name, lambda s: (values(s), torch.ones(len(values(s)))))

    def spread(
        self,
        name: str,
        counts: Callable[[dict[str, Tensor]], Tensor],
    ) -> dict[str, PlainTree]:
        """Compare the pooled distribution of per-episode counts.

        Args:
          name: The test's name.
          counts: Reads the counts per episode and category of a sample.

        Returns:
          entry: ``permutation_tv`` of the two samples.

        """
        entry = permutation_tv(
            counts(self.real),
            counts(self.dream),
            permutations=self.permutations,
            generator=self.generator,
        )
        self.entries.append((name, entry))
        return entry

    def aux(self, index: int, name: str) -> dict[str, PlainTree]:
        """Compare one auxiliary field's mean and its values' distribution.

        Args:
          index: The field's auxiliary index.
          name: The field's name.

        Returns:
          entry: The valid ``values`` range, the ``mean``, and the ``distribution``.

        """
        low, high = (bound - 155 for bound in craftax_schema().scalar_ranges[index])
        values = torch.arange(AUX_VALUES, dtype=torch.float64)
        return {
            "values": [low, high],
            "mean": self.ratio(
                f"aux/{name}/mean",
                lambda s: (
                    (s["aux"][:, index] * values).sum(-1),
                    s["aux"][:, index].sum(-1),
                ),
            ),
            "distribution": self.spread(
                f"aux/{name}",
                lambda s: s["aux"][:, index, low : high + 1],
            ),
        }


def _resample(episodes: int, resamples: int, generator: torch.Generator) -> Tensor:
    """Return bootstrap multiplicities, ``[resamples, episodes]`` float64."""
    picks = torch.randint(episodes, (resamples, episodes), generator=generator)
    weights = torch.zeros(resamples, episodes, dtype=torch.float64)
    return weights.scatter_add_(1, picks, torch.ones_like(weights))


def _tv(a: Tensor, b: Tensor) -> Tensor:
    """Return the total-variation distance between count vectors ``[..., K]``."""
    p = a / a.sum(-1, keepdim=True).clamp(min=1e-12)
    q = b / b.sum(-1, keepdim=True).clamp(min=1e-12)
    return total_variation(p, q)


def _distribution(counts: Tensor) -> list[PlainTree]:
    """Return counts normalized to a distribution."""
    return [float(x) for x in counts.double() / counts.sum().clamp(min=1e-12)]


def _quantiles(values: Tensor) -> dict[str, float]:
    """Return the 10th, 25th, 50th, 75th, and 90th percentiles of ``values``."""
    levels = (0.1, 0.25, 0.5, 0.75, 0.9)
    points = values.double().quantile(torch.tensor(levels, dtype=torch.float64))
    return {f"q{round(100 * q)}": float(v) for q, v in zip(levels, points, strict=True)}


def _holm(
    tests: Sequence[tuple[str, dict[str, PlainTree]]],
    alpha: float = 0.05,
) -> list[PlainTree]:
    """Mark each test ``distinguishable`` by Holm's correction; return their names."""
    ranked = sorted(
        (
            (p, name, entry)
            for name, entry in tests
            if isinstance(p := entry["p_value"], float)
        ),
        key=lambda item: item[0],
    )
    adjusted = from_plain(holm([p for p, _, _ in ranked]).tolist(), list[float])
    names: list[PlainTree] = []
    for (_, name, entry), p_holm in zip(ranked, adjusted, strict=True):
        entry["p_holm"] = p_holm
        entry["distinguishable"] = p_holm < alpha
        if p_holm < alpha:
            names.append(name)
    return names


# Frames 1 on of ``y`` are compared; one that ``x`` lacks, because it ended first,
# differs in every cell and field.
def _pair(x: Segment, y: Segment) -> dict[str, Tensor]:
    """Return the frame, reward, done, and action agreement of one pair."""
    shown = min(len(x.cells), len(y.cells))
    lost = len(y.cells) - shown
    field_diff = torch.cat(
        [
            x.cells[1:shown] != y.cells[1:shown],
            torch.ones(lost, *y.cells.shape[1:], dtype=torch.bool),
        ],
    )
    hud_diff = torch.cat(
        [
            x.aux[1:shown] != y.aux[1:shown],
            torch.ones(lost, *y.aux.shape[1:], dtype=torch.bool),
        ],
    )
    cell_diff = field_diff.any(-1)
    decisions = min(len(x.actions), len(y.actions))
    rx, ry = x.reward[:decisions], y.reward[:decisions]
    return {
        "frame_diff": cell_diff.any(-1) | hud_diff.any(-1),
        "board_diff": cell_diff.any(-1),
        "hud_diff": hud_diff,
        "board_frac": cell_diff.double().mean(-1),
        "field_frac": field_diff.double().mean(-2),
        "shown": torch.tensor(shown - 1),
        "reward_equal": rx == ry,
        "action_equal": x.actions[:decisions] == y.actions[:decisions],
        "nonzero_dream": (rx != 0).sum(),
        "nonzero_real": (ry != 0).sum(),
        "matched_nonzero": ((rx == ry) & (ry != 0)).sum(),
        "end_dream": torch.tensor(
            len(x.actions) - 1 if len(x.done) and bool(x.done[-1]) else -1,
        ),
        "end_real": torch.tensor(
            len(y.actions) - 1 if len(y.done) and bool(y.done[-1]) else -1,
        ),
        "compared": torch.tensor(decisions),
    }


def _first_true(mask: Tensor) -> int | None:
    """Return the 1-based index of the first True of ``mask``, or None."""
    hits = mask.nonzero()
    return int(hits[0, 0]) + 1 if len(hits) else None


def _done_agreement(pairs: Sequence[dict[str, Tensor]]) -> dict[str, PlainTree]:
    """Count terminals that fall on the same decision, and those only one side has."""
    same = dream_only = real_only = 0
    for p in pairs:
        dream, real, compared = (
            int(p["end_dream"]),
            int(p["end_real"]),
            int(p["compared"]),
        )
        same += dream >= 0 and dream == real
        dream_only += 0 <= dream < compared and dream != real
        real_only += 0 <= real < compared and dream != real
    return {
        "dream_terminals": sum(int(p["end_dream"]) >= 0 for p in pairs),
        "real_terminals": sum(int(p["end_real"]) >= 0 for p in pairs),
        "same_decision": same,
        "dream_only": dream_only,
        "real_only": real_only,
    }


def _free_summary(free: Rollout) -> dict[str, PlainTree]:
    """Summarize the free-running rollout as a whole."""
    return {
        "rows": len(free.done),
        "decisions": free.done.shape[1],
        "episodes_started": int(free.starts.sum()),
        "episodes_ended": int(free.done.sum()),
        "deaths": int((free.done & (free.reward == -1)).sum()),
        "first_episode_ended": int(free.done.any(-1).sum()),
        "invalid_frame_rate": float(free.invalid.any(-1).double().mean()),
        "max_floor": int(free.aux[..., FLOOR_AUX].max()),
    }


def _write_bundles(
    directory: Path,
    *,
    free: Rollout,
    features: Sequence[Features],
    first: Rollout,
    windows: Sequence[Window],
    firsts: Sequence[int | None],
    origin: str,
) -> list[PlainTree]:
    """Write viewer bundles of chosen dreams and continuations; return their index."""
    written: list[PlainTree] = []
    for name, row in _free_choices(features).items():
        stop = features[row].length
        path = directory / f"free-{name}"
        write_bundle(
            stream_of(_slice(free, row=row, stop=stop), row=0),
            path,
            title=f"Dream: {name.replace('_', ' ')} (row {row})",
            description=f"{stop:,} decisions from a new world, model actions, temperature 1",
            provenance=f"{origin}, free-running from start; the frame after a "
            "terminal is the next episode's first frame.",
        )
        written.append(
            {
                "name": f"free-{name}",
                "path": str(path),
                "row": row,
                "decisions": stop,
                "reference": False,
            },
        )
    half = len(windows)
    for name, index, rows_offset, mark in _continuation_choices(windows, firsts, half):
        w = windows[index]
        row = rows_offset + index
        stop = len(episodes_of(first, rows=[row])[0].actions)
        path = directory / f"continuation-{name}"
        write_bundle(
            stream_of(_slice(first, row=row, stop=stop), row=0, action_mark=mark),
            path,
            title=f"Continuation: {name.replace('_', ' ')} ({w.kind})",
            reference=w.real,
            description=f"{w.name}: {stop:,} decisions after a {len(w.prefix.actions[0])}-decision "
            "real prefix; the real episode is drawn beside it",
            provenance=f"{origin}, open-loop from a held-out validation prefix; "
            + (
                "recorded actions forced."
                if mark == Mark.FORCED
                else "model actions, "
                "so the real board beside it took different actions."
            ),
        )
        written.append(
            {
                "name": f"continuation-{name}",
                "path": str(path),
                "row": row,
                "window": w.name,
                "decisions": stop,
                "reference": True,
            },
        )
    return written


def _later(first: int | None) -> float:
    """Rank a first divergence, never diverging last."""
    return math.inf if first is None else first


def _free_choices(features: Sequence[Features]) -> dict[str, int]:
    """Return rows of typical, longest, deepest, and a dying first episode."""
    lengths = torch.tensor([f.length for f in features], dtype=torch.float64)
    reward = torch.tensor([f.reward for f in features], dtype=torch.float64)
    deepest = torch.tensor(
        [int((f.floor_first >= 0).nonzero().max()) for f in features],
    )
    choices = {
        "typical": int((lengths - lengths.median()).abs().argmin()),
        "longest": int((lengths * 1e6 + reward).argmax()),
        "deepest_floor": int((deepest * 1e9 + lengths).argmax()),
    }
    died = torch.tensor([f.died for f in features])
    if died.any():
        dying = died.nonzero()[:, 0]
        choices["death"] = int(
            dying[(lengths[dying] - lengths[dying].median()).abs().argmin()],
        )
    return choices


# ``firsts`` holds each recorded-action continuation's first divergent frame, None where
# it never diverged, which ranks last.
def _continuation_choices(
    windows: Sequence[Window],
    firsts: Sequence[int | None],
    half: int,
) -> list[tuple[str, int, int, Mark]]:
    """Return continuations to bundle: name, window, row offset, and action mark."""
    order = sorted(range(len(firsts)), key=lambda i: _later(firsts[i]))
    median, longest = order[len(order) // 2], order[-1]
    chosen = [
        ("recorded_median", median, half, Mark.FORCED),
        ("recorded_longest", longest, half, Mark.FORCED),
        ("model_actions", median, 0, Mark.MODEL),
    ]
    deaths = [i for i, w in enumerate(windows) if w.kind == "pre_death"]
    if deaths:
        chosen.append(("recorded_pre_death", deaths[0], half, Mark.FORCED))
    return chosen


def _showcase(
    dreams: Sequence[Segment],
    *,
    bundles: Sequence[PlainTree],
    first: Rollout,
    windows: Sequence[Window],
    generator: torch.Generator,
) -> list[tuple[str, str, Segment]]:
    """Return a dozen episodes for ``episodes.json``: bundled ones and two random dreams."""
    chosen: list[tuple[str, str, Segment]] = []
    rows: list[int] = []
    for bundle in bundles:
        entry = from_plain(bundle, dict[str, object])
        row = from_plain(entry["row"], int)
        name = str(entry["name"])
        if name.startswith("free-"):
            chosen.append((name, "model", dreams[row]))
            rows.append(row)
            continue
        index = next(i for i, w in enumerate(windows) if w.name == entry["window"])
        chosen.append((name, "model", episodes_of(first, rows=[row])[0]))
        if not name.endswith("model_actions"):
            chosen.append((f"{name}-real", "real", windows[index].real))
    spare = [
        r
        for r in from_plain(
            torch.randperm(len(dreams), generator=generator).tolist(),
            list[int],
        )
        if r not in rows
    ]
    chosen += [(f"free-random-{r}", "model", dreams[r]) for r in spare[:2]]
    return chosen


def _episodes_document(records: Sequence[dict[str, PlainTree]]) -> dict[str, PlainTree]:
    """Return ``episodes.json``: the vocabulary a page needs and the episodes."""
    schema = craftax_schema()
    return {
        "schema": "craftax-dream-episodes/v1",
        "cell_layout": "99 cells row-major (9 rows x 11 columns, player at cell 49), "
        "8 field values per cell in cell_fields order",
        "cell_fields": [{"name": f.name, "valid": f.valid} for f in schema.cell_fields],
        "aux_fields": list(schema.scalar_names),
        "block_names": list(BLOCK_NAMES),
        "item_names": list(ITEM_NAMES),
        "action_names": list(ACTION_NAMES),
        "episodes": list(records),
    }


def _seconds(seconds: Mapping[str, PlainTree], key: str) -> float:
    """Return a timing, at least a microsecond."""
    value = seconds[key]
    assert isinstance(value, float)
    return max(value, 1e-6)


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 hex digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
