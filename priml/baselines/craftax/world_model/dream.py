"""Generated episodes: batched open-loop rollouts of the world model.

Every engine row starts either from ``start`` (a new world) or from its own real
prefix, a held-out episode replayed from the archive and prefilled for any number
of decisions, and then runs open-loop. Actions come from the model's action head,
a ``Policy`` fed the decoded 843-float observation, or a recorded action
stream. A row whose episode ends begins the next one at its following step.

The rollout stores every frame's tokens, per-slot log-probabilities, and
invalid-cell flags, which the viewer's model bundles are built from.
"""

from collections.abc import Sequence

import collections
import dataclasses

from torch import Tensor

import torch

from priml.baselines.craftax.world_model.codec import decode
from priml.baselines.craftax.world_model.engine import (
    Engine,
    Policy,
    Prefix,
)
from priml.baselines.craftax.world_model.schema import FrameSchema


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Rollout:
    """``D`` decisions and ``D + 1`` frames per row.

    Frame ``d`` is observed by decision ``d``; frame ``d + 1`` is its next
    frame, or the first frame of the following episode when decision ``d`` ends
    one. Frame 0 of a prefixed row is the prefix's last, real frame.

    Attributes:
      cells: Cell values, uint8 ``[B, D + 1, cells, fields]``.
      aux: Auxiliary values, int16 ``[B, D + 1, scalars]``.
      starts: Whether each frame begins an episode, bool ``[B, D + 1]``.
      frame_logp: Log-probability of each generated frame slot, 0 for real
        frames, float32 ``[B, D + 1, frame slots]``.
      invalid: Cells whose fields no observation of the game can hold, bool
        ``[B, D + 1, cells]``.
      action: Executed actions, uint8 ``[B, D]``.
      reward: Generated rewards, int16 ``[B, D]``.
      done: Generated terminal flags, bool ``[B, D]``.
      action_logp: Action-head log-probability of each action, float32 ``[B, D]``.
      reward_logp: Log-probability of each reward, float32 ``[B, D]``.
      done_logp: Log-probability of each terminal flag, float32 ``[B, D]``.

    """

    cells: Tensor
    aux: Tensor
    starts: Tensor
    frame_logp: Tensor
    invalid: Tensor
    action: Tensor
    reward: Tensor
    done: Tensor
    action_logp: Tensor
    reward_logp: Tensor
    done_logp: Tensor


def dream(
    engine: Engine,
    *,
    decisions: int,
    prefixes: Sequence[Prefix | None] = (),
    actions: Tensor | Policy | None = None,
) -> Rollout:
    """Generate ``decisions`` decisions in every engine row.

    Args:
      engine: The engine; every row is restarted.
      decisions: Decisions to generate per row.
      prefixes: One entry per row: a one-row ``Prefix`` of real decisions, of
        any length, or None to start a new world; empty starts every row.
      actions: Recorded actions ``[B, decisions]``, a ``Policy``, or None for
        the model's action head.

    Returns:
      rollout: Tokens, log-probabilities, and invalid-cell flags.

    Raises:
      ValueError: If ``decisions`` is not positive, or ``prefixes`` is neither
        empty nor one per row.

    """
    if decisions < 1:
        raise ValueError(f"decisions={decisions} must be positive.")
    if prefixes and len(prefixes) != engine.rows:
        raise ValueError(
            f"Got {len(prefixes)} prefixes; expected one per row ({engine.rows}).",
        )
    state = engine.state
    engine.reset(torch.ones(engine.rows, dtype=torch.bool))
    # One global forward per prefixed row: prefixes differ in length, and this
    # runs once per rollout.
    for row, prefix in enumerate(prefixes):
        if prefix is not None:
            engine.prefill(torch.tensor([row]), prefix)
    begun = engine.start(engine.control())
    frame_logp = torch.where(begun.started[:, None], begun.job.logp[:, 2:], 0.0)
    frames = [_frame(engine, logp=frame_logp)]
    starts = [begun.started]
    record: dict[str, list[Tensor]] = collections.defaultdict(list)
    room = 0
    for step in range(decisions):
        if room == 0:
            room = engine.ensure_room()
        control = engine.control()
        if isinstance(actions, Tensor):
            control.action_forced.fill_(value=True)
            control.action.copy_(actions[:, step])
        elif actions is not None:
            control.action_forced.fill_(value=True)
            control.action.copy_(actions(decode(state.cells, state.aux)))
        decision = engine.decide(control)
        room -= 1
        job = decision.job
        if engine.starting(control):
            begun = engine.start(control)
        else:
            begun = engine.skip_start(control)
        outcome = engine.outcome(job.tokens)
        logp = torch.where(begun.started[:, None], begun.job.logp, job.logp)
        frames.append(_frame(engine, logp=logp[:, 2:]))
        starts.append(begun.started)
        record["action"].append(decision.action.to(torch.uint8))
        record["reward"].append(outcome.reward.to(torch.int16))
        record["done"].append(outcome.done)
        record["action_logp"].append(decision.action_logp)
        record["reward_logp"].append(job.logp[:, 0])
        record["done_logp"].append(job.logp[:, 1])
    cells = torch.stack([cells for cells, _, _ in frames], dim=1)
    return Rollout(
        cells=cells,
        aux=torch.stack([aux for _, aux, _ in frames], dim=1),
        starts=torch.stack(starts, dim=1),
        frame_logp=torch.stack([logp for _, _, logp in frames], dim=1),
        invalid=invalid_cells(cells, schema=engine.model.schema),
        **{name: torch.stack(values, dim=1) for name, values in record.items()},
    )


def invalid_cells(cells: Tensor, *, schema: FrameSchema) -> Tensor:
    """Flag cells whose fields no observation of the game can hold.

    ``compute_observations_numba`` writes an unseen cell as all zeros, and melee,
    passive, and ranged mobs share one occupancy bitmap, so at most one is set.
    Projectiles are not exclusive: the observation keeps the last one written to
    a cell.

    Args:
      cells: Cell values ``[..., cells, fields]``.
      schema: The schema naming the cell fields.

    Returns:
      invalid: Bool ``[..., cells]``.

    """
    names = [field.name for field in schema.cell_fields]
    unseen = cells[..., names.index("visibility")] == 0
    mobs = cells[..., [names.index(name) for name in ("melee", "passive", "ranged")]]
    return (unseen & (cells != 0).any(-1)) | ((mobs != 0).sum(-1) > 1)


def _frame(engine: Engine, *, logp: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return every row's current frame, compactly typed, with its slot log-probabilities."""
    state = engine.state
    return state.cells.to(torch.uint8), state.aux.to(torch.int16), logp.float()
