"""Self-imitation of practice branches: the policy's own legal actions, relearned.

A practice row restored at step 0 of a rollout plays a branch from an archived
state; the branch runs until its episode ends or the rollout does. Each epoch
:class:`BranchImitation` takes the branches of rows ``[0, rows)`` whose every
action was legal into a first-in-first-out archive of ``capacity``, in the
order they completed: by their last step, then by row. Before the first window
it replays one archived branch, round robin from the oldest, from the carry
its rollout started it with, and that window adds

    coefficient * mean_t CE(logits_t with illegal actions at -inf, a_t)

over the branch's steps to its backward. It fills
``AgentWindows.Config.auxiliary``.

With a per-step feature (``LearnerRollout.features``), the archive keeps each
archived step's stored feature, the one its actor read, and the replay reads
those: the actor's input, as the learner's windows read it. The loss scores
the branch's own steps alone -- the actions' cross-entropy, with no value or
return to bootstrap past the branch's end -- and the trunk is causal, so no
step reads a feature beyond the archived horizon.

The shapes are static: the archive holds ``capacity`` whole horizons, a
branch's length masks the steps past it, and the counters and the cursor live
on the device, so the learner epoch still captures as one CUDA graph. The mean
is therefore the masked sum over the horizon divided by the length. Against
torch's ``mean`` over the branch alone, measured at every length to 256:

- on the CPU, which divides too, the gradient is bit for bit; the value, which
  only the metrics read, differs in the last bit at 29 lengths, as the sum
  runs over the whole horizon;
- on CUDA, where ``mean`` multiplies by the fp32 reciprocal of the count, the
  gradient's scale differs in the last bit at 69 lengths and the value at 85.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import (
    TYPE_CHECKING,
    cast,
)

import math

from configgle import Fig
from torch import Tensor
from torch.nn import functional

import torch


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.baselines.craftax.model import Policy
    from priml.baselines.craftax.train_step import (
        CraftaxTrainStep,
        LearnerRollout,
    )


def imitation_loss(
    logits: Tensor,
    actions: Tensor,
    action_mask: Tensor,
    length: Tensor,
    *,
    coefficient: float,
) -> Tensor:
    """Return ``coefficient`` times a branch's mean cross-entropy of its actions.

    Args:
      logits: ``[horizon, num_actions]``, any float dtype; scored in fp32.
      actions: ``[horizon]``, the action ids taken, in any dtype.
      action_mask: ``[horizon, num_actions]``, nonzero where legal.
      length: 0-dim int64, the branch's steps; those past it are left out.
      coefficient: The loss's weight.

    Returns:
      loss: 0-dim fp32; zero, with a zero gradient, when ``length`` is zero.

    """
    losses = masked_cross_entropy(logits, actions, action_mask, length)
    valid = torch.arange(logits.shape[0], device=logits.device) < length
    return coefficient * ((losses * valid).sum() / length.clamp_min(1))


def masked_cross_entropy(
    logits: Tensor,
    actions: Tensor,
    action_mask: Tensor,
    length: Tensor,
) -> Tensor:
    """Return each step's cross-entropy of its action, illegal actions' logits at -inf.

    Args:
      logits: ``[horizon, num_actions]``, any float dtype; scored in fp32.
      actions: ``[horizon]``, the action ids taken, in any dtype.
      action_mask: ``[horizon, num_actions]``, nonzero where legal.
      length: 0-dim int64, the branch's steps.

    Returns:
      losses: ``[horizon]`` fp32. A step past ``length`` scores every action
        legal, so it is finite whatever its mask.

    """
    valid = torch.arange(logits.shape[0], device=logits.device) < length
    # A step past the branch may have no legal action; its row of -inf would be
    # NaN, which the mean's zero weight does not remove (NaN * 0 is NaN).
    legal = (action_mask != 0) | ~valid[:, None]
    return functional.cross_entropy(
        logits.float().masked_fill(~legal, -math.inf),
        actions.long(),
        reduction="none",
    )


class BranchImitation:
    """Relearn archived practice branches' actions, one branch per epoch."""

    class Config(Fig["BranchImitation"]):
        """Which rows' branches, how many are kept, and the loss's weight."""

        rows: int = 64
        """Rows ``[0, rows)`` of each rollout whose restored branches are archived."""

        capacity: int = 32
        """Branches the archive keeps: the newest."""

        coefficient: float = 0.01
        """The loss's weight in the first window's total."""

    def __init__(self, config: Config) -> None:
        """Keep the geometry and the weight.

        Args:
          config: The rows, the capacity and the coefficient.

        Raises:
          ValueError: A count is not positive, or the coefficient is not
            positive and finite.

        """
        for name, count in (("rows", config.rows), ("capacity", config.capacity)):
            if count <= 0:
                raise ValueError(f"{name} must be positive, not {count}")
        coefficient = config.coefficient
        if math.isnan(coefficient) or math.isinf(coefficient) or coefficient <= 0:
            raise ValueError(
                f"coefficient must be positive and finite, not {coefficient}",
            )
        self.rows = config.rows
        self.capacity = config.capacity
        self.coefficient = coefficient
        self.device: torch.device | None = None
        """The step's device, from :meth:`prepare`; a loaded archive goes there."""
        self._archive: _Archive | None = None

    def prepare(self, config: CraftaxTrainStep.Config) -> None:
        """Check that the rows exist and take the step's device.

        Args:
          config: The step's recipe.

        Raises:
          ValueError: There are fewer environments than ``rows``.

        """
        agents = config.env.num_envs
        if self.rows > agents:
            msg = f"imitation reads rows [0, {self.rows}) of {agents} environments"
            raise ValueError(msg)
        self.device = config.parallelism.make().device

    def ingest(self, rollout: LearnerRollout) -> None:
        """Archive the epoch's legal branches of rows ``[0, rows)``.

        A row whose step 0 was restored holds a branch up to, not including,
        the first step after 0 whose terminal is set (the reset's
        observation), or the whole horizon. The archive takes the newest
        ``capacity`` of its old branches and these, in completion order.

        Args:
          rollout: The epoch's rollout, agent-major; the archive takes its
            shapes and dtypes from the first.

        """
        if self._archive is None:
            self._archive = _Archive.allocate(rollout, capacity=self.capacity)
        archive, rows, capacity = self._archive, self.rows, self.capacity
        horizon = rollout.actions.shape[1]
        row = torch.arange(rows, device=rollout.actions.device)
        ended = rollout.terminals[:rows, 1:] != 0
        lengths = 1 + (ended.cumsum(-1) == 0).sum(-1)
        valid = torch.arange(horizon, device=row.device) < lengths[:, None]
        taken = rollout.action_mask[:rows].gather(
            -1,
            rollout.actions[:rows].long()[..., None],
        )
        legal = ((taken[..., 0] != 0) | ~valid).all(-1)
        kept = (rollout.branch_starts[:rows] != 0) & legal
        # Completion order: by the branch's last step, then by its row.
        order = lengths * rows + row
        rank = (kept[None, :] & (order[None, :] < order[:, None])).sum(-1)
        new = kept.sum()
        # Only this epoch's newest ``capacity`` survive, each in the ring slot
        # after the last one filled; no two then share a slot.
        placed = kept & (rank >= new - capacity)
        slots = (archive.inserted + rank) % capacity
        writes = placed[None, :] & (
            slots[None, :] == torch.arange(capacity, device=row.device)[:, None]
        )
        written = writes.any(-1)
        source = (writes * row).sum(-1)
        archived = [
            (archive.observations, rollout.observations, 0),
            (archive.actions, rollout.actions, 0),
            (archive.action_mask, rollout.action_mask, 0),
            (archive.lengths, lengths, 0),
            (archive.initial_states, rollout.initial_states, 1),
        ]
        if archive.features is not None:
            # The archive takes its shapes from the first rollout, so a run with a
            # feature stores one in every rollout.
            assert isinstance(rollout.features, Tensor)
            archived.append((archive.features, rollout.features, 0))
        for target, values, axis in archived:
            shape = [1] * target.ndim
            shape[axis] = capacity
            torch.where(
                written.reshape(shape),
                values.index_select(axis, source),
                target,
                out=target,
            )
        archive.inserted.add_(new)
        torch.clamp_max(archive.count + new, capacity, out=archive.count)

    def loss(self, policy: Policy) -> tuple[Tensor, dict[str, Tensor]]:
        """Replay the cursor's branch and return its loss, then advance the cursor.

        Args:
          policy: The live policy, scored from the branch's carry with no
            episode start and no actions, so it adds no loss of its own, and
            on the branch's stored features if the archive keeps them.

        Returns:
          loss: 0-dim fp32, for the first window's backward; zero while the
            archive is empty.
          metrics: ``imitation/loss`` and ``imitation/branches``, the
            archive's fill.

        """
        archive = self._archive
        assert isinstance(archive, _Archive), "ingest the epoch's rollout first"
        count = archive.count
        # The cursor counts from the oldest branch, which sits ``count`` slots
        # before the next to be filled.
        slot = (
            (archive.inserted - count + archive.cursor % count.clamp_min(1))
            % self.capacity
        ).reshape(1)
        decoded, _, _ = policy.forward_sequence(
            archive.observations.index_select(0, slot),
            archive.initial_states.index_select(1, slot),
            torch.zeros(
                1,
                archive.actions.shape[1],
                dtype=archive.action_mask.dtype,
                device=slot.device,
            ),
            features=None
            if archive.features is None
            else archive.features.index_select(0, slot),
        )
        num_actions = archive.action_mask.shape[-1]
        loss = imitation_loss(
            decoded[0, :, :num_actions],
            archive.actions.index_select(0, slot)[0],
            archive.action_mask.index_select(0, slot)[0],
            # An empty archive has no branch: a zero length masks every step.
            torch.where(count > 0, archive.lengths.index_select(0, slot)[0], 0),
            coefficient=self.coefficient,
        )
        archive.cursor.add_(count > 0)
        return loss, {
            "imitation/loss": loss.detach(),
            "imitation/branches": count.float(),
        }

    def state_dict(self) -> dict[str, Tensor]:
        """Return the archive, its counters and the cursor; empty before any ingest.

        Returns:
          state: Each of the archive's tensors by field name, not copied;
            ``features`` only with a feature.

        """
        archive = self._archive
        if archive is None:
            return {}
        return _archived(archive)

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore a :meth:`state_dict`, copying into the archive a graph addresses.

        Args:
          state: What :meth:`state_dict` returned; an empty one empties the
            archive. It holds ``features`` exactly when the archive does.

        """
        archive = self._archive
        if not state:
            if archive is not None:
                for counter in (archive.count, archive.inserted, archive.cursor):
                    counter.zero_()
            return
        if archive is None:
            assert isinstance(self.device, torch.device), "prepare the imitation first"
            self._archive = _Archive(
                **{
                    name: value.to(self.device, copy=True)
                    for name, value in state.items()
                },
            )
            return
        for name, value in _archived(archive).items():
            value.copy_(state[name])


@dataclass(frozen=True, slots=True, kw_only=True)
class _Archive:
    """The archived branches in ring order, and the counters that place them.

    Attributes:
      observations: ``[capacity, horizon, observation_size]``.
      actions: ``[capacity, horizon]`` fp32.
      action_mask: ``[capacity, horizon, num_actions]``.
      lengths: ``[capacity]`` int64, each branch's steps.
      initial_states: ``[layers, capacity, width]``, each branch's carry at step 0.
      count: 0-dim int64, the branches held.
      inserted: 0-dim int64, the branches ever archived; the next goes to
        ``inserted % capacity``.
      cursor: 0-dim int64, the branches learned from.
      features: ``[capacity, horizon, width]`` in the stored dtype, each
        step's feature as its actor read it; None without a feature.

    """

    observations: Tensor
    actions: Tensor
    action_mask: Tensor
    lengths: Tensor
    initial_states: Tensor
    count: Tensor
    inserted: Tensor
    cursor: Tensor
    features: Tensor | None = None

    @classmethod
    def allocate(cls, rollout: LearnerRollout, *, capacity: int) -> _Archive:
        """Return an empty archive shaped like ``rollout``'s rows.

        Args:
          rollout: A rollout, agent-major: its widths, dtypes and device, and
            whether it stores a feature.
          capacity: The branches the archive holds.

        Returns:
          archive: Zeros, the counters and the cursor at zero.

        """
        states = rollout.initial_states
        counter = rollout.actions.new_zeros((), dtype=torch.int64)
        features = rollout.features
        return cls(
            observations=rollout.observations.new_zeros(
                capacity,
                *rollout.observations.shape[1:],
            ),
            actions=rollout.actions.new_zeros(capacity, rollout.actions.shape[1]),
            action_mask=rollout.action_mask.new_zeros(
                capacity,
                *rollout.action_mask.shape[1:],
            ),
            lengths=counter.new_zeros(capacity),
            initial_states=states.new_zeros(
                states.shape[0],
                capacity,
                *states.shape[2:],
            ),
            count=counter,
            inserted=counter.clone(),
            cursor=counter.clone(),
            features=None
            if features is None
            else features.new_zeros(capacity, *features.shape[1:]),
        )


def _archived(archive: _Archive) -> dict[str, Tensor]:
    """Return the archive's tensors by field name, in field order; no absent feature."""
    tensors = {
        entry.name: cast("Tensor | None", getattr(archive, entry.name))
        for entry in fields(archive)
    }
    return {name: value for name, value in tensors.items() if value is not None}
