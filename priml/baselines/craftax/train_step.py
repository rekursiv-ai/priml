"""One learner minibatch, as PufferLib's ``train_epoch_gpu`` runs it.

The rollout arrives time-major, ``[horizon, agents, ...]``; the learner
reads it agent-major, ``[agents, horizon, ...]``, with rewards scaled, then
clamped to ``[-1, 1]`` (``pufferl.cu:1472-1489``). Each of the epoch's minibatches is
``rows`` consecutive agents starting at ``(index * rows) mod agents``, so the
18 minibatches of ``exp000`` wrap past the 16 that cover the 2,048 agents
once. Every minibatch starts its recurrence from the carry the rollout began
with, never from another minibatch's (``pufferl.cu:1511-1552``).

``score_minibatch`` is the forward and the learning rule; ``learn_minibatch``
adds autograd's backward from the rule's total. :class:`AgentWindows` is
PufferLib's epoch: those minibatches, an optimizer step after each, and an
optional :class:`Auxiliary` loss that the first minibatch adds to its
backward. It fills the step's ``learner`` slot, which another epoch can fill
instead (``update.ShuffledTransitions``). ``CraftaxTrainStep`` is the epoch pipeline
around it: with two rollout slots, the asynchronous rollout one epoch ahead of
the learner; with one, a rollout collected with the current weights before
each epoch learns from it. With ``feature_training`` the learner also trains
the world model behind a feature: each window recomputes its features from
the stored contexts with the learner's copy of the weights
(``world_model.context``), and each epoch's rollout reads the copy published
into the feature's source before it starts, its rows' histories rebuilt
under it.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``src/pufferl.cu``, pin ``6ffa5b10``.

"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Protocol,
    Self,
    TypedDict,
    cast,
    override,
    runtime_checkable,
)

import math
import struct
import time

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor

import numpy as np
import torch

from priml.baselines.craftax.env import CraftaxEnv
from priml.baselines.craftax.evaluation import Evaluation
from priml.baselines.craftax.game.state import LOG_DTYPE
from priml.baselines.craftax.metric import (
    LOG_FIELDS,
    aggregate_logs,
    report_metrics,
)
from priml.baselines.craftax.model import (
    MinGRUPolicy,
    Policy,
    PolicyConfig,
)
from priml.baselines.craftax.rollout import (
    CAPTURE_LOCK,
    FeatureSource,
    PhiloxSampler,
    Rollout,
    Sampler,
)
from priml.baselines.craftax.world_model.context import (
    ContextReplay,
    Contexts,
    JointWorldModel,
)
from priml.baselines.craftax.world_model.feature import WorldModelFeature
from priml.lib.codec import from_plain
from priml.loss.policy_gradient import PPO, TorchPPO
from priml.loss.policy_gradient_kernel import TritonPPO
from priml.math.schedules import Schedule, cosine
from priml.model.min_gru import TritonScan
from priml.optimizers.fused_muon import FusedMuon
from priml.optimizers.lr import remember_initial_lrs
from priml.timer import CheckpointableStepTimer
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing_extensions import TypeIs

    from torch.optim.optimizer import StateDict as OptimizerState

    from priml.baselines.craftax.rollout import RolloutStorage
    from priml.train.custom_types import TrainStepOutput


@dataclass(frozen=True, slots=True, kw_only=True)
class LearnerRollout:
    """A rollout in the learner's layout: agent-major, rewards scaled and clamped.

    Each tensor keeps its :class:`RolloutStorage` dtype.

    Attributes:
      observations: ``[agents, horizon, observation_size]``.
      actions: ``[agents, horizon]`` fp32.
      logprobs: ``[agents, horizon]``.
      rewards: ``[agents, horizon]``, scaled and clamped.
      terminals: ``[agents, horizon]``.
      values: ``[agents, horizon]``.
      action_mask: ``[agents, horizon, num_actions]``.
      initial_states: ``[layers, agents, width]``, the carry at step 0.
      branch_starts: ``[agents]`` uint8, 1 where step 0 is a practice restore.
      features: ``[agents, horizon, width]``, the feature each step's actor
        read; None without a feature.
      contexts: Each step's world-model context, agent-major: what a
        learner that trains the feature's world model recomputes the
        features from; None otherwise.

    """

    observations: Tensor
    actions: Tensor
    logprobs: Tensor
    rewards: Tensor
    terminals: Tensor
    values: Tensor
    action_mask: Tensor
    initial_states: Tensor
    branch_starts: Tensor
    features: Tensor | None = None
    contexts: Contexts | None = None

    @classmethod
    def from_time_major(  # noqa: PLR0917 -- One argument per rollout buffer, as PufferLib stores them.
        cls,
        observations: Tensor,
        actions: Tensor,
        logprobs: Tensor,
        rewards: Tensor,
        terminals: Tensor,
        values: Tensor,
        action_mask: Tensor,
        initial_states: Tensor,
        branch_starts: Tensor,
        *,
        reward_scale: float,
        reward_clip: float,
        features: Tensor | None = None,
        contexts: Contexts | None = None,
    ) -> LearnerRollout:
        """Transpose the time-major buffers; scale, then clamp, the rewards.

        Args:
          observations: ``[horizon, agents, observation_size]``.
          actions: ``[horizon, agents]``.
          logprobs: ``[horizon, agents]``.
          rewards: ``[horizon, agents]``.
          terminals: ``[horizon, agents]``.
          values: ``[horizon, agents]``.
          action_mask: ``[horizon, agents, num_actions]``.
          initial_states: ``[layers, agents, width]``; already agent-major.
          branch_starts: ``[agents]``; already agent-indexed.
          reward_scale: Rewards are multiplied by it first.
          reward_clip: Then clamped to ``[-reward_clip, reward_clip]``.
          features: ``[horizon, agents, width]``; None without a feature.
          contexts: Each step's world-model context, already agent-major;
            None unless the learner trains the feature's world model.

        Returns:
          rollout: The agent-major copy.

        """
        return cls(
            observations=observations.transpose(0, 1).contiguous(),
            actions=actions.transpose(0, 1).contiguous(),
            logprobs=logprobs.transpose(0, 1).contiguous(),
            # Contiguous first: the product and the clamp keep their input's
            # strides. In bf16 the clamp is PufferLib's fp32 clamp then store:
            # rounding is monotone, so the two commute.
            rewards=(rewards.transpose(0, 1).contiguous() * reward_scale).clamp(
                -reward_clip,
                reward_clip,
            ),
            terminals=terminals.transpose(0, 1).contiguous(),
            values=values.transpose(0, 1).contiguous(),
            action_mask=action_mask.transpose(0, 1).contiguous(),
            initial_states=initial_states,
            branch_starts=branch_starts,
            features=None
            if features is None
            else features.transpose(0, 1).contiguous(),
            contexts=contexts,
        )

    def minibatch(self, offset: int, rows: int) -> LearnerRollout:
        """Return the views of ``rows`` agents from ``offset``.

        Args:
          offset: The first agent.
          rows: Agents per minibatch.

        Returns:
          minibatch: Views into this rollout, so nothing is copied.

        """
        return LearnerRollout(
            observations=self.observations[offset : offset + rows],
            actions=self.actions[offset : offset + rows],
            logprobs=self.logprobs[offset : offset + rows],
            rewards=self.rewards[offset : offset + rows],
            terminals=self.terminals[offset : offset + rows],
            values=self.values[offset : offset + rows],
            action_mask=self.action_mask[offset : offset + rows],
            initial_states=self.initial_states[:, offset : offset + rows],
            branch_starts=self.branch_starts[offset : offset + rows],
            features=None
            if self.features is None
            else self.features[offset : offset + rows],
            contexts=None
            if self.contexts is None
            else self.contexts.rows(offset, rows),
        )


def minibatch_offsets(*, agents: int, rows: int, count: int) -> list[int]:
    """Return PufferLib's minibatch starts: ``(index * rows) mod agents``.

    Args:
      agents: Agents in the rollout.
      rows: Agents per minibatch; PufferLib's ``minibatch_size / horizon``.
      count: Minibatches per epoch; PufferLib's ``int(replay_ratio * agents *
        horizon / minibatch_size)``.

    Returns:
      offsets: One per minibatch, in learning order.

    """
    return [(index * rows) % agents for index in range(count)]


def minibatch_count(
    *,
    agents: int,
    horizon: int,
    minibatch_size: int,
    replay_ratio: float,
) -> int:
    """Return PufferLib's minibatch count, truncated as its ``int`` arithmetic is.

    Args:
      agents: Agents in the rollout.
      horizon: Steps per rollout.
      minibatch_size: Transitions per minibatch.
      replay_ratio: Passes over the rollout; 1.17 gives 18 of 16.

    Returns:
      count: ``int(replay_ratio * agents * horizon / minibatch_size)``.

    """
    return int(replay_ratio * (agents * horizon) / minibatch_size)


def score_minibatch(
    policy: Policy,
    objective: PPO,
    minibatch: LearnerRollout,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run one minibatch's forward and learning rule.

    Args:
      policy: The live policy.
      objective: The learning rule, with its coefficients.
      minibatch: ``rows`` agents of the rollout.

    Returns:
      total: The rule's total plus the policy's auxiliary loss, 0-dim;
        autograd differentiates it when the policy's parameters require
        gradients and grad mode is on.
      losses: ``[8]`` fp32, the rule's summed terms.
      auxiliary_loss: 0-dim fp32, the policy's own loss inside ``total``.

    """
    decoded, _, auxiliary_loss = policy.forward_sequence(
        minibatch.observations,
        minibatch.initial_states,
        minibatch.terminals,
        actions=minibatch.actions,
        features=minibatch.features,
    )
    total, losses = objective(
        decoded,
        actions=minibatch.actions,
        action_mask=minibatch.action_mask,
        old_logprobs=minibatch.logprobs,
        rewards=minibatch.rewards,
        terminals=minibatch.terminals,
        values=minibatch.values,
    )
    return total + auxiliary_loss, losses, auxiliary_loss


def learn_minibatch(
    policy: Policy,
    objective: PPO,
    minibatch: LearnerRollout,
    *,
    extra_loss: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Run one minibatch's forward, learning rule and backward.

    Autograd adds each weight's gradient to its ``grad``, which is the
    gradient itself when the ``grad`` is None, as ``zero_grad`` leaves it.

    Args:
      policy: The live policy.
      objective: The learning rule, with its coefficients.
      minibatch: ``rows`` agents of the rollout.
      extra_loss: A 0-dim loss from outside the minibatch, added to its total
        before the one backward; None adds nothing.

    Returns:
      losses: ``[8]`` fp32, the summed terms.
      auxiliary_loss: 0-dim fp32, the policy's own loss, detached.

    """
    total, losses, auxiliary_loss = score_minibatch(policy, objective, minibatch)
    (total if extra_loss is None else total + extra_loss).backward()
    return losses, auxiliary_loss.detach()


def learn_joint_minibatch(
    policy: Policy,
    objective: PPO,
    minibatch: LearnerRollout,
    world_model: JointWorldModel,
    *,
    extra_loss: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Learn one minibatch whose features the trained world model computes now.

    The two-phase gradient of ``world_model.context``: the features are
    recomputed from the minibatch's contexts without a graph, the policy's
    backward (:func:`learn_minibatch`) reads them as a leaf and leaves their
    gradient there, and the world model's backward takes it into its weights.
    Every gradient is added to its weight's ``grad``, as autograd adds it.

    Args:
      policy: The live policy.
      objective: The learning rule, with its coefficients.
      minibatch: ``rows`` agents of the rollout, with their contexts; the
        stored features are only compared with.
      world_model: The trained copy of the feature's world model.
      extra_loss: A 0-dim loss from outside the minibatch, added to the
        policy's total before its backward; None adds nothing.

    Returns:
      losses: ``[8]`` fp32, the summed terms.
      auxiliary_loss: 0-dim fp32, the policy's own loss, detached.
      feature_gap: 0-dim fp32, ``|F - F_stored| / |F_stored|`` over the
        window: the recomputed features against those the actor read,
        rounding alone when the actor ran the same weights.

    Raises:
      ValueError: The minibatch carries no contexts or no stored features,
        or the policy's loss does not read the features.

    """
    stored = minibatch.features
    if minibatch.contexts is None:
        raise ValueError(
            "training the feature's world model needs each step's stored context",
        )
    if stored is None:
        raise ValueError(
            "training the feature's world model compares the features it "
            "recomputes with the stored ones, and none are stored",
        )
    replay = world_model.forward(minibatch.contexts)
    difference = (replay.features.float() - stored.float()).norm()
    feature_gap = difference / stored.float().norm()
    features = replay.features.detach().requires_grad_()
    losses, auxiliary_loss = learn_minibatch(
        policy,
        objective,
        replace(minibatch, features=features),
        extra_loss=extra_loss,
    )
    gradient = features.grad
    if gradient is None:
        raise ValueError(
            "the policy's loss does not read the features, so the world model "
            "has no gradient to learn from",
        )
    world_model.backward(replay, gradient)
    return losses, auxiliary_loss, feature_gap


class Learner(Protocol):
    """How one epoch learns from its rollout: its rule, minibatches and optimizer steps."""

    def prepare(self, config: CraftaxTrainStep.Config) -> None:
        """Check the step's recipe against this learner and take its geometry.

        The step calls it before it builds its policy, optimizer or
        environments, so a recipe the learner cannot run fails at once.

        Args:
          config: The step's recipe: its environments, rollout and horizon.

        """
        ...

    def begin_epoch(self, step: CraftaxTrainStep, epoch: int) -> None:
        """Do epoch ``epoch``'s host work, before its minibatches run.

        The minibatches may run as a replayed CUDA graph, so anything that
        must change between epochs is written here, into tensors they read.

        Args:
          step: The train step, for its geometry and device.
          epoch: The epoch's index, counted across resumes.

        """
        ...

    def __call__(
        self,
        step: CraftaxTrainStep,
        rollout: LearnerRollout,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Run the epoch's minibatches on ``step``'s policy and optimizer.

        Args:
          step: The train step: its policy, optimizer, objective and geometry.
          rollout: The epoch's rollout, agent-major.

        Returns:
          losses: ``[8]`` fp32, the mean of the minibatches' terms.
          metrics: Any further 0-dim values the epoch reports, by name.

        """
        ...

    def loss(
        self,
        step: CraftaxTrainStep,
        rollout: LearnerRollout,
    ) -> tuple[Tensor, Tensor]:
        """Score the rollout's first minibatch by the learner's rule, without updating.

        Args:
          step: The train step: its policy and geometry.
          rollout: The rollout, agent-major.

        Returns:
          total: The minibatch's total loss, 0-dim.
          losses: ``[8]`` fp32, its terms.

        """
        ...

    def state_dict(self) -> dict[str, Tensor]:
        """Return what the learner carries from epoch to epoch; the live tensors."""
        ...

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore a :meth:`state_dict`, copying into any tensor a graph addresses."""
        ...


class Auxiliary(Protocol):
    """A loss beside the learning rule's, fed each epoch's rollout.

    :class:`AgentWindows` calls :meth:`ingest` then :meth:`loss` once per
    epoch, before its first window, and adds the loss to that window's
    backward. Both run inside the epoch's CUDA graph, so they keep static
    shapes, keep their state in device tensors and never read one back to the
    host.
    """

    def prepare(self, config: CraftaxTrainStep.Config) -> None:
        """Check the step's recipe against this loss, before anything is built."""
        ...

    def ingest(self, rollout: LearnerRollout) -> None:
        """Take in what the loss learns from out of the epoch's rollout."""
        ...

    def loss(self, policy: Policy) -> tuple[Tensor, dict[str, Tensor]]:
        """Return the loss for the first window, 0-dim fp32, and metrics by name."""
        ...

    def state_dict(self) -> dict[str, Tensor]:
        """Return the state a resumed run needs; the live tensors."""
        ...

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore a :meth:`state_dict`."""
        ...


class AgentWindows:
    """PufferLib's epoch: windows of whole agents, each scored by its objective.

    The windows are ``rows`` consecutive agents from ``(index * rows) mod
    agents`` (:func:`minibatch_offsets`), the whole horizon each; an optimizer
    step follows each window's ``learn_minibatch``. An :class:`Auxiliary`
    ingests the epoch's rollout first, and its loss joins the first window's
    backward. When the step trains its feature's world model (``step.joint``),
    each window learns by :func:`learn_joint_minibatch` instead, from features
    recomputed with the weights of the moment.
    """

    class Config(Fig["AgentWindows"]):
        """The learning rule, the size and count of the windows, and an auxiliary loss."""

        objective: Makeable[PPO] = field(default_factory=TritonPPO.Config)
        """The learning rule and its coefficients, which score each window."""

        minibatch_size: int = 8_192
        """Transitions per window, whole agents' horizons; PufferLib's
        ``train.minibatch_size``."""

        replay_ratio: float = 1.0
        """Windows per epoch as a fraction of the rollout's size; PufferLib's
        ``train.replay_ratio``."""

        auxiliary: Makeable[Auxiliary] | None = None
        """A loss the first window adds to its backward, fed each epoch's
        rollout; None learns the rule alone."""

    def __init__(self, config: Config) -> None:
        """Build the rule and the auxiliary loss.

        Args:
          config: The rule, the windows' size and count, and the auxiliary.

        Raises:
          ValueError: ``replay_ratio`` is not positive and finite, or the
            rule or the auxiliary refuses a coefficient.

        """
        replay_ratio = config.replay_ratio
        if math.isnan(replay_ratio) or math.isinf(replay_ratio) or replay_ratio <= 0:
            raise ValueError(
                f"replay_ratio must be positive and finite, not {replay_ratio}",
            )
        self.config = config
        self.objective = config.objective.make()
        self.auxiliary = None if config.auxiliary is None else config.auxiliary.make()
        self.rows = 0
        """Agents per window, from :meth:`prepare`."""
        self.offsets: list[int] = []
        """Each window's first agent, in learning order, from :meth:`prepare`."""

    def prepare(self, config: CraftaxTrainStep.Config) -> None:
        """Place the windows on the step's rollout.

        Args:
          config: The step's recipe.

        Raises:
          ValueError: The windows do not tile the rollout, the ratio leaves
            none, or the rule or the auxiliary cannot learn from the rollout.

        """
        horizon = config.rollout.horizon
        agents = config.env.num_envs
        size = self.config.minibatch_size
        if size <= 0 or size % horizon:
            raise ValueError(
                "minibatch_size must be a positive multiple of the horizon",
            )
        rows = size // horizon
        # PufferLib reads ``rows`` agents from each start unwrapped, past the rollout's
        # end when they do not tile it (``pufferl.cu:1512-1520``).
        if agents % rows:
            msg = f"a minibatch of {rows} agents must tile the {agents} environments"
            raise ValueError(msg)
        count = minibatch_count(
            agents=agents,
            horizon=horizon,
            minibatch_size=size,
            replay_ratio=self.config.replay_ratio,
        )
        if count < 1:
            raise ValueError(
                f"replay_ratio {self.config.replay_ratio} leaves no minibatch per epoch",
            )
        self.objective.check_horizon(horizon)
        if self.auxiliary is not None:
            self.auxiliary.prepare(config)
        self.rows = rows
        self.offsets = minibatch_offsets(agents=agents, rows=rows, count=count)

    def begin_epoch(self, step: CraftaxTrainStep, epoch: int) -> None:
        """Nothing to prepare: the windows are the same every epoch."""
        del step, epoch

    def __call__(
        self,
        step: CraftaxTrainStep,
        rollout: LearnerRollout,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Learn from each window, then step.

        Args:
          step: The train step: its policy, optimizer and device.
          rollout: The epoch's rollout, agent-major.

        Returns:
          losses: ``[8]`` fp32, the windows' mean terms.
          metrics: ``auxiliary_loss``, the windows' mean of the policy's own
            loss, and the auxiliary's metrics; when the step trains its
            feature's world model, ``joint/feature_gap``, the first window's
            (:func:`learn_joint_minibatch`).

        """
        total = torch.zeros(
            len(TorchPPO.Config.LOSS_NAMES),
            dtype=torch.float32,
            device=step.device,
        )
        policy_auxiliary = torch.zeros((), dtype=torch.float32, device=step.device)
        extra_loss: Tensor | None = None
        metrics: dict[str, Tensor] = {}
        if self.auxiliary is not None:
            self.auxiliary.ingest(rollout)
            extra_loss, metrics = self.auxiliary.loss(step.model)
        for offset in self.offsets:
            # None, not zeros: the backward then hands each weight its gradient
            # as computed, with no add onto a zeroed buffer.
            step.optimizer.zero_grad(set_to_none=True)
            minibatch = rollout.minibatch(offset, self.rows)
            if step.joint is None:
                losses, auxiliary_loss = learn_minibatch(
                    step.model,
                    self.objective,
                    minibatch,
                    extra_loss=extra_loss,
                )
            else:
                losses, auxiliary_loss, feature_gap = learn_joint_minibatch(
                    step.model,
                    self.objective,
                    minibatch,
                    step.joint,
                    extra_loss=extra_loss,
                )
                # The first window learns from the epoch's starting weights; the
                # later ones from weights its own steps have moved.
                metrics.setdefault("joint/feature_gap", feature_gap)
            # The first window alone learns the auxiliary's loss.
            extra_loss = None
            total += losses
            policy_auxiliary += auxiliary_loss
            step.optimizer.step()
        count = len(self.offsets)
        return total / count, {"auxiliary_loss": policy_auxiliary / count, **metrics}

    def loss(
        self,
        step: CraftaxTrainStep,
        rollout: LearnerRollout,
    ) -> tuple[Tensor, Tensor]:
        """Score the first window by the rule without the backward or the auxiliary.

        A step that trains its feature's world model scores the features its
        current weights compute, as its windows learn from them.

        Args:
          step: The train step: its policy, and the world model it trains.
          rollout: The rollout, agent-major.

        Returns:
          total: The window's total, the policy's own loss included, 0-dim.
          losses: ``[8]`` fp32, the rule's terms.

        Raises:
          ValueError: The step trains its world model but the rollout
            carries no contexts.

        """
        minibatch = rollout.minibatch(self.offsets[0], self.rows)
        if step.joint is not None:
            if minibatch.contexts is None:
                raise ValueError(
                    "training the feature's world model needs each step's stored "
                    "context",
                )
            features = step.joint.forward(minibatch.contexts).features
            minibatch = replace(minibatch, features=features)
        total, losses, _ = score_minibatch(step.model, self.objective, minibatch)
        return total, losses

    def state_dict(self) -> dict[str, Tensor]:
        """Return the auxiliary's state; the windows keep none."""
        return {} if self.auxiliary is None else self.auxiliary.state_dict()

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore the auxiliary's state.

        Args:
          state: What :meth:`state_dict` returned.

        Raises:
          ValueError: There is state but no auxiliary to take it: the
            checkpoint is another recipe's.

        """
        if self.auxiliary is not None:
            self.auxiliary.load_state_dict(state)
        elif state:
            msg = f"windows without an auxiliary carry no state, not {sorted(state)}"
            raise ValueError(msg)


class RateSchedule(Protocol):
    """An epoch's learning rate, from the rate the optimizer was configured with."""

    def __call__(self, base: float, *, step: int, total_steps: int) -> float:
        """Return the rate for epoch ``step`` of ``total_steps``."""
        ...


class ProgressSchedule:
    """One of priml's progress schedules, scaling the configured rate each epoch."""

    class Config(Fig["ProgressSchedule"]):
        """The curve."""

        curve: Makeable[Schedule[float]] = field(
            default_factory=lambda: PartialConfig(cosine),
        )
        """Maps the run's progress, ``step / total_steps``, to the rate's multiplier."""

    def __init__(self, config: Config) -> None:
        self.curve = config.curve.make()

    def __call__(self, base: float, *, step: int, total_steps: int) -> float:
        """Return ``base`` times the curve at ``step / total_steps``."""
        return base * self.curve(step / total_steps)


def cosine_annealing_fp32(
    base: float,
    minimum: float,
    step: int,
    total_steps: int,
) -> float:
    """Return PufferLib's cosine-annealed rate at ``step``, in its fp32 arithmetic.

    ``u = step / total_steps`` in double and ``cos(pi * u)`` from the
    platform's libm, rounded to fp32; the rest is fp32, in this order:
    ``minimum + (0.5 * (base - minimum)) * (1 + cos)``, as PufferLib's C holds
    its rates as ``float``. It exists only to match PufferLib's bits: a
    :class:`ProgressSchedule` over priml's ``cosine`` computes in double and
    multiplies once, and lands on a different fp32 rate at 2,836 of exp000's
    6,663 epoch boundaries, the first at epoch 5.

    Args:
      base: The rate at step 0; rounded to fp32 first.
      minimum: The rate at ``total_steps``; rounded to fp32 first.
      step: The current step.
      total_steps: The step at which the rate reaches ``minimum``.

    Returns:
      lr: The fp32 rate, exact as a Python float.

    References:
      https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``cosine_annealing`` in
        ``src/pufferl.cu``, pin ``6ffa5b10``.

    """
    cosine_fp32 = _fp32(math.cos(math.pi * (step / total_steps)))
    span = _fp32(_fp32(base) - _fp32(minimum))
    return _fp32(_fp32(minimum) + _fp32(_fp32(0.5 * span) * _fp32(1.0 + cosine_fp32)))


@runtime_checkable
class MasterWeights(Protocol):
    """An optimizer that steps fp32 masters behind lower-precision parameters."""

    @property
    def master_weights(self) -> list[Tensor]:
        """The live masters, one per parameter in its groups' order."""
        ...


def load_masters(
    model: Policy,
    optimizer: torch.optim.Optimizer,
    path: Path,
    *,
    others: Sequence[Tensor] = (),
) -> None:
    """Start a policy and its optimizer from a ``state_dict`` of fp32 masters.

    The parameters take the masters rounded to their dtype; the optimizer's
    masters take them exactly, in place, before its first step.

    Args:
      model: The policy, on its device.
      optimizer: Its optimizer, which keeps fp32 masters.
      path: The masters by parameter name, as ``torch.save`` wrote them.
      others: Parameters the optimizer holds beside the policy's, a trained
        world model's, which keep the masters they start from.

    Raises:
      TypeError: The optimizer keeps no masters (:class:`MasterWeights`).
      ValueError: The optimizer holds a parameter that is neither the
        policy's nor among ``others``.

    """
    if not isinstance(optimizer, MasterWeights):
        raise TypeError(
            "a checkpoint of fp32 masters needs an optimizer that keeps "
            "masters (MasterWeights)",
        )
    masters = cast(
        "dict[str, Tensor]",
        torch.load(path, weights_only=True, map_location="cpu"),
    )
    model.load_state_dict(masters)
    names = {id(weight): name for name, weight in model.named_parameters()}
    kept = {id(weight) for weight in others}
    with torch.no_grad():
        # ``master_weights`` follows the optimizer's own group order.
        for parameter, master in zip(
            (
                parameter
                for group in optimizer.param_groups
                for parameter in cast("list[Tensor]", group["params"])
            ),
            optimizer.master_weights,
            strict=True,
        ):
            name = names.get(id(parameter))
            if name is not None:
                master.copy_(masters[name])
            elif id(parameter) not in kept:
                raise ValueError(
                    "the optimizer holds a parameter the policy does not name "
                    "and no other owner claims",
                )


def _fused_muon() -> Makeable[Callable[..., torch.optim.Optimizer]]:
    """Return :class:`FusedMuon` at PufferLib's ``default.ini`` values."""
    return FusedMuon.Config()


def _triton_policy() -> MinGRUPolicy.Config:
    """Return the policy config with the Triton scan, the production choice."""
    config = MinGRUPolicy.Config()
    config.block.scan = TritonScan.Config()
    return config


class CraftaxTrainStep:
    """PufferLib's training loop, one ``train_step`` per epoch (``pufferl.cu:3071``).

    The rollout runs one epoch AHEAD of the learner: a boot rollout fills slot
    0 with the initial weights; each epoch then starts the next rollout into
    the other slot (from a copy of the learner's current weights that the
    step graphs own, ``rollout_start``, ``pufferl.cu:2457``), trains on the
    slot collected during the previous epoch, and waits for the rollout
    before the slots swap. The learner therefore always trains on data one
    epoch stale, and the last epoch does not prefetch. With one slot there is
    nothing to prefetch into: each epoch collects its rollout with the current
    weights, then learns from it, as synchronous PPO does. Per epoch the
    learning rate is the schedule's over ``train_budget_steps``, and ``learner``
    runs the minibatches and their optimizer steps.

    It implements the training loop's step protocol directly rather than
    extending priml's supervised ``TrainStep``. It replaces that step's update
    whole, so the supervised knobs -- the loss, the rate multiplier, clipping,
    accumulation, autocast, compile, EMA -- would be fields nothing reads.
    """

    class Config(Fig["CraftaxTrainStep"]):
        """The policy, its optimizer and schedule, the environments and the learning rule."""

        model: PolicyConfig = field(default_factory=_triton_policy)
        """The learner's policy; the rollout runs a copy of its weights."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=_fused_muon,
        )
        """Builds the optimizer from the policy's ``parameters()``, in their
        order. Each group's ``lr`` is the schedule's base; the step replaces it
        with a device tensor refilled every epoch, so a captured learner graph
        reads the rate live."""

        schedule: Makeable[RateSchedule] = field(
            default_factory=ProgressSchedule.Config,
        )
        """Each epoch's rate over ``train_budget_steps``; priml's cosine to zero
        unless set."""

        learner: Makeable[Learner] = field(default_factory=AgentWindows.Config)
        """How each epoch learns from its rollout -- its rule, minibatches and
        optimizer steps: PufferLib's windows of whole agents unless set."""

        env: CraftaxEnv.Config = field(default_factory=CraftaxEnv.Config)
        """The training environments."""

        sampler: Makeable[Sampler] = field(default_factory=PhiloxSampler.Config)
        """The action streams."""

        rollout: Rollout.Config = field(default_factory=Rollout.Config)
        """The two slots and the horizon."""

        feature: Makeable[FeatureSource] | None = None
        """A per-step feature the actor computes from each observation and its
        history and hands the policy beside it, e.g. a world model's state;
        the rollout stores it for the learner, and the evaluation reads the
        same source. Frozen unless ``feature_training`` trains it; None
        computes none."""

        feature_training: ContextReplay.Config | None = None
        """Train the feature's world model with the policy. The learner trains a
        copy of its weights, which join the optimizer after the policy's in one
        group, so every matrix steps under the same rule, rate and global clip;
        each window recomputes its features from the stored contexts with the
        copy (``world_model.context``), and the learner runs eagerly. Each
        rollout and evaluation reads the copy, published into the source; a
        rollout after the learner has trained it first rebuilds its rows'
        histories under it. ``finalize`` makes the feature ``joint`` exactly
        when this is set, so its rollouts store what the learner recomputes
        the features from, and only then. A
        ``BranchImitation`` auxiliary still reads the stored features,
        detached, so its loss trains the policy alone. None keeps the feature
        frozen."""

        evaluation: Evaluation.Config = field(default_factory=Evaluation.Config)
        """The fresh trainer an evaluation plays in; ``finalize`` fills the
        parts it leaves unset from training's."""

        checkpoint: Path | str | None = None
        """A ``state_dict`` of fp32 masters to start from: loaded into the
        parameters, rounded to their dtype, and exactly into the optimizer's
        masters, so the optimizer must keep them (:class:`MasterWeights`)."""

        reward_scale: float = 1.0
        """The learner's rewards are the stored ones times this, before the
        clamp; the environments, their returns and the evaluation keep theirs.
        It multiplies the rollout's stored dtype, so bf16 storage rounds twice
        unless it is a power of two."""

        reward_clip: float = 1.0
        """Rewards are clamped to ``[-reward_clip, reward_clip]`` for the learner;
        ``math.inf`` leaves them as they are."""

        train_budget_steps: float = math.inf
        """Epochs the run trains, the schedule's horizon and the last prefetch;
        it must be set, since the pipeline needs to know when to stop."""

        parallelism: NoParallel.Config = field(default_factory=NoParallel.Config)
        """The one device the policy, the rollout and the learner run on."""

        @override
        def finalize(self) -> Self:
            if isinstance(self.feature, WorldModelFeature.Config):
                # The learner recomputes the features from what a joint
                # source's rollout stores, and publishes into its model; a
                # joint source no learner trains would store it for nothing.
                self.feature.joint = self.feature_training is not None
            if self.evaluation.env is None:
                # The evaluation plays training's rules, less its training-only
                # options: no stall cap and no practice.
                env = self.evaluation.env = self.env.copy_tree()
                env.stall_cap = None
                env.practice = None
            if self.evaluation.sampler is None:
                self.evaluation.sampler = self.sampler.copy_tree()
            if self.evaluation.rollout is None:
                rollout = self.evaluation.rollout = self.rollout.copy_tree()
                rollout.num_slots = 1
                rollout.bootstrap = False
            return super().finalize()

    # A base, not ``NotRequired``: under postponed annotations ``TypedDict``
    # counts a ``NotRequired`` key among the required ones, and a checkpoint
    # of a frozen feature would be refused as missing it.
    class _Parts(TypedDict):
        """What every checkpoint of the step carries."""

        model: dict[str, Tensor]
        optimizer: OptimizerState
        learner: dict[str, Tensor]
        timer_step: CheckpointableStepTimer.StateDict
        env: dict[str, Tensor]
        rollout: dict[str, Tensor]
        ready: int
        booted: bool

    class StateDict(_Parts, total=False):
        """The learner's checkpoint and the pipeline's; see :meth:`CraftaxTrainStep.state_dict`."""

        world_model: dict[str, Tensor]
        """The trained world model's weights, when the step trains one."""

    def __init__(self, config: Config) -> None:
        """Build the learner, its optimizer, the environments, the actor and the rollout.

        Every setting is checked before the policy, the optimizer and the
        environments are built: the recipe's scalars, the geometry, the
        observation width the env writes and the policy reads, the
        evaluation, and the learner's own recipe.

        Args:
          config: The recipe.

        Raises:
          ValueError: A setting is out of range, the geometry does not tile,
            the run has no epoch count, the evaluation's config would be
            refused, the policy's first stage does not read the env's layout,
            a feature is asked of the symbolic view, ``feature_training``
            has no world-model feature to train or no ``AgentWindows`` to
            learn with, or the learner refuses the recipe.
          TypeError: A checkpoint is set but the optimizer keeps no masters.

        """
        _check_recipe(config)
        self.config = config
        self.learner = config.learner.make()
        self.learner.prepare(config)
        self.device: torch.device = config.parallelism.make().device
        self.model = config.model.make().to(self.device)
        self.schedule = config.schedule.make()
        self.total_steps = int(config.train_budget_steps)
        self.timer_step = CheckpointableStepTimer()
        """Epochs trained: how many, and how long the learner took."""
        self.env = config.env.make()
        self.env.reset()
        self._logs_read = _log_sums(self.env)
        """The env's episode-log sums at the last report; each epoch reports the change."""
        self.actor = config.model.make().to(self.device)
        """The rollout's copy of the weights; the step graphs address its parameters."""
        self.feature = None if config.feature is None else config.feature.make()
        """The actor's per-step feature, which the evaluation shares; None without one."""
        self.rollout = Rollout(
            config.rollout,
            policy=self.actor,
            sampler=config.sampler.make(),
            env=self.env,
            device=self.device,
            feature=self.feature,
        )
        self.joint = _joint_world_model(config, self.feature)
        """The learner's trained copy of the feature's world model; None while
        the feature is frozen."""
        # After the rollout, whose engines put the source's world model on the
        # device in the actor's dtype: a trained copy joins the optimizer from there.
        self.optimizer: torch.optim.Optimizer = config.optimizer.make()(
            [
                *self.model.parameters(),
                *([] if self.joint is None else self.joint.parameters()),
            ],
        )
        # The schedule's base: each group's configured rate, before any epoch
        # overwrites ``lr``.
        remember_initial_lrs([self.optimizer])
        if config.checkpoint is not None:
            load_masters(
                self.model,
                self.optimizer,
                Path(config.checkpoint),
                others=[] if self.joint is None else self.joint.parameters(),
            )
        self.ready = 0
        self.write = 1
        self._booted = False
        self._prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rollout")
        self._pending: Future[RolloutStorage] | None = None
        self._rollout_started = 0.0
        self.rollout_seconds = 0.0
        """Wall time of the last rollout, from its launch to its completion."""
        self.rebuild_seconds = 0.0
        """Wall time of the rebuild of the actor's feature histories before the
        last rollout; 0 when it needed none."""
        self._stale = False
        """Whether the learner has trained the world model since the actor's
        feature histories were last built."""
        cuda = self.device.type == "cuda"
        self._stream = torch.cuda.Stream(device=self.device) if cuda else None
        """The learner's stream: its first epoch warms it, the captures run on it."""
        self._rates = [
            torch.zeros((), dtype=torch.float32, device=self.device)
            for _ in self.optimizer.param_groups
        ]
        """Each group's rate as the optimizer reads it, refilled before every epoch."""
        self._graphs: dict[int, _LearnerGraph] = {}
        self._pool = torch.cuda.graph_pool_handle() if cuda else None
        """The learner graphs' shared memory pool: they never replay at once."""
        self._warm = False

    @property
    def global_step(self) -> int:
        """Epochs trained across the whole run, resumes included."""
        return self.timer_step.global_count

    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        """Pass the loop's tick through: the data comes from the environments."""
        return batch

    def train_step(self, **batch: object) -> TrainStepOutput:
        """Run one epoch: prefetch the next rollout, learn from the ready slot.

        Args:
          **batch: Ignored; the loop's tick.

        Returns:
          result: The epoch's mean loss terms (``TorchPPO.Config.LOSS_NAMES``) as ``model``,
            the total as ``loss``; the metrics add the rate, the timings, the
            learner's own metrics and the environments' practice metrics.

        Raises:
          RuntimeError: The run's epochs are spent.

        """
        del batch
        if self.global_step >= self.total_steps:
            msg = (
                f"the run's {self.total_steps} epochs are spent: past "
                "train_budget_steps every epoch would relearn the last rollout"
            )
            raise RuntimeError(msg)
        synchronous = self.config.rollout.num_slots == 1
        if not synchronous:
            self._boot()
        started = time.perf_counter()
        if synchronous:
            # One slot leaves nothing to prefetch into: this epoch's rollout is
            # collected now, with the weights the epoch then trains.
            self._rollout_start(self.ready)
            self._rollout_finish()
        # A synchronous rollout is ``rollout_seconds``'; a prefetch's launch is
        # the learner's, as it always was.
        learning = time.perf_counter() if synchronous else started
        prefetch = not synchronous and self.global_step + 1 < self.total_steps
        if prefetch:
            self._rollout_start(self.write)
        learner_stream = (
            nullcontext() if self._stream is None else torch.cuda.stream(self._stream)
        )
        with self.timer_step, learner_stream:
            losses, learner_metrics, learning_rate = self._train_epoch(self.ready)
            if self._stream is not None:
                # The learner's own stream only: a device-wide sync would wait
                # for the rollout in flight and report the epoch instead.
                self._stream.synchronize()
            learner_seconds = time.perf_counter() - learning
        if prefetch:
            self._rollout_finish()
            self.ready, self.write = self.write, self.ready
        metrics: dict[str, float | Tensor] = {
            name: losses[index] for index, name in enumerate(TorchPPO.Config.LOSS_NAMES)
        }
        metrics["learning_rate"] = learning_rate
        # PufferLib's own key: transitions trained so far (``pufferl.cu`` logs it).
        metrics["agent_steps"] = float(
            self.global_step * self.env.num_envs * self.config.rollout.horizon,
        )
        # PufferLib's epoch wall is its rollout's: the learner hides under the
        # next rollout. These three say whether the port's does.
        epoch_seconds = time.perf_counter() - started
        metrics["epoch_seconds"] = epoch_seconds
        # PufferLib's SPS: the epoch's transitions over its wall.
        metrics["transitions_per_second"] = (
            self.env.num_envs * self.config.rollout.horizon / epoch_seconds
        )
        metrics["learner_seconds"] = learner_seconds
        # The last pipelined epoch starts no rollout, so it has none to time.
        metrics["rollout_seconds"] = (
            self.rollout_seconds if prefetch or synchronous else 0.0
        )
        if self.joint is not None:
            # The rebuild before the rollout, outside its wall.
            metrics["rebuild_seconds"] = (
                self.rebuild_seconds if prefetch or synchronous else 0.0
            )
        metrics.update(learner_metrics)
        # Read with no rollout in flight: the last one has finished above.
        metrics.update(self.env.practice_metrics())
        metrics.update(self.rollout.feature_metrics())
        metrics.update(self._episode_metrics())
        return {
            "loss": losses[TorchPPO.Config.LOSS_NAMES.index("total_loss")],
            "model": losses,
            "metrics": metrics,
        }

    def train_loss(self, **batch: object) -> TrainStepOutput:
        """Score the ready slot's first minibatch by the learner's rule, without updating.

        Args:
          **batch: Ignored.

        Returns:
          result: The minibatch's loss terms; the total as ``loss``.

        """
        del batch
        self._boot()
        with torch.no_grad():
            total, losses = self.learner.loss(
                self,
                self._learner_rollout(self.rollout.slots[self.ready]),
            )
        return {"loss": total, "model": losses}

    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Score as :meth:`train_loss`; the policy has no evaluation mode.

        Args:
          **batch: Ignored.

        Returns:
          result: As :meth:`train_loss`.

        """
        return self.train_loss(**batch)

    def call_eval(self, *args: object, **kwargs: object) -> Tensor:
        """Score observations from a zero carry.

        Args:
          *args: The observations, positionally.
          **kwargs: Or as ``observation``.

        Returns:
          logits: ``[batch, num_actions]``.

        """
        observation = args[0] if args else kwargs["observation"]
        assert isinstance(observation, Tensor)
        batch = observation.shape[0]
        with torch.no_grad():
            decoded, _ = self.model.forward_fused(
                observation.to(self.device),
                self.model.initial_state(batch, device=self.device),
                None,
            )
        return decoded[:, :-1]

    def make_evaluator(self) -> Evaluation:
        """Build a fresh evaluation trainer around the learner's weights.

        Returns:
          evaluation: New environments, streams and carry; the policy and the
            feature source's weights shared, the source holding the learner's
            trained world model when it trains one. No learner reads the
            evaluation's rollout, so a joint source's is frozen: it keeps no
            frames and stores no learner's inputs.

        """
        feature = self.feature
        if self.joint is not None:
            self.joint.publish()
            assert isinstance(feature, WorldModelFeature)
            feature = feature.frozen()
        return Evaluation(
            self.config.evaluation,
            policy=self.model,
            device=self.device,
            feature=feature,
        )

    def on_epoch_end(self) -> None:
        """Nothing to flush: every update completes within one step."""

    def state_dict(self) -> StateDict:
        """Return the learner's checkpoint and the pipeline's, for a bit-equal resume.

        Between two ``train_step`` calls no rollout is in flight. The ready
        slot holds the rollout the next epoch learns from, collected with the
        previous epoch's weights, and the environments, carries and streams
        are where that rollout left them; all of it is saved, so a resumed run
        continues as if never stopped. The tensors are the live ones, not
        copies: a checkpointer snapshots what it is handed.

        Returns:
          state: ``model``, ``optimizer``, ``learner`` (what it carries across
            epochs) and the epoch count; ``env`` (worlds, ``rand_r`` streams,
            episode logs and the five buffers), ``rollout`` (carries, draw
            counts and the ready slot), ``ready`` and ``booted``; and
            ``world_model``, the trained weights, when the step trains its
            feature's world model.

        """
        state: CraftaxTrainStep.StateDict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learner": self.learner.state_dict(),
            "timer_step": self.timer_step.state_dict(),
            "env": self.env.state_dict(),
            "rollout": self.rollout.state_dict(slot=self.ready),
            "ready": self.ready,
            "booted": self._booted,
        }
        if self.joint is not None:
            state["world_model"] = self.joint.weights()
        return state

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore a :meth:`state_dict`: the learner's, then the pipeline's.

        Args:
          state_dict: What :meth:`state_dict` returned.

        Raises:
          ValueError: A part the checkpoint should carry is missing, or it
            carries a trained world model and the step trains none, or the
            reverse.

        """
        if not _is_step_state(state_dict):
            missing = CraftaxTrainStep.StateDict.__required_keys__ - state_dict.keys()
            msg = f"not a train step's checkpoint: {sorted(missing)} are missing"
            raise ValueError(msg)
        weights = state_dict.get("world_model")
        if (weights is None) != (self.joint is None):
            raise ValueError(
                "a trained world model's weights load into a step that trains "
                "one, and only there",
            )
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.learner.load_state_dict(state_dict["learner"])
        if self._graphs:
            # The load puts new tensors in the optimizer's state, and a captured
            # learner epoch addresses the old ones, so it would train on from the
            # state it was captured over. The next epochs capture afresh, in a new
            # pool: the old one is released with its last graph, and capturing
            # into it again fails torch's allocator assert.
            self._graphs.clear()
            self._pool = torch.cuda.graph_pool_handle()
        self.timer_step.load_state_dict(state_dict["timer_step"])
        self.env.load_state_dict(state_dict["env"])
        self._logs_read = _log_sums(self.env)
        self.ready = state_dict["ready"]
        self.write = 1 - self.ready
        self.rollout.load_state_dict(state_dict["rollout"], slot=self.ready)
        self._booted = state_dict["booted"]
        if self.joint is not None and weights is not None:
            self.joint.load_weights(weights)
            # The source takes them now, as the next rollout's would: an
            # evaluation before it then reads the loaded weights. The
            # rollout's rows begin windows, so no history needs a rebuild.
            self.joint.publish()
            self._stale = False
        if self._stream is not None:
            # The learner reads the restored slot on its own stream.
            self._stream.wait_stream(torch.cuda.current_stream(self.device))

    def close(self) -> None:
        """Wait for a rollout in flight, then stop every thread.

        Each teardown runs even when one before it raised -- a failed rollout
        re-raises here -- and a second call is a no-op.
        """
        with ExitStack() as teardown:
            teardown.callback(self.env.close)
            teardown.callback(self._prefetch.shutdown)
            teardown.callback(self.rollout.close)
            if self._pending is not None:
                self._rollout_finish()

    # Nothing clears the training env's logs, which are running fp32 sums, so each
    # report differences them, in float64, against the sums it last read; with no
    # episode finished since, it reports nothing. Practice branches never reach the
    # logs, so these are natural episodes only.
    def _episode_metrics(self) -> dict[str, float]:
        """Return :func:`report_metrics` as ``env/*``, over the episodes finished since the last report."""
        sums = _log_sums(self.env)
        change = (sums - self._logs_read).astype(np.float32)
        self._logs_read = sums
        mean = aggregate_logs(change.view(LOG_DTYPE).reshape(-1))
        if mean[-1] <= 0:
            return {}
        return {f"env/{name}": value for name, value in report_metrics(mean).items()}

    def _boot(self) -> None:
        """Fill slot 0 with the initial weights, once (``pufferl.cu:3071``)."""
        if self._booted:
            return
        self._rollout_start(0)
        self._rollout_finish()
        self.ready, self.write = 0, 1
        self._booted = True

    def _rollout_start(self, slot: int) -> None:
        """Ready the environments, copy the learner's weights to the actor, launch."""
        # Between rollouts, the boot's included: practice restores its rows here.
        self.env.prepare_rollout()
        with torch.no_grad():
            for actor, learner in zip(
                self.actor.parameters(),
                self.model.parameters(),
                strict=True,
            ):
                actor.copy_(learner)
        if self.joint is not None:
            self.joint.publish()
        rebuilding, self._stale = self._stale, False
        started = time.perf_counter()
        if rebuilding:
            self.rollout.rebuild_features()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        # Timed through the one wait, which also lands the copies above: theirs
        # is a fraction of a millisecond beside a rebuild's seconds.
        self.rebuild_seconds = time.perf_counter() - started if rebuilding else 0.0
        self._rollout_started = time.perf_counter()
        self._pending = self._prefetch.submit(self.rollout.collect, slot)

    def _rollout_finish(self) -> None:
        """Wait for the rollout in flight; it is no longer in flight even if it raised."""
        pending, self._pending = self._pending, None
        if pending is None:
            raise RuntimeError("no rollout is in flight")
        pending.result()
        self.rollout_seconds = time.perf_counter() - self._rollout_started

    def _learner_rollout(self, storage: RolloutStorage) -> LearnerRollout:
        return LearnerRollout.from_time_major(
            storage.observations,
            storage.actions,
            storage.logprobs,
            storage.rewards,
            storage.terminals,
            storage.values,
            storage.action_mask,
            storage.initial_states,
            storage.branch_starts,
            reward_scale=self.config.reward_scale,
            reward_clip=self.config.reward_clip,
            features=storage.features,
            contexts=None if self.joint is None else _stored_contexts(storage),
        )

    # On CUDA the process's first epoch runs eagerly on the learner stream, which
    # compiles every kernel and sizes cuBLAS's workspace there. Each slot's next epoch
    # is captured whole, autograd's backward and the optimizer included, as PufferLib
    # captures its learner (``pufferl.cu:1606-1626``), and every later one replays
    # that graph. Eager, the learner thread spent half a second per epoch issuing
    # launches with the GIL held (measured), longer than the rollout it should hide
    # under, and the rollout's threads waited on it. The rate reaches the optimizer as a
    # device tensor refilled before each launch, as PufferLib copies its ``lr`` to the
    # device.
    def _train_epoch(self, slot: int) -> tuple[Tensor, dict[str, Tensor], float]:
        """PufferLib's ``train_impl``: the rate for the epoch, then every minibatch."""
        self.learner.begin_epoch(self, self.global_step)
        learning_rate = 0.0
        for group, rate in zip(self.optimizer.param_groups, self._rates, strict=True):
            # ``initial_lr`` is the configured rate, recorded before any epoch.
            learning_rate = self.schedule(
                from_plain(cast("object", group["initial_lr"]), float),
                step=self.global_step,
                total_steps=self.total_steps,
            )
            rate.fill_(learning_rate)
            group["lr"] = rate
        storage = self.rollout.slots[slot]
        # A learner that trains the world model runs eagerly: its replay reads
        # each window's contexts to the host to plan its passes, whose count
        # and shapes change from epoch to epoch.
        if self._stream is None or not self._warm or self.joint is not None:
            losses, metrics = self._learn_epoch(storage)
            self._warm = True
            self._stale = self.joint is not None
            return losses, metrics, learning_rate
        if slot not in self._graphs:
            graph = torch.cuda.CUDAGraph()
            with (
                CAPTURE_LOCK,
                torch.cuda.graph(
                    graph,
                    pool=self._pool,
                    stream=self._stream,
                    capture_error_mode="thread_local",
                ),
            ):
                losses, metrics = self._learn_epoch(storage)
            self._graphs[slot] = _LearnerGraph(
                graph=graph,
                losses=losses,
                metrics=metrics,
            )
        learner = self._graphs[slot]
        learner.graph.replay()
        # Copies: the next replay rewrites the graph's own outputs.
        return (
            learner.losses.clone(),
            {name: value.clone() for name, value in learner.metrics.items()},
            learning_rate,
        )

    def _learn_epoch(self, storage: RolloutStorage) -> tuple[Tensor, dict[str, Tensor]]:
        """Run the learner's minibatches and optimizer steps; return the mean terms."""
        return self.learner(self, self._learner_rollout(storage))


@dataclass(frozen=True, slots=True, kw_only=True)
class _LearnerGraph:
    """One slot's captured learner epoch and the loss terms and metrics it writes."""

    graph: torch.cuda.CUDAGraph
    losses: Tensor
    metrics: dict[str, Tensor]


# Numpy, not torch: the logs are the Numba step's structured records, read in place.
def _log_sums(env: CraftaxEnv) -> np.ndarray:
    """Return every environment's episode-log fields as float64 ``[num_envs, LOG_FIELDS]``."""
    logs = np.ascontiguousarray(env.stats["log"])
    return logs.view(np.float32).reshape(len(logs), LOG_FIELDS).astype(np.float64)


def _check_recipe(config: CraftaxTrainStep.Config) -> None:
    """Refuse a recipe the pipeline would run wrongly, before anything is built."""
    if not math.isfinite(config.train_budget_steps) or config.train_budget_steps <= 0:
        raise ValueError(
            "train_budget_steps is the epoch count; it must be finite and positive",
        )
    scale = config.reward_scale
    if math.isnan(scale) or math.isinf(scale) or scale <= 0:
        raise ValueError(f"reward_scale must be positive and finite, not {scale}")
    if math.isnan(config.reward_clip) or config.reward_clip <= 0:
        raise ValueError(f"reward_clip must be positive, not {config.reward_clip}")
    if config.rollout.num_slots not in {1, 2}:
        raise ValueError(
            "the pipeline prefetches into a second slot or collects into its "
            f"only one; num_slots must be 1 or 2, not {config.rollout.num_slots}",
        )
    if config.rollout.horizon <= 0:
        raise ValueError(f"horizon must be positive, not {config.rollout.horizon}")
    width, expected = config.env.observation_size, config.model.observation_size
    if width != expected:
        msg = f"the env writes {width}-float observations; the policy reads {expected}"
        raise ValueError(msg)
    if config.feature is not None and config.env.rules.symbolic_observation:
        raise ValueError(
            "a feature reads the packed observation: the symbolic view is refused",
        )
    if config.feature_training is not None:
        if not isinstance(config.feature, WorldModelFeature.Config):
            raise ValueError(
                "feature_training trains a world-model feature's weights, and "
                "the step reads no such feature",
            )
        if not isinstance(config.learner, AgentWindows.Config):
            raise ValueError(
                "feature_training recomputes the features of whole agents' "
                "windows: the learner must be AgentWindows",
            )
    Evaluation.check(config.evaluation)


def _joint_world_model(
    config: CraftaxTrainStep.Config,
    feature: FeatureSource | None,
) -> JointWorldModel | None:
    """Copy the feature's world model for the learner to train; None keeps it frozen."""
    if config.feature_training is None:
        return None
    # ``_check_recipe`` refused any other feature.
    assert isinstance(config.feature, WorldModelFeature.Config)
    assert isinstance(feature, WorldModelFeature)
    # Under the capture lock, which the rollout holds while it compiles and
    # captures: a compiled replay kernel called during another thread's compile
    # raises (measured in a smoke run), and a compiler that synchronizes the device
    # would break a step graph being captured.
    return JointWorldModel(
        feature.model,
        layers=config.feature.layers,
        replay=config.feature_training.make(),
        guard=CAPTURE_LOCK,
    )


def _stored_contexts(storage: RolloutStorage) -> Contexts | None:
    """Return the contexts a joint rollout stored, agent-major; None if it stored none."""
    cells, aux = storage.frame_cells, storage.frame_aux
    previous, lengths = storage.previous_actions, storage.context_decisions
    anchored = storage.context_anchored
    prefix_cells, prefix_aux = storage.prefix_cells, storage.prefix_aux
    prefix_previous, prefix_count = (
        storage.prefix_previous_actions,
        storage.prefix_decisions,
    )
    if (
        cells is None
        or aux is None
        or previous is None
        or lengths is None
        or anchored is None
        or prefix_cells is None
        or prefix_aux is None
        or prefix_previous is None
        or prefix_count is None
    ):
        return None
    return Contexts(
        cells=cells.transpose(0, 1).contiguous(),
        aux=aux.transpose(0, 1).contiguous(),
        previous_actions=previous.transpose(0, 1).contiguous(),
        lengths=lengths.transpose(0, 1).contiguous(),
        anchored=anchored.transpose(0, 1).contiguous(),
        prefix_cells=prefix_cells,
        prefix_aux=prefix_aux,
        prefix_previous_actions=prefix_previous,
        prefix_counts=prefix_count,
    )


def _is_step_state(state: Mapping[str, object]) -> TypeIs[CraftaxTrainStep.StateDict]:
    """Whether ``state`` has every part a train step's checkpoint carries."""
    return CraftaxTrainStep.StateDict.__required_keys__ <= state.keys()


def _fp32(value: float) -> float:
    """Round to the nearest fp32; on fp32 operands this is fp32 arithmetic."""
    return struct.unpack("<f", struct.pack("<f", value))[0]
