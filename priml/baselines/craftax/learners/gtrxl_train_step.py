"""PPO for a policy that remembers: whole trajectories, windows, and the actor's memory.

The objective is the clipped surrogate ``update.ShuffledTransitions`` uses.
What changes is what a minibatch IS. A memoryless policy learns from
individually shuffled transitions, because each carries everything the
network reads. A policy with a carry cannot: its prediction at a step depends
on the steps before it, so the unit of learning is a contiguous stretch of one
environment's time. That forces the three differences of
:class:`TrajectoryWindows`:

* Minibatches split the ENVIRONMENT axis, never time: each pass shuffles the
  environments and cuts them into ``num_minibatches`` groups of whole
  trajectories, so a worker's history stays intact.
* Gradients flow over fixed WINDOWS of ``window`` steps, not the whole
  rollout: each trajectory is cut into ``horizon / window`` windows, folded
  into the batch, which bounds the activations held for the backward at a
  known cost in truncated credit.
* Each window starts from the carry the ACTOR held at its first step. The
  rollout stores the carry only at step 0, so before the first pass the
  learner replays the policy over the rollout's first ``horizon - window``
  steps, without gradients, with the weights it was collected with -- the
  learner's at that point, since one slot collects each rollout with the
  weights that then learn from it -- and keeps the carry at each window's
  start. The windows read those carries in every pass, so the memory a window
  starts from is the one the actor saw, fixed for the whole update.

The advantages are GAE over each environment's whole rollout, from the values
the rollout recorded, bootstrapped from the value of the observation after its
last step, the rollout's bootstrap row.

The replay costs one more policy pass over the first ``horizon - window``
steps each update, carries the actor already computed. Storing them instead
takes ``horizon x envs x carry`` floats, tens of GB at exp008's size; a
cheaper replay is a speed follow-up, not a change of what is learned.

References:
    https://github.com/Reytuag/transformerXL_PPO_JAX
        The Craftax scoreboard implementation whose learner this ports.
    https://arxiv.org/abs/1707.06347
        Schulman et al. 2017. Proximal policy optimization algorithms.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import math

from configgle import Fig
from torch import Tensor

import torch

from priml.baselines.craftax.learners.update import shuffles
from priml.loss.policy_gradient import (
    TorchPPO,
    categorical_entropy,
    clipped_policy_loss,
)
from priml.math.advantage import generalized_advantage


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.baselines.craftax.model import Policy
    from priml.baselines.craftax.train_step import (
        CraftaxTrainStep,
        LearnerRollout,
    )


class Learning(Protocol):
    """What the windows read of a train step: its policy, optimizer and device."""

    @property
    def model(self) -> Policy:
        """The live policy."""
        ...

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Its optimizer, stepped after each minibatch."""
        ...

    @property
    def device(self) -> torch.device:
        """Where the policy and the rollout live."""
        ...


class TrajectoryWindows:
    """PPO's epoch over shuffled whole trajectories, cut into windows from the actor's carry.

    The defaults are transformerXL_PPO_JAX's.
    """

    class Config(Fig["TrajectoryWindows"]):
        """The passes, the windows, the estimator's and objective's coefficients, the seed."""

        num_passes: int = 4
        """Shuffled passes over the rollout per epoch."""

        num_minibatches: int = 8
        """Groups of whole trajectories each pass is cut into."""

        window: int = 64
        """Contiguous steps that receive gradients together; divides the horizon."""

        discount: float = 0.999
        """Reward discount factor, usually written gamma."""

        trace_decay: float = 0.8
        """Eligibility-trace decay of the advantage estimate, usually written lambda."""

        clip_epsilon: float = 0.2
        """Half-width of the trust region, for the ratio and the value alike."""

        value_coefficient: float = 0.5
        """The value term's weight in the total."""

        entropy_coefficient: float = 0.002
        """The entropy bonus's weight in the total."""

        seed: int = 0
        """Seeds every epoch's shuffles, together with the epoch's index."""

    def __init__(self, config: Config) -> None:
        """Keep the coefficients.

        Args:
          config: The passes, the windows, the coefficients and the seed.

        Raises:
          ValueError: A count is not positive, ``discount`` or
            ``trace_decay`` is outside ``[0, 1]``, another coefficient is
            negative or not finite, or the seed is negative.

        """
        for name, count in (
            ("num_passes", config.num_passes),
            ("num_minibatches", config.num_minibatches),
            ("window", config.window),
        ):
            if count <= 0:
                raise ValueError(f"{name} must be positive, not {count}")
        for name, value in (
            ("discount", config.discount),
            ("trace_decay", config.trace_decay),
        ):
            if math.isnan(value) or value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0, 1], not {value}")
        for name, value in (
            ("clip_epsilon", config.clip_epsilon),
            ("value_coefficient", config.value_coefficient),
            ("entropy_coefficient", config.entropy_coefficient),
        ):
            if math.isnan(value) or math.isinf(value) or value < 0:
                raise ValueError(f"{name} must be finite and not negative, not {value}")
        if config.seed < 0:
            raise ValueError(f"seed must not be negative, not {config.seed}")
        self.config = config
        self.rows = 0
        """Environments per minibatch, from :meth:`prepare`."""
        self.order: Tensor | None = None
        """Each pass's shuffle of the environments, ``[passes, envs]``, refilled
        before every epoch; a captured epoch reads it in place."""

    def prepare(self, config: CraftaxTrainStep.Config) -> None:
        """Place the minibatches and windows on the step's rollout.

        Args:
          config: The step's recipe.

        Raises:
          ValueError: The rollout stores no bootstrap row or has two slots, the
            window does not divide the horizon, or the minibatches do not split
            the environments evenly.

        """
        rollout = config.rollout
        if not rollout.bootstrap:
            raise ValueError(
                "trajectory windows bootstrap each advantage from the observation "
                "after the rollout; set the rollout's bootstrap",
            )
        if rollout.num_slots != 1:
            raise ValueError(
                "trajectory windows replay the actor's carry with the learner's "
                "weights, which are the actor's only when each rollout is "
                "collected with the weights that then learn from it: one slot",
            )
        if rollout.horizon % self.config.window:
            msg = (
                f"window {self.config.window} must divide the horizon {rollout.horizon}"
            )
            raise ValueError(msg)
        if config.env.num_envs % self.config.num_minibatches:
            msg = (
                f"{self.config.num_minibatches} minibatches must split the "
                f"{config.env.num_envs} environments evenly"
            )
            raise ValueError(msg)
        self.rows = config.env.num_envs // self.config.num_minibatches

    def begin_epoch(self, step: Learning, epoch: int) -> None:
        """Draw epoch ``epoch``'s shuffles into the tensor the minibatches read.

        Args:
          step: The train step, for its device.
          epoch: The epoch's index; with the seed, it keys the shuffles, so a
            resumed run draws the ones it would have drawn.

        """
        order = shuffles(
            seed=self.config.seed,
            epoch=epoch,
            transitions=self.rows * self.config.num_minibatches,
            passes=self.config.num_passes,
        )
        if self.order is None:
            self.order = torch.empty_like(order, device=step.device)
        self.order.copy_(order)

    def __call__(
        self,
        step: Learning,
        rollout: LearnerRollout,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Estimate the advantages, replay the carries, then learn from each minibatch.

        Args:
          step: The train step: its policy, optimizer and device.
          rollout: The epoch's rollout with its bootstrap row, agent-major.

        Returns:
          losses: ``[8]`` fp32, the minibatches' mean terms.
          metrics: None beyond the terms.

        """
        # ``begin_epoch`` drew this epoch's shuffles before it.
        assert isinstance(self.order, Tensor)
        advantages, returns = self.advantage(rollout)
        # Before any optimizer step: the weights the rollout was collected with.
        carries = self.carries(step.model, rollout)
        total = torch.zeros(
            len(TorchPPO.Config.LOSS_NAMES),
            dtype=torch.float32,
            device=step.device,
        )
        # Pass by pass, each pass's shuffle cut in order into its minibatches.
        minibatches = self.order.reshape(-1, self.rows)
        for agents in minibatches:
            step.optimizer.zero_grad(set_to_none=True)
            loss, losses = self.score(
                step.model,
                rollout,
                carries,
                agents=agents,
                advantages=advantages,
                returns=returns,
            )
            loss.backward()
            total += losses
            step.optimizer.step()
        return total / minibatches.shape[0], {}

    def loss(
        self,
        step: Learning,
        rollout: LearnerRollout,
    ) -> tuple[Tensor, Tensor]:
        """Score the first ``rows`` environments' windows, unshuffled, without the backward.

        Args:
          step: The train step: its policy and device.
          rollout: The rollout with its bootstrap row, agent-major.

        Returns:
          total: The minibatch's loss, 0-dim fp32.
          losses: ``[8]`` fp32, its terms.

        """
        advantages, returns = self.advantage(rollout)
        return self.score(
            step.model,
            rollout,
            self.carries(step.model, rollout),
            agents=torch.arange(self.rows, device=step.device),
            advantages=advantages,
            returns=returns,
        )

    def state_dict(self) -> dict[str, Tensor]:
        """Return nothing: each epoch's shuffles come from the seed and the epoch."""
        return {}

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Restore nothing; see :meth:`state_dict`.

        Args:
          state: What :meth:`state_dict` returned: nothing.

        Raises:
          ValueError: The state is not empty: the checkpoint is another
            learner's.

        """
        if state:
            msg = f"trajectory windows carry no state, not {sorted(state)}"
            raise ValueError(msg)

    def advantage(self, rollout: LearnerRollout) -> tuple[Tensor, Tensor]:
        """Estimate every transition's advantage and value target by GAE.

        Row ``t`` stores what arrived WITH observation ``t``, so the reward and
        terminal of the transition that leaves it are row ``t + 1``'s; the
        bootstrap row supplies the last transition's and the value after it.

        Args:
          rollout: A rollout of ``horizon + 1`` rows, agent-major.

        Returns:
          advantages: fp32 ``[agents, horizon]``.
          returns: ``advantages`` plus the recorded values.

        """
        values = rollout.values.float()
        advantages, returns = generalized_advantage(
            rewards=rollout.rewards.float()[:, 1:].T,
            values=values[:, :-1].T,
            dones=rollout.terminals.float()[:, 1:].T,
            last_value=values[:, -1],
            discount=self.config.discount,
            trace_decay=self.config.trace_decay,
        )
        return advantages.T, returns.T

    @torch.no_grad()
    def carries(self, policy: Policy, rollout: LearnerRollout) -> Tensor:
        """Replay the policy over the rollout; return the carry at each window's first step.

        Each is the carry the actor read there: emptied where that step begins
        an episode, as the actor empties it before its forward.

        Args:
          policy: The policy, at the weights the rollout was collected with.
          rollout: The rollout, agent-major, with its step-0 carry.

        Returns:
          carries: ``[windows, *carry]``, window ``k``'s at step ``k * window``.

        """
        window = self.config.window
        horizon = rollout.observations.shape[1] - 1
        state = rollout.initial_states
        carries = [state]
        for index in range(horizon - window):
            _, state = policy.forward_fused(
                rollout.observations[:, index],
                state,
                rollout.terminals[:, index],
            )
            if (index + 1) % window == 0:
                starts = rollout.terminals[:, index + 1] != 0
                carries.append(torch.where(starts[None, :, None], 0.0, state))
        return torch.stack(carries)

    def score(
        self,
        policy: Policy,
        rollout: LearnerRollout,
        carries: Tensor,
        *,
        agents: Tensor,
        advantages: Tensor,
        returns: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate the clipped objective on every window of some environments.

        Each environment's horizon is cut into its windows, folded into the
        batch window-major, and each window starts from its replayed carry.

        Args:
          policy: The live policy.
          rollout: The epoch's rollout, agent-major.
          carries: :meth:`carries`' result.
          agents: ``[rows]`` int64, the minibatch's environments.
          advantages: fp32 ``[agents, horizon]``, standardized over the
            minibatch before the ratio's clip reads them.
          returns: fp32 ``[agents, horizon]``, the value targets.

        Returns:
          total: The minibatch's loss plus the policy's auxiliary loss, 0-dim
            fp32; autograd differentiates it.
          losses: ``[8]`` fp32, the objective's terms as
            ``TorchPPO.Config.LOSS_NAMES`` orders them; not differentiated.

        """
        horizon, window = advantages.shape[1], self.config.window

        def windows(value: Tensor) -> Tensor:
            chosen = value[agents, :horizon]
            return chosen.unflatten(1, (-1, window)).transpose(0, 1).flatten(0, 1)

        actions = windows(rollout.actions)
        decoded, _, auxiliary_loss = policy.forward_sequence(
            windows(rollout.observations),
            carries[:, :, agents].transpose(0, 1).flatten(1, 2),
            windows(rollout.terminals),
            actions=actions,
        )
        decoded = decoded.float()
        num_actions = rollout.action_mask.shape[-1]
        # The sampler masks the same way, so the learner scores the distribution
        # the actions were drawn from; an all-legal mask leaves the logits be.
        logits = torch.where(
            windows(rollout.action_mask) != 0,
            decoded[..., :num_actions],
            TorchPPO.Config.MASKED_LOGIT,
        )
        log_probs = logits.log_softmax(dim=-1)
        taken = log_probs.gather(-1, actions.long()[..., None])[..., 0].flatten()
        behavior = windows(rollout.logprobs).float().flatten()
        terms = clipped_policy_loss(
            log_probs=taken,
            behavior_log_probs=behavior,
            advantages=windows(advantages).flatten(),
            values=decoded[..., num_actions].flatten(),
            behavior_values=windows(rollout.values).float().flatten(),
            targets=windows(returns).flatten(),
            entropy=categorical_entropy(log_probs).flatten(),
            clip_epsilon=self.config.clip_epsilon,
        )
        total = (
            terms.policy
            + self.config.value_coefficient * terms.value
            - self.config.entropy_coefficient * terms.entropy
        )
        log_ratio = taken.detach() - behavior
        return total + auxiliary_loss, torch.stack(
            (
                terms.policy,
                terms.value,
                terms.entropy,
                total,
                (-log_ratio).mean(),
                terms.approx_kl,
                terms.clip_fraction,
                log_ratio.exp().mean(),
            ),
        ).detach()
