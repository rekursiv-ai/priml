"""The recurrent PPO epoch: shuffled whole trajectories, replayed from their carries.

Craftax_Baselines' ``ppo_rnn.py`` is its ``ppo.py`` with one change to the
minibatch, made because a recurrent policy's prediction depends on the steps
before it, so its transitions cannot be shuffled one by one.
:class:`ShuffledTrajectories` makes that change to
:class:`~priml.baselines.craftax.learners.update.ShuffledTransitions`:

1. generalized advantage estimation once, over every agent's whole rollout,
   bootstrapped from the value of the observation after its last step (the
   rollout's bootstrap row), as before;
2. ``num_passes`` passes, each a fresh shuffle of the AGENTS cut into
   ``num_minibatches`` minibatches of whole trajectories: time is never cut;
3. per minibatch, the policy replayed over every step of the rollout from the
   carry its agents started it with, resetting the carry where an episode
   starts, then the same standardized advantages, clipped objective, weighted
   value and entropy terms, backward and optimizer step.

The gradient runs through the whole rollout, as the reference's does: the
recurrence has no parallel form, so a shorter window would save nothing.

References:
    https://github.com/MichaelTMatthews/Craftax_Baselines
        Matthews et al. Craftax_Baselines (MIT license), ``ppo_rnn.py``.
    https://arxiv.org/abs/1707.06347
        Schulman et al. 2017. Proximal policy optimization algorithms.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from configgle import Makes
from torch import Tensor

import torch

from priml.baselines.craftax.learners.update import ShuffledTransitions, shuffles
from priml.loss.policy_gradient import (
    TorchPPO,
    categorical_entropy,
    clipped_policy_loss,
)


if TYPE_CHECKING:
    from priml.baselines.craftax.learners.update import LearnerStep
    from priml.baselines.craftax.model import Policy
    from priml.baselines.craftax.train_step import (
        CraftaxTrainStep,
        LearnerRollout,
    )


class ShuffledTrajectories(ShuffledTransitions):
    """PPO's epoch over shuffled whole trajectories, for a policy that keeps a carry.

    It keeps :class:`ShuffledTransitions`' passes, coefficients, seed, checks
    and advantage estimate; the defaults are ``ppo_rnn.py``'s, which are
    ``ppo.py``'s. Its ``order`` holds each pass's shuffle of the agents, not
    of the transitions, and ``rows`` agents make a minibatch.
    """

    class Config(Makes["ShuffledTrajectories"], ShuffledTransitions.Config):
        """ShuffledTransitions' passes, coefficients and seed; a minibatch is whole agents."""

    def __init__(self, config: Config) -> None:
        """Keep the coefficients.

        Args:
          config: The passes, the coefficients and the seed.

        """
        super().__init__(config)
        self.rows = 0
        """Agents per minibatch, from :meth:`prepare`."""

    @override
    def prepare(self, config: CraftaxTrainStep.Config) -> None:
        """Size the minibatches from the step's agents.

        Args:
          config: The step's recipe.

        Raises:
          ValueError: The rollout stores no bootstrap row, or the minibatches
            do not split its agents evenly.

        """
        super().prepare(config)
        agents = config.env.num_envs
        if agents % self.config.num_minibatches:
            msg = (
                f"{self.config.num_minibatches} minibatches must split the "
                f"{agents} agents evenly"
            )
            raise ValueError(msg)
        self.rows = agents // self.config.num_minibatches

    @override
    def begin_epoch(self, step: LearnerStep, epoch: int) -> None:
        """Draw epoch ``epoch``'s shuffles of the agents into the tensor the minibatches read.

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

    @override
    def __call__(
        self,
        step: LearnerStep,
        rollout: LearnerRollout,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Learn from each minibatch of the epoch's shuffled agents.

        Args:
          step: The train step: its policy and optimizer.
          rollout: The epoch's rollout with its bootstrap row, agent-major.

        Returns:
          losses: ``[8]`` fp32, the minibatches' mean terms.
          metrics: None beyond the terms.

        """
        # ``begin_epoch`` drew this epoch's shuffles before it.
        assert isinstance(self.order, Tensor)
        # Pass by pass, each pass's shuffle cut in order into its minibatches.
        minibatches = self.order.reshape(-1, self.rows)
        return self.learn(step.model, step.optimizer, rollout, minibatches), {}

    def learn(
        self,
        policy: Policy,
        optimizer: torch.optim.Optimizer,
        rollout: LearnerRollout,
        minibatches: Tensor,
    ) -> Tensor:
        """Estimate the advantages, then take one optimizer step per minibatch of agents.

        Args:
          policy: The live policy.
          optimizer: Its optimizer.
          rollout: The epoch's rollout with its bootstrap row, agent-major.
          minibatches: ``[count, rows]`` int64, each minibatch's agents, in
            learning order.

        Returns:
          losses: ``[8]`` fp32, the minibatches' mean terms.

        """
        advantages, returns = self.trajectory_advantages(rollout)
        total = advantages.new_zeros(len(TorchPPO.Config.LOSS_NAMES))
        for chosen in minibatches:
            optimizer.zero_grad(set_to_none=True)
            loss, losses = self.score_trajectories(
                policy,
                rollout,
                agents=chosen,
                advantages=advantages[chosen],
                returns=returns[chosen],
            )
            loss.backward()
            total += losses
            optimizer.step()
        return total / minibatches.shape[0]

    @override
    def loss(
        self,
        step: LearnerStep,
        rollout: LearnerRollout,
    ) -> tuple[Tensor, Tensor]:
        """Score the rollout's first ``rows`` agents, unshuffled, without the backward.

        Args:
          step: The train step: its policy and geometry.
          rollout: The rollout with its bootstrap row, agent-major.

        Returns:
          total: The minibatch's loss, 0-dim fp32.
          losses: ``[8]`` fp32, its terms.

        """
        advantages, returns = self.trajectory_advantages(rollout)
        chosen = torch.arange(self.rows, device=step.device)
        return self.score_trajectories(
            step.model,
            rollout,
            agents=chosen,
            advantages=advantages[chosen],
            returns=returns[chosen],
        )

    def trajectory_advantages(self, rollout: LearnerRollout) -> tuple[Tensor, Tensor]:
        """Estimate every agent's advantages and value targets over its whole rollout.

        :meth:`ShuffledTransitions.advantage`'s estimate, one row per agent.

        Args:
          rollout: A rollout of ``horizon + 1`` rows, agent-major.

        Returns:
          advantages: fp32 ``[agents, horizon]``.
          returns: ``advantages`` plus the recorded values, likewise.

        """
        advantages, returns = self.advantage(rollout)
        horizon = rollout.rewards.shape[1] - 1
        return advantages.reshape(-1, horizon), returns.reshape(-1, horizon)

    def score_trajectories(
        self,
        policy: Policy,
        rollout: LearnerRollout,
        *,
        agents: Tensor,
        advantages: Tensor,
        returns: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate the clipped objective on the whole trajectories of some agents.

        The policy replays every step of the rollout, less its bootstrap row,
        from the carry each agent started it with; the objective is
        :meth:`ShuffledTransitions.score`'s, over the flattened steps.

        Args:
          policy: The live policy.
          rollout: The epoch's rollout with its bootstrap row, agent-major.
          agents: ``[rows]`` int64, the minibatch's agents.
          advantages: ``[rows, horizon]`` fp32, standardized over the
            minibatch before the ratio's clip reads them.
          returns: ``[rows, horizon]`` fp32, the value targets.

        Returns:
          total: The minibatch's loss plus the policy's auxiliary loss, 0-dim
            fp32; autograd differentiates it.
          losses: ``[8]`` fp32, the objective's terms as
            ``TorchPPO.Config.LOSS_NAMES`` orders them; not differentiated.

        """
        horizon = advantages.shape[-1]
        actions = rollout.actions[agents, :horizon]
        decoded, _, auxiliary_loss = policy.forward_sequence(
            rollout.observations[agents, :horizon],
            rollout.initial_states[:, agents],
            rollout.terminals[agents, :horizon],
            actions=actions,
        )
        decoded = decoded.float().flatten(0, 1)
        num_actions = rollout.action_mask.shape[-1]
        # The sampler masks the same way, so the learner scores the distribution
        # the actions were drawn from; an all-legal mask leaves the logits be.
        logits = torch.where(
            rollout.action_mask[agents, :horizon].flatten(0, 1) != 0,
            decoded[:, :num_actions],
            TorchPPO.Config.MASKED_LOGIT,
        )
        log_probs = logits.log_softmax(dim=-1)
        taken = log_probs.gather(-1, actions.flatten().long()[:, None])
        behavior = rollout.logprobs[agents, :horizon].flatten().float()
        terms = clipped_policy_loss(
            log_probs=taken[:, 0],
            behavior_log_probs=behavior,
            advantages=advantages.flatten(),
            values=decoded[:, num_actions],
            behavior_values=rollout.values[agents, :horizon].flatten().float(),
            targets=returns.flatten(),
            entropy=categorical_entropy(log_probs),
            clip_epsilon=self.config.clip_epsilon,
        )
        total = (
            terms.policy
            + self.config.value_coefficient * terms.value
            - self.config.entropy_coefficient * terms.entropy
        )
        log_ratio = taken[:, 0].detach() - behavior
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
