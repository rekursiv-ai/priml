"""The standard PPO epoch: advantages over the whole rollout, then shuffled transitions.

PufferLib's epoch (``train_step.AgentWindows``) learns from windows of whole
agents and estimates each window's advantages from the live values.
:class:`ShuffledTransitions` is the PPO update as PureJaxRL and
Craftax_Baselines write it:

1. generalized advantage estimation once, over every agent's whole rollout,
   from the values the rollout recorded, bootstrapped from the value of the
   observation after its last step (the rollout's bootstrap row);
2. ``num_passes`` passes over the rollout, each a fresh shuffle of every
   transition cut into ``num_minibatches`` minibatches;
3. per minibatch, the advantages standardized, the clipped objective
   (:func:`~priml.loss.policy_gradient.clipped_policy_loss`) plus the
   weighted value and entropy terms, autograd's backward and an optimizer
   step.

A transition shuffled away from its neighbours is scored from no carry, so the
policy must keep none: :meth:`ShuffledTransitions.score` refuses one that does.

References:
    https://arxiv.org/abs/1707.06347
        Schulman et al. 2017. Proximal policy optimization algorithms.
    https://github.com/MichaelTMatthews/Craftax_Baselines
        Matthews et al. Craftax_Baselines (MIT license), ``ppo.py``, commit
        ``7ce36fa``.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import math

from configgle import Fig
from torch import Tensor

import numpy as np
import torch

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


def shuffles(*, seed: int, epoch: int, transitions: int, passes: int) -> Tensor:
    """Return epoch ``epoch``'s shuffles of the transitions, one per pass.

    Args:
      seed: The run's shuffle seed.
      epoch: The epoch; with the seed, it keys the draws, so a resumed run
        draws the shuffles it would have drawn.
      transitions: Transitions in the rollout.
      passes: Shuffles to draw.

    Returns:
      order: int64 ``[passes, transitions]``, each row a permutation.

    """
    # NumPy rather than torch: its seed sequence mixes the (seed, epoch) pair into
    # one stream. Torch's CPU generator keeps only a seed's low 32 bits (seeds 5
    # and 2**32 + 5 draw the same permutation), so a pair packed into one seed
    # would collide.
    generator = np.random.default_rng((seed, epoch))
    return torch.from_numpy(
        np.stack([generator.permutation(transitions) for _ in range(passes)]),
    )


class LearnerStep(Protocol):
    """What a shuffled epoch reads of its train step: recipe, device, policy, optimizer.

    ``train_step.CraftaxTrainStep`` is one. The epoch reads nothing of the
    environments or the rollout's slots, so it learns from any rollout handed
    to it.
    """

    @property
    def config(self) -> CraftaxTrainStep.Config:
        """The step's recipe; the epoch reads its rollout's horizon."""
        ...

    @property
    def device(self) -> torch.device:
        """Where the policy learns and the shuffles live."""
        ...

    @property
    def model(self) -> Policy:
        """The live policy."""
        ...

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """The policy's optimizer, stepped once per minibatch."""
        ...


class ShuffledTransitions:
    """PPO's epoch over shuffled transitions, with its own clipped objective.

    The defaults are Craftax_Baselines' ``ppo.py``.
    """

    class Config(Fig["ShuffledTransitions"]):
        """The passes, the estimator's and the objective's coefficients, the seed."""

        num_passes: int = 4
        """Shuffled passes over the rollout per epoch; ``ppo.py``'s ``UPDATE_EPOCHS``."""

        num_minibatches: int = 8
        """Minibatches each pass is cut into; ``ppo.py``'s ``NUM_MINIBATCHES``."""

        discount: float = 0.99
        """Reward discount factor, usually written gamma."""

        trace_decay: float = 0.8
        """Eligibility-trace decay of the advantage estimate, usually written lambda."""

        clip_epsilon: float = 0.2
        """Half-width of the trust region, for the ratio and the value alike."""

        value_coefficient: float = 0.5
        """The value term's weight in the total."""

        entropy_coefficient: float = 0.01
        """The entropy bonus's weight in the total."""

        seed: int = 0
        """Seeds every epoch's shuffles, together with the epoch's index."""

    def __init__(self, config: Config) -> None:
        """Keep the coefficients.

        Args:
          config: The passes, the coefficients and the seed.

        Raises:
          ValueError: A count is not positive, ``discount`` or
            ``trace_decay`` is outside ``[0, 1]``, another coefficient is
            negative or not finite, or the seed is negative.

        """
        for name, count in (
            ("num_passes", config.num_passes),
            ("num_minibatches", config.num_minibatches),
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
        self.size = 0
        """Transitions per minibatch, from :meth:`prepare`."""
        self.order: Tensor | None = None
        """Each pass's shuffle of the transitions, ``[passes, transitions]``,
        refilled before every epoch; a captured epoch reads it in place."""

    def prepare(self, config: CraftaxTrainStep.Config) -> None:
        """Size the minibatches from the step's rollout.

        Args:
          config: The step's recipe.

        Raises:
          ValueError: The rollout stores no bootstrap row, or the minibatches
            do not split its transitions evenly.

        """
        if not config.rollout.bootstrap:
            raise ValueError(
                "shuffled transitions bootstrap each advantage from the observation "
                "after the rollout; set the rollout's bootstrap",
            )
        transitions = config.env.num_envs * config.rollout.horizon
        if transitions % self.config.num_minibatches:
            msg = (
                f"{self.config.num_minibatches} minibatches must split the "
                f"{transitions} transitions evenly"
            )
            raise ValueError(msg)
        self.size = transitions // self.config.num_minibatches

    def begin_epoch(self, step: LearnerStep, epoch: int) -> None:
        """Draw epoch ``epoch``'s shuffles into the tensor the minibatches read.

        Args:
          step: The train step, for its device.
          epoch: The epoch's index; with the seed, it keys the shuffles, so a
            resumed run draws the ones it would have drawn.

        """
        order = shuffles(
            seed=self.config.seed,
            epoch=epoch,
            transitions=self.size * self.config.num_minibatches,
            passes=self.config.num_passes,
        )
        if self.order is None:
            self.order = torch.empty_like(order, device=step.device)
        self.order.copy_(order)

    def __call__(
        self,
        step: LearnerStep,
        rollout: LearnerRollout,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Estimate the advantages, then learn from each shuffled minibatch.

        Args:
          step: The train step: its policy, optimizer and geometry.
          rollout: The epoch's rollout with its bootstrap row, agent-major.

        Returns:
          losses: ``[8]`` fp32, the minibatches' mean terms.
          metrics: None beyond the terms.

        """
        # ``begin_epoch`` drew this epoch's shuffles before it.
        assert isinstance(self.order, Tensor)
        horizon = step.config.rollout.horizon
        advantages, returns = self.advantage(rollout)
        total = torch.zeros(
            len(TorchPPO.Config.LOSS_NAMES),
            dtype=torch.float32,
            device=step.device,
        )
        # Pass by pass, each pass's shuffle cut in order into its minibatches.
        minibatches = self.order.reshape(-1, self.size)
        for chosen in minibatches:
            step.optimizer.zero_grad(set_to_none=True)
            loss, losses = self.score(
                step.model,
                rollout,
                agents=chosen // horizon,
                times=chosen % horizon,
                advantages=advantages[chosen],
                returns=returns[chosen],
            )
            loss.backward()
            total += losses
            step.optimizer.step()
        return total / minibatches.shape[0], {}

    def loss(
        self,
        step: LearnerStep,
        rollout: LearnerRollout,
    ) -> tuple[Tensor, Tensor]:
        """Score the rollout's first ``size`` transitions, unshuffled, without the backward.

        Args:
          step: The train step: its policy and geometry.
          rollout: The rollout with its bootstrap row, agent-major.

        Returns:
          total: The minibatch's loss, 0-dim fp32.
          losses: ``[8]`` fp32, its terms.

        """
        horizon = step.config.rollout.horizon
        advantages, returns = self.advantage(rollout)
        chosen = torch.arange(self.size, device=step.device)
        return self.score(
            step.model,
            rollout,
            agents=chosen // horizon,
            times=chosen % horizon,
            advantages=advantages[chosen],
            returns=returns[chosen],
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
            msg = f"shuffled transitions carry no state, not {sorted(state)}"
            raise ValueError(msg)

    def advantage(self, rollout: LearnerRollout) -> tuple[Tensor, Tensor]:
        """Estimate every transition's advantage and value target by GAE.

        Row ``t`` stores what arrived WITH observation ``t``, so the reward and
        terminal of the transition that leaves it are row ``t + 1``'s; the
        bootstrap row supplies the last transition's and the value after it.

        Args:
          rollout: A rollout of ``horizon + 1`` rows, agent-major.

        Returns:
          advantages: fp32 ``[agents * horizon]``, transition ``(agent, t)`` at
            ``agent * horizon + t``.
          returns: ``advantages`` plus the recorded values, likewise.

        """
        rewards = rollout.rewards.float()
        values = rollout.values.float()
        dones = rollout.terminals.float()
        advantages, returns = generalized_advantage(
            rewards=rewards[:, 1:].T,
            values=values[:, :-1].T,
            dones=dones[:, 1:].T,
            last_value=values[:, -1],
            discount=self.config.discount,
            trace_decay=self.config.trace_decay,
        )
        return advantages.T.reshape(-1), returns.T.reshape(-1)

    def score(
        self,
        policy: Policy,
        rollout: LearnerRollout,
        *,
        agents: Tensor,
        times: Tensor,
        advantages: Tensor,
        returns: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Evaluate the clipped objective on one minibatch of transitions.

        Args:
          policy: The live policy.
          rollout: The epoch's rollout, agent-major.
          agents: ``[size]`` int64, each transition's agent.
          times: ``[size]`` int64, each transition's step.
          advantages: ``[size]`` fp32, standardized here before the ratio's
            clip reads them.
          returns: ``[size]`` fp32, the value targets.

        Returns:
          total: The minibatch's loss plus the policy's auxiliary loss, 0-dim
            fp32; autograd differentiates it.
          losses: ``[8]`` fp32, the objective's terms as
            ``TorchPPO.Config.LOSS_NAMES`` orders them; not differentiated.

        Raises:
          ValueError: The rollout's policy keeps a carry, which a transition
            scored alone would read from the rollout's first step.

        """
        if rollout.initial_states.numel():
            raise ValueError(
                "shuffled transitions score each transition from no carry; "
                "the rollout's policy keeps one",
            )
        decoded, _, auxiliary_loss = policy.forward_sequence(
            rollout.observations[agents, times][:, None],
            rollout.initial_states[:, agents],
            rollout.terminals[agents, times][:, None],
            actions=rollout.actions[agents, times][:, None],
        )
        decoded = decoded[:, 0].float()
        num_actions = rollout.action_mask.shape[-1]
        # The sampler masks the same way, so the learner scores the distribution
        # the actions were drawn from; an all-legal mask leaves the logits be.
        logits = torch.where(
            rollout.action_mask[agents, times] != 0,
            decoded[:, :num_actions],
            TorchPPO.Config.MASKED_LOGIT,
        )
        log_probs = logits.log_softmax(dim=-1)
        taken = log_probs.gather(-1, rollout.actions[agents, times].long()[:, None])
        behavior = rollout.logprobs[agents, times].float()
        terms = clipped_policy_loss(
            log_probs=taken[:, 0],
            behavior_log_probs=behavior,
            advantages=advantages,
            values=decoded[:, num_actions],
            behavior_values=rollout.values[agents, times].float(),
            targets=returns,
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
