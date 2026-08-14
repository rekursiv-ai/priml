"""Proximal policy optimization over the Craftax environment.

One training step is a whole PPO update: collect a fixed rollout with the
current policy, score it with generalized advantage estimation, then take
several optimization passes over shuffled minibatches of that same rollout.
Reusing the data is what makes PPO sample-efficient, and the clipped objective
is what keeps the reuse from moving the policy somewhere the data no longer
describes.

The step owns the environment rather than receiving batches, because on-policy
data cannot be prepared in advance: the next observation depends on the action
this policy just chose.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Any, Self, cast, override

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.baselines.craftax.constants import Action
from priml.baselines.craftax.env import CraftaxEnv
from priml.baselines.craftax.model import ActorCritic
from priml.baselines.craftax.renderer import OBSERVATION_SIZE
from priml.loss.policy_gradient import categorical_entropy, clipped_policy_loss
from priml.math.advantage import explained_variance, generalized_advantage
from priml.runtime import get_device
from priml.train.custom_types import TrainStepOutput


if TYPE_CHECKING:
    from collections.abc import Iterator


class Rollout:
    """One batch of experience, held time-major as ``[steps, envs, ...]``.

    Time-major is the natural layout for the backward advantage recursion,
    and flattening it for optimization is a reshape rather than a transpose.
    """

    __slots__ = (
        "action",
        "advantage",
        "done",
        "log_prob",
        "observation",
        "reward",
        "target",
        "value",
    )

    def __init__(
        self,
        *,
        observation: Tensor,
        action: Tensor,
        log_prob: Tensor,
        value: Tensor,
        reward: Tensor,
        done: Tensor,
        advantage: Tensor,
        target: Tensor,
    ) -> None:
        self.observation = observation
        self.action = action
        self.log_prob = log_prob
        self.value = value
        self.reward = reward
        self.done = done
        self.advantage = advantage
        self.target = target

    def minibatches(
        self,
        *,
        count: int,
        generator: torch.Generator | None = None,
    ) -> Iterator[dict[str, Tensor]]:
        """Shuffle every transition and yield ``count`` equal minibatches.

        Transitions are shuffled across BOTH time and environment: the value
        target already carries the temporal structure, so the optimizer sees
        each transition as an independent sample.

        Args:
          count: Minibatches per pass.
          generator: Source of randomness for the shuffle.

        Yields:
          minibatch: Flat tensors for one optimization step.

        """
        flat = {
            "observation": self.observation.flatten(0, 1),
            "action": self.action.flatten(),
            "log_prob": self.log_prob.flatten(),
            "value": self.value.flatten(),
            "advantage": self.advantage.flatten(),
            "target": self.target.flatten(),
        }
        order = torch.randperm(
            flat["action"].shape[0],
            generator=generator,
            device=flat["action"].device,
        )
        for chunk in order.chunk(count):
            yield {name: value[chunk] for name, value in flat.items()}


class CraftaxTrainStep:
    """Model, environment, and optimizer for one PPO experiment.

    Implements the training-step protocol the loop drives: each call to
    :meth:`train_step` performs one complete PPO update.
    """

    class Config(Fig["CraftaxTrainStep"]):
        """Model, environment, and the PPO hyperparameters."""

        model: ActorCritic.Config = field(default_factory=ActorCritic.Config)
        """Policy and value network.

        Narrowed to the concrete config rather than a ``Makeable``: every
        experiment here trains this network and reaches its fields directly,
        so the narrow belongs here once instead of in each factory."""

        env: CraftaxEnv.Config = field(default_factory=CraftaxEnv.Config)
        """Environment the rollout is collected from.

        Narrowed to the concrete config for the same reason the model is:
        every experiment here trains on this environment and reaches its
        worker count and seed directly."""

        rollout_steps: int = 16
        """Environment steps per worker in one update."""

        num_epochs: int = 4
        """Optimization passes over each rollout."""

        num_minibatches: int = 8
        """Minibatches per pass."""

        learning_rate: float = 3e-4
        """Initial Adam learning rate."""

        anneal_learning_rate: bool = True
        """Decay the rate linearly to zero across the run."""

        total_train_steps: int = 244
        """Updates in the run; the schedule horizon.

        Set it to the run's step budget, or the learning rate anneals past
        the end of training or short of it."""

        discount: float = 0.99
        """Reward discount factor."""

        trace_decay: float = 0.8
        """Advantage-estimation trace decay."""

        clip_epsilon: float = 0.2
        """Trust-region half-width, for both the ratio and the value."""

        entropy_coefficient: float = 0.01
        """Weight on the entropy bonus that keeps the policy exploring."""

        value_coefficient: float = 0.5
        """Weight on the value-regression term."""

        max_grad_norm: float = 1.0
        """Global gradient-norm clip."""

        device: str = "auto"
        """Device to train on; ``"auto"`` picks the best available."""

        compile: bool = False
        """Compile the model with ``torch.compile``."""

        seed: int = 0
        """Seed for action sampling and minibatch shuffling."""

        @override
        def finalize(self) -> Self:
            # The environment renders the observations the model consumes and
            # names the actions it scores, so the two must agree. Deriving the
            # geometry here means an experiment that changes the environment
            # cannot forget to resize the network.
            self.model.observation_size = OBSERVATION_SIZE
            self.model.num_actions = len(Action)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        """Build the model, environment, and optimizer.

        Args:
          config: Model, environment, and PPO settings.

        Raises:
          ValueError: A geometry or coefficient is invalid.

        """
        if config.rollout_steps <= 0 or config.num_epochs <= 0:
            raise ValueError("PPO rollout geometry must be positive")
        if config.num_minibatches <= 0:
            raise ValueError("PPO must have at least one minibatch")
        if config.total_train_steps <= 0:
            raise ValueError("total_train_steps must be positive")
        if not 0.0 <= config.discount <= 1.0:
            raise ValueError("discount must be between zero and one")
        if not 0.0 <= config.trace_decay <= 1.0:
            raise ValueError("trace_decay must be between zero and one")
        if config.clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive")

        self.config = config
        self.device = get_device(config.device)
        self.global_step: int = 0
        self.local_step: int = 0
        self.env = config.env.make()
        # Weight initialization draws from the global stream, so the seed has
        # to reach it for a run to be reproducible from its config alone.
        # The stream is restored afterwards, leaving whatever the caller had.
        saved_rng = torch.get_rng_state()
        torch.manual_seed(config.seed)
        try:
            model = config.model.make().to(self.device)
        finally:
            torch.set_rng_state(saved_rng)
        self.model: nn.Module = torch.compile(model) if config.compile else model
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(config.seed)
        self._observation = self.env.reset()
        self._done = torch.zeros(
            self._observation.shape[0],
            dtype=torch.bool,
            device=self.device,
        )
        self._episode_return = torch.zeros(
            self._observation.shape[0],
            device=self.device,
        )
        self._episode_length = torch.zeros(
            self._observation.shape[0],
            dtype=torch.int64,
            device=self.device,
        )
        self._finished_returns: list[float] = []
        self._finished_lengths: list[int] = []

    @property
    def steps_per_update(self) -> int:
        """Environment interactions consumed by one update."""
        return int(self._observation.shape[0]) * self.config.rollout_steps

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Pass the loop's batch through untouched.

        The rollout is collected here rather than supplied, so the incoming
        batch carries nothing but the loop's cadence.
        """
        return batch

    def train_step(self, **batch: Any) -> TrainStepOutput:
        """Collect a rollout and optimize on it.

        Args:
          **batch: Ignored; the data comes from the environment.

        Returns:
          result: The final minibatch's loss and logits, with the update's
            scalar diagnostics.

        """
        del batch
        rollout = self.collect()
        self._set_learning_rate()
        metrics = self._optimize(rollout)

        self.global_step += 1
        self.local_step += 1
        metrics.update(self._episode_metrics())
        metrics["explained_variance"] = float(
            explained_variance(rollout.value.flatten(), rollout.target.flatten()),
        )
        return {
            "loss": metrics.pop("_loss_tensor"),
            "model": metrics.pop("_logits"),
            "metrics": metrics,
        }

    @torch.no_grad()
    def collect(self) -> Rollout:
        """Run the current policy for a fixed number of steps.

        Returns:
          rollout: The collected experience, already scored with advantages.

        """
        observations: list[Tensor] = []
        actions: list[Tensor] = []
        log_probs: list[Tensor] = []
        values: list[Tensor] = []
        rewards: list[Tensor] = []
        dones: list[Tensor] = []

        for _ in range(self.config.rollout_steps):
            logits, value = self.model(self._observation)
            log_probs_all = logits.log_softmax(-1)
            # Sampled through the step's own generator rather than
            # ``Categorical.sample``, which draws from the global stream: a
            # run must replay from its seed regardless of what else in the
            # process has consumed randomness.
            action = torch.multinomial(
                log_probs_all.exp(),
                1,
                generator=self._generator,
            ).squeeze(-1)

            observations.append(self._observation)
            actions.append(action)
            log_probs.append(log_probs_all.gather(-1, action[:, None])[:, 0])
            values.append(value)

            transition = self.env.step(action)
            self._observation = transition.observation
            self._done = transition.done
            rewards.append(transition.reward)
            dones.append(transition.done)
            self._record_episodes(transition.reward, transition.done)

        _, last_value = self.model(self._observation)
        reward = torch.stack(rewards)
        value = torch.stack(values)
        done = torch.stack(dones)
        advantage, target = generalized_advantage(
            rewards=reward,
            values=value,
            dones=done,
            last_value=last_value,
            discount=self.config.discount,
            trace_decay=self.config.trace_decay,
        )
        return Rollout(
            observation=torch.stack(observations),
            action=torch.stack(actions),
            log_prob=torch.stack(log_probs),
            value=value,
            reward=reward,
            done=done,
            advantage=advantage,
            target=target,
        )

    def train_loss(self, **batch: Any) -> TrainStepOutput:
        """Score a rollout without optimizing.

        Args:
          **batch: Ignored; the data comes from the environment.

        Returns:
          result: The loss and logits of one freshly collected rollout.

        """
        del batch
        rollout = self.collect()
        minibatch = next(rollout.minibatches(count=1, generator=self._generator))
        loss, logits, _ = self._loss(minibatch)
        return {"loss": loss.detach(), "model": logits.detach()}

    def eval_loss(self, **batch: Any) -> TrainStepOutput:
        """Score a rollout in evaluation mode.

        Args:
          **batch: Ignored; the data comes from the environment.

        Returns:
          result: The loss and logits of one freshly collected rollout.

        """
        self.model.eval()
        try:
            return self.train_loss(**batch)
        finally:
            self.model.train()

    def call_eval(self, **batch: Any) -> Tensor:
        """Return action logits for a batch of observations.

        Args:
          **batch: Must contain ``observation``.

        Returns:
          logits: Unnormalized action scores.

        """
        self.model.eval()
        try:
            with torch.no_grad():
                logits, _ = self.model(batch["observation"])
        finally:
            self.model.train()
        return cast("Tensor", logits)

    def on_epoch_end(self) -> None:
        """Nothing to flush: every update completes within one step."""

    def state_dict(self) -> dict[str, Any]:
        """Return model, optimizer, environment, and counters."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "env": self.env.state_dict(),
            "generator": self._generator.get_state(),
            "global_step": self.global_step,
            "observation": self._observation,
            "episode_return": self._episode_return,
            "episode_length": self._episode_length,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore everything :meth:`state_dict` saved."""
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.env.load_state_dict(state_dict["env"])
        self._generator.set_state(state_dict["generator"])
        self.global_step = int(state_dict["global_step"])
        self.local_step = 0
        self._observation = state_dict["observation"]
        self._episode_return = state_dict["episode_return"]
        self._episode_length = state_dict["episode_length"]

    def _optimize(self, rollout: Rollout) -> dict[str, Any]:
        """Take every configured pass over the rollout."""
        metrics: dict[str, Any] = {}
        for _ in range(self.config.num_epochs):
            for minibatch in rollout.minibatches(
                count=self.config.num_minibatches,
                generator=self._generator,
            ):
                loss, logits, terms = self._loss(minibatch)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                metrics = {
                    "policy_loss": float(terms.policy.detach()),
                    "value_loss": float(terms.value.detach()),
                    "entropy": float(terms.entropy.detach()),
                    "approx_kl": float(terms.approx_kl.detach()),
                    "clip_fraction": float(terms.clip_fraction.detach()),
                    "grad_norm": float(grad_norm.detach()),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "_loss_tensor": loss.detach(),
                    "_logits": logits.detach(),
                }
        return metrics

    def _loss(self, minibatch: dict[str, Tensor]) -> tuple[Tensor, Tensor, Any]:
        """Evaluate the clipped objective on one minibatch."""
        logits, value = self.model(minibatch["observation"])
        log_probs = logits.log_softmax(-1)
        chosen = log_probs.gather(-1, minibatch["action"][:, None].long())[:, 0]
        terms = clipped_policy_loss(
            log_probs=chosen,
            behavior_log_probs=minibatch["log_prob"],
            advantages=minibatch["advantage"],
            values=value,
            behavior_values=minibatch["value"],
            targets=minibatch["target"],
            entropy=categorical_entropy(log_probs),
            clip_epsilon=self.config.clip_epsilon,
        )
        loss = (
            terms.policy
            + self.config.value_coefficient * terms.value
            - self.config.entropy_coefficient * terms.entropy
        )
        return loss, logits, terms

    def _set_learning_rate(self) -> None:
        """Anneal the rate linearly across the configured horizon."""
        if not self.config.anneal_learning_rate:
            return
        remaining = 1.0 - self.global_step / self.config.total_train_steps
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate * max(remaining, 0.0)

    def _record_episodes(self, reward: Tensor, done: Tensor) -> None:
        """Accumulate per-worker returns and bank the finished ones."""
        self._episode_return = self._episode_return + reward
        self._episode_length = self._episode_length + 1
        if bool(done.any()):
            self._finished_returns.extend(
                self._episode_return[done].tolist(),
            )
            self._finished_lengths.extend(
                self._episode_length[done].tolist(),
            )
            self._episode_return = self._episode_return * ~done
            self._episode_length = self._episode_length * ~done

    def _episode_metrics(self) -> dict[str, float]:
        """Summarize the episodes that finished during this update."""
        if not self._finished_returns:
            return {"episodes": 0.0}
        returns = self._finished_returns
        lengths = self._finished_lengths
        metrics = {
            "episodes": float(len(returns)),
            "episode_return": sum(returns) / len(returns),
            "episode_length": sum(lengths) / len(lengths),
            "normalized_return_pct": (
                sum(returns) / len(returns) / self.env.reward_ceiling * 100.0
            ),
        }
        self._finished_returns = []
        self._finished_lengths = []
        return metrics
