"""PPO for a GRU policy: whole trajectories, whole rollouts.

The same clipped objective as everywhere else. What differs is the shape of a
minibatch, and it differs for the same reason it does in
:mod:`gtrxl_train_step`: a recurrent prediction depends on the steps before
it, so transitions cannot be shuffled individually.

Simpler than the transformer's version in one way and stricter in another.
Simpler, because a GRU state is one vector: the rollout records the state it
started from and everything else replays exactly, with no per-layer cache to
rebuild. Stricter, because gradients run over the WHOLE rollout rather than a
window -- the reference does this, and a recurrence has no parallel form to
make a shorter window cheaper, so there is nothing to gain by truncating.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING, Any, Self, override

from configgle import Fig
from torch import Tensor

import torch

from priml.baselines.craftax.env import CraftaxEnv
from priml.baselines.craftax.game.constants import Action
from priml.baselines.craftax.game.observation import observation_size
from priml.baselines.craftax.rnn import ActorCriticRNN
from priml.loss.policy_gradient import categorical_entropy, clipped_policy_loss
from priml.math.advantage import explained_variance, generalized_advantage
from priml.runtime import get_device
from priml.train.custom_types import TrainStepOutput


if TYPE_CHECKING:
    from collections.abc import Iterator


type _Callable = Callable[..., Any]
"""One of the model's recurrent entry points, compiled or not."""


class RecurrentRollout:
    """One batch of experience, plus the state the recurrence began from."""

    __slots__ = (
        "action",
        "advantage",
        "done",
        "initial_state",
        "log_prob",
        "observation",
        "previous_done",
        "reward",
        "target",
        "value",
    )

    def __init__(
        self,
        *,
        observation: Tensor,
        previous_done: Tensor,
        initial_state: Tensor,
        action: Tensor,
        log_prob: Tensor,
        value: Tensor,
        reward: Tensor,
        done: Tensor,
        advantage: Tensor,
        target: Tensor,
    ) -> None:
        self.observation = observation
        self.previous_done = previous_done
        self.initial_state = initial_state
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
        """Shuffle whole trajectories and yield ``count`` groups of them.

        The environment axis is shuffled and split; time is never cut. Each
        minibatch is therefore a set of complete trajectories with the exact
        recurrent state each began from, which is what lets the loss replay
        them.

        Args:
          count: Minibatches per pass; must divide the worker count.
          generator: Source of randomness for the trajectory shuffle.

        Yields:
          minibatch: Time-major tensors plus their starting recurrent state.

        """
        order = torch.randperm(
            self.observation.shape[1],
            generator=generator,
            device=self.observation.device,
        )
        named = {
            "observation": self.observation,
            "previous_done": self.previous_done,
            "action": self.action,
            "log_prob": self.log_prob,
            "value": self.value,
            "advantage": self.advantage,
            "target": self.target,
        }
        shuffled = {
            name: _split_environments(value, order=order, count=count)
            for name, value in named.items()
        }
        states = self.initial_state[order].reshape(
            count, -1, self.initial_state.shape[-1]
        )

        for index in range(count):
            minibatch = {name: value[index] for name, value in shuffled.items()}
            minibatch["initial_state"] = states[index]
            yield minibatch


class CraftaxRNNTrainStep:
    """Model, environment, and optimizer for one recurrent PPO experiment."""

    class Config(Fig["CraftaxRNNTrainStep"]):
        """Model, environment, and the PPO hyperparameters."""

        model: ActorCriticRNN.Config = field(default_factory=ActorCriticRNN.Config)
        """Recurrent policy and value network."""

        env: CraftaxEnv.Config = field(default_factory=CraftaxEnv.Config)
        """Environment the rollout is collected from."""

        rollout_steps: int = 64
        """Environment steps per worker in one update."""

        num_epochs: int = 4
        """Optimization passes over each rollout."""

        num_minibatches: int = 8
        """Minibatches per pass; must divide the worker count."""

        learning_rate: float = 2e-4
        """Initial Adam learning rate."""

        anneal_learning_rate: bool = True
        """Decay the rate linearly to zero across the run."""

        total_train_steps: int = 15_258
        """Updates in the run; the schedule horizon."""

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
        """Compile the recurrent entry points with ``torch.compile``."""

        seed: int = 0
        """Seed for action sampling and minibatch shuffling."""

        @override
        def finalize(self) -> Self:
            # The environment renders the observations and names the actions,
            # so an experiment that changes it cannot forget to resize the net.
            self.model.observation_size = observation_size(self.env.view)
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
        saved_rng = torch.get_rng_state()
        torch.manual_seed(config.seed)
        try:
            model = config.model.make()
            model.to(self.device)
        finally:
            torch.set_rng_state(saved_rng)
        self.model = model
        # The compiled handles wrap the two RECURRENT entry points, not
        # ``forward``: a rollout never calls ``forward``, so compiling the
        # module would leave the hot path interpreted.
        self._step = _compiled(model.step, enabled=config.compile)
        self._sequence = _compiled(model.sequence, enabled=config.compile)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
        )
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(config.seed)
        self._observation = self.env.reset()

        num_envs = int(self._observation.shape[0])
        if num_envs % config.num_minibatches:
            raise ValueError("num_minibatches must divide the worker count")
        self._state = self.model.initial_state(num_envs, device=self.device)
        self._previous_done = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._episode_return = torch.zeros(num_envs, device=self.device)
        self._episode_length = torch.zeros(
            num_envs,
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
        """Pass the loop's batch through untouched."""
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
    def collect(self) -> RecurrentRollout:
        """Run the current policy for a fixed number of steps.

        Returns:
          rollout: The collected experience, already scored with advantages.

        """
        initial_state = self._state
        observations: list[Tensor] = []
        previous_dones: list[Tensor] = []
        actions: list[Tensor] = []
        log_probs: list[Tensor] = []
        values: list[Tensor] = []
        rewards: list[Tensor] = []
        dones: list[Tensor] = []

        for _ in range(self.config.rollout_steps):
            observations.append(self._observation)
            previous_dones.append(self._previous_done)

            self._state, logits, value = self._step(
                self._state,
                self._observation,
                self._previous_done,
            )
            log_probs_all = logits.log_softmax(-1)
            action = torch.multinomial(
                log_probs_all.exp(),
                1,
                generator=self._generator,
            ).squeeze(-1)

            actions.append(action)
            log_probs.append(log_probs_all.gather(-1, action[:, None])[:, 0])
            values.append(value)

            transition = self.env.step(action)
            self._observation = transition.observation
            self._previous_done = transition.done
            rewards.append(transition.reward)
            dones.append(transition.done)
            self._record_episodes(transition.reward, transition.done)

        _, _, last_value = self._step(
            self._state,
            self._observation,
            self._previous_done,
        )
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
        return RecurrentRollout(
            observation=torch.stack(observations),
            previous_done=torch.stack(previous_dones),
            initial_state=initial_state,
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
          logits: Unnormalized action scores, computed with a fresh state.

        """
        self.model.eval()
        try:
            with torch.no_grad():
                logits, _ = self.model(batch["observation"])
        finally:
            self.model.train()
        return logits

    def on_epoch_end(self) -> None:
        """Nothing to flush: every update completes within one step."""

    def state_dict(self) -> dict[str, Any]:
        """Return model, optimizer, environment, state, and counters."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "env": self.env.state_dict(),
            "generator": self._generator.get_state(),
            "global_step": self.global_step,
            "observation": self._observation,
            "recurrent_state": self._state,
            "previous_done": self._previous_done,
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
        self._observation = _tensor(state_dict["observation"])
        self._state = _tensor(state_dict["recurrent_state"])
        self._previous_done = _tensor(state_dict["previous_done"])
        self._episode_return = _tensor(state_dict["episode_return"])
        self._episode_length = _tensor(state_dict["episode_length"])

    def _optimize(self, rollout: RecurrentRollout) -> dict[str, Any]:
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
        """Evaluate the clipped objective over one set of trajectories."""
        _, logits, value = self._sequence(
            minibatch["initial_state"],
            minibatch["observation"],
            minibatch["previous_done"],
        )
        log_probs = logits.log_softmax(-1)
        chosen = log_probs.gather(-1, minibatch["action"][..., None].long())[..., 0]
        terms = clipped_policy_loss(
            log_probs=chosen.flatten(),
            behavior_log_probs=minibatch["log_prob"].flatten(),
            advantages=minibatch["advantage"].flatten(),
            values=value.flatten(),
            behavior_values=minibatch["value"].flatten(),
            targets=minibatch["target"].flatten(),
            entropy=categorical_entropy(log_probs).flatten(),
            clip_epsilon=self.config.clip_epsilon,
        )
        loss = (
            terms.policy
            + self.config.value_coefficient * terms.value
            - self.config.entropy_coefficient * terms.entropy
        )
        return loss, logits.flatten(0, 1), terms

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
            self._finished_returns.extend(self._episode_return[done].tolist())
            self._finished_lengths.extend(self._episode_length[done].tolist())
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


def _compiled(function: _Callable, *, enabled: bool) -> _Callable:
    """Compile one bound method, or return it untouched.

    Compiling the recurrent entry points rather than the module is what keeps
    the rollout on the compiled path: a rollout calls ``step``, never
    ``forward``.

    Args:
      function: The bound method to compile.
      enabled: Whether to compile at all.

    Returns:
      callable: The compiled function, or the original.

    """
    return torch.compile(function) if enabled else function


def _tensor(value: Any) -> Tensor:
    """Narrow one checkpoint entry to a tensor.

    A state dict is untyped by construction, and assigning straight from it
    would widen every restored attribute to ``Any`` -- erasing the shapes the
    rest of this file depends on.

    Args:
      value: The checkpoint entry.

    Returns:
      tensor: The same value, typed.

    Raises:
      TypeError: The entry is not a tensor.

    """
    if not isinstance(value, Tensor):
        raise TypeError(f"checkpoint entry must be a tensor, got {type(value)}")
    return value


def _split_environments(value: Tensor, *, order: Tensor, count: int) -> Tensor:
    """Shuffle whole trajectories and expose a leading minibatch axis.

    Args:
      value: Time-major tensor, ``[time, envs, ...]``.
      order: Permutation of the worker axis.
      count: Minibatches to split the workers into.

    Returns:
      split: ``[count, time, envs / count, ...]``.

    """
    shuffled = value[:, order]
    grouped = shuffled.reshape(value.shape[0], count, -1, *value.shape[2:])
    return grouped.transpose(0, 1)
