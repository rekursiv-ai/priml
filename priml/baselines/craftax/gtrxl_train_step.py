"""PPO for a policy that remembers: recurrent rollouts and windowed gradients.

The objective is the same clipped surrogate the feed-forward step uses. What
changes is what a minibatch IS. A memoryless policy can be optimized on
individually shuffled transitions, because each one carries everything the
network reads. A recurrent policy cannot: its prediction at a step depends on
the steps before it, so the unit of optimization has to be a contiguous stretch
of time from one worker.

That forces three things, and they are the whole difference from ``train_step``:

* Minibatches split the ENVIRONMENT axis, never time. Whole trajectories move
  together, so a worker's history stays intact.
* Gradients flow over a fixed WINDOW of steps rather than the whole rollout.
  A 128-step rollout backpropagated end to end would hold every intermediate
  activation; 64-step windows bound that at a known cost in truncated credit.
* Each window's starting memory is REPLAYED, not stored. The rollout records
  the input each layer saw at each step, which is exactly enough to rebuild
  the cache any window began with -- far smaller than the attention states.
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
from priml.baselines.craftax.game.observation import OBSERVATION_SIZE
from priml.baselines.craftax.gtrxl import ActorCriticGTrXL
from priml.loss.policy_gradient import categorical_entropy, clipped_policy_loss
from priml.math.advantage import explained_variance, generalized_advantage
from priml.runtime import get_device
from priml.train.custom_types import TrainStepOutput


if TYPE_CHECKING:
    from collections.abc import Iterator


type _Callable = Callable[..., Any]
"""One of the model's recurrent entry points, compiled or not."""


class RecurrentRollout:
    """One batch of experience, plus the memory needed to replay it.

    Every tensor is time-major ``[steps, envs, ...]``, and ``layer_input``
    additionally carries what each transformer layer read at each step --
    the record that lets a gradient window reconstruct its starting cache.
    """

    __slots__ = (
        "action",
        "advantage",
        "done",
        "layer_input",
        "log_prob",
        "observation",
        "previous_done",
        "reward",
        "target",
        "valid_length",
        "value",
    )

    def __init__(
        self,
        *,
        observation: Tensor,
        previous_done: Tensor,
        valid_length: Tensor,
        layer_input: Tensor,
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
        self.valid_length = valid_length
        self.layer_input = layer_input
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
        initial_memory: Tensor,
        count: int,
        window: int,
        memory_length: int,
        generator: torch.Generator | None = None,
    ) -> Iterator[dict[str, Tensor]]:
        """Shuffle whole trajectories and cut each into gradient windows.

        Args:
          initial_memory: Cache the rollout started from,
            ``[envs, memory_length, layers, embed]``.
          count: Minibatches per pass; must divide the worker count.
          window: Steps that receive gradients together.
          memory_length: Steps of memory the model attends over.
          generator: Source of randomness for the trajectory shuffle.

        Yields:
          minibatch: Time-major window tensors plus their starting memory.

        """
        history = torch.cat(
            (initial_memory.transpose(0, 1), self.layer_input),
            dim=0,
        )
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
        valid_length = _split_environments(self.valid_length, order=order, count=count)
        memories = _split_environments(history, order=order, count=count)

        for index in range(count):
            minibatch = {
                name: _windows(value[index], window=window)
                for name, value in shuffled.items()
            }
            minibatch["memory"] = _window_memories(
                memories[index],
                memory_length=memory_length,
                window=window,
            )
            minibatch["valid_length"] = valid_length[index][::window].reshape(-1)
            yield minibatch


class CraftaxGTrXLTrainStep:
    """Model, environment, and optimizer for one recurrent PPO experiment."""

    class Config(Fig["CraftaxGTrXLTrainStep"]):
        """Model, environment, and the PPO hyperparameters."""

        model: ActorCriticGTrXL.Config = field(
            default_factory=ActorCriticGTrXL.Config,
        )
        """Recurrent policy and value network."""

        env: CraftaxEnv.Config = field(default_factory=CraftaxEnv.Config)
        """Environment the rollout is collected from.

        Narrowed to the concrete config for the same reason the model is:
        every experiment here trains on this environment and reaches its
        worker count and seed directly."""

        rollout_steps: int = 128
        """Environment steps per worker in one update."""

        gradient_window: int = 64
        """Contiguous steps that receive gradients together.

        Must divide ``rollout_steps``. Larger windows carry credit further
        back at a proportional cost in stored activations."""

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

        discount: float = 0.999
        """Reward discount factor.

        Higher than the feed-forward recipe's: a policy that can remember is
        worth pointing at rewards further away."""

        trace_decay: float = 0.8
        """Advantage-estimation trace decay."""

        clip_epsilon: float = 0.2
        """Trust-region half-width, for both the ratio and the value."""

        entropy_coefficient: float = 0.002
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
            # The environment renders the observations and names the actions,
            # so an experiment that changes it cannot forget to resize the
            # network.
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
        if config.num_minibatches <= 0 or config.gradient_window <= 0:
            raise ValueError("PPO rollout geometry must be positive")
        if config.rollout_steps % config.gradient_window:
            raise ValueError("gradient_window must divide rollout_steps")
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
        self._memory, self._valid_length = self.model.initial_state(
            num_envs,
            device=self.device,
        )
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
        memory = self._memory
        rollout = self.collect()
        self._set_learning_rate()
        metrics = self._optimize(rollout, initial_memory=memory)

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
        observations: list[Tensor] = []
        previous_dones: list[Tensor] = []
        valid_lengths: list[Tensor] = []
        layer_inputs: list[Tensor] = []
        actions: list[Tensor] = []
        log_probs: list[Tensor] = []
        values: list[Tensor] = []
        rewards: list[Tensor] = []
        dones: list[Tensor] = []

        for _ in range(self.config.rollout_steps):
            # Recorded BEFORE the step clears it: this is how much memory the
            # window starting here may attend to, which is what the gradient
            # pass has to be told.
            valid_lengths.append(
                torch.where(self._previous_done, 0, self._valid_length),
            )
            observations.append(self._observation)
            previous_dones.append(self._previous_done)

            self._memory, self._valid_length, logits, value = self._step(
                self._memory,
                self._valid_length,
                self._observation,
                self._previous_done,
            )
            log_probs_all = logits.log_softmax(-1)
            action = torch.multinomial(
                log_probs_all.exp(),
                1,
                generator=self._generator,
            ).squeeze(-1)

            layer_inputs.append(self._memory[:, -1])
            actions.append(action)
            log_probs.append(log_probs_all.gather(-1, action[:, None])[:, 0])
            values.append(value)

            transition = self.env.step(action)
            self._observation = transition.observation
            self._previous_done = transition.done
            rewards.append(transition.reward)
            dones.append(transition.done)
            self._record_episodes(transition.reward, transition.done)

        _, _, _, last_value = self._step(
            self._memory,
            self._valid_length,
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
            valid_length=torch.stack(valid_lengths),
            layer_input=torch.stack(layer_inputs),
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
        memory = self._memory
        rollout = self.collect()
        minibatch = next(
            rollout.minibatches(
                initial_memory=memory,
                count=1,
                window=self.config.gradient_window,
                memory_length=self.model.memory_length,
                generator=self._generator,
            ),
        )
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
          logits: Unnormalized action scores, computed with empty memory.

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
        """Return model, optimizer, environment, memory, and counters."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "env": self.env.state_dict(),
            "generator": self._generator.get_state(),
            "global_step": self.global_step,
            "observation": self._observation,
            "memory": self._memory,
            "valid_length": self._valid_length,
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
        self._observation = state_dict["observation"]
        self._memory = state_dict["memory"]
        self._valid_length = state_dict["valid_length"]
        self._previous_done = state_dict["previous_done"]
        self._episode_return = state_dict["episode_return"]
        self._episode_length = state_dict["episode_length"]

    def _optimize(
        self,
        rollout: RecurrentRollout,
        *,
        initial_memory: Tensor,
    ) -> dict[str, Any]:
        """Take every configured pass over the rollout."""
        metrics: dict[str, Any] = {}
        for _ in range(self.config.num_epochs):
            for minibatch in rollout.minibatches(
                initial_memory=initial_memory,
                count=self.config.num_minibatches,
                window=self.config.gradient_window,
                memory_length=self.model.memory_length,
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
        """Evaluate the clipped objective over one set of gradient windows."""
        logits, value = self._sequence(
            minibatch["memory"],
            minibatch["valid_length"],
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

    Compiling the two recurrent entry points rather than the module is what
    keeps the rollout on the compiled path: a rollout calls ``step``, never
    ``forward``, so ``torch.compile(module)`` would compile the one method
    training does not use.

    Args:
      function: The bound method to compile.
      enabled: Whether to compile at all.

    Returns:
      callable: The compiled function, or the original.

    """
    return torch.compile(function) if enabled else function


def _split_environments(
    value: Tensor,
    *,
    order: Tensor,
    count: int,
) -> Tensor:
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


def _windows(value: Tensor, *, window: int) -> Tensor:
    """Cut time into fixed windows and fold the pieces into the batch axis.

    Args:
      value: Time-major tensor, ``[time, envs, ...]``.
      window: Steps per window; must divide ``time``.

    Returns:
      windowed: ``[window, chunks * envs, ...]``, chunk-major.

    """
    chunks = value.shape[0] // window
    reshaped = value.reshape(chunks, window, value.shape[1], *value.shape[2:])
    return reshaped.transpose(0, 1).reshape(
        window,
        chunks * value.shape[1],
        *value.shape[2:],
    )


def _window_memories(
    history: Tensor,
    *,
    memory_length: int,
    window: int,
) -> Tensor:
    """Rebuild the memory each gradient window began with.

    The rollout recorded what every layer read at every step, and the cache a
    window starts from is exactly the ``memory_length`` rows preceding it --
    so no attention state has to be stored to replay it.

    Args:
      history: Layer inputs preceded by the rollout's starting cache,
        ``[memory_length + time, envs, layers, embed]``.
      memory_length: Rows the model attends over.
      window: Steps per gradient window.

    Returns:
      memories: ``[chunks * envs, memory_length, layers, embed]``, ordered to
        match :func:`_windows`.

    """
    rollout_steps = history.shape[0] - memory_length
    starts = torch.arange(0, rollout_steps, window, device=history.device)
    rows = torch.arange(memory_length, device=history.device)
    selected = history[starts[:, None] + rows[None, :]]
    selected = selected.transpose(1, 2)
    return selected.reshape(
        selected.shape[0] * selected.shape[1],
        memory_length,
        *selected.shape[3:],
    )
