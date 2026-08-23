"""Q-learning on fresh experience: no buffer, no target network.

One update collects a rollout with an epsilon-greedy policy, builds
Q(lambda) targets from the network's OWN values, and regresses toward them.
That is the whole algorithm -- there is no second network holding the target
still, and nothing is stored between updates.

What makes that stable is covered in :mod:`pqn`: enough parallel workers to
decorrelate samples, batch renormalization to absorb the shift as the policy
changes, and a multi-step target that leans less on any single bootstrap.

Two things differ from the PPO steps beside it. The loss is a plain squared
error rather than a clipped surrogate, because there is no policy ratio to
trust-region. And the targets are built ONCE per rollout, before any
optimization: recomputing them from the updated network each epoch would be
chasing a value the network had already moved.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING, Any, Self, override

from configgle import Makes, PartialConfig
from torch import Tensor

import torch

from priml.baselines.craftax.env import CraftaxEnv
from priml.baselines.craftax.game.constants import Action
from priml.baselines.craftax.game.observation import observation_size
from priml.baselines.craftax.pqn import RecurrentQNetwork, epsilon_at
from priml.math.advantage import explained_variance, q_lambda_targets
from priml.math.schedules import linear
from priml.train.custom_types import TrainStepOutput
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from collections.abc import Iterator


type _StepFn = Callable[
    [tuple[Tensor, Tensor], Tensor, Tensor, Tensor],
    tuple[tuple[Tensor, Tensor], Tensor],
]
"""The model's one-step entry point, compiled or not.

Typed precisely rather than as ``Callable[..., Any]``: an ``Any`` here would
erase the recurrent state's shape at every call site, and the state being a
PAIR is the thing most easily got wrong."""

type _SequenceFn = _StepFn
"""The windowed entry point, whose signature matches the stepped one."""


class QRollout:
    """One batch of experience with its regression targets already built."""

    __slots__ = (
        "action",
        "cell",
        "hidden",
        "observation",
        "previous_action",
        "previous_done",
        "q_value",
        "reward",
        "target",
    )

    def __init__(
        self,
        *,
        observation: Tensor,
        previous_action: Tensor,
        previous_done: Tensor,
        hidden: Tensor,
        cell: Tensor,
        action: Tensor,
        reward: Tensor,
        q_value: Tensor,
        target: Tensor,
    ) -> None:
        self.observation = observation
        self.previous_action = previous_action
        self.previous_done = previous_done
        self.hidden = hidden
        self.cell = cell
        self.action = action
        self.reward = reward
        self.q_value = q_value
        self.target = target

    def minibatches(
        self,
        *,
        count: int,
        generator: torch.Generator | None = None,
    ) -> Iterator[dict[str, Tensor]]:
        """Shuffle whole trajectories and yield ``count`` groups of them.

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
            "previous_action": self.previous_action,
            "previous_done": self.previous_done,
            "action": self.action,
            "target": self.target,
        }
        shuffled = {
            name: _split_environments(value, order=order, count=count)
            for name, value in named.items()
        }
        width = self.hidden.shape[-1]
        hidden = self.hidden[order].reshape(count, -1, width)
        cell = self.cell[order].reshape(count, -1, width)

        for index in range(count):
            minibatch = {name: value[index] for name, value in shuffled.items()}
            minibatch["hidden"] = hidden[index]
            minibatch["cell"] = cell[index]
            yield minibatch


class CraftaxPQNTrainStep(TrainStep):
    """Model, environment, and optimizer for one Q-learning experiment."""

    class Config(Makes["CraftaxPQNTrainStep"], TrainStep.Config, kw_only=False):
        """Model, environment, and the Q-learning hyperparameters."""

        # ---- Inherited slots, re-defaulted for this recipe. ----

        model: RecurrentQNetwork.Config = field(  # pyright: ignore[reportIncompatibleVariableOverride] -- narrowing a Makeable slot to its concrete Config is the priml idiom; finalize reaches this model's own fields
            default_factory=RecurrentQNetwork.Config,
        )
        """Recurrent Q-network."""

        # ---- This recipe's own. ----

        env: CraftaxEnv.Config = field(default_factory=CraftaxEnv.Config)
        """Environment the rollout is collected from."""

        rollout_steps: int = 128
        """Environment steps per worker in one update."""

        num_epochs: int = 4
        """Optimization passes over each rollout."""

        num_minibatches: int = 4
        """Minibatches per pass; must divide the worker count."""

        learning_rate: float = 3e-4
        """Initial RAdam learning rate."""

        anneal_learning_rate: bool = True
        """Decay the rate linearly to zero across the run."""

        total_train_steps: int = 7_629
        """Updates in the run; the horizon for both schedules."""

        discount: float = 0.99
        """Reward discount factor."""

        trace_decay: float = 0.5
        """Q(lambda) multi-step mixing factor."""

        epsilon_start: float = 1.0
        """Initial probability of taking a random action."""

        epsilon_finish: float = 0.005
        """Floor the exploration rate decays to."""

        epsilon_decay_fraction: float = 0.1
        """Fraction of the run spent decaying the exploration rate."""

        max_grad_norm: float = 0.5
        """Global gradient-norm clip."""

        compile_recurrent_steps: bool = False
        """Compile the recurrent entry points with ``torch.compile``.

        Distinct from the base's ``compile``, which wraps ``forward``: a
        rollout never calls ``forward``."""

        seed: int = 0
        """Seed for exploration and minibatch shuffling."""

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
          config: Model, environment, and Q-learning settings.

        Raises:
          ValueError: A geometry or coefficient is invalid.

        """
        if config.rollout_steps <= 0 or config.num_epochs <= 0:
            raise ValueError("Rollout geometry must be positive")
        if config.num_minibatches <= 0:
            raise ValueError("There must be at least one minibatch")
        if config.total_train_steps <= 0:
            raise ValueError("total_train_steps must be positive")
        if config.discount < 0.0 or config.discount > 1.0:
            raise ValueError("discount must be between zero and one")
        if config.trace_decay < 0.0 or config.trace_decay > 1.0:
            raise ValueError("trace_decay must be between zero and one")
        if config.epsilon_decay_fraction <= 0.0 or config.epsilon_decay_fraction > 1.0:
            raise ValueError("epsilon_decay_fraction must be in (0, 1]")

        # The recipe's own optimizer, into the base's slot before the
        # base reads it, so there is one optimizer rather than an
        # inherited AdamW discarded for this one.
        # RAdam, as the reference uses: its warmup-free variance rectification
        # matters here because the first updates regress toward targets built
        # by a network that has seen almost nothing.
        config.optimizer = PartialConfig(torch.optim.RAdam, lr=config.learning_rate)
        # Weight initialization draws from the global stream, so the seed has
        # to reach it for a run to be reproducible from its config alone. The
        # BASE's build is what gets bracketed: rebuilding after it would leave
        # the optimizer holding the parameters of a discarded model.
        saved_rng = torch.get_rng_state()
        torch.manual_seed(config.seed)
        try:
            super().__init__(config)
        finally:
            torch.set_rng_state(saved_rng)
        self.config: CraftaxPQNTrainStep.Config = config
        self.env = config.env.make()
        model = self.model
        self._step: _StepFn = _compiled(
            model.step, enabled=config.compile_recurrent_steps
        )
        self._sequence: _SequenceFn = _compiled(
            model.sequence,
            enabled=config.compile_recurrent_steps,
        )
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(config.seed)
        self._observation: Tensor = self.env.reset()

        num_envs = int(self._observation.shape[0])
        if num_envs % config.num_minibatches:
            raise ValueError("num_minibatches must divide the worker count")
        self._state = self.model.initial_state(num_envs, device=self.device)
        # Declared, not merely assigned: these are rebound from the rollout
        # and from a checkpoint, and without a declaration the widest of those
        # assignments would set the attribute's type everywhere.
        self._previous_action: Tensor = torch.zeros(
            num_envs,
            dtype=torch.int64,
            device=self.device,
        )
        self._previous_done: Tensor = torch.zeros(
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
    @override
    def model(self) -> RecurrentQNetwork:
        """The Q-network this step trains, at its declared class."""
        model = self._model
        assert isinstance(model, RecurrentQNetwork)
        return model

    @property
    def steps_per_update(self) -> int:
        """Environment interactions consumed by one update."""
        workers = int(self._observation.shape[0])
        return workers * int(self.config.rollout_steps)

    @property
    @override
    def progress_learning_schedule(self) -> float:
        """Fraction of ``total_train_steps`` spent, in ``[0, 1]``.

        The exploration rate keeps its own step-indexed decay, which is a
        different quantity with a different horizon fraction.
        """
        spent = self.global_step / self.config.total_train_steps
        return 1.0 if spent > 1.0 else float(spent)

    @property
    def epsilon(self) -> float:
        """The exploration rate this update collects with."""
        return epsilon_at(
            self.global_step,
            total_updates=self.config.total_train_steps,
            start=self.config.epsilon_start,
            finish=self.config.epsilon_finish,
            decay_fraction=self.config.epsilon_decay_fraction,
        )

    @override
    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Pass the loop's batch through: the rollout is collected here."""
        return batch

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Collect a rollout and regress toward its Q(lambda) targets.

        Args:
          **batch: Ignored; the data comes from the environment.

        Returns:
          result: The final minibatch's loss and values, with the update's
            scalar diagnostics.

        """
        del batch
        epsilon = self.epsilon
        rollout = self.collect()
        self._set_learning_rate()
        # The timer brackets the update, so ``global_step`` and the budget
        # clock advance exactly as they do for every other recipe -- one
        # tick per PPO update, however many optimizer calls it makes.
        with self.timer_step:
            metrics = self._optimize(rollout)

        metrics["epsilon"] = epsilon
        metrics.update(self._episode_metrics())
        chosen = rollout.q_value.gather(-1, rollout.action[..., None])[..., 0]
        metrics["explained_variance"] = float(
            explained_variance(chosen.flatten(), rollout.target.flatten()),
        )
        return {
            "loss": metrics.pop("_loss_tensor"),
            "model": metrics.pop("_values"),
            "metrics": metrics,
        }

    @torch.no_grad()
    def collect(self) -> QRollout:
        """Run the epsilon-greedy policy and build its regression targets.

        Returns:
          rollout: The collected experience with Q(lambda) targets attached.

        """
        hidden, cell = self._state
        observations: list[Tensor] = []
        previous_actions: list[Tensor] = []
        previous_dones: list[Tensor] = []
        actions: list[Tensor] = []
        rewards: list[Tensor] = []
        dones: list[Tensor] = []
        values: list[Tensor] = []
        epsilon = self.epsilon

        # Collection reads the running statistics rather than updating them:
        # a rollout is inference, and folding it in would count every
        # observation twice per update.
        self.model.eval()
        try:
            for _ in range(self.config.rollout_steps):
                observations.append(self._observation)
                previous_actions.append(self._previous_action)
                previous_dones.append(self._previous_done)

                self._state, q_values = self._step(
                    self._state,
                    self._observation,
                    self._previous_action,
                    self._previous_done,
                )
                action = self._explore(q_values, epsilon=epsilon)

                actions.append(action)
                values.append(q_values)

                transition = self.env.step(action)
                self._observation = transition.observation
                self._previous_action = action
                self._previous_done = transition.done
                rewards.append(transition.reward)
                dones.append(transition.done)
                self._record_episodes(transition.reward, transition.done)

            _, bootstrap = self._step(
                self._state,
                self._observation,
                self._previous_action,
                self._previous_done,
            )
        finally:
            self.model.train()

        q_value = torch.stack(values)
        target = q_lambda_targets(
            rewards=torch.stack(rewards),
            q_values=torch.cat((q_value, bootstrap[None])),
            dones=torch.stack(dones),
            discount=self.config.discount,
            trace_decay=self.config.trace_decay,
        )
        return QRollout(
            observation=torch.stack(observations),
            previous_action=torch.stack(previous_actions),
            previous_done=torch.stack(previous_dones),
            hidden=hidden,
            cell=cell,
            action=torch.stack(actions),
            reward=torch.stack(rewards),
            q_value=q_value,
            target=target,
        )

    @override
    def train_loss(self, **batch: object) -> TrainStepOutput:
        """Score a rollout without optimizing.

        Args:
          **batch: Ignored; the data comes from the environment.

        Returns:
          result: The loss and values of one freshly collected rollout.

        """
        del batch
        rollout = self.collect()
        minibatch = next(rollout.minibatches(count=1, generator=self._generator))
        loss, values = self._loss(minibatch)
        return {"loss": loss.detach(), "model": values.detach()}

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Score a rollout in evaluation mode.

        Args:
          **batch: Ignored; the data comes from the environment.

        Returns:
          result: The loss and values of one freshly collected rollout.

        """
        self.model.eval()
        try:
            return self.train_loss(**batch)
        finally:
            self.model.train()

    @override
    def call_eval(self, *, observation: Tensor, **_batch: object) -> Tensor:
        """Return action values for a batch of observations.

        Args:
          observation: Batched observations, ``[batch, observation_size]``.
          **_batch: Ignored; only ``observation`` is scored.

        Returns:
          q_values: Value of each action, computed with a fresh state.

        """
        self.model.eval()
        try:
            with torch.no_grad():
                return self.model.forward(observation)
        finally:
            self.model.train()

    @override
    def on_epoch_end(self) -> None:
        """Nothing to flush: every update completes within one step."""

    @override
    def state_dict(self) -> dict[str, Any]:
        """Return model, optimizer, environment, state, and counters."""
        hidden, cell = self._state
        return super().state_dict() | {
            "env": self.env.state_dict(),
            "generator": self._generator.get_state(),
            "observation": self._observation,
            "hidden": hidden,
            "cell": cell,
            "previous_action": self._previous_action,
            "previous_done": self._previous_done,
            "episode_return": self._episode_return,
            "episode_length": self._episode_length,
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any], **kwargs: Any) -> None:
        """Restore everything :meth:`state_dict` saved."""
        super().load_state_dict(state_dict, **kwargs)
        self.env.load_state_dict(state_dict["env"])
        self._generator.set_state(state_dict["generator"])
        self._observation = _tensor(state_dict["observation"])
        self._state = (_tensor(state_dict["hidden"]), _tensor(state_dict["cell"]))
        self._previous_action = _tensor(state_dict["previous_action"])
        self._previous_done = _tensor(state_dict["previous_done"])
        self._episode_return = _tensor(state_dict["episode_return"])
        self._episode_length = _tensor(state_dict["episode_length"])

    def _explore(self, q_values: Tensor, *, epsilon: float) -> Tensor:
        """Take the greedy action, except at rate ``epsilon``."""
        greedy = q_values.argmax(dim=-1)
        random = torch.randint(
            0,
            q_values.shape[-1],
            greedy.shape,
            generator=self._generator,
            device=q_values.device,
        )
        explore = (
            torch.rand(greedy.shape, generator=self._generator, device=q_values.device)
            < epsilon
        )
        return torch.where(explore, random, greedy)

    def _optimize(self, rollout: QRollout) -> dict[str, Any]:
        """Take every configured pass over the rollout."""
        metrics: dict[str, Any] = {}
        for _ in range(self.config.num_epochs):
            for minibatch in rollout.minibatches(
                count=self.config.num_minibatches,
                generator=self._generator,
            ):
                loss, values = self._loss(minibatch)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                metrics = {
                    "q_loss": float(loss.detach()),
                    "q_mean": float(values.detach().mean()),
                    "grad_norm": float(grad_norm.detach()),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "_loss_tensor": loss.detach(),
                    "_values": values.detach(),
                }
        return metrics

    def _loss(self, minibatch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Regress the taken actions' values toward their targets."""
        _, q_values = self._sequence(
            (minibatch["hidden"], minibatch["cell"]),
            minibatch["observation"],
            minibatch["previous_action"],
            minibatch["previous_done"],
        )
        chosen = q_values.gather(-1, minibatch["action"][..., None].long())[..., 0]
        # Plain squared error: there is no policy ratio here to trust-region,
        # so the clipping the PPO steps do would have nothing to clip.
        loss = ((chosen - minibatch["target"]) ** 2).mean()
        return loss, q_values.flatten(0, 1)

    def _set_learning_rate(self) -> None:
        """Anneal the rate linearly across the configured horizon."""
        if not self.config.anneal_learning_rate:
            return
        multiplier = linear(self.progress_learning_schedule)
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate * multiplier

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


def _compiled(function: _StepFn, *, enabled: bool) -> _StepFn:
    """Compile one bound method, or return it untouched.

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
