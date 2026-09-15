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

from collections.abc import Callable, Mapping
from dataclasses import field
from typing import TYPE_CHECKING, Self, cast, override

from configgle import Makes, PartialConfig
from torch import Tensor

import torch

from priml.baselines.craftax.data import EvaluationActor
from priml.baselines.craftax.env import CraftaxEnv
from priml.baselines.craftax.evaluation import (
    evaluation_mode,
    evaluation_transaction,
)
from priml.baselines.craftax.game.constants import Action
from priml.baselines.craftax.game.observation import observation_size
from priml.baselines.craftax.rnn import ActorCriticRNN
from priml.lib.custom_json import ListCodec
from priml.loss.policy_gradient import (
    ClippedPolicyLoss,
    categorical_entropy,
    clipped_policy_loss,
)
from priml.math.advantage import explained_variance, generalized_advantage
from priml.math.schedules import linear
from priml.optimizers.lr import learning_rate
from priml.train.custom_types import TrainStepOutput
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from collections.abc import Iterator


type _Callable = Callable[..., object]
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
            count,
            -1,
            self.initial_state.shape[-1],
        )

        for index in range(count):
            minibatch = {name: value[index] for name, value in shuffled.items()}
            minibatch["initial_state"] = states[index]
            yield minibatch


def _split_environments(value: Tensor, *, order: Tensor, count: int) -> Tensor:
    """Shuffle whole trajectories and expose a leading minibatch axis."""
    shuffled = value[:, order]
    grouped = shuffled.reshape(value.shape[0], count, -1, *value.shape[2:])
    return grouped.transpose(0, 1)


class CraftaxRNNTrainStep(TrainStep):
    """Model, environment, and optimizer for one recurrent PPO experiment."""

    class Config(
        Makes["CraftaxRNNTrainStep"],
        TrainStep.Config[ActorCriticRNN.Config],
        kw_only=True,
    ):
        """Model, environment, and the PPO hyperparameters."""

        # ---- Inherited slots, re-defaulted for this recipe. ----

        model: ActorCriticRNN.Config = field(default_factory=ActorCriticRNN.Config)
        """Recurrent policy and value network."""

        # ---- This recipe's own. ----

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

        compile_recurrent_steps: bool = False
        """Compile the recurrent entry points with ``torch.compile``.

        Distinct from the base's ``compile``, which wraps ``forward``: a
        rollout never calls ``forward``."""

        seed: int = 0
        """Seed for action sampling and minibatch shuffling."""

        @override
        def finalize(self) -> Self:
            # The environment renders the observations and names the actions,
            # so an experiment that changes it cannot forget to resize the net.
            self.model.observation_size = observation_size(self.env.view)
            self.model.num_actions = len(Action)
            return super().finalize()

    config: Config

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
        if config.discount < 0.0 or config.discount > 1.0:
            raise ValueError("discount must be between zero and one")
        if config.trace_decay < 0.0 or config.trace_decay > 1.0:
            raise ValueError("trace_decay must be between zero and one")
        if config.clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive")

        # The recipe's own optimizer, into the base's slot before the
        # base reads it, so there is one optimizer rather than an
        # inherited AdamW discarded for this one.
        config.optimizer = PartialConfig(
            torch.optim.Adam,
            lr=config.learning_rate,
            eps=1e-5,
        )
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
        self.config = config
        self.env = config.env.make()
        model = self.model
        # The compiled handles wrap the two RECURRENT entry points, not
        # ``forward``: a rollout never calls ``forward``, so compiling the
        # module would leave the hot path interpreted.
        self._step = _compiled(model.step, enabled=config.compile_recurrent_steps)
        self._sequence = _compiled(
            model.sequence,
            enabled=config.compile_recurrent_steps,
        )
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(config.seed)
        self._observation: Tensor = self.env.reset()

        num_envs = int(self._observation.shape[0])
        if num_envs % config.num_minibatches:
            raise ValueError("num_minibatches must divide the worker count")
        self._state: Tensor
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
    @override
    def model(self) -> ActorCriticRNN:
        """The policy this step trains, at its declared class."""
        model = self._model
        assert isinstance(model, ActorCriticRNN)
        return model

    @property
    def steps_per_update(self) -> int:
        """Environment interactions consumed by one update."""
        workers = int(self._observation.shape[0])
        return workers * int(self.config.rollout_steps)

    @property
    @override
    def progress_learning_schedule(self) -> float:
        """Fraction of ``total_train_steps`` spent, in ``[0, 1]``."""
        spent = self.global_step / self.config.total_train_steps
        return 1.0 if spent > 1.0 else float(spent)

    @override
    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        """Pass the loop's batch through: the rollout is collected here."""
        return batch

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
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
        # The timer brackets the update, so ``global_step`` and the budget
        # clock advance exactly as they do for every other recipe -- one
        # tick per PPO update, however many optimizer calls it makes.
        with self.timer_step:
            metrics = self._optimize(rollout)

        metrics.update(self._episode_metrics())
        metrics["explained_variance"] = float(
            explained_variance(rollout.value.flatten(), rollout.target.flatten()),
        )
        loss = metrics.pop("_loss_tensor")
        logits = metrics.pop("_logits")
        assert isinstance(loss, Tensor)
        assert isinstance(logits, Tensor)
        typed_metrics = {
            name: value
            for name, value in metrics.items()
            if isinstance(value, (float, Tensor))
        }
        return {"loss": loss, "model": logits, "metrics": typed_metrics}

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

    @override
    def train_loss(self, **batch: object) -> TrainStepOutput:
        """Score a rollout without optimizing.

        Args:
          **batch: Ignored; the data comes from the environment.

        Returns:
          result: The loss and logits of one freshly collected rollout.

        """
        del batch
        rollout = self.collect()
        minibatch = next(rollout.minibatches(count=1, generator=self._generator))
        loss, logits, _terms = self._loss(minibatch)
        return {"loss": loss.detach(), "model": logits.detach()}

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Score a rollout in evaluation mode.

        Args:
          **batch: Ignored; the data comes from the environment.

        Returns:
          result: The loss and logits of one freshly collected rollout.

        """
        with evaluation_transaction(
            model=self.model,
            save=self.state_dict,
            restore=self.load_state_dict,
        ):
            return self.train_loss(**batch)

    @override
    def call_eval(self, *args: object, **batch: object) -> Tensor:
        """Return action logits for a batch of observations.

        Args:
          *args: Unused; the base signature admits positionals.
          **batch: Batch fields; only ``observation`` is scored.

        Returns:
          logits: Unnormalized action scores, computed with a fresh state.

        """
        assert not args
        observation = batch["observation"]
        assert isinstance(observation, Tensor)
        with evaluation_mode(self.model), torch.no_grad():
            logits, _ = self.model.forward(observation)
        return logits

    def make_evaluation_actor(self) -> EvaluationActor:
        """Build an actor with isolated GRU state.

        Returns:
          result: The EvaluationActor.

        """
        return _EvaluationActor(
            self.model,
            observation_size=self.env.observation_size,
            device=self.device,
        )

    @override
    def on_epoch_end(self) -> None:
        """Nothing to flush: every update completes within one step."""

    class StateDict(TrainStep.StateDict):
        """The base state plus the environment, the GRU state, and the cursor."""

        env: CraftaxEnv.StateDict
        generator: Tensor
        observation: Tensor
        recurrent_state: Tensor
        previous_done: Tensor
        episode_return: Tensor
        episode_length: Tensor
        finished_returns: list[float]
        finished_lengths: list[int]

    @override
    def state_dict(self) -> StateDict:
        """Return model, optimizer, environment, state, and counters."""
        return {
            **super().state_dict(),
            "env": self.env.state_dict(),
            "generator": self._generator.get_state(),
            "observation": self._observation,
            "recurrent_state": self._state,
            "previous_done": self._previous_done,
            "episode_return": self._episode_return,
            "episode_length": self._episode_length,
            "finished_returns": list(self._finished_returns),
            "finished_lengths": list(self._finished_lengths),
        }

    @override
    def load_state_dict(
        self,
        state_dict: Mapping[str, object],
        *,
        strict: bool = True,
        load_optimizer: bool = True,
        remap: Callable[[Mapping[str, Tensor]], Mapping[str, Tensor]] | None = None,
    ) -> None:
        """Restore everything :meth:`state_dict` saved."""
        super().load_state_dict(
            state_dict,
            strict=strict,
            load_optimizer=load_optimizer,
            remap=remap,
        )
        state = cast(CraftaxRNNTrainStep.StateDict, state_dict)
        self.env.load_state_dict(state["env"])
        self._generator.set_state(state["generator"])
        self._observation = state["observation"]
        self._state = state["recurrent_state"]
        self._previous_done = state["previous_done"]
        self._episode_return = state["episode_return"]
        self._episode_length = state["episode_length"]
        self._finished_returns = list(state["finished_returns"])
        self._finished_lengths = list(state["finished_lengths"])

    def _optimize(self, rollout: RecurrentRollout) -> dict[str, object]:
        """Take every configured pass over the rollout."""
        metrics: dict[str, object] = {}
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
                    "learning_rate": learning_rate(self.optimizer),
                    "_loss_tensor": loss.detach(),
                    "_logits": logits.detach(),
                }
        return metrics

    def _loss(
        self, minibatch: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, ClippedPolicyLoss]:
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
        multiplier = linear(self.progress_learning_schedule)
        for group in self.optimizer.param_groups:
            group["lr"] = self.config.learning_rate * multiplier

    def _record_episodes(self, reward: Tensor, done: Tensor) -> None:
        """Accumulate per-worker returns and bank the finished ones."""
        self._episode_return = self._episode_return + reward
        self._episode_length = self._episode_length + 1
        if bool(done.any()):
            self._finished_returns.extend(
                ListCodec.coerce(self._episode_return[done].tolist(), float),
            )
            self._finished_lengths.extend(
                ListCodec.coerce(self._episode_length[done].tolist(), int),
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


class _EvaluationActor:
    """Sample a recurrent policy while owning its GRU state."""

    def __init__(
        self,
        model: ActorCriticRNN,
        *,
        observation_size: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.observation_size = observation_size
        self.device = device
        self._state: Tensor | None = None

    def reset(self, *, num_envs: int, device: torch.device) -> None:
        self._state = self.model.initial_state(num_envs, device=device)

    def act(
        self,
        observation: Tensor,
        previous_done: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        assert self._state is not None
        self._state, logits, _ = self.model.step(
            self._state,
            observation,
            previous_done,
        )
        return torch.multinomial(
            logits.softmax(-1),
            1,
            generator=generator,
        ).squeeze(-1)


# Compiling the recurrent entry points rather than the module is what keeps the rollout
# on the compiled path: a rollout calls ``step``, never ``forward``.
def _compiled[FunctionT: _Callable](function: FunctionT, *, enabled: bool) -> FunctionT:
    """Compile one bound method, or return it untouched."""
    return cast(FunctionT, torch.compile(function) if enabled else function)
