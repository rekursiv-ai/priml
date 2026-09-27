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

On a GPU the policy's action step and the whole update -- every epoch, every
minibatch, every optimizer step -- are replayed as CUDA graphs (``cuda_graph``),
so an update costs one launch instead of Python per minibatch. What a graph
reads and writes must keep its memory, which is why the rollout, the policy's
outputs, and the update's results live in buffers allocated once and
overwritten in place.
"""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, Self, cast, override

from configgle import Makes, PartialConfig
from torch import Tensor

import torch

from priml.baselines.craftax.cuda_graph import CudaGraphed
from priml.baselines.craftax.env import CraftaxEnv
from priml.baselines.craftax.evaluation import (
    evaluation_mode,
    evaluation_transaction,
)
from priml.baselines.craftax.game.constants import Action
from priml.baselines.craftax.game.observation import observation_size
from priml.baselines.craftax.model import ActorCritic
from priml.lib.custom_json import ListCodec
from priml.loss.policy_gradient import (
    ClippedPolicyLoss,
    categorical_entropy,
    clipped_policy_loss,
)
from priml.math.advantage import explained_variance, generalized_advantage
from priml.math.schedules import linear
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from priml.baselines.craftax.data import EvaluationActor
    from priml.train.custom_types import TrainStepOutput


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

    @classmethod
    def zeros(
        cls,
        *,
        steps: int,
        envs: int,
        observation_size: int,
        device: torch.device,
    ) -> Self:
        """Allocate a rollout to fill one step at a time, reused every update.

        Args:
          steps: Environment steps per worker.
          envs: Parallel workers.
          observation_size: Width of one observation.
          device: Device the rollout lives on.

        Returns:
          rollout: Zero-filled storage.

        """
        shape = (steps, envs)
        return cls(
            observation=torch.zeros((*shape, observation_size), device=device),
            action=torch.zeros(shape, dtype=torch.int64, device=device),
            log_prob=torch.zeros(shape, device=device),
            value=torch.zeros(shape, device=device),
            reward=torch.zeros(shape, device=device),
            done=torch.zeros(shape, dtype=torch.bool, device=device),
            advantage=torch.zeros(shape, device=device),
            target=torch.zeros(shape, device=device),
        )

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


class CraftaxTrainStep(TrainStep):
    """Model, environment, and optimizer for one PPO experiment."""

    class Config(
        Makes["CraftaxTrainStep"],
        TrainStep.Config[ActorCritic.Config],
        kw_only=True,
    ):
        """Model, environment, and the PPO hyperparameters."""

        # ---- Inherited slots, re-defaulted for this recipe. ----

        model: ActorCritic.Config = field(default_factory=ActorCritic.Config)
        """Policy and value network."""

        # ---- This recipe's own. ----

        env: CraftaxEnv.Config = field(default_factory=CraftaxEnv.Config)
        """Environment the rollout is collected from."""

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

        seed: int = 0
        """Seed for action sampling and minibatch shuffling."""

        cuda_graphs: bool = True
        """Replay the action step and the whole update as CUDA graphs on a GPU.

        The graphed update runs Adam in its capturable form, which keeps the
        step count and learning rate on the device and rounds differently from
        the eager optimizer in the last bit: a run with this on does not
        reproduce one with it off bit for bit. Off, and on a CPU, the update
        runs eagerly with the eager optimizer."""

        @override
        def finalize(self) -> Self:
            # The environment renders the observations the model consumes and
            # names the actions it scores, so the two must agree. Deriving the
            # geometry here means an experiment that changes the environment
            # cannot forget to resize the network.
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

        # The recipe's own optimizer, put into the base's slot before the base
        # reads it, so there is one optimizer rather than an inherited AdamW
        # discarded for this one. ``learning_rate`` stays the field an
        # experiment sets; the base records it as each group's ``initial_lr``.
        config.optimizer = PartialConfig(
            torch.optim.Adam,
            lr=config.learning_rate,
            eps=1e-5,
        )
        # Weight initialization draws from the global stream, so the seed has
        # to reach it for a run to be reproducible from its config alone. The
        # stream is restored afterwards, leaving whatever the caller had --
        # which is why the base's build is bracketed rather than followed by a
        # second, seeded one.
        saved_rng = torch.get_rng_state()
        torch.manual_seed(config.seed)
        try:
            super().__init__(config)
        finally:
            torch.set_rng_state(saved_rng)
        self.config = config
        self.env = config.env.make()
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(config.seed)
        self._observation: Tensor = self.env.reset()
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

        workers = self._observation.shape[0]
        self._rollout = Rollout.zeros(
            steps=config.rollout_steps,
            envs=workers,
            observation_size=self._observation.shape[1],
            device=self.device,
        )
        # What the procedures read and store; see ``_act`` and ``_update``.
        self._policy_input = torch.zeros_like(self._observation)
        self._action = torch.zeros(workers, dtype=torch.int64, device=self.device)
        self._log_prob = torch.zeros(workers, device=self.device)
        self._value = torch.zeros(workers, device=self.device)
        self._update_scalars = torch.zeros(6, device=self.device)
        self._update_loss = torch.zeros((), device=self.device)
        self._update_logits = torch.zeros(0, device=self.device)

        self._cuda_graphs = config.cuda_graphs and self.device.type == "cuda"
        if self._cuda_graphs:
            _make_capturable(
                self._adam.param_groups,
                self._adam.state,
                self.device,
            )
        self._act_procedure = self._procedure(self._act)
        self._update_procedure = self._procedure(self._update)

    @property
    @override
    def model(self) -> ActorCritic:
        """The policy this step trains, at its declared class."""
        model = self._model
        assert isinstance(model, ActorCritic)
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
        rate = self._set_learning_rate()
        # The timer brackets the update, so ``global_step`` and the budget
        # clock advance exactly as they do for every other recipe -- one
        # tick per PPO update, however many optimizer calls it makes.
        with self.timer_step:
            metrics: dict[str, float | Tensor] = self._optimize(rate)

        metrics.update(self._episode_metrics())
        metrics["explained_variance"] = float(
            explained_variance(rollout.value.flatten(), rollout.target.flatten()),
        )
        loss = metrics.pop("_loss_tensor")
        logits = metrics.pop("_logits")
        assert isinstance(loss, Tensor)
        assert isinstance(logits, Tensor)
        return {"loss": loss, "model": logits, "metrics": metrics}

    @torch.no_grad()
    def collect(self) -> Rollout:
        """Run the current policy for a fixed number of steps.

        Returns:
          rollout: The collected experience, already scored with advantages.
            The step's own storage, overwritten by the next collection.

        """
        rollout = self._rollout
        for step in range(self.config.rollout_steps):
            self._policy_input.copy_(self._observation)
            self._act_procedure()
            rollout.observation[step].copy_(self._observation)
            rollout.action[step].copy_(self._action)
            rollout.log_prob[step].copy_(self._log_prob)
            rollout.value[step].copy_(self._value)

            transition = self.env.step(self._action)
            self._observation = transition.observation
            self._done = transition.done
            rollout.reward[step].copy_(transition.reward)
            rollout.done[step].copy_(transition.done)
            self._record_episodes(transition.reward, transition.done)

        _, last_value = self.model(self._observation)
        advantage, target = generalized_advantage(
            rewards=rollout.reward,
            values=rollout.value,
            dones=rollout.done,
            last_value=last_value,
            discount=self.config.discount,
            trace_decay=self.config.trace_decay,
        )
        rollout.advantage.copy_(advantage)
        rollout.target.copy_(target)
        return rollout

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
        """Score a rollout in evaluation mode, leaving training state intact.

        On-policy scoring has to interact with the world -- there is no held
        out batch to read -- so this collects a rollout like ``train_loss``.
        What it must not do is KEEP the consequences: ``collect`` advances
        ``_observation``/``_done`` and banks finished episodes, so an eval
        pass silently moved the world the next update trains from and folded
        its own episodes into the return/length averages reported as training
        progress. Snapshot and restore both.

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
          logits: Unnormalized action scores.

        """
        if args:
            raise ValueError("Expected not args.")
        observation = batch["observation"]
        assert isinstance(observation, Tensor)
        with evaluation_mode(self.model), torch.no_grad():
            logits, _ = self.model.forward(observation)
        return logits

    def make_evaluation_actor(self) -> EvaluationActor:
        """Build a stateless sampling actor over the live policy.

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
        """The base state plus the environment, its generator, and the rollout cursor."""

        env: CraftaxEnv.StateDict
        generator: Tensor
        observation: Tensor
        done: Tensor
        episode_return: Tensor
        episode_length: Tensor
        finished_returns: list[float]
        finished_lengths: list[int]

    @override
    def state_dict(self) -> StateDict:
        """Return model, optimizer, environment, and counters."""
        return {
            **super().state_dict(),
            "env": self.env.state_dict(),
            "generator": self._generator.get_state(),
            "observation": self._observation,
            "done": self._done,
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
        state = cast(CraftaxTrainStep.StateDict, state_dict)
        self.env.load_state_dict(state["env"])
        self._generator.set_state(state["generator"])
        self._observation = state["observation"]
        self._done = state["done"]
        self._episode_return = state["episode_return"]
        self._episode_length = state["episode_length"]
        self._finished_returns = list(state["finished_returns"])
        self._finished_lengths = list(state["finished_lengths"])
        if self._cuda_graphs:
            # Loading replaced the optimizer's state tensors, which the captured
            # update still addresses, so it is captured afresh.
            _make_capturable(
                self._adam.param_groups,
                self._adam.state,
                self.device,
            )
            self._update_procedure = self._procedure(self._update)

    @property
    def _adam(self) -> torch.optim.Adam:
        """The optimizer, at the class ``__init__`` builds it as."""
        optimizer = self.optimizer
        assert isinstance(optimizer, torch.optim.Adam)
        return optimizer

    def _procedure(self, procedure: Callable[[], None]) -> Callable[[], None]:
        """Return ``procedure``, replayed as a CUDA graph when graphs are on."""
        if self._cuda_graphs:
            return CudaGraphed(procedure, generators=(self._generator,))
        return procedure

    # A CUDA-graph procedure: reads ``_policy_input``, stores the step's outputs.
    def _act(self) -> None:
        """Sample every worker's action from the current policy."""
        logits, value = self.model(self._policy_input)
        log_probs_all = logits.log_softmax(-1)
        # Sampled through the step's own generator rather than
        # ``Categorical.sample``, which draws from the global stream: a run must
        # replay from its seed regardless of what else in the process has
        # consumed randomness.
        action = torch.multinomial(
            log_probs_all.exp(),
            1,
            generator=self._generator,
        ).squeeze(-1)
        # Stored rather than copied into buffers: a tensor made during capture
        # lives in the graph's memory, and every replay refills it.
        self._action = action
        self._log_prob = log_probs_all.gather(-1, action[:, None])[:, 0]
        self._value = value

    def _optimize(self, rate: float) -> dict[str, float | Tensor]:
        """Take every configured pass over the rollout and report the last one."""
        self._update_procedure()
        policy, value, entropy, approx_kl, clip_fraction, grad_norm = ListCodec.coerce(
            self._update_scalars.tolist(),
            float,
        )
        return {
            "policy_loss": policy,
            "value_loss": value,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "grad_norm": grad_norm,
            "learning_rate": rate,
            "_loss_tensor": self._update_loss.clone(),
            "_logits": self._update_logits.clone(),
        }

    # A CUDA-graph procedure: reads the rollout, stores the last minibatch's results.
    def _update(self) -> None:
        """Take every configured pass over the stored rollout."""
        for _ in range(self.config.num_epochs):
            for minibatch in self._rollout.minibatches(
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
                # Kept on the device, not read out per minibatch: each read is a
                # host sync, and only the last minibatch's are reported.
                self._update_scalars = torch.stack(
                    (
                        terms.policy,
                        terms.value,
                        terms.entropy,
                        terms.approx_kl,
                        terms.clip_fraction,
                        grad_norm,
                    ),
                ).detach()
                self._update_loss = loss.detach()
                self._update_logits = logits.detach()

    def _loss(
        self,
        minibatch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, ClippedPolicyLoss]:
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

    def _set_learning_rate(self) -> float:
        """Anneal the rate linearly across the configured horizon, returning it."""
        rate = self.config.learning_rate
        if self.config.anneal_learning_rate:
            rate *= linear(self.progress_learning_schedule)
        for group in self.optimizer.param_groups:
            _write_rate(group, rate)
        return rate

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


# A captured update reads its rate from device memory, so the rate is written into
# that memory; rebinding the group's entry would go unseen by the replay.
def _write_rate(group: dict[str, object], rate: float) -> None:
    """Set one parameter group's learning rate, in place where it is a tensor."""
    current = group["lr"]
    if isinstance(current, Tensor):
        current.fill_(rate)
    else:
        group["lr"] = rate


# ``torch.optim.Adam`` reads both flags per group at every step, so flipping them before
# the first step -- or after a checkpoint restored an eager run's -- is enough.
def _make_capturable(
    param_groups: list[dict[str, object]],
    state: Mapping[Tensor, dict[str, object]],
    device: torch.device,
) -> None:
    """Keep Adam's step counts and learning rates on the device, as capture needs."""
    for group in param_groups:
        group["capturable"] = True
        group["lr"] = torch.as_tensor(group["lr"], device=device)
    for per_parameter in state.values():
        step = per_parameter.get("step")
        if isinstance(step, Tensor):
            per_parameter["step"] = step.to(device=device, dtype=torch.float32)


class _EvaluationActor:
    """Sample a feed-forward policy without owning recurrent state."""

    def __init__(
        self,
        model: ActorCritic,
        *,
        observation_size: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.observation_size = observation_size
        self.device = device

    def reset(self, *, num_envs: int, device: torch.device) -> None:
        del num_envs, device

    def act(
        self,
        observation: Tensor,
        previous_done: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        del previous_done
        logits, _ = self.model(observation)
        return torch.multinomial(
            logits.softmax(-1),
            1,
            generator=generator,
        ).squeeze(-1)
