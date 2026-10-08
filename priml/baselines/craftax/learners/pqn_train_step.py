"""Q-learning on fresh experience: no buffer, no target network.

One update collects a rollout with an epsilon-greedy policy, builds Q(lambda)
targets from the network's OWN values, and regresses toward them. That is the
whole algorithm -- there is no second network holding the target still, and
nothing is stored between updates.

What makes that stable is covered in :mod:`pqn`: enough parallel workers to
decorrelate samples, batch renormalization to absorb the shift as the policy
changes, and a multi-step target that leans less on any single bootstrap.

Two things differ from the PPO epochs beside it. The loss is half the squared
error rather than a clipped surrogate, because there is no policy ratio to
trust-region. And the targets are built ONCE per rollout, before any
optimization: recomputing them from the updated network each epoch would be
chasing a value the network had already moved.

The port's actor collects each rollout (``rollout.Rollout``, one slot, the
current weights, the network as its own policy). The step then rescores the
stored rollout once, without a gradient and with the running normalization the
actor used, from the carry the rollout began with: that is every action's
value under the weights that collected it. The rollout stores beside each
observation the reward and terminal that arrived with it, so transition ``t``
reads row ``t + 1``'s; the last transition's arrive in the env's buffers after
the rollout, beside the observation it leads to, which the rescore's final
carry values for the bootstrap.

:meth:`CraftaxPQNTrainStep.collect` returns that rollout, values and targets
attached, before any optimization reads it, so a subclass that extends it
sees each update's values, rewards and terminals without touching the update.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Self, TypedDict, cast, override

import math
import time

from configgle import Fig, Makeable, Makes, PartialConfig
from torch import Tensor

import numpy as np
import torch

from priml.baselines.craftax.data import CraftaxRollouts
from priml.baselines.craftax.env import CraftaxEnv
from priml.baselines.craftax.evaluation import Evaluation
from priml.baselines.craftax.game.state import ATN_DIM, LOG_DTYPE
from priml.baselines.craftax.learners.update import shuffles
from priml.baselines.craftax.metric import (
    LOG_FIELDS,
    aggregate_logs,
    report_metrics,
)
from priml.baselines.craftax.policies.pqn import (
    EpsilonGreedy,
    GreedySampler,
    PreviousAction,
    RecurrentQNetwork,
)
from priml.baselines.craftax.rollout import (
    EnvBuffers,
    FeatureSource,
    Rollout,
    RolloutStorage,
    Sampler,
)
from priml.baselines.craftax.train_step import ProgressSchedule, RateSchedule
from priml.lib.codec import from_plain
from priml.math.advantage import explained_variance, q_lambda_targets
from priml.math.schedules import linear
from priml.optimizers.lr import remember_initial_lrs
from priml.timer import CheckpointableStepTimer
from priml.train.checkpointer import Checkpointer
from priml.train.custom_types import CheckpointerProtocol
from priml.train.parallelism import NoParallel
from priml.train.tracker import AsyncTracker, TrackerList, WandbTracker
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch.optim.optimizer import StateDict as OptimizerState

    from priml.train.custom_types import TrainStepOutput


@dataclass(frozen=True, slots=True, kw_only=True)
class QRollout:
    """One rollout, time-major, its values and Q(lambda) targets built.

    The first four are views of the rollout's slot, which the next rollout
    rewrites; the rest are the step's own.

    Attributes:
      observations: ``[horizon, agents, observation_size]``, what the actor read.
      previous_actions: ``[horizon, agents, 1]``, the feature each step read.
      starts: ``[horizon, agents]``, nonzero where the step's carry was reset.
      initial_states: ``[2, agents, width]``, the carry at step 0.
      actions: ``[horizon, agents]`` int64, the actions taken.
      q_values: ``[horizon, agents, num_actions]``, every action's value under
        the weights that collected the rollout.
      bootstrap: ``[agents, num_actions]``, the values of the observation the
        last step leads to.
      rewards: ``[horizon, agents]`` fp32, each transition's reward.
      dones: ``[horizon, agents]`` bool, whether each transition ended its
        episode.
      targets: ``[horizon, agents]``, the Q(lambda) regression targets.

    """

    observations: Tensor
    previous_actions: Tensor
    starts: Tensor
    initial_states: Tensor
    actions: Tensor
    q_values: Tensor
    bootstrap: Tensor
    rewards: Tensor
    dones: Tensor
    targets: Tensor


class TrainingEnv(EnvBuffers, Protocol):
    """What the step needs of its environments: the actor's buffers, and their upkeep.

    Attributes:
      stats: Each environment's records; their ``log`` field holds its
        running episode log, ``LOG_DTYPE``.

    """

    stats: NDArray[np.void]

    def reset(self) -> None:
        """Start every environment afresh."""
        ...

    def prepare_rollout(self) -> None:
        """Ready the environments for a rollout's first step."""
        ...

    def state_dict(self) -> dict[str, Tensor]:
        """Return everything a step reads and writes."""
        ...

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Copy a :meth:`state_dict` into the live environments."""
        ...

    def close(self) -> None:
        """Stop the environments' threads."""
        ...


class TrainingEnvConfig(Makeable[TrainingEnv], Protocol):
    """An environments' config: it builds them and states their geometry."""

    @property
    def num_envs(self) -> int:
        """Environments in total."""
        ...

    @property
    def observation_size(self) -> int:
        """Floats per observation."""
        ...


class CraftaxPQNTrainStep:
    """Recurrent Q-learning: each train step collects one rollout and regresses on it.

    It implements the training loop's step protocol directly, as
    ``train_step.CraftaxTrainStep`` does, and shares its env, actor and
    evaluation; only the learning is its own. Its defaults are the reference's
    recipe.
    """

    class Config(Fig["CraftaxPQNTrainStep"]):
        """The Q-network, its optimizer and schedule, the environments and the update."""

        model: RecurrentQNetwork.Config = field(
            default_factory=RecurrentQNetwork.Config,
        )
        """The Q-network; ``finalize`` sizes it from the env."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=lambda: PartialConfig(torch.optim.RAdam, lr=3e-4),
        )
        """Builds the optimizer from the network's ``parameters()``. RAdam, as
        the reference uses: its rectified variance matters because the first
        updates regress toward targets built by a network that has seen almost
        nothing. Each group's ``lr`` is the schedule's base."""

        schedule: Makeable[RateSchedule] = field(
            default_factory=lambda: ProgressSchedule.Config(
                curve=PartialConfig(linear),
            ),
        )
        """Each update's rate over ``train_budget_steps``: linear to zero."""

        env: TrainingEnvConfig = field(default_factory=CraftaxEnv.Config)
        """The training environments; ``CraftaxEnv`` unless a test fakes them."""

        sampler: Makeable[Sampler] = field(default_factory=EpsilonGreedy.Config)
        """The exploring actor's action streams; ``finalize`` gives an
        :class:`~priml.baselines.craftax.policies.pqn.EpsilonGreedy` the update's
        length and the run's."""

        feature: Makeable[FeatureSource] = field(default_factory=PreviousAction.Config)
        """The previous action, the network's second input, which the actor
        and the evaluation compute and the rollout stores."""

        rollout: Rollout.Config = field(
            default_factory=lambda: Rollout.Config(num_slots=1, horizon=128),
        )
        """One slot: each rollout is collected with the weights that learn
        from it. ``horizon`` is the steps per update."""

        evaluation: Evaluation.Config = field(default_factory=Evaluation.Config)
        """The fresh trainer an evaluation plays in; ``finalize`` fills the
        unset parts from training's, with the greedy sampler."""

        num_epochs: int = 4
        """Optimization passes over each rollout."""

        num_minibatches: int = 4
        """Minibatches of whole trajectories per pass; must divide the
        environments."""

        discount: float = 0.99
        """Reward discount factor."""

        trace_decay: float = 0.5
        """Q(lambda) multi-step mixing factor."""

        max_grad_norm: float = 0.5
        """Global gradient-norm clip."""

        seed: int = 0
        """Seeds every update's trajectory shuffles, with the update's index."""

        train_budget_steps: float = math.inf
        """Updates the run trains, the schedules' horizon; it must be set."""

        parallelism: NoParallel.Config = field(default_factory=NoParallel.Config)
        """The one device the network, the actor and the learner run on."""

        @override
        def finalize(self) -> Self:
            # The env renders the observations and names the actions, so an
            # experiment that changes it cannot forget to resize the network.
            self.model.observation_size = self.env.observation_size
            self.model.num_actions = ATN_DIM
            sampler = self.sampler
            if isinstance(sampler, EpsilonGreedy.Config):
                if sampler.steps_per_update == -1:
                    sampler.steps_per_update = self.rollout.horizon
                if sampler.total_updates == -1 and math.isfinite(
                    self.train_budget_steps,
                ):
                    sampler.total_updates = int(self.train_budget_steps)
            if self.evaluation.env is None and isinstance(self.env, CraftaxEnv.Config):
                # The evaluation plays training's rules, less its training-only
                # options: no stall cap and no practice.
                env = self.evaluation.env = self.env.copy_tree()
                env.stall_cap = None
                env.practice = None
            if self.evaluation.sampler is None:
                self.evaluation.sampler = GreedySampler.Config()
            if self.evaluation.rollout is None:
                self.evaluation.rollout = self.rollout.copy_tree()
            return super().finalize()

    class StateDict(TypedDict):
        """What a resumed run needs; see :meth:`CraftaxPQNTrainStep.state_dict`."""

        model: dict[str, Tensor]
        optimizer: OptimizerState
        timer_step: CheckpointableStepTimer.StateDict
        env: dict[str, Tensor]
        rollout: dict[str, Tensor]

    def __init__(self, config: Config) -> None:
        """Build the network, its optimizer, the environments and the actor.

        Args:
          config: The recipe.

        Raises:
          ValueError: A setting is out of range, the minibatches do not split
            the environments, the run has no update count, the rollout keeps
            more than one slot, or the evaluation's config would be refused.

        """
        _check(config)
        self.config = config
        self.device: torch.device = config.parallelism.make().device
        # Inference by default: the actor and the evaluation read the running
        # normalization; only the learner's passes switch to training.
        self.model = config.model.make().to(self.device).eval()
        self.schedule = config.schedule.make()
        self.total_steps = int(config.train_budget_steps)
        self.timer_step = CheckpointableStepTimer()
        """Updates trained: how many, and how long the learner took."""
        self.env = config.env.make()
        self.env.reset()
        self._logs_read = _log_sums(self.env)
        """The env's episode-log sums at the last report; each update reports the change."""
        self.feature = config.feature.make()
        """The previous action, which the evaluation's actor computes too."""
        self.sampler = config.sampler.make()
        """The exploring actor's action streams."""
        self.rollout = Rollout(
            config.rollout,
            policy=self.model,
            sampler=self.sampler,
            env=self.env,
            device=self.device,
            feature=self.feature,
        )
        self.optimizer: torch.optim.Optimizer = config.optimizer.make()(
            self.model.parameters(),
        )
        # The schedule's base: each group's configured rate, before any update
        # overwrites ``lr``.
        remember_initial_lrs([self.optimizer])

    @property
    def global_step(self) -> int:
        """Updates trained across the whole run, resumes included."""
        return self.timer_step.global_count

    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        """Pass the loop's tick through: the data comes from the environments."""
        return batch

    def train_step(self, **batch: object) -> TrainStepOutput:
        """Collect a rollout and regress toward its Q(lambda) targets.

        Args:
          **batch: Ignored; the loop's tick.

        Returns:
          result: The final minibatch's loss as ``loss`` and its action values
            as ``model``; the metrics add the update's rate, exploration,
            explained variance, timings and the episodes finished since the
            last update.

        Raises:
          RuntimeError: The run's updates are spent.

        """
        del batch
        if self.global_step >= self.total_steps:
            msg = f"the run's {self.total_steps} updates are spent"
            raise RuntimeError(msg)
        update = self.global_step
        started = time.perf_counter()
        rollout = self.collect()
        collected = time.perf_counter()
        learning_rate = self._set_learning_rate()
        with self.timer_step:
            loss, q_values, metrics = self._optimize(rollout)
        learned = time.perf_counter()
        chosen = rollout.q_values.gather(-1, rollout.actions[..., None])[..., 0]
        metrics["explained_variance"] = explained_variance(
            chosen.flatten(),
            rollout.targets.flatten(),
        )
        metrics["learning_rate"] = learning_rate
        if isinstance(self.sampler, EpsilonGreedy):
            metrics["epsilon"] = self.sampler.epsilon(update)
        transitions = self.env.num_envs * self.config.rollout.horizon
        metrics["agent_steps"] = float(self.global_step * transitions)
        metrics["transitions_per_second"] = transitions / (learned - started)
        metrics["rollout_seconds"] = collected - started
        metrics["learner_seconds"] = learned - collected
        metrics.update(self._episode_metrics())
        return {"loss": loss, "model": q_values, "metrics": metrics}

    @torch.no_grad()
    def collect(self) -> QRollout:
        """Collect one rollout with the current weights and build its targets.

        Returns:
          rollout: The rollout, with every action's value under the weights
            that collected it and its Q(lambda) targets.

        """
        self.env.prepare_rollout()
        if self.device.type == "cuda":
            # The learner's updates run on this thread's stream and the actor's
            # graphs on the buffers' own: they must land before a replay reads
            # the weights.
            torch.cuda.synchronize(self.device)
        return self._score(self.rollout.collect(0))

    def train_loss(self, **batch: object) -> TrainStepOutput:
        """Score the last rollout's loss under the current weights, without updating.

        Between train steps the env's buffers still follow the last rollout, so
        its targets are rebuilt as :meth:`collect` built them, from the
        current weights.

        Args:
          **batch: Ignored.

        Returns:
          result: The loss over every trajectory and their action values.

        """
        del batch
        with torch.no_grad():
            rollout = self._score(self.rollout.slots[0])
            loss, q_values = self._loss(
                rollout,
                torch.arange(self.env.num_envs, device=self.device),
            )
        return {"loss": loss, "model": q_values}

    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Score as :meth:`train_loss`, which already reads the running normalization.

        Args:
          **batch: Ignored.

        Returns:
          result: As :meth:`train_loss`.

        """
        return self.train_loss(**batch)

    def call_eval(self, *args: object, **kwargs: object) -> Tensor:
        """Value every action from a zero carry, after action 0.

        Args:
          *args: The observations, positionally.
          **kwargs: Or as ``observation``.

        Returns:
          q_values: ``[batch, num_actions]``.

        """
        observation = args[0] if args else kwargs["observation"]
        assert isinstance(observation, Tensor)
        batch = observation.shape[0]
        with torch.no_grad():
            decoded, _ = self.model.forward_fused(
                observation.to(self.device),
                self.model.initial_state(batch, device=self.device),
                None,
                features=torch.zeros(batch, 1, device=self.device),
            )
        return decoded[:, :-1]

    def make_evaluator(self) -> Evaluation:
        """Build a fresh evaluation around the network's current weights.

        Returns:
          evaluation: New environments, a greedy actor and its own carry; the
            weights and the running normalization shared.

        """
        return Evaluation(
            self.config.evaluation,
            policy=self.model,
            device=self.device,
            feature=self.feature,
        )

    def on_epoch_end(self) -> None:
        """Nothing to flush: every update completes within one step."""

    def state_dict(self) -> StateDict:
        """Return what a bit-equal resume needs.

        Returns:
          state: ``model`` (with the running normalization), ``optimizer``,
            the update count, ``env`` (worlds, streams, logs and buffers, the
            last actions among them) and from ``rollout`` the carries and the
            draw counts. The slot is left out: the next update's rollout
            rewrites it whole before anything reads it.

        """
        rollout = self.rollout.state_dict(slot=0)
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "timer_step": self.timer_step.state_dict(),
            "env": self.env.state_dict(),
            "rollout": {"carry": rollout["carry"], "draws": rollout["draws"]},
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore a :meth:`state_dict`.

        Args:
          state_dict: What :meth:`state_dict` returned.

        """
        state = cast(CraftaxPQNTrainStep.StateDict, state_dict)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.timer_step.load_state_dict(state["timer_step"])
        self.env.load_state_dict(state["env"])
        self._logs_read = _log_sums(self.env)
        # The live slot onto itself: the rollout restores a whole slot, and
        # this one is rewritten before it is read.
        self.rollout.load_state_dict(
            {**self.rollout.state_dict(slot=0), **state["rollout"]},
            slot=0,
        )

    def close(self) -> None:
        """Stop the actor's and the environments' threads."""
        self.rollout.close()
        self.env.close()

    def _score(self, storage: RolloutStorage) -> QRollout:
        """Value a stored rollout and its bootstrap, then build its targets."""
        features = storage.features
        # The step always hands the rollout its previous-action feature.
        assert isinstance(features, Tensor)
        decoded, final, _ = self.model.forward_sequence(
            storage.observations.transpose(0, 1),
            storage.initial_states,
            storage.terminals.transpose(0, 1),
            features=features.transpose(0, 1),
        )
        q_values = decoded[..., :-1].transpose(0, 1)
        env = self.env
        terminals = env.terminals.to(self.device)
        bootstrap, _ = self.model.forward_fused(
            env.observations.to(self.device),
            final,
            terminals,
            features=env.actions.to(self.device),
        )
        rewards = torch.cat(
            (storage.rewards[1:], env.rewards.to(self.device)[None]),
        ).float()
        dones = torch.cat((storage.terminals[1:], terminals[None])) != 0
        targets = q_lambda_targets(
            rewards=rewards,
            q_values=torch.cat((q_values, bootstrap[None, :, :-1])),
            dones=dones,
            discount=self.config.discount,
            trace_decay=self.config.trace_decay,
        )
        return QRollout(
            observations=storage.observations,
            previous_actions=features,
            starts=storage.terminals,
            initial_states=storage.initial_states,
            actions=storage.actions.long(),
            q_values=q_values,
            bootstrap=bootstrap[:, :-1],
            rewards=rewards,
            dones=dones,
            targets=targets,
        )

    def _optimize(
        self,
        rollout: QRollout,
    ) -> tuple[Tensor, Tensor, dict[str, float | Tensor]]:
        """Take every pass over the rollout; return the last minibatch's loss, values and metrics."""
        config = self.config
        agents = self.env.num_envs
        order = shuffles(
            seed=config.seed,
            epoch=self.global_step,
            transitions=agents,
            passes=config.num_epochs,
        ).to(self.device)
        loss = grad_norm = q_values = torch.zeros((), device=self.device)
        self.model.train()
        try:
            for trajectories in order.reshape(-1, agents // config.num_minibatches):
                loss, q_values = self._loss(rollout, trajectories)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    config.max_grad_norm,
                )
                self.optimizer.step()
        finally:
            self.model.eval()
        q_values = q_values.detach()
        return (
            loss.detach(),
            q_values,
            {
                "q_loss": loss.detach(),
                "q_mean": q_values.mean(),
                "grad_norm": grad_norm,
            },
        )

    def _loss(self, rollout: QRollout, agents: Tensor) -> tuple[Tensor, Tensor]:
        """Regress the taken actions' values toward their targets, for whole trajectories."""
        decoded, _, _ = self.model.forward_sequence(
            rollout.observations[:, agents].transpose(0, 1),
            rollout.initial_states[:, agents],
            rollout.starts[:, agents].transpose(0, 1),
            features=rollout.previous_actions[:, agents].transpose(0, 1),
        )
        q_values = decoded[..., :-1]
        chosen = q_values.gather(-1, rollout.actions[:, agents].T[..., None])[..., 0]
        # A squared error, not a clipped surrogate: there is no policy ratio to
        # trust-region. Halved, as purejaxql's ``_loss_fn`` (``pqn_rnn_craftax.py``
        # line 395): without the half every gradient doubles, which the 0.5
        # global-norm clip and RAdam's unrectified first steps both read.
        error = chosen - rollout.targets[:, agents].T
        return 0.5 * error.square().mean(), q_values

    def _set_learning_rate(self) -> float:
        """Set every group's rate for this update; return the first's."""
        rates = [
            self.schedule(
                from_plain(cast("object", group["initial_lr"]), float),
                step=self.global_step,
                total_steps=self.total_steps,
            )
            for group in self.optimizer.param_groups
        ]
        for group, rate in zip(self.optimizer.param_groups, rates, strict=True):
            group["lr"] = rate
        return rates[0]

    # Nothing clears the training env's logs, which are running fp32 sums, so each
    # report differences them, in float64, against the sums it last read; with no
    # episode finished since, it reports nothing.
    def _episode_metrics(self) -> dict[str, float]:
        """Return ``report_metrics`` as ``env/*``, over the episodes finished since the last report."""
        sums = _log_sums(self.env)
        change = (sums - self._logs_read).astype(np.float32)
        self._logs_read = sums
        mean = aggregate_logs(change.view(LOG_DTYPE).reshape(-1))
        if mean[-1] <= 0:
            return {}
        return {f"env/{name}": value for name, value in report_metrics(mean).items()}


class CraftaxPQNTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[CraftaxPQNTrainStep.Config, CraftaxRollouts.Config],
):
    """A training loop with the Q-learning step, the update cadence and checkpoints in place.

    A checkpoint holds the network, its optimizer, the environments and the
    actor's carries and streams, so a resumed run continues as if never
    stopped. One is written every 200 updates and the newest two are kept.
    """

    step: CraftaxPQNTrainStep.Config = field(
        default_factory=CraftaxPQNTrainStep.Config,
    )
    """The Q-network, the environments and the update."""

    dataset: CraftaxRollouts.Config = field(default_factory=CraftaxRollouts.Config)
    """One tick per update; the data lives in the step's environments."""

    checkpointer: Makeable[CheckpointerProtocol] | None = field(
        default_factory=lambda: Checkpointer.Config(save_every=200, keep_last_n=2),
    )
    """Every 200 updates, the newest two kept."""

    @override
    def finalize(self) -> Self:
        # Unnamed, W&B makes a name up, and a board of seeds cannot be told apart.
        dashboard = _dashboard(self.tracker)
        if dashboard is not None and not dashboard.name:
            dashboard.name = self.experiment_name
        return super().finalize()


def _check(config: CraftaxPQNTrainStep.Config) -> None:
    """Refuse a recipe the step would run wrongly, before anything is built."""
    if not math.isfinite(config.train_budget_steps) or config.train_budget_steps <= 0:
        raise ValueError(
            "train_budget_steps is the update count; it must be finite and positive",
        )
    if config.rollout.num_slots != 1:
        raise ValueError(
            "each rollout is learned from with the weights that collected it: "
            f"num_slots must be 1, not {config.rollout.num_slots}",
        )
    if config.num_epochs <= 0 or config.num_minibatches <= 0:
        raise ValueError("num_epochs and num_minibatches must be positive")
    if config.env.num_envs % config.num_minibatches:
        raise ValueError("num_minibatches must divide the environments")
    for name, value in (
        ("discount", config.discount),
        ("trace_decay", config.trace_decay),
    ):
        if math.isnan(value) or value < 0 or value > 1:
            raise ValueError(f"{name} must be in [0, 1], not {value}")
    if not math.isfinite(config.max_grad_norm) or config.max_grad_norm <= 0:
        raise ValueError(
            f"max_grad_norm must be positive and finite, not {config.max_grad_norm}",
        )
    Evaluation.check(config.evaluation)


# Numpy, not torch: the logs are the Numba step's structured records, read in place.
def _log_sums(env: TrainingEnv) -> np.ndarray:
    """Return every environment's episode-log fields as float64 ``[num_envs, LOG_FIELDS]``."""
    logs = np.ascontiguousarray(env.stats["log"])
    return logs.view(np.float32).reshape(len(logs), LOG_FIELDS).astype(np.float64)


def _dashboard(tracker: object) -> WandbTracker.Config | None:
    """Return the W&B tracker in a tree of tracker lists and async wrappers, if any."""
    if isinstance(tracker, WandbTracker.Config):
        return tracker
    if isinstance(tracker, AsyncTracker.Config):
        return _dashboard(tracker.tracker)
    if isinstance(tracker, TrackerList.Config):
        for member in tracker.trackers.values():
            found = _dashboard(member)
            if found is not None:
                return found
    return None
