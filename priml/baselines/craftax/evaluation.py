"""PufferLib's final evaluation: a fresh actor around the trained policy.

PufferLib evaluates in a trainer built for the purpose (``eval_make``,
``pufferl.cu:2905-2960``): new environments reset from their seeds, a sampler
whose Philox streams start at draw zero, a zero carry, and no
``reset_every_horizon``. It wipes the episode logs, then plays whole rollouts
until enough episodes have finished (``eval_loop``, ``pufferl.cu:2846-2903``).
One :class:`Evaluation` is that trainer and that loop. It shares only the
policy's weights with training -- the training environments, streams, carry
and rollout slots are never touched -- so an evaluation cannot move a training
bit. ``metric.CraftaxScore`` reduces what it played to the score. The logs
are numpy records because the Numba step writes them (``game``'s package
docstring says why the game is numpy).

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``src/pufferl.cu``, pin ``6ffa5b10``.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import time

from configgle import Fig, Makeable

import numpy as np

from priml.baselines.craftax.env import CraftaxEnv
from priml.baselines.craftax.game.state import LOG_DTYPE
from priml.baselines.craftax.rollout import (
    FeatureSource,
    Rollout,
    Sampler,
)


if TYPE_CHECKING:
    import torch

    from priml.baselines.craftax.model import Policy


@dataclass(frozen=True, slots=True, kw_only=True)
class Played:
    """What one evaluation played.

    Attributes:
      logs: Every environment's episode log, ``LOG_DTYPE [num_envs]``.
      rollouts: Whole rollouts played.
      gameplay_seconds: Wall time spent playing them.

    """

    logs: np.ndarray
    rollouts: int
    gameplay_seconds: float


@runtime_checkable
class Evaluator(Protocol):
    """Plays an evaluation of the policy under training."""

    def play(self) -> Played:
        """Play from a fresh start until the evaluation's episodes have finished."""
        ...

    def close(self) -> None:
        """Release the environments and threads; the evaluator is not reused."""
        ...


@runtime_checkable
class MakesEvaluator(Protocol):
    """Builds a fresh :class:`Evaluator` around the policy being trained."""

    def make_evaluator(self) -> Evaluator:
        """Return a new evaluator; its caller owns it and closes it."""
        ...


class Evaluation:
    """Plays evaluation rollouts in environments of its own."""

    class Config(Fig["Evaluation"]):
        """The evaluation trainer's parts, and how many episodes make a score.

        A part left ``None`` is the training run's, filled by the train step's
        ``finalize``: PufferLib's ``eval_make`` builds from the run's own
        config, so by default a fresh evaluation replays the training geometry,
        seeds and rules.
        """

        env: CraftaxEnv.Config | None = None
        """The environments; ``None`` plays the training environments' config.
        With a feature, each environment's history is held beside training's
        (a 20-layer world model's KV cache is 31.5 MB a row), so a recipe whose
        training rows fill most of the device evaluates on fewer."""

        sampler: Makeable[Sampler] | None = None
        """The action streams, restarted from draw zero; ``None`` is training's."""

        rollout: Rollout.Config | None = None
        """The slot and the horizon; ``None`` plays the training horizon in one
        slot, since nothing learns while an evaluation plays."""

        num_episodes: int = 10_000
        """Finished episodes required; PufferLib's ``base.eval_episodes``.

        Play stops at the FIRST rollout that brings the count to at least this
        many, so the logs usually cover a few more."""

    def __init__(
        self,
        config: Config,
        *,
        policy: Policy,
        device: torch.device,
        feature: FeatureSource | None = None,
    ) -> None:
        """Build the environments and the rollout around ``policy``.

        Args:
          config: The evaluation trainer's parts, every one set.
          policy: The policy under evaluation; only its weights are read.
          device: Where the policy runs.
          feature: Training's per-step feature source, which the policy reads;
            the evaluation's own engines step it, one row per environment of
            its own, and free their memory on :meth:`close`. None without one.

        Raises:
          ValueError: A part is unset, or :meth:`check` refuses the config.

        """
        if config.env is None or config.sampler is None or config.rollout is None:
            raise ValueError(
                "an evaluation needs its env, sampler and rollout; a train "
                "step's finalize fills the unset ones from training's",
            )
        self.check(config)
        self.num_episodes = config.num_episodes
        self.env = config.env.make()
        self.env.reset()
        self._rollout = Rollout(
            config.rollout,
            policy=policy,
            sampler=config.sampler.make(),
            env=self.env,
            device=device,
            feature=feature,
        )
        self._seconds = 0.0

    @classmethod
    def check(cls, config: Config) -> None:
        """Refuse the values the constructor would refuse, before anything is built.

        A train step calls this at construction, so a bad evaluation fails
        before training rather than when the final eval starts. It checks what
        an evaluation sets for itself -- the episode count and the env's
        sizes. An unset part is training's, which the step checks as it builds
        training's own; a sampler or rollout set for the evaluation alone
        checks itself when the evaluation builds it.

        Args:
          config: The evaluation trainer's parts.

        Raises:
          ValueError: ``num_episodes`` is not positive, ``CraftaxEnv.check``
            refuses the env's sizes, or the env sets a training-only option:
            a stall cap, practice or the boss-fight reward. A train step's
            ``finalize`` copies training's env into an unset one, dropping
            them, so a recipe that sets its evaluation's env drops them itself.

        """
        if config.num_episodes <= 0:
            raise ValueError(
                f"num_episodes must be positive, not {config.num_episodes}",
            )
        if config.env is not None:
            CraftaxEnv.check(config.env)
            env = config.env
            if (
                env.stall_cap is not None
                or env.practice is not None
                or env.boss_fight_reward is not None
            ):
                raise ValueError(
                    "an evaluation plays natural, uncapped episodes for the game's "
                    "reward: its env sets none of stall_cap, practice and "
                    "boss_fight_reward, which are training-only",
                )

    def reset(self) -> None:
        """Start afresh, as a new evaluation would.

        Reseeded environments, streams at draw zero, a zero carry, empty
        episode logs and a stopped clock.
        """
        self.env.reset()
        self.env.stats["log"] = np.zeros((), dtype=LOG_DTYPE)
        self._rollout.reset()
        self._seconds = 0.0

    def collect(self) -> None:
        """Play one rollout: ``horizon`` steps on every buffer."""
        start = time.perf_counter()
        self._rollout.collect(0)
        self._seconds += time.perf_counter() - start

    def play(self) -> Played:
        """Play whole rollouts from a fresh start until enough episodes finish.

        Returns:
          played: The logs after the first rollout that brings the finished
            episodes to ``num_episodes`` or more, the rollouts played, and
            their wall time.

        """
        self.reset()
        rollouts = 0
        # Each count is a small whole number, so the float64 sum is exact and
        # stops on the rollout PufferLib's fp32 environment-order sum stops on.
        while float(self.logs["n"].astype(np.float64).sum()) < self.num_episodes:
            self.collect()
            rollouts += 1
        return Played(logs=self.logs, rollouts=rollouts, gameplay_seconds=self._seconds)

    @property
    def logs(self) -> np.ndarray:
        """Every environment's episode log, ``LOG_DTYPE [num_envs]``."""
        return np.ascontiguousarray(self.env.stats["log"])

    @property
    def gameplay_seconds(self) -> float:
        """Wall time spent in :meth:`collect` since :meth:`reset`."""
        return self._seconds

    def close(self) -> None:
        """Stop the rollout's and the environments' threads."""
        self._rollout.close()
        self.env.close()
