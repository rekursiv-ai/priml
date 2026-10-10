"""Episode sources of a capture worker: a trained policy, or uniform random play.

A source plays a capture env (``env.py``) and hands over its episodes as they
end. :class:`PolicySource` loads a policy the port's RL trainer trained -- its
experiment's step config and a ``TrainLoop`` checkpoint -- and drives it with
the port's ``Rollout``, sampling actions under the action mask exactly as in
training, with learning, practice and auxiliary objectives off.
:class:`RandomSource` plays uniformly random legal actions instead, one
SplitMix64 stream per environment, so a worker, its archive and the replay
verifier are qualified end to end with no policy and no GPU.

Both play the game's default rules, which replay reproduces: an experiment
that ends episodes at the necromancer's fall plays on in capture, as replay
does, and only the experiment's observation layout carries over.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import hashlib

from configgle import Fig, Makeable

import numpy as np
import torch

from priml.baselines.craftax.rollout import Rollout, TorchPhiloxSampler
from priml.baselines.craftax.train_step import CraftaxTrainStep
from priml.baselines.craftax.world_model.capture.branches import (
    BranchFeeder,
)
from priml.baselines.craftax.world_model.capture.env import (
    CaptureEnv,
    Schedule,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    rollout_seed,
    splitmix64,
)
from priml.lib.codec import from_plain
from priml.runtime import best_device, get_device


if TYPE_CHECKING:
    from priml.baselines.craftax.world_model.capture.env import Captured


class CaptureError(RuntimeError):
    """Capture stopped recording: the env's failure, raised once every ended episode is taken."""


class EpisodeSource(Protocol):
    """A rollout that records episodes and hands over each one when it ends.

    Every episode's summary carries ``"episode"``, its ordinal among the
    episodes the worker started; resumption reads it back.
    """

    def provenance(self) -> dict[str, str]:
        """Return what the episodes were played by.

        Called after ``start``, so it may name the seeds ``start`` derived.

        Returns:
          provenance: Recorded in every manifest line the worker publishes.

        """
        ...

    def start(
        self,
        *,
        schedule: Schedule,
        run_dir: Path,
    ) -> None:
        """Begin playing the schedule's episodes.

        Args:
          schedule: The worker's seed range, first ordinal and remaining
            budget. Once the ended episodes hold the budget the source starts
            no episode and drains the ones in flight to their end.
          run_dir: Directory of this worker alone, for the source's own files.

        """
        ...

    def poll(self) -> list[Captured]:
        """Play on and return the episodes that ended since the last poll."""
        ...

    def finished(self) -> bool:
        """Return whether no episode will end after the ones ``poll`` still holds."""
        ...

    def close(self) -> None:
        """Stop playing and release the source."""
        ...


def make_source(config: Makeable[EpisodeSource]) -> EpisodeSource:
    """Build a source on the best accelerator unless its config names a device.

    A capture worker is a job, not a ``TrainLoop`` child: no loop has set the
    default device that a source left at ``device=None`` takes.

    Args:
      config: The source's config.

    Returns:
      source: The built source.

    """
    with torch.device(best_device()):
        return config.make()


class PolicySource:
    """A trained policy of the port's RL trainer, recorded by a capture env."""

    class Config(Fig["PolicySource"]):
        """The policy, its weights, the environments and any branches."""

        policy: CraftaxTrainStep.Config = field(default_factory=CraftaxTrainStep.Config)
        """The step config of the experiment that trained the policy: its
        model, sampler, rollout horizon and observation layout."""

        checkpoint: Path | None = None
        """A ``TrainLoop`` checkpoint of that experiment, the policy's weights
        under ``step``/``model``; required. Its SHA-256 is recorded in
        provenance."""

        env: CaptureEnv.Config = field(default_factory=CaptureEnv.Config)
        """The environments and the per-episode settings; the observation
        layout is the policy's."""

        branches: BranchFeeder.Config | None = None
        """Branch pools to record branches of, instead of fresh episodes."""

        device: torch.device | str | None = None
        """Where the policy runs; see ``get_device``."""

    def __init__(self, config: Config) -> None:
        """Hash the checkpoint and build the policy with its weights.

        Args:
          config: The policy, its weights, the environments and branches.

        Raises:
          ValueError: The checkpoint is missing.

        """
        if config.checkpoint is None:
            raise ValueError("Capture needs the arm's checkpoint.")
        with config.checkpoint.open("rb") as weights:
            self.checkpoint_sha256 = hashlib.file_digest(weights, "sha256").hexdigest()
            """SHA-256 of the checkpoint's bytes, recorded in provenance."""
        self.config = config
        self.device = get_device(config.device)
        self.policy = config.policy.model.make().to(self.device)
        # ``weights_only`` admits only tensors and plain containers, which is all
        # ``TrainLoop.state_dict`` holds; ``mmap`` leaves the rest of the
        # pipeline's state, most of the file, unread.
        state = from_plain(
            cast(
                "object",
                torch.load(
                    config.checkpoint,
                    map_location="cpu",
                    weights_only=True,
                    mmap=True,
                ),
            ),
            dict[str, object],
        )
        step = from_plain(state["step"], dict[str, object])
        self.policy.load_state_dict(
            from_plain(step.get("model"), dict[str, torch.Tensor]),
        )
        self.seed = 0
        self.env: CaptureEnv | None = None
        self.rollout: Rollout | None = None
        self.feeder: BranchFeeder | None = None

    def provenance(self) -> dict[str, str]:
        """Return the checkpoint, its hash, and the worker's sampler seed."""
        return {
            "checkpoint": str(self.config.checkpoint),
            "checkpoint_sha256": self.checkpoint_sha256,
            "rollout_seed": str(self.seed),
        }

    def start(self, *, schedule: Schedule, run_dir: Path) -> None:
        """Reset the environments and build the rollout around the policy.

        Args:
          schedule: The worker's seed range, first ordinal and budget.
          run_dir: Unused; the rollout writes no files.

        """
        del run_dir
        cfg = self.config
        step = cfg.policy
        self.feeder = _feeder(cfg.branches, schedule=schedule)
        self.env = CaptureEnv(
            _layout(cfg.env, policy=step),
            schedule=schedule,
            branches=self.feeder,
        )
        sampler = step.sampler.copy_tree()
        # A worker's streams start from the sampler's seed (``seeds.rollout_seed``).
        assert isinstance(sampler, TorchPhiloxSampler.Config)
        self.seed = sampler.seed = rollout_seed(
            arm=schedule.arm,
            worker=schedule.worker,
            base=sampler.seed,
            generation=schedule.generation,
        )
        rollout = step.rollout.copy_tree()
        rollout.num_slots = 1
        rollout.bootstrap = False
        self.rollout = Rollout(
            rollout,
            policy=self.policy,
            sampler=sampler.make(),
            env=self.env,
            device=self.device,
            feature=None if step.feature is None else step.feature.make(),
        )

    def poll(self) -> list[Captured]:
        """Play one rollout, unless capture has finished, and take the ended episodes.

        Returns:
          episodes: The episodes that ended since the last poll, oldest first.

        Raises:
          CaptureError: Capture stopped recording; raised once every ended
            episode is taken.

        """
        if self.env is None or self.rollout is None:
            raise ValueError("Start first.")
        if not self.env.finished():
            with torch.no_grad():
                self.rollout.collect(0)
        return _take(self.env)

    def finished(self) -> bool:
        """Return whether no episode will end after the ones ``poll`` still holds."""
        return self.env is not None and self.env.finished()

    def close(self) -> None:
        """Stop the rollout's threads and the branch feeder."""
        if self.rollout is not None:
            self.rollout.close()
        if self.feeder is not None:
            self.feeder.close()


class RandomSource:
    """Uniformly random legal actions, one SplitMix64 stream per environment."""

    class Config(Fig["RandomSource"]):
        """The environments, the steps per poll and the action seed."""

        env: CaptureEnv.Config = field(default_factory=CaptureEnv.Config)
        """The environments and the per-episode settings."""

        steps_per_poll: int = 64
        """Decisions each environment takes per poll."""

        action_seed: int = 0
        """Base of the action streams, offset by ``seeds.rollout_seed``."""

        branches: BranchFeeder.Config | None = None
        """Branch pools to record branches of, instead of fresh episodes."""

    def __init__(self, config: Config) -> None:
        if config.steps_per_poll <= 0:
            raise ValueError("Random play needs a positive steps_per_poll.")
        self.config = config
        self.seed = 0
        self.streams: list[int] = []
        self.env: CaptureEnv | None = None
        self.feeder: BranchFeeder | None = None

    def provenance(self) -> dict[str, str]:
        """Return the action policy and its seed."""
        return {"policy": "uniform-legal", "rollout_seed": str(self.seed)}

    def start(self, *, schedule: Schedule, run_dir: Path) -> None:
        """Reset the environments; environment ``i`` draws from ``rollout_seed + i``.

        Args:
          schedule: The worker's seed range, first ordinal and budget.
          run_dir: Unused; random play writes no files.

        """
        del run_dir
        cfg = self.config
        self.feeder = _feeder(cfg.branches, schedule=schedule)
        self.env = CaptureEnv(cfg.env, schedule=schedule, branches=self.feeder)
        self.seed = rollout_seed(
            arm=schedule.arm,
            worker=schedule.worker,
            base=cfg.action_seed,
            generation=schedule.generation,
        )
        self.streams = [self.seed + i for i in range(self.env.num_envs)]

    def poll(self) -> list[Captured]:
        """Step every environment ``steps_per_poll`` times and take the ended episodes.

        Returns:
          episodes: The episodes that ended, oldest first.

        Raises:
          CaptureError: Capture stopped recording; raised once every ended
            episode is taken.

        """
        env = self.env
        if env is None:
            raise ValueError("Start first.")
        masks, actions = env.action_mask.numpy(), env.actions.numpy()
        for _ in range(self.config.steps_per_poll):
            for row in range(env.num_envs):
                legal = np.flatnonzero(masks[row, :])
                self.streams[row], draw = splitmix64(self.streams[row])
                actions[row, 0] = legal[draw % len(legal)] if len(legal) else 0
            for buffer in range(env.num_buffers):
                env.step_buffer(buffer)
        return _take(env)

    def finished(self) -> bool:
        """Return whether no episode will end after the ones ``poll`` still holds."""
        return self.env is not None and self.env.finished()

    def close(self) -> None:
        """Stop the branch feeder."""
        if self.feeder is not None:
            self.feeder.close()


def _layout(
    env: CaptureEnv.Config,
    *,
    policy: CraftaxTrainStep.Config,
) -> CaptureEnv.Config:
    """Return ``env`` writing the observation layout ``policy`` reads."""
    env = env.copy_tree()
    env.rules.previous_action = policy.env.rules.previous_action
    env.rules.symbolic_observation = policy.env.rules.symbolic_observation
    return env


def _feeder(
    config: BranchFeeder.Config | None,
    *,
    schedule: Schedule,
) -> BranchFeeder | None:
    """Return the schedule's branch feeder, or None for fresh episodes."""
    if config is None:
        return None
    return BranchFeeder(
        config,
        arm=schedule.arm,
        worker=schedule.worker,
        first_episode=schedule.first_episode,
    )


def _take(env: CaptureEnv) -> list[Captured]:
    """Return the ended episodes, or raise the env's failure once none is left."""
    episodes = env.episodes()
    if not episodes and env.failure:
        raise CaptureError(env.failure)
    return episodes
