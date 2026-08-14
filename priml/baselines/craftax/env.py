"""The environment as a learner sees it: reset, step, and auto-restart.

Episodes end at different times across the batch, so a worker whose episode
just ended is returned to a fresh world on the same step that reports it. The
observation handed back alongside a set ``done`` flag is therefore the RESET
observation, which is what lets a rollout keep a rectangular shape without the
learner tracking per-worker episode boundaries.

Achievement unlocks are reported through the step's ``info`` rather than folded
into the reward, because the score is computed from them at evaluation and a
policy must never read them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from configgle import Fig
from torch import Tensor

import torch

from priml.baselines.craftax import constants, renderer, step, world_gen
from priml.baselines.craftax.state import EnvState
from priml.runtime import get_device


@dataclass(frozen=True, slots=True, kw_only=True)
class CraftaxStep:
    """One transition across every parallel worker.

    Attributes:
      observation: Next observations, post-reset where ``done``.
      reward: Reward earned by the transition.
      done: Whether the transition ended an episode.
      info: Per-achievement unlock indicators, valued 100 at the final step
        of an episode that unlocked one and 0 otherwise. Diagnostics only.

    """

    observation: Tensor
    reward: Tensor
    done: Tensor
    info: dict[str, Tensor]


class CraftaxEnv:
    """Full symbolic Craftax, batched and auto-resetting.

    Attributes:
      num_actions: Size of the discrete action space.
      observation_size: Width of one flattened observation.
      reward_ceiling: Total achievement reward available, which normalizes a
        score into a comparable percentage.

    """

    class Config(Fig["CraftaxEnv"]):
        """Configure the batched environment."""

        num_envs: int = 256
        """Parallel worlds stepped together."""

        device: str = "auto"
        """Device the world lives on; ``"auto"`` picks the best available."""

        seed: int = 0
        """Seed for world generation and every in-game draw.

        The environment owns a generator rather than drawing from the global
        stream: a rollout interleaves environment draws with the policy's
        action sampling, and sharing one stream would make the world depend
        on how many actions had been sampled."""

    def __init__(self, config: Config) -> None:
        """Prepare an unpopulated environment.

        Args:
          config: Batch size, device, and seed.

        Raises:
          ValueError: The batch is empty.

        """
        if config.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_actions = len(constants.Action)
        self.observation_size = renderer.OBSERVATION_SIZE
        self.reward_ceiling = constants.REWARD_CEILING
        self._num_envs = config.num_envs
        self._device = get_device(config.device)
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(config.seed)
        self._state: EnvState | None = None

    @property
    def state(self) -> EnvState:
        """The live world.

        Raises:
          RuntimeError: The environment has not been reset.

        """
        if self._state is None:
            raise RuntimeError("CraftaxEnv must reset before it can be read")
        return self._state

    def reset(self, num_envs: int = 0) -> Tensor:
        """Start fresh episodes in every worker.

        Args:
          num_envs: Workers to run; zero keeps the configured batch size.

        Returns:
          observation: Initial observations, ``[envs, observation_size]``.

        """
        if num_envs:
            self._num_envs = num_envs
        self._state = world_gen.generate_world(
            num_envs=self._num_envs,
            generator=self._generator,
            device=self._device,
        )
        return renderer.render(self._state)

    def step(self, actions: Tensor) -> CraftaxStep:
        """Advance every worker one action, restarting those that finished.

        Args:
          actions: Integer actions, ``[envs]``.

        Returns:
          step: The resulting transition across every worker.

        """
        state, reward = step.step(
            self.state,
            actions,
            generator=self._generator,
        )
        done = step.is_done(state)
        info = _achievement_info(state, done)

        if bool(done.any()):
            # Only the finished workers restart. Generating a whole fresh
            # batch and selecting rows keeps the operation shaped the same
            # whether one worker ended or all of them did.
            fresh = world_gen.generate_world(
                num_envs=state.num_envs,
                generator=self._generator,
                device=self._device,
            )
            state = state.select(done, fresh)

        self._state = state
        return CraftaxStep(
            observation=renderer.render(state),
            reward=reward,
            done=done,
            info=info,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the world and its generator, for checkpointing."""
        return {
            "generator": self._generator.get_state(),
            "num_envs": self._num_envs,
            "state": {} if self._state is None else self._state.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a world saved by :meth:`state_dict`."""
        self._generator.set_state(state_dict["generator"])
        self._num_envs = state_dict["num_envs"]
        saved = state_dict["state"]
        if not saved:
            self._state = None
            return
        if self._state is None:
            self.reset()
        self.state.load_state_dict(saved)


def _achievement_info(state: EnvState, done: Tensor) -> dict[str, Tensor]:
    """Report each achievement's unlock as an end-of-episode percentage.

    The value is 100 where an episode ended having unlocked the achievement
    and 0 otherwise, so averaging the entries over completed episodes gives
    the success rate directly.
    """
    unlocked = state.achievements & done[:, None]
    return {
        f"Achievements/{achievement.name.lower()}": unlocked[:, index].float() * 100.0
        for index, achievement in enumerate(constants.Achievement)
    }
