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

from priml.baselines.craftax.game import constants, observation, step, world_gen
from priml.baselines.craftax.game.state import EnvState
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

        optimistic_reset_ratio: int = 16
        """Workers served by each freshly generated world.

        Generating a world is expensive and most steps end no episode, so
        generating one per worker means throwing nearly all of them away. This
        generates ``num_envs / ratio`` worlds instead and deals them to
        whichever workers finished.

        The cost is a correlation: with more terminal workers in one step than
        worlds generated, two of them restart in the SAME world. At the
        published ratio of 16 that is rare -- episodes run thousands of steps
        and end at scattered times -- and the reference baseline accepts it in
        exchange for the throughput. Set 1 to generate one world per worker."""

    def __init__(self, config: Config) -> None:
        """Prepare an unpopulated environment.

        Args:
          config: Batch size, device, and seed.

        Raises:
          ValueError: The batch is empty, or the reset ratio is invalid.

        """
        if config.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if config.optimistic_reset_ratio <= 0:
            raise ValueError("optimistic_reset_ratio must be positive")
        self.num_actions = len(constants.Action)
        self.observation_size = observation.OBSERVATION_SIZE
        self.reward_ceiling = constants.REWARD_CEILING
        self._num_envs = config.num_envs
        self._reset_ratio = config.optimistic_reset_ratio
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
        return observation.render(self._state)

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
            state = self._restart(state, done)

        self._state = state
        return CraftaxStep(
            observation=observation.render(state),
            reward=reward,
            done=done,
            info=info,
        )

    def _restart(self, state: EnvState, done: Tensor) -> EnvState:
        """Put every finished worker into a fresh world.

        Only ``num_envs / ratio`` worlds are generated, because generating one
        is the single most expensive thing this environment does and a step
        that ends no episode would throw all of them away. The generated
        worlds are dealt to the terminal workers in order and wrap around,
        which is what the ratio buys and costs: fewer worlds generated, and a
        chance that two workers finishing together share one.

        Args:
          state: The stepped world.
          done: Which workers finished, ``[envs]``.

        Returns:
          state: The world with finished workers restarted.

        """
        # Generate exactly as many worlds as there are finished workers, up to
        # the pool the ratio allows. Generation DOES scale with batch size --
        # 25 ms for one world, 182 ms for sixty-four -- so a step that ended
        # three episodes should not pay for sixteen.
        #
        # This is where an eager port beats the reference. JAX must pick one
        # static shape and compile it, so the baseline approximates this with
        # a fixed pool and a two-branch `lax.cond`; here the count is just an
        # integer, and the exact-fit case is also the fast one.
        pool = max(1, state.num_envs // self._reset_ratio)
        wanted = min(int(done.sum()), pool)
        fresh = world_gen.generate_world(
            num_envs=wanted,
            generator=self._generator,
            device=self._device,
        )
        if wanted < state.num_envs:
            # Deal the pool across the batch: the nth finished worker takes
            # world n mod pool. Non-terminal rows index harmlessly, since
            # ``select`` discards them.
            rank = done.to(torch.int64).cumsum(0) - 1
            fresh = fresh.take(rank.clamp_min(0) % wanted)
        return state.select(done, fresh)

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
