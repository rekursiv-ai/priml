"""The environment as a learner sees it: reset, step, and auto-restart.

Episodes end at different times across the batch, so a worker whose episode
just ended is returned to a fresh world on the same step that reports it. The
observation handed back alongside a set ``done`` flag is therefore the RESET
observation, which is what lets a rollout keep a rectangular shape without the
learner tracking per-worker episode boundaries.

Achievement unlocks are reported through the step's ``info`` rather than folded
into the reward, because the score is computed from them at evaluation and a
policy must never read them.

On a GPU a step is replayed as CUDA graphs (``cuda_graph``): a step is thousands
of tiny kernels, and launching them one at a time costs several times what
running them does. The graphs run the same kernels in the same order and draw
the same random numbers, so a graphed run is bit-identical to an eager one.
Capture needs every tensor a step touches to keep its memory, which is why the
world and each step's outputs live in buffers allocated at ``reset`` and
overwritten in place, eagerly or not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import functools

from configgle import Fig, Makeable
from torch import Tensor

import torch

from priml.baselines.craftax.cuda_graph import CudaGraphed
from priml.baselines.craftax.game import constants, observation, step, world_gen
from priml.baselines.craftax.game.state import EnvState, empty_state
from priml.baselines.craftax.restart import (
    RestartFromReserve,
    RestartPolicy,
    RestartStateDict,
)
from priml.runtime import get_device


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence


@dataclass(frozen=True, slots=True, kw_only=True)
class CraftaxStep:
    """One transition across every parallel worker.

    Attributes:
      observation: Next observations, post-reset where ``done``.
      reward: Reward earned by the transition.
      done: Whether the transition ended an episode.
      info: Per-achievement unlock indicators, valued 100 at the final step
        of an episode that unlocked one and 0 otherwise. Diagnostics only.
      terminal_state: The state this transition REACHED, before any finished
        worker was restarted into a fresh world. ``env.state`` already holds
        the replacement by the time a caller sees this, so a renderer wanting
        to draw the frame that ended the episode -- the drowning, the death --
        has no other way to reach it.

    """

    observation: Tensor

    reward: Tensor

    done: Tensor

    info: dict[str, Tensor]

    terminal_state: EnvState


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

        view: tuple[int, int] = (9, 11)
        """Tiles the player can see, ``(rows, columns)``.

        The benchmark's own 9x11, and changing it changes the game: a policy
        trained on a smaller window sees less and its score is not comparable
        to a published one. It is a field rather than a constant because a
        test that only needs the encoding to RUN should not have to pay for
        8,268 floats per observation."""

        optimistic_reset_ratio: int = 16
        """Workers served by each freshly generated world.

        Generating a world is expensive and most steps end no episode, so
        generating one per worker means throwing nearly all of them away. This
        caps generation at ``num_envs / ratio`` worlds at a time, a pool that
        ``restart`` deals to whichever workers finished.

        The cost is a correlation: with more terminal workers in one step than
        the pool holds, two of them restart in the SAME world. At the
        published ratio of 16 that is rare -- episodes run thousands of steps
        and end at scattered times -- and the reference baseline accepts it in
        exchange for the throughput. Set 1 to generate one world per worker."""

        restart: Makeable[RestartPolicy] = field(
            default_factory=RestartFromReserve.Config,
        )
        """When the pool's worlds are generated; see ``restart``.

        A reserve, because generating one world costs most of what generating
        a pool does: a thousand workers then pay for generation every twenty or
        so steps instead of every step. The worlds dealt have the same
        distribution (checked per floor over a thousand of them) and exp011
        trained to scores within seed noise of each other under both policies.
        ``RestartOnDemand`` reproduces the draws every golden was minted with."""

        cuda_graphs: bool = True
        """Replay each step as captured CUDA graphs when the world is on a GPU.

        Bit-identical to stepping eagerly and several times faster, because a
        step's cost is launching its thousands of kernels, not running them.
        False steps eagerly, for profiling one kernel at a time or debugging a
        capture; on a CPU there is nothing to capture."""

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
        if min(config.view) <= 0:
            raise ValueError("view must be positive in both dimensions")
        self.num_actions = len(constants.Action)
        self._view = config.view
        self.observation_size = observation.observation_size(config.view)
        self.reward_ceiling = constants.REWARD_CEILING
        self._num_envs = config.num_envs
        self._reset_ratio = config.optimistic_reset_ratio
        self._restart = config.restart
        self._device = get_device(config.device)
        self._generator = torch.Generator(device=self._device)
        self._generator.manual_seed(config.seed)
        self._cuda_graphs = config.cuda_graphs and self._device.type == "cuda"
        self._stepper: _Stepper | None = None

    @property
    def state(self) -> EnvState:
        """The live world.

        Raises:
          RuntimeError: The environment has not been reset.

        """
        return self._live_stepper().state

    def reset(self, num_envs: int = 0) -> Tensor:
        """Start fresh episodes in every worker.

        Args:
          num_envs: Workers to run; zero keeps the configured batch size.

        Returns:
          observation: Initial observations, ``[envs, observation_size]``.

        """
        if num_envs:
            self._num_envs = num_envs
        world = world_gen.generate_world(
            num_envs=self._num_envs,
            generator=self._generator,
            device=self._device,
        )
        self._stepper = _Stepper(
            world,
            generator=self._generator,
            pool_size=max(1, self._num_envs // self._reset_ratio),
            restart=self._restart.make(),
            view=self._view,
            cuda_graphs=self._cuda_graphs,
        )
        return self._stepper.observation.clone()

    def step(self, actions: Tensor) -> CraftaxStep:
        """Advance every worker one action, restarting those that finished.

        Args:
          actions: Integer actions, ``[envs]``.

        Returns:
          step: The resulting transition across every worker. Its tensors are
            the caller's own; ``terminal_state`` is valid until the next step.

        """
        return self._live_stepper().step(actions)

    class StateDict(TypedDict):
        """Checkpointed environment: generator, batch size, world, and pool.

        ``state`` is empty before the first :meth:`reset`. ``pool`` and
        ``restart`` are absent from a checkpoint written before they existed,
        which resumes with the pool spent.
        """

        generator: Tensor
        num_envs: int
        state: dict[str, Tensor]
        pool: NotRequired[dict[str, Tensor]]
        restart: NotRequired[RestartStateDict]

    def state_dict(self) -> StateDict:
        """Return the world and its generator, for checkpointing.

        Returns:
          result: The generator state, batch size, and flattened world.

        """
        saved: CraftaxEnv.StateDict = {
            "generator": self._generator.get_state(),
            "num_envs": self._num_envs,
            "state": {},
        }
        if self._stepper is not None:
            saved["state"] = self._stepper.state.state_dict()
            saved["pool"] = self._stepper.pool.state_dict()
            saved["restart"] = self._stepper.restart.state_dict()
        return saved

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore a world saved by :meth:`state_dict`.

        Args:
          state_dict: State from :meth:`state_dict` with generator, env state.

        """
        state = cast(CraftaxEnv.StateDict, state_dict)
        generator_state = state["generator"]
        self._generator.set_state(generator_state)
        self._num_envs = state["num_envs"]
        saved = state["state"]
        if not saved:
            self._stepper = None
            return
        # The world is loaded INTO the live buffers, so they must have its size.
        if self._stepper is None or self._stepper.state.num_envs != self._num_envs:
            self.reset()
        stepper = self._live_stepper()
        stepper.state.load_state_dict(saved)
        if "pool" in state and "restart" in state:
            stepper.pool.load_state_dict(state["pool"])
            stepper.restart.load_state_dict(state["restart"])
        else:
            stepper.restart.load_state_dict({"cursor": None})
        self._generator.set_state(generator_state)

    def _live_stepper(self) -> _Stepper:
        """Return the step machinery ``reset`` built."""
        if self._stepper is None:
            raise RuntimeError("CraftaxEnv must reset before it can be read")
        return self._stepper


class _Stepper:
    """One batch's step, run from buffers that keep their memory across steps.

    A step is three procedures around one host read. ``advance`` plays the step
    from the live world into ``reached``; the host then reads how many workers
    finished, and the restart policy decides how many worlds ``generate`` must
    build into the pool; ``commit`` deals pool worlds to the finished workers
    and renders every view. On a GPU each procedure is a replayed CUDA graph,
    with one graph per generated count.
    """

    def __init__(
        self,
        world: EnvState,
        *,
        generator: torch.Generator,
        pool_size: int,
        restart: RestartPolicy,
        view: tuple[int, int],
        cuda_graphs: bool,
    ) -> None:
        num_envs, device = world.num_envs, world.device
        self.state = world
        """The live world, overwritten in place by every step."""

        self.reached = empty_state(num_envs=num_envs, device=device)
        """The world each step reached, before any finished worker restarted."""

        self.pool = empty_state(num_envs=pool_size, device=device)
        """Fresh worlds awaiting the workers that finish."""

        self.restart = restart
        """Decides which pool worlds each step's finished workers take."""

        self._generator = generator
        self._view = view
        self._cuda_graphs = cuda_graphs
        self._actions = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self._reward = torch.zeros(num_envs, device=device)
        self._done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._info = torch.zeros((num_envs, len(constants.Achievement)), device=device)
        self._finished = torch.zeros((), dtype=torch.int64, device=device)
        self._deal_offset = torch.zeros((), dtype=torch.int64, device=device)
        self._deal_modulus = torch.ones((), dtype=torch.int64, device=device)
        self.observation = observation.render(world, view=view)
        """What every worker sees of the live world."""

        self._advance = self._procedure(self._advance_eager, (generator,))
        self._commit = self._procedure(self._commit_eager, ())
        self._generate_by_count: dict[int, Callable[[], None]] = {}

    def step(self, actions: Tensor) -> CraftaxStep:
        """Advance every worker one action, restarting those that finished.

        Args:
          actions: Integer actions, ``[envs]``.

        Returns:
          step: Copies of this step's outputs, and ``reached`` as it stands.

        """
        self._actions.copy_(actions)
        self._advance()
        # The step's one host read: a graph has one static shape, so the host
        # must know how many worlds to generate to replay the graph for them.
        plan = self.restart.plan(
            finished=int(self._finished),
            pool_size=self.pool.num_envs,
        )
        if plan.generate:
            self._generate(plan.generate)
        self._deal_offset.fill_(plan.offset)
        self._deal_modulus.fill_(plan.modulus)
        self._commit()
        return CraftaxStep(
            observation=self.observation.clone(),
            reward=self._reward.clone(),
            done=self._done.clone(),
            info=_achievement_info(self._info.clone()),
            terminal_state=self.reached,
        )

    def _procedure(
        self,
        procedure: Callable[[], None],
        generators: Sequence[torch.Generator],
    ) -> Callable[[], None]:
        """Return ``procedure``, replayed as a CUDA graph when graphs are on."""
        if self._cuda_graphs:
            return CudaGraphed(procedure, generators=generators)
        return procedure

    def _advance_eager(self) -> None:
        """Play one step from the live world into ``reached`` and score it."""
        reached, reward = step.step(
            self.state.shallow_copy(),
            self._actions,
            generator=self._generator,
        )
        done = step.is_done(reached)
        self.reached.copy_(reached)
        self._reward.copy_(reward)
        self._done.copy_(done)
        # 100 where an episode ended having unlocked the achievement and 0
        # otherwise, so averaging over completed episodes gives the success rate
        # directly.
        self._info.copy_((reached.achievements & done[:, None]).float() * 100.0)
        self._finished.copy_(done.sum())

    # JAX must pick one static shape, so the reference approximates an exact count
    # with a fixed pool and a two-branch `lax.cond`; here each count selects the graph
    # captured for it.
    def _generate(self, count: int) -> None:
        """Fill the front of the pool with ``count`` fresh worlds."""
        procedure = self._generate_by_count.get(count)
        if procedure is None:
            procedure = self._procedure(
                functools.partial(self._generate_eager, count),
                (self._generator,),
            )
            self._generate_by_count[count] = procedure
        procedure()

    def _generate_eager(self, count: int) -> None:
        """Generate ``count`` worlds into the front of the pool."""
        fresh = world_gen.generate_world(
            num_envs=count,
            generator=self._generator,
            device=self.state.device,
        )
        self.pool.head(count).copy_(fresh)

    def _commit_eager(self) -> None:
        """Restart the finished workers, then render what every worker sees."""
        # Deal the pool across the batch: the nth finished worker takes world
        # offset + n, wrapping at the modulus, which is where two workers
        # finishing together can share one. Unfinished rows index harmlessly,
        # since ``select`` discards them.
        rank = self._done.to(torch.int64).cumsum(0) - 1
        fresh = self.pool.take(
            (self._deal_offset + rank.clamp_min(0)) % self._deal_modulus,
        )
        self.state.copy_(self.reached.select(self._done, fresh))
        self.observation.copy_(observation.render(self.state, view=self._view))


def _achievement_info(info: Tensor) -> dict[str, Tensor]:
    """Name each achievement's column of the end-of-episode unlock table."""
    return {
        f"Achievements/{achievement.name.lower()}": info[:, index]
        for index, achievement in enumerate(constants.Achievement)
    }
