"""The environment as PufferLib's trainer sees it: buffers of worlds on CPU threads.

2,048 environments step in 4 buffers of 512. A rollout steps one buffer on CPU
threads while the GPU runs the policy for another, and the two sides meet only
through five host buffers: observations, an action mask, rewards and
terminals written by the env, and actions written by the policy. Those
buffers, their shapes and their dtypes are the contract between this module
and the GPU side (``rollout.py``); they are PufferLib's, and a change to one
is a change to ``rollout.py`` as well.

The world state itself is an array of ``STATE_DTYPE`` records that a Numba
kernel steps in place; nothing above ``game/`` reads it during training. It is
numpy rather than torch because Numba compiles numpy, not tensors (``game``'s
package docstring has the measured case). The five buffers are torch tensors,
pinned so the GPU copies them asynchronously, and the kernels write their
``.numpy()`` views of the same memory in place, from the helper threads,
without a copy. A buffer is stepped by ``threads_per_buffer`` threads, each
taking a contiguous share of its environments, as PufferLib's ``#pragma omp
parallel for num_threads(2)`` does: the buffer thread and its helpers split
the buffer's environments, and between steps the helpers spin rather than
sleep, as libgomp's workers do. Handing a helper its share through Python -- a
``ThreadPoolExecutor`` future and its result -- costs the GIL twice per step
and a futex wake each way (measured: 72 us per buffer-step in evaluation and
180 us in training, where the learner holds the GIL).

So a helper thread sits in :func:`serve_numba`, a ``nogil`` kernel that polls its
row of a shared ``int64`` control array for a new ticket, steps the range the
row names, and posts the ticket back as done. The buffer thread's
:func:`lead_numba` posts every helper's range and ticket, steps its own share and
spins until each helper has posted, all in one call. A helper that sees no
work for ``budget`` ticks parks: it marks its row and returns to Python to
block on an event, and :func:`lead_numba` returns ``WAKE`` so the caller wakes the
helpers whose rows are marked. The loads and stores that publish tickets are
sequentially consistent, so a helper that parks just as a ticket lands is
always seen by one side or the other. A helper that leaves :func:`serve_numba` for
good -- stopped, or raising -- marks its row ``GONE``, and the buffer thread's
wait ends there instead of spinning forever for a share nobody will step.

Two training options live here too. A stall cap (:class:`StallCap`) ends
episodes that go too long without reward. Practice
(``practice.FrontierPractice``) saves its donors' worlds and restores rows from
them between rollouts; once every share of a step has joined, the buffer
thread closes the step for it (``game.archive.after_step_numba``), so the donors
save in row order, whatever the thread count.

Control row ``r`` (``ROW`` words, its own 128-byte block): ``TICKET``,
``START``, ``STOP``, ``DONE``, ``PARKED``, ``GONE``.

References:
    https://github.com/PufferAI/PufferLib
        Suarez. PufferLib (MIT license), ``src/pufferl.cu`` and
        ``ocean/craftax/craftax.h``, pin ``6ffa5b10``.

"""

from __future__ import annotations

from dataclasses import field
from functools import cache
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import math
import platform
import threading
import time
import weakref

from llvmlite import ir
from numba.extending import intrinsic

import numba.core.types as nbtypes
import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from llvmlite.ir import IRBuilder, Value
    from numba.core.base import BaseContext
    from numba.core.typing.templates import Signature
    from numpy.typing import NDArray
    from torch import Tensor

    import torch

    from priml.baselines.craftax.game.state import (
        Archive,
        ArchiveView,
        Array1,
        EnvState,
        Records,
    )
    from priml.baselines.craftax.game.step import BatchView
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")  # ~1050 ms; only construction touches it.

from configgle import Fig, Makeable

from priml.baselines.craftax.game.archive import after_step_numba
from priml.baselines.craftax.game.jit import jit
from priml.baselines.craftax.game.state import (
    ACTION_OBS_SIZE,
    ATN_DIM,
    OBS_SIZE,
    STATE_DTYPE,
    STATS_DTYPE,
    SYMBOLIC_OBS_SIZE,
    TRAINING_STATS_DTYPE,
    new_states,
    new_stats,
)
from priml.baselines.craftax.game.step import (
    Batch,
    Rules,
    reset_range_numba,
    step_range_numba,
)
from priml.baselines.craftax.game.world_gen import (
    DUNGEON_LEVEL_CONFIGS,
    SMOOTH_LEVEL_CONFIGS,
    build_pool_numba,
)
from priml.baselines.craftax.learners.practice import FrontierPractice


ROW: Final = 16
"""Words per helper in the control array: 128 bytes, which :func:`control` aligns,
so no two helpers share a cache line -- Apple's lines are 128 bytes, and Intel's
adjacent-line prefetcher fetches 64-byte lines in pairs."""

TICKET: Final = 0
START: Final = 1
STOP: Final = 2
DONE: Final = 3
PARKED: Final = 4
GONE: Final = 5

STOPPED: Final = -1
"""The ticket that tells a helper to return for good."""

SERVE_PARKED: Final = 0
SERVE_STOPPED: Final = 1

STEPPED: Final = 0
""":func:`lead_numba`: every share of the ticket is stepped."""

WAKE: Final = 1
""":func:`lead_numba`: a helper was parked; wake the rows whose ``PARKED`` is set, then
call :func:`follow_numba`."""

ABANDONED: Final = -1
""":func:`lead_numba`: a helper is ``GONE``, so its share will never be stepped."""

_ARRAY_DATA: Final = 4
"""The data pointer's field in Numba's array struct (``ArrayModel``: meminfo, parent,
nitems, itemsize, data, shape, strides)."""


class GameRules:
    """The rules a step plays by, as a config whose ``make()`` is the kernels' ``Rules``."""

    class Config(Fig["Rules"]):
        """One field per rule of ``game.step.Rules``, under its name.

        Each default is ``Rules()``'s (PufferLib's), so the kernels' tuple
        owns them; each field has the original-Craftax value that
        ``docs/differences.md`` names. The world source is not a field here:
        ``CraftaxEnv.Config.restart`` sets ``Rules.fresh_worlds``.
        """

        original_reward: bool = Rules().original_reward
        """Reward achievements unlocked plus 0.1 per point of health gained or lost
        (D1), instead of achievements plus armour gained, or -1 on death."""

        collapse_sleep: bool = Rules().collapse_sleep
        """Play every tick of sleep or rest within one decision, as PufferLib does;
        False makes each tick a step, as original Craftax does (D4)."""

        action_mask: bool = Rules().action_mask
        """Write the legal-action mask; False writes all ones, so a sampler that
        reads it masks nothing, as original Craftax has no mask (D5)."""

        end_on_boss_defeat: bool = Rules().end_on_boss_defeat
        """End the episode when the necromancer is beaten, as original Craftax does (D2)."""

        max_timesteps: int = Rules().max_timesteps
        """Ticks after which an episode ends; PufferLib and original both use 100,000 (D3)."""

        # ``Rules`` spells off as None, which ``make`` restores.
        symbolic_observation: bool = bool(Rules().symbolic_observation)
        """Write original Craftax's 8,268-float one-hot view (D8) instead of
        PufferLib's 843 packed floats; it sets the width of ``observations``."""

        previous_action: bool = bool(Rules().previous_action)
        """Append the id of the action that led to each observation, 43 where
        it starts an episode, to the packed floats (844). A recipe's input
        rather than a difference between PufferLib and original Craftax; the
        symbolic view does not take it."""

        @override
        def make(self) -> Rules:
            """Return the rules as the step's kernels take them."""
            return Rules(
                original_reward=self.original_reward,
                collapse_sleep=self.collapse_sleep,
                action_mask=self.action_mask,
                end_on_boss_defeat=self.end_on_boss_defeat,
                max_timesteps=self.max_timesteps,
                # None, not False, when off: the step then compiles neither view.
                symbolic_observation=True if self.symbolic_observation else None,
                previous_action=True if self.previous_action else None,
            )


class Restart(Protocol):
    """Where an ended episode's next world comes from."""

    def pool(self) -> NDArray[np.void]:
        """Return the ``STATE_DTYPE`` worlds an episode end reads, built once."""
        ...

    def rules(self, rules: Rules) -> Rules:
        """Return ``rules`` with this world source's kernel option set."""
        ...


class WorldPool:
    """Restart each episode in a copy of a world generated at construction.

    World ``k`` is seed ``k``; an episode end draws one from the environment's
    stream, as PufferLib's reset pool does (D6).
    """

    class Config(Fig["WorldPool"]):
        """The pool's size."""

        num_worlds: int = 8_192
        """Worlds generated at construction."""

    def __init__(self, config: Config) -> None:
        """Keep the pool's size.

        Args:
          config: The pool's size.

        Raises:
          ValueError: ``num_worlds`` is not positive.

        """
        if config.num_worlds <= 0:
            raise ValueError(f"num_worlds must be positive, not {config.num_worlds}")
        self.num_worlds = config.num_worlds

    def pool(self) -> NDArray[np.void]:
        """Generate the pool.

        Returns:
          pool: ``STATE_DTYPE [num_worlds]``; world ``k`` from seed ``k``.

        """
        pool = new_states(self.num_worlds)
        build_pool_numba(
            pool,
            pool.view(np.uint8).reshape(self.num_worlds, STATE_DTYPE.itemsize),
            SMOOTH_LEVEL_CONFIGS,
            DUNGEON_LEVEL_CONFIGS,
            0,
        )
        return pool

    def rules(self, rules: Rules) -> Rules:
        """Return ``rules`` unchanged: its default draws from the pool."""
        return rules


class FreshWorlds:
    """Generate each episode's world anew from the environment's own stream (D6).

    As original Craftax does. The pool is one zeroed world, the template each
    new world is generated into.
    """

    class Config(Fig["FreshWorlds"]):
        """Nothing to configure: every world comes from the environment's stream."""

    def __init__(self, config: Config) -> None:
        del config

    def pool(self) -> NDArray[np.void]:
        """Return the one zeroed template world."""
        return new_states(1)

    def rules(self, rules: Rules) -> Rules:
        """Return ``rules`` with the step's world generation compiled in."""
        return rules._replace(fresh_worlds=True)


class StallCap:
    """End a training episode that goes too long without an achievement reward.

    Episodes that stall -- a policy wandering after its last reward -- fill a
    rollout with transitions that teach little. The cap ends one, as an
    ordinary terminal, ``stall_limit`` decisions after its last achievement
    reward. At each reset a draw from the environment's escape stream, its own
    so the game draws as without the cap, leaves ``uncapped_fraction`` of
    episodes uncapped, so long episodes stay in the training distribution for
    an evaluation, which plays uncapped. Training only: an evaluation refuses
    it.
    """

    class Config(Fig["StallCap"]):
        """The limit and the uncapped share."""

        stall_limit: int = 10_000
        """Decisions after an episode's last achievement reward that end it."""

        uncapped_fraction: float = 0.005
        """Share of episodes the draw at each reset leaves uncapped; a whole
        number of thousandths, as the draw is ``rand_r % 1000``."""

    def __init__(self, config: Config) -> None:
        """Keep the limit and the uncapped share in thousandths.

        Args:
          config: The limit and the share.

        Raises:
          ValueError: The limit is not positive or does not fit the step's
            int32 clock, or the share is not a whole number of thousandths in
            ``[0, 1]``.

        """
        if config.stall_limit <= 0:
            raise ValueError(f"stall_limit must be positive, not {config.stall_limit}")
        if config.stall_limit >= 2**31:
            # The step stores it in an int32 field; a wrapped limit ends episodes at once.
            raise ValueError(
                f"stall_limit must fit the step's int32 clock, not {config.stall_limit}",
            )
        fraction = config.uncapped_fraction
        if (
            not (math.isfinite(fraction) and 0 <= fraction <= 1)
            or round(fraction, 3) != fraction
        ):
            raise ValueError(
                "uncapped_fraction must be a whole number of thousandths in "
                f"[0, 1], not {fraction}",
            )
        self.stall_limit = config.stall_limit
        self.uncapped_permille = round(fraction * 1000)

    def rules(self, rules: Rules) -> Rules:
        """Return ``rules`` with the cap compiled into the step."""
        return rules._replace(
            stall_limit=self.stall_limit,
            uncapped_permille=self.uncapped_permille,
        )


class BossFightReward:
    """Pay each necromancer hit and each final-floor kill, on top of the reward.

    The game rewards the first hit and the eighth, the boss's defeat, and
    nothing between, so a policy that has learned the floors hits once and
    waits out the clock. This shaping pays every hit and every kill on the
    last floor, a step that kills the player excepted. Game-specific, and
    training only: an evaluation refuses it, so scores stay the game's.
    """

    class Config(Fig["BossFightReward"]):
        """The reward per hit and per kill."""

        hit: float = 1.0
        """Added per point of the necromancer's progress, before reward scaling."""

        kill: float = 1.5
        """Added per creature killed on the final floor, before reward scaling."""

    def __init__(self, config: Config) -> None:
        """Keep the two rewards.

        Args:
          config: The reward per hit and per kill.

        Raises:
          ValueError: A reward is not finite.

        """
        if not (math.isfinite(config.hit) and math.isfinite(config.kill)):
            raise ValueError(
                f"Boss-fight rewards must be finite, not {config.hit}, {config.kill}",
            )
        self.hit = config.hit
        self.kill = config.kill

    def rules(self, rules: Rules) -> Rules:
        """Return ``rules`` with the shaping compiled into the step."""
        return rules._replace(boss_fight_reward=(self.hit, self.kill))


class CraftaxEnv:
    """PufferLib's vectorized Craftax: ``num_envs`` worlds in ``num_buffers`` groups.

    The five buffer attributes are the contract with the GPU side. Each is a
    CPU tensor -- pinned when CUDA is available, so a device copy can be
    asynchronous -- with a row per environment, and every buffer's rows are
    a contiguous slice of it (``buffer_slice``).

    Attributes:
      observations: ``float32 [num_envs, observation_size]``. What each
        player sees after its last step: PufferLib's 843 packed floats (99
        tiles of 8 integer-valued channels as ``compute_observations_numba`` writes
        them), or with ``rules.symbolic_observation`` original Craftax's 8,268
        one-hot floats; both then 51 scalars. With ``rules.previous_action``
        the packed floats are followed by the id of the action the row's last
        step executed: 43 where that step ended the episode, after
        :meth:`reset`, and a donor's saved action after a practice restore.
        Rewritten by every step.
      action_mask: ``uint8 [num_envs, 43]``. 1 where an action is legal, as
        ``compute_action_mask_numba`` writes it; all ones before the first reset,
        as PufferLib's ``memset`` leaves it. Rewritten by every step.
      rewards: ``float32 [num_envs]``. The reward of each environment's last
        step: achievement deltas plus armour gained, or -1 on death; with
        ``rules.original_reward``, achievement deltas plus 0.1 per point of
        health gained or lost, and no death penalty.
      terminals: ``float32 [num_envs]``. 1.0 where the last step ended an
        episode (the observation is then the reset world's), else 0.0.
      actions: ``float32 [num_envs, 1]``. The action each environment takes
        on its next step, one discrete id stored as a float as PufferLib
        stores it; the env truncates it to ``int``. Written by the sampler.
      save_slots: ``int32 [num_envs]``. The carry slot each row saved into on
        its last step, or -1; rewritten by every step.
      restore_slots: ``int32 [num_envs]``. The carry slot each row starts the
        next rollout from, or -1; written by :meth:`prepare_rollout`, valid
        for that rollout's first step only.
      carry_slots: Rows of the carry archive the rollout keeps for
        ``save_slots`` and ``restore_slots``; 0 without practice.
      save_rows: The rows whose ``save_slots`` can be set, practice's donors;
        empty without practice.
      states: ``STATE_DTYPE [num_envs]``. The live worlds.
      rngs: ``uint32 [num_envs]``. Each environment's ``rand_r`` state.
      stats: ``STATS_DTYPE [num_envs]``. Each environment's accumulators and
        its fp32 episode log; ``TRAINING_STATS_DTYPE`` with a stall cap or
        practice, adding their clocks, escape stream and branch flag.
      pool: ``STATE_DTYPE`` worlds, as ``restart`` builds them.
      practice: The practice archive and its controller, or None.
      num_envs: Environments in total.
      num_buffers: Groups the environments are stepped in.
      envs_per_buffer: Environments in each group.

    """

    class Config(Fig["CraftaxEnv"]):
        """Configure the vectorized environment; defaults are exp000's."""

        num_envs: int = 2_048
        """Parallel worlds, split evenly across the buffers."""

        num_buffers: int = 4
        """Groups stepped independently; one steps while another is on the GPU."""

        threads_per_buffer: int = 2
        """Threads that step one buffer, each taking a contiguous share of it."""

        seed: int = 0
        """PufferLib's ``seed_offset``: environment ``i`` seeds ``rand_r`` with ``seed + i``."""

        rules: GameRules.Config = field(default_factory=GameRules.Config)
        """What a step plays by; PufferLib's rules unless set otherwise."""

        restart: Makeable[Restart] = field(default_factory=WorldPool.Config)
        """Where an ended episode's next world comes from."""

        spin_seconds: float = 0.005
        """How long a helper thread spins for its next share before it sleeps.

        Longer than a rollout's buffer-step (about 1-2 ms), so helpers never
        sleep mid-rollout, and short enough that an idle environment frees
        its CPUs; PufferLib's OpenMP workers spin likewise."""

        stall_cap: StallCap.Config | None = None
        """End a training episode that stalls without achievement reward;
        training only. None plays uncapped and compiles no clock."""

        practice: FrontierPractice.Config | None = None
        """Save donors' worlds as they reach new achievement returns, and restore
        the first rows from them each rollout; training only. None compiles
        none of it."""

        boss_fight_reward: BossFightReward.Config | None = None
        """Pay each necromancer hit and final-floor kill; training only. None
        pays the game's reward alone and compiles none of it."""

        @property
        def observation_size(self) -> int:
            """Floats per observation: the symbolic view's, or the packed one's."""
            if self.rules.symbolic_observation:
                return SYMBOLIC_OBS_SIZE
            return ACTION_OBS_SIZE if self.rules.previous_action else OBS_SIZE

    def __init__(self, config: Config) -> None:
        """Allocate the buffers and the states, and build the world pool.

        Args:
          config: Sizes, seed, rules and world source.

        Raises:
          ValueError: A size :meth:`check` refuses, or one the world source
            refuses.

        """
        self.check(config)
        restart = config.restart.make()
        rules = restart.rules(config.rules.make())
        if config.stall_cap is not None:
            rules = config.stall_cap.make().rules(rules)
        if config.practice is not None:
            rules = rules._replace(practice=True)
        if config.boss_fight_reward is not None:
            rules = config.boss_fight_reward.make().rules(rules)
        self.num_envs = config.num_envs
        self.num_buffers = config.num_buffers
        self.envs_per_buffer = config.num_envs // config.num_buffers
        self._seed = config.seed
        # The kernels read and write these rows through numpy views, so they stay
        # on the host even when the loop builds the step under a CUDA default.
        pinned = torch.cuda.is_available()
        num_envs = config.num_envs
        self.observations: Tensor = torch.zeros(
            (num_envs, config.observation_size),
            dtype=torch.float32,
            device="cpu",
            pin_memory=pinned,
        )
        self.action_mask: Tensor = torch.ones(
            (num_envs, ATN_DIM),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pinned,
        )
        self.rewards: Tensor = torch.zeros(
            num_envs,
            dtype=torch.float32,
            device="cpu",
            pin_memory=pinned,
        )
        self.terminals: Tensor = torch.zeros(
            num_envs,
            dtype=torch.float32,
            device="cpu",
            pin_memory=pinned,
        )
        self.actions: Tensor = torch.zeros(
            (num_envs, 1),
            dtype=torch.float32,
            device="cpu",
            pin_memory=pinned,
        )
        self.save_slots: Tensor = torch.full(
            (num_envs,),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pinned,
        )
        self.restore_slots: Tensor = torch.full(
            (num_envs,),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pinned,
        )
        self.states = new_states(num_envs)
        self.rngs = np.arange(config.seed, config.seed + num_envs, dtype=np.uint32)
        training = rules.stall_limit is not None or rules.practice is not None
        self.stats = new_stats(
            num_envs,
            TRAINING_STATS_DTYPE if training else STATS_DTYPE,
        )
        if training:
            # Seeded once, as the game's stream is seeded by ``seed + i``; a
            # reset reseeds the game's stream but continues this one.
            self.stats["escape"][:, 0] = self.rngs ^ np.uint32(0x9E37_79B9)
        self.practice = (
            None
            if config.practice is None
            else FrontierPractice(
                config.practice,
                num_envs=num_envs,
                envs_per_buffer=self.envs_per_buffer,
                save_slots=self.save_slots.numpy(),
                restore_slots=self.restore_slots.numpy(),
            )
        )
        self.carry_slots = 0 if self.practice is None else self.practice.carry_slots
        self.save_rows = (
            slice(0, 0)
            if self.practice is None
            else slice(self.practice.archive.first_donor, num_envs)
        )
        # The kernels' view of the same memory: numpy for the tensors, so a
        # step reads and writes the pinned buffers in place.
        self._batch = Batch(
            self.states,
            self.rngs,
            self.stats,
            self.actions.numpy(),
            self.observations.numpy(),
            self.action_mask.numpy(),
            self.rewards.numpy(),
            self.terminals.numpy(),
            rules,
        )
        self.pool = restart.pool()
        threads = config.threads_per_buffer
        self._bounds = [
            np.array(
                [start + (stop - start) * k // threads for k in range(threads + 1)],
                dtype=np.int64,
            )
            for start, stop, _ in (
                self.buffer_slice(buffer).indices(num_envs)
                for buffer in range(config.num_buffers)
            )
        ]
        budget = int(config.spin_seconds * ticks_per_second())
        # One team per buffer, so two buffers stepping at once never queue
        # behind each other; the calling thread takes the first share.
        self._teams: list[_Team] = []
        self._close_teams = weakref.finalize(self, _close, self._teams)
        archive = None if self.practice is None else self.practice.archive
        try:
            for bounds in self._bounds:
                self._teams.append(
                    _Team(
                        self._batch,
                        self.pool,
                        archive=archive,
                        bounds=bounds,
                        budget=budget,
                    ),
                )
        except BaseException:
            # A half-built env is never returned to be closed.
            self._close_teams()
            raise

    @classmethod
    def check(cls, config: Config) -> None:
        """Refuse the sizes the constructor would refuse, before anything is built.

        A train step checks its evaluation's env with this at construction,
        so a bad one fails before training rather than at the final eval. The
        world source checks itself when the constructor makes it.

        Args:
          config: Sizes, seed, rules and world source.

        Raises:
          ValueError: The environments do not split evenly across the buffers,
            a count is not positive, the previous action is asked of the
            symbolic view, or practice refuses the geometry.

        """
        if config.num_envs <= 0 or config.num_buffers <= 0:
            raise ValueError("num_envs and num_buffers must be positive")
        if config.num_envs % config.num_buffers:
            raise ValueError("num_envs must be a multiple of num_buffers")
        if config.threads_per_buffer <= 0 or config.rules.max_timesteps <= 0:
            raise ValueError("threads_per_buffer and max_timesteps must be positive")
        if config.rules.symbolic_observation and config.rules.previous_action:
            raise ValueError(
                "previous_action extends the packed observation; the symbolic "
                "view does not take it",
            )
        if config.practice is not None:
            FrontierPractice.check(
                config.practice,
                num_envs=config.num_envs,
                envs_per_buffer=config.num_envs // config.num_buffers,
            )

    def buffer_slice(self, buffer: int) -> slice:
        """Return the rows of every buffer attribute that belong to ``buffer``."""
        start = buffer * self.envs_per_buffer
        return slice(start, start + self.envs_per_buffer)

    def reset(self) -> None:
        """Reseed every environment and start it in a pool world, as ``puf_reset``.

        With practice, every donor then starts an episode, and no row is a branch.
        """
        reset_range_numba(self._batch, self.pool, self._seed, 0, self.num_envs)
        if self.practice is not None:
            self.practice.reset(self._batch)

    def prepare_rollout(self) -> None:
        """Ready the environments for a rollout's first step; call it between rollouts.

        Before every rollout, and never during one. With practice it sizes the
        rollout's practice rows and restores them (``FrontierPractice.prepare``),
        writing ``restore_slots``; without practice it does nothing.
        """
        if self.practice is not None:
            self.practice.prepare(self._batch)

    def practice_metrics(self) -> dict[str, float]:
        """Return practice's metrics by name; empty without practice."""
        return {} if self.practice is None else self.practice.metrics()

    def step_buffer(self, buffer: int) -> None:
        """Step every environment of one buffer by its ``actions`` row.

        Blocks until the buffer's observations, mask, rewards and terminals are
        written. The calling thread steps the first share of the buffer and
        ``threads_per_buffer - 1`` helpers step the rest, with the GIL
        released throughout.

        Args:
          buffer: Which buffer to step.

        """
        self._teams[buffer].step()

    def run_buffer(
        self,
        buffer: int,
        steps: int,
        prepare: Callable[..., int],
        args: tuple[object, ...],
    ) -> None:
        """Step one buffer ``steps`` times, calling ``prepare(*args)`` before each step.

        One ``nogil`` loop runs every step: ``prepare`` -- a ``nogil`` jitted
        function, e.g. a rollout's graph launch and stream wait, which writes
        the ``actions`` rows the step reads -- then the step as
        :meth:`step_buffer` takes it. Python runs only to wake a helper that
        went to sleep.

        Args:
          buffer: Which buffer to step.
          steps: How many times.
          prepare: Returns 0 on success; anything else stops the loop.
          args: ``prepare``'s arguments.

        Raises:
          RuntimeError: ``prepare`` returned nonzero, or a helper thread failed.

        """
        self._teams[buffer].run(steps, prepare, args)

    def close(self) -> None:
        """Stop the helper threads."""
        self._close_teams()

    def state_dict(self) -> dict[str, Tensor]:
        """Return everything a step reads and writes, as tensors sharing the env's memory.

        Call it between steps. The pool is not included: construction rebuilds
        it from the config.

        Returns:
          state: ``states``, ``rngs`` and ``stats`` as bytes, then the five
            buffers; with practice, the two slot buffers and the practice
            state under ``practice.``. The live memory, not copies, so a
            checkpointer snapshots what it is handed.

        """
        state = {
            "states": torch.from_numpy(self.states.view(np.uint8)),
            "rngs": torch.from_numpy(self.rngs.view(np.uint8)),
            "stats": torch.from_numpy(self.stats.view(np.uint8)),
            "observations": self.observations,
            "action_mask": self.action_mask,
            "rewards": self.rewards,
            "terminals": self.terminals,
            "actions": self.actions,
        }
        if self.practice is not None:
            state["save_slots"] = self.save_slots
            state["restore_slots"] = self.restore_slots
            state |= {
                f"practice.{name}": value
                for name, value in self.practice.state_dict().items()
            }
        return state

    def load_state_dict(self, state: Mapping[str, Tensor]) -> None:
        """Copy a :meth:`state_dict` into the live arrays, which the kernels address.

        Args:
          state: What :meth:`state_dict` returned, for an env of this geometry.

        """
        for name, live in self.state_dict().items():
            live.copy_(state[name])


@jit
def serve_numba(
    batch: BatchView,
    pool: Records[EnvState],
    control: Array1[int],
    row: int,
    budget: int,
) -> int:
    """Step every range posted to ``row`` until parked or stopped.

    Args:
      batch: The per-environment arrays.
      pool: The world pool.
      control: The buffer's ``int64`` control array.
      row: This helper's row.
      budget: Ticks (:func:`ticks`) to spin without work before parking.

    Returns:
      status: ``SERVE_PARKED`` after the budget ran out with no work (the
        row's ``PARKED`` is then 1), or ``SERVE_STOPPED`` (its ``GONE`` is then
        1: a ticket that ``STOPPED`` overwrote is never stepped).

    """
    base = row * ROW
    atomic_store(control, base + PARKED, 0)
    seen = atomic_load(control, base + DONE)
    while True:
        idle_since = ticks()
        ticket = atomic_load(control, base + TICKET)
        while ticket == seen:
            if ticks() - idle_since > budget:
                atomic_store(control, base + PARKED, 1)
                ticket = atomic_load(control, base + TICKET)
                if ticket == seen:
                    return SERVE_PARKED
                atomic_store(control, base + PARKED, 0)
                break
            pause()
            ticket = atomic_load(control, base + TICKET)
        if ticket == STOPPED:
            atomic_store(control, base + GONE, 1)
            return SERVE_STOPPED
        step_range_numba(batch, pool, control[base + START], control[base + STOP])
        atomic_store(control, base + DONE, ticket)
        seen = ticket


@jit
def lead_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured).
    batch: BatchView,
    pool: Records[EnvState],
    archive: ArchiveView | None,
    control: Array1[int],
    ticket: int,
    bounds: Array1[int],
) -> int:
    """Post the helpers' shares, then step the first and wait for the rest.

    Returns early, before stepping anything of its own, when a helper is
    parked: the caller wakes the parked helpers and calls :func:`follow_numba`.

    Args:
      batch: The per-environment arrays.
      pool: The world pool.
      archive: Practice's archive, which a completed step is closed for;
        None without practice.
      control: The buffer's control array, one row per helper.
      ticket: A value never posted before on this control array.
      bounds: ``int64 [helpers + 2]``: share ``k`` is ``bounds[k]..bounds[k+1]``;
        share 0 is the caller's, share ``k + 1`` helper ``k``'s.

    Returns:
      status: ``STEPPED``, ``WAKE`` or ``ABANDONED``.

    """
    helpers = bounds.shape[0] - 2
    for k in range(helpers):
        base = k * ROW
        control[base + START] = bounds[k + 1]
        control[base + STOP] = bounds[k + 2]
        atomic_store(control, base + TICKET, ticket)
    for k in range(helpers):
        if atomic_load(control, k * ROW + PARKED):
            return WAKE
    completed = follow_numba(batch, pool, archive, control, ticket, bounds)
    return STEPPED if completed else ABANDONED


@jit
def follow_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured).
    batch: BatchView,
    pool: Records[EnvState],
    archive: ArchiveView | None,
    control: Array1[int],
    ticket: int,
    bounds: Array1[int],
) -> bool:
    """Step share 0 of a posted ``ticket``, then spin until every helper is done.

    Args:
      batch: The per-environment arrays.
      pool: The world pool.
      archive: As :func:`lead_numba`'s.
      control: The buffer's control array.
      ticket: The posted ticket.
      bounds: As :func:`lead_numba`'s.

    Returns:
      completed: True once every share is stepped; False, without waiting
        further, once a helper is ``GONE``.

    """
    step_range_numba(batch, pool, bounds[0], bounds[1])
    for k in range(bounds.shape[0] - 2):
        while atomic_load(control, k * ROW + DONE) != ticket:
            if atomic_load(control, k * ROW + GONE):
                return False
            pause()
    # Once every share has joined, on this one thread: practice's donors save in
    # row order, so the archive follows the steps and not the helpers' timing.
    after_step_numba(batch, archive, bounds[0], bounds[-1])
    return True


@jit
def run_steps_numba(  # noqa: PLR0917 -- Numba nopython rejects keyword-only parameters (measured).
    batch: BatchView,
    pool: Records[EnvState],
    archive: ArchiveView | None,
    control: Array1[int],
    bounds: Array1[int],
    ticket: int,
    steps: int,
    prepare: Callable[..., int],
    args: tuple[object, ...],
) -> tuple[int, int, int, int]:
    """Before each of ``steps`` buffer-steps call ``prepare(*args)``, then :func:`lead_numba`.

    The whole loop runs without the GIL. It stops early when ``prepare``
    returns nonzero, or when a posted step finds a helper parked or gone,
    which the caller handles before resuming.

    Args:
      batch: The per-environment arrays.
      pool: The world pool.
      archive: As :func:`lead_numba`'s.
      control: The buffer's control array.
      bounds: As :func:`lead_numba`'s.
      ticket: The last ticket posted on ``control``; each step posts the next.
      steps: Buffer-steps to run.
      prepare: A ``nogil`` jitted function returning 0 on success.
      args: ``prepare``'s arguments.

    Returns:
      stepped: Steps whose every share completed.
      led: :func:`lead_numba`'s status for the step after those, ``WAKE`` or
        ``ABANDONED``; ``STEPPED`` if no step stopped the loop.
      ticket: The last ticket posted.
      status: ``prepare``'s nonzero result, if one stopped the loop, else 0.

    """
    for stepped in range(steps):
        status = prepare(*args)
        if status != 0:
            return stepped, STEPPED, ticket, status
        ticket += 1
        led = lead_numba(batch, pool, archive, control, ticket, bounds)
        if led != STEPPED:
            return stepped, led, ticket, 0
    return steps, STEPPED, ticket, 0


@jit
def stop_numba(control: Array1[int]) -> None:
    """Post ``STOPPED`` to every helper row."""
    for base in range(0, control.shape[0], ROW):
        atomic_store(control, base + TICKET, STOPPED)


@jit
def fail_numba(control: Array1[int], row: int) -> None:
    """Mark ``row``'s helper ``GONE`` after it raised, so the buffer thread stops waiting."""
    atomic_store(control, row * ROW + GONE, 1)


@jit
def clock_numba() -> int:
    """Return :func:`ticks` to a Python caller, which cannot call an intrinsic."""
    return ticks()


def control(helpers: int) -> NDArray[np.int64]:
    """Return a zeroed control array of ``helpers`` rows, each its own 128-byte block."""
    raw = np.zeros((helpers + 1) * ROW, dtype=np.int64)
    # The first word at a multiple of ROW words; words are 8 bytes, ROW a power of two.
    first = -(raw.ctypes.data >> 3) & (ROW - 1)
    return raw[first : first + helpers * ROW]


@cache
def ticks_per_second() -> float:
    """Return the rate of :func:`ticks`, measured once per process over 10 ms.

    The first :func:`clock_numba` call loads or compiles its kernel, which can take
    longer than the window; counted inside it, a 5 ms spin budget became
    0.7 ms (measured), below a buffer-step. So that call comes first.

    Returns:
      rate: Ticks per second.

    """
    clock_numba()
    started, started_ticks = time.perf_counter(), clock_numba()
    time.sleep(0.01)
    return (clock_numba() - started_ticks) / (time.perf_counter() - started)


def _atomic_load_signature(
    typingctx: object,
    array: nbtypes.Type,
    index: nbtypes.Type,
) -> tuple[Signature, Callable[..., Value]] | None:
    """Type :data:`atomic_load` as ``int64(array, intp)`` on a C-contiguous ``int64`` vector."""
    del typingctx, index
    if not _is_word_vector(array):
        return None
    return nbtypes.int64(array, nbtypes.intp), _emit_atomic_load


def _atomic_store_signature(
    typingctx: object,
    array: nbtypes.Type,
    index: nbtypes.Type,
    value: nbtypes.Type,
) -> tuple[Signature, Callable[..., Value]] | None:
    """Type :data:`atomic_store` as ``void(array, intp, int64)`` on the same vectors."""
    del typingctx, index, value
    if not _is_word_vector(array):
        return None
    return nbtypes.void(array, nbtypes.intp, nbtypes.int64), _emit_atomic_store


def _pause_signature(typingctx: object) -> tuple[Signature, Callable[..., Value]]:
    """Type :data:`pause` as ``void()``."""
    del typingctx
    return nbtypes.void(), _emit_pause


def _ticks_signature(typingctx: object) -> tuple[Signature, Callable[..., Value]]:
    """Type :data:`ticks` as ``int64()``."""
    del typingctx
    return nbtypes.int64(), _emit_ticks


atomic_load = cast(
    "Callable[[Array1[int], int], int]",
    intrinsic(_atomic_load_signature),
)
"""Load ``array[index]`` of an ``int64`` array, sequentially consistent (kernels only)."""

atomic_store = cast(
    "Callable[[Array1[int], int, int], None]",
    intrinsic(_atomic_store_signature),
)
"""Store ``value`` to ``array[index]`` of an ``int64`` array, sequentially consistent."""

pause = cast("Callable[[], None]", intrinsic(_pause_signature))
"""Hint a spin-wait to the core: x86 ``pause``, arm64 ``yield`` (kernels only)."""

ticks = cast("Callable[[], int]", intrinsic(_ticks_signature))
"""Read the cycle counter: ``rdtsc`` on x86, ``cntvct`` on arm64 (kernels only)."""


def _is_word_vector(array: nbtypes.Type) -> bool:
    """Whether ``array`` is a C-contiguous 1-D ``int64`` array, which ``gep`` can index."""
    return (
        isinstance(array, nbtypes.Array)
        and array.ndim == 1
        and array.layout == "C"
        and array.dtype == nbtypes.int64
    )


def _item_pointer(builder: IRBuilder, args: Sequence[Value]) -> Value:
    """Return the address of ``args[0][args[1]]``, a word of a contiguous vector."""
    data = builder.extract_value(args[0], _ARRAY_DATA)
    return builder.gep(data, [args[1]], inbounds=True)


def _emit_atomic_load(
    context: BaseContext,
    builder: IRBuilder,
    signature: Signature,
    args: Sequence[Value],
) -> Value:
    """Emit a ``load atomic seq_cst`` of the element."""
    del context, signature
    return builder.load_atomic(_item_pointer(builder, args), "seq_cst", 8)


def _emit_atomic_store(
    context: BaseContext,
    builder: IRBuilder,
    signature: Signature,
    args: Sequence[Value],
) -> Value:
    """Emit a ``store atomic seq_cst`` of ``args[2]`` to the element."""
    del signature
    builder.store_atomic(args[2], _item_pointer(builder, args), "seq_cst", 8)
    return context.get_dummy_value()


def _emit_pause(
    context: BaseContext,
    builder: IRBuilder,
    signature: Signature,
    args: Sequence[Value],
) -> Value:
    """Emit the platform's spin-wait hint."""
    del signature, args
    if platform.machine() in {"arm64", "aarch64"}:
        hint = builder.module.declare_intrinsic(
            "llvm.aarch64.hint",
            fnty=ir.FunctionType(ir.VoidType(), [ir.IntType(32)]),
        )
        builder.call(hint, [ir.Constant(ir.IntType(32), 1)])
    else:
        hint = builder.module.declare_intrinsic(
            "llvm.x86.sse2.pause",
            fnty=ir.FunctionType(ir.VoidType(), []),
        )
        builder.call(hint, [])
    return context.get_dummy_value()


def _emit_ticks(
    context: BaseContext,
    builder: IRBuilder,
    signature: Signature,
    args: Sequence[Value],
) -> Value:
    """Emit a read of LLVM's ``readcyclecounter``."""
    del context, signature, args
    counter = builder.module.declare_intrinsic(
        "llvm.readcyclecounter",
        fnty=ir.FunctionType(ir.IntType(64), []),
    )
    return builder.call(counter, [])


class _Team:
    """One buffer's helper threads (none at one thread), handed shares through :func:`lead_numba`."""

    def __init__(
        self,
        batch: Batch,
        pool: NDArray[np.void],
        *,
        archive: Archive | None,
        bounds: NDArray[np.int64],
        budget: int,
    ) -> None:
        helpers = len(bounds) - 2
        self._batch = batch
        self._pool = pool
        self._archive = archive
        self._bounds = bounds
        self._control = control(helpers)
        self._wakes = [threading.Event() for _ in range(helpers)]
        self._ticket = 0
        self._closed = False
        self._threads: list[threading.Thread] = []
        for row, wake in enumerate(self._wakes):
            thread = threading.Thread(
                target=_serve,
                args=(batch, pool, self._control, row, budget, wake),
                name=f"craftax-env-helper-{row}",
                daemon=True,
            )
            try:
                thread.start()
            except BaseException:
                # No team to close later: stop the helpers already serving.
                self.close()
                raise
            self._threads.append(thread)

    def step(self) -> None:
        """Step the buffer once: every share.

        Raises:
          RuntimeError: The team is closed, or a helper failed or was stopped.

        """
        self._check_open()
        self._ticket += 1
        status = lead_numba(
            self._batch,
            self._pool,
            self._archive,
            self._control,
            self._ticket,
            self._bounds,
        )
        if status != STEPPED:
            self._complete(status)

    def run(
        self,
        steps: int,
        prepare: Callable[..., int],
        args: tuple[object, ...],
    ) -> None:
        """Run ``steps`` of ``prepare`` then a step, resuming after each wake.

        Args:
          steps: Buffer-steps to run.
          prepare: A ``nogil`` jitted function returning 0 on success.
          args: ``prepare``'s arguments.

        Raises:
          RuntimeError: ``prepare`` returned nonzero, the team is closed, or a
            helper failed or was stopped.

        """
        self._check_open()
        while steps:
            stepped, led, self._ticket, status = run_steps_numba(
                self._batch,
                self._pool,
                self._archive,
                self._control,
                self._bounds,
                self._ticket,
                steps,
                prepare,
                args,
            )
            steps -= stepped
            if status:
                raise RuntimeError(f"The buffer's prepare step returned {status}.")
            if led != STEPPED:
                self._complete(led)
                steps -= 1

    def close(self) -> None:
        """Post the stop to every helper, wake the sleeping ones, and join them."""
        self._closed = True
        stop_numba(self._control)
        for wake in self._wakes:
            wake.set()
        for thread in self._threads:
            thread.join()

    def _check_open(self) -> None:
        """Raise if closed: the helpers are gone, and a posted share would wait on none."""
        if self._closed:
            raise RuntimeError("The environment is closed.")

    def _complete(self, status: int) -> None:
        """Finish the posted step that :func:`lead_numba` left, waking parked helpers."""
        if status == WAKE:
            for row, wake in enumerate(self._wakes):
                if self._control[row * ROW + PARKED]:
                    wake.set()
            if follow_numba(
                self._batch,
                self._pool,
                self._archive,
                self._control,
                self._ticket,
                self._bounds,
            ):
                return
        raise RuntimeError(
            "An environment helper thread failed or was stopped by close(); "
            "see its traceback.",
        )


def _serve(  # noqa: PLR0917 -- A thread target's positional arguments.
    batch: Batch,
    pool: NDArray[np.void],
    control: NDArray[np.int64],
    row: int,
    budget: int,
    wake: threading.Event,
) -> None:
    """Run one helper: serve shares, and sleep on ``wake`` whenever it parks."""
    try:
        while serve_numba(batch, pool, control, row, budget) == SERVE_PARKED:
            wake.wait()
            wake.clear()
    except BaseException:
        fail_numba(control, row)
        raise


def _close(teams: list[_Team]) -> None:
    """Stop every team; the finalizer of a closed or collected environment."""
    for team in teams:
        team.close()
