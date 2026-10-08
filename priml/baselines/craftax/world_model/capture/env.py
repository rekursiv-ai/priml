"""The environment capture records: the port's game, recording every episode it plays.

:class:`CaptureEnv` has ``CraftaxEnv``'s five buffers (``rollout.EnvBuffers``),
so the port's ``Rollout`` drives it with any policy, and it plays the game's
default rules (``Rules()``), which replay reproduces, while it records. A
buffer steps in one ``nogil`` kernel (``step.record_rows_numba``), so buffers step
in parallel on the rollout's threads; Python handles only the rows whose
episode ended or whose chunk filled.

Per recorded decision it takes the epsilon override, the action byte, the
floor, the state hash before every 256th decision, and whether health lies off
its 0.05-HP grid; after the step, the reward and done flag. When the episode
ends it takes the terminal state's hash and a summary line: the ordinal, the
environment, the return, death and timeout, the achievements and, per floor,
whether it was reached, its decisions, kills, clearance and descent. Episodes
are handed over whole, in the order they end (:meth:`CaptureEnv.episodes`),
as records (:class:`Captured`): their token frames are not taken live, since
the game and replay are one implementation; replay regenerates them, and
checks every hash taken here on the way (``shards.py``).

Four per-episode settings serve later dataset versions and leave the first's
capture as it was when unset:

- epsilon range: each episode draws its own epsilon uniformly from
  ``[epsilon, epsilon_high]`` from its sampling seed, and replaces the
  policy's action by a uniform legal one with that probability;
- stall limit: a training episode that goes ``stall_limit`` decisions without
  a positive reward is handed over there, truncated: its last decision is not
  terminal. Its environment then reaches the timeout at its next, unrecorded
  decision. Validation episodes are never truncated;
- branches: every recorded episode starts from a state of another episode
  (``branches.py``), as a training episode of its parent's world seed;
- fixed worlds: episode ``n`` starts from the reset of ``world_seeds[n % k]``
  instead of a world of its own, keeping its split and sampling seed, so many
  episodes start from one State and stream and differ by their actions alone.

Recording starts at a reset. Once the recorded episodes that ended hold the
schedule's ``budget`` decisions, no further episode is recorded (the drain),
and :meth:`CaptureEnv.finished` once none is in flight; so is a branch
capture whose pool is used up. The other environments keep playing unrecorded
worlds drawn from their own streams. A failure -- an action outside the action
space, a reward outside the schema, an episode longer than ``max_decisions``
or a seed range used up -- drops that episode and stops recording; the
episodes that ended before it are still handed over, and ``failure`` names it.

Buffers step on the rollout's threads, so the schedule (the next ordinal, the
budget, the branch queue) is taken under a lock, and with several buffers the
order of their resets, so which world each environment draws, follows the
threads' timing. One buffer resets its environments in row order, so a capture
from one seed then records the same episodes every time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field
from typing import TYPE_CHECKING, Final, Protocol, Self, override

import dataclasses
import itertools
import threading

from configgle import Fig
from numpy.typing import NDArray

import numpy as np

from priml.baselines.craftax.env import GameRules
from priml.baselines.craftax.game.state import (
    ACTION_OBS_SIZE,
    ATN_DIM,
    MONSTERS_KILLED_TO_CLEAR_LEVEL,
    NO_ACTION,
    NUM_LEVELS,
    OBS_SIZE,
    SYMBOLIC_OBS_SIZE,
    env_state,
    new_states,
    new_stats,
)
from priml.baselines.craftax.game.step import (
    Rules,
    observe_numba,
    write_previous_action_numba,
)
from priml.baselines.craftax.game.world_gen import (
    DUNGEON_LEVEL_CONFIGS,
    SMOOTH_LEVEL_CONFIGS,
    generate_world_numba,
)
from priml.baselines.craftax.lib.arrays import ints
from priml.baselines.craftax.world_model.archive import (
    FloorTrace,
    Receipt,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    episode_seed,
    sampling_seed,
    splitmix64,
)
from priml.baselines.craftax.world_model.capture.step import (
    BAD_ACTION,
    ENDED,
    FULL,
    HASH_STRIDE,
    RECORDED_END,
    TOO_LONG,
    Rows,
    fnv1a_numba,
    record_rows_numba,
)
from priml.baselines.craftax.world_model.replay import (
    load,
    reset_world,
    save,
    xor_bytes,
)
from priml.lib.codec import from_plain, loads


if TYPE_CHECKING:
    import torch
else:
    from wrapt import lazy_import

    # Only construction and an episode's close touch torch here; ``archive``
    # imports it anyway, so this saves nothing today.
    torch = lazy_import("torch")


EPSILON_STREAM: Final = 0x5851_F42D_4C95_7F2D
"""XORed into a sampling seed to seed the episode's epsilon draw: a stream of its
own, so the override draws as it does with a fixed epsilon."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Schedule:
    """Which episodes a capture records: its worker's seed range and its budget.

    Attributes:
      arm: Behaviour-mixture arm, 0-3.
      worker: Capture worker, 0-3.
      generation: Seed generation, 0-9 (``seeds.py``).
      first_episode: Ordinal of the first episode to record.
      budget: Decisions of ended recorded episodes after which no further
        episode is recorded.

    """

    arm: int = 0
    worker: int = 0
    generation: int = 0
    first_episode: int = 0
    budget: int


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BranchStart:
    """Where one branch starts: a state of its parent.

    Attributes:
      world_seed: The parent's world seed.
      origin: The parent's state before the branch decision, XORed with the
        reset world of ``world_seed`` (``replay.origin``).
      point: What the summary records under ``"branch"``.

    """

    world_seed: int
    origin: bytes
    point: Mapping[str, object]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Captured:
    """One recorded episode as capture hands it over: what replay regenerates it from.

    Attributes:
      receipt: How to regenerate the episode.
      actions: Executed actions, uint8 ``[T]``.
      hashes: State hashes before decisions 0, 256, ... and after the last, as
        uint64 bits in int64 ``[ceil(T / 256) + 1]``.
      floors: Where the floor changes, and whether the episode died.
      summary: The summary line.
      origin: A branch's start state, as ``archive.Episode.origin``.
      truncated: Whether the last decision is not terminal.

    """

    receipt: Receipt
    actions: torch.Tensor
    hashes: torch.Tensor
    floors: FloorTrace
    summary: Mapping[str, object]
    origin: bytes = b""
    truncated: bool = False


class Branches(Protocol):
    """The branch starts of a branch capture, in ordinal order."""

    def start(self, ordinal: int) -> BranchStart | None:
        """Return branch ``ordinal``'s start, waiting for it; None once the pool is used up."""
        ...


class CaptureEnv:
    """Worlds the rollout steps as it steps ``CraftaxEnv``, recording every episode.

    Attributes:
      observations: float32 ``[num_envs, observation_size]``.
      action_mask: uint8 ``[num_envs, 43]``.
      rewards: float32 ``[num_envs]``.
      terminals: float32 ``[num_envs]``.
      actions: float32 ``[num_envs, 1]``, written by the rollout.
      num_envs: Environments.
      num_buffers: Buffers the rollout steps in turn.
      states: ``STATE_DTYPE [num_envs]``, the live worlds.
      rngs: uint32 ``[num_envs]``, each world's stream.
      failure: Empty, or why recording stopped.

    """

    class Config(Fig["CaptureEnv"]):
        """The environments and the per-episode settings."""

        num_envs: int = 64
        """Environments."""

        num_buffers: int = 1
        """Buffers; ``num_envs`` must divide among them."""

        rules: GameRules.Config = field(default_factory=GameRules.Config)
        """The observation layout the policy reads. The game plays its default
        rules, which replay reproduces, so only ``previous_action`` and
        ``symbolic_observation`` may differ from their defaults."""

        epsilon: float = 0.0
        """Probability of replacing the policy's action by a uniform legal one,
        or the lower end of each episode's draw of it; held as float32."""

        epsilon_high: float = 0.0
        """Upper end of each episode's epsilon draw; at most ``epsilon`` draws none."""

        stall_limit: int | None = None
        """Decisions without a positive reward that truncate a training episode,
        at least 1; None truncates none."""

        max_decisions: int = 100_000
        """Longest episode recorded: the timeout's 100,000 ticks."""

        chunk_decisions: int = 4_096
        """Decisions each environment records before Python moves them into
        its episode, a positive multiple of 256: about 7 bytes each."""

        world_seeds: tuple[int, ...] = ()
        """World seeds the episodes start from in turn, ordinal ``n`` from
        ``world_seeds[n % len(world_seeds)]``; empty gives every episode its own
        (``seeds.episode_seed``). A branch starts from its parent's world."""

        @property
        def observation_size(self) -> int:
            """Floats per observation: the symbolic view's, or the packed one's."""
            if self.rules.symbolic_observation:
                return SYMBOLIC_OBS_SIZE
            return ACTION_OBS_SIZE if self.rules.previous_action else OBS_SIZE

        @override
        def finalize(self) -> Self:
            if self.num_buffers <= 0 or self.num_envs % self.num_buffers:
                raise ValueError(
                    f"num_envs={self.num_envs} must divide among "
                    f"num_buffers={self.num_buffers} > 0.",
                )
            if self.epsilon < 0 or self.epsilon_high > 1 or self.max_decisions <= 0:
                raise ValueError(
                    "Expected 0 <= epsilon, epsilon_high <= 1 and max_decisions > 0.",
                )
            if self.chunk_decisions <= 0 or self.chunk_decisions % HASH_STRIDE:
                raise ValueError(
                    f"chunk_decisions={self.chunk_decisions} is not a positive "
                    "multiple of 256.",
                )
            # The kernel truncates once a stall reaches the limit, so 0 would
            # cut every training episode after its first decision.
            if self.stall_limit is not None and self.stall_limit < 1:
                raise ValueError(
                    f"stall_limit={self.stall_limit} is not positive; None "
                    "truncates no episode.",
                )
            return super().finalize()

    def __init__(
        self,
        config: Config,
        *,
        schedule: Schedule,
        branches: Branches | None = None,
    ) -> None:
        """Reset every environment into its first world.

        Args:
          config: The environments and the per-episode settings.
          schedule: The worker's seed range and budget.
          branches: The branch starts of a branch capture; None records fresh
            episodes.

        Raises:
          ValueError: The rules change the game, not just its observation.

        """
        config = config.copy_tree().finalize()
        self.rules = config.rules.make()
        game = self.rules._replace(previous_action=None, symbolic_observation=None)
        if game != Rules():
            raise ValueError(
                "Capture plays the game's default rules, which replay "
                "reproduces; only the observation layout may change.",
            )
        self.config = config
        self.schedule = schedule
        self.num_envs = config.num_envs
        self.num_buffers = config.num_buffers
        count = config.num_envs
        pinned = torch.cuda.is_available()
        self.observations = torch.zeros(
            count,
            config.observation_size,
            pin_memory=pinned,
        )
        self.action_mask = torch.zeros(
            count,
            ATN_DIM,
            dtype=torch.uint8,
            pin_memory=pinned,
        )
        self.rewards = torch.zeros(count, pin_memory=pinned)
        self.terminals = torch.zeros(count, pin_memory=pinned)
        self.actions = torch.zeros(count, 1, pin_memory=pinned)
        self.states = new_states(count)
        # An unrecorded world draws from the environment's own stream, which
        # starts at its index.
        self.rngs = np.arange(count, dtype=np.uint32)
        self.failure = ""
        self._rows = _rows(
            self,
            chunk=config.chunk_decisions,
        )
        self._played = Rules()
        self._branches = branches
        self._lock = threading.Lock()
        self._ordinal = schedule.first_episode
        self._ended_decisions = 0
        self._draining = False
        self._finished: list[Captured] = []
        self._open: list[_Open | None] = [None] * count
        for env in range(count):
            self._reset(env)
            self._observe(env, previous=NO_ACTION)

    def buffer_slice(self, buffer: int) -> slice:
        """Return the rows of every buffer attribute that belong to ``buffer``."""
        rows = self.num_envs // self.num_buffers
        return slice(buffer * rows, (buffer + 1) * rows)

    def step_buffer(self, buffer: int) -> None:
        """Step one buffer by its ``actions`` rows, recording as it goes.

        The rows step in one ``nogil`` kernel; the rows it flags are then
        handled in row order: chunks moved, episodes handed over, failures
        kept and worlds reset.

        Args:
          buffer: Which buffer to step.

        """
        rows = self.buffer_slice(buffer)
        start, stop, _ = rows.indices(self.num_envs)
        stall = self.config.stall_limit
        record_rows_numba(
            self._rows,
            start,
            stop,
            -1 if stall is None else stall,
            self.config.max_decisions,
            self.rules.max_timesteps,
            self._played,
            self.rules,
        )
        for env in ints(np.flatnonzero(self._rows.flags[rows]) + start):
            self._handle(env)

    def receipt(self, env: int) -> Receipt | None:
        """Return the receipt of the episode ``env`` records, or None while unrecorded."""
        episode = self._open[env]
        return None if episode is None else episode.receipt

    def decisions(self, env: int) -> int:
        """Return the decisions ``env``'s recorded episode holds; 0 while unrecorded."""
        return self._rows.decisions.item(env) if self._open[env] is not None else 0

    def returned(self, env: int) -> np.float32:
        """Return the achievement return of ``env``'s recorded episode; 0 while unrecorded."""
        if self._open[env] is None:
            return np.float32(0.0)
        return np.float32(self._rows.returned.item(env))

    def episodes(self) -> list[Captured]:
        """Return the episodes ended since the last call, in the order they ended."""
        with self._lock:
            finished, self._finished = self._finished, []
        return finished

    def in_flight(self) -> int:
        """Return how many recorded episodes have not ended."""
        return sum(episode is not None for episode in self._open)

    def finished(self) -> bool:
        """Return whether no episode will end after the ones ``episodes`` still holds."""
        with self._lock:
            stopped = bool(self.failure) or self._draining
        return stopped and not self.in_flight()

    def drain(self) -> None:
        """Record no further episode; the ones in flight play to their end."""
        with self._lock:
            self._draining = True

    def _handle(self, env: int) -> None:
        """Handle one flagged row: its chunk, its episode's end or failure, its reset."""
        flags = self._rows.flags.item(env)
        episode = self._open[env]
        if flags & TOO_LONG:
            self._fail(
                env,
                f"An episode exceeds {self.config.max_decisions} decisions.",
            )
        elif flags & BAD_ACTION:
            self._fail(
                env,
                f"An action of environment {env} is outside 0-{ATN_DIM - 1}.",
            )
        elif episode is not None and flags & (FULL | RECORDED_END):
            self._flush(env, episode)
            if flags & RECORDED_END:
                self._publish(env, episode)
        if flags & ENDED:
            self._reset(env)
            self._observe(env, previous=NO_ACTION)

    def _flush(self, env: int, episode: _Open) -> None:
        """Move ``env``'s chunk into its episode."""
        rows = self._rows
        fill = rows.fill.item(env)
        episode.actions.append(rows.chunk_actions[env, :fill].copy())
        episode.rewards.append(rows.chunk_rewards[env, :fill].copy())
        episode.done.append(rows.chunk_done[env, :fill].copy())
        episode.floors.append(rows.chunk_floors[env, :fill].copy())
        episode.hashes.append(rows.chunk_hashes[env, : rows.hash_fill[env]].copy())
        rows.fill[env] = 0
        rows.hash_fill[env] = 0

    def _publish(self, env: int, episode: _Open) -> None:
        """Hand over ``env``'s episode; its last hash is of the State then."""
        rows = self._rows
        self._open[env] = None
        rewards = np.concatenate(episode.rewards)
        integral = (
            np.equal(rewards, np.floor(rewards)) & (rewards >= -1) & (rewards <= 234)
        )
        if not integral.all():
            self._fail(env, "A reward lies outside the schema.")
            return
        tokens = rewards.astype(np.int16)
        done = np.concatenate(episode.done).astype(np.bool_)
        floors = np.concatenate(episode.floors).astype(np.int64)
        state = env_state(self.states, env)
        death = bool(state.player_health <= 0)
        closed = Captured(
            receipt=episode.receipt,
            actions=torch.from_numpy(np.concatenate(episode.actions)),
            hashes=torch.from_numpy(
                np.concatenate([*episode.hashes, rows.final_hash[env : env + 1]]).view(
                    np.int64,
                ),
            ),
            floors=_floor_trace(
                floors,
                died=done.item(-1) and tokens.item(-1) == -1,
            ),
            summary=_summary(
                episode,
                floors=[*ints(floors), int(state.player_level)],
                killed=np.array(state.monsters_killed, dtype=np.int64),
                achievements=ints(np.flatnonzero(state.achievements)),
                returned=np.float32(rows.returned.item(env)),
                health_snaps=rows.health_snaps.item(env),
                death=death,
                timeout=not death and state.timestep >= self.rules.max_timesteps,
                truncated=not done.item(-1),
                ranged=bool(
                    np.float32(self.config.epsilon_high)
                    > np.float32(self.config.epsilon),
                ),
            ),
            origin=episode.origin,
            truncated=not done.item(-1),
        )
        with self._lock:
            self._finished.append(closed)
            self._ended_decisions += len(closed.actions)
            if self._ended_decisions >= self.schedule.budget:
                self._draining = True

    def _fail(self, env: int, reason: str) -> None:
        """Drop ``env``'s episode and stop recording; the first reason is kept."""
        self._open[env] = None
        self._rows.recording[env] = 0
        with self._lock:
            self.failure = self.failure or reason

    def _reset(self, env: int) -> None:
        """Put ``env`` in its next world: recorded from the schedule, or its own."""
        branch: BranchStart | None = None
        with self._lock:
            record = not self.failure and not self._draining
            ordinal = self._ordinal
            if record and self._branches is not None:
                branch = self._branches.start(ordinal)
                record = branch is not None
                self._draining |= branch is None
            self._ordinal += record
        if not record:
            # A generated world zeroes the State first.
            self._rows.state_bytes[env] = 0
            generate_world_numba(
                env_state(self.states, env),
                self.rngs[env : env + 1],
                SMOOTH_LEVEL_CONFIGS,
                DUNGEON_LEVEL_CONFIGS,
            )
            return
        try:
            self._open[env] = self._start(env, ordinal, branch=branch)
        except ValueError as error:
            self._fail(env, str(error))

    def _start(self, env: int, ordinal: int, *, branch: BranchStart | None) -> _Open:
        """Set ``env`` to episode ``ordinal``'s start and open its record."""
        schedule = self.schedule
        worlds = self.config.world_seeds
        if branch is None:
            split, world_seed = episode_seed(
                ordinal,
                arm=schedule.arm,
                worker=schedule.worker,
                generation=schedule.generation,
            )
            if worlds:
                world_seed = worlds[ordinal % len(worlds)]
        else:
            split, world_seed = TRAIN, branch.world_seed
        seed = sampling_seed(
            ordinal,
            arm=schedule.arm,
            worker=schedule.worker,
            environment=env,
            generation=schedule.generation,
        )
        states, rng = reset_world(world_seed)
        if branch is not None:
            states, rng = load(xor_bytes(save(states, rng), right=branch.origin))
        self.states[env] = states[0]
        self.rngs[env] = rng[0]
        epsilon = np.float32(self.config.epsilon)
        high = np.float32(self.config.epsilon_high)
        if high > epsilon:
            _, draw = splitmix64(seed ^ EPSILON_STREAM)
            span = float(np.float32(high - epsilon))
            epsilon = np.float32(epsilon + np.float32((draw >> 11) * 2.0**-53 * span))
        rows = self._rows
        for name in ("decisions", "fill", "hash_fill", "last_gain", "health_snaps"):
            getattr(rows, name)[env] = 0
        rows.returned[env] = 0
        rows.epsilon[env] = epsilon
        rows.epsilon_rng[env] = seed
        rows.split[env] = split
        rows.recording[env] = 1
        return _Open(
            receipt=Receipt(
                world_seed=world_seed,
                sampling_seed=seed,
                initial_state_hash=int(fnv1a_numba(rows.state_bytes[env, :])),
                arm=schedule.arm,
                split=split,
            ),
            ordinal=ordinal,
            environment=env,
            epsilon=epsilon,
            killed=np.array(
                env_state(self.states, env).monsters_killed,
                dtype=np.int64,
            ),
            origin=b"" if branch is None else branch.origin,
            point=None if branch is None else branch.point,
        )

    def _observe(self, env: int, *, previous: int) -> None:
        """Write ``env``'s observation and mask rows, as the game's step ends."""
        observation = self._rows.observations[env, :]
        observe_numba(
            env_state(self.states, env),
            observation,
            self._rows.masks[env, :],
            self.rules,
        )
        write_previous_action_numba(observation, previous, self.rules)


@dataclasses.dataclass(slots=True, kw_only=True)
class _Open:
    """An episode being recorded: its receipt, identity, start and moved chunks.

    Attributes:
      killed: Monsters killed per floor at the start, int64 ``[9]``.

    """

    receipt: Receipt
    ordinal: int
    environment: int
    epsilon: np.float32
    killed: NDArray[np.int64]
    origin: bytes
    point: Mapping[str, object] | None
    actions: list[NDArray[np.uint8]] = field(default_factory=list)
    rewards: list[NDArray[np.float32]] = field(default_factory=list)
    done: list[NDArray[np.uint8]] = field(default_factory=list)
    floors: list[NDArray[np.uint8]] = field(default_factory=list)
    hashes: list[NDArray[np.uint64]] = field(default_factory=list)


def _rows(env: CaptureEnv, *, chunk: int) -> Rows:
    """Return the step kernel's arrays over ``env``'s worlds and buffers."""
    count = env.num_envs
    return Rows(
        states=env.states,
        state_bytes=env.states.view(np.uint8).reshape(count, -1),
        rngs=env.rngs,
        stats=new_stats(count),
        actions=env.actions.numpy(),
        masks=env.action_mask.numpy(),
        observations=env.observations.numpy(),
        rewards=env.rewards.numpy(),
        terminals=env.terminals.numpy(),
        recording=np.zeros(count, np.uint8),
        decisions=np.zeros(count, np.int64),
        fill=np.zeros(count, np.int64),
        chunk_actions=np.zeros((count, chunk), np.uint8),
        chunk_rewards=np.zeros((count, chunk), np.float32),
        chunk_done=np.zeros((count, chunk), np.uint8),
        chunk_floors=np.zeros((count, chunk), np.uint8),
        chunk_hashes=np.zeros((count, chunk // HASH_STRIDE), np.uint64),
        hash_fill=np.zeros(count, np.int64),
        final_hash=np.zeros(count, np.uint64),
        epsilon=np.zeros(count, np.float32),
        epsilon_rng=np.zeros(count, np.uint64),
        last_gain=np.zeros(count, np.int64),
        health_snaps=np.zeros(count, np.int64),
        returned=np.zeros(count, np.float32),
        split=np.zeros(count, np.uint8),
        flags=np.zeros(count, np.uint8),
    )


def _floor_trace(floors: NDArray[np.int64], *, died: bool) -> FloorTrace:
    """Return the floor trace of an episode's per-decision floors (``index.floor_trace``)."""
    starts = np.concatenate(
        [[0], np.flatnonzero(np.not_equal(floors[1:], floors[:-1])) + 1],
    )
    return FloorTrace(
        changes=tuple(zip(ints(starts), ints(floors[starts]), strict=True)),
        died=died,
    )


# ``floors`` is each decision's floor, then the floor after the last. A floor is reached
# when the episode stood on it; its kills are the monsters killed there since the start,
# a count that changes only while the player is on it; it is cleared when reached with
# eight kills or more; and descended when the player went down its ladder. The text,
# with ``%.9g`` floats, is the form every archive's summaries were parsed from, so a
# value reads back as the same JSON number.
def _summary(
    episode: _Open,
    *,
    floors: list[int],
    killed: NDArray[np.int64],
    achievements: list[int],
    returned: np.float32,
    health_snaps: int,
    death: bool,
    timeout: bool,
    truncated: bool,
    ranged: bool,
) -> dict[str, object]:
    """Return an episode's summary line, written as text and parsed as JSON."""
    visits = np.bincount(floors[:-1], minlength=NUM_LEVELS)
    reached = np.zeros(NUM_LEVELS, dtype=np.int64)
    reached[floors] = 1
    descended = np.zeros(NUM_LEVELS, dtype=np.int64)
    for here, there in itertools.pairwise(floors):
        if there == here + 1:
            descended[here] = 1
    kills = killed - episode.killed
    cleared = reached & (killed >= MONSTERS_KILLED_TO_CLEAR_LEVEL)
    text = (
        f'{{"episode":{episode.ordinal},"environment":{episode.environment},'
        f'"return":{_g9(returned)},"death":{int(death)},'
        f'"timeout":{int(timeout)},'
    )
    if ranged:
        text += f'"epsilon":{_g9(episode.epsilon)},'
    if truncated:
        text += '"truncated":1,'
    if health_snaps:
        text += f'"health_snaps":{health_snaps},'
    text += f'"achievements":[{",".join(map(str, achievements))}],"floors":['
    text += ",".join(
        f'{{"reached":{reached[k]},"decisions":{visits[k]},"kills":{kills[k]},'
        f'"cleared":{cleared[k]},"descended":{descended[k]}}}'
        for k in range(NUM_LEVELS)
    )
    summary = from_plain(loads(text + "]}"), dict[str, object])
    if episode.point is not None:
        summary["branch"] = dict(episode.point)
    return summary


def _g9(value: np.float32) -> str:
    """Return a float32 as ``printf("%.9g")`` writes it promoted to double."""
    return f"{float(value):.9g}"
