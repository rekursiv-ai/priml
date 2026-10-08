"""Tests for the vectorized environment's buffer contract and its helper threads."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hashlib
import math
import platform
import sys
import threading
import time

from llvmlite import ir
from numba import njit
from numba.core.registry import cpu_target

import numba.core.types as nbtypes
import numpy as np
import pytest
import torch

from priml.baselines.craftax.env import (
    ROW,
    SERVE_STOPPED,
    CraftaxEnv,
    FreshWorlds,
    StallCap,
    WorldPool,
    _atomic_load_signature,
    _atomic_store_signature,
    _emit_atomic_load,
    _emit_atomic_store,
    _emit_pause,
    _emit_ticks,
    _pause_signature,
    _ticks_signature,
    clock_numba,
    lead_numba,
    ticks_per_second,
)
from priml.baselines.craftax.game.jit import jit
from priml.baselines.craftax.game.rng import rand_r_numba
from priml.baselines.craftax.game.state import (
    ACTION_OBS_SIZE,
    NUM_BLOCK_TYPES,
    OBS_SIZE,
    STATE_DTYPE,
    STATS_DTYPE,
    SYMBOLIC_OBS_SIZE,
    SYMBOLIC_TILE_CHANNELS,
    TRAINING_STATS_DTYPE,
)
from priml.baselines.craftax.game.step import Rules
from priml.baselines.craftax.game.testing import ir_builder
from priml.baselines.craftax.game.world_gen import (
    DUNGEON_LEVEL_CONFIGS,
    SMOOTH_LEVEL_CONFIGS,
)
from priml.baselines.craftax.learners.practice import FrontierPractice
from priml.baselines.craftax.testing import (
    assert_golden,
    digest,
    env_digests,
    masked_random_actions_numba,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numba.core.base import BaseContext
    from numba.core.typing.templates import Signature

    from priml.baselines.craftax.game.state import Array1, Array2


def _env(num_envs: int = 8, num_buffers: int = 2, num_worlds: int = 3) -> CraftaxEnv:
    cfg = CraftaxEnv.Config()
    cfg.num_envs = num_envs
    cfg.num_buffers = num_buffers
    cfg.restart = _pool(num_worlds)
    return cfg.make()


def _pool(num_worlds: int) -> WorldPool.Config:
    config = WorldPool.Config()
    config.num_worlds = num_worlds
    return config


@pytest.mark.compute_large_fixture
def test_buffer_attributes_have_the_contracted_shapes_and_dtypes() -> None:
    env = _env()
    assert env.observations.shape == (8, 843)
    assert env.observations.dtype == torch.float32
    assert env.action_mask.shape == (8, 43)
    assert env.action_mask.dtype == torch.uint8
    assert env.rewards.shape == (8,)
    assert env.rewards.dtype == torch.float32
    assert env.terminals.shape == (8,)
    assert env.terminals.dtype == torch.float32
    assert env.actions.shape == (8, 1)
    assert env.actions.dtype == torch.float32


@pytest.mark.compute_large_fixture
def test_buffers_are_contiguous_host_tensors_pinned_where_cuda_exists() -> None:
    env = _env()
    for buffer in (
        env.observations,
        env.action_mask,
        env.rewards,
        env.terminals,
        env.actions,
    ):
        assert buffer.device.type == "cpu"
        assert buffer.is_contiguous()
        assert buffer.is_pinned() == torch.cuda.is_available()


@pytest.mark.compute_large_fixture
def test_buffers_stay_on_the_host_when_built_under_another_default_device() -> None:
    """The loop builds its step under the runtime's device; the C writes host rows."""
    with torch.device("meta"):
        env = _env()
    for buffer in (
        env.observations,
        env.action_mask,
        env.rewards,
        env.terminals,
        env.actions,
        env.save_slots,
        env.restore_slots,
    ):
        assert buffer.device.type == "cpu"


@pytest.mark.compute_large_fixture
def test_mask_starts_all_ones_and_the_rest_zero() -> None:
    env = _env()
    assert torch.equal(env.action_mask, torch.ones(8, 43, dtype=torch.uint8))
    assert not env.observations.any()
    assert not env.rewards.any()
    assert not env.terminals.any()
    assert not env.actions.any()


@pytest.mark.compute_large_fixture
def test_state_arrays_follow_the_configured_sizes() -> None:
    env = _env(num_envs=6, num_buffers=3, num_worlds=5)
    assert env.states.dtype == STATE_DTYPE
    assert env.states.shape == (6,)
    assert env.pool.shape == (5,)
    assert env.stats.dtype == STATS_DTYPE
    assert env.stats.shape == (6,)
    assert np.equal(env.stats["first_undefined_step"], -1).all()
    assert env.rngs.dtype == np.uint32
    assert env.rngs.tolist() == [0, 1, 2, 3, 4, 5]


@pytest.mark.compute_large_fixture
def test_seed_is_an_offset_added_to_the_env_index() -> None:
    cfg = CraftaxEnv.Config()
    cfg.num_envs = 4
    cfg.num_buffers = 1
    cfg.restart = _pool(1)
    cfg.seed = 100
    assert cfg.make().rngs.tolist() == [100, 101, 102, 103]


@pytest.mark.compute_large_fixture
def test_buffer_slices_tile_the_rows_in_order() -> None:
    env = _env(num_envs=8, num_buffers=2)
    assert env.envs_per_buffer == 4
    assert env.buffer_slice(0) == slice(0, 4)
    assert env.buffer_slice(1) == slice(4, 8)
    assert env.observations[env.buffer_slice(1)].shape == (4, 843)


@pytest.mark.parametrize(
    ("num_envs", "num_buffers", "num_worlds", "threads", "max_timesteps"),
    [
        (7, 2, 3, 1, 10),
        (0, 1, 3, 1, 10),
        (8, 0, 3, 1, 10),
        (8, 2, 0, 1, 10),
        (8, 2, 3, 0, 10),
        (8, 2, 3, 1, 0),
    ],
)
def test_invalid_sizes_are_rejected(
    num_envs: int,
    num_buffers: int,
    num_worlds: int,
    threads: int,
    max_timesteps: int,
) -> None:
    cfg = CraftaxEnv.Config()
    cfg.num_envs = num_envs
    cfg.num_buffers = num_buffers
    cfg.restart = _pool(num_worlds)
    cfg.threads_per_buffer = threads
    cfg.rules.max_timesteps = max_timesteps
    with pytest.raises(ValueError, match="must be"):
        cfg.make()


def _original_rules_env() -> CraftaxEnv:
    """Return 4 environments on fresh worlds with every rule of original Craftax set."""
    cfg = CraftaxEnv.Config()
    cfg.num_envs = 4
    cfg.num_buffers = 2
    # One thread: a helper would compile the step with every option on for two
    # more kernels (serve, follow), and these tests are about the rules.
    cfg.threads_per_buffer = 1
    cfg.restart = FreshWorlds.Config()
    rules = cfg.rules
    rules.original_reward = True
    rules.collapse_sleep = False
    rules.action_mask = False
    rules.end_on_boss_defeat = True
    rules.max_timesteps = 7
    rules.symbolic_observation = True
    return cfg.make()


def test_the_config_passes_the_rules_to_the_step() -> None:
    # A unit test because construction compiles no step kernel. A reset here
    # would compile the step for these rules, 64 s on a cold cache (measured on
    # the EPYC); the next test plays it.
    env = _original_rules_env()
    assert env._batch.rules == Rules(
        original_reward=True,
        collapse_sleep=False,
        action_mask=False,
        end_on_boss_defeat=True,
        max_timesteps=7,
        fresh_worlds=True,
        symbolic_observation=True,
    )
    assert env.pool.shape == (1,)
    assert env.pool.tobytes() == bytes(STATE_DTYPE.itemsize)
    assert env.observations.shape == (4, SYMBOLIC_OBS_SIZE)
    env.close()


@pytest.mark.compute_large_fixture
def test_the_original_rules_reset_into_the_unmasked_symbolic_view_and_step() -> None:
    env = _original_rules_env()
    env.reset()
    assert env.action_mask.all()
    # Every lit tile is one-hot in block, item and the visible flag.
    tiles = env.observations[:, : 99 * SYMBOLIC_TILE_CHANNELS].reshape(
        4,
        99,
        -1,
    )
    lit = tiles[:, :, -1] == 1
    assert lit.any()
    assert (tiles[:, :, :NUM_BLOCK_TYPES].sum(-1)[lit] == 1).all()
    env.step_buffer(0)
    env.close()


@pytest.mark.compute_large_fixture
def test_the_default_config_plays_the_default_rules() -> None:
    # Apart from the options test, so each pays one compile: together, the step
    # with every option on and this pool's world generator took 57 s of the
    # timeout's 60 on a cold cache (measured on the Xeon).
    env = _env()
    assert env._batch.rules == Rules()
    env.close()


@pytest.mark.compute_large_fixture
def test_reset_writes_first_observations_and_clears_the_step_outputs() -> None:
    env = _env(num_envs=4, num_buffers=2, num_worlds=5)
    env.reset()
    assert env.states[0:1].tobytes() != bytes(80_248)
    assert (env.observations[:, 792:] != 0).any()
    assert env.action_mask[:, 0].tolist() == [1, 1, 1, 1]
    assert not env.rewards.any()
    assert not env.terminals.any()
    env.close()


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize(
    ("threads", "spin_seconds"),
    [(1, 0.005), (2, 0.005), (3, 0.005), (3, 0.0), (66, 0.0)],
    ids=["1", "2", "3", "3-sleeping-helpers", "66-sleeping-helpers"],
)
def test_stepping_a_buffer_gives_the_same_bits_at_every_thread_count(
    threads: int,
    spin_seconds: float,
) -> None:
    # With no spin budget every helper parks after each share, so every step
    # also takes the wake path.
    def run(threads: int) -> tuple[bytes, np.ndarray, np.ndarray]:
        cfg = CraftaxEnv.Config()
        cfg.num_envs = 12
        cfg.num_buffers = 2
        cfg.restart = _pool(4)
        cfg.threads_per_buffer = threads
        cfg.spin_seconds = spin_seconds
        env = cfg.make()
        env.reset()
        draws = np.random.default_rng(0)
        for _ in range(8):
            for buffer in range(2):
                rows = env.buffer_slice(buffer)
                legal = env.action_mask[rows].numpy()
                start, stop, _ = rows.indices(env.num_envs)
                choice = draws.integers(0, 43, size=stop - start)
                env.actions[rows, 0] = torch.from_numpy(
                    np.where(
                        legal[np.arange(len(legal)), choice],
                        choice,
                        0,
                    ).astype(np.float32),
                )
                env.step_buffer(buffer)
        result = (
            env.states.tobytes(),
            env.observations.numpy().copy(),
            env.rewards.numpy().copy(),
        )
        env.close()
        return result

    reference = run(1)
    actual = run(threads)
    assert actual[0] == reference[0]
    assert np.array_equal(actual[1], reference[1])
    assert np.array_equal(actual[2], reference[2])


@jit
def _pick_actions(
    masks: Array2[int],
    actions: Array2[np.float32],
    start: int,
    stop: int,
    calls: Array1[int],
) -> int:
    """Stand in for a rollout's prepare: write a legal action per row, count the call."""
    calls[0] += 1
    for i in range(start, stop):
        choice = (calls[0] * 7 + i * 13) % 43
        actions[i, 0] = np.float32(choice if masks[i, choice] else 0)
    return 0


@jit
def _fail_third_call(calls: Array1[int]) -> int:
    calls[0] += 1
    return 7 if calls[0] == 3 else 0


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize(
    ("threads", "spin_seconds"),
    [(1, 0.005), (2, 0.005), (3, 0.0)],
    ids=["1", "2", "3-sleeping-helpers"],
)
def test_run_buffer_matches_preparing_and_stepping_from_python(
    threads: int,
    spin_seconds: float,
) -> None:
    def make() -> CraftaxEnv:
        cfg = CraftaxEnv.Config()
        cfg.num_envs = 12
        cfg.num_buffers = 2
        cfg.restart = _pool(4)
        cfg.threads_per_buffer = threads
        cfg.spin_seconds = spin_seconds
        env = cfg.make()
        env.reset()
        return env

    looped, ran = make(), make()
    for buffer in range(2):
        start, stop, _ = looped.buffer_slice(buffer).indices(looped.num_envs)
        calls = np.zeros(1, dtype=np.int64)
        for _ in range(8):
            _pick_actions(
                looped.action_mask.numpy(), looped.actions.numpy(), start, stop, calls,
            )  # fmt: skip
            looped.step_buffer(buffer)
        calls = np.zeros(1, dtype=np.int64)
        ran.run_buffer(
            buffer,
            8,
            _pick_actions,
            (
                ran.action_mask.numpy(),
                ran.actions.numpy(),
                start,
                stop,
                calls,
            ),
        )
        assert calls[0] == 8
    assert ran.states.tobytes() == looped.states.tobytes()
    assert ran.stats.tobytes() == looped.stats.tobytes()
    assert torch.equal(ran.observations, looped.observations)
    assert torch.equal(ran.action_mask, looped.action_mask)
    assert torch.equal(ran.rewards, looped.rewards)
    assert torch.equal(ran.terminals, looped.terminals)
    looped.close()
    ran.close()


@pytest.mark.compute_large_fixture
def test_run_buffer_stops_at_a_failed_prepare_after_the_steps_before_it() -> None:
    env = _env()
    env.reset()
    calls = np.zeros(1, dtype=np.int64)
    with pytest.raises(RuntimeError, match="returned 7"):
        env.run_buffer(0, 8, _fail_third_call, (calls,))
    assert calls[0] == 3
    rows = env.buffer_slice(0)
    assert env.stats["steps"][rows].tolist() == [2] * len(
        range(*rows.indices(env.num_envs)),
    )
    env.close()


@pytest.mark.compute_large_fixture
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_failed_helper_makes_the_step_raise_instead_of_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*args: object) -> int:
        del args
        raise ValueError("a broken helper")

    monkeypatch.setattr("priml.baselines.craftax.env.serve_numba", broken)
    env = _env()
    env.reset()
    with pytest.raises(RuntimeError, match="helper thread failed"):
        env.step_buffer(0)
    env.close()


@pytest.mark.compute_large_fixture
def test_a_closed_environment_refuses_to_step() -> None:
    env = _env()
    env.reset()
    env.step_buffer(0)
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.step_buffer(0)


@pytest.mark.compute_large_fixture
def test_a_share_posted_after_its_helper_stopped_is_not_waited_for() -> None:
    # As when close() lands between a step posting its ticket and the helper
    # reading it: the helper leaves serve without doing the share.
    env = _env()
    env.reset()
    env.step_buffer(0)
    team = env._teams[0]
    env.close()
    status = lead_numba(
        env._batch,
        env.pool,
        team._archive,
        team._control,
        team._ticket + 1,
        team._bounds,
    )
    assert status < 0


@pytest.mark.compute_large_fixture
def test_the_spin_budget_spans_spin_seconds_of_the_cycle_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A fresh dispatcher: its first call compiles, as a process's first clock
    # read loads the kernel, which a budget measured across it must not count.
    # The rate is measured once per process, so this test measures it anew.
    fresh_clock = njit(nogil=True)(clock_numba.py_func)
    monkeypatch.setattr("priml.baselines.craftax.env.clock_numba", fresh_clock)
    ticks_per_second.cache_clear()
    budgets: list[int] = []

    def record(
        batch: object,
        pool: object,
        control: object,
        row: int,
        budget: int,
    ) -> int:
        del batch, pool, control, row
        budgets.append(budget)
        return SERVE_STOPPED

    monkeypatch.setattr("priml.baselines.craftax.env.serve_numba", record)
    cfg = CraftaxEnv.Config()
    cfg.num_envs = 2
    cfg.num_buffers = 1
    cfg.restart = FreshWorlds.Config()
    cfg.spin_seconds = 0.005
    cfg.make().close()
    # The reference rate, with the clock's kernel compiled by now.
    started, started_ticks = time.perf_counter(), fresh_clock()
    time.sleep(0.05)
    rate = (fresh_clock() - started_ticks) / (time.perf_counter() - started)
    assert budgets == [pytest.approx(cfg.spin_seconds * rate, rel=0.2)]


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize("threads", [2, 3, 4, 5, 8])
def test_each_helper_row_of_the_control_array_is_its_own_cache_block(
    threads: int,
) -> None:
    # 128 bytes: Apple's cache line, and the pair of 64-byte lines Intel's
    # adjacent-line prefetcher fetches together.
    cfg = CraftaxEnv.Config()
    cfg.num_envs = 8
    cfg.num_buffers = 1
    cfg.restart = _pool(1)
    cfg.threads_per_buffer = threads
    env = cfg.make()
    control = env._teams[0]._control
    assert ROW * control.itemsize == 128
    assert control.ctypes.data % 128 == 0
    env.close()


@pytest.mark.compute_large_fixture
def test_a_helper_that_cannot_start_stops_the_helpers_before_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[threading.Thread] = []
    start = threading.Thread.start

    def start_one(thread: threading.Thread) -> None:
        if started:
            raise RuntimeError("can't start new thread")
        started.append(thread)
        start(thread)

    monkeypatch.setattr(threading.Thread, "start", start_one)
    cfg = CraftaxEnv.Config()
    cfg.num_envs = 6
    cfg.num_buffers = 1
    cfg.restart = _pool(1)
    cfg.threads_per_buffer = 3
    with pytest.raises(RuntimeError, match="start new thread"):
        cfg.make()
    started[0].join(timeout=10)
    assert not started[0].is_alive()


def test_atomic_load_types_a_contiguous_int64_vector_as_a_word_load() -> None:
    vector = nbtypes.Array(nbtypes.int64, 1, "C")
    assert _atomic_load_signature(None, vector, nbtypes.intp) == (
        nbtypes.int64(vector, nbtypes.intp),
        _emit_atomic_load,
    )


@pytest.mark.parametrize(
    "array",
    [
        nbtypes.Array(nbtypes.int32, 1, "C"),
        nbtypes.Array(nbtypes.int64, 2, "C"),
        nbtypes.Array(nbtypes.int64, 1, "A"),
        nbtypes.int64,
    ],
    ids=["int32", "matrix", "strided", "scalar"],
)
def test_atomic_load_types_nothing_but_a_contiguous_int64_vector(
    array: nbtypes.Type,
) -> None:
    # None tells Numba this overload does not apply, so it rejects the call.
    assert _atomic_load_signature(None, array, nbtypes.intp) is None


def test_the_store_the_pause_and_the_clock_type_their_calls() -> None:
    vector = nbtypes.Array(nbtypes.int64, 1, "C")
    assert _atomic_store_signature(None, vector, nbtypes.intp, nbtypes.int64) == (
        nbtypes.void(vector, nbtypes.intp, nbtypes.int64),
        _emit_atomic_store,
    )
    scalar = nbtypes.int64
    assert _atomic_store_signature(None, scalar, nbtypes.intp, scalar) is None
    assert _pause_signature(None) == (nbtypes.void(), _emit_pause)
    assert _ticks_signature(None) == (nbtypes.int64(), _emit_ticks)


@pytest.mark.parametrize(
    ("emit", "emitted"),
    [
        (_emit_atomic_load, "load atomic i64"),
        (_emit_atomic_store, "store atomic i64"),
        (_emit_ticks, 'call i64 @"llvm.readcyclecounter"()'),
    ],
    ids=["load", "store", "ticks"],
)
def test_each_word_intrinsic_emits_its_instruction(
    emit: Callable[
        [BaseContext, ir.IRBuilder, Signature, Sequence[ir.Value]],
        ir.Value,
    ],
    emitted: str,
) -> None:
    builder, args = _vector_builder()
    emit(cpu_target.target_context, builder, nbtypes.void(), args)
    assert emitted in str(builder.module)


@pytest.mark.parametrize(
    ("machine", "hint"),
    [
        ("arm64", 'call void @"llvm.aarch64.hint"(i32 1)'),
        ("x86_64", 'call void @"llvm.x86.sse2.pause"()'),
    ],
)
def test_the_pause_emits_the_platforms_spin_wait_hint(
    monkeypatch: pytest.MonkeyPatch,
    machine: str,
    hint: str,
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: machine)
    builder, args = _vector_builder()
    _emit_pause(cpu_target.target_context, builder, nbtypes.void(), args)
    assert hint in str(builder.module)


def _vector_builder() -> tuple[ir.IRBuilder, tuple[ir.Argument, ...]]:
    """Return a builder over ``(vector, index, value)``, the vector as Numba's array."""
    word = ir.IntType(64)
    byte = ir.IntType(8).as_pointer()
    # ``ArrayModel``'s fields: meminfo, parent, nitems, itemsize, data, shape, strides.
    vector = ir.LiteralStructType(
        [byte, byte, word, word, word.as_pointer(), word, word],
    )
    return ir_builder(vector, word, word)


def test_the_pool_builds_world_k_from_seed_k_with_the_shipped_level_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pool hands ``build_pool_numba`` its rows, their bytes and the shipped configs.

    The worlds are ``build_pool_numba``'s, which ``world_gen_test`` holds to seed-by-seed
    generation; compiling it here would cost seconds on a cold cache.
    """
    monkeypatch.setattr(
        "priml.baselines.craftax.env.build_pool_numba",
        _stamp_seeds,
    )
    pool = _pool(3).make().pool()
    assert pool.dtype == STATE_DTYPE
    assert pool.shape == (3,)
    assert pool.view(np.uint8).reshape(3, -1)[:, 0].tolist() == [0, 1, 2]


def _stamp_seeds(
    pool: np.ndarray,
    pool_bytes: np.ndarray,
    smooth_configs: np.ndarray,
    dungeon_configs: np.ndarray,
    first_seed: int,
) -> None:
    """Stand in for ``build_pool_numba``: write each row's seed into its first byte."""
    assert smooth_configs is SMOOTH_LEVEL_CONFIGS
    assert dungeon_configs is DUNGEON_LEVEL_CONFIGS
    assert pool_bytes.shape == (len(pool), STATE_DTYPE.itemsize)
    pool_bytes[:, 0] = first_seed + np.arange(len(pool))


@pytest.mark.compute_large_fixture
def test_the_state_dict_restores_every_array_a_step_reads() -> None:
    """A loaded state dict puts the env back where it was saved, bit for bit."""
    env = _env(num_envs=4, num_buffers=2, num_worlds=3)
    try:
        env.reset()
        streams = np.uint32(0x9E37_79B9) ^ np.arange(1, 5, dtype=np.uint32)
        _play(env, streams, steps=2)
        saved = {name: value.clone() for name, value in env.state_dict().items()}
        expected = env_digests(env)
        _play(env, streams, steps=3)
        assert env_digests(env) != expected
        env.load_state_dict(saved)
        assert env_digests(env) == expected
    finally:
        env.close()


def _play(env: CraftaxEnv, streams: np.ndarray, *, steps: int) -> None:
    """Step every buffer ``steps`` times on the oracle traces' action stream."""
    for _ in range(steps):
        masked_random_actions_numba(
            streams,
            env.action_mask.numpy(),
            env.actions.numpy(),
        )
        for buffer in range(env.num_buffers):
            env.step_buffer(buffer)


def test_defaults_are_exp000() -> None:
    cfg = CraftaxEnv.Config()
    assert (cfg.num_envs, cfg.num_buffers, cfg.threads_per_buffer) == (2048, 4, 2)
    assert cfg.restart == _pool(8192)
    assert cfg.rules.make() == Rules()
    assert cfg.stall_cap is None
    assert cfg.practice is None
    assert cfg.observation_size == OBS_SIZE


def test_the_previous_action_widens_the_packed_observation_by_one() -> None:
    cfg = CraftaxEnv.Config()
    cfg.rules.previous_action = True
    assert cfg.observation_size == ACTION_OBS_SIZE
    assert cfg.rules.make() == Rules(previous_action=True)
    cfg.rules.symbolic_observation = True
    with pytest.raises(ValueError, match="symbolic"):
        CraftaxEnv.check(cfg)


def test_practice_refuses_donors_beyond_the_last_buffer() -> None:
    cfg = CraftaxEnv.Config()
    cfg.num_envs = 8
    cfg.num_buffers = 2
    practice = cfg.practice = FrontierPractice.Config()
    practice.num_donors = 4
    CraftaxEnv.check(cfg)
    practice.num_donors = 5
    with pytest.raises(ValueError, match="last buffer"):
        CraftaxEnv.check(cfg)


def test_a_stall_cap_compiles_its_limit_and_permille_into_the_rules() -> None:
    cap = StallCap.Config().make()
    assert (cap.stall_limit, cap.uncapped_permille) == (10_000, 5)
    assert cap.rules(Rules()) == Rules(stall_limit=10_000, uncapped_permille=5)


@pytest.mark.parametrize(
    ("stall_limit", "uncapped_fraction", "permille"),
    [(1, 0.0, 0), (2**31 - 1, 1.0, 1000), (10, 0.25, 250)],
)
def test_a_stall_cap_takes_every_limit_and_share_its_step_can_play(
    stall_limit: int,
    uncapped_fraction: float,
    permille: int,
) -> None:
    config = StallCap.Config()
    config.stall_limit = stall_limit
    config.uncapped_fraction = uncapped_fraction
    cap = config.make()
    assert (cap.stall_limit, cap.uncapped_permille) == (stall_limit, permille)


@pytest.mark.parametrize(
    ("stall_limit", "uncapped_fraction", "match"),
    [
        (0, 0.005, r"^stall_limit must be positive"),
        # The step's clock is int32: 2**31 would wrap and end every episode at once.
        (2**31, 0.005, r"^stall_limit must fit the step's int32 clock"),
        (
            10,
            0.0005,
            r"^uncapped_fraction must be a whole number of thousandths in \[0, 1\]",
        ),
        (
            10,
            -0.001,
            r"^uncapped_fraction must be a whole number of thousandths in \[0, 1\]",
        ),
        (
            10,
            1.001,
            r"^uncapped_fraction must be a whole number of thousandths in \[0, 1\]",
        ),
        (
            10,
            math.nan,
            r"^uncapped_fraction must be a whole number of thousandths in \[0, 1\]",
        ),
        (
            10,
            math.inf,
            r"^uncapped_fraction must be a whole number of thousandths in \[0, 1\]",
        ),
        (
            10,
            -math.inf,
            r"^uncapped_fraction must be a whole number of thousandths in \[0, 1\]",
        ),
    ],
)
def test_a_stall_cap_refuses_what_its_draw_cannot_play(
    stall_limit: int,
    uncapped_fraction: float,
    match: str,
) -> None:
    config = StallCap.Config()
    config.stall_limit = stall_limit
    config.uncapped_fraction = uncapped_fraction
    with pytest.raises(ValueError, match=match):
        config.make()


def _options_env(threads: int = 1, *, fresh_worlds: bool = False) -> CraftaxEnv:
    """Return 8 environments in 2 buffers with every training option on, reset."""
    cfg = CraftaxEnv.Config()
    cfg.num_envs = 8
    cfg.num_buffers = 2
    cfg.threads_per_buffer = threads
    cfg.restart = FreshWorlds.Config() if fresh_worlds else _pool(4)
    cfg.rules.previous_action = True
    cap = cfg.stall_cap = StallCap.Config()
    cap.stall_limit = 20
    cap.uncapped_fraction = 0.25
    # One point per level, so a random player's first achievements save.
    practice = cfg.practice = FrontierPractice.Config()
    practice.num_donors = 3
    practice.num_levels = 4
    practice.level_width = 1.0
    practice.entries_per_level = 2
    practice.entries_per_world = 1
    env = cfg.make()
    env.reset()
    return env


def _practise(env: CraftaxEnv, streams: np.ndarray, *, rollouts: int) -> list[str]:
    """Play ``rollouts`` of 8 steps, preparing before each; digest every array after it."""
    lines: list[str] = []
    for _ in range(rollouts):
        env.prepare_rollout()
        _play(env, streams, steps=8)
        lines += [f"{name} {digest(value)}" for name, value in env.state_dict().items()]
    return lines


@pytest.mark.compute_large_fixture
def test_without_practice_its_plumbing_is_inert() -> None:
    env = _env(num_envs=4, num_buffers=2, num_worlds=3)
    try:
        assert env.carry_slots == 0
        assert env.save_rows == slice(0, 0)
        assert env.practice is None
        for slots in (env.save_slots, env.restore_slots):
            assert slots.dtype == torch.int32
            assert slots.tolist() == [-1] * 4
            assert slots.is_pinned() == torch.cuda.is_available()
        env.reset()
        before = env_digests(env)
        env.prepare_rollout()
        assert env_digests(env) == before
        assert env.practice_metrics() == {}
        assert env.stats.dtype == STATS_DTYPE
        assert list(env.state_dict()) == [
            "states",
            "rngs",
            "stats",
            "observations",
            "action_mask",
            "rewards",
            "terminals",
            "actions",
        ]
    finally:
        env.close()


@pytest.mark.compute_large_fixture
def test_the_escape_streams_are_seeded_once_and_draw_once_per_reset() -> None:
    env = _options_env()
    try:
        assert env.stats.dtype == TRAINING_STATS_DTYPE
        # The rows that can save: the three donors, the last buffer's last rows.
        assert env.save_rows == slice(5, 8)
        assert env.carry_slots == 8
        drawn = np.arange(8, dtype=np.uint32) ^ np.uint32(0x9E37_79B9)
        for _ in range(2):
            for i in range(8):
                rand_r_numba(drawn[i : i + 1])
            assert env.stats["escape"][:, 0].tolist() == drawn.tolist()
            env.reset()
    finally:
        env.close()


@pytest.mark.compute_large_fixture
@pytest.mark.parametrize("fresh_worlds", [False, True])
def test_practice_plays_the_same_bits_at_every_thread_count(fresh_worlds: bool) -> None:
    """Donors save in row order after each step's shares join, whoever stepped them.

    In fresh worlds too (exp109), where every reset generates the world that
    donors save from.
    """
    runs: list[list[str]] = []
    for threads in (1, 3):
        env = _options_env(threads, fresh_worlds=fresh_worlds)
        try:
            streams = np.uint32(0x9E37_79B9) ^ np.arange(1, 9, dtype=np.uint32)
            runs.append(_practise(env, streams, rollouts=6))
            practice = env.practice
            assert practice is not None
            assert practice.archive.sizes.sum() > 0
            assert practice.controller["restores"] > 0
        finally:
            env.close()
    assert runs[0] == runs[1]


@pytest.mark.compute_large_fixture
def test_practice_resumes_bit_identically_from_the_state_dict() -> None:
    env = _options_env()
    try:
        streams = np.uint32(0x9E37_79B9) ^ np.arange(1, 9, dtype=np.uint32)
        _practise(env, streams, rollouts=3)
        saved = {name: value.clone() for name, value in env.state_dict().items()}
        resumed_streams = streams.copy()
        expected = _practise(env, streams, rollouts=3)
        env.load_state_dict(saved)
        assert _practise(env, resumed_streams, rollouts=3) == expected
    finally:
        env.close()


@pytest.mark.compute_large_fixture
@pytest.mark.skipif(
    sys.platform == "darwin",
    reason=(
        "libm differs from glibc's in the last bit; golden is glibc x86-64. "
        "At 64 environments macOS's play departed from it at steps 77-127"
    ),
)
@pytest.mark.skipif(
    sys.platform == "linux" and platform.machine() != "x86_64",
    reason="golden is glibc x86-64; glibc 2.35 and 2.39 play it alike, this CPU is unmeasured",
)
@pytest.mark.parametrize(
    ("name", "rules", "fresh"),
    [
        ("exp000", {}, False),
        (
            "exp002",
            {
                "original_reward": True,
                "end_on_boss_defeat": True,
                "collapse_sleep": False,
                "action_mask": False,
                "symbolic_observation": True,
            },
            True,
        ),
        ("original_reward", {"original_reward": True}, False),
        ("tick_sleep", {"collapse_sleep": False}, False),
        ("no_action_mask", {"action_mask": False}, False),
        ("fresh_worlds", {}, True),
        ("symbolic", {"symbolic_observation": True}, False),
    ],
)
def test_the_environments_play_their_golden(
    name: str,
    rules: dict[str, bool],
    fresh: bool,
) -> None:
    """exp000's and exp002's environments, and each original option alone, frozen.

    Each plays from seed 0 -- 2,048 environments in 2 buffers of 2 threads,
    2,048 pool worlds unless fresh -- for 256 steps: past the first episode
    ends (from step ~30) and their resets, and into the night, whose light the
    spawns and the view read. Fewer environments miss defects that the 2,048
    on 8,192 worlds caught (measured): mining coal yields 2, a strength
    level-up adds 2, a hostile kill feeds, a move off the map wraps. Goldens
    red, of 7, with each planted:

        environments   coal   strength   hostile kill   edge wrap
        64             0      0          0              0
        512            4      4          1              0
        1,024          4      4          2              0
        2,048          5      1          5              3

    The actions are the oracle traces' xorshift stream over the last step's
    mask. The golden holds the pool, every array after the reset, a digest of
    every array after each step, and every array at the end, all as digests,
    so its size does not grow with the environments. exp002's options are ``docs/differences.md``'s D1, D2, D4, D5,
    D6 and D8; D2 alone is left out, since no random player reaches the
    necromancer, so its golden would be exp000's.
    """
    config = CraftaxEnv.Config()
    config.num_envs = 2_048
    config.num_buffers = 2
    for field_name, value in rules.items():
        setattr(config.rules, field_name, value)
    config.restart = FreshWorlds.Config() if fresh else _pool(2_048)
    env = config.make()
    try:
        lines = [f"pool {digest(env.pool.view(np.uint8))}"]
        env.reset()
        lines += [f"reset {entry}" for entry in env_digests(env)]
        streams = np.uint32(0x9E37_79B9) ^ np.arange(
            1,
            env.num_envs + 1,
            dtype=np.uint32,
        )
        for step in range(1, 257):
            masked_random_actions_numba(
                streams,
                env.action_mask.numpy(),
                env.actions.numpy(),
            )
            for buffer in range(env.num_buffers):
                env.step_buffer(buffer)
            lines.append(f"step {step:04d} {_step_digest(env)}")
        lines += [f"final {entry}" for entry in env_digests(env)]
    finally:
        env.close()
    assert_golden(test_file=__file__, name=f"env_{name}", lines=lines)


def _step_digest(env: CraftaxEnv) -> str:
    """Return 16 hex digits of the sha256 of the worlds, streams, stats and buffers."""
    hasher = hashlib.sha256()
    for array in (
        env.states.view(np.uint8),
        env.rngs,
        env.stats.view(np.uint8),
        env.observations.numpy(),
        env.action_mask.numpy(),
        env.rewards.numpy(),
        env.terminals.numpy(),
    ):
        hasher.update(np.ascontiguousarray(array))
    return hasher.hexdigest()[:16]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
