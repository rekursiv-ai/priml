"""Unit tests for the actor at tiny sizes: the rollout engine, then the sampler.

A fake env supplies random transitions through the buffer contract and the
torch sampler draws on the same Philox streams the kernel uses, so the
storage layout, the device-side step index, the carry's reset
and snapshot, the saved state and the per-buffer stream discipline are all
pinned without a GPU. The stream and row-bound contracts of the captured
graphs are checked on a GPU, and a golden freezes exp000's first two
rollouts there.

The sampler's Philox reference is checked against Random123's known answers
and the torch sampler against that reference on the CPU, and the kernel
against both on a GPU, which it alone runs on. The goldens freeze the
actions, log-probabilities and draw counts for fixed logits and exp000's
seed: the torch sampler on any CPU, the kernel on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Final,
    cast,
    override,
)

import copy
import ctypes
import functools

from configgle import Fig
from torch import Tensor, nn

import numpy as np
import pytest
import torch

from priml.baselines.craftax.game.state import OBS_SIZE
from priml.baselines.craftax.lib.arrays import ints
from priml.baselines.craftax.lib.compat import ExactPhiloxSampler
from priml.baselines.craftax.model import MinGRUPolicy
from priml.baselines.craftax.policies.actor_critic import ActorCritic
from priml.baselines.craftax.rollout import (
    PhiloxSampler,
    Rollout,
    RolloutStorage,
    Sampled,
    Sampler,
    StepGraph,
    TorchPhiloxSampler,
    _philox,
    philox_uniform,
    philox_uniforms,
)
from priml.baselines.craftax.testing import (
    FakeEnv,
    assert_golden,
    digest,
    env_digests,
    fill_portable,
    gpu_key,
    host_agnostic_pipeline,
    portable_uniform,
    require_golden,
    smoke_feature,
    tiny_exp000_step,
    tiny_policy,
    tiny_train_step,
)
from priml.baselines.craftax.train_step import (
    CraftaxTrainStep,
    LearnerRollout,
)
from priml.baselines.craftax.world_model.codec import decode
from priml.baselines.craftax.world_model.feature import (
    DonorHistory,
    FeatureEngine,
    Refill,
    Sliding,
    WorldModelFeature,
)
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.scripts.feature_gates import (
    Span,
    reference_features,
)
from priml.baselines.craftax.world_model.testing import random_segment
from priml.loss.policy_gradient import TorchPPO
from priml.model.linear import Linear
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Mapping,
    )

    from priml.baselines.craftax.rollout import FeatureStep
    from priml.baselines.craftax.world_model.feature import (
        RecentDecisions,
    )
    from priml.baselines.craftax.world_model.model import WorldModel


def _rollout(
    *,
    horizon: int = 3,
    seed: int = 0,
    device: str = "cpu",
    state_dtype: torch.dtype = torch.bfloat16,
    output_dtype: torch.dtype = torch.bfloat16,
    dtype: torch.dtype = torch.float32,
) -> tuple[Rollout, FakeEnv, MinGRUPolicy]:
    config = tiny_policy()
    config.state_dtype = state_dtype
    config.output_dtype = output_dtype
    torch.manual_seed(seed)
    policy = config.make().to(device)
    env = FakeEnv.make(num_envs=8, num_buffers=2, seed=seed)
    rollout_config = Rollout.Config()
    rollout_config.horizon = horizon
    rollout_config.dtype = dtype
    sampler = PhiloxSampler if device == "cuda" else TorchPhiloxSampler
    rollout = Rollout(
        rollout_config,
        policy=policy,
        sampler=sampler.Config().make(),
        env=env,
        device=torch.device(device),
    )
    return rollout, env, policy


def test_storage_has_time_major_shapes_and_dtypes() -> None:
    rollout, env, policy = _rollout(horizon=5)
    storage = rollout.slots[0]
    assert storage.observations.shape == (5, 8, 843)
    assert storage.observations.dtype == torch.bfloat16
    assert storage.actions.shape == (5, 8)
    assert storage.actions.dtype == torch.float32
    assert storage.action_mask.shape == (5, 8, 43)
    assert storage.initial_states.shape == (2, 8, 8)
    assert storage.initial_states.dtype == policy.state_dtype
    # What the learning rule reads is in the rollout's dtype, fp32 by default;
    # the observations, masks and terminals in the policy's.
    assert storage.logprobs.dtype == storage.values.dtype == torch.float32
    assert storage.rewards.dtype == torch.float32
    assert storage.terminals.dtype == storage.action_mask.dtype == policy.dtype
    assert len(rollout.slots) == 2
    assert len(rollout.graphs) == 2
    assert len(rollout.graphs[0]) == env.num_buffers
    # One worker per buffer: every buffer steps on its own thread.
    assert rollout.workers._max_workers == env.num_buffers
    rollout.close()


def test_a_step_stores_row_t_and_advances_the_index() -> None:
    rollout, env, _ = _rollout(horizon=3)
    graph = rollout.graphs[0][0]
    before = env.observations[env.buffer_slice(0)].clone()
    graph.replay()
    assert int(graph.step) == 1
    storage = rollout.slots[0]
    assert torch.equal(storage.observations[0, :4], before.bfloat16())
    assert torch.equal(storage.observations[1, :4], torch.zeros_like(before).bfloat16())
    # The actions reached the env's pinned row, as fp32 ids.
    assert torch.equal(env.actions[:4, 0], storage.actions[0, :4])
    assert (storage.actions[0, :4] >= 0).all()
    assert torch.equal(graph.draws, torch.ones(4, dtype=torch.int64))
    rollout.close()


def test_a_terminal_resets_the_carry_and_step_0_snapshots_it() -> None:
    rollout, env, policy = _rollout(horizon=3)
    graph = rollout.graphs[0][1]
    rows = env.buffer_slice(1)
    graph.state.copy_(torch.randn_like(graph.state))
    env.terminals[rows] = torch.tensor([1.0, 0.0, 0.0, 1.0])
    carry = graph.state.clone()
    graph.replay()
    storage = rollout.slots[0]
    snapshot = storage.initial_states[:, rows]
    # Rows that ended an episode start from zero; the rest from their carry.
    assert torch.equal(snapshot[:, 0], torch.zeros_like(snapshot[:, 0]))
    assert torch.equal(snapshot[:, 3], torch.zeros_like(snapshot[:, 3]))
    assert torch.equal(snapshot[:, 1], carry[:, 1])
    assert torch.equal(snapshot[:, 2], carry[:, 2])
    assert torch.equal(storage.terminals[0, rows], env.terminals[rows].bfloat16())
    # After the step the carry moved on, and a later step leaves the snapshot alone.
    assert not torch.equal(graph.state, snapshot)
    env.terminals[rows] = 0.0
    graph.replay()
    assert torch.equal(storage.initial_states[:, rows], snapshot)
    assert int(graph.step) == 2
    del policy
    rollout.close()


def test_collect_fills_every_row_and_steps_the_env_each_time() -> None:
    rollout, env, _ = _rollout(horizon=3)
    storage = rollout.collect(0)
    assert storage is rollout.slots[0]
    assert all(len(rows) == 3 for rows in env.stepped)
    for buffer in range(env.num_buffers):
        rows = env.buffer_slice(buffer)
        for time in range(3):
            assert torch.equal(
                env.stepped[buffer][time][:, 0],
                storage.actions[time, rows],
            )
    assert not torch.equal(storage.observations[0], storage.observations[1])
    # The second slot shares the buffers' carry and streams, and starts at t=0.
    again = rollout.collect(1)
    assert again is rollout.slots[1]
    assert rollout.graphs[1][0].state is rollout.graphs[0][0].state
    assert all(int(graph.step) == 3 for graph in rollout.graphs[1])
    # The learner reads a slot through its time-major loader.
    learner = LearnerRollout.from_time_major(
        again.observations,
        again.actions,
        again.logprobs,
        again.rewards,
        again.terminals,
        again.values,
        again.action_mask,
        again.initial_states,
        again.branch_starts,
        reward_scale=1.0,
        reward_clip=1.0,
    )
    assert learner.observations.shape == (8, 3, 843)
    rollout.close()


def test_fp32_storage_keeps_what_bf16_storage_rounds() -> None:
    """fp32 and bf16 storage of one rollout: the bf16 rows are the fp32 ones rounded.

    The sampler's log-probabilities and the env's rewards are fp32, so only the
    fp32 storage keeps them; a reward of 0.1 stays 0.1 there.
    """
    exact, exact_env, _ = _rollout(horizon=3, seed=2)
    rounded, rounded_env, _ = _rollout(horizon=3, seed=2, dtype=torch.bfloat16)
    exact_env.rewards.fill_(0.1)
    rounded_env.rewards.fill_(0.1)
    ours, theirs = exact.collect(0), rounded.collect(0)
    assert torch.equal(ours.actions, theirs.actions)
    for name in ("logprobs", "values", "rewards"):
        value = cast("torch.Tensor", getattr(ours, name))
        assert value.dtype == torch.float32
        assert torch.equal(
            value.bfloat16(),
            cast("torch.Tensor", getattr(theirs, name)),
        ), name
    assert torch.equal(ours.rewards[0], torch.full((8,), 0.1))
    assert not torch.equal(ours.logprobs, theirs.logprobs.float())
    assert not torch.equal(ours.rewards, theirs.rewards.float())
    exact.close()
    rounded.close()


def test_an_fp32_carry_advances_unrounded_through_the_rollout() -> None:
    """The step graph keeps the policy's fp32 carry, and snapshots it unrounded."""
    rollout, env, policy = _rollout(horizon=3, seed=3, state_dtype=torch.float32)
    storage = rollout.collect(0)
    state = rollout.graphs[0][0].state
    assert state.dtype == storage.initial_states.dtype == policy.state_dtype
    assert not torch.equal(state, state.bfloat16().float())
    carry = state.clone()
    # No episode ends, so the next rollout's snapshot is buffer 0's carry as it was.
    env.terminals.zero_()
    rollout.collect(1)
    assert torch.equal(rollout.slots[1].initial_states[:, :4], carry)
    rollout.close()


def test_replay_without_a_stream_runs_eagerly_each_time() -> None:
    rollout, env, _ = _rollout(horizon=2)
    graph = rollout.graphs[0][0]
    assert graph.stream is None
    graph.replay()
    graph.replay()
    assert int(graph.step) == 2
    assert not graph.graphs
    del env
    rollout.close()


def test_storage_allocate_matches_the_env_and_policy_geometry() -> None:
    config = tiny_policy()
    policy = config.make()
    env = FakeEnv.make(num_envs=6, num_buffers=3, seed=1)
    storage = RolloutStorage.allocate(
        horizon=4,
        env=env,
        policy=policy,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert storage.rewards.shape == (4, 6)
    assert storage.rewards.dtype == torch.float64
    assert storage.initial_states.shape == (2, 6, 8)
    graph = StepGraph(
        policy=policy,
        sampler=TorchPhiloxSampler.Config().make(),
        env=env,
        storage=storage,
        buffer=2,
        stream=None,
    )
    assert graph.rows == slice(4, 6)
    assert graph.state.shape == (2, 2, 8)


def test_a_rollout_resumes_from_its_state_as_if_never_stopped() -> None:
    """Carries, draw counts and a slot round-trip; the next rollout is bit-equal."""
    straight, straight_env, _ = _rollout(horizon=3, seed=4)
    straight.collect(0)
    state = {name: value.clone() for name, value in straight.state_dict(slot=0).items()}
    env_state = {
        name: cast("torch.Tensor", getattr(straight_env, name)).clone()
        for name in ("observations", "action_mask", "rewards", "terminals", "actions")
    }
    generators = [generator.get_state() for generator in straight_env.generators]
    expected = straight.collect(1)

    resumed, resumed_env, _ = _rollout(horizon=3, seed=4)
    resumed.load_state_dict(state, slot=0)
    for name, value in env_state.items():
        cast("torch.Tensor", getattr(resumed_env, name)).copy_(value)
    for generator, saved in zip(resumed_env.generators, generators, strict=True):
        generator.set_state(saved)
    assert torch.equal(resumed.slots[0].observations, straight.slots[0].observations)
    actual = resumed.collect(1)
    for name in ("observations", "actions", "logprobs", "initial_states"):
        assert torch.equal(
            cast("torch.Tensor", getattr(actual, name)),
            cast("torch.Tensor", getattr(expected, name)),
        ), name
    for graph, twin in zip(resumed.graphs[0], straight.graphs[0], strict=True):
        assert torch.equal(graph.state, twin.state)
        assert torch.equal(graph.draws, twin.draws)
    straight.close()
    resumed.close()


def test_reset_restarts_every_carry_and_stream() -> None:
    rollout, _, _ = _rollout(horizon=2)
    rollout.collect(0)
    assert all(int(graph.draws.sum()) > 0 for graph in rollout.graphs[0])
    rollout.reset()
    for graph in rollout.graphs[0]:
        assert not graph.state.any()
        assert not graph.draws.any()
    rollout.close()


@pytest.mark.parametrize(("num_slots", "horizon"), [(0, 3), (2, 0)])
def test_a_rollout_refuses_an_empty_geometry(num_slots: int, horizon: int) -> None:
    config = Rollout.Config()
    config.num_slots = num_slots
    config.horizon = horizon
    with pytest.raises(ValueError, match=r"^num_slots and horizon must be positive$"):
        Rollout(
            config,
            policy=tiny_policy().make(),
            sampler=TorchPhiloxSampler.Config().make(),
            env=FakeEnv.make(num_envs=4, num_buffers=2, seed=0),
            device=torch.device("cpu"),
        )


def test_a_one_step_rollout_is_accepted() -> None:
    config = Rollout.Config()
    config.horizon = 1
    Rollout(
        config,
        policy=tiny_policy().make(),
        sampler=TorchPhiloxSampler.Config().make(),
        env=FakeEnv.make(num_envs=4, num_buffers=2, seed=0),
        device=torch.device("cpu"),
    ).close()


def _bootstrap_rollout(*, device: str = "cpu") -> tuple[Rollout, FakeEnv, ActorCritic]:
    """Return one slot of horizon 3 with its bootstrap row, for a tiny fp32 MLP."""
    config = ActorCritic.Config()
    config.observation_size = OBS_SIZE
    config.channels_hidden = 8
    config.num_layers = 1
    torch.manual_seed(0)
    policy = config.make().to(device)
    env = FakeEnv.make(num_envs=8, num_buffers=2, seed=0)
    rollout_config = Rollout.Config()
    rollout_config.num_slots = 1
    rollout_config.horizon = 3
    rollout_config.bootstrap = True
    sampler = PhiloxSampler if device == "cuda" else TorchPhiloxSampler
    rollout = Rollout(
        rollout_config,
        policy=policy,
        sampler=sampler.Config().make(),
        env=env,
        device=torch.device(device),
    )
    return rollout, env, policy


def test_a_bootstrap_row_holds_what_the_last_step_led_to() -> None:
    """Row ``horizon`` is the env where the rollout left it, and the policy's value there.

    The env steps ``horizon`` times, not one more, and the next rollout starts
    from that same observation.
    """
    rollout, env, policy = _bootstrap_rollout()
    storage = rollout.collect(0)
    assert storage.observations.shape == (4, 8, OBS_SIZE)
    assert all(len(rows) == 3 for rows in env.stepped)
    # The fp32 policy's storage keeps the env's fp32 values unrounded.
    assert torch.equal(storage.observations[3], env.observations)
    assert torch.equal(storage.rewards[3], env.rewards)
    assert torch.equal(storage.terminals[3], env.terminals)
    # Per buffer, as the step scores them: a GEMM over other rows rounds otherwise.
    for buffer in range(env.num_buffers):
        rows = env.buffer_slice(buffer)
        with torch.no_grad():
            decoded, _ = policy.forward_fused(
                env.observations[rows],
                policy.initial_state(4),
                None,
            )
        assert torch.equal(storage.values[3, rows], decoded[:, -1])
    last = storage.observations[3].clone()
    rollout.collect(0)
    assert torch.equal(storage.observations[0], last)
    assert all(len(rows) == 6 for rows in env.stepped)
    rollout.close()


class _RecordingSampler(TorchPhiloxSampler):
    """The torch sampler, keeping each step's draws."""

    def __init__(self, config: TorchPhiloxSampler.Config) -> None:
        super().__init__(config)
        self.sampled: list[Sampled] = []

    @override
    def __call__(
        self,
        decoded: Tensor,
        action_mask: Tensor,
        draws: Tensor,
        *,
        buffer: int,
        dtype: torch.dtype | None = None,
    ) -> Sampled:
        sampled = super().__call__(
            decoded,
            action_mask,
            draws,
            buffer=buffer,
            dtype=dtype,
        )
        self.sampled.append(sampled)
        return sampled


def test_a_one_buffer_rollout_stores_its_steps_under_the_host_agnostic_pipeline() -> (
    None
):
    """Each row holds the step's draws, where one buffer's rows span the whole slot.

    Those rows' slice is ``alias``, which the harness upcast to a copy, so every
    store landed in the copy and the slot kept zeros.
    """
    torch.manual_seed(0)
    policy = tiny_policy().make()
    env = FakeEnv.make(num_envs=4, num_buffers=1, seed=0)
    config = Rollout.Config()
    config.num_slots = 1
    config.horizon = 3
    sampler = _RecordingSampler(TorchPhiloxSampler.Config())
    rollout = Rollout(
        config,
        policy=policy,
        sampler=sampler,
        env=env,
        device=torch.device("cpu"),
    )
    try:
        with host_agnostic_pipeline():
            storage = rollout.collect(0)
    finally:
        rollout.close()
    assert len(sampler.sampled) == 3
    for step, sampled in enumerate(sampler.sampled):
        assert torch.equal(storage.logprobs[step], sampled.logprobs), step
        assert torch.equal(storage.values[step], sampled.values), step
        assert torch.equal(storage.actions[step], sampled.actions), step
    assert bool((storage.logprobs < 0).all())


class _EndingEnv(FakeEnv):
    """A fake env whose every step also ends the episode of each buffer's first row."""

    @override
    def step_buffer(self, buffer: int) -> None:
        super().step_buffer(buffer)
        first = range(*self.buffer_slice(buffer).indices(self.num_envs))[0]
        self.terminals[first] = 1.0


def _carried_rollout(
    *,
    bootstrap: bool,
    device: str = "cpu",
) -> tuple[Rollout, _EndingEnv, MinGRUPolicy]:
    """Return one slot of horizon 3 for the tiny MinGRU, which keeps a carry."""
    torch.manual_seed(0)
    policy = tiny_policy().make().to(device)
    env = _EndingEnv.make(num_envs=8, num_buffers=2, seed=0)
    assert isinstance(env, _EndingEnv)
    config = Rollout.Config()
    config.num_slots = 1
    config.horizon = 3
    config.bootstrap = bootstrap
    sampler = PhiloxSampler if device == "cuda" else TorchPhiloxSampler
    rollout = Rollout(
        config,
        policy=policy,
        sampler=sampler.Config().make(),
        env=env,
        device=torch.device(device),
    )
    return rollout, env, policy


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.gpu_triton)],
)
def test_a_bootstrap_row_puts_back_the_carry_its_forward_advanced(device: str) -> None:
    """Each rollout then steps from the carry the last one left, as without the row.

    The twin without the row draws the same transitions and actions, so its
    carry and its next rollout's are what the bootstrap row must leave.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    carried, _, _ = _carried_rollout(bootstrap=True, device=device)
    plain, _, _ = _carried_rollout(bootstrap=False, device=device)
    try:
        for _ in range(2):
            with_row, without_row = carried.collect(0), plain.collect(0)
            for graph, twin in zip(carried.graphs[0], plain.graphs[0], strict=True):
                assert torch.equal(graph.state, twin.state)
            assert torch.equal(with_row.initial_states, without_row.initial_states)
            assert torch.equal(with_row.values[:3], without_row.values)
    finally:
        carried.close()
        plain.close()


def test_the_bootstrap_value_reads_the_last_carry_reset_where_done() -> None:
    """Row ``horizon``'s value is a forward from the carry the last step left."""
    rollout, env, policy = _carried_rollout(bootstrap=True)
    try:
        storage = rollout.collect(0)
        ended = env.terminals != 0
        assert bool(ended.any())
        assert not bool(ended.all())
        # Per buffer, as the step scores them: a GEMM over other rows rounds otherwise.
        for graph in rollout.graphs[0]:
            with torch.no_grad():
                decoded, _ = policy.forward_fused(
                    env.observations[graph.rows],
                    graph.state,
                    env.terminals[graph.rows],
                )
            assert torch.equal(
                storage.values[3, graph.rows],
                decoded[:, -1].to(storage.values.dtype),
            )
    finally:
        rollout.close()


@pytest.mark.gpu_triton
def test_a_captured_rollout_stores_an_fp32_policys_rows_unrounded() -> None:
    """The ingest and the sampler round to the storage's dtype, fp32 here, not bf16."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    rollout, env, _ = _bootstrap_rollout(device="cuda")
    storage = rollout.collect(0)
    torch.cuda.synchronize()
    assert storage.rewards.dtype == torch.float32
    assert torch.equal(storage.rewards[3].cpu(), env.rewards)
    assert torch.equal(storage.observations[3].cpu(), env.observations)
    assert not torch.equal(storage.rewards, storage.rewards.bfloat16().float())
    assert not torch.equal(storage.logprobs, storage.logprobs.bfloat16().float())
    assert all(len(rows) == 3 for rows in env.stepped)
    rollout.close()


@pytest.mark.gpu_triton
def test_a_one_step_capture_writes_no_row_past_the_horizon() -> None:
    """The capture's warmup and the stores stay inside ``horizon`` rows (ROLL-2)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    rollout, env, _ = _rollout(horizon=1, device="cuda")
    # A canary row just past the slot: storage carved from a larger buffer.
    storage = rollout.slots[0]
    canary = torch.zeros(
        2,
        *storage.observations.shape[1:],
        dtype=storage.observations.dtype,
        device="cuda",
    )
    carved = RolloutStorage(
        observations=canary[:1],
        actions=storage.actions,
        logprobs=storage.logprobs,
        values=storage.values,
        rewards=storage.rewards,
        terminals=storage.terminals,
        action_mask=storage.action_mask,
        initial_states=storage.initial_states,
        branch_starts=storage.branch_starts,
    )
    graph = StepGraph(
        policy=rollout.graphs[0][0].policy,
        sampler=PhiloxSampler.Config().make(),
        env=env,
        storage=carved,
        buffer=0,
        stream=rollout.streams[0],
    )
    graph.replay()
    torch.cuda.synchronize()
    assert not canary[1].any()
    assert canary[0].any()
    rollout.close()


@pytest.mark.gpu_triton
def test_replay_runs_on_the_graphs_stream_whatever_the_callers_is() -> None:
    """A replay issued while the caller's stream is busy does not wait for it (ROLL-1)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    rollout, _, _ = _rollout(horizon=4, device="cuda")
    graph = rollout.graphs[0][0]
    graph.replay()
    stream = graph.stream
    assert stream is not None
    stream.synchronize()
    graph.step.zero_()
    torch.cuda.synchronize()
    # Park the caller's stream; a replay issued there would wait behind it.
    torch.cuda._sleep(2_000_000_000)
    graph.replay()
    stream.synchronize()
    # Read on the graph's stream: the parked one would hold the copy back.
    with torch.cuda.stream(stream):
        assert graph.step.item() == 1
    torch.cuda.synchronize()
    rollout.close()


def test_the_tiny_pipelines_first_two_rollouts_match_their_golden() -> None:
    """The tiny pipeline's boot rollout and the next, frozen on every host.

    exp000's recipe in its torch forms (the scan, the Philox sampler at seed
    73, a bf16 carry and bf16 storage), from portable seed-73 weights, plays 4
    real environments in 2 buffers for two horizons of 2 into the two slots,
    as training's boot rollout and first prefetch do before the learner moves
    a weight. The golden holds every slot tensor with the carries and draw
    counts after each rollout, then the environments.
    """
    config = tiny_train_step()
    config.rollout.horizon = 2
    with host_agnostic_pipeline():
        lines = _rollout_entries(config, device=torch.device("cpu"))
    assert_golden(test_file=__file__, name="rollout_tiny", lines=lines)


@pytest.mark.gpu_triton
def test_exp000s_first_two_rollouts_match_their_golden() -> None:
    """exp000's boot rollout and the next at test size, from portable weights, frozen.

    exp000's kernels on the tiny pipeline's environments: its policy (the
    exact scan, a bf16 carry), its exact Philox sampler (seed 73) and its bf16
    storage, in captured step graphs that the env's ``nogil`` loop launches,
    for two horizons of 8.
    """
    host = gpu_key()
    require_golden(test_file=__file__, name="rollout_exp000_tiny", host=host)
    lines = _rollout_entries(tiny_exp000_step(), device=torch.device("cuda"))
    assert_golden(
        test_file=__file__,
        name="rollout_exp000_tiny",
        lines=lines,
        host=host,
    )


def _rollout_entries(
    config: CraftaxTrainStep.Config,
    *,
    device: torch.device,
) -> list[str]:
    """Collect both slots from portable seed-73 weights; digest each, then the env."""
    policy = config.model.make()
    assert isinstance(policy, MinGRUPolicy)
    fill_portable(policy, seed=73)
    policy.to(device)
    env = config.env.make()
    env.reset()
    rollout = Rollout(
        config.rollout,
        policy=policy,
        sampler=config.sampler.make(),
        env=env,
        device=device,
    )
    lines: list[str] = []
    try:
        for slot in range(2):
            rollout.collect(slot)
            if device.type == "cuda":
                # The slots and carries are written on the buffers' streams.
                torch.cuda.synchronize()
            lines += [
                f"slot {slot} {name} {digest(value)}"
                for name, value in rollout.state_dict(slot=slot).items()
            ]
        return lines + env_digests(env)
    finally:
        rollout.close()
        env.close()


class _PractisingEnv:
    """A fake env with practice's slots: rows save into the entries a test scripts.

    It records, as each scripted save lands, the carry the row's next forward
    will read -- the rollout's carry after this step's forward, zeroed where
    the step ended an episode -- which the save's entry must then hold; and
    with ``engines`` set, the feature history the row's next step extends,
    empty where the step ended an episode, with the action it just took.
    """

    def __init__(
        self,
        env: FakeEnv | _FramedEnv,
        *,
        carry_slots: int,
        save_rows: slice,
        saves: dict[tuple[int, int], dict[int, int]],
    ) -> None:
        self.env = env
        self.save_rows = save_rows
        self.observations = env.observations
        self.action_mask = env.action_mask
        self.rewards = env.rewards
        self.terminals = env.terminals
        self.actions = env.actions
        self.num_envs = env.num_envs
        self.num_buffers = env.num_buffers
        pinned = torch.cuda.is_available()
        self.save_slots, self.restore_slots = (
            torch.full((env.num_envs,), -1, dtype=torch.int32, pin_memory=pinned)
            for _ in range(2)
        )
        self.carry_slots = carry_slots
        self.saves = saves
        self.steps = [0] * env.num_buffers
        self.carries: list[Tensor] = []
        self.saved: dict[int, Tensor] = {}
        self.engines: list[FeatureEngine] = []
        self.histories: dict[int, dict[str, Tensor]] = {}

    def buffer_slice(self, buffer: int) -> slice:
        return self.env.buffer_slice(buffer)

    def step_buffer(self, buffer: int) -> None:
        rows = self.buffer_slice(buffer)
        start = rows.indices(self.num_envs)[0]
        self.env.step_buffer(buffer)
        self.save_slots[rows] = -1
        self.restore_slots[rows] = -1
        for row, slot in self.saves.get((buffer, self.steps[buffer]), {}).items():
            self.save_slots[row] = slot
            carry = self.carries[buffer][:, row - start].cpu()
            ended = bool(self.terminals[row])
            self.saved[slot] = torch.zeros_like(carry) if ended else carry.clone()
            if self.engines:
                self.histories[slot] = _history(
                    self.engines[buffer],
                    row - start,
                    previous_action=self.actions[row, 0],
                    ended=ended,
                )
        self.steps[buffer] += 1


# Its policy and sampler are :func:`_rollout`'s with an fp32 carry; 6 envs in 2 buffers,
# so no dimension ties the carry's width of 8.
def _practising(
    device: str,
    *,
    carry_slots: int = 5,
    num_slots: int = 2,
) -> tuple[Rollout, _PractisingEnv]:
    """Return a rollout whose donors save mid-rollout (row 5) and on its last step (row 4)."""
    env = _PractisingEnv(
        FakeEnv.make(num_envs=6, num_buffers=2, seed=0),
        carry_slots=carry_slots,
        save_rows=slice(4, 6),
        saves={(1, 0): {5: 2}, (1, 2): {4: 4}},
    )
    config = tiny_policy()
    config.state_dtype = torch.float32
    config.output_dtype = torch.bfloat16
    torch.manual_seed(0)
    policy = config.make().to(device)
    rollout_config = Rollout.Config()
    rollout_config.num_slots = num_slots
    rollout_config.horizon = 3
    sampler = PhiloxSampler if device == "cuda" else TorchPhiloxSampler
    rollout = Rollout(
        rollout_config,
        policy=policy,
        sampler=sampler.Config().make(),
        env=env,
        device=torch.device(device),
    )
    env.carries = [graph.state for graph in rollout.graphs[0]]
    return rollout, env


def _assert_restored_rows_start_from_their_donors_carries(
    device: str,
    num_slots: int,
) -> None:
    rollout, env = _practising(device, num_slots=num_slots)
    try:
        assert rollout.archive is not None
        assert rollout.archive.shape == (5, 2, 8)
        # Only the buffer holding the saving rows saves, and only those rows.
        first, second = rollout.graphs[0]
        assert first.saving is None
        assert second.saving is not None
        assert second.saving.rows == slice(1, 3)
        assert second.saving.carries.shape == (7, 2, 8)
        rollout.collect(0)
        if device == "cuda":
            torch.cuda.synchronize()
        # The mid-rollout save landed through a step, the last step's through the tail.
        assert set(env.saved) == {2, 4}
        for slot, carry in env.saved.items():
            assert torch.equal(rollout.archive[slot].cpu(), carry), slot
        assert not rollout.slots[0].branch_starts.any()
        # As the env's prepare does: rows 0 and 1 restored from entries 2 and 4.
        env.restore_slots[:2] = torch.tensor([2, 4], dtype=torch.int32)
        env.terminals[:2] = 0.0
        storage = rollout.collect(num_slots - 1)
        if device == "cuda":
            torch.cuda.synchronize()
        assert storage.branch_starts.tolist() == [1, 1, 0, 0, 0, 0]
        assert torch.equal(storage.initial_states[:, 0].cpu(), env.saved[2])
        assert torch.equal(storage.initial_states[:, 1].cpu(), env.saved[4])
        state = rollout.state_dict(slot=num_slots - 1)
        assert state["archive"] is rollout.archive
        assert state["branch_starts"] is storage.branch_starts
    finally:
        rollout.close()


@pytest.mark.parametrize("num_slots", [1, 2])
def test_a_restored_row_starts_from_its_donors_saved_carry(num_slots: int) -> None:
    _assert_restored_rows_start_from_their_donors_carries("cpu", num_slots)


def test_an_env_with_no_carry_slots_gets_no_carry_archive() -> None:
    """As ``CraftaxEnv`` without practice: the slots exist, and no carry is kept."""
    rollout, _ = _practising("cpu", carry_slots=0)
    try:
        assert rollout.archive is None
        assert all(graph.saving is None for graph in rollout.graphs[0])
        storage = rollout.collect(0)
        assert "archive" not in rollout.state_dict(slot=0)
        assert not storage.branch_starts.any()
    finally:
        rollout.close()


def test_without_practice_the_rollout_keeps_no_carries() -> None:
    rollout, _, _ = _rollout(horizon=2)
    try:
        assert rollout.archive is None
        storage = rollout.collect(0)
        assert not storage.branch_starts.any()
        assert storage.branch_starts.dtype == torch.uint8
        assert list(rollout.state_dict(slot=0)) == [
            "carry",
            "draws",
            "observations",
            "actions",
            "logprobs",
            "values",
            "rewards",
            "terminals",
            "action_mask",
            "initial_states",
        ]
    finally:
        rollout.close()


@pytest.mark.gpu_triton
def test_a_restored_row_starts_from_its_donors_saved_carry_on_the_gpu() -> None:
    """The captured step saves through its own ops; the tail and the gather are eager."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    _assert_restored_rows_start_from_their_donors_carries("cuda", 2)


@pytest.mark.gpu_triton
def test_practice_adds_its_upload_and_its_save_to_the_saving_buffers_graph_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A buffer without saving rows captures the plain step; the other adds two things.

    The save slots' upload, and the save's kernels. The ingest kernel is the
    same in all three.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    monkeypatch.setattr(
        torch.cuda,
        "CUDAGraph",
        functools.partial(torch.cuda.CUDAGraph, keep_graph=True),
    )
    plain, _, _ = _rollout(horizon=3, device="cuda", state_dtype=torch.float32)
    practising, _ = _practising("cuda")
    kinds: list[dict[int, int]] = []
    for graph in (plain.graphs[0][0], *practising.graphs[0]):
        graph.replay()
        assert list(graph.graphs) == [0]
        kinds.append(_graph_node_kinds(graph.graphs[0].raw_cuda_graph()))
    plain.close()
    practising.close()
    kernel, memcpy = 0, 1  # ``CUgraphNodeType``: CU_GRAPH_NODE_TYPE_KERNEL, _MEMCPY.
    assert kinds[1] == kinds[0]
    assert kinds[2][memcpy] == kinds[0][memcpy] + 1
    assert kinds[2][kernel] > kinds[0][kernel]
    assert set(kinds[2]) == set(kinds[0])


def _graph_node_kinds(graph: int) -> dict[int, int]:
    """Return how many nodes of each ``CUgraphNodeType`` a ``cudaGraph_t`` holds."""
    driver = ctypes.CDLL("libcuda.so.1")
    # Typed as their restype leaves them: a CUresult.
    get_nodes = cast("Callable[..., int]", driver.cuGraphGetNodes)
    node_type = cast("Callable[..., int]", driver.cuGraphNodeGetType)
    count = ctypes.c_size_t(0)
    status = get_nodes(ctypes.c_void_p(graph), None, ctypes.byref(count))
    assert status == 0, status
    nodes = (ctypes.c_void_p * count.value)()
    status = get_nodes(ctypes.c_void_p(graph), nodes, ctypes.byref(count))
    assert status == 0, status
    kinds: dict[int, int] = {}
    for node in cast("list[int | None]", list(nodes)):
        kind = ctypes.c_int(0)
        assert node_type(ctypes.c_void_p(node), ctypes.byref(kind)) == 0
        kinds[kind.value] = kinds.get(kind.value, 0) + 1
    return kinds


@pytest.mark.parametrize(
    ("counter", "key", "expected"),
    [
        # Random123's known-answer vectors for philox4x32-10.
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        (
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF),
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ],
)
def test_the_philox_reference_matches_the_known_answers(
    counter: tuple[int, int, int, int],
    key: tuple[int, int],
    expected: tuple[int, int, int, int],
) -> None:

    c0, c1, c2, c3 = (np.array(word, dtype=np.uint64) for word in counter)
    k0, k1 = (np.array(word, dtype=np.uint64) for word in key)
    words = _philox((c0, c1, c2, c3), (k0, k1))
    assert tuple(int(word) for word in words) == expected


def test_uniforms_are_in_the_open_unit_interval_and_stream_dependent() -> None:
    a = philox_uniforms(73, 0, 64)
    b = philox_uniforms(73, 1, 64)
    c = philox_uniforms(74, 0, 64)
    assert a.dtype == np.float32
    assert (a > 0).all()
    assert (a <= 1).all()
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # Draw n is word n mod 4 of block n // 4: the stream is a prefix-stable sequence.
    assert np.array_equal(philox_uniforms(73, 0, 16), a[:16])


def test_the_vectorized_stream_matches_the_scalar_reference() -> None:
    agents = np.array([0, 5, 5, 511], dtype=np.uint64)
    draws = np.array([0, 3, 9, 1], dtype=np.uint64)
    seed = 2**32 + 73
    expected = [
        philox_uniforms(seed, int(a), int(d) + 1)[-1]
        for a, d in zip(ints(agents), ints(draws), strict=True)
    ]
    assert np.array_equal(
        philox_uniform(seed, agents=agents, draws=draws),
        np.array(expected, dtype=np.float32),
    )


def test_the_seeds_high_word_is_part_of_the_key() -> None:
    assert not np.array_equal(
        philox_uniforms(73, 0, 8),
        philox_uniforms(2**32 + 73, 0, 8),
    )


def test_the_torch_sampler_draws_the_reference_stream() -> None:
    philox = TorchPhiloxSampler.Config().make()
    draws = philox.draws(2, device="cpu")
    decoded = torch.zeros(2, 44).bfloat16()
    mask = torch.ones(2, 43).bfloat16()
    first = philox(decoded, mask, draws, buffer=0)
    second = philox(decoded, mask, draws, buffer=0)
    assert torch.equal(draws, torch.tensor([2, 2]))
    # Uniform logits: the action is the draw's bin of 43.
    expected = [
        int(np.float32(philox_uniforms(73, agent, 2).item(step)) * 43)
        for step, agent in ((0, 0), (0, 1), (1, 0), (1, 1))
    ]
    assert [int(a) for a in (*first.actions, *second.actions)] == expected
    assert torch.equal(first.values, decoded[:, 43])


def test_the_torch_sampler_picks_legal_actions_even_where_the_last_is_masked() -> None:
    philox = TorchPhiloxSampler.Config().make()
    draws = philox.draws(64, device="cpu")
    generator = torch.Generator().manual_seed(1)
    decoded = torch.randn(64, 44, generator=generator).bfloat16()
    mask = (torch.rand(64, 43, generator=generator) > 0.5).bfloat16()
    mask[:, 0] = 1
    mask[:, 42] = 0
    for _ in range(4):
        sampled = philox(decoded, mask, draws, buffer=2)
        assert (mask[torch.arange(64), sampled.actions.long()] != 0).all()


@pytest.mark.parametrize("sampler", [TorchPhiloxSampler, PhiloxSampler])
@pytest.mark.parametrize("seed", [-1, 2**64])
def test_a_seed_outside_curands_64_bits_is_refused(
    sampler: type[TorchPhiloxSampler | PhiloxSampler],
    seed: int,
) -> None:
    config = sampler.Config()
    config.seed = seed
    with pytest.raises(ValueError, match="seed"):
        config.make()


@pytest.mark.parametrize("sampler", [TorchPhiloxSampler, PhiloxSampler])
def test_the_config_seeds_and_counts(
    sampler: type[TorchPhiloxSampler | PhiloxSampler],
) -> None:
    philox = sampler.Config().make()
    assert philox.seed == 73
    draws = philox.draws(5, device="cpu")
    assert draws.shape == (5,)
    assert draws.dtype == torch.int64
    assert int(draws.sum()) == 0


def test_the_kernel_refuses_a_cpu_batch_rather_than_changing_its_bits() -> None:
    philox = PhiloxSampler.Config().make()
    draws = philox.draws(2, device="cpu")
    with pytest.raises(ValueError, match="TorchPhiloxSampler"):
        philox(torch.zeros(2, 44).bfloat16(), torch.ones(2, 43), draws, buffer=0)
    assert not draws.any()


def _cuda_batch(agents: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    generator = torch.Generator(device="cuda").manual_seed(0)
    decoded = (
        torch.randn(agents, 44, device="cuda", generator=generator) * 2
    ).bfloat16()
    mask = (torch.rand(agents, 43, device="cuda", generator=generator) > 0.3).bfloat16()
    mask[:, 5] = 1
    mask[::7, 42] = 0
    return decoded, mask


@pytest.mark.gpu_triton
def test_the_kernel_draws_the_reference_streams_and_a_legal_action() -> None:
    decoded, mask = _cuda_batch(512)
    philox = PhiloxSampler.Config().make()
    draws = philox.draws(512, device="cuda")
    reference = TorchPPO.Config().make()
    for step in range(3):
        sampled = philox(decoded, mask, draws, buffer=1)
        assert torch.equal(draws, torch.full_like(draws, step + 1))
        assert torch.equal(sampled.values, decoded[:, 43])
        actions = sampled.actions.long().cpu()
        assert (mask.cpu()[torch.arange(512), actions] != 0).all()
        # The draw decides the action through the masked CDF: replay it on
        # the CPU with the reference stream and the reference probabilities.
        logps = reference.log_probs(decoded[None], sampled.actions[None], mask[None])
        cdf = logps.logps[0].float().exp().cumsum(-1).cpu()
        uniform = torch.tensor(
            [philox_uniforms(74, agent, step + 1)[step] for agent in range(512)],
        )
        expected = (uniform[:, None] < cdf).int().argmax(-1)
        fell_through = (uniform[:, None] >= cdf).all(-1)
        legal = mask.cpu() != 0
        last_legal = 42 - legal.flip(-1).int().argmax(-1)
        expected = torch.where(fell_through, last_legal, expected)
        # The two CDFs agree except within an ulp of the draw; a handful of
        # agents may differ there, never more.
        assert int((expected != actions).sum()) <= 2
        torch.testing.assert_close(
            sampled.logprobs.float().cpu(),
            logps.logps[0].float().cpu()[torch.arange(512), actions],
            rtol=1e-2,
            atol=1e-2,
        )


@pytest.mark.gpu_triton
def test_the_kernel_keys_on_the_seeds_high_word() -> None:
    """A seed past 2^32 draws curand's stream, not its low word's (SAMP-1)."""
    decoded, mask = _cuda_batch(512)
    config = PhiloxSampler.Config()
    config.seed = 2**32 + 73
    philox = config.make()
    torch_config = TorchPhiloxSampler.Config()
    torch_config.seed = config.seed
    torch_philox = torch_config.make()
    kernel_draws = philox.draws(512, device="cuda")
    cpu_draws = torch_philox.draws(512, device="cpu")
    for _ in range(3):
        kernel = philox(decoded, mask, kernel_draws, buffer=1)
        reference = torch_philox(decoded.cpu(), mask.cpu(), cpu_draws, buffer=1)
        # The two paths share the stream and differ only by the exp's ulps.
        assert int((kernel.actions.cpu() != reference.actions).sum()) <= 2


def test_the_torch_sampler_rounds_its_logprobs_and_values_to_the_dtype_asked() -> None:
    """In fp32 they are the sampler's own; in the decoder's bf16, those rounded once."""
    decoded, mask, draws = _portable_draw_inputs(64)
    sampler = TorchPhiloxSampler.Config().make()
    exact = sampler(decoded, mask, draws.clone(), buffer=1, dtype=torch.float32)
    rounded = sampler(decoded, mask, draws.clone(), buffer=1)
    _assert_rounded_once(exact, rounded)


def test_the_torch_sampler_matches_its_golden() -> None:
    """Twelve draws for 64 agents at exp000's seed, frozen on every host.

    Portable logits and masks inside ``host_agnostic_numerics``; each of the
    four buffers is its own stream, and the draw counts start uneven.
    """
    decoded, mask, draws = _portable_draw_inputs(64)
    with host_agnostic_numerics():
        lines = _sampler_entries(
            TorchPhiloxSampler.Config().make(),
            decoded,
            mask,
            draws,
        )
    assert_golden(test_file=__file__, name="sampler_tiny", lines=lines)


@pytest.mark.gpu_triton
def test_the_kernel_matches_its_golden_on_the_gpu() -> None:
    """Twelve draws for 64 agents at exp000's seed: the torch sampler's bits, on any GPU.

    The exact kernel draws, rounds and scores as the torch sampler does inside
    ``host_agnostic_numerics``, so it shares that test's golden and no GPU's.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    decoded, mask, draws = (value.cuda() for value in _portable_draw_inputs(64))
    lines = _sampler_entries(ExactPhiloxSampler.Config().make(), decoded, mask, draws)
    assert_golden(test_file=__file__, name="sampler_tiny", lines=lines)


@pytest.mark.gpu_triton
def test_the_kernel_rounds_its_logprobs_and_values_to_the_dtype_asked() -> None:
    """The kernel's fp32 log-probabilities, rounded, are its bf16 ones."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    decoded, mask, draws = (value.cuda() for value in _portable_draw_inputs(512))
    sampler = PhiloxSampler.Config().make()
    exact = sampler(decoded, mask, draws.clone(), buffer=1, dtype=torch.float32)
    rounded = sampler(decoded, mask, draws.clone(), buffer=1)
    _assert_rounded_once(exact, rounded)


@pytest.mark.gpu_triton
def test_the_step_graph_stores_fp32_rewards_logprobs_and_carries_unrounded() -> None:
    """On the GPU the ingest and emit kernels store each row in its tensor's dtype."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    rollout, env, _ = _rollout(
        horizon=2,
        device="cuda",
        state_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    env.rewards.fill_(0.1)
    graph = rollout.graphs[0][0]
    graph.replay()
    assert graph.stream is not None
    graph.stream.synchronize()
    storage = rollout.slots[0]
    assert torch.equal(storage.rewards[0, :4].cpu(), torch.full((4,), 0.1))
    assert storage.logprobs.dtype == graph.state.dtype == torch.float32
    assert not torch.equal(
        storage.logprobs[0, :4],
        storage.logprobs[0, :4].bfloat16().float(),
    )
    assert not torch.equal(graph.state, graph.state.bfloat16().float())
    values = storage.values[0, :4]
    assert not torch.equal(values, values.bfloat16().float())
    rollout.close()


def _assert_rounded_once(exact: Sampled, rounded: Sampled) -> None:
    """Assert the same draws, with ``rounded``'s values ``exact``'s rounded to bf16."""
    assert torch.equal(exact.actions, rounded.actions)
    assert exact.logprobs.dtype == exact.values.dtype == torch.float32
    assert rounded.logprobs.dtype == rounded.values.dtype == torch.bfloat16
    assert torch.equal(exact.logprobs.bfloat16(), rounded.logprobs)
    assert torch.equal(exact.values.bfloat16(), rounded.values)
    assert not torch.equal(exact.logprobs, rounded.logprobs.float())


def _portable_draw_inputs(agents: int) -> tuple[Tensor, Tensor, Tensor]:
    """Draw bf16 logits and value, a mask with every seventh last action masked, counts."""
    generator = torch.Generator().manual_seed(0)
    decoded = portable_uniform(agents, 44, bound=4.0, generator=generator).bfloat16()
    mask = (torch.rand(agents, 43, generator=generator) > 0.3).bfloat16()
    mask[:, 5] = 1
    mask[::7, 42] = 0
    return decoded, mask, torch.arange(agents) % 5


def _sampler_entries(
    philox: Sampler,
    decoded: Tensor,
    mask: Tensor,
    draws: Tensor,
) -> list[str]:
    """Draw three times on each of four buffers; digest the draws and the counts."""
    lines: list[str] = []
    for buffer in range(4):
        for step in range(3):
            sampled = philox(decoded, mask, draws, buffer=buffer)
            prefix = f"buffer {buffer} step {step}"
            lines += [f"{prefix} actions {digest(sampled.actions)}"]
            lines += [f"{prefix} logprobs {digest(sampled.logprobs)}"]
            lines += [f"{prefix} values {digest(sampled.values)}"]
            lines += [f"{prefix} draws {digest(draws)}"]
    return lines


FEATURE_WIDTH: Final = 5
"""The fake feature's width: what it saw at each step, one value per column."""


def test_a_feature_steps_on_the_upload_and_the_store_holds_what_the_policy_read() -> (
    None
):
    rollout, env, policy = _featured(horizon=5, hook=2)
    try:
        storage = rollout.collect(0)
        assert storage.features is not None
        assert storage.features.dtype == policy.dtype
        for buffer, engine in enumerate(rollout.engines):
            assert isinstance(engine, _CountingStep)
            rows = env.buffer_slice(buffer)
            # The blocks: two of the hook interval, then the rest.
            assert engine.rooms == [2, 2, 1]
            for time, (observation, terminals, previous) in enumerate(engine.seen):
                assert torch.equal(
                    observation.bfloat16(),
                    storage.observations[time, rows, 0],
                )
                assert torch.equal(terminals.bfloat16(), storage.terminals[time, rows])
                want = storage.actions[time - 1, rows] if time else torch.zeros(4)
                assert torch.equal(previous, want)
                fed = torch.stack(
                    [observation, terminals, previous, torch.full((4,), time)],
                    dim=-1,
                )
                assert torch.equal(
                    storage.features[time, rows, :4],
                    fed.to(policy.dtype),
                )
        assert rollout.feature_metrics() == {
            "feature/reprefill_rows": 2 * 3 * 2.0,
            "feature/rms": 1.0,
        }
        learner = LearnerRollout.from_time_major(
            storage.observations,
            storage.actions,
            storage.logprobs,
            storage.rewards,
            storage.terminals,
            storage.values,
            storage.action_mask,
            storage.initial_states,
            storage.branch_starts,
            reward_scale=1.0,
            reward_clip=1.0,
            features=storage.features,
        )
        assert learner.features is not None
        assert torch.equal(learner.features, storage.features.transpose(0, 1))
        minibatch = learner.minibatch(2, 3)
        assert minibatch.features is not None
        assert torch.equal(minibatch.features, learner.features[2:5])
    finally:
        rollout.close()


def test_the_learners_window_over_the_store_scores_what_the_actor_scored() -> None:
    """A frozen world model's feature: the learner re-reads exactly the actor's input.

    Rows of 8 positions, re-prefilled from their last 2 decisions, in blocks of 2
    steps: ``feature_test`` holds the engine to the training forward at any size.
    """
    config = _world_model_feature()
    history = config.history = Refill.Config()
    history.t_max = 8
    history.keep = 2
    config.hook_interval = 2
    source = config.make()
    policy_config = tiny_policy()
    proj = policy_config.proj_feature = Linear.Config()
    proj.channels_in = source.width
    torch.manual_seed(14)
    policy = policy_config.make()
    env = _FramedEnv.make(num_envs=8, num_buffers=2, seed=15)
    rollout_config = Rollout.Config()
    # Three blocks: by the third, the rows that ran 4 steps without a reset
    # have filled their 8 positions and re-prefill.
    rollout_config.horizon = 6
    rollout = Rollout(
        rollout_config,
        policy=policy,
        sampler=TorchPhiloxSampler.Config().make(),
        env=env,
        device=torch.device("cpu"),
        feature=source,
    )
    try:
        storage = rollout.collect(0)
        assert storage.features is not None
        assert storage.features.any()
        with torch.no_grad():
            decoded, _, _ = policy.forward_sequence(
                storage.observations.transpose(0, 1),
                storage.initial_states,
                storage.terminals.transpose(0, 1),
                features=storage.features.transpose(0, 1),
            )
            unfed, _, _ = policy.forward_sequence(
                storage.observations.transpose(0, 1),
                storage.initial_states,
                storage.terminals.transpose(0, 1),
                features=torch.zeros_like(storage.features).transpose(0, 1),
            )
        values = decoded[..., -1].transpose(0, 1).to(storage.values.dtype)
        assert torch.equal(values, storage.values)
        assert not torch.equal(unfed, decoded)
        metrics = rollout.feature_metrics()
        assert metrics["feature/reprefill_rows"] > 0
    finally:
        rollout.close()


def test_with_a_feature_the_store_is_saved_and_a_resume_begins_windows() -> None:
    rollout, _, _ = _featured(horizon=3, hook=3)
    try:
        storage = rollout.collect(0)
        assert storage.features is not None
        state = rollout.state_dict(slot=0)
        assert list(state)[-1] == "features"
        assert state["features"] is storage.features
        rollout.load_state_dict(
            {name: value.clone() for name, value in state.items()},
            slot=1,
        )
        assert rollout.slots[1].features is not None
        assert torch.equal(rollout.slots[1].features, storage.features)
        for engine in rollout.engines:
            assert isinstance(engine, _CountingStep)
            assert bool(engine.window.all())
        rollout.reset()
        for engine in rollout.engines:
            assert isinstance(engine, _CountingStep)
            assert engine.resets == 1
    finally:
        rollout.close()
    assert all(
        isinstance(engine, _CountingStep) and engine.released
        for engine in rollout.engines
    )


def test_a_feature_is_refused_a_bootstrap_row() -> None:
    config = Rollout.Config()
    config.horizon = 2
    config.bootstrap = True
    # A carry-free policy, so the bootstrap row alone is what is refused.
    policy = ActorCritic.Config()
    policy.channels_hidden = 3
    policy.num_layers = 1
    with pytest.raises(ValueError, match="a feature's history advances once"):
        Rollout(
            config,
            policy=policy.make(),
            sampler=TorchPhiloxSampler.Config().make(),
            env=FakeEnv.make(num_envs=6, num_buffers=2, seed=0),
            device=torch.device("cpu"),
            feature=_CountingFeature.Config().make(),
        )


@pytest.mark.parametrize("donor", [False, True], ids=["fresh_window", "donor"])
@pytest.mark.parametrize("sliding", [False, True], ids=["refill", "sliding"])
def test_a_restored_rows_feature_resumes_its_donors_saved_history_or_a_window(
    *,
    sliding: bool,
    donor: bool,
) -> None:
    _assert_restored_rows_read_their_donors_histories(
        "cpu",
        sliding=sliding,
        donor=donor,
    )


@pytest.mark.gpu_triton
@pytest.mark.gpu_torch_cuda
@pytest.mark.parametrize("sliding", [False, True], ids=["refill", "sliding"])
def test_a_captured_rollout_saves_and_restores_feature_histories(
    *,
    sliding: bool,
) -> None:
    """The step graph saves the donors' histories; each plan of the feature captures."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    _assert_restored_rows_read_their_donors_histories(
        "cuda",
        sliding=sliding,
        donor=True,
    )


# The archive holds each donor's history as its next step would extend it, and a
# restored row's first feature is an engine's restored the same way.
def _assert_restored_rows_read_their_donors_histories(
    device: str,
    *,
    sliding: bool,
    donor: bool,
) -> None:
    """Rows 5 and 4 save mid-rollout and on its last step; rows 0 and 1 are restored."""
    config = _world_model_feature()
    if sliding:
        window = config.history = Sliding.Config()
        window.decisions = 4
    if donor:
        config.practice = DonorHistory.Config()
    source = config.make()
    env = _PractisingEnv(
        _FramedEnv.make(num_envs=6, num_buffers=2, seed=20),
        carry_slots=5,
        save_rows=slice(4, 6),
        saves={(1, 0): {5: 2}, (1, 2): {4: 4}},
    )
    policy_config = tiny_policy()
    proj = policy_config.proj_feature = Linear.Config()
    proj.channels_in = source.width
    torch.manual_seed(21)
    policy = policy_config.make().to(device)
    rollout_config = Rollout.Config()
    rollout_config.horizon = 3
    sampler = PhiloxSampler if device == "cuda" else TorchPhiloxSampler
    rollout = Rollout(
        rollout_config,
        policy=policy,
        sampler=sampler.Config().make(),
        env=env,
        device=torch.device(device),
        feature=source,
    )
    try:
        env.carries = [graph.state for graph in rollout.graphs[0]]
        env.engines = [_narrowed(engine) for engine in rollout.engines]
        rollout.collect(0)
        if device == "cuda":
            torch.cuda.synchronize()
        if donor:
            histories = rollout.histories
            assert histories is not None
            assert set(env.histories) == {2, 4}
            for slot, saved in env.histories.items():
                for name, value in saved.items():
                    assert torch.equal(histories[name][slot].cpu(), value), (slot, name)
        else:
            assert rollout.histories is None
        env.restore_slots[:2] = torch.tensor([2, 4], dtype=torch.int32)
        env.terminals[:2] = 0.0
        observations = env.observations[:3].clone().to(device)
        storage = rollout.collect(1)
        reference = _narrowed(source.make_engine(rows=3, device=torch.device(device)))
        previous = reference.restore_history(
            rollout.histories,
            torch.tensor([2, 4, -1]),
        )
        reference.ensure_room(1)
        feature = reference(
            observations,
            torch.zeros(3, device=device),
            torch.zeros(3, device=device) if previous is None else previous,
        )
        assert storage.features is not None
        assert torch.equal(
            storage.features[0, :2],
            feature[:2].to(storage.features.dtype),
        )
        decisions, anchored = reference.context()
        if not donor:
            assert decisions[:2].tolist() == [1, 1]
            assert anchored[:2].tolist() == [False, False]
    finally:
        rollout.close()


@pytest.mark.parametrize("sliding", [False, True], ids=["refill", "sliding"])
def test_a_joint_slot_stores_the_inputs_each_of_its_features_reads(
    *,
    sliding: bool,
) -> None:
    """The training forward over each step's stored context gives its stored feature.

    Two rollouts: the second after new weights and a rebuild, its rows 0 and 1
    restored from saved histories. Rows reset, re-prefill (``Refill``) or
    slide (``Sliding``) within each.
    """
    rollout, env, source = _joint_rollout("cpu", sliding=sliding)
    try:
        assert rollout.histories is not None
        assert "obs" not in rollout.histories
        storage = rollout.collect(0)
        old = copy.deepcopy(source.model)
        _publish(source.model)
        rollout.rebuild_features()
        env.restore_slots[:2] = torch.tensor([2, 4], dtype=torch.int32)
        env.terminals[:2] = 0.0
        restored = rollout.collect(1)
        assert restored.prefix_decisions is not None
        assert restored.prefix_decisions[:2].max() > 0
        assert storage.context_anchored is not None
        assert restored.context_anchored is not None
        assert not (storage.context_anchored & restored.context_anchored).all()
        for slot, model in ((storage, old), (restored, source.model)):
            assert slot.features is not None
            torch.testing.assert_close(
                slot.features,
                _replayed_features(slot, model),
                rtol=0,
                atol=2e-6,
            )
        state = rollout.state_dict(slot=1)
        assert list(state)[-10:-1] == [
            "frame_cells",
            "frame_aux",
            "previous_actions",
            "context_decisions",
            "context_anchored",
            "prefix_cells",
            "prefix_aux",
            "prefix_previous_actions",
            "prefix_decisions",
        ]
    finally:
        rollout.close()


def test_rebuilding_a_frozen_features_history_is_refused() -> None:
    rollout, _, _ = _featured(horizon=2, hook=2)
    try:
        assert rollout.slots[0].frame_cells is None
        with pytest.raises(ValueError, match="joint"):
            rollout.rebuild_features()
    finally:
        rollout.close()


@pytest.mark.gpu_triton
@pytest.mark.gpu_torch_cuda
@pytest.mark.parametrize("sliding", [False, True], ids=["refill", "sliding"])
def test_a_captured_joint_rollout_stores_and_rebuilds_as_the_eager_one(
    *,
    sliding: bool,
) -> None:
    """Graphs store the learner's inputs and rebuild as eager steps do."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    runs: list[list[Tensor]] = []
    for captured in (False, True):
        rollout, env, source = _joint_rollout("cuda", sliding=sliding)
        try:
            if not captured:
                for graphs in rollout.graphs:
                    for graph in graphs:
                        graph.stream = None
            stored: list[Tensor] = []
            for slot in range(2):
                if slot:
                    _publish(source.model)
                    rollout.rebuild_features()
                    env.restore_slots[:2] = torch.tensor([2, 4], dtype=torch.int32)
                    env.terminals[:2] = 0.0
                rollout.collect(slot)
                torch.cuda.synchronize()
                stored += [
                    value.cpu()
                    for name, value in rollout.state_dict(slot=slot).items()
                    if name not in {"carry", "draws", "archive"}
                ]
            runs.append(stored)
        finally:
            rollout.close()
    for index, (want, got) in enumerate(zip(runs[0], runs[1], strict=True)):
        assert torch.equal(want, got), index


@pytest.mark.gpu_triton
@pytest.mark.gpu_torch_cuda
def test_a_captured_rollout_steps_a_world_model_feature_as_the_eager_one() -> None:
    """A capture's warmup leaves the feature's state and inputs unchanged (G3a, rollout).

    The second slot's graphs capture mid-episode, where the feature reads each
    row's previous action from the env's host buffer, which the warmup writes.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    # One source for both rollouts: each makes engines of its own.
    source = _world_model_feature().make()
    runs: list[list[Tensor]] = []
    for captured in (False, True):
        source_policy = tiny_policy()
        proj = source_policy.proj_feature = Linear.Config()
        proj.channels_in = 36
        torch.manual_seed(16)
        policy = source_policy.make().cuda()
        env = _FramedEnv.make(num_envs=8, num_buffers=2, seed=17)
        rollout_config = Rollout.Config()
        rollout_config.horizon = 6
        rollout = Rollout(
            rollout_config,
            policy=policy,
            sampler=PhiloxSampler.Config().make(),
            env=env,
            device=torch.device("cuda"),
            feature=source,
        )
        try:
            if not captured:
                for graphs in rollout.graphs:
                    for graph in graphs:
                        graph.stream = None
            stored: list[Tensor] = []
            for slot in range(rollout_config.num_slots):
                storage = rollout.collect(slot)
                torch.cuda.synchronize()
                assert storage.features is not None
                stored += [storage.features.cpu(), storage.actions.cpu()]
            runs.append(stored)
        finally:
            rollout.close()
    eager, graphed = runs
    for name, want, got in zip(
        ["slot 0 features", "slot 0 actions", "slot 1 features", "slot 1 actions"],
        eager,
        graphed,
        strict=True,
    ):
        assert torch.equal(want, got), name


class _CountingFeature:
    """A feature that reports what each step fed it, as its columns."""

    class Config(Fig["_CountingFeature"]):
        """The interval between blocks."""

        hook_interval: int = 2
        """Steps between ``ensure_room`` calls."""

    def __init__(self, config: Config) -> None:
        self.width = FEATURE_WIDTH
        self.hook_interval = config.hook_interval
        self.joint = False
        self.context_decisions = 1

    def make_engine(self, *, rows: int, device: torch.device) -> _CountingStep:
        """Return a fresh counting step."""
        return _CountingStep(rows=rows, device=device)

    def history_archive(self, entries: int, *, device: torch.device) -> None:
        """Keep no history: a restored row begins a window."""
        del entries, device


class _CountingStep:
    """Columns: the first observed float, the terminal, the previous action, the step."""

    def __init__(self, *, rows: int, device: torch.device) -> None:
        self.count = torch.zeros((), device=device)
        self.window = torch.zeros(rows, dtype=torch.bool, device=device)
        self.seen: list[tuple[Tensor, Tensor, Tensor]] = []
        self.rooms: list[int] = []
        self.resets = 0
        self.released = False
        self.plan = 0

    def __call__(
        self,
        observation: Tensor,
        terminals: Tensor,
        previous_action: Tensor,
    ) -> Tensor:
        """Record the inputs; return them, the step and the row index as the feature."""
        self.seen.append(
            (observation[:, 0].clone(), terminals.clone(), previous_action.clone()),
        )
        rows = len(terminals)
        feature = torch.stack(
            [
                observation[:, 0],
                terminals,
                previous_action,
                self.count.expand(rows),
                torch.arange(rows, device=terminals.device).float(),
            ],
            dim=-1,
        )
        self.count.add_(1)
        return feature

    def capture_state(self) -> list[Tensor]:
        """Return the step counter."""
        return [self.count]

    def ensure_room(self, steps: int) -> dict[str, float]:
        """Record the block's length."""
        self.rooms.append(steps)
        return {"feature/rms": 1.0, "feature/reprefill_rows": 2.0}

    def reset(self) -> None:
        """Count the reset."""
        self.resets += 1

    def begin_window(self, rows: Tensor) -> None:
        """Mark the rows."""
        self.window |= rows.to(self.window.device)

    def save_history(
        self,
        archive: Mapping[str, Tensor],
        entries: Tensor,
        rows: slice,
        *,
        previous_action: Tensor,
        fresh: Tensor,
    ) -> None:
        """Save nothing: the source keeps no archive, so no rollout calls this."""
        del archive, entries, rows, previous_action, fresh

    def restore_history(
        self,
        archive: Mapping[str, Tensor] | None,
        entries: Tensor,
    ) -> None:
        """Mark the restored rows as windows."""
        assert archive is None
        self.begin_window(entries >= 0)

    def context(self) -> tuple[Tensor, Tensor]:
        """Return one decision per row, unanchored: the double keeps no history."""
        rows = len(self.window)
        return self.window.new_ones(rows, dtype=torch.long), self.window.new_zeros(rows)

    def last_decisions(self, n: int) -> RecentDecisions:
        """Refuse: the double is not joint, so the rollout never asks."""
        raise AssertionError(n)

    def rebuild(self) -> None:
        """Refuse: the double is not joint, so the rollout never asks."""
        raise AssertionError

    def release(self) -> None:
        """Record the release."""
        self.released = True


@dataclass(slots=True, kw_only=True)
class _FramedEnv:
    """``FakeEnv``'s transitions with game frames for observations, as a world model reads."""

    inner: FakeEnv
    observations: Tensor
    action_mask: Tensor
    rewards: Tensor
    terminals: Tensor
    actions: Tensor
    num_envs: int
    num_buffers: int
    frames: Tensor

    @classmethod
    def make(cls, *, num_envs: int, num_buffers: int, seed: int) -> _FramedEnv:
        """Wrap a ``FakeEnv`` whose observations are decoded random token frames."""
        inner = FakeEnv.make(num_envs=num_envs, num_buffers=num_buffers, seed=seed)
        segment = random_segment(craftax_schema(), 31, seed=seed)
        env = cls(
            inner=inner,
            observations=inner.observations,
            action_mask=inner.action_mask,
            rewards=inner.rewards,
            terminals=inner.terminals,
            actions=inner.actions,
            num_envs=num_envs,
            num_buffers=num_buffers,
            frames=decode(segment.cells, segment.aux),
        )
        for buffer in range(num_buffers):
            env.frame(buffer)
        return env

    def buffer_slice(self, buffer: int) -> slice:
        """Return the rows of ``buffer``, as the inner env's."""
        return self.inner.buffer_slice(buffer)

    def step_buffer(self, buffer: int) -> None:
        """Step the inner env, then show each row a frame of its own."""
        self.inner.step_buffer(buffer)
        self.frame(buffer)

    def frame(self, buffer: int) -> None:
        """Write a frame drawn by the buffer's generator into each of its rows."""
        rows = self.buffer_slice(buffer)
        pick = torch.randint(
            len(self.frames),
            (len(range(*rows.indices(self.num_envs))),),
            generator=self.inner.generators[buffer],
        )
        self.observations[rows] = self.frames[pick]


def _featured(*, horizon: int, hook: int) -> tuple[Rollout, FakeEnv, MinGRUPolicy]:
    """Return a CPU rollout of the counting feature and a policy reading it."""
    config = tiny_policy()
    proj = config.proj_feature = Linear.Config()
    proj.channels_in = FEATURE_WIDTH
    torch.manual_seed(18)
    policy = config.make()
    env = FakeEnv.make(num_envs=8, num_buffers=2, seed=19)
    rollout_config = Rollout.Config()
    rollout_config.horizon = horizon
    feature = _CountingFeature.Config()
    feature.hook_interval = hook
    rollout = Rollout(
        rollout_config,
        policy=policy,
        sampler=TorchPhiloxSampler.Config().make(),
        env=env,
        device=torch.device("cpu"),
        feature=feature.make(),
    )
    return rollout, env, policy


def _world_model_feature() -> WorldModelFeature.Config:
    """Return the smoke feature hooked every 4 steps: rows re-prefill within a block."""
    config = smoke_feature()
    config.weights = _SmokeWeights.Config()
    config.hook_interval = 4
    return config


class _SmokeWeights:
    """``smoke_feature``'s world model, built once a process and copied for each source.

    The build finalizes ``exp_smoke``'s whole config and draws its ~300 weights:
    30 ms on x86, a third of a test that makes a source. ``feature_test`` checks
    ``InitialWeights`` itself.
    """

    class Config(Fig["_SmokeWeights"]):
        """Nothing to configure: the weights are ``smoke_feature``'s."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self) -> WorldModel:
        """Return a copy of the model, at the seeded init and in eval mode."""
        return copy.deepcopy(_smoke_model())


@functools.cache
def _smoke_model() -> WorldModel:
    """Build ``smoke_feature``'s world model."""
    return smoke_feature().weights.make()()


# One buffer of 4 rows, whose rows 3 and 2 save entries 2 and 4 in the first rollout;
# the feature is float32, and the policy, so the store holds the features unrounded.
# Rows of 8 positions re-prefilled from 2 decisions, or windows of 4, in blocks of 2
# steps over rollouts of 3: each training forward reads the full schema's frames.
def _joint_rollout(
    device: str,
    *,
    sliding: bool,
) -> tuple[Rollout, _PractisingEnv, WorldModelFeature]:
    """Return a joint feature's practising rollout: 2 of 4 rows save histories."""
    config = _world_model_feature()
    if sliding:
        window = config.history = Sliding.Config()
        window.decisions = 4
    else:
        refill = config.history = Refill.Config()
        refill.t_max = 8
        refill.keep = 2
    config.hook_interval = 2
    config.practice = DonorHistory.Config()
    config.joint = True
    source = config.make()
    env = _PractisingEnv(
        _FramedEnv.make(num_envs=4, num_buffers=1, seed=22),
        carry_slots=5,
        save_rows=slice(2, 4),
        saves={(0, 0): {3: 2}, (0, 2): {2: 4}},
    )
    policy_config = tiny_policy(dtype=torch.float32)
    proj = policy_config.proj_feature = Linear.Config()
    proj.channels_in = source.width
    torch.manual_seed(23)
    policy = policy_config.make().to(device)
    rollout_config = Rollout.Config()
    rollout_config.horizon = 3
    sampler = PhiloxSampler if device == "cuda" else TorchPhiloxSampler
    rollout = Rollout(
        rollout_config,
        policy=policy,
        sampler=sampler.Config().make(),
        env=env,
        device=torch.device(device),
        feature=source,
    )
    env.carries = [graph.state for graph in rollout.graphs[0]]
    return rollout, env, source


def _publish(model: nn.Module) -> None:
    """Move every weight of ``model`` in place, as a joint learner's publication."""
    generator = torch.Generator().manual_seed(24)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            parameter.add_(0.2 * parameter.abs().mean() * noise.to(parameter.device))


# Each row's decisions are its prefix, then its steps; step ``t``'s context is its last
# ``context_decisions`` of them, and the action taken at each is the next one's previous
# action. Steps whose contexts extend one another share a forward, read at each one's
# ``obs``.
def _replayed_features(storage: RolloutStorage, model: WorldModel) -> Tensor:
    """Return the training forward's feature of each context the store describes."""
    cells, aux, previous, decisions, anchored, prefix_cells, prefix_aux = (
        storage.frame_cells,
        storage.frame_aux,
        storage.previous_actions,
        storage.context_decisions,
        storage.context_anchored,
        storage.prefix_cells,
        storage.prefix_aux,
    )
    assert cells is not None
    assert aux is not None
    assert previous is not None
    assert decisions is not None
    assert anchored is not None
    assert prefix_cells is not None
    assert prefix_aux is not None
    assert storage.prefix_previous_actions is not None
    assert storage.prefix_decisions is not None
    horizon, agents = decisions.shape
    prefix = prefix_cells.shape[1]
    steps = torch.arange(horizon, device=decisions.device)[:, None]
    # Every context reaches no further back than the stored prefix.
    assert bool((steps - decisions + 1 >= -storage.prefix_decisions).all())
    history = (
        torch.cat([prefix_cells, cells.transpose(0, 1)], dim=1).cpu(),
        torch.cat([prefix_aux, aux.transpose(0, 1)], dim=1).cpu(),
        torch.cat([storage.prefix_previous_actions, previous.transpose(0, 1)], dim=1)
        .roll(-1, dims=1)
        .cpu(),
    )
    spans: list[Span] = []
    for agent in range(agents):
        for t in range(horizon):
            first = prefix + t - int(decisions[t, agent]) + 1
            starts = bool(anchored[t, agent])
            span = spans[-1] if spans else None
            if span is None or (span.row, span.first, span.starts_episode) != (
                agent,
                first,
                starts,
            ):
                span = Span(row=agent, first=first, starts_episode=starts, steps=[])
                spans.append(span)
            span.steps.append(prefix + t)
    features = reference_features(
        model,
        layers=1,
        cells=history[0],
        aux=history[1],
        actions=history[2],
        spans=spans,
    )
    return features[:, prefix:].transpose(0, 1).to(decisions.device)


def _history(
    engine: FeatureEngine,
    row: int,
    *,
    previous_action: Tensor,
    ended: bool,
) -> dict[str, Tensor]:
    """Return a copy of what a save of ``row`` holds: its history, empty if it ended."""
    length = engine.length[row]
    return {
        "obs": engine.obs_ring[row].cpu().clone(),
        "actions": engine.action_ring[row].cpu().clone(),
        "count": engine.count[row].cpu().clone(),
        "length": (torch.zeros_like(length) if ended else length).cpu().clone(),
        "previous_action": previous_action.clone(),
    }


def _narrowed(engine: FeatureStep) -> FeatureEngine:
    """Narrow a world-model source's step to its engine."""
    assert isinstance(engine, FeatureEngine)
    return engine


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
