"""Tests for the capture environment: its records, their replay, and the v2 settings.

Every capture plays the game's kernels as Python (``eager``) on tiny worlds of
grass, whose clock runs out a few decisions after the reset, so an episode
is a few decisions long.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final

import dataclasses

import numpy as np
import pytest
import torch

from priml.baselines.craftax.eager import eager, tiny_world
from priml.baselines.craftax.game import step
from priml.baselines.craftax.game.state import (
    DEFAULT_MAX_TIMESTEPS,
    Action,
    env_state,
    env_stats,
)
from priml.baselines.craftax.game.step import Rules
from priml.baselines.craftax.rollout import Rollout, TorchPhiloxSampler
from priml.baselines.craftax.testing import tiny_policy
from priml.baselines.craftax.world_model import replay
from priml.baselines.craftax.world_model.archive import Episode
from priml.baselines.craftax.world_model.capture.env import (
    BranchStart,
    Captured,
    CaptureEnv,
    Schedule,
)
from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    episode_seed,
    splitmix64,
)
from priml.baselines.craftax.world_model.capture.source import (
    RandomSource,
)
from priml.baselines.craftax.world_model.index import (
    FLOOR_AUX,
    floor_trace,
)
from priml.lib.codec import from_plain


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


_DECISIONS: Final = 3
"""Decisions an episode of ``_SHORT`` plays before its clock runs out."""

_SHORT: Final = partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - _DECISIONS)


@pytest.fixture(autouse=True)
def cpu_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture as a host without CUDA does: unpinned buffers, no CUDA context to start.

    On a GPU host the first pinned buffer of a process starts the CUDA context,
    120 ms on the x86 host, which these tests of play and recording never use.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def short() -> Generator[None]:
    with eager(world=_SHORT):
        yield


def test_a_capture_without_buffers_is_refused() -> None:
    with pytest.raises(ValueError, match="num_buffers=0"):
        CaptureEnv(
            CaptureEnv.Config(num_envs=2, num_buffers=0),
            schedule=Schedule(budget=1),
        )


def test_a_chunk_off_the_hash_stride_is_refused() -> None:
    with pytest.raises(ValueError, match="multiple of 256"):
        CaptureEnv(
            CaptureEnv.Config(num_envs=2, chunk_decisions=100),
            schedule=Schedule(budget=1),
        )


def test_a_stall_limit_of_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="stall_limit=0"):
        CaptureEnv.Config(num_envs=2, stall_limit=0).finalize()


def test_rules_that_change_the_game_are_refused() -> None:
    config = CaptureEnv.Config(num_envs=2)
    config.rules.end_on_boss_defeat = True
    with pytest.raises(ValueError, match="default rules"):
        CaptureEnv(config, schedule=Schedule(budget=1))


def test_the_previous_action_layout_reads_none_after_a_reset(short: None) -> None:
    del short
    config = CaptureEnv.Config(num_envs=2)
    config.rules.previous_action = True
    env = CaptureEnv(config, schedule=Schedule(budget=1_000))
    assert env.observations.shape == (2, 844)
    assert env.observations[:, -1].tolist() == [43.0, 43.0]
    env.actions[:, 0] = torch.tensor([5.0, 7.0])
    env.step_buffer(0)
    assert env.observations[:, -1].tolist() == [5.0, 7.0]


@pytest.mark.parametrize("action", [-1.0, 43.0, float("nan")])
def test_an_action_outside_the_schema_stops_recording(
    short: None,
    action: float,
) -> None:
    del short
    env = CaptureEnv(CaptureEnv.Config(num_envs=2), schedule=Schedule(budget=1_000))
    env.actions[0, 0] = action
    env.step_buffer(0)
    assert "outside 0-42" in env.failure
    assert env.receipt(0) is None
    # The stopped row holds nothing; the other its first decision.
    assert (env.decisions(0), env.returned(0), env.decisions(1)) == (0, 0.0, 1)
    assert env.in_flight() == 1
    _play(env, decisions=_DECISIONS)
    assert env.finished()
    # Its episode over, the other holds none either, though its count stays.
    assert env.decisions(1) == 0


def test_a_recording_row_reads_its_decisions_and_achievement_return(
    short: None,
) -> None:
    # The tiny world's tree stands above the start: facing it, DO collects wood.
    del short
    env = CaptureEnv(CaptureEnv.Config(num_envs=1), schedule=Schedule(budget=10**9))
    for action in (Action.UP, Action.DO):
        env.actions[0, 0] = float(action.value)
        env.step_buffer(0)
    assert (env.decisions(0), env.returned(0)) == (2, 1.0)


def test_random_play_records_what_replay_record_records(short: None) -> None:
    # Each action drawn among the legal ones by splitmix64 from the episode's
    # sampling seed, as replay.record draws; replay.record equals the reference
    # recordings (the parity suite checks it), so equal records here make this
    # capture the reference's. Two rounds: each row resets into a second episode.
    del short
    env = CaptureEnv(CaptureEnv.Config(num_envs=2), schedule=Schedule(budget=10**9))
    streams = [0, 0]
    finished: list[Captured] = []
    while len(finished) < 4:
        for e in range(2):
            receipt = env.receipt(e)
            assert receipt is not None
            if not env.decisions(e):
                streams[e] = receipt.sampling_seed
            streams[e], draw = splitmix64(streams[e])
            legal = env.action_mask[e].nonzero().flatten()
            env.actions[e, 0] = float(legal[draw % len(legal)])
        env.step_buffer(0)
        finished += env.episodes()
    for episode in finished:
        receipt = episode.receipt
        recorded = replay.record(
            world_seed=receipt.world_seed,
            sampling_seed=receipt.sampling_seed,
            max_decisions=len(episode.actions),
        )
        assert recorded.receipt == dataclasses.replace(receipt, split=0)
        assert torch.equal(episode.actions, recorded.actions)
        assert torch.equal(episode.hashes, recorded.hashes)
        assert episode.floors == floor_trace(
            aux=recorded.aux,
            reward=recorded.reward,
            done=recorded.done,
        )


def test_a_full_chunk_moves_into_its_episode_and_hashes_keep_their_stride() -> None:
    # Two decisions on a world as a 256-decision chunk fills: the row is set as
    # if 255 decisions were recorded, so the first fills the chunk and the
    # second is decision 256, which takes a stride hash, and ends the episode.
    with eager(world=partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - 2)):
        env = CaptureEnv(
            CaptureEnv.Config(num_envs=1, chunk_decisions=256),
            schedule=Schedule(budget=10**9),
        )
        rows = env._rows
        rows.fill[0] = rows.decisions[0] = 255
        rows.hash_fill[0] = 0
        env.actions[0, 0] = 2.0
        env.step_buffer(0)
        assert (rows.fill.item(0), rows.decisions.item(0)) == (0, 256)
        stride = replay.fnv1a_numba(env.states.view(np.uint8))
        # The row resets once its episode ends: its last State, played on a copy.
        after, rng, stats = env.states.copy(), env.rngs.copy(), rows.stats.copy()
        step.play_numba(env_state(after, 0), rng, env_stats(stats, 0), 1, Rules())
        final = replay.fnv1a_numba(after.view(np.uint8))
        env.actions[0, 0] = 1.0
        env.step_buffer(0)
    (episode,) = env.episodes()
    assert len(episode.actions) == 257
    assert episode.actions[-2:].tolist() == [2, 1]
    assert episode.hashes.numpy().view(np.uint64).tolist() == [stride, final]


def test_the_summary_counts_floors_and_achievements_from_the_episode(
    short: None,
) -> None:
    del short
    env = CaptureEnv(CaptureEnv.Config(num_envs=2), schedule=Schedule(budget=10**9))
    (episode, *_) = _play(env, decisions=_DECISIONS, episodes=1)
    frames = _replayed(episode)
    summary = episode.summary
    floors = from_plain(summary["floors"], list[dict[str, object]])
    decisions = [from_plain(f["decisions"], int) for f in floors]
    visited = frames.aux[:, FLOOR_AUX].bincount(minlength=9).tolist()
    assert decisions == visited == [_DECISIONS, *[0] * 8]
    assert from_plain(floors[0]["reached"], int) == 1
    assert from_plain(summary["episode"], int) == 0
    died = bool(frames.done[-1]) and int(frames.reward[-1]) == -1
    assert from_plain(summary["death"], int) == int(died)
    assert from_plain(summary["timeout"], int) == 1
    assert episode.floors.died == died
    assert "epsilon" not in summary
    assert "truncated" not in summary
    assert list(summary)[:5] == ["episode", "environment", "return", "death", "timeout"]


def test_epsilon_one_replaces_every_action_by_a_legal_draw() -> None:
    config = CaptureEnv.Config(num_envs=2, epsilon=1.0)
    with eager(world=partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - 6)):
        episodes = _play(
            CaptureEnv(config, schedule=Schedule(budget=10**9)),
            decisions=6,
            episodes=2,
        )
        statuses = [replay.verify(episode) for episode in episodes]
        again = _play(
            CaptureEnv(config, schedule=Schedule(budget=10**9)),
            decisions=6,
            episodes=2,
        )
    for episode in episodes:
        # The policy plays NOOP only; the override draws every action.
        assert int((episode.actions != 0).sum()) > len(episode.actions) // 2
    assert statuses == [replay.MATCHED] * 2
    assert torch.equal(episodes[0].actions, again[0].actions)


def test_an_epsilon_range_draws_each_episodes_epsilon_into_its_summary(
    short: None,
) -> None:
    del short
    config = CaptureEnv.Config(num_envs=2, epsilon=0.25, epsilon_high=0.75)
    episodes = _play(
        CaptureEnv(config, schedule=Schedule(budget=10**9)),
        decisions=_DECISIONS,
        episodes=2,
    )
    drawn = [from_plain(e.summary["epsilon"], float) for e in episodes]
    assert all(0.25 <= value <= 0.75 for value in drawn)
    assert drawn[0] != drawn[1]


def test_the_stall_cap_truncates_training_episodes_that_replay() -> None:
    config = CaptureEnv.Config(num_envs=2, stall_limit=2)
    with eager(world=partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - 10)):
        env = CaptureEnv(config, schedule=Schedule(budget=10**9))
        (episode, *_) = _play(env, decisions=10, episodes=1)
        frames = _replayed(episode)
        row = from_plain(episode.summary["environment"], int)
        assert env.receipt(row) is None
        env.step_buffer(0)
    assert episode.truncated
    assert not bool(frames.done[-1])
    assert episode.summary["truncated"] == 1
    assert episode.receipt.split == TRAIN
    assert len(episode.actions) == 2
    assert bool((frames.reward <= 0).all())
    # The environment ends its next, unrecorded, decision at the timeout, and
    # records the next world.
    assert float(env.terminals[row]) == 1.0
    assert env.receipt(row) is not None


def test_a_budget_drains_the_capture(short: None) -> None:
    del short
    env = CaptureEnv(CaptureEnv.Config(num_envs=3), schedule=Schedule(budget=1))
    episodes = _play(env, decisions=2 * _DECISIONS)
    assert env.finished()
    ordinals = {from_plain(e.summary["episode"], int) for e in episodes}
    assert ordinals == {0, 1, 2}


def test_a_budget_of_one_decision_records_one_episode_per_environment_on_its_world(
    short: None,
) -> None:
    # As a ghost pilot plays: rows reset in order at construction, so
    # environment ``i`` records ordinal ``i``, from world ``world_seeds[i]``,
    # whichever of the two buffers steps first.
    del short
    config = CaptureEnv.Config(num_envs=4, num_buffers=2, world_seeds=tuple(range(16)))
    env = CaptureEnv(config, schedule=Schedule(arm=2, budget=1))
    episodes: list[Captured] = []
    for _ in range(2 * _DECISIONS):
        for buffer in (1, 0):
            env.step_buffer(buffer)
        episodes += env.episodes()
    assert env.finished()
    rows = sorted(
        (
            from_plain(e.summary["episode"], int),
            from_plain(e.summary["environment"], int),
            e.receipt.world_seed,
        )
        for e in episodes
    )
    assert rows == [(i, i, i) for i in range(4)]


def test_without_fixed_worlds_every_episode_plays_its_own_seed(short: None) -> None:
    del short
    env = CaptureEnv(
        CaptureEnv.Config(num_envs=3),
        schedule=Schedule(arm=2, worker=1, generation=3, budget=1),
    )
    episodes = _play(env, decisions=2 * _DECISIONS)
    assert len(episodes) == 3
    for episode in episodes:
        ordinal = from_plain(episode.summary["episode"], int)
        receipt = episode.receipt
        assert (receipt.split, receipt.world_seed) == episode_seed(
            ordinal,
            arm=2,
            worker=1,
            generation=3,
        )


# Random play on three rows over two worlds, one decision a poll, until 12
# decisions have ended, so rows reset into their next episodes mid-capture.
def test_fixed_worlds_start_every_episode_from_its_worlds_state_and_stream(
    tmp_path: Path,
    short: None,
) -> None:
    del short
    worlds = (3, 11)
    config = RandomSource.Config(steps_per_poll=1)
    config.env.num_envs = 3
    config.env.world_seeds = worlds
    source = config.make()
    source.start(schedule=Schedule(budget=12), run_dir=tmp_path)
    env = source.env
    assert env is not None
    starts = {seed: replay.save(*replay.reset_world(seed)) for seed in worlds}
    started: list[int] = []
    episodes: list[Captured] = []
    while not source.finished():
        # A recorded row without a decision has just reset into its episode.
        for row in range(env.num_envs):
            receipt = env.receipt(row)
            if receipt is not None and not env.decisions(row):
                state = replay.save(env.states[row : row + 1], env.rngs[row : row + 1])
                assert state == starts[receipt.world_seed]
                started.append(receipt.world_seed)
        episodes += source.poll()
    ordinals = [from_plain(e.summary["episode"], int) for e in episodes]
    assert sorted(ordinals) == list(range(len(episodes)))
    assert sorted(started) == sorted(e.receipt.world_seed for e in episodes)
    assert len(episodes) > env.num_envs
    for episode, ordinal in zip(episodes, ordinals, strict=True):
        assert episode.receipt.world_seed == worlds[ordinal % len(worlds)]
        assert replay.verify(episode) == replay.MATCHED
    # One start, a different action stream per episode.
    assert len({bytes(e.actions.numpy()) for e in episodes}) == len(episodes)


def test_a_branch_starts_from_its_parents_state_and_replays() -> None:
    with eager(world=partial(tiny_world, timestep=DEFAULT_MAX_TIMESTEPS - 6)):
        parent = replay.record(world_seed=7, sampling_seed=11, max_decisions=6)
        decision = len(parent.actions) // 2
        origin = replay.origin(parent, decision=decision)
        starts = _Starts(
            starts=[
                BranchStart(world_seed=7, origin=origin, point={"decision": decision}),
            ],
        )
        env = CaptureEnv(
            CaptureEnv.Config(num_envs=2),
            schedule=Schedule(budget=10**9, generation=2),
            branches=starts,
        )
        assert env.receipt(1) is None
        (branch,) = _play(env, decisions=6)
        status = replay.verify(branch)
    assert env.finished()
    assert branch.origin == origin
    assert branch.receipt.world_seed == 7
    assert len(branch.actions) == len(parent.actions) - decision
    assert branch.summary["branch"] == {"decision": decision}
    assert status == replay.MATCHED


def test_the_ports_rollout_captures_episodes_that_replay(short: None) -> None:
    # One rollout of a tiny policy: every row's episode ends within its horizon.
    del short
    env = CaptureEnv(
        CaptureEnv.Config(num_envs=4, num_buffers=2),
        schedule=Schedule(budget=10**9),
    )
    torch.manual_seed(0)
    config = Rollout.Config()
    config.horizon = _DECISIONS
    config.num_slots = 1
    rollout = Rollout(
        config,
        policy=tiny_policy(dtype=torch.float32).make(),
        sampler=TorchPhiloxSampler.Config().make(),
        env=env,
        device=torch.device("cpu"),
    )
    try:
        rollout.collect(0)
    finally:
        rollout.close()
    finished = env.episodes()
    assert len(finished) == env.num_envs
    for episode in finished:
        assert replay.verify(episode) == replay.MATCHED
        frames = _replayed(episode)
        assert episode.floors == floor_trace(
            aux=frames.aux,
            reward=frames.reward,
            done=frames.done,
        )


@dataclasses.dataclass(slots=True, kw_only=True)
class _Starts:
    """Branch starts handed out in order, then none."""

    starts: list[BranchStart]

    def start(self, ordinal: int) -> BranchStart | None:
        return self.starts[ordinal] if ordinal < len(self.starts) else None


def _replayed(episode: Captured) -> Episode:
    """Return a captured episode with the frames replay regenerates of it."""
    empty = torch.empty(0)
    return replay.replay(
        Episode(
            receipt=episode.receipt,
            actions=episode.actions,
            hashes=episode.hashes,
            cells=empty,
            aux=empty,
            reward=empty,
            done=empty,
            summary=episode.summary,
            origin=episode.origin,
            truncated=episode.truncated,
        ),
    )


def _play(env: CaptureEnv, *, decisions: int, episodes: int = 0) -> list[Captured]:
    """Step every environment with NOOP until ``episodes`` end, or it finishes."""
    finished: list[Captured] = []
    for _ in range(decisions):
        env.actions[:, 0] = 0.0
        env.step_buffer(0)
        finished += env.episodes()
        if (episodes and len(finished) >= episodes) or env.finished():
            break
    return finished


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
