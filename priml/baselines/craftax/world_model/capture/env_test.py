"""Tests for the capture environment: its records, their replay, and the v2 settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dataclasses

import pytest
import torch

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
    from pathlib import Path


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


def test_the_previous_action_layout_reads_none_after_a_reset() -> None:
    config = CaptureEnv.Config(num_envs=2)
    config.rules.previous_action = True
    env = CaptureEnv(config, schedule=Schedule(budget=1_000))
    assert env.observations.shape == (2, 844)
    assert env.observations[:, -1].tolist() == [43.0, 43.0]
    env.actions[:, 0] = torch.tensor([5.0, 7.0])
    env.step_buffer(0)
    assert env.observations[:, -1].tolist() == [5.0, 7.0]


@pytest.mark.parametrize("action", [-1.0, 43.0, float("nan")])
def test_an_action_outside_the_schema_stops_recording(action: float) -> None:
    env = CaptureEnv(CaptureEnv.Config(num_envs=2), schedule=Schedule(budget=1_000))
    env.actions[0, 0] = action
    env.step_buffer(0)
    assert "outside 0-42" in env.failure
    assert env.receipt(0) is None
    assert env.in_flight() == 1
    _play(env, decisions=1_000)
    assert env.finished()


def test_random_play_records_what_replay_record_records() -> None:
    # Each action drawn among the legal ones by splitmix64 from the episode's
    # sampling seed, as replay.record draws; replay.record equals the reference
    # recordings (the parity suite checks it), so equal records here make this
    # capture the reference's. A chunk of 256 is moved many times an episode.
    env = CaptureEnv(
        CaptureEnv.Config(num_envs=3, chunk_decisions=256),
        schedule=Schedule(budget=10**9),
    )
    streams = [0, 0, 0]
    finished: list[Captured] = []
    while len(finished) < 8:
        for e in range(3):
            receipt = env.receipt(e)
            assert receipt is not None
            if not env.decisions(e):
                streams[e] = receipt.sampling_seed
            streams[e], draw = splitmix64(streams[e])
            legal = env.action_mask[e].nonzero().flatten()
            env.actions[e, 0] = float(legal[draw % len(legal)])
        env.step_buffer(0)
        finished += env.episodes()
    assert max(len(e.actions) for e in finished) > 256
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


def test_the_summary_counts_floors_and_achievements_from_the_episode() -> None:
    env = CaptureEnv(CaptureEnv.Config(num_envs=2), schedule=Schedule(budget=10**9))
    (episode, *_) = _play(env, decisions=10_000, episodes=1)
    frames = _replayed(episode)
    summary = episode.summary
    floors = from_plain(summary["floors"], list[dict[str, object]])
    decisions = [from_plain(f["decisions"], int) for f in floors]
    visited = frames.aux[:, FLOOR_AUX].bincount(minlength=9).tolist()
    assert decisions == visited
    assert from_plain(floors[0]["reached"], int) == 1
    assert from_plain(summary["episode"], int) == 0
    died = bool(frames.done[-1]) and int(frames.reward[-1]) == -1
    assert from_plain(summary["death"], int) == int(died)
    assert episode.floors.died == died
    assert "epsilon" not in summary
    assert "truncated" not in summary
    assert list(summary)[:5] == ["episode", "environment", "return", "death", "timeout"]


def test_epsilon_one_replaces_every_action_by_a_legal_draw() -> None:
    config = CaptureEnv.Config(num_envs=2, epsilon=1.0)
    episodes = _play(
        CaptureEnv(config, schedule=Schedule(budget=10**9)),
        decisions=10_000,
        episodes=2,
    )
    for episode in episodes:
        # The policy plays NOOP only; the override draws every action.
        assert int((episode.actions != 0).sum()) > len(episode.actions) // 2
        assert replay.verify(episode) == replay.MATCHED
    again = _play(
        CaptureEnv(config, schedule=Schedule(budget=10**9)),
        decisions=10_000,
        episodes=2,
    )
    assert torch.equal(episodes[0].actions, again[0].actions)


def test_an_epsilon_range_draws_each_episodes_epsilon_into_its_summary() -> None:
    config = CaptureEnv.Config(num_envs=2, epsilon=0.25, epsilon_high=0.75)
    episodes = _play(
        CaptureEnv(config, schedule=Schedule(budget=10**9)),
        decisions=10_000,
        episodes=2,
    )
    drawn = [from_plain(e.summary["epsilon"], float) for e in episodes]
    assert all(0.25 <= value <= 0.75 for value in drawn)
    assert drawn[0] != drawn[1]


def test_the_stall_cap_truncates_training_episodes_that_replay() -> None:
    config = CaptureEnv.Config(num_envs=2, stall_limit=40)
    env = CaptureEnv(config, schedule=Schedule(budget=10**9))
    (episode, *_) = _play(env, decisions=200, episodes=1)
    frames = _replayed(episode)
    assert episode.truncated
    assert not bool(frames.done[-1])
    assert episode.summary["truncated"] == 1
    assert episode.receipt.split == TRAIN
    assert len(episode.actions) >= 40
    assert bool((frames.reward[-40:] <= 0).all())
    # The environment ends its next, unrecorded, decision at the timeout, and
    # records the next world.
    row = from_plain(episode.summary["environment"], int)
    assert env.receipt(row) is None
    env.step_buffer(0)
    assert float(env.terminals[row]) == 1.0
    assert env.receipt(row) is not None


def test_a_budget_drains_the_capture() -> None:
    env = CaptureEnv(CaptureEnv.Config(num_envs=3), schedule=Schedule(budget=1))
    episodes = _play(env, decisions=10_000)
    assert env.finished()
    ordinals = {from_plain(e.summary["episode"], int) for e in episodes}
    assert ordinals == {0, 1, 2}


def test_without_fixed_worlds_every_episode_plays_its_own_seed() -> None:
    env = CaptureEnv(
        CaptureEnv.Config(num_envs=3),
        schedule=Schedule(arm=2, worker=1, generation=3, budget=1),
    )
    episodes = _play(env, decisions=10_000)
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


# Random play on three rows over two worlds, one decision a poll, until 1,000
# decisions have ended, so rows reset into their next episodes mid-capture.
def test_fixed_worlds_start_every_episode_from_its_worlds_state_and_stream(
    tmp_path: Path,
) -> None:
    worlds = (3, 11)
    config = RandomSource.Config(steps_per_poll=1)
    config.env.num_envs = 3
    config.env.world_seeds = worlds
    source = config.make()
    source.start(schedule=Schedule(budget=1_000), run_dir=tmp_path)
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
    parent = replay.record(world_seed=7, sampling_seed=11, max_decisions=100_000)
    decision = len(parent.actions) // 2
    origin = replay.origin(parent, decision=decision)
    starts = _Starts(
        starts=[BranchStart(world_seed=7, origin=origin, point={"decision": decision})],
    )
    env = CaptureEnv(
        CaptureEnv.Config(num_envs=2),
        schedule=Schedule(budget=10**9, generation=2),
        branches=starts,
    )
    assert env.receipt(1) is None
    (branch,) = _play(env, decisions=100_000)
    assert env.finished()
    assert branch.origin == origin
    assert branch.receipt.world_seed == 7
    assert branch.summary["branch"] == {"decision": decision}
    assert replay.verify(branch) == replay.MATCHED


# Three whole episodes of the policy's rollout, 64-step horizons until they
# end: 0.38 s warm on x86.
@pytest.mark.compute_large_fixture
def test_the_ports_rollout_captures_episodes_that_replay() -> None:
    env = CaptureEnv(
        CaptureEnv.Config(num_envs=4, num_buffers=2),
        schedule=Schedule(budget=10**9),
    )
    torch.manual_seed(0)
    config = Rollout.Config()
    config.horizon = 64
    config.num_slots = 1
    rollout = Rollout(
        config,
        policy=tiny_policy().make(),
        sampler=TorchPhiloxSampler.Config().make(),
        env=env,
        device=torch.device("cpu"),
    )
    finished: list[Captured] = []
    try:
        while len(finished) < 3:
            rollout.collect(0)
            finished += env.episodes()
    finally:
        rollout.close()
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
