"""Tests for the capture seed schedule: world seeds, the split, and sampler seeds."""

from __future__ import annotations

import pytest

from priml.baselines.craftax.world_model.capture.seeds import (
    TRAIN,
    VALIDATION,
    episode_seed,
    rollout_seed,
    sampling_seed,
)


def test_first_episodes_follow_the_plan_formula() -> None:
    assert episode_seed(0, arm=0, worker=0, generation=0) == (TRAIN, 100_000_000)
    assert episode_seed(1, arm=0, worker=0, generation=0) == (TRAIN, 100_000_001)
    assert episode_seed(0, arm=2, worker=3, generation=0) == (
        TRAIN,
        100_000_000 + 80_000_000 + 30_000_000,
    )


def test_every_twentieth_started_episode_is_validation() -> None:
    splits = [episode_seed(n, arm=1, worker=2, generation=0)[0] for n in range(60)]
    assert [n for n, split in enumerate(splits) if split == VALIDATION] == [19, 39, 59]
    assert episode_seed(19, arm=1, worker=2, generation=0) == (
        VALIDATION,
        300_000_000 + 40_000_000 + 20_000_000,
    )
    assert episode_seed(39, arm=1, worker=2, generation=0)[1] == 360_000_001


def test_indices_count_within_each_split() -> None:
    assert episode_seed(18, arm=0, worker=0, generation=0)[1] == 100_000_018
    assert episode_seed(20, arm=0, worker=0, generation=0)[1] == 100_000_019
    assert episode_seed(40, arm=0, worker=0, generation=0)[1] == 100_000_038


def test_seeds_are_disjoint_across_arms_workers_and_splits() -> None:
    seeds = {
        episode_seed(n, arm=arm, worker=worker, generation=0)[1]
        for arm in range(4)
        for worker in range(4)
        for n in range(40)
    }
    assert len(seeds) == 4 * 4 * 40


def test_generation_ranges_are_disjoint_and_fit_the_games_seed() -> None:
    last = 10_000_000 * 20 - 1
    for generation in range(10):
        low = episode_seed(0, arm=0, worker=0, generation=generation)[1]
        high = episode_seed(last, arm=3, worker=3, generation=generation)[1]
        assert low == 100_000_000 + 400_000_000 * generation
        assert high == 459_999_999 + 400_000_000 * generation
        assert high < 1 << 32


@pytest.mark.parametrize(
    ("ordinal", "arm", "worker", "generation"),
    [
        (-1, 0, 0, 0),
        (0, 4, 0, 0),
        (0, 0, 4, 0),
        (10_000_000 * 20, 0, 0, 0),
        (0, 0, 0, 10),
    ],
)
def test_out_of_range_arguments_are_rejected(
    ordinal: int,
    arm: int,
    worker: int,
    generation: int,
) -> None:
    with pytest.raises(ValueError, match="range"):
        episode_seed(ordinal, arm=arm, worker=worker, generation=generation)


def test_sampling_seed_packs_its_fields() -> None:
    seed = sampling_seed(5, arm=3, worker=2, environment=7, generation=0)
    assert seed == 3 << 62 | 2 << 60 | 7 << 40 | 5
    assert sampling_seed(5, arm=3, worker=2, environment=7, generation=9) == (
        seed | 9 << 36
    )


@pytest.mark.parametrize(
    ("ordinal", "environment", "generation"),
    [(-1, 0, 0), (1 << 36, 0, 0), (0, 1 << 20, 0), (0, 0, 10)],
)
def test_sampling_seed_rejects_out_of_range_fields(
    ordinal: int,
    environment: int,
    generation: int,
) -> None:
    with pytest.raises(ValueError, match="range"):
        sampling_seed(
            ordinal,
            arm=0,
            worker=0,
            environment=environment,
            generation=generation,
        )


def test_rollout_seeds_keep_every_buffer_stream_distinct() -> None:
    seeds = [
        rollout_seed(arm=a, worker=w, base=73, generation=g)
        for a in range(4)
        for w in range(4)
        for g in range(10)
    ]
    assert seeds[0] == 73
    buffers = {seed + buffer for seed in seeds for buffer in range(64)}
    assert len(buffers) == 16 * 10 * 64


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
