"""Tests for the arm factories: the design's behaviour mixture, the policies and v2."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from priml.baselines.craftax.world_model.capture import experiments
from priml.baselines.craftax.world_model.capture.source import (
    PolicySource,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.craftax.world_model.capture.worker import (
        CaptureWorker,
    )


ARMS = (
    experiments.exp103_s74,
    experiments.exp103_s74_epsilon,
    experiments.exp102_s73,
    experiments.exp000_s73,
)
V2_FRESH = (
    experiments.exp103_s74_v2,
    experiments.exp103_s74_epsilon_v2,
    experiments.exp102_s73_v2,
    experiments.exp000_s73_v2,
)
V2_BRANCH = (
    experiments.exp103_s74_branch,
    experiments.exp103_s74_epsilon_branch,
    experiments.exp102_s73_branch,
    experiments.exp000_s73_branch,
)


def test_arms_follow_the_plan_order_and_shares() -> None:
    configs = [factory() for factory in ARMS]
    assert [config.arm for config in configs] == [0, 1, 2, 3]
    per_worker = [config.decisions for config in configs]
    assert sum(per_worker) * 4 == 3_500_000_000
    assert [d * 4 * 100 // 3_500_000_000 for d in per_worker] == [70, 10, 10, 10]
    assert [_source(c).env.epsilon for c in configs] == [0.0, 0.05, 0.0, 0.0]


def test_arms_read_their_policys_observation_layout() -> None:
    sources = [_source(factory()) for factory in ARMS]
    assert [s.policy.env.rules.previous_action for s in sources] == [
        True,
        True,
        True,
        False,
    ]


@pytest.mark.parametrize("factory", ARMS)
def test_every_arm_refuses_weights_of_another_sha256(
    factory: Callable[[], CaptureWorker.Config],
    tmp_path: Path,
) -> None:
    source = _source(factory())
    assert source.checkpoint is not None
    assert source.checkpoint.parent == Path(
        "/opt/scratch/artifacts/craftax/world-model/behaviour-policies",
    )
    source.checkpoint = tmp_path / "policy.pt"
    source.checkpoint.write_bytes(b"not the arm's weights")
    with pytest.raises(ValueError, match="wrong SHA-256"):
        source.make()


def test_v2_fresh_arms_are_generation_1_and_stall_capped() -> None:
    configs = [factory() for factory in V2_FRESH]
    assert [c.arm for c in configs] == [0, 1, 2, 3]
    assert {(c.generation, c.root.name) for c in configs} == {(1, "archive-v2")}
    envs = [_source(c).env for c in configs]
    assert [(e.epsilon, e.epsilon_high) for e in envs] == [
        (0.0, 0.0),
        (0.04, 0.06),
        (0.0, 0.0),
        (0.0, 0.0),
    ]
    assert [e.stall_limit for e in envs] == [2_000, 2_000, 10_000, 10_000]
    assert all(_source(c).branches is None for c in configs)


def test_v2_branch_arms_are_generation_2_with_their_pools() -> None:
    for arm, (branch, fresh) in enumerate(zip(V2_BRANCH, V2_FRESH, strict=True)):
        config = branch()
        assert (config.arm, config.generation) == (arm, 2)
        source = _source(config)
        assert source.branches is not None
        assert source.branches.pools == config.root / "pools"
        assert source.env == _source(fresh()).env


def test_v2_budgets_total_6_66_billion() -> None:
    fresh = [factory().decisions * 4 for factory in V2_FRESH]
    branch = [factory().decisions * 4 for factory in V2_BRANCH]
    assert abs(sum(fresh) + sum(branch) - 6_660_000_000) < 4 * 4


def test_the_verifier_reads_the_first_archive() -> None:
    assert experiments.verifier().root == experiments.exp103_s74().root


def test_the_e2e_arms_shrink_the_arms_into_one_root() -> None:
    arms = [
        experiments.e2e_exp103_s74(),
        experiments.e2e_exp103_s74_epsilon(),
        experiments.e2e_exp102_s73(),
        experiments.e2e_exp000_s73(),
    ]
    assert [c.arm for c in arms] == [0, 1, 2, 3]
    assert [c.decisions for c in arms] == [700_000, 100_000, 100_000, 100_000]
    assert {c.root for c in arms} == {experiments.e2e_verifier().root}
    assert {_source(c).env.num_envs for c in arms} == {64}
    assert [_source(c).checkpoint for c in arms] == [
        _source(factory()).checkpoint for factory in ARMS
    ]


def _source(config: CaptureWorker.Config) -> PolicySource.Config:
    """Return a worker's policy source."""
    source = config.source
    assert isinstance(source, PolicySource.Config)
    return source


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
