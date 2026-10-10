"""Tests for the arm factories: the design's behaviour mixture, the policies and v2."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from priml.baselines.craftax.world_model import (
    experiments as world_model_experiments,
)
from priml.baselines.craftax.world_model.capture import experiments
from priml.baselines.craftax.world_model.capture.branches import (
    BranchFeeder,
)
from priml.baselines.craftax.world_model.capture.source import (
    PolicySource,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.craftax.world_model.capture.worker import (
        CaptureWorker,
    )


ARMS = (
    experiments.arm0,
    experiments.arm1,
    experiments.arm2,
    experiments.arm3,
)
V2_FRESH = (
    experiments.arm0_v2,
    experiments.arm1_v2,
    experiments.arm2_v2,
    experiments.arm3_v2,
)
V2_BRANCH = (
    experiments.arm0_branch,
    experiments.arm1_branch,
    experiments.arm2_branch,
    experiments.arm3_branch,
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


def test_each_arm_reads_its_policy_runs_final_checkpoint() -> None:
    runs = Path("/opt/scratch/runs/craftax")
    assert [_source(_finalized(factory)).checkpoint for factory in ARMS] == [
        runs / "exp103/checkpoints/step_00038146.pt",
        runs / "exp103/checkpoints/step_00038146.pt",
        runs / "exp102/checkpoints/step_00038146.pt",
        runs / "exp000/checkpoints/step_00006662.pt",
    ]


def test_base_dir_moves_every_path_of_a_capture(tmp_path: Path) -> None:
    config = experiments.arm0_v2()
    config.base_dir = tmp_path
    pools = _source(config).branches = BranchFeeder.Config()
    pools.pools = config.root / "pools"
    config = config.copy_tree().finalize()
    source = _source(config)
    assert source.branches is not None
    archive = tmp_path / "datasets/craftax/world-model/archive-v2"
    assert [config.root, config.run_root, source.checkpoint, source.branches.pools] == [
        archive,
        tmp_path / "runs/craftax/world-model/capture",
        tmp_path / "runs/craftax/exp103/checkpoints/step_00038146.pt",
        archive / "pools",
    ]
    verifier = experiments.verifier()
    verifier.base_dir = tmp_path
    verifier = verifier.copy_tree().finalize()
    assert (verifier.root, verifier.log_dir) == (
        tmp_path / "datasets/craftax/world-model/archive-v1",
        tmp_path / "artifacts/craftax/world-model/verifier",
    )


def test_the_smoke_arm_captures_exp_smoke_into_the_world_models_smoke_corpus() -> None:
    config = _finalized(experiments.smoke)
    assert _source(config).checkpoint == Path(
        "/opt/scratch/runs/craftax/exp_smoke/checkpoints/step_00000004.pt",
    )
    reader = world_model_experiments.exp_smoke().copy_tree().finalize().dataset
    assert config.root == reader.working_dir


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


@pytest.mark.parametrize("factory", V2_BRANCH)
def test_a_v2_branch_arm_without_its_pools_raises_its_todo(
    factory: Callable[[], CaptureWorker.Config],
) -> None:
    with pytest.raises(NotImplementedError, match="TODO"):
        factory()
    assert "Not yet reproducible" in (factory.__doc__ or "")


def test_v2_fresh_budgets_are_the_tables_millions() -> None:
    fresh = [round(factory().decisions * 4 / 1_000_000) for factory in V2_FRESH]
    assert fresh == [3_406, 257, 369, 272]


def test_the_verifier_reads_the_first_archive() -> None:
    assert experiments.verifier().root == experiments.arm0().root


def test_the_e2e_arms_shrink_the_arms_into_one_root() -> None:
    arms = [
        experiments.e2e_arm0(),
        experiments.e2e_arm1(),
        experiments.e2e_arm2(),
        experiments.e2e_arm3(),
    ]
    assert [c.arm for c in arms] == [0, 1, 2, 3]
    assert [c.decisions for c in arms] == [700_000, 100_000, 100_000, 100_000]
    assert {c.root for c in arms} == {experiments.e2e_verifier().root}
    assert {_source(c).env.num_envs for c in arms} == {64}
    assert [_source(c).checkpoint for c in arms] == [
        _source(factory()).checkpoint for factory in ARMS
    ]


def _finalized(factory: Callable[[], CaptureWorker.Config]) -> CaptureWorker.Config:
    """Return a factory's config with its paths resolved."""
    return factory().copy_tree().finalize()


def _source(config: CaptureWorker.Config) -> PolicySource.Config:
    """Return a worker's policy source."""
    source = config.source
    assert isinstance(source, PolicySource.Config)
    return source


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
