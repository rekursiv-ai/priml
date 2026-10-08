"""Check the joint smoke's config: a joint arm's step, placed on CUDA."""

from __future__ import annotations

import pytest

from priml.baselines.craftax.scripts.joint_smoke import (
    ARMS,
    smoke_config,
)
from priml.baselines.craftax.world_model.context import ContextReplay
from priml.baselines.craftax.world_model.feature import (
    DonorHistory,
    FreshWindow,
    Refill,
    Sliding,
    WorldModelFeature,
)


@pytest.mark.parametrize("experiment", sorted(ARMS))
def test_the_smoke_drives_a_joint_arm_on_cuda(experiment: str) -> None:
    config = smoke_config(experiment, slots=1, episodes=64)
    final = config.copy_tree().finalize()
    assert final.parallelism.device == "cuda"
    assert final.rollout.num_slots == 1
    assert final.evaluation.num_episodes == 64
    assert final.env.practice is not None
    assert isinstance(final.feature, WorldModelFeature.Config)
    assert final.feature.joint
    assert isinstance(final.feature_training, ContextReplay.Config)


@pytest.mark.parametrize("experiment", sorted(ARMS))
def test_the_smoke_reads_exact_windows_with_donor_histories_on_request(
    experiment: str,
) -> None:
    refill, exact = (
        smoke_config(experiment, slots=2, episodes=0, sliding=sliding)
        .copy_tree()
        .finalize()
        .feature
        for sliding in (False, True)
    )
    assert isinstance(refill, WorldModelFeature.Config)
    assert isinstance(refill.history, Refill.Config)
    assert isinstance(refill.practice, FreshWindow.Config)
    assert isinstance(exact, WorldModelFeature.Config)
    assert exact.joint
    assert isinstance(exact.history, Sliding.Config)
    assert exact.history.decisions == 512
    assert isinstance(exact.practice, DonorHistory.Config)
    assert (refill.hook_interval, exact.hook_interval) == (128, 32)


def test_the_smoke_keeps_the_arms_slots_and_evaluation_by_default() -> None:
    final = smoke_config("exp112", slots=2, episodes=0).copy_tree().finalize()
    assert final.rollout.num_slots == 2
    assert final.evaluation.num_episodes == 10_000


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
