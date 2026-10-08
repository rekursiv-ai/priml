"""Check the speed script's synthetic contexts are the shapes it claims."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.craftax.testing import tiny_policy
from priml.baselines.craftax.train_step import learn_joint_minibatch
from priml.baselines.craftax.world_model.context import (
    ContextReplay,
    JointWorldModel,
    plan_replay,
)
from priml.baselines.craftax.world_model.scripts.joint_speed import (
    _rollout,
    synthetic_contexts,
)
from priml.baselines.craftax.world_model.testing import (
    random_contexts,
    small_schema,
    tiny_model,
)
from priml.loss.policy_gradient import TorchPPO
from priml.model.linear import Linear


def test_refill_contexts_cycle_from_256_kept_decisions_to_512() -> None:
    contexts = synthetic_contexts(
        "refill",
        agents=3,
        horizon=256,
        generator=torch.Generator().manual_seed(0),
    )
    lengths = contexts.lengths
    assert int(lengths.min()) == 257
    assert int(lengths.max()) == 512
    assert float(lengths.float().mean()) == (257 + 512) / 2
    assert not bool(contexts.anchored.any())
    # Each row restarts once in 256 steps: two passes.
    plan = plan_replay(contexts, bin_tokens=4_096, pass_tokens=16_384)
    assert plan.passes == 2 * 3


def test_sliding_contexts_begin_their_episodes_and_never_fill() -> None:
    contexts = synthetic_contexts(
        "sliding",
        agents=5,
        horizon=256,
        generator=torch.Generator().manual_seed(1),
    )
    assert bool(contexts.anchored.all())
    assert int(contexts.lengths.max()) <= 512
    assert int(contexts.prefix_counts.max()) < 511
    # Rows 0 and 4 restart their episodes once; the others are one pass each.
    plan = plan_replay(contexts, bin_tokens=4_096, pass_tokens=16_384)
    assert plan.passes == 5 + 2


@pytest.mark.parametrize("shape", ["refill", "sliding"])
def test_every_frame_token_is_in_its_fields_range(shape: str) -> None:
    contexts = synthetic_contexts(
        shape,
        agents=2,
        horizon=256,
        generator=torch.Generator().manual_seed(2),
    )
    assert contexts.cells.dtype == torch.uint8
    assert contexts.aux.dtype == torch.int16
    assert contexts.prefix_cells.shape == (2, 511, 99, 8)


@pytest.mark.compute_training
def test_the_synthetic_rollout_learns_through_the_joint_minibatch() -> None:
    """Every field the joint learner reads, the stored features included, is there."""
    config = tiny_policy(dtype=torch.float32)
    proj = config.proj_feature = Linear.Config()
    proj.channels_in = 36  # The tiny world model's width.
    policy = config.make()
    model = tiny_model(small_schema(), global_layers=2)
    contexts = random_contexts(
        torch.tensor(((1, 2, 3), (2, 3, 4))),
        torch.ones(2, 3, dtype=torch.bool),
        schema=small_schema(),
        slots=4,
        counts=torch.tensor((0, 1)),
        seed=1,
    )
    rollout = _rollout(
        policy,
        observation_size=config.observation_size,
        contexts=contexts,
        horizon=3,
        width=36,
        generator=torch.Generator().manual_seed(2),
    )
    replay = ContextReplay.Config()
    replay.bin_tokens = 1
    joint = JointWorldModel(model, layers=2, replay=replay.make())
    losses, _, gap = learn_joint_minibatch(
        policy,
        TorchPPO(TorchPPO.Config()),
        rollout,
        joint,
    )
    assert bool(losses.isfinite().all())
    assert bool(gap.isfinite())
    assert all(leaf.grad is not None for leaf in joint.parameters())


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
