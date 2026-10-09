"""Check the speed script's synthetic contexts are the shapes it claims."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.craftax.testing import tiny_policy
from priml.baselines.craftax.world_model.context import plan_replay
from priml.baselines.craftax.world_model.scripts.joint_speed import (
    _rollout,
    synthetic_contexts,
)
from priml.baselines.craftax.world_model.testing import (
    random_contexts,
    small_schema,
)


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


def test_the_synthetic_rollout_is_in_the_joint_learners_layout() -> None:
    """Every field the joint learner reads, the stored features and contexts included."""
    config = tiny_policy(dtype=torch.float32)
    policy = config.make()
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
    # ``LearnerRollout``'s layout: agent-major, the policy's dtype, 43 actions.
    observations, features = rollout.observations, rollout.features
    assert observations.shape == (2, 3, config.observation_size)
    assert features is not None
    assert features.shape == (2, 3, 36)
    assert observations.dtype == features.dtype == policy.dtype
    for steps in (rollout.logprobs, rollout.rewards, rollout.terminals, rollout.values):
        assert steps.shape == (2, 3)
    assert rollout.action_mask.shape == (2, 3, 43)
    actions = rollout.actions
    assert torch.equal(actions, actions.floor())
    assert 0 <= int(actions.min()) <= int(actions.max()) < 43
    assert bool((rollout.logprobs <= 0).all())
    assert torch.equal(
        rollout.initial_states,
        policy.initial_state(2, device=torch.device("cpu")),
    )
    assert (rollout.branch_starts.dtype, rollout.branch_starts.shape) == (
        torch.uint8,
        (2,),
    )
    assert rollout.contexts is contexts


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
