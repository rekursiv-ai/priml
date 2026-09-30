"""Unit tests for ARC1 ACT exploration and feedback branches."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.arcagi1.act import (
    FeedbackCarry,
    ForcedContinue,
    SampledMinimum,
)


def test_exploration_and_feedback_validation() -> None:
    fired = torch.tensor([True, True])
    steps = torch.tensor([1, 3])
    generator = torch.Generator().manual_seed(0)
    assert SampledMinimum.Config(prob=0).make()(
        fired,
        steps=steps,
        max_steps=3,
        generator=generator,
    ).tolist() == [True, True]
    assert ForcedContinue.Config(prob=1).make()(
        fired,
        steps=steps,
        max_steps=3,
        generator=generator,
    ).tolist() == [False, False]
    carry = FeedbackCarry.Config(corruption_rate=0).make()
    grid = torch.full((2, 3), 4)
    assert torch.equal(carry.corrupt(grid), grid)
    with pytest.raises(ValueError, match="corruption_rate"):
        FeedbackCarry.Config(corruption_rate=2).make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
