"""Unit tests for ARC1 losses and reductions."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.arcagi1.loss import (
    CrossEntropyTokens,
    MeanOverActive,
    MeanOverBatch,
    StablemaxTokens,
)


def test_token_losses_and_reductions() -> None:
    logits = torch.zeros(2, 3, 4)
    labels = torch.tensor([[0, 1, -1], [2, 3, 1]])
    cross = CrossEntropyTokens.Config(label_smoothing=0.1).make()(
        logits,
        labels,
        ignore_index=-1,
    )
    stable = StablemaxTokens.Config().make()(logits, labels, ignore_index=-1)
    assert cross.shape == labels.shape
    assert stable.dtype == torch.float64
    active = torch.tensor([True, False])
    values = torch.tensor([2.0, 4.0])
    assert MeanOverActive.Config().make()(values, active=active) == 2
    assert (
        MeanOverActive.Config().make()(values, active=torch.zeros(2, dtype=torch.bool))
        == 0
    )
    assert MeanOverBatch.Config(batch_size=2).make()(values, active=active) == 1
    with pytest.raises(ValueError, match="batch_size must be positive"):
        MeanOverBatch.Config(batch_size=0).make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
