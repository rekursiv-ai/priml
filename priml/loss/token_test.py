"""Tests for tokenwise losses and pool reductions."""

from __future__ import annotations

import pytest
import torch

from priml.loss.token import (
    CrossEntropyTokens,
    MeanOverActive,
    MeanOverBatch,
    StablemaxTokens,
)
from priml.math.loss import log_stablemax_via_probabilities


def test_token_losses_and_reductions() -> None:
    """Keep token loss and active-slot reduction behavior unchanged."""
    logits = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0], [2.0, 0.0, 1.0, 3.0]],
            [[0.0, 1.0, 2.0, 4.0], [4.0, 2.0, 1.0, 0.0], [1.0, 3.0, 0.0, 2.0]],
        ],
    )
    labels = torch.tensor([[0, 1, -1], [2, 3, 1]])
    cross = CrossEntropyTokens.Config(label_smoothing=0.1).make()(
        logits,
        labels,
        ignore_index=-1,
    )
    unsmoothed = CrossEntropyTokens.Config().make()(
        logits,
        labels,
        ignore_index=-1,
    )
    stable = StablemaxTokens.Config().make()(logits, labels, ignore_index=-1)

    assert cross.shape == labels.shape
    assert not torch.equal(cross, unsmoothed)
    assert stable.dtype == torch.float64
    active = torch.tensor([True, False])
    values = torch.tensor([2.0, 4.0])
    assert MeanOverActive.Config().make()(values, active=active) == 2
    assert (
        MeanOverActive.Config().make()(
            values,
            active=torch.tensor([True, True]),
        )
        == 3
    )
    assert (
        MeanOverActive.Config().make()(values, active=torch.zeros(2, dtype=torch.bool))
        == 0
    )
    assert MeanOverBatch.Config(batch_size=2).make()(values, active=active) == 1
    assert MeanOverBatch.Config(batch_size=1).make()(values, active=active) == 2
    with pytest.raises(ValueError, match="batch_size must be positive") as error:
        MeanOverBatch.Config(batch_size=0).make()
    assert str(error.value) == (
        "batch_size must be positive; got 0. It is normally inherited from "
        "the pool during finalize."
    )


def test_stablemax_tokens_inject_source_probability_normalizer() -> None:
    """Select the source probability-normalize-then-log arithmetic explicitly."""
    logits = torch.tensor(
        [
            [[-2.0, 0.5, 3.0, -0.25], [1.0, -4.0, 0.25, 2.0], [0.0, 2.0, -1.0, 0.5]],
            [[3.0, -2.0, 0.25, 1.5], [-0.5, 2.0, 1.0, -3.0], [1.25, 0.5, -2.5, 3.5]],
        ],
    )
    labels = torch.tensor([[2, 0, -100], [0, 1, 3]])
    values = logits.double()
    stable = torch.where(
        values < 0,
        1 / (1 - values + 1e-30),
        values + 1,
    )
    log_probabilities = (stable / stable.sum(dim=-1, keepdim=True)).log()
    counted = labels != -100
    selected = log_probabilities.gather(
        -1,
        labels.masked_fill(~counted, 0).unsqueeze(-1),
    ).squeeze(-1)
    expected = -torch.where(counted, selected, 0.0)
    loss = StablemaxTokens.Config(
        log_normalizer=log_stablemax_via_probabilities,
    ).make()(logits, labels, ignore_index=-100)

    assert loss.dtype == torch.float64
    assert torch.equal(loss, expected)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
