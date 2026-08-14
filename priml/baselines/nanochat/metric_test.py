"""Tests for the bits-per-byte metric."""

from __future__ import annotations

import math

import pytest
import torch

from priml.baselines.nanochat.metric import BitsPerByte


def _metric() -> BitsPerByte:
    return BitsPerByte.Config().make()


def test_it_converts_nats_per_byte_to_bits() -> None:
    """A known loss over a known byte count has one right answer.

    Pinned against a closed form rather than a recorded number: two tokens of
    one byte each at ln(2) nats is exactly 1 bit per byte.
    """
    metric = _metric()
    metric.update(
        torch.full((1, 2), math.log(2)),
        label=torch.tensor([[1, 1]]),
        token_bytes=torch.tensor([0, 1]),
    )
    assert metric.compute()["bpb"] == pytest.approx(1.0)


def test_a_longer_token_lowers_the_score() -> None:
    """The point of the metric: the same surprise over more bytes is cheaper.

    A per-token score would rank these two equal, which is exactly the bias
    that makes cross-entropy incomparable across tokenizers.
    """

    def score(*, token_bytes: list[int]) -> float:
        metric = _metric()
        metric.update(
            torch.full((1, 2), math.log(2)),
            label=torch.tensor([[1, 1]]),
            token_bytes=torch.tensor(token_bytes),
        )
        return metric.compute()["bpb"]

    assert score(token_bytes=[0, 4]) < score(token_bytes=[0, 1])


def test_zero_byte_tokens_leave_both_sums() -> None:
    """Document markers are formatting, not text.

    Charging the model for predicting one would score a convention, and a
    model cannot be right or wrong about where a document was cut.
    """
    metric = _metric()
    # Token 0 carries no bytes; its enormous loss must not reach the score.
    metric.update(
        torch.tensor([[math.log(2), 1e6]]),
        label=torch.tensor([[1, 0]]),
        token_bytes=torch.tensor([0, 1]),
    )
    assert metric.compute()["bpb"] == pytest.approx(1.0)


def test_counts_accumulate_across_batches() -> None:
    """The ratio is computed once at the end, not averaged per batch.

    A mean of per-batch ratios would weight a short final batch equally with
    a full one.
    """
    metric = _metric()
    token_bytes = torch.tensor([0, 1])
    metric.update(
        torch.full((1, 4), math.log(2)),
        label=torch.ones(1, 4, dtype=torch.int64),
        token_bytes=token_bytes,
    )
    metric.update(
        torch.full((1, 1), 3 * math.log(2)),
        label=torch.ones(1, 1, dtype=torch.int64),
        token_bytes=token_bytes,
    )
    # (4 + 3) bits over 5 bytes, not the mean of 1.0 and 3.0.
    assert metric.compute()["bpb"] == pytest.approx(7 / 5)


def test_a_shape_disagreement_is_rejected() -> None:
    """Silently broadcasting would pair each loss with another token's length."""
    metric = _metric()
    with pytest.raises(ValueError, match="targets"):
        metric.update(
            torch.zeros(1, 3),
            label=torch.ones(1, 2, dtype=torch.int64),
            token_bytes=torch.tensor([0, 1]),
        )


def test_an_empty_eval_refuses_rather_than_scoring_zero() -> None:
    """Zero is the BEST possible score, so an empty eval must not report it.

    Lower is better here, so a run whose eval loader yielded nothing -- a cap
    below one batch, a misconfigured split -- would otherwise win every
    comparison it entered, and look like a result rather than a failure.
    """
    with pytest.raises(ValueError, match="no scored tokens"):
        _metric().compute()


def test_state_round_trips() -> None:
    metric = _metric()
    metric.update(
        torch.full((1, 2), math.log(2)),
        label=torch.ones(1, 2, dtype=torch.int64),
        token_bytes=torch.tensor([0, 1]),
    )
    restored = _metric()
    restored.load_state_dict(metric.state_dict())
    assert restored.compute() == metric.compute()


def test_reset_clears_both_sums() -> None:
    """After a reset the metric holds nothing, so it refuses to score."""
    metric = _metric()
    metric.update(
        torch.full((1, 2), math.log(2)),
        label=torch.ones(1, 2, dtype=torch.int64),
        token_bytes=torch.tensor([0, 1]),
    )
    metric.reset()
    assert metric.state_dict() == {"nats": 0.0, "bytes": 0}
    with pytest.raises(ValueError, match="no scored tokens"):
        metric.compute()


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
