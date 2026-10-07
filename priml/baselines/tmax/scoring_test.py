"""Tests for Qwen3.5 teacher-forced packed scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.utils.checkpoint

from priml.baselines.tmax.scoring import (
    _projection_weight,
    response_logprobs,
    score_hidden_labels,
)
from priml.baselines.tmax.train_step import tiny_qwen35_config


if TYPE_CHECKING:
    from collections.abc import Callable


def test_each_packed_segment_is_scored_with_a_fresh_hybrid_state() -> None:
    """Score each packed segment with fresh hybrid-model state."""
    torch.manual_seed(7)
    model = tiny_qwen35_config().make().eval()
    tokens = torch.tensor([1, 2, 3, 4, 5, 6])
    segments = torch.tensor([1, 1, 1, 2, 2, 2])
    positions = torch.tensor([0, 1, 2, 0, 1, 2])
    actual = response_logprobs(
        model,
        tokens=tokens,
        segments=segments,
        positions=positions,
        pad_token_id=0,
    )
    first = response_logprobs(
        model,
        tokens=tokens[:3],
        segments=segments[:3],
        positions=positions[:3],
        pad_token_id=0,
    )
    second = response_logprobs(
        model,
        tokens=tokens[3:],
        segments=segments[3:],
        positions=positions[3:],
        pad_token_id=0,
    )
    torch.testing.assert_close(actual[0, :2], first[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[0, 3:5], second[0], rtol=0.0, atol=0.0)
    assert torch.isnan(actual[0, 2])


def test_scoring_rejects_non_positive_temperature() -> None:
    """Reject a non-positive temperature before dividing logits."""
    model = tiny_qwen35_config().make()
    with pytest.raises(ValueError, match="positive"):
        response_logprobs(
            model,
            tokens=torch.tensor([1, 2]),
            segments=torch.ones(2, dtype=torch.long),
            positions=torch.arange(2),
            pad_token_id=0,
            temperature=0.0,
        )


def test_scoring_rejects_a_row_with_fewer_than_two_tokens() -> None:
    """Require one token to condition on and another to score."""
    model = tiny_qwen35_config().make()
    with pytest.raises(ValueError, match="at least two tokens"):
        response_logprobs(
            model,
            tokens=torch.tensor([1]),
            segments=torch.ones(1, dtype=torch.long),
            positions=torch.arange(1),
            pad_token_id=0,
        )


def test_a_one_token_segment_is_left_unscored() -> None:
    """A trajectory with no next token of its own has nothing to score."""
    torch.manual_seed(7)
    model = tiny_qwen35_config().make().eval()
    actual = response_logprobs(
        model,
        tokens=torch.tensor([1, 2, 3]),
        segments=torch.tensor([1, 1, 2]),
        positions=torch.tensor([0, 1, 0]),
        pad_token_id=0,
    )
    assert torch.isfinite(actual[0, 0])
    assert torch.isnan(actual[0, 1])


def test_a_non_unit_temperature_rescales_the_logits() -> None:
    """Apply a non-unit temperature to the logits."""
    torch.manual_seed(7)
    model = tiny_qwen35_config().make().eval()
    tokens = torch.tensor([1, 2, 3])
    segments = torch.ones(3, dtype=torch.long)
    positions = torch.arange(3)
    warm = response_logprobs(
        model,
        tokens=tokens,
        segments=segments,
        positions=positions,
        pad_token_id=0,
    )
    cold = response_logprobs(
        model,
        tokens=tokens,
        segments=segments,
        positions=positions,
        pad_token_id=0,
        temperature=2.0,
    )
    assert not torch.allclose(warm, cold)


def test_chunked_head_matches_full_logits_scoring() -> None:
    """Match full-logit scores when projecting the head in chunks."""
    torch.manual_seed(19)
    full = tiny_qwen35_config(vocab_size=64).make().eval()
    tokens = torch.arange(1, 13)
    segments = torch.tensor([1] * 6 + [2] * 6)
    positions = torch.tensor([0, 1, 2, 3, 4, 5] * 2)
    expected = response_logprobs(
        full,
        tokens=tokens,
        segments=segments,
        positions=positions,
        pad_token_id=0,
        temperature=1.5,
    )
    actual = response_logprobs(
        full,
        tokens=tokens,
        segments=segments,
        positions=positions,
        pad_token_id=0,
        temperature=1.5,
        head_chunk_size=2,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)


def test_chunked_head_backpropagates_into_the_registered_model() -> None:
    """Backpropagate through every parameter on the chunked fp32 path."""
    torch.manual_seed(31)
    model = tiny_qwen35_config(vocab_size=64).make().train()
    scores = response_logprobs(
        model,
        tokens=torch.arange(1, 9),
        segments=torch.ones(8, dtype=torch.long),
        positions=torch.arange(8),
        pad_token_id=0,
        head_chunk_size=2,
        fp32_head=True,
    )
    scores.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_each_head_chunk_has_its_own_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply activation checkpointing to each output-head chunk."""
    calls: list[int] = []

    def checkpoint(
        function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        hidden: torch.Tensor,
        labels: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        """Record one checkpointed chunk and evaluate it."""
        calls.append(hidden.shape[1])
        return function(hidden, labels)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", checkpoint)
    head = torch.nn.Linear(4, 8, bias=False)
    score_hidden_labels(
        head,
        torch.randn(2, 6, 4),
        torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]),
        temperature=1.0,
        chunk_size=2,
        fp32_head=False,
    )
    assert calls == [2, 2, 1]


def test_chunked_head_rejects_non_positive_chunk_size() -> None:
    """Reject a non-positive output-head chunk size."""
    model = tiny_qwen35_config().make()
    assert model.proj_out is not None
    with pytest.raises(ValueError, match="positive"):
        score_hidden_labels(
            model.proj_out,
            torch.randn(2, 3, 16),
            torch.tensor([[1, 2], [2, 3]]),
            temperature=1.0,
            chunk_size=0,
            fp32_head=False,
        )


def test_the_fp32_head_casts_its_operands_before_the_matmul() -> None:
    """Upstream's ``--lm_head_fp32`` casts both operands before the matmul.

    The released recipe patches the head so the hidden states AND the weight
    are fp32 when the projection runs. Casting the logits afterwards would
    keep the softmax reduction in fp32 while every product had already been
    rounded to bfloat16 -- the drift the TMax paper's Figure 4 shows between
    the trainer's and the engines' logprobs.
    """
    torch.manual_seed(23)
    model = tiny_qwen35_config(vocab_size=64).make().eval()
    head = model.proj_out
    assert head is not None
    weight = _projection_weight(head)
    assert weight is not None
    hidden = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    labels = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        scores = score_hidden_labels(
            head,
            hidden,
            labels,
            temperature=1.0,
            chunk_size=None,
            fp32_head=True,
        )
    logits = hidden[:, :-1].float() @ weight.float()
    expected = torch.gather(
        logits,
        -1,
        labels.unsqueeze(-1),
    ).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    assert scores.dtype == torch.float32
    torch.testing.assert_close(
        scores,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_the_fp32_head_projects_a_tied_head_through_the_embedding() -> None:
    """The released 4B checkpoint ties its head, so the patch must borrow too."""
    torch.manual_seed(29)
    model = tiny_qwen35_config(vocab_size=64, tie_word_embeddings=True).make().eval()
    head = model.proj_out
    assert head is not None
    # The tied head borrows the embedding's weight, so the expected
    # projection reads it through the same borrowing.
    weight = _projection_weight(head)
    assert weight is not None
    hidden = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    labels = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        scores = score_hidden_labels(
            head,
            hidden,
            labels,
            temperature=1.0,
            chunk_size=None,
            fp32_head=True,
        )
    logits = hidden[:, :-1].float() @ weight.float()
    expected = torch.gather(
        logits,
        -1,
        labels.unsqueeze(-1),
    ).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    assert scores.dtype == torch.float32
    torch.testing.assert_close(
        scores,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_the_fp32_head_rejects_a_model_without_a_head() -> None:
    """Require a projection weight for fp32 output-head scoring."""
    with pytest.raises(TypeError, match="projection weight"):
        score_hidden_labels(
            torch.nn.Identity(),
            torch.randn(2, 3, 4),
            torch.tensor([[1, 2], [2, 1]]),
            temperature=1.0,
            chunk_size=None,
            fp32_head=True,
        )


def test_scoring_rejects_two_dimensional_segments() -> None:
    """Require segment ids to be a flat per-token vector."""
    model = tiny_qwen35_config().make()
    with pytest.raises(TypeError, match="one-dimensional"):
        response_logprobs(
            model,
            tokens=torch.tensor([1, 2]),
            # The invalid leading axis is the degenerate input under test.
            segments=torch.ones(1, 2, dtype=torch.long),
            positions=torch.arange(2),
            pad_token_id=0,
        )


def test_scoring_rejects_unordered_segments() -> None:
    """Reject segment ids that decrease within a packed row."""
    model = tiny_qwen35_config().make()
    with pytest.raises(ValueError, match="non-decreasing"):
        response_logprobs(
            model,
            tokens=torch.tensor([1, 2]),
            segments=torch.tensor([2, 1]),
            positions=torch.arange(2),
            pad_token_id=0,
        )
