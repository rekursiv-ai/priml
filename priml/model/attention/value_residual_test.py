"""Tests for value-residual attention."""

from __future__ import annotations

import pytest
import torch

from priml.model.attention.value_residual import (
    ValueResidualAttention,
    _rotate_half,
    blend_values,
)


def _factors(tokens: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    # ValueResidualAttention.forward broadcasts rotary factors over batch and heads.
    return torch.ones(1, tokens, 1, width // 2), torch.zeros(1, tokens, 1, width // 2)


def test_blend_values_and_rotate_half() -> None:
    values = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    first = torch.zeros_like(values)
    assert torch.equal(blend_values(values, first, torch.tensor(0.25)), values * 0.75)
    assert torch.equal(_rotate_half(values), torch.tensor([[[-2.0, 1.0, -4.0, 3.0]]]))


def test_value_residual_attention_forward_variants_and_cost() -> None:
    config = ValueResidualAttention.Config(channels=8, heads=2)
    attention = config.make()
    x = torch.randn(2, 3, 8)
    rope = _factors(3, 4)
    output, raw = attention(x, rope, None)
    assert output.shape == x.shape
    assert raw.shape == (2, 2, 3, 4)
    reference = torch.randn_like(raw)
    mixed, _ = attention(x, rope, reference)
    assert mixed.shape == x.shape
    assert config.cost(seq_len=3, batch_size=2, dtype=torch.float32).params > 0
    no_norm = ValueResidualAttention.Config(
        channels=8,
        heads=2,
        qk_norm=False,
        value_residual=False,
        reference_rope=True,
    ).make()
    no_norm_output, _ = no_norm(x, rope, None)
    assert no_norm_output.shape == x.shape


def test_value_residual_rejects_nondivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ValueResidualAttention.Config(channels=8, heads=3).make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
