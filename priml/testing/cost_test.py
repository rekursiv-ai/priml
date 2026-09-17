"""Tests for priml.testing.cost."""

from __future__ import annotations

from dataclasses import replace
from typing import override

from configgle import Makes

import pytest
import torch

from priml.model.cost import Cost
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.testing.cost import assert_cost_matches_torch


def test_a_linear_matches_torch_exactly() -> None:
    analytical = assert_cost_matches_torch(
        Linear.Config(8, 16, bias=True),
        build_input=lambda: torch.randn(4, 8, requires_grad=True),
        num_tokens=4,
    )
    assert analytical.params == 8 * 16 + 16


def test_a_swiglu_matches_torch_exactly() -> None:
    assert_cost_matches_torch(
        SwiGLU.Config(8, 8, channels_hidden=16, round_to=1),
        build_input=lambda: torch.randn(4, 8, requires_grad=True),
        num_tokens=4,
    )


def test_a_matmul_free_leaf_measures_zero() -> None:
    analytical = assert_cost_matches_torch(
        RMSNorm.Config(8, elementwise_affine=True),
        build_input=lambda: torch.randn(4, 8, requires_grad=True),
        num_tokens=4,
    )
    assert analytical.training.flops.matmul == 0
    assert analytical.training.flops.elementwise > 0


def test_a_wrong_matmul_count_is_caught() -> None:
    """Positive control: an estimate off by a factor must fail."""
    with pytest.raises(AssertionError, match="matmul FLOPs/token"):
        assert_cost_matches_torch(
            Linear.Config(8, 16),
            build_input=lambda: torch.randn(4, 8, requires_grad=True),
            num_tokens=4,
            expected_ratio=2.0,
        )


class _OvercountingLinear(Linear):
    """A Linear whose cost claims one parameter it does not own."""

    class Config(Makes["_OvercountingLinear"], Linear.Config, kw_only=False):
        @override
        def cost(
            self,
            *,
            rows: float = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            true = super().cost(rows=rows, itemsize=itemsize, **kwargs)
            return replace(true, params=true.params + 1)


def test_a_wrong_param_count_is_caught() -> None:
    """Positive control: the params check names both numbers."""
    with pytest.raises(
        AssertionError,
        match=r"cost\.params=129 but the module owns 128",
    ):
        assert_cost_matches_torch(
            _OvercountingLinear.Config(8, 16),
            build_input=lambda: torch.randn(4, 8, requires_grad=True),
            num_tokens=4,
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
