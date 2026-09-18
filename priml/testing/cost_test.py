"""Tests for priml.testing.cost."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, override

from configgle import Makes

import pytest
import torch

from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.testing.cost import assert_cost_matches_torch, measured_traffic


if TYPE_CHECKING:
    from priml.cost import Cost


def test_a_linear_matches_torch_exactly() -> None:
    analytical = assert_cost_matches_torch(
        Linear.Config(8, 16, bias=True),
        build_input=lambda: torch.randn(4, 8, requires_grad=True),
        seq_len=4,
        batch_size=1,
        num_tokens=4,
        dtype=None,
    )
    assert analytical.params == 8 * 16 + 16


def test_a_swiglu_matches_torch_exactly() -> None:
    assert_cost_matches_torch(
        SwiGLU.Config(8, 8, channels_hidden=16, round_to=1),
        build_input=lambda: torch.randn(4, 8, requires_grad=True),
        seq_len=4,
        batch_size=1,
        num_tokens=4,
        dtype=None,
    )


def test_a_matmul_free_leaf_measures_zero() -> None:
    analytical = assert_cost_matches_torch(
        RMSNorm.Config(8, elementwise_affine=True),
        build_input=lambda: torch.randn(4, 8, requires_grad=True),
        seq_len=4,
        batch_size=1,
        num_tokens=4,
        dtype=None,
    )
    assert analytical["flops", "matmul"].sum() == 0
    assert analytical["flops", "elementwise"].sum() > 0


def test_matmul_bytes_are_checked_against_dispatched_operands() -> None:
    """The matmul silo's traffic is exactly what torch's ``mm`` operands weigh.

    Bias-free, so the projection is one ``mm`` forward and two back; every
    operand of those three products is read or written once, which is the
    analytical convention.
    """
    analytical = assert_cost_matches_torch(
        Linear.Config(8, 16),
        build_input=lambda: torch.randn(4, 8, requires_grad=True),
        seq_len=4,
        batch_size=1,
        num_tokens=4,
        dtype=None,
    )
    # Per token: x (8) + W (128 / 4 rows) + y (16) each way, times three products.
    assert analytical["bytes", "matmul"].sum() == 4 * 3 * (8 + 32 + 16)


def test_measured_traffic_is_reported_per_silo() -> None:
    """``measured_traffic`` tallies what torch moved, per silo, per token."""
    traffic = measured_traffic(
        Linear.Config(8, 16),
        build_input=lambda: torch.randn(4, 8, requires_grad=True),
        num_tokens=4,
    )
    assert traffic["matmul"] == 4 * 3 * (8 + 32 + 16)
    assert traffic["selection"] == 0
    assert traffic["sort"] == 0


class _UndercountingMatmul(Linear):
    """A Linear whose cost forgets the weight's own traffic."""

    class Config(Makes["_UndercountingMatmul"], Linear.Config, kw_only=False):
        @override
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            true = super().cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            return replace(
                true,
                cells=(true.only("flops") + true.only("bytes") / 2).cells,
            )


def test_wrong_matmul_bytes_are_caught() -> None:
    """Positive control: FLOPs can be right while the operand traffic is wrong."""
    with pytest.raises(ValueError, match="matmul bytes/token"):
        assert_cost_matches_torch(
            _UndercountingMatmul.Config(8, 16),
            build_input=lambda: torch.randn(4, 8, requires_grad=True),
            seq_len=4,
            batch_size=1,
            num_tokens=4,
            dtype=None,
        )


def test_a_wrong_matmul_count_is_caught() -> None:
    """Positive control: an estimate off by a factor must fail."""
    with pytest.raises(ValueError, match="matmul FLOPs/token"):
        assert_cost_matches_torch(
            Linear.Config(8, 16),
            build_input=lambda: torch.randn(4, 8, requires_grad=True),
            seq_len=4,
            batch_size=1,
            num_tokens=4,
            dtype=None,
            expected_ratio=2.0,
        )


class _OvercountingLinear(Linear):
    """A Linear whose cost claims one parameter it does not own."""

    class Config(Makes["_OvercountingLinear"], Linear.Config, kw_only=False):
        @override
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            true = super().cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )
            return replace(true, params=true.params + 1)


def test_a_wrong_param_count_is_caught() -> None:
    """Positive control: the params check names both numbers."""
    with pytest.raises(
        ValueError,
        match=r"cost\.params=129 but the module owns 128",
    ):
        assert_cost_matches_torch(
            _OvercountingLinear.Config(8, 16),
            build_input=lambda: torch.randn(4, 8, requires_grad=True),
            seq_len=4,
            batch_size=1,
            num_tokens=4,
            dtype=None,
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
