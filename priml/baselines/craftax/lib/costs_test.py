"""Tests for the costs the port's configs share."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn import functional

import pytest
import torch

from priml.baselines.craftax.lib.costs import (
    activation_cost,
    broadcast_add_cost,
    cast_cost,
    concat_cost,
    gather_cost,
    logits_cost,
    residual_cost,
    weight_gradient_only,
)
from priml.cost import matmul_cost
from priml.model.embedding import Embedding
from priml.model.swiglu import silu


if TYPE_CHECKING:
    from priml.math.custom_types import TensorFn


@pytest.mark.parametrize(
    ("activation", "primal", "adjoint"),
    [(functional.silu, 5, 5), (functional.gelu, 5, 11), (torch.tanh, 1, 3)],
    ids=["silu", "gelu", "tanh"],
)
def test_torchs_activations_count_their_ops_per_value(
    activation: TensorFn,
    primal: int,
    adjoint: int,
) -> None:
    counted = activation_cost(activation, elements=6, dtype=None)
    assert counted["flops", "primal", "elementwise"].sum() == 6 * primal
    assert counted["flops", "adjoint", "elementwise"].sum() == 6 * adjoint


def test_an_activation_with_its_own_cost_is_priced_by_it() -> None:
    assert activation_cost(silu, elements=6, dtype=None) == activation_cost(
        functional.silu,
        elements=6,
        dtype=None,
    )
    with pytest.raises(TypeError, match="no cost"):
        activation_cost(torch.sin, elements=6, dtype=None)


def test_a_broadcast_add_owns_its_table_and_sums_its_gradient_over_the_copies() -> None:
    added = broadcast_add_cost(params=4, rows=3, dtype=None)
    assert added.params == 4
    assert added["flops", "primal", "elementwise"].sum() == 3 * 4
    assert added["flops", "adjoint", "reduction"].sum() == 4 * (3 - 1)


def test_a_gather_is_a_lookup_that_owns_nothing() -> None:
    gathered = gather_cost(rows=3, width=4, source=5, dtype=None)
    lookup = Embedding.Config(channels_in=5, channels_out=4).cost(
        seq_len=1,
        batch_size=3,
        dtype=None,
    )
    assert gathered.cells == lookup.cells
    assert lookup.params == 5 * 4
    assert gathered.params == gathered.params_active == 0


def test_without_the_inputs_gradient_a_product_is_twice_its_primal() -> None:
    both = matmul_cost(channels_in=3, channels_out=4, bias=True, rows=5, dtype=None)
    alone = weight_gradient_only(both)
    assert alone["flops", "matmul"].sum() == 2 * (2 * 5 * 3 * 4)
    assert both["flops", "matmul"].sum() == 6 * 5 * 3 * 4
    assert alone["bytes", "adjoint", "matmul"].sum() == (
        both["bytes", "primal", "matmul"].sum()
    )
    # The bias's gradient still sums over the rows.
    assert alone["flops", "adjoint", "reduction"].sum() == 4 * (5 - 1)
    assert alone.params == both.params == 3 * 4 + 4


def test_a_float32_head_skips_its_copy_and_a_bfloat16_one_pays_it() -> None:
    fp32 = logits_cost(
        channels_in=3,
        channels_out=4,
        rows=5,
        weight=False,
        dtype=torch.float32,
    )
    assert fp32 == matmul_cost(
        channels_in=3,
        channels_out=4,
        rows=5,
        weight=False,
        dtype=torch.float32,
    )
    bf16 = logits_cost(
        channels_in=3,
        channels_out=4,
        rows=5,
        weight=True,
        dtype=torch.bfloat16,
    )
    copied = bf16["bytes", "elementwise"].sum()
    assert copied == cast_cost(elements=5 * 4, dtype=torch.bfloat16).sum()
    assert bf16.params == 3 * 4


def test_copies_move_values_and_compute_nothing() -> None:
    for copied in (
        cast_cost(elements=6, dtype=None),
        concat_cost(elements=6, dtype=None),
    ):
        assert copied["flops"].sum() == 0
    assert cast_cost(elements=6, dtype=None)["bytes"].sum() == 2 * 2 * 6 * 4
    assert concat_cost(elements=6, dtype=None)["bytes"].sum() == 2 * 6 * 4


def test_a_residual_add_is_one_add_each_way() -> None:
    added = residual_cost(channels=4, rows=3, dtype=None)
    assert added["flops", "primal", "elementwise"].sum() == 3 * 4
    assert added["flops", "adjoint", "elementwise"].sum() == 3 * 4
    assert added.params == 0


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
