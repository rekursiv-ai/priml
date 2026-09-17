"""Tests for priml.model.cost."""

from __future__ import annotations

from dataclasses import fields
from typing import cast

import importlib
import inspect
import math
import pathlib

from configgle import Fig

import pytest
import torch

from priml.model.cost import (
    Bytes,
    Compute,
    Cost,
    Flops,
    HasCost,
    KernelStats,
    Total,
    cost,
    elementwise_cost,
    matmul_cost,
    mbu,
    mfu,
    reduction_cost,
    utilization,
)
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.softcap import SoftCap
from priml.model.special import Skip
from priml.model.swiglu import SwiGLU


# -- KernelStats -------------------------------------------------------------


def test_kernel_stats_has_the_five_silos_and_nothing_else() -> None:
    silos = ["matmul", "elementwise", "reduction", "selection", "sort"]
    assert [f.name for f in fields(KernelStats)] == silos
    assert [f.name for f in fields(Flops)] == silos
    assert [f.name for f in fields(Bytes)] == silos


def test_kernel_stats_default_is_zero_and_total_sums_every_silo() -> None:
    assert KernelStats().total == 0
    assert Flops(matmul=1, elementwise=2, reduction=3, selection=4, sort=5).total == 15


def test_kernel_stats_add_is_silo_wise_and_keeps_the_subclass() -> None:
    total = Flops(matmul=1, reduction=3) + Flops(matmul=10, sort=5)
    assert type(total) is Flops
    assert total == Flops(matmul=11, reduction=3, sort=5)


def test_add_keeps_the_left_operand_type() -> None:
    """At runtime the left operand's type wins.

    ``Self`` on ``__add__`` makes ``Flops + Bytes`` a checker error, so the
    operator form is not written here.
    """
    mixed = KernelStats.__add__(Flops(matmul=1), Bytes(matmul=1))
    assert type(mixed) is Flops
    assert mixed.matmul == 2


def test_kernel_stats_scales_by_a_number_both_sides_but_not_bool() -> None:
    f = Flops(matmul=2, elementwise=3)
    assert 3 * f == f * 3 == Flops(matmul=6, elementwise=9)
    assert f * 0.5 == Flops(matmul=1, elementwise=1.5)
    assert f / 2 == Flops(matmul=1, elementwise=1.5)
    assert type(f * 3) is type(3 * f) is type(f / 2) is Flops
    with pytest.raises(TypeError):
        _ = f * True


def test_kernel_stats_hadamard_product_and_quotient_return_the_base_type() -> None:
    """``flops * seconds`` and ``flops / bytes`` are new units, so not Flops."""
    flops = Flops(matmul=6, elementwise=3, reduction=1)
    moved = Bytes(matmul=2, elementwise=3)
    intensity = flops / moved
    assert type(intensity) is KernelStats
    assert intensity.matmul == 3.0
    assert intensity.elementwise == 1.0
    assert intensity.reduction == float("inf")
    assert math.isnan(intensity.selection)
    assert math.isnan(intensity.sort)
    product = flops * moved
    assert type(product) is KernelStats
    assert product == KernelStats(matmul=12, elementwise=9)


# -- Compute -----------------------------------------------------------------


def test_compute_adds_and_repeats_both_halves() -> None:
    a = Compute(flops=Flops(matmul=1), bytes=Bytes(elementwise=2))
    b = Compute(flops=Flops(reduction=3), bytes=Bytes(elementwise=4))
    assert a + b == Compute(
        flops=Flops(matmul=1, reduction=3),
        bytes=Bytes(elementwise=6),
    )
    assert 3 * a == a * 3 == Compute(flops=Flops(matmul=3), bytes=Bytes(elementwise=6))
    with pytest.raises(TypeError):
        _ = a * True


def test_compute_intensity_is_flops_over_bytes_per_silo() -> None:
    c = Compute(flops=Flops(matmul=6, reduction=1), bytes=Bytes(matmul=2))
    assert c.intensity.matmul == 3
    assert c.intensity.reduction == float("inf")


def test_compute_total_sums_every_silo() -> None:
    c = Compute(
        flops=Flops(matmul=60, elementwise=4, reduction=1),
        bytes=Bytes(matmul=6, elementwise=10, selection=4),
    )
    assert c.total == Total(flops=65, bytes=20)
    assert c.total.flops / c.total.bytes == 65 / 20


# -- Cost --------------------------------------------------------------------


def test_cost_default_is_zero_everywhere() -> None:
    zero = Cost()
    assert zero.training == Compute()
    assert zero.params == zero.params_active == zero.bytes_state == 0


def test_cost_add_is_fieldwise() -> None:
    a = Cost(primal=Compute(flops=Flops(matmul=1)), params=3, bytes_state=7)
    b = Cost(
        primal=Compute(flops=Flops(matmul=10)),
        adjoint=Compute(flops=Flops(reduction=2)),
        params=30,
    )
    assert a + b == Cost(
        primal=Compute(flops=Flops(matmul=11)),
        adjoint=Compute(flops=Flops(reduction=2)),
        params=33,
        bytes_state=7,
    )


def test_sum_over_children_with_an_empty_start() -> None:
    parts = [Cost(primal=Compute(flops=Flops(matmul=1))), Cost(params=4)]
    assert sum(parts, Cost()) == Cost(primal=Compute(flops=Flops(matmul=1)), params=4)
    assert sum([], Cost()) == Cost()


def test_cost_mul_broadcasts_by_int_both_sides_and_rejects_bool_and_float() -> None:
    c = Cost(primal=Compute(flops=Flops(matmul=2)), params=3)
    assert c * 3 == 3 * c == Cost(primal=Compute(flops=Flops(matmul=6)), params=9)
    with pytest.raises(TypeError):
        _ = c * True
    # ``__mul__`` returns ``NotImplemented`` for a fraction; the operator then
    # raises, which is what a caller sees.
    assert Cost.__mul__(c, 1.5) is NotImplemented  # ty: ignore[invalid-argument-type] -- The rejection is under test.  # pyright: ignore[reportArgumentType] -- The rejection is under test.


def test_cost_training_sums_both_passes() -> None:
    c = Cost(
        primal=Compute(flops=Flops(matmul=6), bytes=Bytes(matmul=1, elementwise=1)),
        adjoint=Compute(
            flops=Flops(matmul=12, reduction=1),
            bytes=Bytes(elementwise=2),
        ),
    )
    assert c.training.flops == Flops(matmul=18, reduction=1)
    assert c.training.bytes == Bytes(matmul=1, elementwise=3)


def test_tile_repeats_work_by_rows_and_ownership_by_copies() -> None:
    """A shared norm over four head rows runs four times and exists once."""
    one = Cost(
        primal=Compute(
            flops=Flops(matmul=2, elementwise=3),
            bytes=Bytes(matmul=7, elementwise=11),
        ),
        adjoint=Compute(
            flops=Flops(matmul=4, reduction=5),
            bytes=Bytes(matmul=7, elementwise=11),
        ),
        params=7,
        params_active=7,
        bytes_state=13,
    )
    tiled = one.tile(4)
    assert tiled.primal.flops == 4 * one.primal.flops
    assert tiled.adjoint.flops == 4 * one.adjoint.flops
    assert tiled.primal.bytes == 4 * one.primal.bytes
    assert tiled.adjoint.bytes == 4 * one.adjoint.bytes
    assert tiled.params == tiled.params_active == 7
    assert tiled.bytes_state == 52
    twice = one.tile(4, copies=2)
    assert twice.params == twice.params_active == 14
    assert twice.primal.bytes.matmul == 28
    assert one.tile(4, copies=4) == 4 * one


# -- primitives --------------------------------------------------------------


@pytest.mark.parametrize(("rows", "reduction"), [(1, 0), (2, 1.5), (4, 2.25)])
def test_matmul_cost_bias_adds_primal_and_reduces_adjoint(
    rows: int,
    reduction: float,
) -> None:
    c = matmul_cost(channels_in=2, channels_out=3, bias=True, rows=rows)
    assert c == Cost(
        primal=Compute(
            flops=Flops(matmul=12, elementwise=3),
            bytes=Bytes(
                matmul=4 * (2 + 3 + 6 / rows),
                elementwise=4 * (6 + 3 / rows),
            ),
        ),
        adjoint=Compute(
            flops=Flops(matmul=24, reduction=reduction),
            bytes=Bytes(
                matmul=4 * (4 + 6 + 12 / rows),
                reduction=4 * (3 + 3 / rows),
            ),
        ),
        params=9,
        params_active=9,
    )


def test_matmul_cost_without_a_weight_owns_nothing_but_reads_both_operands() -> None:
    """``QK^T`` has activation traffic, despite owning no parameters."""
    c = matmul_cost(channels_in=8, channels_out=32, weight=False)
    assert c.primal.flops == Flops(matmul=2 * 8 * 32)
    assert c.adjoint.flops == Flops(matmul=4 * 8 * 32)
    assert c.params == c.params_active == 0
    assert c.primal.bytes == Bytes(matmul=4 * (8 + 32 + 8 * 32))
    assert c.adjoint.bytes == 2 * c.primal.bytes


def test_matmul_cost_adjoint_is_twice_primal_in_the_matmul_silo() -> None:
    c = matmul_cost(channels_in=5, channels_out=7)
    assert c.adjoint.flops.matmul == 2 * c.primal.flops.matmul


@pytest.mark.parametrize(("rows", "reduction"), [(1, 0), (2, 1.5), (4, 2.25)])
def test_elementwise_cost_reduces_each_parameter_over_the_rows(
    rows: int,
    reduction: float,
) -> None:
    """The same N-1 rule ``matmul_cost`` applies to a bias, applied to ``params``."""
    c = elementwise_cost(
        primal=7,
        adjoint=11,
        channels=5,
        params=3,
        rows=rows,
    )
    assert c == Cost(
        primal=Compute(
            flops=Flops(elementwise=7),
            bytes=Bytes(elementwise=4 * (10 + 3 / rows)),
        ),
        adjoint=Compute(
            flops=Flops(elementwise=11, reduction=reduction),
            bytes=Bytes(
                elementwise=4 * (15 + 3 / rows + 3),
                reduction=4 * (3 + 3 / rows),
            ),
        ),
        params=3,
        params_active=3,
    )
    assert elementwise_cost(primal=1, adjoint=2) == Cost(
        primal=Compute(flops=Flops(elementwise=1)),
        adjoint=Compute(flops=Flops(elementwise=2)),
    )


@pytest.mark.parametrize("itemsize", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("weight", [False, True])
def test_matmul_operand_geometry(itemsize: int, rows: int, weight: bool) -> None:
    counted = matmul_cost(
        channels_in=3,
        channels_out=5,
        rows=rows,
        weight=weight,
        itemsize=itemsize,
    )
    assert counted.primal.bytes.matmul == itemsize * (3 + 5 + 15 / rows)
    assert counted.adjoint.bytes.matmul == itemsize * (6 + 10 + 30 / rows)
    assert counted.primal.flops.matmul == 30
    assert counted.adjoint.flops.matmul == 60
    assert counted.params == (15 if weight else 0)


def test_elementwise_explicit_operand_geometry_is_independent_of_flops() -> None:
    counted = elementwise_cost(
        primal=0,
        adjoint=99,
        channels=7,
        inputs=2,
        outputs=1,
        adjoint_inputs=3,
        adjoint_outputs=2,
        itemsize=2,
    )
    assert counted.primal.bytes == Bytes(elementwise=42)
    assert counted.adjoint.bytes == Bytes(elementwise=70)
    assert counted.primal.flops.total == 0
    assert counted.adjoint.flops.total == 99


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (1, 0.0, math.inf),
        (-1, 0.0, -math.inf),
        (1, -0.0, -math.inf),
        (-1, -0.0, math.inf),
    ],
)
def test_silo_division_preserves_zero_denominator_sign(
    numerator: float,
    denominator: float,
    expected: float,
) -> None:
    quotient = Flops(matmul=numerator) / Bytes(matmul=denominator)
    assert quotient.matmul == expected


def test_empty_cost_has_undefined_intensity() -> None:
    intensity = Cost().training.intensity
    assert all(
        math.isnan(value)
        for value in (
            intensity.matmul,
            intensity.elementwise,
            intensity.reduction,
            intensity.selection,
            intensity.sort,
        )
    )


def test_scalar_division_does_not_overflow_a_reciprocal() -> None:
    assert Flops(matmul=1e-320) / 1e-320 == Flops(matmul=1)
    quotient = Flops(matmul=2) / 0.0
    assert quotient.matmul == math.inf
    assert math.isnan(quotient.elementwise)


def test_reduction_empty_group_writes_identity_without_arithmetic() -> None:
    assert reduction_cost(input_elements=0) == Compute(bytes=Bytes(reduction=4))


def test_fractional_sharing_rows_remain_at_least_one() -> None:
    counted = matmul_cost(channels_in=2, channels_out=3, rows=1.5)
    assert counted.primal.bytes.matmul == 4 * (2 + 3 + 6 / 1.5)
    with pytest.raises(ValueError, match="rows"):
        elementwise_cost(primal=1, adjoint=1, params=1, rows=0.5)


@pytest.mark.parametrize("rows", [0, -1, math.nan, math.inf])
def test_primitives_reject_invalid_row_count(rows: float) -> None:
    with pytest.raises(ValueError, match="rows"):
        matmul_cost(channels_in=2, channels_out=3, rows=rows)
    with pytest.raises(ValueError, match="rows"):
        elementwise_cost(primal=1, adjoint=1, rows=rows)
    with pytest.raises(ValueError, match="rows"):
        reduction_cost(input_elements=3, rows=rows)


@pytest.mark.parametrize("itemsize", [0, -1])
def test_primitives_reject_nonpositive_itemsize(itemsize: int) -> None:
    with pytest.raises(ValueError, match="itemsize"):
        matmul_cost(channels_in=2, channels_out=3, itemsize=itemsize)
    with pytest.raises(ValueError, match="itemsize"):
        elementwise_cost(primal=1, adjoint=1, itemsize=itemsize)


@pytest.mark.parametrize(
    ("input_elements", "output_groups", "rows", "itemsize"),
    [(12, 3, 4, 2), (0, 0, 1, 4), (1, 1, 1, 4), (2.5, 0.5, 2, 8)],
)
def test_reduction_tensor_io(
    input_elements: float,
    output_groups: float,
    rows: int,
    itemsize: int,
) -> None:
    counted = reduction_cost(
        input_elements=input_elements,
        output_groups=output_groups,
        rows=rows,
        itemsize=itemsize,
    )
    assert counted.flops == Flops(
        reduction=(input_elements - output_groups) / rows,
    )
    assert counted.bytes == Bytes(
        reduction=itemsize * (input_elements + output_groups) / rows,
    )


# -- metrics -----------------------------------------------------------------


def test_utilization_is_per_silo_and_matmul_is_mfu() -> None:
    c = Cost(
        primal=Compute(flops=Flops(matmul=2, elementwise=3_000_000)),
        adjoint=Compute(flops=Flops(matmul=4, reduction=5_000_000)),
    )
    peak = KernelStats(matmul=100, elementwise=1e9, reduction=1e9, selection=1, sort=1)
    achieved = utilization(c, tokens_per_sec=10, peak=peak)
    assert achieved.matmul == 0.6
    assert achieved.elementwise == 0.03
    assert achieved.reduction == 0.05
    assert achieved.selection == achieved.sort == 0
    assert mfu(c, tokens_per_sec=10, peak_flops_per_sec=100) == achieved.matmul


def test_mbu_reads_active_weights_once_and_state_per_position() -> None:
    c = Cost(params=1000, params_active=100, bytes_state=4)
    achieved = mbu(
        c,
        batch=2,
        context_len=8,
        steps_per_sec=10,
        itemsize=2,
        peak_bytes_per_sec=1e4,
    )
    assert achieved == (100 * 2 + 2 * 8 * 4) * 10 / 1e4


# -- dispatch ----------------------------------------------------------------


class _Priced:
    class Config(Fig["_Priced"]):
        width: int = 4

        def cost(self, **kwargs: object) -> Cost:
            del kwargs
            return Cost(params=self.width)

    def __init__(self, config: Config) -> None:
        del config


class _Unpriced:
    class Config(Fig["_Unpriced"]):
        pass

    def __init__(self, config: Config) -> None:
        del config


def test_cost_calls_the_protocol() -> None:
    assert isinstance(_Priced.Config(), HasCost)
    assert cost(_Priced.Config(), seq_len=8) == Cost(params=4)


class _Rows:
    """A priced config that reports the row count it was handed."""

    class Config(Fig["_Rows"]):
        def cost(self, *, rows: float = 1, **kwargs: object) -> Cost:
            del kwargs
            return Cost(params=int(rows))

    def __init__(self, config: Config) -> None:
        del config


def test_cost_derives_shared_rows_from_seq_len_and_batch_size() -> None:
    """A caller states the batch geometry; a container that set rows wins."""
    assert cost(_Rows.Config()) == Cost(params=1)
    assert cost(_Rows.Config(), seq_len=8) == Cost(params=8)
    assert cost(_Rows.Config(), seq_len=8, batch_size=4) == Cost(params=32)
    assert cost(_Rows.Config(), batch_size=4) == Cost(params=4)
    assert cost(_Rows.Config(), seq_len=8, batch_size=4, rows=2) == Cost(params=2)


def test_seq_len_and_batch_size_stay_on_the_bus() -> None:
    linear = Linear.Config(2, 3)
    linear.bias = True
    assert cost(linear, seq_len=2, batch_size=2) == cost(linear, seq_len=4)
    assert cost(linear, seq_len=4).adjoint.flops.reduction == 2.25


def test_cost_rejects_a_config_without_a_cost() -> None:
    """A silent zero is the CPU-SDPA bug; a missing arm must raise, naming itself."""
    assert not isinstance(_Unpriced.Config(), HasCost)
    with pytest.raises(TypeError, match=r"_Unpriced\.Config"):
        cost(_Unpriced.Config(), seq_len=8)


def test_every_model_config_is_priced() -> None:
    """A module Config that cannot price itself fails here, not at a run's MFU line.

    Walks every ``nn.Module`` with a ``Fig`` Config under ``priml.model``,
    ``priml.baselines``, and ``priml.loss`` -- everything a
    ``TrainStep.Config`` model or loss slot can hold -- and asserts the
    ``HasCost`` protocol. torchtitan makes its FLOP method abstract on the
    model Config; this is the same gate without an ABC. ``private`` and
    ``scripts`` trees are one-off tooling, not modules a run prices.
    """
    root = pathlib.Path(__file__).parents[1]
    # The package prefix is read off this module's own name rather than
    # spelled out, so the walk resolves under either import root.
    package = __name__.rsplit(".", 2)[0]
    unpriced: list[str] = []
    for path in sorted(
        [
            *root.glob("model/**/*.py"),
            *root.glob("baselines/**/*.py"),
            *root.glob("loss/**/*.py"),
        ],
    ):
        parts = path.relative_to(root).with_suffix("").parts
        if path.name.endswith("_test.py") or {"private", "scripts"} & set(parts):
            continue
        module_name = ".".join((package, *parts)).removesuffix(".__init__")
        module = importlib.import_module(module_name)
        # ``vars`` rather than ``inspect.getmembers``: the latter reads every
        # attribute, which resolves a ``wrapt.lazy_import`` proxy and imports
        # an optional accelerator dependency the CPU path never needs.
        for name, owner in cast(dict[str, object], vars(module)).items():
            if type(owner) is not type or owner.__module__ != module_name:
                continue
            if not issubclass(owner, torch.nn.Module):
                continue
            config = getattr(owner, "Config", None)
            if not (inspect.isclass(config) and issubclass(config, Fig)):
                continue
            if not hasattr(config, "cost"):
                unpriced.append(f"{module_name}.{name}")
    assert unpriced == []


# -- leaves that exercise the silos ------------------------------------------


def test_unsupported_activation_fails_at_cost_not_at_forward() -> None:
    """An injected activation without a cost is a TypeError when priced."""
    ffn = SwiGLU.Config(4, 4)
    ffn.channels_hidden = 8
    ffn.act = torch.sin
    model = ffn.copy_tree().finalize().make()
    assert model(torch.randn(2, 4)).shape == (2, 4)
    with pytest.raises(TypeError, match="has no cost"):
        cost(ffn.copy_tree().finalize())


def test_scalar_leaf_formulas_and_composition() -> None:
    linear = Linear.Config(2, 3)
    linear.bias = True
    counted = cost(linear, rows=4)
    assert counted.primal.flops.elementwise == 3
    assert counted.adjoint.flops.reduction == 2.25
    norm = cost(RMSNorm.Config(4).finalize())
    assert norm.primal.flops.elementwise + norm.primal.flops.reduction == 14
    capped = cost(SoftCap.Config(2, 3).finalize())
    assert capped.primal.flops.elementwise == 9
    assert capped.adjoint.flops.elementwise == 15
    skip = Skip.Config()
    skip.inner = Linear.Config(4, 4)
    assert cost(skip).primal.flops.elementwise == 4
    assert cost(skip).adjoint.flops.elementwise == 4
    ffn = SwiGLU.Config(4, 4)
    ffn.channels_hidden = 8
    ffn = ffn.finalize()
    assert cost(ffn).primal.flops.elementwise == 6 * 8
    assert cost(ffn).adjoint.flops.elementwise == 7 * 8


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
