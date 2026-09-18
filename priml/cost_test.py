"""Tests for priml.cost."""

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

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    peak,
    reduction_cost,
    resolve_dtype,
    utilization,
)
from priml.custom_types import HasCost
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.softcap import SoftCap
from priml.model.special import Skip
from priml.model.swiglu import SwiGLU


# -- Cost ------------------------------------------------------------------


BF = torch.bfloat16
F32 = torch.float32
I64 = torch.int64


def _t() -> Cost:
    return Cost(
        cells={
            ("flops", "primal", "matmul", BF): 60,
            ("flops", "primal", "elementwise", F32): 4,
            ("flops", "adjoint", "matmul", BF): 120,
            ("bytes", "adjoint", "selection", I64): 2,
        },
        params=5,
    )


def test_full_key_reads_a_float_and_a_missing_cell_is_zero() -> None:
    t = _t()
    assert t["flops", "primal", "matmul", BF] == 60
    assert t["flops", "primal", "sort", F32] == 0


def test_partial_keys_slice_and_drop_the_fixed_leading_axes() -> None:
    t = _t()
    assert t["flops", "primal"].cells == {("matmul", BF): 60, ("elementwise", F32): 4}
    assert t["flops", "adjoint", "matmul"].cells == {(BF,): 120}
    assert t["flops"].params == 0


def test_axes_are_named_by_their_values_in_any_order() -> None:
    t = _t()
    assert t["flops", "matmul"].cells == {("primal", BF): 60, ("adjoint", BF): 120}
    assert t[I64].cells == {("bytes", "adjoint", "selection"): 2}
    assert t[F32].sum() == 4


def test_an_unknown_axis_value_is_rejected_by_name() -> None:
    with pytest.raises(KeyError, match="'gemm' is not a measure"):
        _ = _t()["gemm"]


def test_a_dtype_may_be_named_by_string_or_alias() -> None:
    t = _t()
    assert t["flops", "primal", "matmul", "bfloat16"] == 60
    assert t["flops", "primal", "matmul", "bf16"] == 60
    assert t["bf16", "flops"].cells == t[BF, "flops"].cells
    assert t["f32", "flops"].cells == t[F32, "flops"].cells


def test_intensity_is_a_virtual_measure_on_a_model_cost() -> None:
    c = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 60,
            ("bytes", "primal", "matmul", BF): 4,
            ("flops", "adjoint", "matmul", BF): 120,
            ("bytes", "adjoint", "matmul", BF): 4,
            ("bytes", "primal", "selection", I64): 2,
        },
    )
    assert c["intensity", "primal", "matmul", BF] == 15
    assert c["matmul", BF, "intensity"].cells == {("primal",): 15, ("adjoint",): 30}
    assert c["intensity", "primal", "selection", I64] == 0
    assert (c["matmul", BF, "intensity"] / 15 - 1).cells == {("adjoint",): 1}
    assert c["intensity"].params == 0


def test_sum_totals_every_remaining_cell() -> None:
    assert _t().sum() == 186
    assert _t()["flops", "primal"].sum() == 64
    assert Cost().sum() == 0


def test_add_is_cell_wise_and_keeps_sparse_zeros_out() -> None:
    a = Cost(cells={("flops", "primal", "matmul", BF): 1})
    b = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 2,
            ("flops", "adjoint", "matmul", BF): 3,
        },
    )
    assert a + b == Cost(
        cells={
            ("flops", "primal", "matmul", BF): 3,
            ("flops", "adjoint", "matmul", BF): 3,
        },
    )


def test_cell_wise_division_over_the_written_cells() -> None:
    """Sparse: a cell neither side wrote is zero, not ``0 / 0``."""
    t = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 6,
            ("flops", "primal", "reduction", F32): 1,
            ("bytes", "primal", "matmul", BF): 2,
            ("bytes", "primal", "sort", F32): 5,
        },
        params=3,
    )
    ratio = t["flops"] / t["bytes"]
    assert ratio["primal", "matmul", BF] == 3
    assert ratio["primal", "reduction", F32] == math.inf
    assert ratio["primal", "sort", F32] == 0
    assert ratio["adjoint", "sort", F32] == 0
    assert ratio.params == 0
    assert (t / 2)["flops", "primal", "matmul", BF] == 3


def test_equality_ignores_explicit_zero_cells() -> None:
    assert Cost(cells={("flops", "primal", "matmul", BF): 0}) == Cost()


def test_cost_is_one_table_and_three_owned_integers() -> None:
    names = [f.name for f in fields(Cost)]
    assert names == ["cells", "params", "params_active", "bytes_state"]


def test_cost_add_sums_cells_and_ownership() -> None:
    a = Cost(cells={("flops", "primal", "matmul", F32): 1}, params=3, bytes_state=7)
    b = Cost(cells={("flops", "primal", "matmul", F32): 10}, params=4)
    assert a + b == Cost(
        cells={("flops", "primal", "matmul", F32): 11},
        params=7,
        bytes_state=7,
    )


def test_tile_scales_work_by_rows_and_ownership_by_copies() -> None:
    c = Cost(
        cells={("bytes", "adjoint", "reduction", F32): 2},
        params=5,
        params_active=5,
    )
    tiled = c.tile(3, copies=2)
    assert tiled["bytes", "adjoint", "reduction", F32] == 6
    assert tiled.params == tiled.params_active == 10


def test_matmul_cost_is_tagged_with_its_dtype() -> None:
    c = matmul_cost(channels_in=2, channels_out=3, bias=True, rows=4, dtype=BF)
    assert c["flops", "primal", "matmul", BF] == 12
    assert c["flops", "adjoint", "matmul", BF] == 24
    assert c["flops", "primal", "elementwise", BF] == 3
    assert c["flops", "adjoint", "reduction", BF] == 2.25
    assert c["bytes", "primal", "matmul", BF] == 2 * (2 + 3 + 6 / 4)
    assert c["bytes", "adjoint", "matmul", BF] == 2 * 2 * (2 + 3 + 6 / 4)
    assert c["bytes", "primal", "elementwise", BF] == 2 * (6 + 3 / 4)
    assert c["bytes", "adjoint", "reduction", BF] == 2 * (3 + 3 / 4)
    assert c["bytes", F32] == Cost()
    assert c.params == c.params_active == 9


def test_dtype_defaults_to_the_torch_default() -> None:
    c = matmul_cost(channels_in=2, channels_out=3)
    assert c["bytes", "primal", "matmul", torch.get_default_dtype()] == 4 * (2 + 3 + 6)
    assert c["bytes", "primal", "matmul", BF] == 0


def test_elementwise_and_reduction_costs_carry_the_dtype() -> None:
    ew = elementwise_cost(primal=7, adjoint=11, channels=5, params=3, rows=4, dtype=BF)
    assert ew["bytes", "primal", "elementwise", BF] == 2 * (10 + 3 / 4)
    assert ew["bytes", "adjoint", "elementwise", BF] == 2 * (15 + 3 / 4 + 3)
    assert ew["bytes", "adjoint", "reduction", BF] == 2 * (3 + 3 / 4)
    red = reduction_cost(input_elements=12, output_groups=3, rows=4, dtype=I64)
    assert red["flops", "primal", "reduction", I64] == 9 / 4
    assert red["bytes", "primal", "reduction", I64] == 8 * 15 / 4
    assert red["flops", "adjoint"] == Cost()


def test_intensity_reads_as_a_ratio_of_slices() -> None:
    c = matmul_cost(channels_in=2, channels_out=3, rows=4, dtype=BF)
    assert c["flops", "primal", "matmul"].sum() / c[
        "bytes",
        "primal",
        "matmul",
    ].sum() == (12 / (2 * (2 + 3 + 6 / 4)))
    assert c["flops"].sum() / c["bytes"].sum() == 36 / (3 * 2 * (2 + 3 + 6 / 4))


# -- primitives --------------------------------------------------------------


def test_matmul_cost_without_a_weight_owns_nothing_but_reads_both_operands() -> None:
    c = matmul_cost(channels_in=2, channels_out=3, weight=False, rows=4)
    assert c.params == c.params_active == 0
    assert c["flops", "primal", "matmul"].sum() == 12
    assert c["bytes", "primal", "matmul"].sum() == 4 * (2 + 3 + 6 / 4)


def test_matmul_cost_adjoint_is_twice_primal_in_the_matmul_cells() -> None:
    c = matmul_cost(channels_in=5, channels_out=7, rows=3)
    assert (
        c["flops", "adjoint", "matmul"].sum()
        == 2 * c["flops", "primal", "matmul"].sum()
    )
    assert (
        c["bytes", "adjoint", "matmul"].sum()
        == 2 * c["bytes", "primal", "matmul"].sum()
    )


def test_elementwise_explicit_operand_geometry_is_independent_of_flops() -> None:
    c = elementwise_cost(
        primal=1,
        adjoint=1,
        channels=3,
        inputs=2,
        outputs=1,
        adjoint_inputs=3,
        adjoint_outputs=2,
        dtype=BF,
    )
    assert c["bytes", "primal", "elementwise", BF] == 2 * 3 * 3
    assert c["bytes", "adjoint", "elementwise", BF] == 2 * 3 * 5
    assert c["flops", "primal", "elementwise", BF] == 1


def test_reduction_empty_group_writes_identity_without_arithmetic() -> None:
    c = reduction_cost(input_elements=0)
    assert c["flops", "primal", "reduction", F32] == 0
    assert c["bytes", "primal", "reduction", F32] == 4


def test_reduction_in_the_adjoint_phase() -> None:
    c = reduction_cost(input_elements=6, output_groups=2, phase="adjoint")
    assert c["flops", "adjoint", "reduction", F32] == 4
    assert c["flops", "primal", "reduction", F32] == 0


def test_fractional_sharing_rows_remain_at_least_one() -> None:
    counted = matmul_cost(channels_in=2, channels_out=3, rows=1.5)
    assert counted["bytes", "primal", "matmul", F32] == 4 * (2 + 3 + 6 / 1.5)
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


def test_scalar_division_does_not_overflow_a_reciprocal() -> None:
    key = ("flops", "primal", "matmul", F32)
    tiny = Cost(cells={key: 1e-320})
    assert tiny / 1e-320 == Cost(cells={key: 1})
    assert (Cost(cells={key: 2}) / 0.0)[key] == math.inf


def test_equal_costs_hash_alike_and_zero_cells_do_not_change_the_hash() -> None:
    a = Cost(cells={("flops", "primal", "matmul", BF): 2}, params=3)
    b = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 2,
            ("bytes", "adjoint", "sort", BF): 0,
        },
        params=3,
    )
    assert hash(a) == hash(b)
    assert hash(a) != hash(Cost(cells={("flops", "primal", "matmul", BF): 2}))


def test_division_by_zero_follows_ieee() -> None:
    key = ("flops", "primal", "matmul", F32)
    assert math.isnan((Cost(cells={key: math.nan}) / 0.0)[key])
    assert (Cost(cells={key: -2}) / 0.0)[key] == -math.inf
    assert (Cost(cells={key: 2}) / -0.0)[key] == -math.inf


def test_repr_is_a_grid_with_totals_and_intensity() -> None:
    c = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 64_000,
            ("bytes", "primal", "matmul", BF): 32,
            ("bytes", "primal", "selection", I64): 8,
            ("flops", "adjoint", "matmul", BF): 1.5e9,
        },
        params=7,
    )
    text = repr(c)
    lines = text.splitlines()
    assert lines[0] == "Cost(params=7, params_active=0, bytes_state=0)"
    measure, dtype, primal, selection, adjoint, total = lines[1:]
    assert measure.split() == ["flops", "flops", "bytes", "bytes"]
    assert dtype.split() == ["bf16", "int64", "bf16", "int64", "intensity"]
    assert primal.split() == ["primal", "matmul", "64K", "-", "32", "-", "2K"]
    assert selection.split() == ["primal", "selection", "-", "-", "-", "8", "-"]
    assert adjoint.split() == ["adjoint", "matmul", "1.5G", "-", "-", "-", "inf"]
    assert total.split() == ["total", "1.5G", "-", "32", "8", "37.5M"]


def test_repr_of_a_slice_drops_the_fixed_axes_and_the_empty_table_says_so() -> None:
    c = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 3,
            ("bytes", "primal", "matmul", BF): 6,
        },
    )
    assert repr(c["flops"]).splitlines()[1:] == [
        "              bf16",
        "primal matmul    3",
    ]
    assert repr(c["flops", BF]).splitlines()[1:] == [
        "       matmul",
        "primal      3",
    ]
    assert repr(c["flops", "primal", "matmul"]).splitlines()[1:] == [
        " bf16",
        "    3",
    ]
    assert repr(Cost()).splitlines()[1] == "(empty)"


# -- metrics -----------------------------------------------------------------


def test_utilization_without_a_duration_is_intensity_over_the_ridge() -> None:
    ridge = 989 / 3.35
    c = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 2 * ridge,
            ("bytes", "primal", "matmul", BF): 1,
            ("flops", "adjoint", "sort", F32): 1,
            ("bytes", "adjoint", "sort", F32): 1,
            ("flops", "primal", "matmul", torch.int32): 1,
            ("bytes", "primal", "matmul", torch.int32): 1,
        },
    )
    ratio = utilization(c, device="h100", seq_len=4, batch_size=2)
    assert ratio["primal", "matmul", BF] == pytest.approx(2)
    assert ratio["adjoint", "sort", F32] == pytest.approx(3.35 / 67)
    assert ratio["primal", "matmul", torch.int32] == math.inf
    assert ratio.params == 0


def test_utilization_with_a_duration_is_achieved_over_the_roofline_ceiling() -> None:
    # 8 tokens a step: one compute-bound matmul, one memory-bound sort.
    c = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 989e12 / 8,
            ("bytes", "primal", "matmul", BF): 1,
            ("flops", "adjoint", "sort", F32): 1,
            ("bytes", "adjoint", "sort", F32): 1,
        },
    )
    achieved = utilization(c, device="h100", seq_len=4, batch_size=2, duration_sec=2)
    assert achieved["primal", "matmul", BF] == pytest.approx(0.5)
    # Sort at intensity 1 is capped at bandwidth, 3.35e12 FLOP/s, not 67e12.
    assert achieved["adjoint", "sort", F32] == pytest.approx(8 / 2 / 3.35e12)


def test_peak_prices_matmul_per_dtype_and_every_other_silo_at_the_vector_rate() -> None:
    h100 = peak()["h100"]
    assert h100[BF, "flops", "matmul"] == 989e12
    assert h100[F32, "flops", "matmul"] == 494e12
    assert h100[BF, "flops", "elementwise"] == h100[F32, "flops", "reduction"] == 67e12
    assert h100[torch.float8_e4m3fn, "flops", "sort"] == 67e12
    assert h100[I64, "flops", "sort"] == 67e12
    assert h100[I64, "flops", "matmul"] == 0
    assert h100[BF, "bytes", "matmul"] == h100[F32, "bytes", "sort"] == 3.35e12
    assert h100[BF, "intensity", "matmul"] == 989 / 3.35
    assert h100.params == 0


def test_peak_intensity_is_the_ridge() -> None:
    assert peak()["b200", "matmul", "fp4", "intensity"] == 9000 / 8
    assert peak()["rtx5090", "elementwise", F32, "intensity"] == 104.8 / 1.792


# -- dispatch ----------------------------------------------------------------


class _Priced:
    class Config(Fig["_Priced"]):
        width: int = 4

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            del seq_len, batch_size, dtype, kwargs
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
    assert cost(_Priced.Config(), seq_len=8, batch_size=1, dtype=None) == Cost(params=4)


class _Rows:
    """A priced config that reports the sharing rows it was handed."""

    class Config(Fig["_Rows"]):
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            del kwargs
            del dtype
            return Cost(params=int(seq_len * batch_size))

    def __init__(self, config: Config) -> None:
        del config


def test_resolve_dtype_falls_back_to_the_torch_default() -> None:
    assert resolve_dtype(None) == torch.get_default_dtype()
    assert resolve_dtype(BF) == BF


def test_cost_hands_the_bus_through() -> None:
    assert cost(_Rows.Config(), seq_len=1, batch_size=1, dtype=None) == Cost(params=1)
    assert cost(_Rows.Config(), seq_len=8, batch_size=4, dtype=None) == Cost(params=32)


def test_a_linear_reads_its_rows_from_the_geometry() -> None:
    linear = Linear.Config(2, 3)
    linear.bias = True
    assert cost(linear, seq_len=2, batch_size=2, dtype=None) == cost(
        linear,
        seq_len=4,
        batch_size=1,
        dtype=None,
    )
    assert (
        cost(linear, seq_len=4, batch_size=1, dtype=None)[
            "flops",
            "adjoint",
            "reduction",
        ].sum()
        == 2.25
    )


def test_cost_rejects_a_config_without_a_cost() -> None:
    """A silent zero is the CPU-SDPA bug; a missing arm must raise, naming itself."""
    assert not isinstance(_Unpriced.Config(), HasCost)
    with pytest.raises(TypeError, match=r"_Unpriced\.Config"):
        cost(_Unpriced.Config(), seq_len=8, batch_size=1, dtype=None)


def test_every_model_config_is_priced() -> None:
    """A module Config that cannot price itself fails here, not at a run's MFU line.

    Walks every ``nn.Module`` with a ``Fig`` Config under ``priml.model``,
    ``priml.baselines``, and ``priml.loss`` -- everything a
    ``TrainStep.Config`` model or loss slot can hold -- and asserts the
    ``HasCost`` protocol. torchtitan makes its FLOP method abstract on the
    model Config; this is the same gate without an ABC. ``private`` and
    ``scripts`` trees are one-off tooling, not modules a run prices.

    Attention kernels are the one exemption: a kernel config holds no shapes,
    so the owner of the projections prices it with
    :func:`attention_kernel_cost`; a kernel that priced itself would need
    the bus of shape arguments this design removed.
    """
    root = pathlib.Path(__file__).parent
    # The package prefix is read off this module's own name rather than
    # spelled out, so the walk resolves under either import root.
    package = __name__.rsplit(".", 1)[0]
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
            if not issubclass(owner, torch.nn.Module) or _is_attention_kernel(owner):
                continue
            config = getattr(owner, "Config", None)
            if not (inspect.isclass(config) and issubclass(config, Fig)):
                continue
            if not hasattr(config, "cost"):
                unpriced.append(f"{module_name}.{name}")
    assert unpriced == []


def _is_attention_kernel(owner: type[torch.nn.Module]) -> bool:
    """Whether ``owner`` is a kernel slot: called on ``(q, k, v)``, priced by its owner."""
    forward = inspect.signature(owner.forward)
    return [*forward.parameters][1:4] == ["q", "k", "v"]


# -- leaves that exercise the silos ------------------------------------------


def test_unsupported_activation_fails_at_cost_not_at_forward() -> None:
    """An injected activation without a cost is a TypeError when priced."""
    ffn = SwiGLU.Config(4, 4)
    ffn.channels_hidden = 8
    ffn.act = torch.sin
    model = ffn.copy_tree().finalize().make()
    assert model(torch.randn(2, 4)).shape == (2, 4)
    with pytest.raises(TypeError, match="has no price"):
        cost(ffn.copy_tree().finalize(), seq_len=1, batch_size=1, dtype=None)


def test_scalar_leaf_formulas_and_composition() -> None:
    linear = Linear.Config(2, 3)
    linear.bias = True
    counted = cost(linear, seq_len=4, batch_size=1, dtype=None)
    assert counted["flops", "primal", "elementwise"].sum() == 3
    assert counted["flops", "adjoint", "reduction"].sum() == 2.25
    norm = cost(RMSNorm.Config(4).finalize(), seq_len=1, batch_size=1, dtype=None)
    assert (
        norm["flops", "primal", "elementwise"].sum()
        + norm["flops", "primal", "reduction"].sum()
        == 14
    )
    capped = cost(SoftCap.Config(2, 3).finalize(), seq_len=1, batch_size=1, dtype=None)
    assert capped["flops", "primal", "elementwise"].sum() == 9
    assert capped["flops", "adjoint", "elementwise"].sum() == 15
    skip = Skip.Config()
    skip.inner = Linear.Config(4, 4)
    assert (
        cost(skip, seq_len=1, batch_size=1, dtype=None)[
            "flops",
            "primal",
            "elementwise",
        ].sum()
        == 4
    )
    assert (
        cost(skip, seq_len=1, batch_size=1, dtype=None)[
            "flops",
            "adjoint",
            "elementwise",
        ].sum()
        == 4
    )
    ffn = SwiGLU.Config(4, 4)
    ffn.channels_hidden = 8
    ffn = ffn.finalize()
    assert (
        cost(ffn, seq_len=1, batch_size=1, dtype=None)[
            "flops",
            "primal",
            "elementwise",
        ].sum()
        == 6 * 8
    )
    assert (
        cost(ffn, seq_len=1, batch_size=1, dtype=None)[
            "flops",
            "adjoint",
            "elementwise",
        ].sum()
        == 7 * 8
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
