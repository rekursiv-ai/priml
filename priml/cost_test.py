"""Tests for priml.cost."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Final, cast

import functools
import gc
import importlib
import inspect
import math
import pathlib
import weakref

from configgle import Fig

import pytest
import torch

from priml.baselines.sudoku.experiments import exp000
from priml.cost import (
    MEASURES,
    Cost,
    Report,
    _column_order,
    _div,
    _grid,
    _kind,
    _si,
    cost,
    elementwise_cost,
    intensity,
    map_cost,
    matmul_cost,
    peak,
    reduction_cost,
    resolve_dtype,
    set_cost,
    traffic,
    utilization,
)
from priml.custom_types import HasCost
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.softcap import SoftCap
from priml.model.special import Skip
from priml.model.swiglu import SwiGLU


if TYPE_CHECKING:
    from collections.abc import Callable


# -- Cost ------------------------------------------------------------------


BF: Final = torch.bfloat16
F32: Final = torch.float32
I64: Final = torch.int64


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


def test_full_key_reads_an_integer_and_a_missing_cell_is_zero() -> None:
    t = _t()
    assert t["flops", "primal", "matmul", BF] == 60
    assert t["flops", "primal", "sort", F32] == 0


@pytest.mark.parametrize("value", [1.5, True, -1])
def test_cost_rejects_invalid_cells(value: object) -> None:
    expected = (
        f"cell must be nonnegative; got {value}."
        if value == -1
        else f"cell must be an integer; got {value!r}."
    )
    with pytest.raises((TypeError, ValueError)) as exc_info:
        Cost(cells={("flops", "primal", "matmul", BF): cast(int, value)})
    assert str(exc_info.value) == expected


@pytest.mark.parametrize("field", ["params", "params_active", "bytes_state"])
@pytest.mark.parametrize("value", [1.5, True, -1])
def test_cost_rejects_invalid_owned_counts(field: str, value: object) -> None:
    expected = (
        f"{field} must be nonnegative; got {value}."
        if value == -1
        else f"{field} must be an integer; got {value!r}."
    )
    with pytest.raises((TypeError, ValueError)) as exc_info:
        (
            Cost(params=cast(int, value))
            if field == "params"
            else Cost(params_active=cast(int, value))
            if field == "params_active"
            else Cost(bytes_state=cast(int, value))
        )
    assert str(exc_info.value) == expected


def test_partial_keys_slice_and_drop_the_fixed_leading_axes() -> None:
    t = _t()
    assert t["flops", "primal"].cells == {
        ("matmul", BF): 60,
        ("elementwise", F32): 4,
    }
    assert t["flops", "adjoint", "matmul"].cells == {(BF,): 120}
    assert t["flops"].params == 0


def test_axes_are_named_by_their_values_in_any_order() -> None:
    t = _t()
    assert t["flops", "matmul"].cells == {
        ("primal", BF): 60,
        ("adjoint", BF): 120,
    }
    assert t[I64].cells == {("bytes", "adjoint", "selection"): 2}
    assert t[F32].sum() == 4


def test_an_unknown_axis_value_is_rejected_by_name() -> None:
    with pytest.raises(KeyError, match="'gemm' is not a measure"):
        _ = _t()["gemm"]


def test_an_empty_table_still_rejects_an_unknown_axis_value() -> None:
    with pytest.raises(KeyError, match="'gemm' is not a measure"):
        _ = Cost()["gemm"]
    with pytest.raises(KeyError, match="'gemm' is not a measure"):
        _ = Report()["gemm"]


def test_an_empty_table_answers_by_the_axes_it_was_asked() -> None:
    assert Cost()["flops", "primal", "matmul", BF] == 0
    assert Cost()["flops", "primal"] == Cost()
    assert Report()["primal", "matmul", BF] == 0.0
    assert Report()["primal"] == Report()


def test_one_axis_named_twice_is_rejected() -> None:
    with pytest.raises(KeyError, match="'adjoint' names the phase axis twice"):
        _ = _t()["primal", "adjoint"]


def test_missing_axis_error_names_the_requested_axis() -> None:
    partial = Cost(cells={("flops", "primal", "matmul"): 1})

    with pytest.raises(KeyError) as exc_info:
        _ = partial[torch.float64]

    assert exc_info.value.args == ("torch.float64 names no axis of a 3-axis table.",)


def test_a_dtype_may_be_named_by_string_or_alias() -> None:
    t = _t()
    assert t["flops", "primal", "matmul", "bfloat16"] == 60
    assert t["flops", "primal", "matmul", "bf16"] == 60
    assert t["bf16", "flops"].cells == t[BF, "flops"].cells
    assert t["f32", "flops"].cells == t[F32, "flops"].cells


def test_cost_has_only_execution_measures() -> None:
    c = Cost(cells={("flops", "primal", "matmul", BF): 60})
    assert MEASURES == ("flops", "bytes")
    with pytest.raises(KeyError, match="intensity") as exc_info:
        _ = c["intensity"]
    assert exc_info.value.args == ("'intensity' is a reporting-only measure",)
    with pytest.raises(ValueError, match="reporting intensity") as exc_info:
        Cost(cells={("intensity", "primal", "matmul", BF): 1})
    assert str(exc_info.value) == ("Cost cells cannot contain reporting intensity.")


def test_intensity_is_a_separate_float_reporting_table() -> None:
    c = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 60,
            ("bytes", "primal", "matmul", BF): 4,
            ("flops", "adjoint", "matmul", BF): 120,
            ("bytes", "adjoint", "matmul", BF): 4,
            ("bytes", "primal", "selection", I64): 2,
        },
    )
    ratio = intensity(c)
    assert ratio["primal", "matmul", BF] == 15
    assert ratio["matmul", BF].cells == {
        ("primal",): 15.0,
        ("adjoint",): 30.0,
    }
    assert ratio["primal", "selection", I64] == 0.0
    assert all(isinstance(value, float) for value in ratio.cells.values())
    assert intensity(Cost())["primal", "matmul", BF] == 0.0


def test_intensity_of_work_without_traffic_is_infinite() -> None:
    result = intensity(
        Cost(cells={("flops", "primal", "sort", BF): 5}),
    )
    assert result["primal", "sort", BF] == math.inf


def test_only_keeps_one_axis_value_without_dropping_the_axis() -> None:
    """Unlike indexing, ``only`` keeps the key shape so the result adds back."""
    t = _t()
    primal = t.only("primal")
    assert primal.cells == {
        ("flops", "primal", "matmul", BF): 60,
        ("flops", "primal", "elementwise", F32): 4,
    }
    assert primal.params == 0
    assert t.only("adjoint") + primal == Cost(cells=t.cells)
    assert t.only("primal").only("adjoint") == Cost()
    with pytest.raises(KeyError, match="'gemm' is not"):
        t.only("gemm")


def test_relabel_moves_cells_to_another_axis_value() -> None:
    """A frozen layer's adjoint is its primal relabeled; the two then add."""
    t = _t()
    frozen = t.only("primal") + t.only("primal").relabel("adjoint")
    assert frozen["flops", "adjoint", "matmul", BF] == 60
    assert frozen["flops", "adjoint", "selection", I64] == 0
    assert frozen.params == 0


def test_empty_cost_only_and_relabel_still_validate_axis_names() -> None:
    for transform in (Cost.only, Cost.relabel):
        with pytest.raises(KeyError) as exc_info:
            transform(Cost(), "gemm")

        assert exc_info.value.args == (
            "'gemm' is not a measure, phase, kernel, dtype or device.",
        )


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


def test_cost_has_no_division_operator() -> None:
    assert "__truediv__" not in Cost.__dict__
    assert "__sub__" not in Cost.__dict__


def test_equality_ignores_explicit_zero_cells() -> None:
    assert Cost(cells={("flops", "primal", "matmul", BF): 0}) == Cost()


def test_cost_is_one_table_and_three_owned_integers() -> None:
    names = [f.name for f in fields(Cost)]
    assert names == ["cells", "params", "params_active", "bytes_state"]


def test_cost_add_sums_cells_and_ownership() -> None:
    a = Cost(
        cells={("flops", "primal", "matmul", F32): 1},
        params=3,
        params_active=2,
        bytes_state=7,
    )
    b = Cost(
        cells={("flops", "primal", "matmul", F32): 10},
        params=4,
        params_active=3,
    )
    assert a + b == Cost(
        cells={("flops", "primal", "matmul", F32): 11},
        params=7,
        params_active=5,
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

    stateful = Cost(bytes_state=3).tile(2)
    assert stateful.bytes_state == 6


def test_matmul_cost_counts_the_whole_invocation_in_integers() -> None:
    c = matmul_cost(
        channels_in=2,
        channels_out=3,
        bias=True,
        rows=4,
        dtype=BF,
    )
    assert c["flops", "primal", "matmul", BF] == 48
    assert c["flops", "adjoint", "matmul", BF] == 96
    assert c["flops", "primal", "elementwise", BF] == 12
    assert c["flops", "adjoint", "reduction", BF] == 9
    assert c["bytes", "primal", "matmul", BF] == 2 * (4 * 2 + 4 * 3 + 6)
    assert c["bytes", "adjoint", "matmul", BF] == 2 * 2 * (4 * 2 + 4 * 3 + 6)
    assert c["bytes", "primal", "elementwise", BF] == 2 * (2 * 4 * 3 + 3)
    assert c["bytes", "adjoint", "reduction", BF] == 2 * (4 * 3 + 3)
    assert all(isinstance(value, int) for value in c.cells.values())
    assert c["bytes", F32] == Cost()
    assert c.params == c.params_active == 9


def test_matmul_cost_prices_a_packed_weight_at_its_stored_bytes() -> None:
    # 12 weights at 2.5 bits plus two fp16 group scales: 30 bits + 4 bytes.
    packed = 4 + 4
    c = matmul_cost(
        channels_in=4,
        channels_out=3,
        rows=2,
        dtype=BF,
        weight_bytes=packed,
        dequant_flops=3,
    )
    moved = 2 * (2 * 4 + 2 * 3) + packed
    assert c["bytes", "primal", "matmul", BF] == moved
    assert c["bytes", "adjoint", "matmul", BF] == 2 * moved
    assert c["flops", "primal", "matmul", BF] == 2 * 2 * 12
    assert c["flops", "primal", "elementwise", BF] == 3 * 12
    assert c["bytes", "primal", "elementwise"] == Cost()
    assert c.params == c.params_active == 12


def test_matmul_cost_dequantizes_once_per_invocation_not_per_row() -> None:
    one = matmul_cost(channels_in=8, channels_out=8, rows=1, weight_bytes=40)
    many = matmul_cost(channels_in=8, channels_out=8, rows=64, weight_bytes=40)
    assert one["bytes", "primal", "matmul"].sum() - 4 * 16 == 40
    assert many["bytes", "primal", "matmul"].sum() - 4 * 64 * 16 == 40
    dq = matmul_cost(
        channels_in=8,
        channels_out=8,
        rows=3,
        weight_bytes=40,
        dequant_flops=5,
    )
    assert dq["flops", "primal", "elementwise"].sum() == 5 * 8 * 8


def test_matmul_cost_rejects_dequant_without_a_packed_weight() -> None:
    with pytest.raises(ValueError, match="dequant_flops") as exc_info:
        matmul_cost(channels_in=2, channels_out=3, dequant_flops=1)
    assert str(exc_info.value) == (
        "dequant_flops needs a packed matrix; set weight_bytes."
    )


def test_packed_weight_moves_decode_toward_the_memory_roofline() -> None:
    dense = matmul_cost(channels_in=4096, channels_out=4096, dtype=BF)
    packed = matmul_cost(
        channels_in=4096,
        channels_out=4096,
        dtype=BF,
        weight_bytes=4096 * 4096 * 5 // 16,
    )
    dense_bytes = dense["bytes", "primal"].sum()
    packed_bytes = packed["bytes", "primal"].sum()
    assert packed_bytes * 6 < dense_bytes


@pytest.mark.parametrize("rows", [1.5])
def test_matmul_cost_rejects_fractional_execution_rows(rows: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        matmul_cost(channels_in=2, channels_out=3, rows=cast(int, rows))


def test_dtype_defaults_to_the_torch_default() -> None:
    c = matmul_cost(channels_in=2, channels_out=3)
    assert c["bytes", "primal", "matmul", torch.get_default_dtype()] == 4 * (2 + 3 + 6)
    assert c["bytes", "primal", "matmul", BF] == 0


def test_elementwise_and_reduction_costs_carry_the_dtype() -> None:
    ew = elementwise_cost(
        primal=28,
        adjoint=44,
        channels=5,
        params=3,
        rows=4,
        dtype=BF,
    )
    assert ew["flops", "primal", "elementwise", BF] == 28
    assert ew["flops", "adjoint", "elementwise", BF] == 44
    assert ew["flops", "adjoint", "reduction", BF] == 9
    assert ew["bytes", "primal", "elementwise", BF] == 2 * (4 * 10 + 3)
    assert ew["bytes", "adjoint", "elementwise", BF] == 2 * (4 * 15 + 4 * 3 + 3)
    assert ew["bytes", "adjoint", "reduction", BF] == 2 * (4 * 3 + 3)
    assert ew.params == ew.params_active == 3
    red = reduction_cost(input_elements=12, output_groups=3, dtype=I64)
    assert red["flops", "primal", "reduction", I64] == 9
    assert red["bytes", "primal", "reduction", I64] == 8 * 15
    assert red["flops", "adjoint"] == Cost()


def test_intensity_reads_as_a_ratio_of_slices() -> None:
    c = matmul_cost(channels_in=2, channels_out=3, rows=4, dtype=BF)
    report = intensity(c)
    assert report["primal", "matmul", BF] == 48 / 52
    assert report["adjoint", "matmul", BF] == 96 / 104


# -- primitives --------------------------------------------------------------


def test_matmul_cost_without_a_weight_owns_nothing_but_reads_both_operands() -> None:
    c = matmul_cost(channels_in=2, channels_out=3, weight=False, rows=4)
    assert c.params == c.params_active == 0
    assert c["flops", "primal", "matmul"].sum() == 48
    assert c["bytes", "primal", "matmul"].sum() == 4 * (4 * 2 + 4 * 3 + 6)


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


def test_elementwise_default_channel_count_is_zero() -> None:
    cost = elementwise_cost(primal=1, adjoint=2, dtype=BF)
    assert cost["bytes", "primal", "elementwise", BF] == 0
    assert cost["bytes", "adjoint", "elementwise", BF] == 0


def test_elementwise_rejects_zero_rows() -> None:
    with pytest.raises(ValueError, match="rows must be at least one\\.") as exc_info:
        elementwise_cost(primal=0, adjoint=0, rows=0)

    assert exc_info.value.args == ("rows must be at least one.",)


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


def test_traffic_keeps_the_requested_dtype_and_prices_its_itemsize() -> None:
    result = traffic("primal", "selection", elements=5, dtype=BF, flops=2)
    assert result["flops", "primal", "selection", BF] == 2
    assert result["bytes", "primal", "selection", BF] == 10
    assert result["bytes", "primal", "selection", F32] == 0


def test_reduction_empty_group_writes_identity_without_arithmetic() -> None:
    c = reduction_cost(input_elements=0)
    assert c["flops", "primal", "reduction", F32] == 0
    assert c["bytes", "primal", "reduction", F32] == 4


def test_reduction_in_the_adjoint_phase() -> None:
    c = reduction_cost(input_elements=6, output_groups=2, phase="adjoint")
    assert c["flops", "adjoint", "reduction", F32] == 4
    assert c["flops", "primal", "reduction", F32] == 0


GEOMETRY_PRIMITIVES: Final[list[Callable[[int], Cost]]] = [
    lambda n: matmul_cost(channels_in=2, channels_out=3, rows=n),
    lambda n: elementwise_cost(primal=n, adjoint=1),
    lambda n: reduction_cost(input_elements=n),
    lambda n: traffic("primal", "elementwise", elements=n),
    lambda n: Cost().tile(repetitions=n),
    lambda n: Cost().tile(1, copies=n),
]


@pytest.mark.parametrize("primitive", GEOMETRY_PRIMITIVES)
def test_primitives_reject_fractional_geometry(
    primitive: Callable[[int], Cost],
) -> None:
    with pytest.raises(TypeError, match="integer"):
        primitive(cast(int, cast(object, 1.5)))


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (0, "rows must be at least one."),
        (-1, "rows must be at least one."),
        (math.nan, "rows must be an integer; got nan."),
        (math.inf, "rows must be an integer; got inf."),
    ],
)
def test_matmul_rejects_invalid_row_count(rows: object, expected: str) -> None:
    with pytest.raises((TypeError, ValueError)) as exc_info:
        matmul_cost(channels_in=2, channels_out=3, rows=cast(int, rows))
    assert str(exc_info.value) == expected


def test_tile_rejects_each_negative_count_independently() -> None:
    with pytest.raises(ValueError, match="nonnegative") as exc_info:
        Cost().tile(-1, copies=0)
    assert str(exc_info.value) == "repetitions and copies must be nonnegative."
    with pytest.raises(ValueError, match="nonnegative") as exc_info:
        Cost().tile(0, copies=-1)
    assert str(exc_info.value) == "repetitions and copies must be nonnegative."


@pytest.mark.parametrize("count", [1.5, True])
def test_cost_tile_reports_the_invalid_integer_field(count: object) -> None:
    with pytest.raises(TypeError, match="repetitions") as exc_info:
        Cost().tile(cast(int, count))
    assert str(exc_info.value) == (f"repetitions must be an integer; got {count!r}.")
    with pytest.raises(TypeError, match="copies") as exc_info:
        Cost().tile(1, copies=cast(int, count))
    assert str(exc_info.value) == f"copies must be an integer; got {count!r}."


def test_empty_cost_and_report_indexing_preserves_scalar_and_slice_shapes() -> None:
    empty = Cost()
    assert empty["flops"] == Cost()
    assert empty["flops", "primal", "matmul"] == Cost()
    assert empty["flops", "primal", "matmul", BF] == 0
    report = Report()
    assert report["h100", BF] == Report()
    assert report["h100", BF, "intensity"] == 0.0


def test_relabel_adds_cells_that_collide_at_the_destination() -> None:
    source = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 2,
            ("flops", "adjoint", "matmul", BF): 3,
        },
    )
    assert source.relabel("adjoint") == Cost(
        cells={("flops", "adjoint", "matmul", BF): 5},
    )


def test_report_retains_nan_cells_and_discards_zero_cells() -> None:
    nan = Report(cells={("h100", BF, "intensity", "matmul"): math.nan})
    assert len(nan.cells) == 1
    assert math.isnan(nan.sum())
    assert Report(cells={("h100", BF, "intensity", "matmul"): 0.0}) == Report()


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
    assert hash(a) != hash(
        Cost(cells={("flops", "primal", "matmul", BF): 2}),
    )


def test_repr_is_a_grid_with_totals() -> None:
    c = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 64_000,
            ("bytes", "primal", "matmul", BF): 32,
            ("bytes", "primal", "selection", I64): 8,
            ("flops", "adjoint", "matmul", BF): 1_500_000_000,
        },
        params=7,
    )
    text = repr(c)
    lines = text.splitlines()
    assert lines[0] == "Cost(params=7, params_active=0, bytes_state=0)"
    measure, dtype, primal, selection, adjoint, total = lines[1:]
    assert measure.split() == ["flops", "bytes", "bytes", "intensity"]
    assert dtype.split() == ["bf16", "bf16", "int64", "bf16"]
    assert primal.split() == ["primal", "matmul", "64K", "32", "-", "2K"]
    assert selection.split() == ["primal", "selection", "-", "-", "8", "-"]
    assert adjoint.split() == ["adjoint", "matmul", "1.5G", "-", "-", "inf"]
    assert total.split() == ["total", "1.5G", "32", "8", "46.88M"]
    assert all(type(value) is int for value in c.cells.values())
    assert all("intensity" not in key for key in c.cells)


def test_repr_intensity_total_divides_totals_instead_of_adding_ratios() -> None:
    c = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 60,
            ("bytes", "primal", "matmul", BF): 4,
            ("flops", "adjoint", "matmul", BF): 120,
            ("bytes", "adjoint", "matmul", BF): 12,
        },
    )
    assert repr(c).splitlines()[-1].split() == ["total", "180", "16", "11.25"]
    assert repr(c[BF]).splitlines()[-1].split() == [
        "total",
        "180",
        "16",
        "11.25",
    ]
    assert repr(c["primal"]).splitlines()[-1].split() == ["matmul", "60", "4", "15"]


def test_sudoku_cost_repr_includes_derived_intensity() -> None:
    c = exp000().finalize().step.model.cost(batch_size=1, dtype=BF)
    text = repr(c)
    assert "intensity" in text.splitlines()[1]
    assert text.splitlines()[-1].startswith("total")
    assert all(type(value) is int for value in c.cells.values())


def test_report_repr_renders_float_reporting_cells() -> None:
    text = repr(
        Report(cells={("h100", BF, "intensity", "matmul"): 989 / 3.35}),
    )
    assert "intensity" in text
    assert "295.2" in text


def test_si_formatting_uses_inclusive_decimal_boundaries_and_signs() -> None:
    assert [_si(value) for value in (1_000, 1_000_000, 1_000_000_000, 1e12)] == [
        "1K",
        "1M",
        "1G",
        "1T",
    ]
    assert _si(-1_000) == "-1K"


def test_cost_error_names_for_every_geometry_and_owned_value() -> None:
    cases = (
        (lambda: elementwise_cost(primal=-1, adjoint=0), "primal"),
        (lambda: elementwise_cost(primal=0, adjoint=-1), "adjoint"),
        (lambda: elementwise_cost(primal=0, adjoint=0, channels=-1), "channels"),
        (lambda: elementwise_cost(primal=0, adjoint=0, params=-1), "params"),
        (lambda: elementwise_cost(primal=0, adjoint=0, rows=-1), "rows"),
        (lambda: elementwise_cost(primal=0, adjoint=0, inputs=-1), "inputs"),
        (lambda: elementwise_cost(primal=0, adjoint=0, outputs=-1), "outputs"),
        (
            lambda: elementwise_cost(primal=0, adjoint=0, adjoint_inputs=-1),
            "adjoint_inputs",
        ),
        (
            lambda: elementwise_cost(primal=0, adjoint=0, adjoint_outputs=-1),
            "adjoint_outputs",
        ),
        (lambda: matmul_cost(channels_in=-1, channels_out=2), "channels_in"),
        (lambda: matmul_cost(channels_in=2, channels_out=-1), "channels_out"),
        (
            lambda: matmul_cost(channels_in=2, channels_out=2, weight_bytes=-1),
            "weight_bytes",
        ),
        (
            lambda: matmul_cost(channels_in=2, channels_out=2, dequant_flops=-1),
            "dequant_flops",
        ),
        (lambda: reduction_cost(input_elements=-1), "input_elements"),
        (lambda: reduction_cost(input_elements=1, output_groups=-1), "output_groups"),
        (lambda: traffic("primal", "sort", elements=1, flops=-1), "flops"),
        (lambda: traffic("primal", "sort", elements=-1), "elements"),
    )
    for call, name in cases:
        with pytest.raises(ValueError, match="nonnegative") as exc_info:
            call()
        assert str(exc_info.value) == f"{name} must be nonnegative; got -1."


def test_tile_zero_repetitions_and_copies_are_independent() -> None:
    original = Cost(
        cells={("flops", "primal", "matmul", BF): 7},
        params=3,
        params_active=2,
        bytes_state=5,
    )
    assert original.tile(0) == Cost(params=3, params_active=2)
    assert original.tile(2, copies=0) == Cost(
        cells={("flops", "primal", "matmul", BF): 14},
        bytes_state=10,
    )


def test_grid_orders_dtype_columns_by_name_when_item_sizes_tie() -> None:
    text = repr(
        Report(
            cells={
                ("h100", "intensity", "matmul", F32): 3.0,
                ("h100", "intensity", "matmul", BF): 2.0,
            },
        ),
    )
    assert text.splitlines()[0].split() == ["intensity", "intensity"]
    assert text.splitlines()[1].split() == ["bf16", "f32"]


def test_report_sum_totals_all_values_and_empty_report() -> None:
    assert (
        Report(
            cells={
                ("h100", BF, "intensity", "matmul"): 2.5,
                ("h100", F32, "intensity", "sort"): 3.5,
            },
        ).sum()
        == 6.0
    )
    assert type(Report().sum()) is float
    assert Report().sum() == 0.0


def test_axis_kinds_and_column_orders_are_exact() -> None:
    assert [_kind(axis) for axis in (BF, "flops", "primal", "matmul", "h100")] == [
        "dtype",
        "measure",
        "phase",
        "kernel",
        "device",
    ]
    assert _column_order(BF) == (2, "torch.bfloat16")
    assert _column_order("matmul") == (5, "matmul")
    assert _column_order("unlisted") == (-1, "unlisted")


def test_grid_formats_unmeasured_and_rows_only_tables_exactly() -> None:
    assert _grid({("left", "right"): 1}) == "     right\nleft     1"
    assert _grid({("flops", "primal"): 12}) == "       flops\nprimal    12"


def test_grid_formats_a_zero_axis_table() -> None:
    assert _grid({(): 12}) == "12"


def test_grid_sorts_unrecognized_labels_before_known_axes() -> None:
    text = _grid({("zebra", "value"): 2, ("adjoint", "value"): 1})
    assert [line.split() for line in text.splitlines()] == [
        ["value"],
        ["zebra", "2"],
        ["adjoint", "1"],
        ["total", "3"],
    ]


def test_grid_derives_each_dtype_intensity_from_its_matching_byte_column() -> None:
    result = Cost(
        cells={
            ("bytes", "primal", "matmul", BF): 2,
            ("flops", "primal", "matmul", F32): 6,
            ("bytes", "primal", "matmul", F32): 3,
        },
    )
    lines = [
        line.split() for line in _grid(result.cells, derive_intensity=True).splitlines()
    ]
    assert lines[-1] == ["primal", "matmul", "6", "2", "3", "2"]


def test_grid_drops_globally_empty_measure_columns() -> None:
    text = _grid({("flops", BF): 0, ("bytes", F32): 1})
    lines = [line.split() for line in text.splitlines()]
    assert lines[0] == ["bytes"]
    assert sorted(lines[1:-1]) == [
        ["torch.bfloat16", "-"],
        ["torch.float32", "1"],
    ]
    assert lines[-1] == ["total", "1"]


def test_division_handles_nan_and_signed_zero_denominators() -> None:
    assert math.isnan(_div(math.nan, 0.0))
    assert _div(3.0, -0.0) == -math.inf
    assert _div(-3.0, -0.0) == math.inf


def test_report_grid_omits_a_total_for_intensity_but_cost_grid_includes_it() -> None:
    cells: dict[tuple[object, ...], float] = {
        ("h100", "intensity", "matmul", BF): 2.0,
        ("h100", "intensity", "sort", BF): 3.0,
    }
    report_lines = repr(Report(cells=cells)).splitlines()
    assert report_lines[0].split() == ["intensity"]
    assert all(not line.startswith("total") for line in report_lines[2:])
    assert (
        repr(
            Cost(
                cells={
                    ("flops", "primal", "matmul", BF): 2,
                    ("flops", "adjoint", "matmul", BF): 3,
                },
            ),
        )
        .splitlines()[-1]
        .split()[0]
        == "total"
    )


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


@pytest.mark.parametrize("use_report", [False, True])
def test_utilization_without_a_duration_is_intensity_over_the_ridge(
    use_report: bool,
) -> None:
    c = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 2 * 989_000_000_000_000,
            ("bytes", "primal", "matmul", BF): 3_350_000_000_000,
            ("flops", "adjoint", "sort", F32): 1,
            ("bytes", "adjoint", "sort", F32): 1,
            ("flops", "primal", "matmul", torch.int32): 1,
            ("bytes", "primal", "matmul", torch.int32): 1,
        },
    )
    device: str | Report = peak()["h100"] if use_report else "h100"
    ratio = utilization(c, device=device)
    assert ratio["primal", "matmul", BF] == pytest.approx(2)
    assert ratio["adjoint", "sort", F32] == pytest.approx(3.35 / 67)
    assert ratio["primal", "matmul", torch.int32] == math.inf


@pytest.mark.parametrize("use_report", [False, True])
def test_utilization_with_a_duration_is_achieved_over_the_roofline_ceiling(
    use_report: bool,
) -> None:
    # 8 tokens a step: one compute-bound matmul, one memory-bound sort.
    c = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 989_000_000_000_000,
            ("bytes", "primal", "matmul", BF): 1,
            ("flops", "adjoint", "sort", F32): 1,
            ("bytes", "adjoint", "sort", F32): 1,
        },
    )
    device: str | Report = peak()["h100"] if use_report else "h100"
    achieved = utilization(c, device=device, duration_sec=2)
    assert achieved["primal", "matmul", BF] == pytest.approx(0.5)
    # Sort at intensity 1 is capped at bandwidth, 3.35e12 FLOP/s, not 67e12.
    assert achieved["adjoint", "sort", F32] == pytest.approx(1 / 2 / 3.35e12)


def test_utilization_treats_missing_bandwidth_as_zero() -> None:
    cost = Cost(
        cells={
            ("flops", "primal", "elementwise", F32): 2,
            ("bytes", "primal", "elementwise", F32): 1,
        },
    )
    ceiling = Report(cells={(F32, "flops", "elementwise"): 1.0})

    assert (
        utilization(cost, device=ceiling, duration_sec=1.0)[
            "primal",
            "elementwise",
            F32,
        ]
        == math.inf
    )


def test_utilization_handles_off_roofline_dtypes_and_memory_bound_work() -> None:
    no_dtype_peak = Cost(
        cells={("flops", "primal", "matmul", torch.complex64): 10},
    )
    assert (
        utilization(
            no_dtype_peak,
            device="h100",
            duration_sec=2,
        )["primal", "matmul", torch.complex64]
        == math.inf
    )

    memory_bound = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 1,
            ("bytes", "primal", "matmul", BF): 1_000,
        },
    )
    achieved = utilization(memory_bound, device="h100", duration_sec=1)
    assert achieved["primal", "matmul", BF] == pytest.approx(1 / 3.35e9)


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


def test_peak_intensity_is_the_ridge() -> None:
    assert peak()["b200", "matmul", "fp4", "intensity"] == 9000 / 8
    assert peak()["rtx5090", "elementwise", F32, "intensity"] == 104.8 / 1.792


def test_peak_names_a_form_factor_without_moving_the_bare_name() -> None:
    # A bare name keeps meaning the form factor the table was sourced from, so
    # every recorded result is unchanged by the form-factor entries existing.
    # The table holds TB/s: 2.039 * 1e12 rounds to 2039000000000.0002, not 2.039e12.
    assert peak()["a100", BF, "bytes", "matmul"] == 2.039 * 1e12
    assert peak()["h100", BF, "bytes", "matmul"] == 3.35e12
    # The other forms move less memory. A100 compute is identical across forms;
    # the H100 PCIe board clocks lower, so its datasheet rates are lower too.
    assert peak()["a100-40g", BF, "bytes", "matmul"] == 1.555 * 1e12
    assert peak()["a100-80g-pcie", BF, "bytes", "matmul"] == 1.935 * 1e12
    assert peak()["h100-pcie", BF, "bytes", "matmul"] == 2.0e12
    assert (
        peak()["a100-40g", BF, "flops", "matmul"]
        == peak()["a100", BF, "flops", "matmul"]
    )
    assert peak()["h100-pcie", BF, "flops", "matmul"] == 756.5e12
    assert peak()["h100", BF, "flops", "matmul"] == 989e12
    assert peak()["h100-pcie", BF, "flops", "elementwise"] == 51e12


def test_peak_ridge_moves_with_the_form_factor() -> None:
    # The whole point of naming the form: the same model reads as differently
    # tuned on the two boards, which is what picking the wrong ridge would hide.
    sxm = peak()["h100", BF, "intensity", "matmul"]
    pcie = peak()["h100-pcie", BF, "intensity", "matmul"]
    assert sxm == 989 / 3.35
    assert pcie == 756.5 / 2.0
    # Sudoku exp000's transformer sits at intensity 272.
    assert 272 / sxm - 1 == pytest.approx(-0.07866532, abs=1e-6)
    assert 272 / pcie - 1 < -0.25
    a100_sxm = peak()["a100", BF, "intensity", "matmul"]
    a100_40g = peak()["a100-40g", BF, "intensity", "matmul"]
    assert a100_40g / a100_sxm == pytest.approx(2.039 / 1.555)
    a100_pcie = peak()["a100-80g-pcie", BF, "intensity", "matmul"]
    assert a100_pcie / a100_sxm == pytest.approx(2.039 / 1.935)


# -- dispatch ----------------------------------------------------------------


class _Costed:
    class Config(Fig["_Costed"]):
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


class _Uncosted:
    class Config(Fig["_Uncosted"]):
        pass

    def __init__(self, config: Config) -> None:
        del config


def test_cost_calls_the_protocol() -> None:
    assert isinstance(_Costed.Config(), HasCost)
    assert cost(_Costed.Config(), seq_len=8, batch_size=1, dtype=None) == Cost(params=4)


class _Rows:
    """A costed config that reports the sharing rows it was handed."""

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
        == 9
    )


def test_a_costed_function_is_a_cost_leaf() -> None:
    """A plain callable carries its per-element cost the way a Config does."""

    @set_cost(map_cost(primal=5, adjoint=5, adjoint_inputs=2))
    def act(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    assert act(torch.zeros(2)).shape == (2,)
    c = cost(act, channels=3, dtype=BF)
    assert c["flops", "primal", "elementwise", BF] == 15
    assert c["flops", "adjoint", "elementwise", BF] == 15
    assert c["bytes", "primal", "elementwise", BF] == 2 * (3 + 3)
    assert c["bytes", "adjoint", "elementwise", BF] == 2 * (2 * 3 + 3)
    assert c.params == 0


def test_set_cost_keeps_no_reference_to_the_function() -> None:
    """The cost travels on the function; no registry outlives it."""

    @set_cost(map_cost(primal=1, adjoint=1))
    def act(x: torch.Tensor) -> torch.Tensor:
        return x

    alive = weakref.ref(act)
    del act
    gc.collect()

    assert alive() is None


def test_map_cost_forwards_every_operand_count() -> None:
    mapped = map_cost(
        primal=2,
        adjoint=3,
        inputs=2,
        outputs=4,
        adjoint_inputs=5,
        adjoint_outputs=3,
    )
    result = mapped(channels=3, dtype=BF)
    assert result["flops", "primal", "elementwise", BF] == 6
    assert result["flops", "adjoint", "elementwise", BF] == 9
    assert result["bytes", "primal", "elementwise", BF] == 2 * 3 * (2 + 4)
    assert result["bytes", "adjoint", "elementwise", BF] == 2 * 3 * (5 + 3)

    unary = map_cost(primal=2, adjoint=3)(channels=3, dtype=BF)
    assert unary["bytes", "adjoint", "elementwise", BF] == 2 * 3 * 3


def test_a_partial_of_a_costed_function_keeps_its_cost() -> None:
    @set_cost(map_cost(primal=3, adjoint=2))
    def shifted(x: torch.Tensor, *, threshold: float) -> torch.Tensor:
        return torch.relu(x - threshold).square()

    bound = functools.partial(shifted, threshold=0.75)
    assert cost(bound, channels=4, dtype=F32) == cost(
        shifted,
        channels=4,
        dtype=F32,
    )


def test_cost_prefers_a_cost_method_on_a_partial_over_the_function_registry() -> None:
    @set_cost(map_cost(primal=1, adjoint=1))
    def activation(x: torch.Tensor) -> torch.Tensor:
        return x

    class CostedPartial(functools.partial[torch.Tensor]):
        def cost(self, **kwargs: object) -> Cost:
            del kwargs
            return Cost(params=7)

    bound = CostedPartial(activation)
    assert cost(bound, channels=3, dtype=BF) == Cost(params=7)


def test_an_uncosted_function_is_rejected_by_name() -> None:
    def gelu(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

    with pytest.raises(TypeError) as exc_info:
        cost(gelu, channels=4, dtype=None)

    assert str(exc_info.value) == (
        f"{gelu.__qualname__} has no cost(); every slot under a costed container "
        "must implement HasCost, or be a function decorated with @set_cost."
    )


def test_cost_rejects_a_config_without_a_cost() -> None:
    """A silent zero is the CPU-SDPA bug; a missing arm must raise, naming itself."""
    assert not isinstance(_Uncosted.Config(), HasCost)
    with pytest.raises(TypeError, match=r"_Uncosted\.Config"):
        cost(_Uncosted.Config(), seq_len=8, batch_size=1, dtype=None)


def test_every_model_config_is_priced() -> None:
    """A module Config that cannot cost itself fails here, not at a run's MFU line.

    Walks every ``nn.Module`` with a ``Fig`` Config under ``priml.model``,
    ``priml.baselines``, and ``priml.loss`` -- everything a
    ``TrainStep.Config`` model or loss slot can hold -- and asserts the
    ``HasCost`` protocol. torchtitan makes its FLOP method abstract on the
    model Config; this is the same gate without an ABC. ``private`` and
    ``scripts`` trees are one-off tooling, not modules a run costs.

    Attention kernels are the one exemption: a kernel config holds no shapes,
    so the owner of the projections costs it with
    :func:`attention_kernel_cost`; a kernel that costed itself would need
    the bus of shape arguments this design removed.
    """
    root = pathlib.Path(__file__).parent
    # The package prefix is read off this module's own name rather than
    # spelled out, so the walk resolves under either import root.
    package = __name__.rsplit(".", 1)[0]
    uncosted: list[str] = []
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
            if not issubclass(owner, torch.nn.Module) or [
                *inspect.signature(owner.forward).parameters,
            ][1:4] == ["q", "k", "v"]:
                continue
            config = getattr(owner, "Config", None)
            if not (inspect.isclass(config) and issubclass(config, Fig)):
                continue
            if not hasattr(config, "cost"):
                uncosted.append(f"{module_name}.{name}")
    assert uncosted == []


# -- leaves that exercise the silos ------------------------------------------


def test_unsupported_activation_fails_at_cost_not_at_forward() -> None:
    """An injected activation without a cost is a TypeError when costed."""
    ffn = SwiGLU.Config(4, 4)
    ffn.channels_hidden = 8
    ffn.act = torch.sin
    model = ffn.copy_tree().finalize().make()
    assert model(torch.randn(2, 4)).shape == (2, 4)
    with pytest.raises(TypeError, match="sin has no cost"):
        cost(ffn.copy_tree().finalize(), seq_len=1, batch_size=1, dtype=None)


def test_scalar_leaf_formulas_and_composition() -> None:
    linear = Linear.Config(2, 3)
    linear.bias = True
    counted = cost(linear, seq_len=4, batch_size=1, dtype=None)
    assert counted["flops", "primal", "elementwise"].sum() == 12
    assert counted["flops", "adjoint", "reduction"].sum() == 9
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
