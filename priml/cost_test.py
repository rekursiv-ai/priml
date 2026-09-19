"""Tests for priml.cost."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, cast

import functools
import importlib
import inspect
import math
import pathlib


if TYPE_CHECKING:
    from collections.abc import Callable

from configgle import Fig

import pytest
import torch

from priml.cost import (
    MEASURES,
    Cost,
    Report,
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


# -- Cost ------------------------------------------------------------------


BF = torch.bfloat16
F32 = torch.float32
I64 = torch.int64
_FRACTIONAL: object = 1.5


def _cost_with_owned_count(field: str, value: int) -> Cost:
    if field == "params":
        return Cost(params=value)
    if field == "params_active":
        return Cost(params_active=value)
    return Cost(bytes_state=value)


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


@pytest.mark.parametrize("value", [_FRACTIONAL, True, -1])
def test_cost_rejects_invalid_cells(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="cell"):
        Cost(cells={("flops", "primal", "matmul", BF): cast(int, value)})


@pytest.mark.parametrize("field", ["params", "params_active", "bytes_state"])
@pytest.mark.parametrize("value", [_FRACTIONAL, True, -1])
def test_cost_rejects_invalid_owned_counts(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        _cost_with_owned_count(field, cast(int, value))


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


def test_cost_has_only_execution_measures() -> None:
    c = Cost(cells={("flops", "primal", "matmul", BF): 60})
    assert MEASURES == ("flops", "bytes")
    with pytest.raises(KeyError, match="intensity"):
        _ = c["intensity"]
    with pytest.raises(ValueError, match="reporting intensity"):
        Cost(cells={("intensity", "primal", "matmul", BF): 1})


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
    assert isinstance(ratio, Report)
    assert ratio["primal", "matmul", BF] == 15
    assert ratio["matmul", BF].cells == {("primal",): 15.0, ("adjoint",): 30.0}
    assert ratio["primal", "selection", I64] == 0.0
    assert all(isinstance(value, float) for value in ratio.cells.values())
    assert intensity(Cost())["primal", "matmul", BF] == 0.0


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

    stateful = Cost(bytes_state=3).tile(2)
    assert stateful.bytes_state == 6


def test_matmul_cost_counts_the_whole_invocation_in_integers() -> None:
    c = matmul_cost(channels_in=2, channels_out=3, bias=True, rows=4, dtype=BF)
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


def test_matmul_cost_rejects_fractional_execution_rows() -> None:
    with pytest.raises(TypeError, match="integer"):
        matmul_cost(channels_in=2, channels_out=3, rows=cast(int, _FRACTIONAL))


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


@pytest.mark.parametrize(
    "call",
    [
        lambda: matmul_cost(channels_in=2, channels_out=3, rows=cast(int, _FRACTIONAL)),
        lambda: elementwise_cost(primal=cast(int, _FRACTIONAL), adjoint=1),
        lambda: reduction_cost(input_elements=cast(int, _FRACTIONAL)),
        lambda: traffic("primal", "elementwise", elements=cast(int, _FRACTIONAL)),
        lambda: Cost().tile(cast(int, _FRACTIONAL)),
        lambda: Cost().tile(1, copies=cast(int, _FRACTIONAL)),
    ],
)
def test_primitives_reject_fractional_geometry(call: Callable[[], object]) -> None:
    with pytest.raises(TypeError, match="integer"):
        call()


@pytest.mark.parametrize("rows", [0, -1, math.nan, math.inf])
def test_matmul_rejects_invalid_row_count(rows: object) -> None:
    with pytest.raises((TypeError, ValueError), match=r"rows|integer"):
        matmul_cost(channels_in=2, channels_out=3, rows=cast(int, rows))


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
    assert measure.split() == ["flops", "flops", "bytes", "bytes"]
    assert dtype.split() == ["bf16", "int64", "bf16", "int64"]
    assert primal.split() == ["primal", "matmul", "64K", "-", "32", "-"]
    assert selection.split() == ["primal", "selection", "-", "-", "-", "8"]
    assert adjoint.split() == ["adjoint", "matmul", "1.5G", "-", "-", "-"]
    assert total.split() == ["total", "1.5G", "-", "32", "8"]


def test_report_repr_renders_float_reporting_cells() -> None:
    text = repr(
        Report(cells={("h100", BF, "intensity", "matmul"): 989 / 3.35}),
    )
    assert "intensity" in text
    assert "295.2" in text


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
    ratio = utilization(c, device="h100")
    assert isinstance(ratio, Report)
    assert ratio["primal", "matmul", BF] == pytest.approx(2)
    assert ratio["adjoint", "sort", F32] == pytest.approx(3.35 / 67)
    assert ratio["primal", "matmul", torch.int32] == math.inf


def test_utilization_with_a_duration_is_achieved_over_the_roofline_ceiling() -> None:
    # 8 tokens a step: one compute-bound matmul, one memory-bound sort.
    c = Cost(
        cells={
            ("flops", "primal", "matmul", BF): 989_000_000_000_000,
            ("bytes", "primal", "matmul", BF): 1,
            ("flops", "adjoint", "sort", F32): 1,
            ("bytes", "adjoint", "sort", F32): 1,
        },
    )
    achieved = utilization(c, device="h100", duration_sec=2)
    assert isinstance(achieved, Report)
    assert achieved["primal", "matmul", BF] == pytest.approx(0.5)
    # Sort at intensity 1 is capped at bandwidth, 3.35e12 FLOP/s, not 67e12.
    assert achieved["adjoint", "sort", F32] == pytest.approx(1 / 2 / 3.35e12)


def test_peak_prices_matmul_per_dtype_and_every_other_silo_at_the_vector_rate() -> None:
    h100 = peak()["h100"]
    assert isinstance(h100, Report)
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


def test_a_partial_of_a_costed_function_keeps_its_cost() -> None:
    @set_cost(map_cost(primal=3, adjoint=2))
    def shifted(x: torch.Tensor, *, threshold: float) -> torch.Tensor:
        return torch.relu(x - threshold).square()

    bound = functools.partial(shifted, threshold=0.75)
    assert cost(bound, channels=4, dtype=F32) == cost(shifted, channels=4, dtype=F32)


def test_an_uncosted_function_is_rejected_by_name() -> None:
    def gelu(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)

    with pytest.raises(TypeError, match="gelu has no cost"):
        cost(gelu, channels=4, dtype=None)


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
            if not issubclass(owner, torch.nn.Module) or _is_attention_kernel(owner):
                continue
            config = getattr(owner, "Config", None)
            if not (inspect.isclass(config) and issubclass(config, Fig)):
                continue
            if not hasattr(config, "cost"):
                uncosted.append(f"{module_name}.{name}")
    assert uncosted == []


def _is_attention_kernel(owner: type[torch.nn.Module]) -> bool:
    """Whether ``owner`` is a kernel slot: called on ``(q, k, v)``, costed by its owner."""
    forward = inspect.signature(owner.forward)
    return [*forward.parameters][1:4] == ["q", "k", "v"]


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
