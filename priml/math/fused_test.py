"""Compare whole fused-rounding outputs with the independent C arithmetic API."""

from __future__ import annotations

from functools import cache
from typing import Protocol, cast

import ctypes
import math

import pytest
import torch

from priml.math.fused import _two_sum, fma32
from priml.testing.bfb import host_agnostic_numerics


@pytest.mark.parametrize("canonical", [False, True])
def test_all_binary32_exponents_match_c_fmaf(canonical: bool) -> None:
    generator = torch.Generator().manual_seed(71)
    operands = torch.randint(
        -(1 << 31),
        1 << 31,
        (3, 4096),
        dtype=torch.int32,
        generator=generator,
    ).view(torch.float32)
    x, y, addend = operands.unbind()
    expected = torch.tensor(
        [
            _libm().fmaf(float(a), float(b), float(c))
            for a, b, c in zip(x, y, addend, strict=True)
        ],
        dtype=torch.float32,
    )
    if canonical:
        with host_agnostic_numerics():
            actual = fma32(x, y=y, addend=addend)
    else:
        actual = fma32(x, y=y, addend=addend)
    assert torch.equal(actual.isnan(), expected.isnan())
    assert torch.equal(
        actual[~expected.isnan()].view(torch.int32),
        expected[~expected.isnan()].view(torch.int32),
    )


def test_exact_midpoint_residual_changes_the_rounding_direction() -> None:
    x = torch.tensor([1 + 2**-23, 1 + 2**-23, -1 - 2**-23, -1 - 2**-23])
    y = torch.full_like(x, 1.5)
    addend = torch.tensor([-(2**-80), 2**-80, -(2**-80), 2**-80])
    expected = torch.tensor([1.5 + 2**-23, 1.5 + 2**-22, -1.5 - 2**-22, -1.5 - 2**-23])
    assert torch.equal(fma32(x, y=y, addend=addend), expected)
    assert not torch.equal(
        (x.double() * y.double() + addend.double()).float(),
        expected,
    )


def test_broadcasting_retains_signed_zero_subnormal_and_overflow_bits() -> None:
    # fma32 broadcasts a singleton axis against the independent trailing axis.
    x = torch.tensor([[0.0], [-0.0], [2**-149], [float.fromhex("0x1.fffffep+127")]])
    y = torch.tensor([0.5, 2.0, -1.0])
    addend = torch.tensor(-0.0)
    expected = torch.tensor(
        [[_libm().fmaf(float(a), float(b), -0.0) for b in y] for a in x.flatten()],
    )
    assert torch.equal(
        fma32(x, y=y, addend=addend).view(torch.int32),
        expected.view(torch.int32),
    )


def test_special_values_preserve_ieee_results() -> None:
    x = torch.tensor([math.inf, -math.inf, 0.0, math.nan, 2.0])
    y = torch.tensor([2.0, 2.0, math.inf, 1.0, 3.0])
    addend = torch.tensor([-math.inf, 1.0, 1.0, 1.0, math.inf])
    actual = fma32(x, y=y, addend=addend)
    expected = torch.tensor([math.nan, -math.inf, math.nan, math.nan, math.inf])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_other_operand_dtypes_are_rejected() -> None:
    for position in range(3):
        operands = [torch.ones(2) for _ in range(3)]
        operands[position] = operands[position].double()
        with pytest.raises(TypeError) as exc:
            fma32(operands[0], y=operands[1], addend=operands[2])
        assert str(exc.value) == "Fused binary32 arithmetic requires binary32 operands."


def test_fused_preprocessing_does_not_record_an_autograd_graph() -> None:
    x = torch.tensor([1.3, 2.7, -0.9], requires_grad=True)
    result = fma32(x, y=x, addend=x)
    assert not result.requires_grad
    assert result.grad_fn is None


def test_overflow_midpoint_retains_the_sign_of_a_lost_minimum_subnormal() -> None:
    x = torch.tensor(
        [18_631 * 2**50, 18_631 * 2**50, -18_631 * 2**50, -18_631 * 2**50],
        dtype=torch.float32,
    )
    y = torch.full_like(x, 1801 * 2**53)
    addend = torch.tensor([-(2**-149), 2**-149, -(2**-149), 2**-149])
    maximum = torch.finfo(torch.float32).max
    expected = torch.tensor([maximum, math.inf, -math.inf, -maximum])
    assert torch.equal(
        fma32(x, y=y, addend=addend).view(torch.int32),
        expected.view(torch.int32),
    )


def test_two_sum_retains_the_complete_roundoff_with_either_larger_operand() -> None:
    x = torch.tensor([2**80, 1, -(2**80), -1, 1.5, 2**80, 2**80], dtype=torch.float64)
    y = torch.tensor(
        [1, 2**80, -1, -(2**80), 2**-80, -(2**80), -1],
        dtype=torch.float64,
    )
    total, residual = _two_sum(x, y=y)
    expected_total = torch.tensor(
        [2**80, 2**80, -(2**80), -(2**80), 1.5, 0, 2**80],
        dtype=torch.float64,
    )
    expected_residual = torch.tensor([1, 1, -1, -1, 2**-80, 0, -1], dtype=torch.float64)
    assert torch.equal(total, expected_total)
    assert torch.equal(residual, expected_residual)


@cache
def _libm() -> _Libm:
    library = ctypes.CDLL(None)
    library.fmaf.restype = ctypes.c_float
    library.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    return cast("_Libm", library)


class _Libm(Protocol):
    def fmaf(self, x: float, y: float, addend: float, /) -> float: ...


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
