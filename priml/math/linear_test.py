"""Tests for linear math kernels."""

from __future__ import annotations

import torch

from priml.math.linear import cast_parameters_to_input


def test_cast_parameters_to_input_matches_explicit_cast_order() -> None:
    inputs = torch.randn(2, 3, dtype=torch.bfloat16)
    weight = torch.randn(4, 3)
    bias = torch.randn(4)

    actual = cast_parameters_to_input(inputs, weight, bias)
    expected = torch.nn.functional.linear(
        inputs,
        weight.to(inputs.dtype),
        bias.to(inputs.dtype),
    )

    assert torch.equal(actual, expected)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
