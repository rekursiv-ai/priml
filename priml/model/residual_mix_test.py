"""Canonical tests for ``ResidualMix``.

Regenerate canonical artifacts through pytest so Priml's deterministic setup
applies::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest priml/model/residual_mix_test.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden
from torch import Tensor, nn

import torch

from priml.model.residual_mix import ResidualMix
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def test_residual_mix_config_pprint() -> None:
    config = ResidualMix.Config(num_layers=2, running=0.5, original=0.25)
    assert_pprint_golden(
        test_file=__file__,
        name="residual_mix",
        config=config,
    )


def test_residual_mix_forward_and_open_kwargs() -> None:
    module = ResidualMix.Config(
        num_layers=2,
        running=0.5,
        original=0.25,
    ).make()
    x = torch.tensor([[2.0, 4.0]])
    original = torch.tensor([[8.0, 12.0]])

    output = module(x, original=original, layer=1, message=object())

    assert torch.equal(output, torch.tensor([[3.0, 5.0]]))


def test_residual_mix_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="residual_mix",
        build_module=lambda: ResidualMix.Config(num_layers=2).make(),
        build_input=lambda: (
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
        ),
        seed=0,
        run=_run_residual,
    )


def _run_residual(module: nn.Module, inputs: tuple[Tensor, Tensor]) -> Tensor:
    """Run the residual module for the golden harness."""
    residual = module
    assert isinstance(residual, ResidualMix)
    return residual(
        inputs[0],
        original=inputs[1],
        layer=1,
        message=object(),
    )


def test_residual_mix_cost_is_two_scalars_per_layer() -> None:
    """Two learned scalar weights per layer contribute only elementwise work."""
    config = ResidualMix.Config(num_layers=3)
    cost = assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randn(2, 4, requires_grad=True),
            torch.randn(2, 4, requires_grad=True),
        ),
        num_tokens=2,
        bus={"channels_in": 4},
        run=_mix_layer_one,
    )
    assert cost.training.flops.matmul == 0
    assert cost.primal.flops.elementwise == 3 * 4 * 3
    assert cost.adjoint.flops.elementwise == 4 * 4 * 3
    # Each scalar's gradient sums over the row, then over the two rows.
    assert cost.adjoint.flops.reduction == 2 * (4 - 1) * 3 + 2 * 3 * (2 - 1) / 2
    assert cost.params == 2 * 3
    assert cost.primal.bytes.elementwise == 4 * (7 * 4 * 3 + 2 * 3 / 2)
    assert cost.adjoint.bytes.elementwise == 4 * (10 * 4 * 3 + 2 * 3 / 2)
    assert cost.adjoint.bytes.reduction == 4 * (2 * (4 + 1) * 3 + 6 * (1 + 1 / 2))


def _mix_layer_one(module: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
    """Mix layer one of the running and original streams."""
    assert isinstance(module, ResidualMix)
    return module(inputs[0], original=inputs[1], layer=1)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
