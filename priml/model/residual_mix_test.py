"""Canonical tests for ``ResidualMix``.

Regenerate canonical artifacts through pytest so Priml's deterministic setup
applies::

    uv --quiet run --frozen pytest priml/model/residual_mix_test.py --regenerate-b4b
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from torch import Tensor, nn

import torch

from priml.model.residual_mix import ResidualMix
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch
from priml.testing.golden import assert_pprint_golden


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
    def run(module: nn.Module, inputs: tuple[Tensor, Tensor]) -> Tensor:
        assert isinstance(module, ResidualMix)
        return module(inputs[0], original=inputs[1], layer=1, message=object())

    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="residual_mix",
        build_module=lambda: ResidualMix.Config(num_layers=2).make(),
        build_input=lambda: (torch.randn(2, 3, 4), torch.randn(2, 3, 4)),
        seed=0,
        run=run,
    )


def test_residual_mix_cost_is_two_scalars_per_layer() -> None:
    """Two learned scalar weights per layer contribute only elementwise work."""
    config = ResidualMix.Config(num_layers=3, channels_in=4)

    def run(module: nn.Module, inputs: tuple[Tensor, ...]) -> Tensor:
        assert isinstance(module, ResidualMix)
        return module(inputs[0], original=inputs[1], layer=1)

    cost = assert_cost_matches_torch(
        config,
        build_input=lambda: (
            torch.randn(2, 4, requires_grad=True),
            torch.randn(2, 4, requires_grad=True),
        ),
        seq_len=2,
        batch_size=1,
        dtype=None,
        run=run,
    )
    assert cost["flops", "matmul"].sum() == 0
    assert cost["flops", "primal", "elementwise"].sum() == 3 * 4 * 3 * 2
    assert cost["flops", "adjoint", "elementwise"].sum() == 4 * 4 * 3 * 2
    # Each scalar's gradient sums over the row, then over the two rows.
    assert cost["flops", "adjoint", "reduction"].sum() == 2 * (
        4 - 1
    ) * 3 * 2 + 2 * 3 * (2 - 1)
    assert cost.params == 2 * 3
    assert cost["bytes", "primal", "elementwise"].sum() == 4 * (7 * 4 * 3 * 2 + 2 * 3)
    assert cost["bytes", "adjoint", "elementwise"].sum() == 4 * (10 * 4 * 3 * 2 + 2 * 3)
    assert cost["bytes", "adjoint", "reduction"].sum() == 4 * (
        2 * (4 + 1) * 3 * 2 + 6 * (2 + 1)
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
