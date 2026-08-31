"""Canonical tests for ``ResidualMix``.

Regenerate canonical artifacts through pytest so Priml's deterministic setup
applies::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest priml/model/residual_mix_test.py
"""

from __future__ import annotations

from pathlib import Path

from configgle.testing import assert_pprint_golden

import torch

from priml.model.residual_mix import ResidualMix
from priml.testing.bfb import assert_bfb_against_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


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
        golden_dir=_TESTDATA,
        golden_name="residual_mix",
        build_module=lambda: ResidualMix.Config(num_layers=2).make(),
        build_input=lambda: (
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
        ),
        seed=0,
        run=lambda module, inputs: module(
            inputs[0],
            original=inputs[1],
            layer=1,
            message=object(),
        ),
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
