"""Tests for softcap module."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.cost import Cost, cost
from priml.model.softcap import SoftCap
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def test_softcap_config_pprint() -> None:
    config = SoftCap.Config(cap=2.0, channels_in=4, channels_out=4)
    assert_pprint_golden(
        test_file=__file__,
        name="soft_cap",
        config=config,
    )


def test_softcap_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="soft_cap",
        build_module=lambda: SoftCap.Config(
            cap=2.0,
            channels_in=4,
            channels_out=4,
        ).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_softcap_infers_either_channel_boundary() -> None:
    assert SoftCap.Config(channels_in=4).finalize().channels_out == 4
    assert SoftCap.Config(channels_out=8).finalize().channels_in == 8


def test_softcap_bounds_output() -> None:
    m = SoftCap.Config(cap=5.0, channels_in=4, channels_out=4).make()
    # Saturating tanh reaches exactly 1.0 in float32, so the bound is closed.
    out = m(torch.randn(8, 4) * 100.0)
    assert out.abs().max() <= 5.0


def test_softcap_rejects_non_positive_or_infinite_cap() -> None:
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            _ = SoftCap.Config(cap=bad, channels_in=4, channels_out=4).make()


def test_softcap_rejects_integer_dtype() -> None:
    """An integer ``dtype`` silently severs the gradient, so it must raise.

    ``Tensor.to(torch.int64)`` is not differentiable: the cast returns a leaf
    with ``requires_grad=False``, so the wrapped projection stops training
    while the forward pass keeps producing plausible values -- measured as
    ``requires_grad`` False on the output of an otherwise ordinary readout.
    """
    with pytest.raises(ValueError, match="floating point"):
        _ = SoftCap.Config(
            cap=5.0,
            channels_in=4,
            channels_out=4,
            dtype=torch.int64,
        ).make()


def test_softcap_keeps_gradient_at_default_dtype() -> None:
    m = SoftCap.Config(cap=5.0, channels_in=4, channels_out=4).make()
    out = m(torch.randn(2, 4))
    assert out.requires_grad
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters())


def test_softcap_cost_separates_projection_and_squash() -> None:
    """The squash contributes scalar work independently of its projection."""
    config = SoftCap.Config(cap=5.0, channels_in=4, channels_out=6)
    model_cost = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randn(3, 4, requires_grad=True),
        seq_len=3,
        batch_size=1,
        dtype=None,
    )
    inner = cost(
        config.copy_tree().finalize().inner,
        seq_len=3,
        batch_size=1,
        dtype=None,
    )
    assert inner.params == 4 * 6
    f32 = torch.float32
    assert model_cost == inner + Cost(
        cells={
            ("flops", "primal", "elementwise", f32): 3 * 6 * 3,
            ("flops", "adjoint", "elementwise", f32): 5 * 6 * 3,
            ("bytes", "primal", "elementwise", f32): 4 * 6 * 6 * 3,
            ("bytes", "adjoint", "elementwise", f32): 4 * 7 * 6 * 3,
        },
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
