"""Tests for the squared-ReLU feed-forward."""

from __future__ import annotations

import pytest
import torch

from priml.model.relu_squared import ReluSquared


def test_a_fresh_block_is_the_identity_on_its_residual_stream() -> None:
    """The output projection is zero-initialized, which is the recipe.

    A stack of these starts as shallow as the task needs and deepens as
    training proceeds; a nonzero init would make every layer contribute from
    step one and change what the schedule is tuned against.
    """
    ffn = ReluSquared.Config(channels_in=8).make()
    assert torch.equal(ffn(torch.randn(2, 4, 8)), torch.zeros(2, 4, 8))


def test_the_nonlinearity_is_relu_squared() -> None:
    """Squared, not plain: the square is what carries what a gate otherwise
    would, so a plain ReLU is a different model at the same parameter count.
    """
    torch.manual_seed(0)
    ffn = ReluSquared.Config(channels_in=8).make()
    x = torch.randn(2, 4, 8)
    hidden = torch.relu(ffn.up_proj(x))
    torch.testing.assert_close(
        ffn.down_proj(hidden.square()),
        ffn(x),
        rtol=0,
        atol=0,
    )


def test_expansion_sets_the_hidden_width() -> None:
    """One matrix in and one out, so the hidden width is the only knob."""
    ffn = ReluSquared.Config(channels_in=8, expansion=3).make()
    assert ffn.up_proj.weight.shape == (24, 8)
    assert ffn.down_proj.weight.shape == (8, 24)


def test_an_uninherited_width_is_rejected() -> None:
    """The sentinel means the block never propagated one, which would
    otherwise surface as a zero-sized matmul deep in the forward.
    """
    with pytest.raises(ValueError, match="channels_in must be positive"):
        ReluSquared.Config().make()


def test_reset_parameters_reinitializes_both_projections() -> None:
    """Meta-device materialization drives init through this alone, so a
    projection it skips would train on ``to_empty``'s garbage.
    """
    torch.manual_seed(0)
    ffn = ReluSquared.Config(channels_in=8).make()
    with torch.no_grad():
        ffn.up_proj.weight.fill_(float("nan"))
        ffn.down_proj.weight.fill_(float("nan"))
    ffn.reset_parameters()
    assert not torch.isnan(ffn.up_proj.weight).any()
    assert not torch.isnan(ffn.down_proj.weight).any()


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
