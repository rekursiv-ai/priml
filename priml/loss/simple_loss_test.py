"""Tests for SimpleLoss."""

from __future__ import annotations

from torch.nn import functional

import pytest
import torch

from priml.loss.simple_loss import SimpleLoss


def test_simple_loss_basic():
    """Test SimpleLoss with default binary_cross_entropy_with_logits."""
    cfg = SimpleLoss.Config()
    loss = cfg.make()

    prediction = torch.randn(2, 3)
    batch = {"label": torch.randn(2, 3)}

    result = loss(prediction, **batch)

    assert "loss" in result
    assert result["loss"].shape == (2, 3)  # reduction="none"


def test_simple_loss_with_kwargs():
    """Test SimpleLoss with additional kwargs."""
    cfg = SimpleLoss.Config(
        kwargs={
            "reduction": "none",
            "pos_weight": torch.tensor([2.0, 1.0, 1.0, 1.0, 1.0]),
        },
    )
    loss = cfg.make()

    prediction = torch.randn(4, 5)
    batch = {"label": torch.randn(4, 5)}

    result = loss(prediction, **batch)

    assert "loss" in result
    assert result["loss"].shape == (4, 5)


def test_simple_loss_custom_target_key():
    """Test SimpleLoss with custom target key."""
    cfg = SimpleLoss.Config(target_key="target")
    loss = cfg.make()

    prediction = torch.randn(3, 10)
    batch = {"target": torch.randn(3, 10)}

    result = loss(prediction, **batch)

    assert "loss" in result
    assert result["loss"].shape == (3, 10)


def test_simple_loss_missing_target():
    """Test SimpleLoss raises KeyError when target missing."""
    cfg = SimpleLoss.Config(target_key="label")
    loss = cfg.make()

    prediction = torch.randn(2, 3)
    batch = {"other": torch.tensor([0, 1])}

    with pytest.raises(KeyError) as exc_info:
        loss(prediction, **batch)
    assert "label" in str(exc_info.value)
    assert "Available keys" in str(exc_info.value)


def test_simple_loss_mse():
    """Test SimpleLoss with MSE loss function."""
    cfg = SimpleLoss.Config(
        loss_fn=functional.mse_loss,
        target_key="target",
    )
    loss = cfg.make()

    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch = {"target": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}

    result = loss(prediction, **batch)

    assert "loss" in result
    # Perfect match should have zero loss
    assert torch.allclose(result["loss"], torch.zeros(2))


def test_simple_loss_reduction_mean():
    """Test SimpleLoss with reduction='mean' via kwargs."""
    cfg = SimpleLoss.Config(kwargs={"reduction": "mean"})
    loss = cfg.make()

    prediction = torch.randn(4, 5)
    batch = {"label": torch.randn(4, 5)}

    result = loss(prediction, **batch)

    assert "loss" in result
    assert result["loss"].ndim == 0  # Scalar


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
