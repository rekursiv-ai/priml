"""Tests for the gradient compressors, against values written out by hand."""

from __future__ import annotations

from configgle import PartialConfig

import pytest
import torch

from priml.baselines.mlsbench.opt.gradient_compression.compressors import (
    ErrorFeedback,
    identity,
    qsgd,
    scaled_sign,
    top_k,
)


def test_identity_returns_the_gradient_itself() -> None:
    grad = torch.arange(6.0).reshape(2, 3)
    assert identity(grad, name="w") is grad


def test_top_k_keeps_the_largest_magnitudes_and_at_least_one() -> None:
    grad = torch.tensor([[0.5, -3.0, 1.0], [2.0, -0.25, 0.0]])
    assert torch.equal(
        top_k(grad, name="w", ratio=0.34),
        torch.tensor([[0.0, -3.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    # 6 * 0.01 rounds down to zero, so the floor of one entry applies.
    assert torch.equal(
        top_k(grad, name="w", ratio=0.01),
        torch.tensor([[0.0, -3.0, 0.0], [0.0, 0.0, 0.0]]),
    )


def test_scaled_sign_counts_zero_as_positive() -> None:
    grad = torch.tensor([[1.0, -2.0], [0.0, 5.0], [-1.0, 1.0]])
    assert torch.equal(
        scaled_sign(grad, name="w"),
        torch.tensor([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]) * 10 / 6,
    )


def test_qsgd_rounds_up_exactly_when_the_draw_is_below_the_remainder() -> None:
    grad = torch.tensor([0.375, -0.5, 0.0])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        draws = torch.rand(3)
        torch.manual_seed(0)
        out = qsgd(grad, name="w", levels=4, clip_norm=1.0)
    norm = grad.norm()
    level = 4 / norm * grad.abs()
    expected = (level.floor() + (draws < level - level.floor()).float()) * grad.sign()
    assert torch.equal(out, norm / 4 * expected)


def test_qsgd_clips_a_large_gradient_to_the_norm() -> None:
    grad = torch.tensor([3.0, 4.0])
    out = qsgd(grad, name="w", levels=8, clip_norm=1.0)
    assert out.norm() <= 1.0
    assert torch.equal(out.sign(), grad.sign())


def test_qsgd_returns_a_zero_gradient_without_drawing() -> None:
    before = torch.get_rng_state()
    out = qsgd(torch.zeros(2, 3), name="w", levels=4, clip_norm=1.0)
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(out, torch.zeros(2, 3))


def test_error_feedback_carries_what_the_last_step_dropped() -> None:
    config = ErrorFeedback.Config()
    config.compressor = PartialConfig(top_k, ratio=0.5)
    feedback = config.make()
    first = feedback(torch.tensor([3.0, 1.0]), name="w")
    assert torch.equal(first, torch.tensor([3.0, 0.0]))
    assert torch.equal(feedback.residuals["w"], torch.tensor([0.0, 1.0]))
    second = feedback(torch.tensor([0.5, 1.0]), name="w")
    assert torch.equal(second, torch.tensor([0.0, 2.0]))
    assert torch.equal(feedback.residuals["w"], torch.tensor([0.5, 0.0]))
    assert torch.equal(
        feedback(torch.tensor([1.0, 0.0]), name="b"),
        torch.tensor([1.0, 0.0]),
    )


def test_error_feedback_weights_the_residual_by_beta() -> None:
    config = ErrorFeedback.Config()
    config.compressor = PartialConfig(top_k, ratio=0.5)
    config.beta = 0.5
    feedback = config.make()
    _ = feedback(torch.tensor([3.0, 2.0]), name="w")
    assert torch.equal(
        feedback(torch.tensor([0.0, 0.0]), name="w"),
        torch.tensor([0.0, 1.0]),
    )


def test_error_feedback_state_round_trips() -> None:
    feedback = ErrorFeedback.Config().make()
    feedback.residuals["w"] = torch.tensor([1.0, 2.0])
    restored = ErrorFeedback.Config().make()
    restored.load_state_dict(feedback.state_dict())
    assert torch.equal(restored.residuals["w"], torch.tensor([1.0, 2.0]))
    assert restored.residuals is not feedback.residuals


@pytest.mark.parametrize("shape", [(2, 3), (3,), (2, 3, 4)])
def test_every_compressor_keeps_the_shape(shape: tuple[int, ...]) -> None:
    grad = torch.arange(1.0, 1.0 + torch.Size(shape).numel()).reshape(shape)
    for out in (
        top_k(grad, name="w", ratio=0.5),
        scaled_sign(grad, name="w"),
        qsgd(grad, name="w", levels=4, clip_norm=1.0),
    ):
        assert out.shape == shape


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
