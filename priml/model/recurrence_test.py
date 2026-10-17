"""Tests for reusable latent recurrence schedules."""

from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.model.recurrence import DeepRecurrence, SingleLatentRecurrence


def test_deep_recurrence_reports_every_stack_pass_as_gradient_bearing() -> None:
    config = DeepRecurrence.Config(fast_cycles=3)

    assert config.forward_passes == 4
    assert config.grad_passes == 4
    assert config.forward_additions == 5
    assert config.grad_additions == 5


def test_single_latent_recurrence_reports_real_pass_counts() -> None:
    config = SingleLatentRecurrence.Config(fast_cycles=4, inner_grad_loops=2)

    assert config.forward_passes == 4
    assert config.grad_passes == 2
    assert config.outer_grad_cycles == 1


def test_single_latent_recurrence_handles_more_grad_loops_than_fast_cycles() -> None:
    config = SingleLatentRecurrence.Config(fast_cycles=2, inner_grad_loops=3)

    assert config.forward_passes == 3
    assert config.grad_passes == 3


def test_single_latent_without_truncation_reports_all_inner_passes() -> None:
    config = SingleLatentRecurrence.Config(fast_cycles=4, inner_grad_loops=0)

    assert config.forward_passes == 4
    assert config.grad_passes == 4
    assert config.forward_additions == 4
    assert config.grad_additions == 4


def test_single_latent_rejects_negative_inner_gradient_loops() -> None:
    with pytest.raises(ValueError, match="inner_grad_loops must be >= 0"):
        SingleLatentRecurrence.Config(fast_cycles=4, inner_grad_loops=-1).make()


@pytest.mark.parametrize(
    ("fast_cycles", "inner_grad_loops", "expected_grad_flags"),
    [
        (4, 2, [False, False, True, True]),
        (4, 0, [True, True, True, True]),
        (2, 3, [True, True, True]),
    ],
)
def test_single_latent_refine_tracks_actual_gradient_passes(
    fast_cycles: int,
    inner_grad_loops: int,
    expected_grad_flags: list[bool],
) -> None:
    recurrence = SingleLatentRecurrence.Config(
        fast_cycles=fast_cycles,
        inner_grad_loops=inner_grad_loops,
    ).make()
    state = torch.zeros(2, 3, 4, requires_grad=True)
    embedding = torch.ones_like(state)
    weight = torch.nn.Parameter(torch.tensor(2.0))
    grad_flags: list[bool] = []

    def mix(z: Tensor, cos_sin: tuple[Tensor, Tensor] | None) -> Tensor:
        del cos_sin
        grad_flags.append(torch.is_grad_enabled())
        return z * weight

    slow, _ = recurrence.refine(mix, embedding, state, state, None)
    slow.sum().backward()

    assert grad_flags == expected_grad_flags
    assert weight.grad is not None


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
