"""Tests for reusable latent recurrence schedules."""

from __future__ import annotations

from torch import Tensor

import pytest
import torch

from priml.cost import Cost
from priml.model.custom_types import CoreOutput
from priml.model.recurrence import (
    DeepRecurrence,
    SingleLatentRecurrence,
    two_latent_refine,
)


def _add_one(z: Tensor, cos_sin: tuple[Tensor, Tensor] | None) -> Tensor:
    del cos_sin
    return z + 1


def test_deep_recurrence_costs_nothing() -> None:
    assert DeepRecurrence.Config().cost(seq_len=3, batch_size=2) == Cost()


@pytest.mark.parametrize(("slow_cycles", "fast_cycles"), [(0, 2), (2, 0)])
def test_deep_recurrence_rejects_a_cycle_count_below_one(
    slow_cycles: int,
    fast_cycles: int,
) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        DeepRecurrence.Config(slow_cycles=slow_cycles, fast_cycles=fast_cycles).make()


def test_two_latent_refine_updates_fast_cycles_times_then_the_slow_latent() -> None:
    z_slow = torch.zeros(2, 3, 4)
    z_fast = torch.zeros(2, 3, 4)
    input_emb = torch.full((2, 3, 4), 10.0)

    new_slow, new_fast = two_latent_refine(
        _add_one,
        input_emb,
        z_slow,
        z_fast,
        None,
        fast_cycles=3,
    )

    # Each fast pass reads z_fast + z_slow + input_emb, then adds one.
    assert torch.equal(new_fast, torch.full((2, 3, 4), 33.0))
    assert torch.equal(new_slow, torch.full((2, 3, 4), 34.0))


def test_deep_recurrence_refine_is_the_two_latent_update() -> None:
    recurrence = DeepRecurrence.Config(fast_cycles=2).make()
    z_slow = torch.randn(2, 3, 4)
    z_fast = torch.randn(2, 3, 4)
    input_emb = torch.randn(2, 3, 4)

    actual = recurrence.refine(_add_one, input_emb, z_slow, z_fast, None)
    expected = two_latent_refine(
        _add_one,
        input_emb,
        z_slow,
        z_fast,
        None,
        fast_cycles=2,
    )

    for actual_latent, expected_latent in zip(actual, expected, strict=True):
        assert torch.equal(actual_latent, expected_latent)


class _Core:
    """A core that logs whether it ran with gradient and on detached inputs."""

    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.grad_flags: list[bool] = []
        self.cos_sins: list[tuple[Tensor, Tensor] | None] = []
        self.inputs_require_grad: list[tuple[bool, bool, bool]] = []

    def __call__(
        self,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> CoreOutput:
        self.grad_flags.append(torch.is_grad_enabled())
        self.cos_sins.append(cos_sin)
        self.inputs_require_grad.append(
            (input_emb.requires_grad, z_slow.requires_grad, z_fast.requires_grad),
        )
        z_slow = z_slow + input_emb * self.weight
        z_fast = z_fast + 1
        return CoreOutput(z_slow * 2, z_slow.mean(-1).mean(-1), z_slow, z_fast)


@pytest.mark.parametrize("collect", [False, True])
def test_deep_recurrence_forward_carries_gradient_through_the_last_cycle_only(
    collect: bool,
) -> None:
    recurrence = DeepRecurrence.Config(slow_cycles=3).make()
    core = _Core()
    input_emb = torch.ones(2, 3, 4, requires_grad=True)
    z_slow = torch.zeros(2, 3, 4, requires_grad=True)
    z_fast = torch.zeros(2, 3, 4, requires_grad=True)
    cos = torch.ones(3, 2, requires_grad=True)
    sin = torch.zeros(3, 2, requires_grad=True)

    out = recurrence(
        core,
        input_emb,
        z_slow,
        z_fast,
        (cos, sin),
        collect_intermediates=collect,
    )

    assert core.grad_flags == [False, False, True]
    # The no-grad cycles see detached inputs; the final one sees the originals.
    assert core.inputs_require_grad[:2] == [(False, False, False)] * 2
    assert core.inputs_require_grad[2] == (True, False, False)
    first, second, last = core.cos_sins
    assert first is not None
    assert second is not None
    assert not first[0].requires_grad
    assert not second[1].requires_grad
    assert last == (cos, sin)
    # Two no-grad cycles each add 2 * 1; the last adds another 2 * 1.
    assert torch.equal(out.z_slow, torch.full((2, 3, 4), 6.0))
    assert torch.equal(out.z_fast, torch.full((2, 3, 4), 3.0))
    assert torch.equal(out.logits, out.z_slow * 2)
    assert out.halt.shape == (2,)
    out.logits.sum().backward()
    assert core.weight.grad is not None
    if collect:
        assert len(out.all_logits) == 3
        assert torch.equal(out.all_logits[0], torch.full((2, 3, 4), 4.0))
        assert torch.equal(out.all_logits[1], torch.full((2, 3, 4), 8.0))
        assert torch.equal(out.all_logits[2], out.logits)
        assert not out.all_logits[2].requires_grad
    else:
        assert out.all_logits == ()


def test_deep_recurrence_forward_without_rotary_passes_none_to_every_cycle() -> None:
    recurrence = DeepRecurrence.Config(slow_cycles=2).make()
    core = _Core()

    recurrence(
        core,
        torch.ones(2, 3, 4),
        torch.zeros(2, 3, 4),
        torch.zeros(2, 3, 4),
        None,
        collect_intermediates=False,
    )

    assert core.cos_sins == [None, None]


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
