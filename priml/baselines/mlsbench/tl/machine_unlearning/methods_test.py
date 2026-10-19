"""Tests for the unlearning methods.

Each test pins one behavior a method's reference states in its arithmetic: what
it reads from the forget batch, how its teachers are made, and when its phases
run. ``reference_test.py`` checks every method's numbers against the scripts.
"""

from __future__ import annotations

from typing import Final

import pytest
import torch

from priml.baselines.mlsbench.tl.machine_unlearning.methods import (
    BadTeacher,
    NegativeGradient,
    RetainFinetune,
    Scrub,
    reinitialize,
)


CLASSES: Final = 3

FEATURES: Final = 4


def tiny() -> tuple[torch.nn.Linear, torch.optim.Optimizer]:
    """Return a seeded linear classifier and its Adam optimizer."""
    torch.manual_seed(0)
    model = torch.nn.Linear(FEATURES, CLASSES)
    return model, torch.optim.Adam(model.parameters(), lr=0.001)


def batches() -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Return a retained batch of two and a forget batch of five."""
    generator = torch.Generator().manual_seed(1)
    retain = (torch.randn(2, FEATURES, generator=generator), torch.tensor([1, 2]))
    forget = (torch.randn(5, FEATURES, generator=generator), torch.zeros(5).long())
    return retain, forget


def test_retain_finetune_ignores_the_forget_batch() -> None:
    retain, forget = batches()
    losses: list[torch.Tensor] = []
    for other in (forget, (forget[0] * 3, forget[1] + 1)):
        model, optimizer = tiny()
        method = RetainFinetune.Config().make()
        losses.append(
            method.unlearn_step(model, retain, other, optimizer, step=0, epoch=0),
        )
    assert torch.equal(losses[0], losses[1])
    model, _ = tiny()
    expected = torch.nn.functional.cross_entropy(model.forward(retain[0]), retain[1])
    assert torch.equal(losses[0], expected.detach())


def test_negative_gradient_subtracts_the_weighted_forget_loss() -> None:
    retain, forget = batches()
    model, optimizer = tiny()
    with torch.no_grad():
        ce = torch.nn.functional.cross_entropy
        expected = ce(model.forward(retain[0]), retain[1]) - 0.5 * ce(
            model.forward(forget[0]),
            forget[1],
        )
    loss = (
        NegativeGradient.Config()
        .make()
        .unlearn_step(
            model,
            retain,
            forget,
            optimizer,
            step=0,
            epoch=0,
        )
    )
    assert torch.equal(loss, expected)


def test_reinitialize_resets_norms_and_redraws_every_weight() -> None:
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3),
        torch.nn.BatchNorm2d(4),
        torch.nn.Linear(4, CLASSES),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(7.0)
    reinitialize(model)
    conv, norm, linear = model
    assert isinstance(conv, torch.nn.Conv2d)
    assert isinstance(norm, torch.nn.BatchNorm2d)
    assert isinstance(linear, torch.nn.Linear)
    assert (conv.weight != 7).all()
    # The reference leaves a convolution's bias as it was.
    assert (torch.as_tensor(conv.bias) == 7).all()
    assert torch.equal(norm.weight, torch.ones(4))
    assert torch.equal(norm.bias, torch.zeros(4))
    assert (linear.weight != 7).all()
    assert torch.equal(torch.as_tensor(linear.bias), torch.zeros(CLASSES))


def test_bad_teacher_freezes_a_copy_and_a_redrawn_copy_on_its_first_step() -> None:
    retain, forget = batches()
    model, optimizer = tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    method = BadTeacher.Config().make()
    _ = method.unlearn_step(model, retain, forget, optimizer, step=0, epoch=0)
    competent, incompetent = method.competent, method.incompetent
    assert competent is not None
    assert incompetent is not None
    for key, value in competent.state_dict().items():
        assert torch.equal(value, before[key])
    assert not torch.equal(incompetent.state_dict()["weight"], before["weight"])
    for teacher in (competent, incompetent):
        assert not teacher.training
        assert not any(p.requires_grad for p in teacher.parameters())
    _ = method.unlearn_step(model, retain, forget, optimizer, step=1, epoch=0)
    assert method.competent is competent


def zeroed(model: torch.nn.Module) -> None:
    """Zero every parameter, so the network predicts the uniform distribution."""
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()


def test_bad_teacher_distills_retained_from_the_original_forget_from_the_random() -> (
    None
):
    retain, forget = batches()
    model, optimizer = tiny()
    with torch.no_grad():
        student = model.forward(torch.cat([retain[0], forget[0]]))
    is_forget = torch.tensor([0.0] * 2 + [1.0] * 5).view(-1, 1)
    uniform = torch.softmax(torch.zeros(7, CLASSES), dim=1)
    target = is_forget * uniform + (1.0 - is_forget) * torch.softmax(student, dim=1)
    expected = torch.nn.functional.kl_div(
        torch.log_softmax(student, dim=1),
        target,
        reduction="batchmean",
    )
    config = BadTeacher.Config()
    config.reinit = zeroed
    loss = config.make().unlearn_step(model, retain, forget, optimizer, step=0, epoch=0)
    assert torch.equal(loss, expected)
    assert loss > 0.01


class StepCounter:
    """An optimizer post-step hook that counts the updates."""

    def __init__(self) -> None:
        self.steps = 0

    def __call__(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        """Count one update."""
        del optimizer, args, kwargs
        self.steps += 1


@pytest.mark.parametrize(("epoch", "updates"), [(0, 2), (1, 2), (2, 1)])
def test_scrub_runs_its_max_step_only_in_the_first_two_epochs(
    epoch: int,
    updates: int,
) -> None:
    retain, forget = batches()
    model, optimizer = tiny()
    method = Scrub.Config().make()
    counter = StepCounter()
    _ = optimizer.register_step_post_hook(counter)
    _ = method.unlearn_step(model, retain, forget, optimizer, step=0, epoch=epoch)
    assert counter.steps == updates


def test_scrub_min_loss_is_gamma_times_cross_entropy_without_the_kl() -> None:
    retain, forget = batches()
    model, optimizer = tiny()
    with torch.no_grad():
        ce = torch.nn.functional.cross_entropy(model.forward(retain[0]), retain[1])
    config = Scrub.Config()
    config.alpha = 0.0
    loss = config.make().unlearn_step(
        model,
        retain,
        forget,
        optimizer,
        step=0,
        epoch=2,
    )
    assert torch.equal(loss, 0.99 * ce)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
