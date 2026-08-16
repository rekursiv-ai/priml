"""Tests for TrainStep's model/optimizer/timer/EMA half.

The supervised ``train_step`` and gradient accumulation are covered by
``train_step_test.py``; this file covers what every recipe inherits whether or
not it uses that step -- the optimizer build, the budgets and progress, the
timers, EMA-aware evaluation, and what a checkpoint carries.
"""

from __future__ import annotations

from typing import Any, override

from configgle import Fig, PartialConfig
from torch import Tensor, nn

import pytest
import torch

from priml.math.schedules import linear, warmup
from priml.timer import CheckpointableStepTimer
from priml.train.ema import EMA, NoEMA
from priml.train.parallelism import NoParallel
from priml.train.train_step import TrainStep


class _Tiny(nn.Module):
    """Linear model with a deterministic forward for eval comparison."""

    class Config(Fig["_Tiny"], make_with_kwargs=True):
        dim: int = -1

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)

    @override
    def forward(self, x: Tensor, **_kwargs: Any) -> Tensor:
        """Apply the linear layer."""
        return self.fc(x)


def _make(ema_config: Any) -> TrainStep:
    return TrainStep.Config(
        model=_Tiny.Config(dim=4),
        optimizer=PartialConfig(torch.optim.SGD, lr=0.0),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
        ema=ema_config,
    ).make()


def test_call_eval_param_dict_uses_shadow_not_live_weights() -> None:
    """#338: param_dict EMA call_eval must evaluate averaged (shadow) weights.

    param_dict mode keeps ``shadow_model is None`` by design (FSDP-safe), so
    the old ``self.ema.shadow_model or self.model`` fell back to LIVE weights.
    Drive the live weights away from the shadow and confirm ``call_eval``
    matches the shadow-swapped forward, not the live forward.
    """
    torch.manual_seed(0)
    learnable = _make(EMA.Config(decay=0.5, shadow_kind="param_dict"))

    x = torch.randn(2, 4)

    # Seed the shadow at the current weights, then mutate live weights so the
    # shadow (average) diverges from live.
    learnable.ema(learnable.model)
    with torch.no_grad():
        for p in learnable.model.parameters():
            p.add_(5.0)
    learnable.ema(learnable.model)  # shadow = 0.5*orig + 0.5*(orig+5)

    # Reference: forward with the shadow swapped in.
    with torch.inference_mode(), learnable.ema.apply_to(learnable.model):
        shadow_out = learnable.model(x).clone()
    live_out = learnable.model(x).clone()

    eval_out = learnable.call_eval(x)

    # The shadow and live forwards must differ (precondition), and call_eval
    # must match the shadow.
    assert not torch.allclose(shadow_out, live_out), "shadow == live; test inert"
    torch.testing.assert_close(eval_out, shadow_out)


def test_call_eval_module_shadow_uses_shadow() -> None:
    """#338: module-kind EMA call_eval still evaluates the shadow module."""
    torch.manual_seed(0)
    learnable = _make(EMA.Config(decay=0.5, shadow_kind="module"))

    x = torch.randn(2, 4)
    learnable.ema(learnable.model)
    with torch.no_grad():
        for p in learnable.model.parameters():
            p.add_(5.0)
    learnable.ema(learnable.model)

    with torch.inference_mode(), learnable.ema.apply_to(learnable.model):
        shadow_out = learnable.model(x).clone()
    torch.testing.assert_close(learnable.call_eval(x), shadow_out)


def test_call_eval_no_ema_uses_live() -> None:
    """#338: NoEMA call_eval evaluates live weights (apply_to is a no-op)."""
    torch.manual_seed(0)
    learnable = _make(NoEMA.Config())

    x = torch.randn(2, 4)
    with torch.inference_mode():
        live_out = learnable.model(x).clone()
    torch.testing.assert_close(learnable.call_eval(x), live_out)


def _learnable_with(**config_kwargs: Any) -> TrainStep:
    return TrainStep.Config(
        model=_Tiny.Config(dim=4),
        optimizer=PartialConfig(torch.optim.SGD, lr=0.0),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
        ema=NoEMA.Config(),
        **config_kwargs,
    ).make()


def test_load_strict_false_tolerates_missing_keys() -> None:
    """strict=False loads a checkpoint missing model keys w/o error.

    Strict load (the default) raises on the absent key; the finetuning policy
    lets the missing parameter keep its fresh initialization.
    """
    full = _learnable_with()
    state = full.state_dict()
    del state["model"]["fc.weight"]  # simulate a checkpoint lacking this param

    strict = _learnable_with()
    with pytest.raises(RuntimeError, match="Missing key"):
        strict.load_state_dict(state)  # default policy is strict

    lenient = _learnable_with()
    lenient.load_state_dict(state, strict=False)  # must not raise


def test_parameter_remap_transforms_before_load() -> None:
    """Remap kwarg rewrites the saved model dict before it is loaded."""
    source = _learnable_with()
    with torch.no_grad():
        for p in source.model.parameters():
            p.fill_(3.0)
    state = source.state_dict()
    # Rename fc.weight -> renamed.weight in the checkpoint; remap puts it back.
    state["model"] = {"renamed.weight": state["model"]["fc.weight"]}

    def _remap(sd: Any) -> Any:
        return {"fc.weight": sd["renamed.weight"]}

    target = _learnable_with()
    target.load_state_dict(state, remap=_remap)
    assert float(target.model.fc.weight.detach()[0, 0]) == 3.0


def _adam_learnable() -> TrainStep:
    """A learnable whose optimizer (Adam) keeps per-parameter state."""
    return TrainStep.Config(
        model=_Tiny.Config(dim=4),
        optimizer=PartialConfig(torch.optim.Adam, lr=0.1),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
        ema=NoEMA.Config(),
    ).make()


def test_load_optimizer_false_skips_optimizer_restore() -> None:
    """Finetuning must not restore the old optimizer state.

    ``load_optimizer=False`` keeps the fresh optimizer, letting
    ``strict=False`` finetuning succeed without a mismatched optimizer.
    """
    source = _adam_learnable()
    source.model(torch.randn(2, 4)).sum().backward()
    source.optimizer.step()  # populates Adam state
    state = source.state_dict()
    del state["model"]["fc.weight"]  # architecture changed

    finetune = _adam_learnable()
    finetune.load_state_dict(state, strict=False, load_optimizer=False)
    assert finetune.optimizer.state_dict()["state"] == {}


def test_load_optimizer_true_restores_optimizer_by_default() -> None:
    """The default policy restores optimizer state (ordinary resume)."""
    source = _adam_learnable()
    source.model(torch.randn(2, 4)).sum().backward()
    source.optimizer.step()
    state = source.state_dict()

    target = _adam_learnable()
    target.load_state_dict(state)
    assert target.optimizer.state_dict()["state"], "optimizer state not restored"


def _scheduled(**config_kwargs: Any) -> TrainStep:
    """A learnable with a nonzero rate, so a schedule has something to scale."""
    return TrainStep.Config(
        model=_Tiny.Config(dim=4),
        optimizer=PartialConfig(torch.optim.SGD, lr=1.0),
        parallelism=NoParallel.Config(device="cpu"),
        compile=None,
        ema=NoEMA.Config(),
        **config_kwargs,
    ).make()


def test_progress_reads_the_step_budget() -> None:
    learnable = _scheduled(train_budget_steps=10)
    assert learnable.progress_learning_schedule == 0.0
    learnable.timer_step.global_count = 5
    assert learnable.progress_learning_schedule == pytest.approx(0.5)


def test_progress_reads_the_time_budget() -> None:
    """A run budgeted in TIME cannot know its step count in advance."""
    learnable = _scheduled(train_budget_sec=100.0)
    learnable.timer_step.global_sec = 25.0
    assert learnable.progress_learning_schedule == pytest.approx(0.25)


def test_progress_reads_the_epoch_budget_once_the_loader_is_bound() -> None:
    """Passes over the data are a third axis, and a bound one.

    The count lives on the loader -- only it knows when the data ran out -- so
    an epoch budget is reachable only through the timer it hands over.
    """
    learnable = _scheduled(train_budget_epochs=4)
    loader_timer = CheckpointableStepTimer()
    learnable.bind_epoch_timer(loader_timer)
    loader_timer.global_count = 3
    assert learnable.progress_learning_schedule == pytest.approx(0.75)


def test_an_unbound_epoch_budget_is_unreachable_rather_than_wrong() -> None:
    """A stream with no pass leaves the count at zero, not at a wrong number.

    Nothing ticks the private timer, so the budget simply never binds -- which
    is the right reading, and the reason binding is a call rather than a
    silent assignment someone can forget to make.
    """
    learnable = _scheduled(train_budget_epochs=4)
    assert learnable.progress_learning_schedule == 0.0


def test_binding_shares_the_object_rather_than_copying_the_count() -> None:
    """A copy would be right only where someone remembered to update it.

    The loop ticks the DATASET's timer, so a learnable holding a snapshot
    would anneal against a count frozen at bind time.
    """
    learnable = _scheduled(train_budget_epochs=4)
    loader_timer = CheckpointableStepTimer()
    learnable.bind_epoch_timer(loader_timer)
    assert learnable.timer_epoch is loader_timer


def test_progress_takes_whichever_budget_binds_first() -> None:
    """Declaring both must anneal to zero as the FIRST of them binds.

    The minimum would instead leave the rate high when the run stopped, which
    is the case annealing exists to prevent.
    """
    learnable = _scheduled(train_budget_steps=10, train_budget_sec=100.0)
    learnable.timer_step.global_count = 8
    learnable.timer_step.global_sec = 20.0
    assert learnable.progress_learning_schedule == pytest.approx(0.8)


def test_progress_is_clamped_at_one() -> None:
    """A budget is checked between steps, so the last one lands past it."""
    learnable = _scheduled(train_budget_steps=10)
    learnable.timer_step.global_count = 40
    assert learnable.progress_learning_schedule == 1.0


def test_an_unset_budget_contributes_nothing() -> None:
    """Infinite is the identity for progress, not an error.

    It is what lets one formula serve a run budgeted in steps, in seconds, or
    in both, with no branch anywhere.
    """
    learnable = _scheduled()
    learnable.timer_step.global_count = 10_000
    learnable.timer_step.global_sec = 10_000.0
    assert learnable.progress_learning_schedule == 0.0


@pytest.mark.parametrize("field", ["train_budget_steps", "train_budget_sec"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_a_budget_that_cannot_be_divided_by_is_rejected(
    field: str,
    value: float,
) -> None:
    """NaN is the one that bites: it fails no comparison, so it would pass.

    A NaN budget divides into a NaN progress, and every schedule then returns
    NaN -- the run trains at an undefined rate rather than failing.
    """
    with pytest.raises(ValueError, match=field):
        _scheduled(**{field: value})


def test_the_schedule_scales_the_rate_the_recipe_was_tuned_at() -> None:
    """Each group is scaled from its own ``initial_lr``, never its current one.

    Scaling the live rate would compound: a schedule at 0.5 applied twice
    would leave the run at a quarter of its rate rather than half.
    """
    learnable = _scheduled(
        train_budget_steps=10,
        learning_rate_scheduler=PartialConfig(linear),
    )
    learnable.timer_step.global_count = 5
    learnable.apply_learning_rate()
    assert learnable.optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    learnable.apply_learning_rate()
    assert learnable.optimizer.param_groups[0]["lr"] == pytest.approx(0.5)


def test_the_rate_is_scheduled_before_the_update_it_applies_to() -> None:
    """Scaled afterwards, the run's FIRST update lands at the unscheduled rate.

    With a schedule that is zero at the start, a correctly-ordered step moves
    no weights at all.
    """
    learnable = _scheduled(
        train_budget_steps=10,
        learning_rate_scheduler=PartialConfig(warmup, fraction=0.5),
    )
    before = learnable.model.fc.weight.detach().clone()
    learnable.model(torch.randn(2, 4)).sum().backward()
    learnable.step()
    torch.testing.assert_close(learnable.model.fc.weight.detach(), before)


def test_each_activity_is_counted_separately() -> None:
    """A forward, an evaluation, and an update are three different costs.

    One counter over all of them could not answer what a run actually spent
    training, which is the number the budget bounds.
    """
    learnable = _scheduled(train_budget_steps=10)
    learnable.model(torch.randn(2, 4)).sum().backward()
    learnable(torch.randn(2, 4))
    learnable.call_eval(torch.randn(2, 4))
    learnable.call_eval(torch.randn(2, 4))
    learnable.step()

    assert learnable.timer_forward.global_count == 1
    assert learnable.timer_eval.global_count == 2
    assert learnable.timer_step.global_count == 1


def test_the_budget_clock_charges_updates_only() -> None:
    """Compilation and evaluation are not training, so they are not charged.

    Excluded by construction rather than by subtraction: the clock runs INSIDE
    the update, so anything outside one was never counted -- which is what
    makes ``progress`` a measure of training rather than of wall time.
    """
    learnable = _scheduled(train_budget_sec=100.0)
    learnable.call_eval(torch.randn(2, 4))
    learnable(torch.randn(2, 4))
    assert learnable.timer_step.global_sec == 0.0
    assert learnable.progress_learning_schedule == 0.0

    learnable.model(torch.randn(2, 4)).sum().backward()
    learnable.step()
    assert learnable.timer_step.global_sec > 0.0
    assert learnable.progress_learning_schedule > 0.0


def test_a_call_that_raised_is_still_counted() -> None:
    """It happened, and hiding it would misreport the step being debugged."""
    learnable = _scheduled()
    with pytest.raises(RuntimeError):
        learnable(torch.randn(2, 9))  # wrong width for the linear
    assert learnable.timer_forward.global_count == 1


def test_a_resume_keeps_the_global_count_and_restarts_the_local_one() -> None:
    """The pair separates this job's work from the whole run's.

    A warmup exclusion or a first-step branch reads the local count; a
    schedule and a stop condition read the global one.
    """
    source = _scheduled()
    source.model(torch.randn(2, 4)).sum().backward()
    source.step()
    source.step()

    target = _scheduled()
    target.load_state_dict(source.state_dict())
    assert target.timer_step.global_count == 2
    assert target.timer_step.local_count == 0
    assert target.timer_step.global_sec == source.timer_step.global_sec
    assert target.timer_step.local_sec == 0.0


def test_a_checkpoint_without_a_timer_still_loads() -> None:
    """A timer the checkpoint does not name keeps its fresh zero.

    Refusing would make adding a timer invalidate every checkpoint written
    before it existed.
    """
    source = _scheduled()
    state = source.state_dict()
    del state["timer_forward"]

    target = _scheduled()
    target.load_state_dict(state)
    assert target.timer_forward.global_count == 0


def test_the_clock_survives_a_resume() -> None:
    """Progress reads it, so a restarted clock re-anneals the rate from the top.

    The run would then repeat decay it had already applied -- training its
    tail at a rate the recipe places in its opening.
    """
    source = _scheduled(train_budget_sec=100.0)
    source.timer_step.global_sec = 40.0
    source.timer_step.global_count = 3

    target = _scheduled(train_budget_sec=100.0)
    target.load_state_dict(source.state_dict())
    assert target.timer_step.global_sec == pytest.approx(40.0)
    assert target.progress_learning_schedule == pytest.approx(0.4)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
