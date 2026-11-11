"""Tests for WeightProgramming's binding, numerical edits, and native state."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, cast, override

import pickle

from configgle import Makes
from torch import Tensor, nn

import pytest
import torch

from priml.optimizers.weight_programming import (
    UpdateOutput,
    WeightProgramming,
    _no_batch,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class _GradientProgram(WeightProgramming):
    """Stateless gradient-assisted programmer for testing."""

    class Config(Makes["Callable[..., _GradientProgram]"], WeightProgramming.Config):
        """A stateless gradient-assisted programmer."""

    @override
    def program_weights(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> UpdateOutput:
        del next_batch
        for group in self.param_groups:
            for param in cast("list[nn.Parameter]", group["params"]):
                gradient = torch.autograd.grad(
                    param.square().sum(),
                    param,
                )[0]
                with torch.no_grad():
                    param.sub_(gradient)
        return UpdateOutput(updated=True)


class _CalibratedProgram(WeightProgramming):
    """Input-dependent programmer: different inputs yield different edits."""

    class Config(
        Makes["Callable[..., _CalibratedProgram]"],
        WeightProgramming.Config,
    ):
        """A calibrated programmer."""

    @override
    def program_weights(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> UpdateOutput:
        batch = next_batch()
        media = batch["media"]
        assert isinstance(media, Tensor)
        for group in self.param_groups:
            for param in cast("list[nn.Parameter]", group["params"]):
                output = (param * media).sum()
                gradient = torch.autograd.grad(output, param)[0]
                with torch.no_grad():
                    param.sub_(0.1 * gradient)
        return UpdateOutput(
            updated=True,
            metrics={"calibration_input_mean": float(media.mean())},
        )


class _ScaledProgram(WeightProgramming):
    """Programmer with a config-derived scale and mutable step count."""

    class Config(
        Makes["Callable[..., _ScaledProgram]"],
        WeightProgramming.Config,
    ):
        """Scaled programmer."""

        scale: float = 0.1
        """Per-step scaling factor."""

    @override
    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        config: WeightProgramming.Config,
    ) -> None:
        assert isinstance(config, _ScaledProgram.Config)
        super().__init__(params, config=config)
        self.defaults["scale"] = config.scale
        for group in self.param_groups:
            group.setdefault("scale", config.scale)
            group.setdefault("step_count", 0)

    @override
    def program_weights(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> UpdateOutput:
        del model, next_batch
        for group in self.param_groups:
            step_count = cast(int, group["step_count"]) + 1
            group["step_count"] = step_count
            scale = cast(float, group["scale"])
            for param in cast("list[nn.Parameter]", group["params"]):
                with torch.no_grad():
                    param.add_(scale * step_count)
        return UpdateOutput(updated=True)


class _AccumulatingProgram(WeightProgramming):
    """Stores running count in Optimizer.state for continuation testing."""

    class Config(
        Makes["Callable[..., _AccumulatingProgram]"],
        WeightProgramming.Config,
    ):
        """Accumulating programmer."""

    @override
    def program_weights(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> UpdateOutput:
        del model, next_batch
        for group in self.param_groups:
            for param in cast("list[nn.Parameter]", group["params"]):
                state = cast("dict[str, Tensor]", self.state[param])
                if "count" not in state:
                    state["count"] = torch.tensor(0)
                state["count"] = state["count"] + 1
                with torch.no_grad():
                    param.add_(state["count"] * 0.01)
        return UpdateOutput(updated=True)


def test_no_batch_raises_outside_training() -> None:
    """The default next_batch sentinel raises outside a training update."""
    with pytest.raises(RuntimeError, match=r"^Calibration input is unavailable"):
        _no_batch()


def test_config_make_returns_constructor() -> None:
    """Config.make() returns a callable; calling it with params builds a programmer."""
    config = _GradientProgram.Config()
    constructor = config.make()
    params = [nn.Parameter(torch.randn(2, 3))]
    programmer = constructor(params)
    assert programmer.param_groups[0]["params"] == params
    assert programmer.param_groups[0]["lr"] == 0.0
    assert programmer.output == UpdateOutput()


def test_step_without_bind_raises() -> None:
    """step() outside bind context raises RuntimeError."""
    config = _GradientProgram.Config()
    programmer = config.make()([nn.Parameter(torch.randn(2, 3))])
    with pytest.raises(RuntimeError, match="requires a bound model"):
        programmer.step()


def test_bind_enables_step() -> None:
    """bind() context enables step() to delegate to program_weights."""
    model = nn.Linear(2, 3)
    config = _GradientProgram.Config()
    programmer = config.make()(list(model.parameters()))
    before = model.weight.detach().clone()
    with programmer.bind(model, next_batch=_no_batch):
        programmer.step()
    assert programmer.output.updated
    assert not torch.equal(model.weight.data, before)


def test_step_closure_returns_value() -> None:
    """step(closure) returns the closure's return value."""
    model = nn.Linear(2, 3)
    config = _GradientProgram.Config()
    programmer = config.make()(list(model.parameters()))
    with programmer.bind(model, next_batch=_no_batch):
        result = programmer.step(lambda: 42.0)
    assert result == 42.0


def test_step_closure_can_differentiate_owned_weights() -> None:
    model = nn.Linear(3, 4, bias=False)
    programmer = _ScaledProgram.Config().make()(model.parameters())
    media = torch.ones(2, 3)
    expected = model(media).square().sum().detach()

    def closure() -> Tensor:
        loss = model(media).square().sum()
        loss.backward()
        return loss

    with programmer.bind(model, next_batch=_no_batch), torch.no_grad():
        result = programmer.step(closure)
    assert isinstance(result, Tensor)
    assert torch.equal(result, expected)
    assert model.weight.grad is not None
    assert not model.weight.requires_grad


def test_failed_closure_restores_owned_eligibility_and_binding() -> None:
    model = nn.Linear(3, 4, bias=False)
    programmer = _ScaledProgram.Config().make()(model.parameters())

    def closure() -> Tensor:
        model(torch.ones(2, 3)).sum().backward()
        raise RuntimeError("closure failed")

    with (
        pytest.raises(RuntimeError, match="closure failed"),
        programmer.bind(model, next_batch=_no_batch),
    ):
        programmer.step(closure)
    assert not model.weight.requires_grad
    assert not programmer.output.updated
    with pytest.raises(RuntimeError, match="requires a bound model"):
        programmer.step()


def test_step_without_closure_returns_none() -> None:
    """step() without a closure returns None."""
    model = nn.Linear(2, 3)
    config = _GradientProgram.Config()
    programmer = config.make()(list(model.parameters()))
    with programmer.bind(model, next_batch=_no_batch):
        result = programmer.step()
    assert result is None


@pytest.mark.parametrize("media", [[2.0, -3.0, 4.0], [-1.0, 5.0, 2.0]])
def test_calibrated_step_uses_next_batch(media: list[float]) -> None:
    """The supplied input determines the exact numerical edit."""
    model = nn.Linear(3, 4, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(12).reshape(4, 3))
    before = model.weight.detach().clone()
    values = torch.tensor(media)
    batch: dict[str, object] = {"media": values}
    config = _CalibratedProgram.Config()
    programmer = config.make()(list(model.parameters()))
    with programmer.bind(model, next_batch=lambda: batch):
        assert torch.equal(model.weight, before)
        assert programmer.output == UpdateOutput()
        programmer.step()
    assert torch.equal(model.weight, before - 0.1 * values)
    assert programmer.output == UpdateOutput(
        updated=True,
        metrics={"calibration_input_mean": float(values.mean())},
    )


def test_bind_restores_on_normal_exit() -> None:
    """After bind context exits normally, step() raises again."""
    model = nn.Linear(2, 3)
    config = _GradientProgram.Config()
    programmer = config.make()(list(model.parameters()))
    with programmer.bind(model, next_batch=_no_batch):
        programmer.step()
    with pytest.raises(RuntimeError, match="requires a bound model"):
        programmer.step()


def test_bind_restores_on_failure() -> None:
    """After an exception inside bind, step() raises as if never bound."""

    class _Failing(WeightProgramming):
        class Config(Makes["Callable[..., _Failing]"], WeightProgramming.Config):
            """Failing programmer."""

        @override
        def program_weights(
            self,
            model: nn.Module,
            *,
            next_batch: Callable[[], dict[str, object]],
        ) -> UpdateOutput:
            raise RuntimeError("Calibration failed")

    model = nn.Linear(2, 3)
    config = _Failing.Config()
    programmer = config.make()(list(model.parameters()))
    with (
        pytest.raises(RuntimeError, match="Calibration failed"),
        programmer.bind(
            model,
            next_batch=_no_batch,
        ),
    ):
        programmer.step()
    with pytest.raises(RuntimeError, match="requires a bound model"):
        programmer.step()


def test_bind_is_not_reentrant() -> None:
    """Calling bind() while already bound raises."""
    model = nn.Linear(2, 3)
    config = _GradientProgram.Config()
    programmer = config.make()(list(model.parameters()))
    with (
        programmer.bind(model, next_batch=_no_batch),
        pytest.raises(
            RuntimeError,
            match="already bound",
        ),
        programmer.bind(model, next_batch=_no_batch),
    ):
        pass


def test_stateless_state_dict_round_trip() -> None:
    """state_dict includes type identity and delegates to torch."""
    programmer = _GradientProgram.Config().make()([nn.Parameter(torch.randn(2, 3))])
    state = programmer.state_dict()
    expected_type = f"{_GradientProgram.__module__}.{_GradientProgram.__qualname__}"
    assert state["programmer"] == expected_type
    assert "param_groups" in state
    programmer.load_state_dict(state)


def test_state_dict_rejects_foreign_type() -> None:
    """load_state_dict rejects state from a different programmer type."""
    programmer = _GradientProgram.Config().make()([nn.Parameter(torch.randn(2, 3))])
    state = programmer.state_dict()
    state["programmer"] = "some.other.Programmer"
    with pytest.raises(ValueError, match="State was saved by"):
        programmer.load_state_dict(state)


def test_state_dict_rejects_missing_programmer_tag() -> None:
    """Plain optimizer state (no programmer tag) is rejected."""
    programmer = _GradientProgram.Config().make()([nn.Parameter(torch.randn(2, 3))])
    sgd = torch.optim.SGD([nn.Parameter(torch.randn(2, 3))], lr=0.1)
    with pytest.raises(ValueError, match="no programmer tag"):
        programmer.load_state_dict(sgd.state_dict())


def test_state_dict_rejects_cross_type() -> None:
    """State from one programmer type cannot load into another."""
    gradient = _GradientProgram.Config().make()([nn.Parameter(torch.randn(2, 3))])
    calibrated = _CalibratedProgram.Config().make()([nn.Parameter(torch.randn(2, 3))])
    with pytest.raises(ValueError, match="State was saved by"):
        calibrated.load_state_dict(gradient.state_dict())


def test_stateful_state_dict_round_trip() -> None:
    """Per-parameter state survives state_dict/load_state_dict."""
    param = nn.Parameter(torch.ones(2, 3))
    programmer = _AccumulatingProgram.Config().make()([param])
    model = nn.Linear(2, 3)

    with programmer.bind(model, next_batch=_no_batch):
        programmer.step()
        programmer.step()

    saved = programmer.state_dict()

    param2 = nn.Parameter(torch.ones(2, 3))
    programmer2 = _AccumulatingProgram.Config().make()([param2])
    programmer2.load_state_dict(saved)

    count = cast(Tensor, programmer2.state[param2]["count"])
    assert count.item() == 2

    before = param2.detach().clone()
    model2 = nn.Linear(2, 3)
    with programmer2.bind(model2, next_batch=_no_batch):
        programmer2.step()
    # Resumed count=3: delta = 3 * 0.01 = 0.03.
    assert torch.equal(param2.data, before + 0.03)


def test_truncated_checkpoint_eligibility_rejects_before_programming() -> None:
    """Truncated saved_requires_grad rejects before numerical edits."""
    model = nn.Linear(3, 4)
    assert model.bias is not None
    programmer = _GradientProgram.Config().make()(list(model.parameters()))

    state = deepcopy(programmer.state_dict())
    groups = cast("list[dict[str, object]]", state["param_groups"])
    groups[0]["saved_requires_grad"] = [True]
    programmer.load_state_dict(state)

    before_weight = model.weight.detach().clone()
    before_bias = model.bias.detach().clone()

    with (
        programmer.bind(model, next_batch=_no_batch),
        pytest.raises(ValueError, match="zip"),
    ):
        programmer.step()

    assert not model.weight.requires_grad
    assert not model.bias.requires_grad
    assert torch.equal(model.weight.data, before_weight)
    assert torch.equal(model.bias.data, before_bias)


def test_on_epoch_end_default() -> None:
    """Default on_epoch_end is callable and returns None."""
    config = _GradientProgram.Config()
    programmer = config.make()([nn.Parameter(torch.randn(2, 3))])
    assert programmer.on_epoch_end() is None


def test_init_freezes_owned_params() -> None:
    """WeightProgramming.__init__ freezes all owned parameters."""
    param = nn.Parameter(torch.randn(2, 3))
    assert param.requires_grad
    _ = _GradientProgram.Config().make()([param])
    assert not param.requires_grad


def test_step_refreezes_after_program_weights() -> None:
    """After step(), requires_grad is back to False."""
    model = nn.Linear(2, 3)
    programmer = _GradientProgram.Config().make()(list(model.parameters()))
    assert not model.weight.requires_grad
    with programmer.bind(model, next_batch=_no_batch):
        programmer.step()
    assert not model.weight.requires_grad


def test_step_refreezes_on_failure() -> None:
    """After a failed program_weights, requires_grad is back to False."""

    class _Boom(WeightProgramming):
        class Config(Makes["Callable[..., _Boom]"], WeightProgramming.Config):
            """Boom."""

        @override
        def program_weights(
            self,
            model: nn.Module,
            *,
            next_batch: Callable[[], dict[str, object]],
        ) -> UpdateOutput:
            raise RuntimeError("boom")

    model = nn.Linear(2, 3)
    programmer = _Boom.Config().make()(list(model.parameters()))
    assert not model.weight.requires_grad
    with (
        pytest.raises(RuntimeError, match="boom"),
        programmer.bind(model, next_batch=_no_batch),
    ):
        programmer.step()
    assert not model.weight.requires_grad


def test_add_param_group_extends_ownership() -> None:
    """Adding a param group freezes and includes it in programming."""
    param_a = nn.Parameter(torch.randn(2, 3))
    programmer = _GradientProgram.Config().make()([param_a])
    assert not param_a.requires_grad

    param_b = nn.Parameter(torch.randn(4, 5))
    programmer.add_param_group({"params": [param_b]})
    assert not param_b.requires_grad
    assert len(programmer.param_groups) == 2

    before_a = param_a.detach().clone()
    before_b = param_b.detach().clone()
    model = nn.Linear(2, 3)
    with programmer.bind(model, next_batch=_no_batch):
        programmer.step()
    # Both groups updated: eligibility restored during program_weights.
    assert not torch.equal(param_a.data, before_a)
    assert not torch.equal(param_b.data, before_b)
    assert not param_b.requires_grad


def test_add_param_group_preserves_frozen_eligibility() -> None:
    """Originally frozen params stay frozen during program_weights."""
    observed_grad: list[bool] = []

    class _Observer(WeightProgramming):
        class Config(Makes["Callable[..., _Observer]"], WeightProgramming.Config):
            """Observer."""

        @override
        def program_weights(
            self,
            model: nn.Module,
            *,
            next_batch: Callable[[], dict[str, object]],
        ) -> UpdateOutput:
            del model, next_batch
            observed_grad.extend(
                p.requires_grad
                for group in self.param_groups
                for p in cast("list[nn.Parameter]", group["params"])
            )
            return UpdateOutput(updated=True)

    trainable = nn.Parameter(torch.randn(2, 3))
    programmer = _Observer.Config().make()([trainable])

    frozen = nn.Parameter(torch.randn(4), requires_grad=False)
    programmer.add_param_group({"params": [frozen]})
    assert not frozen.requires_grad

    model = nn.Linear(2, 3)
    with programmer.bind(model, next_batch=_no_batch):
        programmer.step()
    # Trainable restored True; frozen preserved False.
    assert observed_grad == [True, False]
    assert not trainable.requires_grad
    assert not frozen.requires_grad


def test_deepcopy_preserves_subclass_state() -> None:
    """Deepcopy preserves subclass state: exact numerical continuation."""
    config = _ScaledProgram.Config()
    config.scale = 0.5
    model = nn.Linear(2, 3)
    programmer = config.make()(list(model.parameters()))
    _ = torch.optim.lr_scheduler.StepLR(programmer, step_size=1)

    before = model.weight.detach().clone()
    with programmer.bind(model, next_batch=_no_batch):
        programmer.step()
    # Step 1: scale(0.5) * step_count(1) = 0.5.
    assert torch.equal(model.weight.data, before + 0.5)

    after_step1 = model.weight.detach().clone()
    model_copy, programmer_copy = deepcopy((model, programmer))

    with programmer_copy.bind(model_copy, next_batch=_no_batch):
        programmer_copy.step()
    # Step 2 on copy: scale(0.5) * step_count(2) = 1.0.
    assert torch.equal(model_copy.weight.data, after_step1 + 1.0)
    # Original unchanged after copy advances.
    assert torch.equal(model.weight.data, after_step1)


def test_pickle_preserves_subclass_state_without_native_hooks_or_binding() -> None:
    model = nn.Linear(2, 3, bias=False)
    config = _ScaledProgram.Config()
    config.scale = 0.5
    programmer = config.make()(model.parameters())
    _ = programmer.register_step_pre_hook(lambda _o, _a, _k: None)
    _ = programmer.register_step_post_hook(lambda _o, _a, _k: None)
    _ = programmer.register_state_dict_pre_hook(lambda _o: None)
    _ = programmer.register_state_dict_post_hook(lambda _o, _s: None)
    _ = programmer.register_load_state_dict_pre_hook(lambda _o, _s: None)
    _ = programmer.register_load_state_dict_post_hook(lambda _o: None)
    with programmer.bind(model, next_batch=dict):
        programmer.step()
        payload = pickle.dumps((model, programmer))

    model_copy, programmer_copy = cast(
        tuple[nn.Linear, _ScaledProgram],
        pickle.loads(payload),
    )
    with pytest.raises(RuntimeError, match="requires a bound model"):
        programmer_copy.step()
    assert programmer_copy.output == UpdateOutput()
    before = model_copy.weight.detach().clone()
    with programmer_copy.bind(model_copy, next_batch=_no_batch):
        programmer_copy.step()
    assert torch.equal(model_copy.weight, before + 1.0)
    assert torch.equal(model.weight, before)
    assert not model_copy.weight.requires_grad


def test_scheduler_attached_pickle_round_trips_subclass_state() -> None:
    """Pickle succeeds when a scheduler has wrapped the optimizer's step."""
    model = nn.Linear(2, 3, bias=False)
    config = _ScaledProgram.Config()
    config.scale = 0.5
    programmer = config.make()(model.parameters())
    _ = torch.optim.lr_scheduler.StepLR(programmer, step_size=1)
    with programmer.bind(model, next_batch=dict):
        programmer.step()
    payload = pickle.dumps((model, programmer))
    model_copy, programmer_copy = cast(
        tuple[nn.Linear, _ScaledProgram],
        pickle.loads(payload),
    )
    before = model_copy.weight.detach().clone()
    with programmer_copy.bind(model_copy, next_batch=_no_batch):
        programmer_copy.step()
    assert torch.equal(model_copy.weight, before + 1.0)


def test_deepcopy_preserves_autograd_eligibility() -> None:
    """Deepcopy preserves gradient eligibility: autograd works in the copy."""
    model = nn.Linear(2, 3)
    programmer = _GradientProgram.Config().make()(list(model.parameters()))
    assert not model.weight.requires_grad

    model_copy, programmer_copy = deepcopy((model, programmer))

    before = model_copy.weight.detach().clone()
    with programmer_copy.bind(model_copy, next_batch=_no_batch):
        programmer_copy.step()
    # Autograd succeeded: requires_grad was correctly restored in the copy.
    assert not torch.equal(model_copy.weight.data, before)
    assert not model_copy.weight.requires_grad
    # Original unchanged.
    assert not model.weight.requires_grad


def test_bind_resets_output() -> None:
    """Entering bind clears output from a prior binding."""
    model = nn.Linear(2, 3)
    programmer = _GradientProgram.Config().make()(list(model.parameters()))
    with programmer.bind(model, next_batch=_no_batch):
        programmer.step()
    assert programmer.output.updated

    with programmer.bind(model, next_batch=_no_batch):
        pass
    assert not programmer.output.updated


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
