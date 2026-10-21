"""Exercise actual shared update slots, EMA evaluation and weighted accumulation."""

from collections.abc import Callable
from contextlib import nullcontext
from typing import TYPE_CHECKING, cast

from torch import Tensor, nn

import numpy as np
import pytest
import torch

from priml.baselines.wellbore_geology.testing import (
    numerical_tensors,
    probe_batches,
    probe_recipe,
    tiny_batches,
    tiny_model_config,
)
from priml.baselines.wellbore_geology.train_step import WellboreTrainStep
from priml.optimizers.composite import CompositeOptimizer
from priml.testing.golden import mismatches
from priml.train.ema import NoEMA
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from priml.baselines.wellbore_geology.custom_types import Batch


def test_update_order_and_final_partial_accumulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = probe_recipe()
    step = config.step.make()
    assert isinstance(step, WellboreTrainStep)
    batches = probe_batches()
    initial = next(step.model.parameters()).detach().clone()
    events: list[str] = []
    optimizer_step: Callable[[Callable[[], Tensor | float] | None], bool] = (
        step.optimizer_step
    )
    apply_rate = step.apply_learning_rate
    zero = step.optimizer.zero_grad
    ema = step.ema

    def update(closure: Callable[[], Tensor | float] | None = None) -> bool:
        events.append("optimizer")
        return optimizer_step(closure)

    def rate(*, schedule_step: int | None = None) -> None:
        events.append(f"scheduler:{schedule_step}")
        apply_rate(schedule_step=schedule_step)

    def zero_grad(set_to_none: bool = False) -> None:
        events.append("zero")
        zero(set_to_none=set_to_none)

    def average(model: torch.nn.Module) -> None:
        events.append("ema")
        ema(model)

    monkeypatch.setattr(step, "optimizer_step", update)
    monkeypatch.setattr(step, "apply_learning_rate", rate)
    monkeypatch.setattr(step.optimizer, "zero_grad", zero_grad)
    monkeypatch.setattr(step, "ema", average)
    outputs = [step.train_step(**step.preprocess_batch(batch)) for batch in batches]
    assert [tuple(result["model"].shape) for result in outputs] == [
        (3, 4),
        (3, 4),
        (2, 4),
    ]
    assert [tuple(result["loss"].shape) for result in outputs] == [(3,), (3,), (2,)]
    update_events = ["scheduler:None", "optimizer", "zero", "ema"]
    assert events == update_events
    assert step.global_step == 1
    assert step.accumulation_steps == 1
    assert step.accumulated_samples == 2
    assert not torch.equal(initial, next(step.model.parameters()))
    step.on_epoch_end()
    assert events == update_events * 2
    assert step.global_step == 2
    assert step.accumulation_steps == step.accumulated_samples == 0
    assert all(parameter.grad is None for parameter in step.model.parameters())
    step.on_epoch_end()
    assert events == update_events * 2


def test_no_ema_cpu_precision_and_zero_clip_are_real_slots() -> None:
    config = probe_recipe().step
    config.ema = NoEMA.Config()
    config.gradient_clip_norm = 0
    config.dtype_autocast = torch.bfloat16
    config.accumulate_grad_batches = 1
    step = config.make()
    batches = probe_batches()
    result = step.train_step(**step.preprocess_batch(batches[0]))
    assert result["model"].dtype is torch.float32
    assert step.global_step == 1
    assert isinstance(step.optimizer, CompositeOptimizer)
    for state in cast("dict[object, dict[str, object]]", step.optimizer.state).values():
        average = state["exp_avg"]
        assert isinstance(average, Tensor)
        assert torch.count_nonzero(average).item() == 0
    with pytest.raises(NotImplementedError, match="native worker streams"):
        step.load_state_dict({})


def test_complete_prediction_and_eval_loss_use_actual_slots() -> None:
    config = probe_recipe().step
    config.model = tiny_model_config(dropout=0.05)
    step = config.make()
    batches = tiny_batches()
    prediction, metadata = step.predict(batches, model=step.model, move_cpu=True)
    batch = cast("Batch", batches[0])
    with torch.no_grad():
        expected, _ = step.model(batch["media"])
    assert np.array_equal(prediction, expected.numpy())
    assert prediction.dtype == np.float32
    assert metadata == batch["meta"]
    assert all(parameter.grad is None for parameter in step.model.parameters())
    assert all(parameter.device.type == "cpu" for parameter in step.model.parameters())
    evaluated = step.eval_loss(**step.preprocess_batch(batches[0]))
    assert tuple(evaluated["model"].shape) == (2, 16)
    assert tuple(evaluated["loss"].shape) == (2,)
    assert evaluated["loss"].isfinite().all()
    assert set(evaluated.get("metrics", {})) == {
        "regression",
        "alignment",
        "GR_penalty",
    }


def test_unequal_microbatches_weight_each_sample_once() -> None:
    config = probe_recipe().step
    actual, reference = config.make(), config.make()
    reference.model.load_state_dict(actual.model.state_dict())
    batches = [probe_batches()[0], probe_batches()[2]]
    for batch in batches:
        values = cast("Batch", batch)
        loss = reference.loss(reference.model(values["media"]), **batch)["loss"]
        loss.sum().backward()
        actual.train_step(**actual.preprocess_batch(batch))
    for parameter in reference.model.parameters():
        assert parameter.grad is not None
        parameter.grad.div_(5)
    torch.nn.utils.clip_grad_norm_(
        reference.model.parameters(),
        config.gradient_clip_norm,
    )
    reference.apply_learning_rate()
    reference.optimizer.step()
    assert actual.global_step == 1
    for wanted, observed in (
        (reference.model.state_dict(), actual.model.state_dict()),
        (reference.optimizer.state_dict(), actual.optimizer.state_dict()),
    ):
        assert (
            mismatches(
                numerical_tensors(wanted),
                numerical_tensors(observed),
                ulps=True,
            )
            == []
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("precision", [None, torch.bfloat16, torch.float16])
def test_prediction_device_and_precision_boundary_retains_complete_bins(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
    precision: torch.dtype | None,
) -> None:

    config = probe_recipe().step
    config.model = tiny_model_config(dropout=0)
    config.ema = NoEMA.Config()
    config.dtype_autocast = precision
    config.autocast_cache_enabled = True
    model = config.model.make().eval()
    batches = tiny_batches()
    batch = cast("Batch", batches[0])
    with torch.no_grad():
        expected, _ = model(batch["media"])
    transfers: dict[int, list[tuple[tuple[object, ...], dict[str, object]]]] = {}
    precisions: list[tuple[str, torch.dtype | None, bool]] = []

    def transfer(value: Tensor, *args: object, **kwargs: object) -> Tensor:
        transfers.setdefault(id(value), []).append((args, kwargs))
        return value

    def autocast(
        *,
        device_type: str,
        dtype: torch.dtype | None,
        cache_enabled: bool,
    ) -> nullcontext[None]:
        precisions.append((device_type, dtype, cache_enabled))
        return nullcontext()

    monkeypatch.setattr(torch.Tensor, "to", transfer)
    monkeypatch.setattr(torch.amp, "autocast", autocast)
    config.parallelism = NoParallel.Config(device=device)
    step = config.make()
    prediction, metadata = step.predict(batches, model=model)
    assert np.array_equal(prediction, expected.numpy())
    assert metadata == batch["meta"]
    assert precisions == (
        [(device, precision, True)]
        if device == "cuda" and precision is not None
        else []
    )
    for name in ("media", "typewell_aux"):
        assert transfers[id(batches[0][name])] == [
            ((torch.device(device),), {"non_blocking": device == "cuda"}),
        ]
    live = step.call_eval(batch["media"])
    assert isinstance(live, tuple)


def test_model_constructor_stays_cpu_before_runtime_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    placement = NoParallel.__call__
    initialized: list[str] = []

    def observe(strategy: NoParallel, model: nn.Module) -> nn.Module:
        initialized.extend(parameter.device.type for parameter in model.parameters())
        assert initialized == ["cpu", "cpu"]
        return placement(strategy, model)

    monkeypatch.setattr(NoParallel, "__call__", observe)
    config = probe_recipe().step
    with torch.device("meta"):
        step = config.make()
    assert initialized == ["cpu", "cpu"]
    assert all(parameter.device.type == "cpu" for parameter in step.model.parameters())


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
