"""Small deterministic records shared by ETTh1 tests and source verification."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from torch import Tensor, nn

import torch

from priml.baselines.etth1.experiments import exp000
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from priml.baselines.etth1.train_step import Etth1TrainStep


class _OptimizerState(Protocol):
    def state_dict(self) -> Mapping[str, object]: ...


def tiny_config() -> Etth1TrainStep.Config:
    """Keep exp000's recipe, shrinking only forecasting geometry.

    Returns:
      cfg: Seeded training step with a five-step history and three-step horizon.

    """
    cfg = exp000().step
    cfg.seed = exp000().seed
    cfg.model.seq_len = 5
    cfg.model.pred_len = 3
    cfg.model.channels = 4
    return cfg


def tiny_batches() -> Iterator[dict[str, Tensor]]:
    """Yield three distinct nonrandom batches with no broadcastable axes.

    Yields:
      batch: Float32 media shaped (2, 5, 4) and labels shaped (2, 3, 4).

    """
    for index in range(3):
        yield {
            "media": torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4) / 17
            - 1
            + index / 8,
            "label": torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) / 19
            - 1
            - index / 8,
        }


def model_record(model: nn.Module) -> dict[str, Tensor]:
    """Capture all initialized parameters, a complete forward, and RNG position.

    Args:
      model: Initialized forecasting model accepting a tiny batch's media.

    Returns:
      record: Initial parameters by name, input, output, and RNG fingerprint.

    """
    batch = next(tiny_batches())
    return {
        **{
            f"initial/{key}": value.detach().clone()
            for key, value in model.state_dict().items()
        },
        "rng": rng_fingerprint(),
        "input": batch["media"],
        "output": cast(Tensor, model(batch["media"])).detach().clone(),
    }


def update_record(
    model: nn.Module,
    optimizer: _OptimizerState,
    output: Tensor,
    loss: Tensor,
) -> dict[str, Tensor]:
    """Record every output, gradient, updated parameter, and optimizer state.

    Args:
      model: Forecasting model after an optimization update.
      optimizer: Optimizer holding the updated state.
      output: Predictions from the update's forward pass.
      loss: Mean squared error from the same forward pass.

    Returns:
      record: Detached tensors for output, loss, RNG, parameters, gradients,
        and flattened optimizer state.

    """
    record = {
        "output": output.detach().clone(),
        "loss": loss.detach().reshape(1).clone(),
        "rng": rng_fingerprint(),
        **{
            f"parameter/{key}": value.detach().clone()
            for key, value in model.state_dict().items()
        },
        **{
            f"gradient/{key}": parameter.grad.detach().clone()
            for key, parameter in model.named_parameters()
            if parameter.grad is not None
        },
    }
    _flatten_state(record, prefix="optimizer", value=optimizer.state_dict())
    return record


def training_record(step: Etth1TrainStep) -> dict[str, Tensor]:
    """Run three updates across the first schedule decay and capture all state.

    Args:
      step: Training step initialized with the tiny reference recipe.

    Returns:
      record: Initial model record followed by each update under stepN keys.

    """
    record = model_record(step.model)
    for index, batch in enumerate(tiny_batches()):
        step.timer_epoch.global_count = index
        output = step.train_step(**batch)
        record.update(
            {
                f"step{index + 1}/{key}": value
                for key, value in update_record(
                    step.model,
                    optimizer=step.optimizer,
                    output=output["model"],
                    loss=output["loss"],
                ).items()
            },
        )
    return record


def golden_record(record: Mapping[str, Tensor], *, training: bool) -> dict[str, Tensor]:
    """Keep initial weights, losses, and the final state for replay.

    Args:
      record: A full model or training record.
      training: Whether ``record`` came from ``training_record``.

    Returns:
      golden: The subset of ``record`` the checked-in golden stores.

    """
    return {
        key: value
        for key, value in record.items()
        if key.startswith("initial/")
        or (
            training
            and (
                key in {"step1/loss", "step2/loss", "step3/loss", "step3/rng"}
                or key.startswith(("step3/parameter/", "step3/optimizer/"))
            )
        )
        or (not training and key in {"output", "rng"})
    }


def _flatten_state(record: dict[str, Tensor], prefix: str, value: object) -> None:
    if isinstance(value, Tensor):
        record[prefix] = value.detach().clone()
    elif isinstance(value, (int, float, bool)):
        record[prefix] = torch.tensor(value)
    elif isinstance(value, dict):
        for key, item in cast("Mapping[str | int, object]", value).items():
            # This metadata is PRIML-only; keep the optimizer values below it.
            if key == "initial_lr":
                continue
            _flatten_state(record, prefix=f"{prefix}/{key}", value=item)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(cast("list[object]", value)):
            _flatten_state(record, prefix=f"{prefix}/{index}", value=item)
    elif value is not None:
        raise TypeError(f"Unsupported optimizer state: {type(value).__name__}")
