"""Train the complete depth posterior through shared Priml optimization slots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import field
from typing import TYPE_CHECKING, cast, override

import contextlib

from configgle import Makeable, Makes
from torch import Tensor

import numpy as np
import torch

from priml.baselines.wellbore_geology.custom_types import (
    Batch,
    BatchLoader,
    InitializableEMA,
)
from priml.baselines.wellbore_geology.loss import WellboreLoss
from priml.baselines.wellbore_geology.model import SequenceModel
from priml.loss.custom_types import LossOutput
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from priml.train.custom_types import TrainStepOutput


class WellboreTrainStep(TrainStep):
    """Keep posterior loss formation local and reuse shared update behavior."""

    class Config(
        Makes["WellboreTrainStep"],
        TrainStep.Config[SequenceModel.Config],
        kw_only=True,
    ):
        model: SequenceModel.Config = field(default_factory=SequenceModel.Config)
        """Complete four-stage posterior and regression model."""

        loss: Makeable[Callable[..., LossOutput]] = field(
            default_factory=WellboreLoss.Config,
        )
        """Complete posterior objective and all three unweighted terms."""

    def __init__(self, config: Config) -> None:
        with torch.device("cpu"):
            super().__init__(config)
        self.config: WellboreTrainStep.Config = config  # pyrefly: ignore[bad-override-mutable-attribute] -- Narrowed to this class's own config; nothing assigns the base's.
        self.model.format_backbone(self.device)
        if isinstance(self.ema, InitializableEMA):
            self.ema.initialize(self.model)

    @property
    @override
    def model(self) -> SequenceModel:
        return cast("SequenceModel", self._model)

    @override
    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        """Move every complete training tensor while retaining host row metadata."""
        moved = super().preprocess_batch(batch)
        values = cast("Batch", batch)
        moved["aux_targets"] = {
            name: value.to(self.device, non_blocking=True)
            for name, value in values["aux_targets"].items()
        }
        return moved

    @override
    def train_step(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Accumulate the posterior loss through the shared sample-weighted update.

        Args:
          **preprocessed_batch: Complete collated tensors and original-row metadata.

        Returns:
          result: Per-sample loss, full posterior expectation and all loss terms.

        """
        batch = cast("Batch", preprocessed_batch)
        result = self.train_on_output(
            self(batch["media"]),
            closure=None,
            **preprocessed_batch,
        )
        return {
            "loss": result["loss"].detach(),
            "model": result["model"].detach(),
            "metrics": {
                name: value.detach() if isinstance(value, Tensor) else value
                for name, value in result.get("metrics", {}).items()
            },
        }

    @override
    def on_epoch_end(self) -> None:
        """Apply the final short accumulation instead of carrying it over."""
        if not self.accumulation_steps:
            return
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(self.accumulated_samples)
        self.step()
        self.accumulation_steps = 0
        self.accumulated_samples = 0

    @override
    def eval_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Evaluate the complete posterior objective through the shared EMA slot."""
        batch = cast("Batch", preprocessed_batch)
        return self._loss_on_output(
            self.call_eval(batch["media"]),
            **preprocessed_batch,
        )

    def predict(
        self,
        loader: BatchLoader,
        *,
        model: SequenceModel | None = None,
        move_cpu: bool = False,
    ) -> tuple[NDArray[np.float32], list[dict[str, object]]]:
        """Predict every bin and retain every original-row observation.

        Args:
          loader: Complete validation wells in declared order.
          model: Explicit selected checkpoint, or None for the current EMA shadow.
          move_cpu: Release the explicit model after complete prediction.

        Returns:
          prediction: Every float32 bin prediction in loader order.
          metadata: Every original-row observation in matching well order.

        """
        if model is not None:
            model.to(self.device).eval()
            model.format_backbone(self.device)
        predictions: list[NDArray[np.float32]] = []
        metadata: list[dict[str, object]] = []
        with torch.no_grad():
            for raw in loader:
                batch = cast("Batch", self.preprocess_batch(raw))
                with self._autocast_scope():
                    output = (
                        self.call_eval(batch["media"])
                        if model is None
                        else model(batch["media"])
                    )
                prediction, _ = cast("tuple[Tensor, dict[str, Tensor]]", output)
                predictions.append(prediction.float().cpu().numpy())
                metadata.extend(batch["meta"])
        if model is not None and move_cpu:
            model.cpu()
        return np.concatenate(predictions), metadata

    @override
    def load_state_dict(
        self,
        state_dict: Mapping[str, object],
        *,
        strict: bool = True,
        load_optimizer: bool = True,
        remap: Callable[[Mapping[str, Tensor]], Mapping[str, Tensor]] | None = None,
    ) -> None:
        """Reject resume until persistent-worker streams can be restored exactly."""
        del state_dict, strict, load_optimizer, remap
        raise NotImplementedError(
            "Wellbore native worker streams do not support exact mid-fold resume.",
        )

    @override
    def _loss_on_output(
        self,
        output: object,
        **preprocessed_batch: object,
    ) -> TrainStepOutput:
        """Form the loss under autocast and expose the expectation as the model output."""
        prediction, _ = cast("tuple[Tensor, dict[str, Tensor]]", output)
        with self._autocast_scope():
            losses = self.loss(output, **preprocessed_batch)
        return {
            "loss": losses["loss"],
            "model": prediction,
            "metrics": {
                name: value for name, value in losses.items() if name != "loss"
            },
        }

    @override
    def _autocast_scope(self) -> contextlib.AbstractContextManager[object]:
        if self.device.type == "cpu":
            return contextlib.nullcontext()
        return super()._autocast_scope()
