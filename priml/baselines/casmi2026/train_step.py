"""Native FPNet fine-tuning and the budgeted run around it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import inspect

from configgle import Fig, Makeable, Makes, PartialConfig
from torch import Tensor
from treekle import from_plain, to_plain

import torch

from priml.baselines.casmi2026.data import SpectrumData
from priml.baselines.casmi2026.model import FPNet, PeakAttention
from priml.loss.custom_types import LossOutput
from priml.loss.simple_loss import SimpleLoss
from priml.optimizers.composite import CompositeOptimizer
from priml.train.budget import BudgetedRun, Export
from priml.train.train_loop import TrainLoop
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from priml.train.custom_types import OptimizerProtocol, TrainStepOutput


class FingerprintLoss:
    """Score the fingerprint logits, the first model output, one value per spectrum."""

    class Config(Fig["FingerprintLoss"]):
        loss: Makeable[Callable[..., LossOutput]] = field(
            default_factory=SimpleLoss.Config,
        )
        """Unreduced per-bit loss on the float32 fingerprint logits."""

    def __init__(self, config: Config) -> None:
        self.loss = config.loss.make()

    def __call__(
        self,
        prediction: tuple[Tensor, Tensor],
        **batch: object,
    ) -> LossOutput:
        """Average the per-bit loss within each spectrum.

        Args:
          prediction: Fingerprint logits, then the element predictions.
          **batch: The batch carrying the selected-bit target.

        Returns:
          result: The mean loss of each spectrum, under ``loss``.

        """
        return {"loss": self.loss(prediction[0].float(), **batch)["loss"].mean(-1)}


class FingerprintFineTune(TrainStep):
    """Fine-tune released FPNet weights with per-bit binary cross entropy."""

    class Config(Makes["FingerprintFineTune"], TrainStep.Config[FPNet.Config]):
        model: FPNet.Config = field(default_factory=FPNet.Config)
        """The source-compatible spectral transformer."""

        pretrained: Path | None = None
        """Released checkpoint to initialize; None trains a new model."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=lambda: PartialConfig(torch.optim.AdamW, lr=1e-5),
        )
        """New fine-tuning recipe; the public optimizer was not released."""

        loss: Makeable[Callable[..., LossOutput]] = field(
            default_factory=FingerprintLoss.Config,
        )
        """Per-spectrum binary cross entropy on the fingerprint logits."""

    def __init__(self, config: Config) -> None:
        # The source recipe backpropagates the batch mean in one update, which
        # TrainStep.train_on_output's sum-then-divide rounds differently.
        if config.accumulate_grad_batches != 1:
            raise ValueError("Fingerprint source recipe requires accumulation=1.")
        super().__init__(config)
        optimizers: list[OptimizerProtocol | torch.optim.Optimizer] = [self.optimizer]
        while optimizers:
            optimizer = optimizers.pop()
            if isinstance(optimizer, CompositeOptimizer):
                optimizers.extend(optimizer.optimizers)
            closure = inspect.signature(optimizer.step).parameters.get("closure")
            # An optimizer without the attribute and one with None are both falsy.
            needs_closure = getattr(
                optimizer,
                "requires_closure",
                False,
            )  # pragma: no mutate
            if needs_closure or (
                closure is not None
                and cast(object, closure.default) is inspect.Parameter.empty
            ):
                raise ValueError(
                    "Fingerprint mean-backward recipe does not support closure optimizers.",
                )
        if config.pretrained is not None:
            payload = cast(
                object,
                torch.load(config.pretrained, map_location="cpu", weights_only=True),
            )
            checkpoint = from_plain(payload, dict[str, object])
            state = from_plain(checkpoint["model"], dict[str, Tensor])
            model = self.model
            assert isinstance(model, FPNet)
            model.load_released_state(state)

    @override
    def train_step(self, **preprocessed_batch: object) -> TrainStepOutput:
        output = from_plain(self(**preprocessed_batch), tuple[Tensor, Tensor])
        loss = self.loss(output, **preprocessed_batch)["loss"]
        loss.mean().backward()
        self.step()
        detached = (output[0].detach(), output[1].detach())
        return {"loss": loss.detach(), "model": cast(Tensor, detached)}


class CasmiTrainLoop(
    Makes["TrainLoop"],
    TrainLoop.Config[FingerprintFineTune.Config, SpectrumData.Config],
):
    """Narrow the native dataset and train-step slots for campaign forks."""

    step: FingerprintFineTune.Config = field(default_factory=FingerprintFineTune.Config)
    """Source-compatible FPNet with an explicitly new optimization recipe."""

    dataset: SpectrumData.Config = field(default_factory=SpectrumData.Config)
    """Official-identity-disjoint spectra from the prepared public CSR store."""


class FpnetExport:
    """Publish the evaluated EMA weights with the geometry that rebuilds them."""

    class Config(Fig["FpnetExport"]):
        """Declare the FPNet export; it has no tunables."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, loop: TrainLoop, output: Path) -> None:
        """Write ``fpnet.pt`` under ``output``.

        Args:
          loop: The trained loop, whose step is a fingerprint fine-tune.
          output: The run directory.

        """
        step = loop.step
        assert isinstance(step, FingerprintFineTune)
        model = step.config.model
        assert isinstance(model, FPNet.Config)
        assert isinstance(model.block.attn, PeakAttention.Config)
        with step.ema.apply_to(step.model):
            torch.save(
                {
                    "model": from_plain(
                        step.model.state_dict(),
                        dict[str, torch.Tensor],
                    ),
                    "config": to_plain(model),
                    "step": step.global_step,
                    "nbits": model.channels_out,
                    "d": model.channels_hidden,
                    "layers": model.num_layers,
                    "heads": model.block.attn.attention.num_heads,
                    "note": "New training recipe; not public-best training parity.",
                },
                output / "fpnet.pt",
            )


class CasmiRun(Makes["BudgetedRun"], BudgetedRun.Config):
    """Narrow the budgeted run to the native loop and its FPNet export."""

    loop: TrainLoop.Config = field(default_factory=CasmiTrainLoop)
    """Native dataset, training, final metrics and fresh checkpoint policy."""

    export: Makeable[Export] | None = field(default_factory=FpnetExport.Config)
    """Publication of the evaluated weights, charged to the budget."""


def casmi_loop(run: BudgetedRun.Config) -> CasmiTrainLoop:
    """Narrow a run's loop slot to the native dataset and fine-tune step."""
    assert isinstance(run.loop, CasmiTrainLoop)
    return run.loop
