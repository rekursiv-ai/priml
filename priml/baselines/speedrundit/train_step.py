"""One SR-DiT optimizer update.

The recipe drives the update itself rather than calling the inherited
:meth:`TrainStep.step`, for one reason: the objective returns six named terms
and the step publishes each of them, so the loss cannot go through the base's
single-tensor ``loss`` slot. The optimizer, the EMA, and the step counter are
the base's; the gradient clip is called here, between backward and the timed
update. The ``with self.timer_step:`` bracket is what advances ``global_step``
so every cadence above still means what it says.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import field
from typing import TYPE_CHECKING, cast, override

from configgle import Makeable, Makes, PartialConfig
from torch import Tensor, nn

import torch

from priml.baselines.speedrundit.loss import SpeedrunDiTLoss
from priml.baselines.speedrundit.model import SpeedrunDiT
from priml.optimizers.parameter_filter import everything
from priml.train.custom_types import EMAProtocol
from priml.train.ema import EMA
from priml.train.grad_clip import clip_grad_norm_
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from priml.train.custom_types import TrainStepOutput


__all__ = ["InitialWeightsEMA", "SpeedrunDiTTrainStep"]


class SpeedrunDiTTrainStep(TrainStep):
    """Flow-matching update over prepared latents."""

    class Config(
        Makes["SpeedrunDiTTrainStep"],
        TrainStep.Config[SpeedrunDiT.Config],
        kw_only=True,
    ):
        """Configuration for SpeedrunDiTTrainStep."""

        # ---- Inherited slots, re-defaulted for this recipe. ----

        model: SpeedrunDiT.Config = field(default_factory=SpeedrunDiT.Config)
        """The velocity field."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=lambda: PartialConfig(
                torch.optim.AdamW,
                lr=1e-4,
                betas=(0.9, 0.999),
                weight_decay=0.0,
                eps=1e-8,
            ),
        )
        """A single AdamW group over every parameter.

        Not a ``CompositeOptimizer``: the reference hands AdamW
        ``model.parameters()`` whole, and a router that skipped the frozen
        position table would build a different parameter list and a different
        optimizer state."""

        gradient_clip_norm: float = 1.0
        """Global gradient-norm ceiling."""

        ema: Makeable[EMAProtocol] = field(
            default_factory=lambda: InitialWeightsEMA.Config(
                decay=0.9999,
                update_after_step=0,
                track_buffers=False,
                shadow_kind="module",
                select=everything,
            ),
        )
        """Weight-averaging shadow, seeded at construction and updated after
        each optimizer step.

        Seeded from the initial weights, as the reference seeds its copy
        (``train.py:225``). It averages the frozen position table too, as the
        reference's ``update_ema`` walks every ``named_parameter`` -- a drift
        its own author flags as unintended (``train.py:70``) but which the
        reference's scores were measured under."""

        dtype_autocast: torch.dtype | None = torch.bfloat16
        """Autocast dtype for the model forward; ``None`` runs in full precision.

        The forward alone, with its outputs cast back to float32: the
        objective's arithmetic stays float32, as ``accelerate.prepare`` leaves
        it in the reference."""

        # ---- This recipe's own. ----

        objective: SpeedrunDiTLoss.Config = field(
            default_factory=SpeedrunDiTLoss.Config,
        )
        """The four-term flow-matching objective.

        A slot of its own rather than the base's ``loss``: that one is typed
        to return a single tensor from ``(prediction, **batch)``, and this
        objective drives the model itself so it can build the noised input the
        model consumes."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config: SpeedrunDiTTrainStep.Config = config
        self.objective = config.objective.make()
        # Snapshot before any update, as ``train.py:225`` does; a no-op for any
        # averager that is not this recipe's own.
        if isinstance(self.ema, InitialWeightsEMA):
            self.ema.snapshot(self.model)

    @property
    @override
    def model(self) -> SpeedrunDiT:
        """The velocity field, narrowed.

        Returns:
          model: The built model.

        """
        built = super().model
        assert isinstance(built, SpeedrunDiT)
        return built

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Take one optimizer step.

        Args:
          **batch: A :class:`~priml.baselines.speedrundit.data.SpeedrunDiTBatch`.

        Returns:
          output: The total loss, a probe of the velocity, and every term.

        """
        self.model.train()
        result = self._evaluate(batch)
        result.loss.backward()
        # Clipped outside the timer bracket: the bracket is what advances
        # ``global_step``, and the clip is not part of the update it counts.
        if self.config.gradient_clip_norm != float("inf"):
            _ = clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
            )
        # This bracket is what advances ``global_step``, so every cadence
        # above -- eval, checkpoint, the schedule horizon -- counts one
        # optimizer update per pass through it.
        with self.timer_step:
            self.apply_learning_rate()
            self.optimizer.step()
            self.model.zero_grad(set_to_none=True)
            self.ema(self.model)
        return {
            "loss": result.loss.detach(),
            "model": result.denoising.detach(),
            "metrics": _metrics(result),
        }

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Score one batch without updating anything.

        Args:
          **batch: A :class:`~priml.baselines.speedrundit.data.SpeedrunDiTBatch`.

        Returns:
          output: The total loss and a probe of the velocity error.

        """
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode(), self.ema.apply_to(self.model):
                result = self._evaluate(batch)
        finally:
            self.model.train(was_training)
        return {
            "loss": result.loss.detach(),
            "model": result.denoising.detach(),
            "metrics": _metrics(result),
        }

    # The base's autocast wraps the whole loss; this recipe's reference casts only the
    # forward, so the objective reduces in float32. Wrapping the objective instead
    # measured different gradients at step one.
    def _forward(
        self,
        media: Tensor,
        time: Tensor,
        label: Tensor,
        cls_token: Tensor,
        /,
    ) -> SpeedrunDiT.Output:
        """Run the model under autocast and return float32 outputs."""
        dtype = self.config.dtype_autocast
        context = (
            nullcontext()
            if dtype is None
            else torch.amp.autocast(
                device_type=self.device.type,
                dtype=dtype,
                cache_enabled=self.config.autocast_cache_enabled,
            )
        )
        with context:
            out = self.model(media, time, label, cls_token)
        return SpeedrunDiT.Output(
            velocity=out.velocity.float(),
            projections=[p.float() for p in out.projections],
            cls_velocity=out.cls_velocity.float(),
        )

    def _evaluate(self, batch: dict[str, object]) -> SpeedrunDiTLoss.Output:
        """Run the objective over one batch."""
        media = batch["media"]
        label = batch["label"]
        cls_token = batch["cls_token"]
        features = batch.get("features", [])
        assert isinstance(media, Tensor)
        assert isinstance(label, Tensor)
        assert isinstance(cls_token, Tensor)
        assert isinstance(features, list)
        return self.objective(
            self._forward,
            media=media,
            label=label,
            cls_token=cls_token,
            features=cast("list[Tensor]", features),
        )


class InitialWeightsEMA(EMA):
    """An EMA whose shadow starts at the weights before the first update.

    The shared :class:`~priml.train.ema.EMA` seeds lazily on its first call,
    from the weights one optimizer step in. The reference copies its shadow
    from the initial weights instead (``train.py:225``), and the two differ
    by that first step's update, decayed but never exactly zero. Only this
    recipe reproduces that choice, so it lives here rather than on ``EMA``.
    """

    class Config(Makes["InitialWeightsEMA"], EMA.Config):
        """Configuration for InitialWeightsEMA; see :class:`EMA.Config`."""

    def snapshot(self, model: nn.Module) -> None:
        """Seed the shadow from ``model`` now, advancing no counter.

        Args:
          model: The model at its initial weights.

        """
        self._lazy_initialize(model)


def _metrics(result: SpeedrunDiTLoss.Output) -> dict[str, float | Tensor]:
    """Publish every term of the objective, keyed for the tracker."""
    return {
        "denoising": result.denoising.detach().mean(),
        "cls": result.cls.detach().mean(),
        "projection": result.projection.detach(),
        "cfm": result.cfm.detach(),
        "cfm_cls": result.cfm_cls.detach(),
    }
