"""priml training step for the SpeedrunDiT objective and optimizer split."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING, Literal, cast, override

from configgle import Makeable, Makes
from torch import Tensor, nn

import torch

from priml.baselines.speedrundit.model import ModelOutput, SpeedrunDiT
from priml.baselines.speedrundit.objective import LossTerms, SpeedrunObjective
from priml.baselines.speedrundit.optimizers import speedrundit_optimizer
from priml.model.dinov2 import DinoV2Teacher
from priml.model.invae import LATENT_SCALE
from priml.train.custom_types import EMAProtocol
from priml.train.ema import EMA
from priml.train.train_step import TrainStep, _assert_uniform_microbatch_count


if TYPE_CHECKING:
    from priml.train.custom_types import TrainStepOutput


class SpeedrunTrainStep(TrainStep):
    """Frozen DINO targets plus the four-term latent flow objective."""

    class Config(
        Makes["SpeedrunTrainStep"], TrainStep.Config[SpeedrunDiT.Config], kw_only=True
    ):
        model: SpeedrunDiT.Config = field(default_factory=SpeedrunDiT.Config)
        """Latent SiT student architecture."""
        teacher: Makeable[nn.Module] = field(default_factory=DinoV2Teacher.Config)
        """Frozen representation teacher."""
        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=speedrundit_optimizer
        )
        """AdamW and Muon parameter split."""
        dtype_autocast: torch.dtype | None = torch.bfloat16
        """Autocast dtype for the student and teacher."""
        compile: (
            Makeable[Callable[[Callable[..., object]], Callable[..., object]]] | None
        ) = None
        """Optional training compilation; disabled in the reference branch."""
        ema: Makeable[EMAProtocol] = field(
            default_factory=lambda: cast(
                Makeable[EMAProtocol], EMA.Config(decay=0.9999, track_buffers=False)
            )
        )
        """Exponential moving average of student parameters."""
        gradient_clip_norm: float = 1.0
        """Global gradient norm clipping threshold."""
        latent_scale: float = LATENT_SCALE
        """Scale applied to sampled INVAE latents."""
        projection_coeff: float = 0.5
        """REG projection loss weight."""
        cls_coeff: float = 0.03
        """CLS flow loss weight."""
        cfm_coeff: float = 0.05
        """Contrastive flow loss weight."""
        cfm_weighting: Literal["uniform", "linear"] = "uniform"
        """Time weighting used by contrastive flow matching."""
        time_shifting: bool = True
        """Apply the reference resolution-dependent time shift."""
        shift_base: int = 4096
        """Reference latent dimension for time shifting."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config: SpeedrunTrainStep.Config = config
        self.teacher = (
            config.teacher.make().to(self.device).eval().requires_grad_(False)
        )
        self.objective = SpeedrunObjective(
            projection_coeff=config.projection_coeff,
            cls_coeff=config.cls_coeff,
            cfm_coeff=config.cfm_coeff,
            cfm_weighting=config.cfm_weighting,
            shift_time=config.time_shifting,
            shift_base=config.shift_base,
        )

    @override
    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        moved = super().preprocess_batch(batch)
        latent = moved["latent"]
        assert isinstance(latent, Tensor)
        moved["latent"] = latent.float() * self.config.latent_scale
        return moved

    def _terms(self, batch: dict[str, object], *, evaluate: bool) -> LossTerms:
        image, latent, label = (batch["image"], batch["latent"], batch["label"])
        assert isinstance(image, Tensor)
        assert isinstance(latent, Tensor)
        assert isinstance(label, Tensor)
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.config.dtype_autocast or torch.bfloat16,
                enabled=self.config.dtype_autocast is not None,
            ),
        ):
            teacher_features = cast(tuple[Tensor, ...], self.teacher(image))
        if evaluate:
            model = cast(Callable[..., ModelOutput], self.call_eval)
        else:
            model = cast(Callable[..., ModelOutput], self.__call__)
        return self.objective(model, latent, label, teacher_features)

    @staticmethod
    def _result(terms: LossTerms) -> TrainStepOutput:
        return {
            "loss": terms.loss.detach(),
            "model": terms.output.velocity.detach(),
            "metrics": {
                "velocity_loss": terms.velocity.mean().detach(),
                "cls_loss": terms.cls.mean().detach(),
                "projection_loss": terms.projection.mean().detach(),
                "cfm_loss": terms.cfm.mean().detach(),
            },
        }

    @override
    def train_step(self, **preprocessed_batch: object) -> TrainStepOutput:
        terms = self._terms(preprocessed_batch, evaluate=False)
        terms.mean_loss.backward()
        self.accumulated_samples += terms.velocity.numel()
        self.accumulation_steps += 1
        if self.accumulation_steps >= self.accumulate_grad_batches:
            _assert_uniform_microbatch_count(self.accumulated_samples)
            if self.accumulation_steps > 1:
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(self.accumulation_steps)
            self.step()
            self.accumulation_steps = 0
            self.accumulated_samples = 0
        return self._result(terms)

    @override
    def train_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        return self._result(self._terms(preprocessed_batch, evaluate=False))

    @override
    def eval_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        with torch.inference_mode():
            return self._result(self._terms(preprocessed_batch, evaluate=True))
