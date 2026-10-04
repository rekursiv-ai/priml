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
from priml.model.vision_ae.custom_types import LatentNormalizer
from priml.train.custom_types import EMAProtocol
from priml.train.ema import EMA
from priml.train.train_step import TrainStep, _assert_uniform_microbatch_count


if TYPE_CHECKING:
    from priml.train.custom_types import TrainStepOutput


class SpeedrunTrainStep(TrainStep):
    """Frozen DINO targets plus the four-term latent flow objective."""

    class Config(
        Makes["SpeedrunTrainStep"],
        TrainStep.Config[SpeedrunDiT.Config],
        kw_only=True,
    ):
        model: SpeedrunDiT.Config = field(default_factory=SpeedrunDiT.Config)
        """Latent SiT student architecture."""

        teacher: Makeable[nn.Module] = field(default_factory=DinoV2Teacher.Config)
        """Frozen representation teacher."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=speedrundit_optimizer,
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
                Makeable[EMAProtocol],
                EMA.Config(
                    decay=0.9999,
                    track_buffers=False,
                    shadow_kind="param_dict",
                ),
            ),
        )
        """Exponential moving average of the student's parameters.

        Kept as parameter shadows: a module-copy shadow cannot deepcopy a
        composable distributed model.
        """

        gradient_clip_norm: float = 1.0
        """Global gradient norm clipping threshold."""

        latent_norm: Makeable[LatentNormalizer] | None = None
        """Maps decoded raw latents into diffusion space.

        ``None`` is filled at loop finalize with the normalizer published for
        the dataset's autoencoder; set it to train on another."""

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
        if config.latent_norm is None:
            raise ValueError(
                "latent_norm is unset; SpeedrunTrainLoop.finalize fills it from the "
                "dataset's autoencoder, or set it on the step directly.",
            )
        super().__init__(config)
        if getattr(self.optimizer, "requires_closure", False):
            raise ValueError(
                "SpeedrunTrainStep cannot drive a closure-based optimizer: its "
                "objective draws fresh times and noise on every call, so a closure "
                "would recompute a different loss.",
            )
        self.config: SpeedrunTrainStep.Config = config
        self.reference_samples = 0
        self.latent_norm: LatentNormalizer = config.latent_norm.make()
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
        moved["latent"] = self.latent_norm.normalize(latent.float())
        return moved

    @override
    def train_step(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Backpropagate one micro-batch; step once ``accumulate_grad_batches`` ran.

        The update's gradient is the sample-weighted mean over the micro-batches,
        as one batch of them all would give, except that the contrastive term
        pairs samples within each micro-batch.

        Args:
          **preprocessed_batch: ``image``, ``latent``, and ``label`` on device.

        Returns:
          output: Per-sample ``loss``, the student's velocity as ``model``, each
            loss term's mean, and ``skipped_steps`` when non-finite gradients skip
            updates.

        """
        terms = self._terms(preprocessed_batch, evaluate=False)
        samples = terms.velocity.numel()
        if self.accumulation_steps == 0:
            self.reference_samples = samples
        # Weighted against the first micro-batch rather than divided by the total:
        # a lone or equal-size micro-batch then backpropagates its own mean, bit for
        # bit, while unequal ones still weigh by their sample counts.
        (terms.mean_loss * (samples / self.reference_samples)).backward()
        self.accumulated_samples += samples
        self.accumulation_steps += 1
        if self.accumulation_steps >= self.accumulate_grad_batches:
            _assert_uniform_microbatch_count(self.accumulated_samples)
            share = self.accumulated_samples / self.reference_samples
            if share != 1:
                for parameter in self.model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(share)
            self.step()
            self.accumulation_steps = 0
            self.accumulated_samples = 0
        result = self._result(terms)
        if self.skip_step_on_nonfinite_grad:
            result.setdefault("metrics", {})["skipped_steps"] = self.skipped_steps
        return result

    @override
    def train_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        return self._result(self._terms(preprocessed_batch, evaluate=False))

    @override
    def eval_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        with torch.inference_mode():
            return self._result(self._terms(preprocessed_batch, evaluate=True))

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
            teacher_features = cast("object", self.teacher(image))
        # The slot takes any module, so its output is checked rather than cast: a
        # bare tensor would otherwise fail later as an ambiguous tensor truth value.
        if not isinstance(teacher_features, tuple):
            raise TypeError(
                "The teacher must return a tuple of feature maps; got "
                f"{type(teacher_features).__name__}.",
            )
        if evaluate:
            model = cast(Callable[..., ModelOutput], self.call_eval)
        else:
            model = cast(Callable[..., ModelOutput], self.__call__)
        return self.objective(
            model,
            latents=latent,
            labels=label,
            teacher_features=cast("tuple[Tensor, ...]", teacher_features),
        )

    @classmethod
    def _result(cls, terms: LossTerms) -> TrainStepOutput:
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
