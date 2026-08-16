"""Training step for the CIFAR-10 baseline.

Owns the model, its optimizers, the learning-rate schedule, and the
augmentation policy. Augmentation lives here rather than in the input pipeline
because it is an experimental variable: an experiment changes crop padding or
enables cutout by setting a field, without a second copy of the data.

The optimizer and the schedule are INJECTED, not chosen from a fixed set. A
config field naming one of two strings would mean every new optimizer needs a
branch added here, so a user could not try Lion or Adafactor without patching
the library. Instead the config carries a callable; the builders live in
:mod:`priml.optimizers.partition`.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import field
from pathlib import Path
from typing import Any, Self, cast, override

import math

from configgle import Makeable, Makes, PartialConfig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.cifar10.model import ResNet
from priml.data.augmentation_gpu import pad_crop_flip
from priml.math.schedules import Schedule, cosine
from priml.math.stats import PcaDecompose, pca_eigh
from priml.optimizers import CompositeOptimizer, apply_lr_scale
from priml.train.custom_types import TrainStepOutput
from priml.train.train_step import TrainStep


class Cifar10TrainStep(TrainStep):
    """Model plus optimization for one CIFAR-10 experiment.

    Takes the model, optimizer, and device placement from
    :class:`~priml.train.train_step.TrainStep`. What stays here is the
    augmentation policy, the smoothed loss, the fitted whitening layer, and a
    warmup counted in whole steps.
    """

    class Config(Makes["Cifar10TrainStep"], TrainStep.Config, kw_only=False):
        """Model, optimization, schedule, and augmentation for one run."""

        # ---- Inherited slots, re-defaulted for this recipe. ----

        model: Makeable[nn.Module] = field(default_factory=ResNet.Config)
        """Network to train."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=lambda: CompositeOptimizer.Config(
                optimizers=[
                    PartialConfig(
                        torch.optim.AdamW,
                        lr=1e-3,
                        betas=(0.9, 0.999),
                        weight_decay=5e-2,
                    ),
                ],
            ),
        )
        """Builds the optimizer from the model."""

        # ---- This recipe's own. ----

        total_train_steps: int = 4_000
        """Schedule horizon. Set it to the run's step budget."""

        schedule: Makeable[Schedule[float]] = field(
            default_factory=lambda: PartialConfig(cosine),
        )
        """Maps progress in ``[0, 1]`` to a learning-rate multiplier.

        Read instead of the base's ``learning_rate_scheduler``, since
        ``_apply_schedule`` drives the rate itself."""

        warmup_fraction: float = 0.02
        """Fraction of the horizon spent ramping the learning rate up."""

        label_smoothing: float = 0.1
        """Cross-entropy label smoothing."""

        translate_pad: int = 4
        """Reflect padding before the random crop, in pixels."""

        cutout_size: int = 0
        """Side of the random cutout square; 0 disables it."""

        derandomized_flip: bool = False
        """Flip every image on odd steps and none on even, instead of at random."""

        use_tta: bool = False
        """Average logits over mirrored and shifted crops at evaluation."""

        whiten_num_images: int = 5_000
        """Images used to fit a whitening layer, when the model has one."""

        whiten_cache_path: Path | str = ""
        """File caching fitted whitening weights; empty disables the cache."""

        whiten_decompose: PcaDecompose = pca_eigh
        """Eigendecomposition backing the whitening layer's PCA fit.

        The default reaches ``linalg.eigh``, which MPS lacks; pass
        ``pca_power`` there.
        """

        @override
        def finalize(self) -> Self:
            if self.warmup_fraction < 0 or self.warmup_fraction >= 1:
                raise ValueError(
                    f"warmup_fraction must be in [0, 1); got {self.warmup_fraction}.",
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if config.total_train_steps <= 0:
            raise ValueError(
                f"total_train_steps must be positive; got {config.total_train_steps}.",
            )
        super().__init__(config)
        self.config: Cifar10TrainStep.Config = config
        if self.device.type != "mps":
            # cuDNN's convolutions want this layout; MPS rejects the format on
            # some torch releases. Applied after the parallel strategy placed
            # the module, so it is the placed one that gets re-laid-out.
            self.model.to(memory_format=torch.channels_last)
        self.schedule: Schedule[float] = config.schedule.make()
        # A whitening layer is fitted from data, so it cannot be initialized in
        # the constructor -- the first training batch supplies the images.
        self._whitened = not hasattr(self.model, "init_whiten")

    @override
    def train_step(self, **batch: Any) -> TrainStepOutput:
        """Augment, forward, backward, and step the optimizers.

        Args:
          **batch: Preprocessed batch with ``media`` and ``label``.

        Returns:
          result: ``loss`` (per-example) and ``model`` (logits).

        """
        media: Tensor = batch["media"]
        label: Tensor = batch["label"]
        self._maybe_init_whiten(media)
        media = self._augment(media)

        self.model.train()
        with self._autocast():
            logits = self.model(media)
            loss = self._loss(logits, label)
        loss.sum().backward()

        # Not the inherited ``step``: the rate is scaled from a schedule this
        # recipe owns (a step-indexed warmup the goldens freeze), the clip is
        # unconditional there and gated here, and the counters advance on the
        # step timer either way.
        if math.isfinite(self.config.gradient_clip_norm):
            _ = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
            )
        with self.timer_step:
            self._apply_schedule()
            self.optimizer.step()
            self.model.zero_grad(set_to_none=True)

        return {"loss": loss.detach(), "model": logits.detach()}

    @override
    def train_loss(self, **batch: Any) -> TrainStepOutput:
        """Compute the training loss without a backward pass."""
        media: Tensor = batch["media"]
        label: Tensor = batch["label"]
        self.model.train()
        with torch.no_grad(), self._autocast():
            logits = self.model(media)
            loss = self._loss(logits, label)
        return {"loss": loss, "model": logits}

    @override
    def eval_loss(self, **batch: Any) -> TrainStepOutput:
        """Compute the evaluation loss and logits."""
        media: Tensor = batch["media"]
        label: Tensor = batch["label"]
        logits = self.call_eval(media=media)
        return {"loss": self._loss(logits, label), "model": logits}

    @override
    def call_eval(self, **batch: Any) -> Tensor:
        """Return evaluation logits, optionally averaged over augmentations."""
        media: Tensor = batch["media"]
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            if self.config.use_tta:
                return _tta_logits(self.model, media)
            logits = self.model(media)
            assert isinstance(logits, Tensor)
            return logits

    @override
    def on_epoch_end(self) -> None:
        """Do nothing: this step accumulates nothing across a boundary."""

    @override
    def state_dict(self) -> dict[str, Any]:
        """Extend the base state with whether the whitening layer was fitted.

        Fitting happens once, from the first batch seen, so a resume that
        forgot it would re-fit against a model whose weights had already moved.
        """
        state = super().state_dict()
        state["whitened"] = self._whitened
        return state

    @override
    def load_state_dict(self, state_dict: dict[str, Any], **kwargs: Any) -> None:
        """Restore state produced by :meth:`state_dict`."""
        super().load_state_dict(state_dict, **kwargs)
        self._whitened = state_dict["whitened"]

    @property
    @override
    def progress_learning_schedule(self) -> float:
        """Fraction of ``total_train_steps`` spent, in ``[0, 1]``."""
        spent = self.global_step / self.config.total_train_steps
        return 1.0 if spent > 1.0 else float(spent)

    def _apply_schedule(self) -> None:
        """Scale every parameter group's learning rate for the current step.

        The ramp stays indexed by STEP rather than composed as a
        :func:`~priml.math.schedules.warmup` factor: it counts
        ``int(warmup_fraction * total)`` whole steps and reads one step ahead,
        which is not the same number as a fraction of progress -- and the
        goldens freeze the rate this produces.
        """
        config = self.config
        multiplier = self.schedule(self.progress_learning_schedule)
        warmup_steps = int(config.warmup_fraction * config.total_train_steps)
        if warmup_steps and self.global_step < warmup_steps:
            multiplier *= (self.global_step + 1) / warmup_steps
        apply_lr_scale([self.optimizer], multiplier)

    def _augment(self, media: Tensor) -> Tensor:
        """Apply the configured training augmentation to a batch."""
        config = self.config
        flip = self.global_step % 2 == 1 if config.derandomized_flip else None
        return pad_crop_flip(
            media,
            media.shape[-1],
            pad=config.translate_pad,
            flip=flip,
            cutout_size=config.cutout_size,
        )

    def _loss(self, logits: Tensor, label: Tensor) -> Tensor:
        """Return per-example smoothed cross-entropy."""
        return functional.cross_entropy(
            logits.float(),
            label,
            label_smoothing=self.config.label_smoothing,
            reduction="none",
        )

    def _maybe_init_whiten(self, media: Tensor) -> None:
        """Fit the model's whitening layer once, from the first batch seen."""
        if self._whitened:
            return
        model = cast("Any", self.model)
        configured = self.config.whiten_cache_path
        cache = Path(configured) if configured else None
        if cache is not None and cache.is_file():
            model.whiten.weight.data.copy_(
                torch.load(cache, map_location=self.device, weights_only=True),
            )
        else:
            model.init_whiten(
                media[: self.config.whiten_num_images],
                decompose=self.config.whiten_decompose,
            )
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.whiten.weight.data, cache)
        self._whitened = True

    @contextmanager
    def _autocast(self) -> Generator[None]:
        """Enter autocast when a mixed-precision dtype is configured."""
        dtype = self.config.dtype_autocast
        if dtype is None:
            yield
            return
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=dtype,
            cache_enabled=False,
        ):
            yield


def _tta_logits(model: nn.Module, media: Tensor) -> Tensor:
    """Average logits over the mirror pair of three overlapping crops.

    Six forward passes: the image and two one-pixel-shifted crops, each paired
    with its horizontal mirror. Shifts come from a reflect-padded copy, so no
    crop introduces a border the network never saw in training.

    Args:
      model: Network in evaluation mode.
      media: ``(B, 3, H, W)`` images.

    Returns:
      logits: ``(B, num_classes)`` averaged class scores.

    """

    def mirrored(view: Tensor) -> Tensor:
        averaged = 0.5 * (model(view) + model(view.flip(-1)))
        assert isinstance(averaged, Tensor)
        return averaged

    size = media.shape[-1]
    padded = functional.pad(media, (1,) * 4, "reflect")
    return (
        mirrored(media)
        + mirrored(padded[..., 0:size, 0:size])
        + mirrored(padded[..., 2 : size + 2, 2 : size + 2])
    ) / 3
