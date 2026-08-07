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

from configgle import Fig, Makeable, PartialConfig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.cifar10.model import PcaFn, ResNet
from priml.data.augmentation_gpu import pad_crop_flip
from priml.math.stats import pca
from priml.optimizers import (
    CompositeOptimizer,
    apply_lr_scale,
    remember_initial_lrs,
)
from priml.runtime import get_device
from priml.train.custom_types import TrainStepOutput
from priml.train.schedules import Schedule, cosine


class Cifar10TrainStep:
    """Model plus optimization for one CIFAR-10 experiment.

    Implements ``TrainStepProtocol``: the training loop calls
    :meth:`train_step` per batch and :meth:`eval_loss` per evaluation, and
    persists the whole thing through :meth:`state_dict`.
    """

    class Config(Fig["Cifar10TrainStep"]):
        """Model, optimization, schedule, and augmentation for one run."""

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
        """Builds the optimizer from the model.

        Injected, not selected from a fixed set: a new optimizer is a config a
        caller supplies, never a branch added here. A split recipe routes
        parameters with ``CompositeOptimizer.Config.select`` and still presents
        as one optimizer."""

        total_train_steps: int = 4_000
        """Schedule horizon. Set it to the run's step budget."""

        schedule: Makeable[Schedule] = field(
            default_factory=lambda: PartialConfig(cosine),
        )
        """Maps ``(step, total_steps)`` to a learning-rate multiplier.

        Any callable of that shape works, so a caller supplies a schedule
        rather than choosing from an enumeration -- see
        :mod:`priml.train.schedules` for the built-in shapes."""

        warmup_fraction: float = 0.02
        """Fraction of the horizon spent ramping the learning rate up."""

        label_smoothing: float = 0.1
        """Cross-entropy label smoothing."""

        gradient_clip_norm: float = math.inf
        """Global gradient-norm cap; infinite disables clipping."""

        translate_pad: int = 4
        """Reflect padding before the random crop, in pixels."""

        cutout_size: int = 0
        """Side of the random cutout square; 0 disables it."""

        derandomized_flip: bool = False
        """Flip every image on odd steps and none on even, instead of at random."""

        use_tta: bool = False
        """Average logits over mirrored and shifted crops at evaluation."""

        device: str = "auto"
        """Device to train on ("auto" picks the best available)."""

        dtype_autocast: torch.dtype | None = None
        """Autocast dtype; ``None`` trains in full precision."""

        compile: bool = False
        """Compile the model with ``torch.compile``."""

        whiten_num_images: int = 5_000
        """Images used to fit a whitening layer, when the model has one."""

        whiten_cache_path: Path | str = ""
        """File caching fitted whitening weights; empty disables the cache."""

        whiten_fit: PcaFn = pca
        """PCA fitter for the whitening layer.

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
        self.config = config
        self.device = get_device(config.device)
        self.global_step: int = 0
        self.local_step: int = 0
        self.model: nn.Module = config.model.make()
        self.model.to(self.device)
        if self.device.type != "mps":
            self.model.to(memory_format=torch.channels_last)
        if config.compile:
            self.model = cast("nn.Module", torch.compile(self.model))
        self.schedule: Schedule = config.schedule.make()
        self.optimizer: torch.optim.Optimizer = config.optimizer.make()(self.model)
        remember_initial_lrs([self.optimizer])
        # A whitening layer is fitted from data, so it cannot be initialized in
        # the constructor -- the first training batch supplies the images.
        self._whitened = not hasattr(self.model, "init_whiten")

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move a batch to the training device."""
        return {
            key: value.to(self.device, non_blocking=True)
            if isinstance(value, Tensor)
            else value
            for key, value in batch.items()
        }

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

        if math.isfinite(self.config.gradient_clip_norm):
            _ = nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
            )
        self._apply_schedule()
        self.optimizer.step()
        self.model.zero_grad(set_to_none=True)

        self.global_step += 1
        self.local_step += 1
        return {"loss": loss.detach(), "model": logits.detach()}

    def train_loss(self, **batch: Any) -> TrainStepOutput:
        """Compute the training loss without a backward pass."""
        media: Tensor = batch["media"]
        label: Tensor = batch["label"]
        self.model.train()
        with torch.no_grad(), self._autocast():
            logits = self.model(media)
            loss = self._loss(logits, label)
        return {"loss": loss, "model": logits}

    def eval_loss(self, **batch: Any) -> TrainStepOutput:
        """Compute the evaluation loss and logits."""
        media: Tensor = batch["media"]
        label: Tensor = batch["label"]
        logits = self.call_eval(media=media)
        return {"loss": self._loss(logits, label), "model": logits}

    def call_eval(self, **batch: Any) -> Tensor:
        """Return evaluation logits, optionally averaged over augmentations."""
        media: Tensor = batch["media"]
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            if self.config.use_tta:
                return _tta_logits(self.model, media)
            return self.model(media)

    def on_epoch_end(self) -> None:
        """Do nothing: this step holds no state across epoch boundaries."""

    def state_dict(self) -> dict[str, Any]:
        """Return model, optimizer, and progress state for checkpointing."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "whitened": self._whitened,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.global_step = state_dict["global_step"]
        self.local_step = 0
        self._whitened = state_dict["whitened"]

    def _apply_schedule(self) -> None:
        """Scale every parameter group's learning rate for the current step."""
        config = self.config
        total = config.total_train_steps
        multiplier = self.schedule(self.global_step, total)
        warmup_steps = int(config.warmup_fraction * total)
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
                fit=self.config.whiten_fit,
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
        return 0.5 * (model(view) + model(view.flip(-1)))

    size = media.shape[-1]
    padded = functional.pad(media, (1,) * 4, "reflect")
    return (
        mirrored(media)
        + mirrored(padded[..., 0:size, 0:size])
        + mirrored(padded[..., 2 : size + 2, 2 : size + 2])
    ) / 3
