"""Training step reproducing the ffcv-imagenet ResNet-50 recipe.

What ffcv-imagenet fixes and this step carries: SGD with Nesterov-free
momentum, weight decay on everything but BatchNorm, label-smoothed
cross-entropy, float16 autocast with loss scaling, a cyclic learning rate
interpolated per iteration between per-epoch knots, progressive resizing of
the training images, and flip test-time augmentation at a fixed test
resolution.

The optimizer and the schedule are injected slots; the recipe's values are
their defaults, so a fork swaps either without touching this file.

References:
  https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import TYPE_CHECKING, cast, override

from configgle import Makeable, Makes, PartialConfig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.imagenet.model import TorchvisionResNet
from priml.math.custom_types import TensorFn
from priml.math.schedules import Schedule, cyclic
from priml.optimizers import (
    CompositeOptimizer,
    apply_lr_scale,
    complement,
    matching,
)
from priml.train.train_step import TrainStep


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.train.custom_types import TrainStepOutput


def sgd_no_bn_decay() -> CompositeOptimizer.Config:
    """Return ffcv-imagenet's SGD: momentum 0.9, decay off the BatchNorm weights.

    The rate is 1 because the schedule is applied as a multiplier on
    ``initial_lr``; ffcv-imagenet likewise builds the optimizer at ``lr=1``
    and writes the scheduled rate in every step.

    Returns:
      config: A two-group composite, BatchNorm parameters first.

    """
    bn = PartialConfig(torch.optim.SGD)
    bn.lr = 1.0
    bn.momentum = 0.9
    bn.weight_decay = 0.0
    rest = PartialConfig(torch.optim.SGD)
    rest.lr = 1.0
    rest.momentum = 0.9
    rest.weight_decay = 1e-4
    cfg = CompositeOptimizer.Config()
    on_bn = matching("bn")
    cfg.select = [on_bn, complement(on_bn)]
    cfg.optimizers = [bn, rest]
    return cfg


class ImageNetTrainStep(TrainStep):
    """Model plus optimization for one ImageNet experiment."""

    class Config(Makes["ImageNetTrainStep"], TrainStep.Config, kw_only=True):
        """Model, optimization, schedule, and resolution policy for one run."""

        model: Makeable[nn.Module] = field(default_factory=TorchvisionResNet.Config)
        """Network to train."""

        optimizer: Makeable[Callable[..., torch.optim.Optimizer]] = field(
            default_factory=sgd_no_bn_decay,
        )
        """Builds the optimizer from the model."""

        dtype_autocast: torch.dtype | None = torch.float16
        """ffcv-imagenet trains under float16 autocast with loss scaling."""

        schedule: Makeable[Schedule[float]] = field(
            default_factory=lambda: PartialConfig(cyclic, peak=2 / 16),
        )
        """Maps progress in ``[0, 1]`` to a learning-rate multiplier.

        The default is ffcv-imagenet's cyclic ramp: ``1e-4`` at start, 1 at
        ``lr_peak_epoch / epochs``, 0 at the end. ``peak`` is that ratio."""

        learning_rate: float = 0.5
        """Peak rate the multiplier scales; ffcv-imagenet's ``lr.lr``."""

        label_smoothing: float = 0.1
        """Cross-entropy label smoothing."""

        resolution_min: int = 160
        """Training resolution until ``resize_start`` of the run."""

        resolution_max: int = 192
        """Training resolution from ``resize_end`` of the run."""

        resize_start: float = 11 / 16
        """Progress at which the resolution begins ramping."""

        resize_end: float = 13 / 16
        """Progress at which the resolution reaches ``resolution_max``."""

        use_tta: bool = True
        """Average logits over the image and its mirror at evaluation."""

        loss_scale_init: float = 2.0**16
        """Initial ``GradScaler`` scale; torch's default."""

        mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
        """Per-channel mean a uint8 batch is normalized with, on the device."""

        std: tuple[float, float, float] = (0.229, 0.224, 0.225)
        """Per-channel std a uint8 batch is normalized with, on the device."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.config: ImageNetTrainStep.Config = config
        self.schedule: Schedule[float] = config.schedule.make()
        # The peak rate lives on the step, so ``initial_lr`` is written here
        # rather than through every optimizer group's own ``lr``.
        for group in self.optimizer.param_groups:
            group["initial_lr"] = config.learning_rate
        self.model.to(memory_format=torch.channels_last)
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            init_scale=config.loss_scale_init,
            enabled=config.dtype_autocast == torch.float16,
        )

    @override
    def train_step(self, **batch: object) -> TrainStepOutput:
        """Resize, forward, scaled backward, and step the optimizer.

        Args:
          **batch: Preprocessed batch with ``image`` (B, 3, 1, H, W) and
            ``label`` (B,).

        Returns:
          result: ``loss`` (per-example) and ``model`` (logits).

        """
        media = self._media(batch)
        label = batch["label"]
        assert isinstance(label, Tensor)
        media = self._resize(media, self.resolution)

        self.model.train()
        with self._autocast():
            logits = self._classifier(media)
            loss = self._loss(logits, label)
            # The fused mean differs from ``loss.mean()`` in the last bits under
            # label smoothing; ffcv-imagenet differentiates the fused one.
            objective = functional.cross_entropy(
                logits,
                label,
                label_smoothing=self.config.label_smoothing,
            )
        self.scaler.scale(objective).backward()
        with self.timer_step:
            apply_lr_scale(
                [self.optimizer],
                self.schedule(self.progress_learning_schedule),
            )
            assert isinstance(self.optimizer, torch.optim.Optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.model.zero_grad(set_to_none=True)
        return {"loss": loss.detach(), "model": logits.detach()}

    @override
    def eval_loss(self, **batch: object) -> TrainStepOutput:
        """Compute the evaluation loss and logits."""
        label = batch["label"]
        assert isinstance(label, Tensor)
        logits = self.call_eval(image=batch["image"])
        assert isinstance(logits, Tensor)
        return {"loss": self._loss(logits, label), "model": logits}

    @override
    def call_eval(self, *args: object, **batch: object) -> object:
        """Return evaluation logits, summed with the mirrored view under TTA."""
        if args:
            raise ValueError("Expected not args.")
        media = self._media(batch)
        self.model.eval()
        with torch.inference_mode(), self._autocast():
            logits = self._classifier(media)
            if self.config.use_tta:
                # Summed, not averaged, as in the reference: argmax is unchanged
                # and the reported loss is theirs.
                logits = logits + self._classifier(media.flip(-1))
            return logits

    @property
    def resolution(self) -> int:
        """Training side length at the current progress, a multiple of 32."""
        config = self.config
        low = int(config.resolution_min)
        high = int(config.resolution_max)
        start = float(config.resize_start)
        end = float(config.resize_end)
        spent = self.progress_learning_schedule
        if spent <= start:
            return low
        if spent >= end:
            return high
        interp = low + (spent - start) / (end - start) * (high - low)
        return round(interp / 32) * 32

    @override
    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        """Move the batch; normalize a uint8 image on the device, as ffcv does.

        A uint8 ``(B, 3, H, W)`` batch crosses the bus at a quarter the bytes
        of float32 and is normalized here. A float batch arrives normalized
        with the pipeline's unit frame axis ``(B, 3, 1, H, W)``.
        """
        moved = super().preprocess_batch(batch)
        image = moved["image"]
        assert isinstance(image, Tensor)
        if image.dtype == torch.uint8:
            mean = torch.tensor(self.config.mean, device=image.device) * 255
            std = torch.tensor(self.config.std, device=image.device) * 255
            image = (image.float() - mean.view(-1, 1, 1)) / std.view(-1, 1, 1)
        else:
            image = image.squeeze(2)
        moved["image"] = image.contiguous(memory_format=torch.channels_last)
        return moved

    class StateDict(TrainStep.StateDict):
        """Base state plus the loss scaler."""

        scaler: dict[str, object]

    @override
    def state_dict(self) -> StateDict:
        """Extend the base state with the loss scaler."""
        return {**super().state_dict(), "scaler": self.scaler.state_dict()}

    @override
    def load_state_dict(
        self,
        state_dict: Mapping[str, object],
        *,
        strict: bool = True,
        load_optimizer: bool = True,
        remap: Callable[[Mapping[str, Tensor]], Mapping[str, Tensor]] | None = None,
    ) -> None:
        """Restore state produced by :meth:`state_dict`."""
        state = cast(ImageNetTrainStep.StateDict, state_dict)
        super().load_state_dict(
            state,
            strict=strict,
            load_optimizer=load_optimizer,
            remap=remap,
        )
        if load_optimizer:
            self.scaler.load_state_dict(state["scaler"])

    def _media(self, batch: dict[str, object]) -> Tensor:
        media = batch["image"]
        assert isinstance(media, Tensor)
        return media

    def _resize(self, media: Tensor, resolution: int) -> Tensor:
        """Resize to the epoch's resolution; a no-op when already there."""
        if media.shape[-1] == resolution and media.shape[-2] == resolution:
            return media
        return functional.interpolate(
            media,
            size=(resolution, resolution),
            mode="area",
        )

    def _loss(self, logits: Tensor, label: Tensor) -> Tensor:
        """Return per-example smoothed cross-entropy."""
        return functional.cross_entropy(
            logits.float(),
            label,
            label_smoothing=self.config.label_smoothing,
            reduction="none",
        )

    def _autocast(self) -> torch.autocast:
        return torch.autocast(
            self.device.type,
            dtype=self.config.dtype_autocast,
            enabled=self.config.dtype_autocast is not None,
        )

    @property
    def _classifier(self) -> TensorFn:
        """The model, typed as a batch of images to logits."""
        return cast(TensorFn, self.model)
