"""Image augmentation processors for training.

``media_tensor`` arrives in decoder layout ``(C, F, H, W)``; torchvision's
transforms take ``(..., C, H, W)``, so the per-sample transforms here move the
frame axis in front of the channels for the call and back after.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import math
import random

from configgle import Fig
from torch import Tensor

import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from torchvision.transforms import v2
else:
    from wrapt import lazy_import

    v2 = lazy_import("torchvision.transforms.v2")  # ~400 ms; only 3 stages use it.


__all__ = [
    "ColorJitter",
    "GetCenterCropBoxFromDimensions",
    "GetRandomResizedCropBoxFromDimensions",
    "MixupCutmix",
    "Normalize",
    "RandAugment",
    "RandomErasing",
    "RandomHorizontalFlip",
]


class GetRandomResizedCropBoxFromDimensions:
    """Compute random crop box for RandomResizedCrop.

    Sets crop box (y, x, h, w) that will be used by CropDuringDecodeImage.

    Thread Safety:
        Uses Python's global RNG (random.uniform, random.randint). When used
        in ParMap with num_threads > 1, RNG calls from multiple threads will
        interleave. This is deterministic within a worker process but may vary
        across runs due to thread scheduling. For strict reproducibility, use
        num_threads=1.
    """

    class Config(Fig["GetRandomResizedCropBoxFromDimensions"]):
        size: tuple[int, int] = (224, 224)
        """Target (height, width) the crop is later resized to."""

        scale: tuple[float, float] = (0.08, 1.0)
        """Crop area as a fraction of the source, drawn uniformly."""

        ratio: tuple[float, float] = (3 / 4, 4 / 3)
        """Aspect-ratio bounds, drawn log-uniformly within them."""

    class Input(TypedDict, total=False):
        height: int
        width: int
        crop: tuple[int, int, int, int]
        target_height: int
        target_width: int

    class Output(TypedDict, total=False):
        height: int
        width: int
        crop: tuple[int, int, int, int]
        target_height: int
        target_width: int

    def __init__(self, config: Config):
        self.size = config.size
        self.scale = config.scale
        self.ratio = config.ratio

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Apply to the input."""
        for sample_in in samples:
            sample = sample_in
            height = sample.get("height")
            width = sample.get("width")

            if height is None or width is None:
                yield sample
                continue

            # Compute random crop parameters (adapted from torchvision)
            area = height * width
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))

            for _ in range(10):
                target_area = area * random.uniform(self.scale[0], self.scale[1])  # noqa: S311 -- These draws randomize augmentation, not security-sensitive behavior.
                aspect_ratio = math.exp(random.uniform(log_ratio[0], log_ratio[1]))  # noqa: S311 -- These draws randomize augmentation, not security-sensitive behavior.

                w = round((target_area * aspect_ratio) ** 0.5)
                h = round((target_area / aspect_ratio) ** 0.5)

                if 0 < w <= width and 0 < h <= height:
                    x = random.randint(0, width - w)  # noqa: S311 -- These draws randomize augmentation, not security-sensitive behavior.
                    y = random.randint(0, height - h)  # noqa: S311 -- These draws randomize augmentation, not security-sensitive behavior.
                    sample["crop"] = (y, x, h, w)
                    sample["target_height"] = self.size[0]
                    sample["target_width"] = self.size[1]
                    yield sample
                    break
            else:
                min_ratio, max_ratio = self.ratio
                w = round(min(width, height * max_ratio))
                h = round(min(height, width / min_ratio))
                x = (width - w) // 2
                y = (height - h) // 2
                sample["crop"] = (y, x, h, w)
                sample["target_height"] = self.size[0]
                sample["target_width"] = self.size[1]
                yield sample


class GetCenterCropBoxFromDimensions:
    """Compute center crop box.

    Sets crop box (y, x, h, w) that will be used by CropDuringDecodeImage.
    """

    class Config(Fig["GetCenterCropBoxFromDimensions"]):
        size: tuple[int, int] = (224, 224)
        """Target (height, width) the crop is later resized to."""

        ratio: float = 1.0
        """Fraction of the largest centered box at ``size``'s aspect that is
        kept; ``224 / 256`` is the resize-256-then-crop-224 evaluation
        convention. In ``(0, 1]``."""

    class Input(TypedDict, total=False):
        height: int
        width: int
        crop: tuple[int, int, int, int]
        target_height: int
        target_width: int

    class Output(TypedDict, total=False):
        height: int
        width: int
        crop: tuple[int, int, int, int]
        target_height: int
        target_width: int

    def __init__(self, config: Config):
        # A ratio above 1 asked for a box larger than the image, at negative
        # offsets.
        if math.isnan(config.ratio) or config.ratio <= 0 or config.ratio > 1:
            raise ValueError(f"ratio must be in (0, 1]; got {config.ratio}.")
        self.size = config.size
        self.ratio = config.ratio

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Apply to the input."""
        target_h, target_w = self.size
        for sample_in in samples:
            sample = sample_in
            height = sample.get("height")
            width = sample.get("width")

            if height is None or width is None:
                yield sample
                continue

            # The largest box at the target's aspect, so the later resize to
            # ``size`` does not stretch it.
            fit = min(height / target_h, width / target_w)
            crop_h = int(self.ratio * fit * target_h)
            crop_w = int(self.ratio * fit * target_w)
            x = (width - crop_w) // 2
            y = (height - crop_h) // 2

            sample["crop"] = (y, x, crop_h, crop_w)
            sample["target_height"] = self.size[0]
            sample["target_width"] = self.size[1]
            yield sample


class RandomHorizontalFlip:
    """Random horizontal flip augmentation.

    Thread Safety:
        Uses Python's global RNG (random.random). When used in ParMap with
        num_threads > 1, RNG calls from multiple threads will interleave.
        For strict reproducibility, use num_threads=0.
    """

    class Config(Fig["RandomHorizontalFlip"]):
        p: float = 0.5
        """Probability a given sample is flipped."""

    class Input(TypedDict, total=False):
        media_tensor: Tensor

    class Output(TypedDict, total=False):
        media_tensor: Tensor

    def __init__(self, config: Config):
        self.p = config.p

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Apply to the input."""
        for sample in samples:
            x = sample.get("media_tensor")
            if x is None:
                yield sample
                continue
            if random.random() < self.p:  # noqa: S311 -- These draws randomize augmentation, not security-sensitive behavior.
                sample["media_tensor"] = torch.flip(x, dims=[-1])
            yield sample


class ColorJitter:
    """Apply color jitter augmentation.

    Thread Safety:
        Uses torchvision transforms with internal RNG. When used in ParMap
        with num_threads > 1, RNG calls from multiple threads will interleave.
        For strict reproducibility, use num_threads=1.
    """

    class Config(Fig["ColorJitter"]):
        brightness: float = 0.4
        """Max brightness shift, as a fraction either way."""

        contrast: float = 0.4
        """Max contrast shift, as a fraction either way."""

        saturation: float = 0.4
        """Max saturation shift, as a fraction either way."""

        hue: float = 0.1
        """Max hue rotation, in turns either way (0.5 is a half circle)."""

    class Input(TypedDict, total=False):
        media_tensor: Tensor

    class Output(TypedDict, total=False):
        media_tensor: Tensor

    def __init__(self, config: Config):
        self.transform = v2.ColorJitter(
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue=config.hue,
        )

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Apply to the input."""
        for sample in samples:
            x = sample.get("media_tensor")
            if x is None:
                yield sample
                continue
            sample["media_tensor"] = _per_frame(self.transform, x)
            yield sample


class RandomErasing:
    """Random erasing augmentation (cutout).

    Thread Safety:
        Uses torchvision transforms with internal RNG. When used in ParMap
        with num_threads > 1, RNG calls from multiple threads will interleave.
        For strict reproducibility, use num_threads=1.
    """

    class Config(Fig["RandomErasing"]):
        p: float = 0.25
        """Probability a given sample has a region erased."""

        scale: tuple[float, float] = (0.02, 0.33)
        """Erased area as a fraction of the image."""

        ratio: tuple[float, float] = (0.3, 3.3)
        """Aspect-ratio bounds of the erased region."""

        value: float = 0.0
        """Fill written into the erased region."""

    class Input(TypedDict, total=False):
        media_tensor: Tensor

    class Output(TypedDict, total=False):
        media_tensor: Tensor

    def __init__(self, config: Config):
        self.transform = v2.RandomErasing(
            p=config.p,
            scale=config.scale,
            ratio=config.ratio,
            value=config.value,
        )

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Apply to the input."""
        for sample in samples:
            x = sample.get("media_tensor")
            if x is None:
                yield sample
                continue
            sample["media_tensor"] = _per_frame(self.transform, x)
            yield sample


class RandAugment:
    """RandAugment, as torchvision's ``v2.RandAugment`` implements it.

    The ops, magnitude table, random sign of the geometric ops, and the
    shear-as-matrix-coefficient convention are torchvision's, which follow the
    paper's reference code. timm's ``rand-m9`` divides by a fixed 10 levels;
    torchvision spaces ``num_magnitude_bins`` values from zero, so 11 bins make
    ``magnitude`` a timm level. timm's ``mstd`` noise and its extra
    SolarizeAdd op are not reproduced.

    Reference: https://arxiv.org/abs/1909.13719

    Thread Safety:
        Uses PyTorch's global RNG. When used in ParMap with num_threads > 1,
        RNG calls from multiple threads will interleave. For strict
        reproducibility, use num_threads=0.
    """

    class Config(Fig["RandAugment"]):
        num_ops: int = 2
        """Ops drawn (with replacement) and applied in sequence per sample."""

        magnitude: int = 9
        """Strength index into ``num_magnitude_bins``: 0 is the identity,
        ``num_magnitude_bins - 1`` the strongest."""

        num_magnitude_bins: int = 11
        """Discretization of the strength range; ``magnitude`` indexes it."""

    class Input(TypedDict, total=False):
        media_tensor: Tensor
        """uint8, decoder layout ``(C, F, H, W)``."""

    class Output(TypedDict, total=False):
        media_tensor: Tensor

    def __init__(self, config: Config):
        if config.magnitude < 0 or config.magnitude >= config.num_magnitude_bins:
            raise ValueError(
                f"magnitude must be in [0, {config.num_magnitude_bins}); "
                f"got {config.magnitude}.",
            )
        self.transform = v2.RandAugment(
            num_ops=config.num_ops,
            magnitude=config.magnitude,
            num_magnitude_bins=config.num_magnitude_bins,
        )

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Apply to the input."""
        for sample in samples:
            x = sample.get("media_tensor")
            if x is None:
                yield sample
                continue
            # Its solarize and posterize are uint8 ops; a float image is
            # rejected here rather than transformed on the wrong scale.
            if x.dtype != torch.uint8:
                raise TypeError(f"RandAugment expects a uint8 tensor, got {x.dtype}")
            sample["media_tensor"] = _per_frame(self.transform, x)
            yield sample


class Normalize:
    """Normalize a uint8 tensor with mean and std (ImageNet defaults)."""

    class Config(Fig["Normalize"]):
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
        """Normalization mean.
        ImageNet: (0.485, 0.456, 0.406)
        CLIP: (0.48145466, 0.4578275, 0.40821073)"""

        std: tuple[float, float, float] = (0.229, 0.224, 0.225)
        """Normalization std.
        ImageNet: (0.229, 0.224, 0.225)
        CLIP: (0.26862954, 0.26130258, 0.27577711)"""

        device: torch.device | str = "cpu"
        """Device the cached scale/shift tensors live on."""

        dtype: torch.dtype = torch.float32
        """Width the normalized output is produced in."""

    class Input(TypedDict, total=False):
        media_tensor: Tensor

    class Output(TypedDict, total=False):
        media_tensor: Tensor

    def __init__(self, config: Config):
        # Precompute for fused addcmul: (x - mean*255) / (std*255)
        # = x * (1/(std*255)) + (-mean/std)
        # = torch.addcmul(-mean/std, x, 1/(std*255))
        bias = tuple(-config.mean[i] / config.std[i] for i in range(3))
        scale = tuple(1.0 / (s * 255.0) for s in config.std)
        self.bias = torch.tensor(bias, device=config.device, dtype=config.dtype).view(
            -1,
            1,
            1,
            1,
        )
        self.scale = torch.tensor(scale, device=config.device, dtype=config.dtype).view(
            -1,
            1,
            1,
            1,
        )
        self.device = config.device
        self.dtype = config.dtype

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Apply to the input."""
        for sample in samples:
            x = sample.get("media_tensor")
            if x is None:
                yield sample
                continue
            if x.dtype != torch.uint8:
                raise TypeError(f"Normalize expects uint8 tensor, got {x.dtype}")
            x = x.to(device=self.device, dtype=self.dtype)
            sample["media_tensor"] = x.mul_(self.scale).add_(self.bias)
            yield sample


class MixupCutmix:
    """Mixup and CutMix augmentation for batched samples."""

    class Config(Fig["MixupCutmix"]):
        mixup_alpha: float = 0.8
        """Beta concentration for the mixup blend; 0 disables mixup."""

        cutmix_alpha: float = 1.0
        """Beta concentration for the cutmix box; 0 disables cutmix."""

        prob: float = 1.0
        """Probability a batch is mixed at all."""

        switch_prob: float = 0.5
        """Given a batch is mixed, the chance of cutmix over mixup."""

        label_smoothing: float = 0.1
        """Mass moved off the true class before mixing."""

        num_classes: int = 1000
        """Label range; sets the width of the one-hot targets."""

        field: str = "media_tensor"
        """Batched image field mixed in place: ``(B, ...)`` with H, W last."""

    Input = dict[str, object]
    """A batch: the image ``field`` and ``label`` -- a ``(B,)`` tensor, or the
    list of ints a ``Batcher`` stacks from scalar labels."""

    Output = dict[str, object]

    def __init__(self, config: Config):
        self.mixup_alpha = config.mixup_alpha
        self.cutmix_alpha = config.cutmix_alpha
        self.prob = config.prob
        self.switch_prob = config.switch_prob
        self.label_smoothing = config.label_smoothing
        self.num_classes = config.num_classes
        self.field = config.field

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Apply to the input."""
        for sample in samples:
            images = sample.get(self.field)
            raw_labels = sample.get("label")

            if not isinstance(images, Tensor) or raw_labels is None:
                yield sample
                continue
            # Tensorized once, up front: the mixing paths index the labels
            # with a permutation tensor, which a Batcher-stacked list refuses.
            labels = torch.as_tensor(raw_labels, dtype=torch.long)

            if random.random() > self.prob:  # noqa: S311 -- These draws randomize augmentation, not security-sensitive behavior.
                # No mixing, just label smoothing.
                labels_onehot = _one_hot_with_smoothing(
                    labels,
                    self.num_classes,
                    self.label_smoothing,
                )
                sample["label"] = labels_onehot
                yield sample
                continue

            batch_size = images.size(0)
            if batch_size < 2:
                # Can't mix single sample.
                labels_onehot = _one_hot_with_smoothing(
                    labels,
                    self.num_classes,
                    self.label_smoothing,
                )
                sample["label"] = labels_onehot
                yield sample
                continue

            # Choose mixup or cutmix.
            use_cutmix = random.random() < self.switch_prob  # noqa: S311 -- These draws randomize augmentation, not security-sensitive behavior.

            if use_cutmix and self.cutmix_alpha > 0:
                # CutMix.
                lam = random.betavariate(self.cutmix_alpha, self.cutmix_alpha)
                rand_index = torch.randperm(batch_size, device=images.device)

                # Generate random box (handle both (B,C,H,W) and (B,C,F,H,W))
                *_, h, w = images.shape
                cut_ratio = (1.0 - lam) ** 0.5
                cut_h = int(h * cut_ratio)
                cut_w = int(w * cut_ratio)
                cx = random.randint(0, w)  # noqa: S311 -- These draws randomize augmentation, not security-sensitive behavior.
                cy = random.randint(0, h)  # noqa: S311 -- These draws randomize augmentation, not security-sensitive behavior.
                bbx1 = max(cx - cut_w // 2, 0)
                bby1 = max(cy - cut_h // 2, 0)
                bbx2 = min(cx + cut_w // 2, w)
                bby2 = min(cy + cut_h // 2, h)

                # Apply cutmix.
                images[..., bby1:bby2, bbx1:bbx2] = images[
                    rand_index,
                    ...,
                    bby1:bby2,
                    bbx1:bbx2,
                ]

                # Adjust lambda based on actual box size.
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))

                # Mix labels.
                labels_a = _one_hot_with_smoothing(
                    labels,
                    self.num_classes,
                    self.label_smoothing,
                )
                labels_b = _one_hot_with_smoothing(
                    labels[rand_index],
                    self.num_classes,
                    self.label_smoothing,
                )
                mixed_labels = lam * labels_a + (1 - lam) * labels_b

            elif self.mixup_alpha > 0:
                # Mixup.
                lam = random.betavariate(self.mixup_alpha, self.mixup_alpha)
                rand_index = torch.randperm(batch_size, device=images.device)

                # Mix images.
                images = lam * images + (1 - lam) * images[rand_index]

                # Mix labels.
                labels_a = _one_hot_with_smoothing(
                    labels,
                    self.num_classes,
                    self.label_smoothing,
                )
                labels_b = _one_hot_with_smoothing(
                    labels[rand_index],
                    self.num_classes,
                    self.label_smoothing,
                )
                mixed_labels = lam * labels_a + (1 - lam) * labels_b
            else:
                # No mixing, just label smoothing.
                mixed_labels = _one_hot_with_smoothing(
                    labels,
                    self.num_classes,
                    self.label_smoothing,
                )

            sample[self.field] = images
            sample["label"] = mixed_labels
            yield sample


# Torchvision reads ``(..., C, H, W)``; handed decoder layout it took the frame axis for
# the channels, so saturation and hue no-oped on a one-frame image and raised on a
# multi-frame clip.
def _per_frame(transform: Callable[[Tensor], Tensor], x: Tensor) -> Tensor:
    """Apply an image transform to each frame of a ``(C, F, H, W)`` tensor."""
    return transform(x.moveaxis(-4, -3)).moveaxis(-3, -4)


def _one_hot_with_smoothing(
    labels: Tensor | list[int],
    num_classes: int,
    smoothing: float,
) -> Tensor:
    """Convert labels to one-hot with label smoothing."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    batch_size = labels.size(0)
    one_hot = torch.zeros(
        batch_size,
        num_classes,
        device=labels.device,
        dtype=torch.float32,
    )
    one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

    return one_hot * (1 - smoothing) + smoothing / num_classes
