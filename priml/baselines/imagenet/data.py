"""ImageNet data: the ffcv-imagenet input pipelines, and the DeiT ones.

:class:`ImageNetData` defaults to the ffcv-imagenet recipe: random-resized
crop to the epoch's training resolution and a flip, decoded straight into a
uint8 batch; centre crop at 224/256 of the short side, resized to 256 for
evaluation. Normalization runs on the device, in the train step. The DeiT
augmentation stack (colour jitter, RandAugment, erasing, MixUp) is the pair of
pipelines a fork injects instead -- ``exp000`` carries none of it.

References:
  https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py
  https://arxiv.org/abs/2012.12877
    Touvron et al. 2021. Training data-efficient image transformers.

"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, TypedDict, cast, override

from configgle import Fig, Makeable

from priml.custom_types import HasNormalizedWorkingDirPattern
from priml.data.pipeline.batching import Batcher
from priml.data.pipeline.dataset import DataPipeline
from priml.data.pipeline.parallel import PrefetchBuffer
from priml.data.pipeline.tensorize import AsTensor
from priml.data.processors.augmentation import (
    ColorJitter,
    GetCenterCropBoxFromDimensions,
    GetRandomResizedCropBoxFromDimensions,
    MixupCutmix,
    Normalize,
    RandAugment,
    RandomErasing,
    RandomHorizontalFlip,
)
from priml.data.processors.bytes import (
    CropDuringDecodeImage,
    GetBytesFromFile,
    GetDimensionsFromBytes,
)
from priml.data.processors.decode_batch import DecodeCropResizeBatch
from priml.data.processors.fields import FieldRenameKeys
from priml.data.processors.labels import ImagenetSynsetToIndex
from priml.data.processors.resize import Interpolate
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource
from priml.paths import resolve_working_dir
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch.utils.data import DataLoader

    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")  # ~1050 ms; only the dtype defaults need it.


NUM_TRAIN_SAMPLES: Final = 1_281_167
NUM_CLASSES: Final = 1_000


def ffcv_train_data_pipeline() -> DataPipeline.Config:
    """Return the ffcv-imagenet training pipeline: crop, flip, uint8 out.

    Samples carry only bytes and a crop box until ``Batcher``; the batch is
    then decoded, cropped, resized, and flipped into one uint8 tensor across
    threads. Normalization runs on the device in the train step, as in ffcv.

    Returns:
      config: Pipeline yielding ``image`` (B, 3, res, res) uint8 and
        ``label`` (B,) int64 batches.

    """
    cfg = DataPipeline.Config()
    source = ExtractedImageNetSource.Config()
    source.split = "train"
    source.shuffle = True
    cfg.source = source
    crop = GetRandomResizedCropBoxFromDimensions.Config()
    crop.size = (160, 160)
    batcher = Batcher.Config()
    batcher.size = 512
    batcher.field_names = ["media", "crop", "target_height", "target_width", "label"]
    batcher.drop_remainder = True
    decode = DecodeCropResizeBatch.Config()
    decode.flip_p = 0.5
    # ffcv's fast IDCT: its pixels, bit-for-bit, and faster than the accurate one.
    decode.fast_dct = True
    cfg.processors.extend(
        [
            GetBytesFromFile.Config(),
            GetDimensionsFromBytes.Config(),
            ImagenetSynsetToIndex.Config(),
            crop,
            FieldRenameKeys.Config(
                {
                    "media": "media",
                    "crop": "crop",
                    "target_height": "target_height",
                    "target_width": "target_width",
                    "label": "label",
                    "*": None,
                },
            ),
            batcher,
            decode,
            AsTensor.Config(include=["label"], dtype=torch.int64),
            PrefetchBuffer.Config(size=2),
        ],
    )
    return cfg


def ffcv_eval_data_pipeline() -> DataPipeline.Config:
    """Return the ffcv-imagenet validation pipeline: centre crop, uint8 out.

    Returns:
      config: Pipeline yielding ``image`` (B, 3, 256, 256) uint8 and
        ``label`` (B,) int64 batches; the short batch is kept.

    """
    cfg = DataPipeline.Config()
    source = ExtractedImageNetSource.Config()
    source.split = "val"
    source.validation_labels_file = Path("validation_labels.txt")
    cfg.source = source
    crop = GetCenterCropBoxFromDimensions.Config()
    crop.size = (256, 256)
    # ffcv's DEFAULT_CROP_RATIO: 224/256 of the short side, resized to 256 (FixRes).
    crop.ratio = 224 / 256
    batcher = Batcher.Config()
    batcher.size = 512
    batcher.field_names = ["media", "crop", "target_height", "target_width", "label"]
    decode = DecodeCropResizeBatch.Config()
    decode.fast_dct = True
    cfg.processors.extend(
        [
            ImagenetSynsetToIndex.Config(),
            GetBytesFromFile.Config(),
            GetDimensionsFromBytes.Config(),
            crop,
            FieldRenameKeys.Config(
                {
                    "media": "media",
                    "crop": "crop",
                    "target_height": "target_height",
                    "target_width": "target_width",
                    "label": "label",
                    "*": None,
                },
            ),
            batcher,
            decode,
            AsTensor.Config(include=["label"], dtype=torch.int64),
            PrefetchBuffer.Config(size=4),
        ],
    )
    return cfg


def deit_train_data_pipeline() -> DataPipeline.Config:
    """Return the DeiT/timm training pipeline, augmentations on uint8.

    DeiT's order is RandomResizedCrop(224) -> flip -> ColorJitter(0.4) ->
    RandAugment(2, 9) -> Normalize -> RandomErasing(0.25) -> MixUp/CutMix.
    Here every augmentation runs on uint8 before ``Normalize`` (2-3x faster),
    and ``Interpolate`` to the target size comes last.

    Returns:
      config: Pipeline yielding ``image`` (B, 3, 1, 224, 224) bfloat16 and
        mixed ``label`` (B, 1000) float32 batches.

    """
    cfg = DataPipeline.Config()
    source = ExtractedImageNetSource.Config()
    source.split = "train"
    source.shuffle = True
    cfg.source = source
    cfg.processors.extend(
        [
            GetBytesFromFile.Config(),
            GetDimensionsFromBytes.Config(),
            ImagenetSynsetToIndex.Config(),
            GetRandomResizedCropBoxFromDimensions.Config(),
            CropDuringDecodeImage.Config(),
            RandomHorizontalFlip.Config(),
            ColorJitter.Config(),
            RandAugment.Config(),
            RandomErasing.Config(),
            Normalize.Config(dtype=torch.bfloat16),
            Interpolate.Config(),
            FieldRenameKeys.Config(
                {
                    "media_tensor": "image",
                    "label": "label",
                    "*": None,
                },
            ),
            Batcher.Config(
                size=512,
                field_names=["image", "label"],
                drop_remainder=True,
            ),
            AsTensor.Config(include=["label"]),
            MixupCutmix.Config(),
            PrefetchBuffer.Config(size=2),
        ],
    )
    return cfg


def deit_eval_data_pipeline() -> DataPipeline.Config:
    """Return the DeiT validation pipeline: centre crop to 224, normalized.

    Returns:
      config: Pipeline yielding ``image`` (B, 3, 1, 224, 224) bfloat16 and
        ``label`` (B,) int64 batches.

    """
    cfg = DataPipeline.Config()
    source = ExtractedImageNetSource.Config()
    source.split = "val"
    source.validation_labels_file = Path("validation_labels.txt")
    cfg.source = source
    cfg.processors.extend(
        [
            ImagenetSynsetToIndex.Config(),
            GetBytesFromFile.Config(),
            GetDimensionsFromBytes.Config(),
            GetCenterCropBoxFromDimensions.Config(),
            CropDuringDecodeImage.Config(),
            Normalize.Config(dtype=torch.bfloat16),
            Interpolate.Config(),
            FieldRenameKeys.Config(
                {
                    "media_tensor": "image",
                    "label": "label",
                    "*": None,
                },
            ),
            Batcher.Config(
                size=512,
                field_names=["image", "label"],
                drop_remainder=True,
            ),
            AsTensor.Config(include=["label"], dtype=torch.int64),
            PrefetchBuffer.Config(size=4),
        ],
    )
    return cfg


class ImageNetData:
    """ImageNet as separate train and eval pipelines over the extracted archive.

    Implements ``DatasetProtocol`` for ``TrainLoop``.
    """

    class Config(Fig["ImageNetData"]):
        train_data_pipeline: Makeable[DataPipeline] = field(
            default_factory=ffcv_train_data_pipeline,
        )
        """Source and processors producing training batches."""

        eval_data_pipeline: Makeable[DataPipeline] = field(
            default_factory=ffcv_eval_data_pipeline,
        )
        """Source and processors producing evaluation batches."""

        num_workers: int = 0
        """DataLoader worker processes per loader. ``DecodeCropResizeBatch``
        decodes on its own threads, so a worker process only adds a batch copy
        across the boundary; the per-sample DeiT pipelines want several."""

        prefetch_factor: int = 2
        """Batches prefetched by each DataLoader worker."""

        base_dir: Path | str | None = None
        """Resource root supplied by the trainer."""

        working_dir: Path | str = "/"
        """Transparent scope: the source owns the ``/datasets/imagenet`` root."""

        @override
        def finalize(self) -> Self:
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            for pipeline in (self.train_data_pipeline, self.eval_data_pipeline):
                if (
                    isinstance(pipeline, HasNormalizedWorkingDirPattern)
                    and pipeline.base_dir is None
                ):
                    pipeline.base_dir = self.working_dir
            return super().finalize()

    def __init__(self, config: Config) -> None:
        self.num_workers = config.num_workers
        self.prefetch_factor = config.prefetch_factor
        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the data; ticked by the loop when the loader runs out."""

        self.train_pipe = config.train_data_pipeline.make()
        self.eval_pipe = config.eval_data_pipeline.make()

    def train_dataloader(self) -> DataLoader[dict[str, object]]:
        """Build the training dataloader."""
        return self.train_pipe.create_loader(
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def eval_dataloader(self) -> DataLoader[dict[str, object]]:
        """Build the evaluation dataloader."""
        return self.eval_pipe.create_loader(
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    class StateDict(TypedDict):
        """The pass count; the pipelines carry no resumable position."""

        timer_epoch: CheckpointableStepTimer.StateDict

    def state_dict(self) -> StateDict:
        """Return the pass count."""
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the pass count.

        Args:
          state_dict: Payload produced by :meth:`state_dict`.

        """
        state = cast(ImageNetData.StateDict, state_dict)
        self.timer_epoch.load_state_dict(state["timer_epoch"])
