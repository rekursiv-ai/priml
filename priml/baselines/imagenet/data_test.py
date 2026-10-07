"""Tests for the ImageNet dataset and its pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import io

from configgle import Fig
from PIL import Image
from torch import Tensor

import pytest
import torch

from priml.baselines.imagenet.data import (
    ImageNetData,
    deit_eval_data_pipeline,
    deit_train_data_pipeline,
    ffcv_eval_data_pipeline,
    ffcv_train_data_pipeline,
)
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
from priml.lib.codec import from_plain


if TYPE_CHECKING:
    from collections.abc import Iterator


def test_working_dir_propagates_through_imagenet_pipelines(tmp_path: Path) -> None:
    config = ImageNetData.Config()
    config.base_dir = tmp_path

    finalized = config.copy_tree().finalize()

    # The dataset and pipeline are transparent scopes; the source owns the
    # ``/datasets/imagenet`` root, so it resolves beneath the injected base_dir.
    assert finalized.working_dir == tmp_path
    for pipeline in (finalized.train_data_pipeline, finalized.eval_data_pipeline):
        assert isinstance(pipeline, DataPipeline.Config)
        assert pipeline.working_dir == tmp_path
        assert isinstance(pipeline.source, ExtractedImageNetSource.Config)
        assert pipeline.source.working_dir == tmp_path / "datasets" / "imagenet"


def _fake_jpeg() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (256, 255), color="red").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_ffcv_pipeline_configs_pin_order_and_reference_parameters() -> None:
    """The ffcv pipelines preserve source, crop, batch and decode contracts."""
    train_config = ffcv_train_data_pipeline()
    eval_config = ffcv_eval_data_pipeline()
    assert isinstance(train_config.source, ExtractedImageNetSource.Config)
    assert train_config.source.split == "train"
    assert train_config.source.shuffle is True
    assert isinstance(eval_config.source, ExtractedImageNetSource.Config)
    assert eval_config.source.split == "val"
    assert eval_config.source.validation_labels_file == Path("validation_labels.txt")
    assert [type(processor) for processor in train_config.processors] == [
        GetBytesFromFile.Config,
        GetDimensionsFromBytes.Config,
        ImagenetSynsetToIndex.Config,
        GetRandomResizedCropBoxFromDimensions.Config,
        FieldRenameKeys.Config,
        Batcher.Config,
        DecodeCropResizeBatch.Config,
        AsTensor.Config,
        PrefetchBuffer.Config,
    ]
    assert [type(processor) for processor in eval_config.processors] == [
        ImagenetSynsetToIndex.Config,
        GetBytesFromFile.Config,
        GetDimensionsFromBytes.Config,
        GetCenterCropBoxFromDimensions.Config,
        FieldRenameKeys.Config,
        Batcher.Config,
        DecodeCropResizeBatch.Config,
        AsTensor.Config,
        PrefetchBuffer.Config,
    ]
    train_crop = next(
        processor
        for processor in train_config.processors
        if isinstance(processor, GetRandomResizedCropBoxFromDimensions.Config)
    )
    eval_crop = next(
        processor
        for processor in eval_config.processors
        if isinstance(processor, GetCenterCropBoxFromDimensions.Config)
    )
    train_batcher = next(
        processor
        for processor in train_config.processors
        if isinstance(processor, Batcher.Config)
    )
    eval_batcher = next(
        processor
        for processor in eval_config.processors
        if isinstance(processor, Batcher.Config)
    )
    train_decode = next(
        processor
        for processor in train_config.processors
        if isinstance(processor, DecodeCropResizeBatch.Config)
    )
    eval_decode = next(
        processor
        for processor in eval_config.processors
        if isinstance(processor, DecodeCropResizeBatch.Config)
    )
    assert train_crop.size == (160, 160)
    assert eval_crop.size == (256, 256)
    assert eval_crop.ratio == 224 / 256
    assert train_batcher.size == eval_batcher.size == 512
    assert (
        train_batcher.field_names
        == eval_batcher.field_names
        == [
            "media",
            "crop",
            "target_height",
            "target_width",
            "label",
        ]
    )
    assert train_batcher.drop_remainder is True
    assert eval_batcher.drop_remainder is False
    assert next(
        p for p in train_config.processors if isinstance(p, FieldRenameKeys.Config)
    ).mappings == {
        "media": "media",
        "crop": "crop",
        "target_height": "target_height",
        "target_width": "target_width",
        "label": "label",
        "*": None,
    }
    assert next(
        p for p in eval_config.processors if isinstance(p, FieldRenameKeys.Config)
    ).mappings == {
        "media": "media",
        "crop": "crop",
        "target_height": "target_height",
        "target_width": "target_width",
        "label": "label",
        "*": None,
    }
    train_tensor = next(
        p for p in train_config.processors if isinstance(p, AsTensor.Config)
    )
    eval_tensor = next(
        p for p in eval_config.processors if isinstance(p, AsTensor.Config)
    )
    assert train_tensor.include == eval_tensor.include == ["label"]
    assert train_tensor.dtype == eval_tensor.dtype == torch.int64
    assert train_decode.fast_dct is eval_decode.fast_dct is True
    assert train_decode.flip_p == 0.5
    assert (
        next(
            p for p in train_config.processors if isinstance(p, PrefetchBuffer.Config)
        ).size
        == 2
    )
    assert (
        next(
            p for p in eval_config.processors if isinstance(p, PrefetchBuffer.Config)
        ).size
        == 4
    )


def test_deit_pipeline_configs_pin_augmentation_order_and_outputs() -> None:
    """DeiT augmentation order, precision, batching and label mixing are fixed."""
    train = deit_train_data_pipeline()
    evaluate = deit_eval_data_pipeline()
    assert isinstance(train.source, ExtractedImageNetSource.Config)
    assert train.source.split == "train"
    assert train.source.shuffle is True
    assert [type(processor) for processor in train.processors] == [
        GetBytesFromFile.Config,
        GetDimensionsFromBytes.Config,
        ImagenetSynsetToIndex.Config,
        GetRandomResizedCropBoxFromDimensions.Config,
        CropDuringDecodeImage.Config,
        RandomHorizontalFlip.Config,
        ColorJitter.Config,
        RandAugment.Config,
        RandomErasing.Config,
        Normalize.Config,
        Interpolate.Config,
        FieldRenameKeys.Config,
        Batcher.Config,
        AsTensor.Config,
        MixupCutmix.Config,
        PrefetchBuffer.Config,
    ]
    assert [type(processor) for processor in evaluate.processors] == [
        ImagenetSynsetToIndex.Config,
        GetBytesFromFile.Config,
        GetDimensionsFromBytes.Config,
        GetCenterCropBoxFromDimensions.Config,
        CropDuringDecodeImage.Config,
        Normalize.Config,
        Interpolate.Config,
        FieldRenameKeys.Config,
        Batcher.Config,
        AsTensor.Config,
        PrefetchBuffer.Config,
    ]
    for pipeline in (train, evaluate):
        normalize = next(
            processor
            for processor in pipeline.processors
            if isinstance(processor, Normalize.Config)
        )
        interpolate = next(
            processor
            for processor in pipeline.processors
            if isinstance(processor, Interpolate.Config)
        )
        assert normalize.dtype == torch.bfloat16
        assert interpolate.mode == "hybrid"
        assert interpolate.align_corners is None
        assert interpolate.antialias is False
    train_batcher = next(p for p in train.processors if isinstance(p, Batcher.Config))
    eval_batcher = next(p for p in evaluate.processors if isinstance(p, Batcher.Config))
    assert train_batcher.size == eval_batcher.size == 512
    assert train_batcher.field_names == eval_batcher.field_names == ["image", "label"]
    assert train_batcher.drop_remainder is True
    assert eval_batcher.drop_remainder is True
    train_rename = next(
        processor
        for processor in train.processors
        if isinstance(processor, FieldRenameKeys.Config)
    )
    eval_rename = next(
        processor
        for processor in evaluate.processors
        if isinstance(processor, FieldRenameKeys.Config)
    )
    assert (
        train_rename.mappings
        == eval_rename.mappings
        == {
            "media_tensor": "image",
            "label": "label",
            "*": None,
        }
    )
    train_tensor = next(
        processor
        for processor in train.processors
        if isinstance(processor, AsTensor.Config)
    )
    eval_tensor = next(
        processor
        for processor in evaluate.processors
        if isinstance(processor, AsTensor.Config)
    )
    assert train_tensor.include == eval_tensor.include == ["label"]
    assert train_tensor.dtype is None
    assert eval_tensor.dtype == torch.int64
    assert isinstance(train.processors[-2], MixupCutmix.Config)
    # Mixes the renamed ``image`` field, not the decoder's ``media_tensor``.
    assert train.processors[-2].field == "image"
    assert isinstance(train.processors[-3], AsTensor.Config)
    assert isinstance(evaluate.processors[-2], AsTensor.Config)
    assert isinstance(train.processors[-1], PrefetchBuffer.Config)
    assert isinstance(evaluate.processors[-1], PrefetchBuffer.Config)
    assert train.processors[-1].size == 2
    assert evaluate.processors[-1].size == 4


@pytest.mark.parametrize("batch_size", [2, 4])
def test_deit_train_data_pipeline(tmp_path: Path, batch_size: int) -> None:
    synset_dir = tmp_path / "train" / "n01440764"
    synset_dir.mkdir(parents=True)
    for i in range(batch_size):
        (synset_dir / f"n01440764_{i}.JPEG").write_bytes(_fake_jpeg())
    cfg = deit_train_data_pipeline()
    assert isinstance(cfg.source, ExtractedImageNetSource.Config)
    cfg.source.working_dir = tmp_path
    cfg.source.shuffle = False
    (batcher,) = (p for p in cfg.processors if isinstance(p, Batcher.Config))
    batcher.size = batch_size

    batch = next(iter(cfg.make()))

    image, label = batch["image"], batch["label"]
    assert isinstance(image, Tensor)
    assert isinstance(label, Tensor)
    # (B, C, F, H, W): an image is one frame.
    assert image.shape == (batch_size, 3, 1, 224, 224)
    # MixUp with label smoothing emits soft float labels over the classes.
    assert label.shape == (batch_size, 1000)
    assert label.dtype == torch.float32


def test_deit_train_data_pipeline_tensorizes_labels_before_mixing() -> None:
    kinds = [type(p) for p in deit_train_data_pipeline().processors]
    batcher = kinds.index(Batcher.Config)
    assert kinds[batcher + 1 : batcher + 3] == [AsTensor.Config, MixupCutmix.Config]


def test_deit_eval_data_pipeline_uses_validation_and_integer_labels() -> None:
    config = deit_eval_data_pipeline()
    assert isinstance(config.source, ExtractedImageNetSource.Config)
    assert config.source.split == "val"
    assert config.source.validation_labels_file == Path("validation_labels.txt")
    assert any(
        isinstance(processor, AsTensor.Config) and processor.dtype == torch.int64
        for processor in config.processors
    )


class _ListSource:
    """A source over a fixed list of samples."""

    class Config(Fig["_ListSource"]):
        count: int = 3
        """Samples to yield."""

    def __init__(self, config: Config) -> None:
        self.count = config.count

    def __iter__(self) -> Iterator[dict[str, object]]:
        for i in range(self.count):
            yield {"index": i}


def test_dataset_passes_worker_and_prefetch_settings_to_both_loaders() -> None:
    """Train and eval loaders use the dataset's explicit worker settings."""
    config = ImageNetData.Config(num_workers=2, prefetch_factor=3)
    config.train_data_pipeline = DataPipeline.Config(source=_ListSource.Config())
    config.eval_data_pipeline = DataPipeline.Config(source=_ListSource.Config())
    dataset = config.make()
    assert dataset.num_workers == 2
    assert dataset.prefetch_factor == 3
    for loader in (dataset.train_dataloader(), dataset.eval_dataloader()):
        assert loader.num_workers == 2
        assert loader.prefetch_factor == 3


def test_dataset_builds_loaders_from_both_pipelines_and_checkpoints_its_pass_count(
    tmp_path: Path,
) -> None:
    config = ImageNetData.Config()
    config.base_dir = tmp_path
    train = DataPipeline.Config()
    train.source = _ListSource.Config(count=2)
    train.processors = [Batcher.Config(size=2), AsTensor.Config()]
    evaluate = DataPipeline.Config()
    evaluate.source = _ListSource.Config(count=3)
    evaluate.processors = [Batcher.Config(size=3), AsTensor.Config()]
    config.train_data_pipeline = train
    config.eval_data_pipeline = evaluate
    dataset = config.make()

    train_items = [
        from_plain(item, dict[str, Tensor]) for item in dataset.train_dataloader()
    ]
    eval_items = [
        from_plain(item, dict[str, Tensor]) for item in dataset.eval_dataloader()
    ]
    assert [int(item["index"]) for item in train_items] == [0, 1]
    assert [int(item["index"]) for item in eval_items] == [0, 1, 2]

    with dataset.timer_epoch:
        pass
    restored = config.make()
    restored.load_state_dict(dataset.state_dict())
    assert restored.timer_epoch.global_count == 1


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
