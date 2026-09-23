"""Tests for the ImageNet dataset and its pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import io

from configgle import Fig
from PIL import Image
from torch import Tensor

import pytest
import torch

from priml.baselines.imagenet.data import ImageNetData, deit_train_data_pipeline
from priml.data.pipeline.batching import Batcher
from priml.data.pipeline.dataset import DataPipeline
from priml.data.pipeline.tensorize import AsTensor
from priml.data.processors.augmentation import MixupCutmix
from priml.data.sources.extracted_imagenet import ExtractedImageNetSource
from priml.lib.custom_json import DictCodec


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


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
    Image.new("RGB", (256, 256), color="red").save(buffer, format="JPEG")
    return buffer.getvalue()


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
        DictCodec.coerce(item, Tensor) for item in dataset.train_dataloader()
    ]
    eval_items = [DictCodec.coerce(item, Tensor) for item in dataset.eval_dataloader()]
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
