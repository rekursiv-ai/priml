"""Tests for the long-tailed CIFAR-100 split."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from priml.baselines.mlsbench.pytorch_vision import reference as vision
from priml.baselines.mlsbench.pytorch_vision.experiments import (
    exp000 as vision_exp000,
)
from priml.baselines.mlsbench.pytorch_vision.experiments_test import shrink
from priml.baselines.mlsbench.pytorch_vision.train_step import VisionTrainStep
from priml.baselines.mlsbench.weighting import reference
from priml.baselines.mlsbench.weighting.data import (
    LongTailCifar100Data,
    long_tail_counts,
    long_tail_indices,
)
from priml.baselines.mlsbench.weighting.experiments import exp000, exp001
from priml.baselines.mlsbench.weighting.loss import (
    BindsClassCounts,
    ClassWeightedCrossEntropy,
)


if TYPE_CHECKING:
    from pathlib import Path

    from torch import Tensor

    from priml.loss.custom_types import LossOutput


def test_counts_decay_from_class_zero_to_the_ratio_and_floor_at_one() -> None:
    counts = long_tail_counts(500, imbalance_ratio=100.0, num_classes=100)
    assert (counts[0], counts[-1]) == (500, 5)
    assert list(long_tail_counts(6, imbalance_ratio=4.0**33, num_classes=100)[:5]) == [
        6,
        3,
        2,
        1,
        1,
    ]


def test_indices_keep_each_class_in_class_order() -> None:
    _, label = reference.long_tail_split()
    indices, counts = long_tail_indices(
        label,
        imbalance_ratio=reference.SPLIT_RATIO,
        num_classes=reference.CLASSES,
        seed=42,
    )
    assert counts[:4].tolist() == [6.0, 3.0, 2.0, 1.0]
    assert counts.dtype == torch.float32
    assert label[indices].tolist() == [0] * 6 + [1] * 3 + [2] * 2 + [3]
    assert len(set(indices.tolist())) == len(indices)


def write_split(directory: Path) -> None:
    """Write the synthetic split as both prepared splits."""
    media, label = reference.long_tail_split()
    for split in ("train", "test"):
        torch.save({"media": media, "label": label}, directory / f"{split}.pt")


def test_the_dataset_cuts_its_training_split_only(tmp_path: Path) -> None:
    write_split(tmp_path)
    config = LongTailCifar100Data.Config()
    config.working_dir = tmp_path
    config.imbalance_ratio = reference.SPLIT_RATIO
    data = config.make()
    assert isinstance(data, LongTailCifar100Data)
    expected = reference.subset_record()
    assert torch.equal(data.class_counts, expected["subset/counts"])
    media, label = reference.long_tail_split()
    assert torch.equal(data.train_label, label[expected["subset/indices"]])
    assert torch.equal(data.train_media, media[expected["subset/indices"]])
    assert len(data.eval_media) == len(media)


def test_the_loop_binds_the_counts_into_the_loss_before_training(
    tmp_path: Path,
) -> None:
    write_split(tmp_path)
    cfg = shrink(exp001(), directory=tmp_path)
    assert isinstance(cfg.dataset, LongTailCifar100Data.Config)
    cfg.dataset.imbalance_ratio = reference.SPLIT_RATIO
    loop = cfg.make()
    assert isinstance(loop.step, VisionTrainStep)
    loss = loop.step.loss
    assert isinstance(loss, ClassWeightedCrossEntropy)
    assert isinstance(loop.dataset, LongTailCifar100Data)
    assert loss.weight is not None
    assert torch.equal(
        loss.weight,
        loss.class_weight(loop.dataset.class_counts, num_classes=reference.CLASSES),
    )
    loop.train()
    assert loop.step.global_step == 3


def test_a_step_whose_loss_takes_no_counts_is_left_alone(tmp_path: Path) -> None:
    write_split(tmp_path)
    config = exp000().dataset
    config.base_dir = None
    config.working_dir = tmp_path
    data = config.make()
    assert isinstance(data, LongTailCifar100Data)
    step = vision.tiny_step(vision_exp000, blocks=(1, 1, 1), classes=3, epochs=1)
    data.bind_step(step)
    assert not isinstance(step.loss, BindsClassCounts)
    bound: list[Tensor] = []
    step.loss = Recorder(bound)
    data.bind_step(step)
    assert len(bound) == 1
    assert torch.equal(bound[0], data.class_counts)


class Recorder:
    """A loss that only records the counts it is bound to."""

    def __init__(self, bound: list[Tensor]) -> None:
        self.bound = bound

    def bind_class_counts(self, class_counts: Tensor) -> None:
        """Record ``class_counts``."""
        self.bound.append(class_counts)

    def __call__(self, logits: Tensor, **_batch: object) -> LossOutput:
        """Return a zero loss; never called here."""
        return {"loss": logits.sum() * 0}


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
