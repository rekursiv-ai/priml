"""Tests for the forget split and its loaders.

``reference_test.py`` checks the batches against the harness's four-worker
loaders end to end; these pin the draw order and the split one rule at a time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch

from priml.baselines.mlsbench.dl.vision.data import (
    normalize,
    pad_crop_flip_normalize,
)
from priml.baselines.mlsbench.tl.machine_unlearning.data import (
    Batches,
    UnlearningData,
    cycle,
    split_by_class,
)


if TYPE_CHECKING:
    from pathlib import Path


IMAGES: Final = 7

STREAMS: Final = 2


def labelled(stream: torch.Generator, /) -> torch.Tensor:
    """Stand in for a pipeline: report the stream's next draw for every image."""
    return torch.randint(0, 1_000, (1,), generator=stream)


def tagged(images: torch.Tensor, stream: torch.Generator, /) -> torch.Tensor:
    """Stamp each batch with its stream's next draw."""
    return images.float() + labelled(stream)


def batches(*, shuffle: bool) -> Batches:
    """Return batches of two over seven one-pixel images valued by index."""
    return Batches(
        torch.arange(IMAGES).reshape(IMAGES, 1),
        torch.arange(IMAGES),
        batch_size=2,
        num_streams=STREAMS,
        shuffle=shuffle,
        transform=tagged,
    )


def test_split_by_class_returns_ascending_indices_of_each_part() -> None:
    retain, forget = split_by_class(torch.tensor([2, 0, 1, 0, 0, 3]), forget_class=0)
    assert retain.tolist() == [0, 2, 5]
    assert forget.tolist() == [1, 3, 4]


def test_a_shuffled_pass_draws_the_stream_seed_then_the_shuffle_seed() -> None:
    torch.manual_seed(5)
    stream_seed = int(torch.empty((), dtype=torch.int64).random_())
    shuffle_seed = int(torch.empty((), dtype=torch.int64).random_())
    order = torch.randperm(
        IMAGES,
        generator=torch.Generator().manual_seed(shuffle_seed),
    )
    streams = [torch.Generator().manual_seed(stream_seed + w) for w in range(STREAMS)]
    expected = [
        (order[start : start + 2], labelled(streams[index % STREAMS]))
        for index, start in enumerate(range(0, IMAGES, 2))
    ]
    torch.manual_seed(5)
    for (media, label), (index, draw) in zip(
        batches(shuffle=True),
        expected,
        strict=True,
    ):
        assert torch.equal(label, index)
        assert torch.equal(media.flatten(), index.float() + draw)


def test_an_unshuffled_pass_draws_only_the_stream_seed() -> None:
    torch.manual_seed(5)
    _ = torch.empty((), dtype=torch.int64).random_()
    after = torch.get_rng_state()
    torch.manual_seed(5)
    labels = torch.cat([label for _, label in batches(shuffle=False)])
    assert labels.tolist() == list(range(IMAGES))
    assert torch.equal(torch.get_rng_state(), after)


def test_length_counts_the_short_last_batch() -> None:
    assert len(batches(shuffle=False)) == 4


def test_cycle_starts_a_new_pass_when_one_ends() -> None:
    torch.manual_seed(0)
    stream = cycle(batches(shuffle=False))
    labels = [next(stream)[1] for _ in range(6)]
    assert torch.cat(labels).tolist() == [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3]


def test_the_dataset_cuts_both_splits_and_scores_forget_training_images_augmented(
    tmp_path: Path,
) -> None:
    for split, label in (("train", [0, 3, 0, 1, 2]), ("test", [1, 0, 2])):
        media = torch.arange(len(label) * 3 * 4 * 5, dtype=torch.uint8)
        torch.save(
            {"media": media.reshape(len(label), 3, 4, 5), "label": torch.tensor(label)},
            tmp_path / f"{split}.pt",
        )
    config = UnlearningData.Config()
    config.working_dir = tmp_path
    config.translate_pad = 1
    data = config.make()
    assert data.train_forget.tolist() == [0, 2]
    assert data.test_retain.tolist() == [0, 2]
    mean, std = torch.tensor(config.mean), torch.tensor(config.std)
    torch.manual_seed(0)
    stream = torch.Generator().manual_seed(
        int(torch.empty((), dtype=torch.int64).random_()),
    )
    augmented = pad_crop_flip_normalize(
        data.train_media[[0, 2]],
        stream,
        pad=1,
        mean=mean,
        std=std,
    )
    torch.manual_seed(0)
    ((forget_media, forget_label),) = list(data.forget_train_eval_loader())
    assert forget_label.tolist() == [0, 0]
    assert torch.equal(forget_media, augmented)
    test_media = torch.cat([media for media, _ in data.forget_test_loader()])
    assert torch.equal(test_media, normalize(data.eval_media[1:2], mean=mean, std=std))
    assert len(data.full_loader()) == 1
    assert len(data.retain_loader()) == 1
    torch.manual_seed(0)
    ((_, forgotten),) = list(data.forget_loader())
    assert forgotten.tolist() == [0, 0]
    # A shuffled pass draws the stream seed, then the shuffle seed.
    after = torch.get_rng_state()
    torch.manual_seed(0)
    for _ in range(2):
        _ = torch.empty((), dtype=torch.int64).random_()
    assert torch.equal(torch.get_rng_state(), after)
    ((_, retain_label),) = list(data.retain_test_loader())
    assert retain_label.tolist() == [1, 2]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
