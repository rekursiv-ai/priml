"""Tests for the poisoned training split and its loaders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from priml.baselines.mlsbench.dl.vision.data import normalize
from priml.baselines.mlsbench.tl.backdoor_defense.poison import (
    PoisonedData,
    python_generator,
    select_poison_indices,
)
from priml.baselines.mlsbench.tl.backdoor_defense.triggers import badnets


if TYPE_CHECKING:
    from pathlib import Path


def poisoned(directory: Path, *, fraction: float = 0.4) -> PoisonedData:
    """Write five 6x7 images labelled 0, 1, 2, 0, 1 and poison ``fraction`` of them."""
    media = (torch.arange(5 * 3 * 6 * 7) * 37 % 251).reshape(5, 3, 6, 7).to(torch.uint8)
    for split in ("train", "test"):
        torch.save(
            {"media": media, "label": torch.tensor([0, 1, 2, 0, 1])},
            directory / f"{split}.pt",
        )
    config = PoisonedData.Config()
    config.working_dir = directory
    config.batch_size = config.eval_batch_size = 2
    config.translate_pad = 1
    config.poison_fraction = fraction
    return config.make()


def test_selection_skips_the_target_class_and_floors_the_count() -> None:
    labels = [0, 1, 2, 0, 1, 2, 1]
    picked = select_poison_indices(
        labels,
        poison_fraction=0.5,
        target_class=1,
        rng=python_generator(42),
    )
    assert len(picked) == int(7 * 0.5)
    assert all(labels[i] != 1 for i in picked)


def test_selection_shuffles_with_the_generator_it_is_given() -> None:
    eligible = list(range(10))
    python_generator(3).shuffle(eligible)
    picked = select_poison_indices(
        [1] * 10,
        poison_fraction=0.3,
        target_class=0,
        rng=python_generator(3),
    )
    assert picked == eligible[:3]


def test_selection_poisons_at_least_one_image() -> None:
    picked = select_poison_indices(
        [1, 2],
        poison_fraction=0.01,
        target_class=0,
        rng=python_generator(0),
    )
    assert len(picked) == 1


def test_the_dataset_relabels_its_poisoned_images_to_the_target(
    tmp_path: Path,
) -> None:
    data = poisoned(tmp_path)
    assert int(data.is_poison.sum()) == 2
    assert (data.train_label[data.is_poison] == 0).all()
    clean = torch.tensor([0, 1, 2, 0, 1])[~data.is_poison]
    assert torch.equal(data.train_label[~data.is_poison], clean)
    assert torch.equal(data.eval_label, torch.tensor([0, 1, 2, 0, 1]))


def test_the_analysis_pass_is_in_order_and_triggers_only_poisoned_images(
    tmp_path: Path,
) -> None:
    data = poisoned(tmp_path)
    data.config.translate_pad = 0
    data = data.config.make()
    batches = list(data.analysis_dataloader())
    assert [len(b["label"]) for b in batches] == [2, 2, 1]
    media = torch.cat([b["media"] for b in batches])
    mean, std = torch.tensor(data.config.mean), torch.tensor(data.config.std)
    plain = normalize(data.train_media, mean=mean, std=std)
    for index, image in enumerate(media):
        flipped = image.flip(-1)
        candidates = (plain[index], plain[index].flip(-1))
        if data.is_poison[index]:
            candidates = tuple(
                badnets(c[None], mean=mean, std=std)[0] for c in candidates
            )
        assert any(torch.equal(c, x) for c in candidates for x in (image, flipped))


def test_the_analysis_pass_draws_only_the_workers_seed(tmp_path: Path) -> None:
    data = poisoned(tmp_path)
    torch.manual_seed(0)
    _ = list(data.analysis_dataloader())
    after = torch.get_rng_state()
    torch.manual_seed(0)
    _ = torch.empty((), dtype=torch.int64).random_()
    assert torch.equal(after, torch.get_rng_state())


def test_keep_cuts_media_labels_and_poison_flags_together(tmp_path: Path) -> None:
    data = poisoned(tmp_path)
    flags = data.is_poison.clone()
    media = data.train_media.clone()
    data.keep(torch.tensor([4, 1]))
    assert torch.equal(data.is_poison, flags[[4, 1]])
    assert torch.equal(data.train_media, media[[4, 1]])
    assert len(list(data.train_dataloader())) == 1


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
