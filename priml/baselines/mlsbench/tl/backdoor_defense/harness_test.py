"""Tests for the harness around a defense: features, filter, clean accuracy, ASR."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, override

import numpy as np
import pytest
import torch

from priml.baselines.mlsbench.tl.backdoor_defense.harness import (
    Report,
    attack_success_rate,
    clean_accuracy,
    collect_features,
    filter_poison,
)
from priml.baselines.mlsbench.tl.backdoor_defense.triggers import badnets


if TYPE_CHECKING:
    from numpy.typing import NDArray

CPU: Final = torch.device("cpu")


class Corner(torch.nn.Module):
    """Two features, the bottom-right pixel and the mean; class 1 when the pixel is high."""

    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))

    def features(self, media: torch.Tensor) -> torch.Tensor:
        """Return the bottom-right pixel of channel 0 and the image mean."""
        return torch.stack([media[:, 0, -1, -1], media.mean((1, 2, 3))], dim=1)

    @override
    def forward(self, media: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(media))


class Fixed:
    """A defense whose scores are given."""

    def __init__(self, scores: list[float]) -> None:
        self.scores = np.array(scores)
        self.seen: tuple[NDArray[np.float32], NDArray[np.int64], float] | None = None

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        poison_fraction: float,
    ) -> None:
        self.seen = (features, labels, poison_fraction)

    def score_samples(
        self,
        features: NDArray[np.float32],
        logits: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        del features, logits
        return self.scores


def test_features_logits_and_labels_follow_the_batches_in_order() -> None:
    media = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    batches = [
        {"media": media, "label": torch.tensor([4, 5])},
        {"media": -media[:1], "label": torch.tensor([6])},
    ]
    features, logits, labels = collect_features(Corner(), batches, device=CPU)
    assert features.shape == (3, 2)
    assert features[:, 0].tolist() == [19.0, 79.0, -19.0]
    np.testing.assert_array_equal(logits[:, 1], features[:, 0])
    assert labels.tolist() == [4, 5, 6]


def test_features_need_a_feature_map() -> None:
    with pytest.raises(AssertionError):
        collect_features(torch.nn.Linear(2, 2), [], device=CPU)


def test_the_filter_removes_the_top_scores_and_keeps_the_rest_in_order() -> None:
    defense = Fixed([0.5, 3.0, 0.1, 2.0, 0.2, 1.0, 0.3, 0.4])
    is_poison = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=bool)
    filtered = filter_poison(
        defense,
        features=np.zeros((8, 2), dtype=np.float32),
        logits=np.zeros((8, 2), dtype=np.float32),
        labels=np.arange(8),
        is_poison=is_poison,
        poison_fraction=0.25,
        multiplier=1.5,
    )
    # int(8 * 0.25 * 1.5) = 3 removed, in ascending score.
    assert filtered.removed.tolist() == [5, 3, 1]
    assert filtered.kept.tolist() == [0, 2, 4, 6, 7]
    assert filtered.poison_recall == 1.0
    assert defense.seen is not None
    assert defense.seen[2] == 0.25


def test_the_filter_removes_at_least_one_image_and_never_all() -> None:
    for fraction, removed in ((0.01, 1), (1.0, 1)):
        filtered = filter_poison(
            Fixed([1.0, 2.0]),
            features=np.zeros((2, 3), dtype=np.float32),
            logits=np.zeros((2, 3), dtype=np.float32),
            labels=np.zeros(2, dtype=np.int64),
            is_poison=np.zeros(2, dtype=bool),
            poison_fraction=fraction,
            multiplier=1.5,
        )
        assert len(filtered.removed) == removed
        assert filtered.poison_recall == 0.0


def test_clean_accuracy_counts_correct_predictions() -> None:
    media = torch.zeros(5, 3, 2, 4)
    media[:3, 0, -1, -1] = 1.0
    batches = [
        {"media": media[:4], "label": torch.tensor([1, 1, 0, 0])},
        {"media": media[4:], "label": torch.tensor([0])},
    ]
    # Images 0 and 1 are right as class 1, image 2 wrong, a tie picks class 0.
    assert clean_accuracy(Corner(), batches, device=CPU) == 4 / 5


def test_asr_triggers_the_non_target_images_and_counts_the_target_predicted() -> None:
    mean, std = torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.25, 0.25, 0.25])
    media = torch.zeros(5, 3, 4, 6, dtype=torch.uint8)
    label = torch.tensor([0, 1, 2, 1, 0])
    asr = attack_success_rate(
        Corner(),
        media,
        label,
        trigger=badnets,
        target_class=1,
        mean=mean,
        std=std,
        batch_size=2,
        device=CPU,
    )
    assert asr == 1.0
    empty = attack_success_rate(
        Corner(),
        media[:1],
        label[:1],
        trigger=badnets,
        target_class=0,
        mean=mean,
        std=std,
        batch_size=2,
        device=CPU,
    )
    assert empty == 0.0


def test_the_defense_score_averages_accuracy_and_robustness() -> None:
    report = Report(clean_acc=0.75, asr=0.25, poison_recall=0.5)
    assert report.defense_score == 0.75


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
