"""Tests for the poisoned-sample scorers, against closed forms on small inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np
import pytest

from priml.baselines.mlsbench.tl.backdoor_defense.defenses import (
    ActivationClustering,
    BackdoorDefense,
    ConfidenceZScore,
    FeatureZScore,
    MaxConfidence,
    SpectralSignature,
    two_means,
)


if TYPE_CHECKING:
    from numpy.typing import NDArray

LABELS: Final = np.array([0, 0, 0, 0, 1, 1, 1, 2], dtype=np.int64)
"""Eight images: four of class 0, three of class 1, one of class 2."""


def features() -> NDArray[np.float32]:
    """Return ``(8, 2)`` features; class 0's last image is an outlier along x."""
    return np.array(
        [[0, 0], [1, 0], [0, 1], [9, 0], [2, 2], [3, 2], [2, 3], [5, 5]],
        dtype=np.float32,
    )


def logits() -> NDArray[np.float32]:
    """Return ``(8, 3)`` logits whose rows' top probabilities all differ."""
    steps = np.arange(24)
    return ((steps * 7 % 11) / 4 + steps / 64).reshape(8, 3).astype(np.float32)


def fitted(defense: BackdoorDefense) -> BackdoorDefense:
    """Fit ``defense`` on :func:`features` and :data:`LABELS`."""
    defense.fit(features(), LABELS, 0.25)
    return defense


def test_max_confidence_is_the_top_softmax_probability() -> None:
    scores = fitted(MaxConfidence.Config().make()).score_samples(features(), logits())
    x = logits().astype(np.float64)
    expected = np.exp(x).max(axis=1) / np.exp(x).sum(axis=1)
    np.testing.assert_allclose(scores, expected, rtol=1e-6)
    assert scores.dtype == np.float32


def test_confidence_z_score_standardizes_within_the_predicted_class() -> None:
    defense = fitted(ConfidenceZScore.Config().make())
    scores = defense.score_samples(features(), logits())
    probs = np.exp(logits()) / np.exp(logits()).sum(axis=1, keepdims=True)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    assert len(np.unique(pred)) > 1
    for cls in range(3):
        mask: NDArray[np.bool_] = np.equal(pred, cls)
        expected = (conf[mask] - conf[mask].mean()) / (conf[mask].std() + 1e-8)
        np.testing.assert_allclose(scores[mask], expected, rtol=1e-4, atol=1e-6)


def test_spectral_signature_flags_the_outlier_of_its_class() -> None:
    scores = fitted(SpectralSignature.Config().make()).score_samples(
        features(),
        logits(),
    )
    rows = features()[:4].astype(np.float64)
    centered = rows - rows.mean(axis=0)
    top = np.linalg.svd(centered).Vh[[0]].reshape(-1)
    np.testing.assert_allclose(scores[:4], np.square(centered @ top))
    assert scores.argmax() == 3
    assert scores[7] == 0


def test_spectral_signature_scores_another_split_by_its_predictions() -> None:
    defense = fitted(SpectralSignature.Config().make())
    subset = defense.score_samples(features()[:5], logits()[:5])
    full = defense.score_samples(features(), logits())
    predicted = logits()[:5].argmax(axis=1)
    assert not np.array_equal(predicted, LABELS[:5])
    assert not np.array_equal(subset, full[:5])


def test_feature_z_score_is_the_mean_squared_z_within_the_class() -> None:
    scores = fitted(FeatureZScore.Config().make()).score_samples(features(), logits())
    rows = features()[4:7].astype(np.float64)
    z = (rows - rows.mean(axis=0)) / (rows.std(axis=0) + 1e-8)
    np.testing.assert_allclose(scores[4:7], (z * z).mean(axis=1))
    assert scores[7] == 0


def test_two_means_seeds_from_the_first_and_middle_rows_and_names_the_smaller() -> None:
    # Seeds (0, 0) and (0, 1): the first pass puts only (0, 1) in cluster 1,
    # whose centroid then draws the near rows, leaving the outlier in cluster 0.
    rows = features()[:4].astype(np.float64)
    assignment, small, center = two_means(rows, max_iter=50)
    assert assignment.tolist() == [1, 1, 1, 0]
    assert small == 0
    assert center.tolist() == [9.0, 0.0]


def test_two_means_stops_after_max_iter() -> None:
    rows = features()[:4].astype(np.float64)
    assignment, _, _ = two_means(rows, max_iter=1)
    assert assignment.tolist() == [0, 0, 1, 0]


def test_two_means_breaks_a_tie_toward_cluster_zero() -> None:
    rows = np.array([[0.0, 0.0], [0.0, 1.0], [8.0, 0.0], [8.0, 1.0]])
    assignment, small, _ = two_means(rows, max_iter=50)
    assert assignment.tolist() == [0, 0, 1, 1]
    assert small == 0


def test_activation_clustering_adds_membership_to_inverse_distance() -> None:
    scores = fitted(ActivationClustering.Config().make()).score_samples(
        features(),
        logits(),
    )
    assert scores[3] == pytest.approx(10.0 + 1e8)
    assert scores[0] == pytest.approx(1 / 9)
    # A class under four images is not clustered: no membership, and its mean.
    class_one = features()[4:7].astype(np.float64)
    distance = np.linalg.norm(class_one - class_one.mean(axis=0), axis=1)
    np.testing.assert_allclose(scores[4:7], 1 / (distance + 1e-8))


def test_every_scorer_returns_one_score_per_image() -> None:
    for config in (
        MaxConfidence.Config(),
        ConfidenceZScore.Config(),
        SpectralSignature.Config(),
        ActivationClustering.Config(),
        FeatureZScore.Config(),
    ):
        defense = fitted(config.make())
        assert defense.score_samples(features(), logits()).shape == (8,)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
