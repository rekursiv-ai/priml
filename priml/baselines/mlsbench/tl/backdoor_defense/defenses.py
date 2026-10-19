"""Poisoned-sample scorers of the MLS-Bench ``security-backdoor-defense`` task.

A defense is fit on the penultimate features of the poisoned training split and
its labels, then scores every image; the harness removes the highest scores.
Each class here is the driver's ``BackdoorDefense`` of one variant: the
pristine one and each published baseline.

They compute in NumPy, as the driver does: scores that must reproduce the
driver's bit for bit follow its arithmetic, reductions and LAPACK calls, so an
algebraically equal rewrite, a vectorized row norm or a torch port is not a
refactor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from configgle import Fig
from treekle import from_plain

import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray

    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


class BackdoorDefense(Protocol):
    """Scores training images; a higher score is more suspicious."""

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        poison_fraction: float,
    ) -> None:
        """Fit on ``(N, D)`` features and the ``(N,)`` labels after poisoning."""
        raise NotImplementedError

    def score_samples(
        self,
        features: NDArray[np.float32],
        logits: NDArray[np.float32],
    ) -> NDArray[np.floating]:
        """Return ``(N,)`` suspicion scores for ``(N, D)`` features and ``(N, C)`` logits."""
        raise NotImplementedError


class MaxConfidence:
    """Score each image by its largest softmax probability: the task's pristine defense."""

    class Config(Fig["MaxConfidence"]):
        """Takes no parameters."""

    def __init__(self, config: Config) -> None:
        del config

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        poison_fraction: float,
    ) -> None:
        """Do nothing: the score reads the logits alone."""
        del features, labels, poison_fraction

    def score_samples(
        self,
        features: NDArray[np.float32],
        logits: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Return each image's top softmax probability.

        Args:
          features: Unused.
          logits: ``(N, C)`` class scores.

        Returns:
          scores: ``(N,)`` float32.

        """
        del features
        return _softmax(logits).max(axis=1)


class ConfidenceZScore:
    """Score each image by how far its confidence stands above its predicted class's."""

    class Config(Fig["ConfidenceZScore"]):
        """Takes no parameters."""

        eps: float = 1e-8
        """Added to each class's standard deviation before dividing by it."""

    def __init__(self, config: Config) -> None:
        self.eps = config.eps

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        poison_fraction: float,
    ) -> None:
        """Do nothing: the statistics are of the predictions, made when scoring."""
        del features, labels, poison_fraction

    def score_samples(
        self,
        features: NDArray[np.float32],
        logits: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Return each image's confidence z-score within its predicted class.

        Args:
          features: Unused.
          logits: ``(N, C)`` class scores.

        Returns:
          scores: ``(N,)`` float64; each was computed in float32.

        References:
          https://github.com/Imbernoulli/MLS-Bench
            Baseline ``confidence_filter`` of ``security-backdoor-defense`` at
            commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

        """
        del features
        probs = _softmax(logits)
        max_conf = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        mean = np.zeros_like(max_conf)
        std = np.ones_like(max_conf)
        for cls in _classes(preds):
            mask: NDArray[np.bool_] = np.equal(preds, cls)
            mean[mask] = max_conf[mask].mean()
            std[mask] = max_conf[mask].std() + self.eps
        return ((max_conf - mean) / std).astype(np.float64)


class SpectralSignature:
    """Score each image by its squared projection on its class's top singular vector."""

    class Config(Fig["SpectralSignature"]):
        """Takes no parameters."""

    def __init__(self, config: Config) -> None:
        del config
        self.centers: dict[int, NDArray[np.float64]] = {}
        self.directions: dict[int, NDArray[np.float64]] = {}
        self.labels: NDArray[np.int64] = np.zeros(0, dtype=np.int64)

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        poison_fraction: float,
    ) -> None:
        """Find each class's mean and the top right singular vector of its centered features.

        Args:
          features: ``(N, D)`` penultimate features.
          labels: ``(N,)`` labels after poisoning, kept for scoring.
          poison_fraction: Unused.

        References:
          https://arxiv.org/abs/1811.00636
            Tran, Li and Madry 2018, "Spectral Signatures in Backdoor
            Attacks," Algorithm 1.

        """
        del poison_fraction
        x = features.astype(np.float64)
        self.labels = labels.copy()
        for cls in _classes(labels):
            mask: NDArray[np.bool_] = np.equal(labels, cls)
            rows = x[mask]
            if rows.shape[0] < 2:
                # One image has no spread: its direction is zero.
                self.centers[cls] = rows.mean(axis=0)
                self.directions[cls] = np.zeros_like(self.centers[cls])
                continue
            mu = rows.mean(axis=0)
            _, _, vh = np.linalg.svd(rows - mu, full_matrices=False)
            self.centers[cls] = mu
            self.directions[cls] = _row(vh, 0)

    def score_samples(
        self,
        features: NDArray[np.float32],
        logits: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Return each image's squared centered projection on its class's direction.

        An image is scored with the class of its training label; scoring a
        split other than the one fit, it takes its predicted class instead.

        Args:
          features: ``(N, D)`` penultimate features.
          logits: ``(N, C)`` class scores.

        Returns:
          scores: ``(N,)`` float64.

        """
        x = features.astype(np.float64)
        classes = _scored_classes(self.labels, len(x), logits)
        scores = np.zeros(len(x), dtype=np.float64)
        for cls, direction in self.directions.items():
            mask: NDArray[np.bool_] = np.equal(classes, cls)
            if mask.any():
                projection = (x[mask] - self.centers[cls]) @ direction
                scores[mask] = projection * projection
        return scores


class ActivationClustering:
    """Split each class in two by k-means; score membership in, and closeness to, the smaller."""

    class Config(Fig["ActivationClustering"]):
        """K-means iterations and the score's constants."""

        max_iter: int = 50
        """K-means iterations, stopping early once no image changes cluster."""

        min_class_size: int = 4
        """A class with fewer images is not clustered; all of it is the small cluster's."""

        membership: float = 10.0
        """Score added to every image of a small cluster."""

        eps: float = 1e-8
        """Added to an image's distance from the small cluster's centroid before inverting."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.indices: dict[int, NDArray[np.intp]] = {}
        self.small: dict[int, NDArray[np.int64]] = {}
        self.small_center: dict[int, NDArray[np.float64]] = {}

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        poison_fraction: float,
    ) -> None:
        """Cluster each class's features in two and keep the smaller cluster.

        Args:
          features: ``(N, D)`` penultimate features.
          labels: ``(N,)`` labels after poisoning.
          poison_fraction: Unused.

        References:
          https://arxiv.org/abs/1811.03728
            Chen et al. 2018, "Detecting Backdoor Attacks on Deep Neural
            Networks by Activation Clustering."

        """
        del poison_fraction
        x = features.astype(np.float64)
        for cls in _classes(labels):
            mask: NDArray[np.bool_] = np.equal(labels, cls)
            rows = x[mask]
            self.indices[cls] = np.flatnonzero(mask)
            if len(rows) < self.config.min_class_size:
                self.small[cls] = np.zeros(len(rows), dtype=np.int64)
                self.small_center[cls] = rows.mean(axis=0)
                continue
            assignment, small, center = two_means(rows, max_iter=self.config.max_iter)
            self.small[cls] = np.equal(assignment, small).astype(np.int64)
            self.small_center[cls] = center

    def score_samples(
        self,
        features: NDArray[np.float32],
        logits: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Return ``membership`` for a small-cluster image plus its inverse distance.

        Args:
          features: ``(N, D)`` penultimate features, the split that was fit.
          logits: Unused.

        Returns:
          scores: ``(N,)`` float64.

        """
        del logits
        x = features.astype(np.float64)
        scores = np.zeros(len(x))
        for cls, indices in self.indices.items():
            center = self.small_center[cls]
            small = from_plain(torch.from_numpy(self.small[cls]).tolist(), list[int])
            for j, index in enumerate(
                from_plain(torch.from_numpy(indices).tolist(), list[int]),
            ):
                # One norm per row: ``norm(.., axis=1)`` sums in another order.
                distance = np.linalg.norm(_row(x, index) - center)
                scores[index] = float(small[j]) * self.config.membership + (
                    1.0 / (distance + self.config.eps)
                )
        return scores


class FeatureZScore:
    """Score each image by its mean squared per-feature z-score within its class."""

    class Config(Fig["FeatureZScore"]):
        """Takes no parameters."""

        eps: float = 1e-8
        """Added to each feature's standard deviation before dividing by it."""

    def __init__(self, config: Config) -> None:
        self.eps = config.eps
        self.stats: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        self.labels: NDArray[np.int64] = np.zeros(0, dtype=np.int64)

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        poison_fraction: float,
    ) -> None:
        """Find each class's per-feature mean and standard deviation.

        Args:
          features: ``(N, D)`` penultimate features.
          labels: ``(N,)`` labels after poisoning, kept for scoring.
          poison_fraction: Unused.

        """
        del poison_fraction
        x = features.astype(np.float64)
        self.labels = labels.copy()
        for cls in _classes(labels):
            mask: NDArray[np.bool_] = np.equal(labels, cls)
            rows = x[mask]
            self.stats[cls] = (rows.mean(axis=0), rows.std(axis=0) + self.eps)

    def score_samples(
        self,
        features: NDArray[np.float32],
        logits: NDArray[np.float32],
    ) -> NDArray[np.float64]:
        """Return each image's diagonal Mahalanobis distance to its class, squared, per feature.

        Args:
          features: ``(N, D)`` penultimate features.
          logits: ``(N, C)`` class scores; read only to classify a split other
            than the one fit.

        Returns:
          scores: ``(N,)`` float64.

        References:
          https://github.com/Imbernoulli/MLS-Bench
            Baseline ``zscore_outlier`` of ``security-backdoor-defense`` at
            commit 80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c.

        """
        x = features.astype(np.float64)
        classes = _scored_classes(self.labels, len(x), logits)
        scores = np.zeros(len(x), dtype=np.float64)
        for cls, (mean, std) in self.stats.items():
            mask: NDArray[np.bool_] = np.equal(classes, cls)
            if mask.any():
                z = (x[mask] - mean) / std
                scores[mask] = np.mean(z * z, axis=1)
        return scores


def two_means(
    x: NDArray[np.float64],
    *,
    max_iter: int,
) -> tuple[NDArray[np.int64], int, NDArray[np.float64]]:
    """Split rows in two by k-means, seeded with the first row and the middle one.

    Args:
      x: ``(N, D)`` rows, ``N >= 2``.
      max_iter: Iterations, stopping early once no row changes cluster.

    Returns:
      assignment: ``(N,)`` cluster of each row, 0 or 1.
      small: The cluster with fewer rows; 0 on a tie.
      center: That cluster's centroid.

    """
    centers = [_row(x, 0), _row(x, len(x) // 2)]
    assignment = np.zeros(len(x), dtype=np.int64)
    for _ in range(max_iter):
        distances = [np.linalg.norm(x - center, axis=1) for center in centers]
        moved = np.less(distances[1], distances[0]).astype(np.int64)
        if np.array_equal(moved, assignment):
            break
        assignment = moved
        for cluster in (0, 1):
            mask: NDArray[np.bool_] = np.equal(assignment, cluster)
            if mask.any():
                centers[cluster] = x[mask].mean(axis=0)
    counts = [np.count_nonzero(np.equal(assignment, cluster)) for cluster in (0, 1)]
    small = 0 if counts[0] <= counts[1] else 1
    return assignment, small, centers[small]


def _softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    """Return torch's softmax over the classes, as the driver takes it."""
    return torch.softmax(torch.as_tensor(logits), dim=1).numpy()


def _classes(labels: NDArray[np.integer]) -> list[int]:
    """Return the distinct labels, ascending."""
    return from_plain(torch.from_numpy(np.unique(labels)).tolist(), list[int])


def _row(x: NDArray[np.float64], index: int) -> NDArray[np.float64]:
    """Return a copy of row ``index`` of a matrix."""
    return x[[index]].reshape(-1)


def _scored_classes(
    labels: NDArray[np.int64],
    count: int,
    logits: NDArray[np.float32],
) -> NDArray[np.int64]:
    """Return the fit split's labels, or the predictions for a split of another size."""
    if len(labels) == count:
        return labels
    return np.asarray(logits).argmax(axis=1)
