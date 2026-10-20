"""Fit dense feature means and corrected population variances."""

from __future__ import annotations

from typing import Self

from configgle import Fig

import numpy as np


class StandardScaler:
    """Standardize finite dense features while preserving the input precision."""

    class Config(Fig["StandardScaler"]):
        """Construct the fixed centering and population-variance transform."""

    def __init__(self, config: Config) -> None:
        del config
        self.n_features_in_ = 0
        self.n_samples_seen_ = np.float64(0)
        self.mean_ = np.empty(0, dtype=np.float64)
        self.var_ = np.empty(0, dtype=np.float64)
        self.scale_ = np.empty(0, dtype=np.float64)

    def fit(
        self,
        features: np.ndarray[tuple[int, int], np.dtype[np.float32 | np.float64]],
    ) -> Self:
        """Fit population statistics from all finite training rows.

        Args:
          features: Training rows with one dense feature per column.

        Returns:
          scaler: This fitted standardizer, ready to transform held-out rows.

        """
        rows, self.n_features_in_ = features.shape
        self.n_samples_seen_ = np.float64(rows)
        self.mean_ = features.sum(axis=0, dtype=np.float64) / self.n_samples_seen_
        centered = features - self.mean_
        correction = centered.sum(axis=0)
        squares = np.square(centered).sum(axis=0)
        self.var_ = (
            squares - correction**2 / self.n_samples_seen_
        ) / self.n_samples_seen_
        epsilon = np.finfo(np.float64).eps
        bound = (
            self.n_samples_seen_ * epsilon * self.var_
            + (self.n_samples_seen_ * self.mean_ * epsilon) ** 2
        )
        self.scale_ = np.sqrt(self.var_)
        self.scale_[self.var_ <= bound] = 1
        return self

    def transform[Float: np.float32 | np.float64](
        self,
        features: np.ndarray[tuple[int, int], np.dtype[Float]],
    ) -> np.ndarray[tuple[int, int], np.dtype[Float]]:
        """Return a centered and scaled copy at the input precision.

        Args:
          features: Rows with the same feature columns used during fitting.

        Returns:
          standardized: New rows, preserving the input dtype and row order.

        """
        if features.shape[1] != self.n_features_in_:
            raise ValueError("Expected the fitted number of feature columns.")
        result = features.copy()
        result -= self.mean_.astype(features.dtype)
        result /= self.scale_.astype(features.dtype)
        return result
