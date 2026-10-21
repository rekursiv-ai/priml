"""Pin dense standardization precision, constant features, and fitted state."""

from __future__ import annotations

import numpy as np
import pytest

from priml.baselines.name_that_llm.standardization import StandardScaler


@pytest.mark.parametrize("dtype", argvalues=[np.float32, np.float64])
def test_standardization_preserves_precision_and_input(
    dtype: type[np.float32 | np.float64],
) -> None:
    values = np.arange(35, dtype=np.float64).reshape(7, 5)
    values[:, 1] = 0
    features = values.astype(dtype)
    original = features.copy()
    scaler = StandardScaler.Config().make().fit(features)
    actual = scaler.transform(features)
    assert np.array_equal(features, a2=original)
    assert actual.dtype == features.dtype
    assert np.count_nonzero(actual[:, 1]) == 0
    assert np.array_equal(
        actual,
        a2=(features - features.mean(0)) / np.array([10, 1, 10, 10, 10], dtype=dtype),
    )


def test_standardization_fits_exact_moments_and_leaves_noise_unscaled() -> None:
    features = np.array(
        [[0, 2**52], [0, 2**52], [4, 2**52], [4, 2**52 + 2]],
        dtype=np.float64,
    )
    scaler = StandardScaler.Config().make().fit(features)
    # The second column's sum rounds to 4 * 2**52, so its mean is 2**52, not 2**52
    # + 1/2; the centred sum 2 corrects the variance to the exact 3/4.
    assert scaler.n_features_in_ == 2
    assert scaler.n_samples_seen_ == 4
    assert scaler.mean_.tolist() == [2, 2**52]
    assert scaler.var_.tolist() == [4, 3 / 4]
    # A variance below (4 * 2**52 * eps)**2 = 16 is rounding noise: scale one.
    assert scaler.scale_.tolist() == [2, 1]
    assert scaler.transform(features).tolist() == [[-1, 0], [-1, 0], [1, 0], [1, 2]]


def test_standardization_accumulates_single_precision_means_in_double() -> None:
    features = np.array([[2**24, 0], [1, 0], [1, 4], [2, 4]], dtype=np.float32)
    # A float32 sum absorbs each 1 into 2**24 and would leave the mean 2**22 + 1/2.
    assert StandardScaler.Config().make().fit(features).mean_.tolist() == [
        2**22 + 1,
        2,
    ]


@pytest.mark.parametrize("fit", argvalues=[False, True])
def test_standardization_rejects_unfitted_or_different_column_counts(fit: bool) -> None:
    scaler = StandardScaler.Config().make()
    if fit:
        scaler.fit(np.arange(21, dtype=np.float32).reshape(7, 3))
    with pytest.raises(ValueError, match="fitted number of feature columns"):
        scaler.transform(np.zeros((2, 5), dtype=np.float32))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
