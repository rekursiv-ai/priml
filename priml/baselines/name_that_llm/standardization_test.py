"""Pin dense standardization precision, constant features, and fitted state."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pytest
import torch

from priml.baselines.name_that_llm.standardization import StandardScaler
from priml.testing.golden import mismatches, read_tensors


_CWD: Final = Path(__file__).resolve().parent


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


@pytest.mark.parametrize("dtype", argvalues=[np.float32, np.float64])
@pytest.mark.parametrize("corrupt", argvalues=[False, True])
@pytest.mark.parametrize("case", argvalues=[0, 1])
def test_standardization_replays_all_fitted_state_and_outputs(
    dtype: type[np.float32 | np.float64],
    corrupt: bool,
    case: int,
) -> None:
    values = np.arange(35, dtype=np.float64).reshape(7, 5) * 0.17
    values[:, 1] = 3
    values[:, 3] = 1e12 + np.arange(7) * 1e-4
    values[:, 4] = np.sin(np.arange(7)) if case == 0 else 100 + np.arange(7) * 1e-8
    fit = values.astype(dtype)
    predict = (values[:3] * 0.71).astype(dtype)
    scaler = StandardScaler.Config().make().fit(fit)
    if corrupt:
        scaler.mean_[:1] = np.nextafter(scaler.mean_[:1], np.inf)
    actual = {
        "n_features_in_": torch.tensor(scaler.n_features_in_),
        "n_samples_seen_": torch.from_numpy(np.asarray(scaler.n_samples_seen_)),
        "mean_": torch.from_numpy(scaler.mean_),
        "var_": torch.from_numpy(scaler.var_),
        "scale_": torch.from_numpy(scaler.scale_),
        "fit": torch.from_numpy(scaler.transform(fit)),
        "predict": torch.from_numpy(scaler.transform(predict)),
    }
    assert set(vars(scaler)) == set(actual) - {"fit", "predict"}
    prefix = f"{np.dtype(dtype).name}/{case}/"
    expected = {
        name.removeprefix(prefix): value
        for name, value in read_tensors(
            _CWD / "testdata" / "standardization.pt",
        ).items()
        if name.startswith(prefix)
    }
    report = mismatches(expected, actual=actual)
    if corrupt:
        assert any(item.startswith("mean_:") for item in report), report
    else:
        assert not report, "\n".join(report)


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
