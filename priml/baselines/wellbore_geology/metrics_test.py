"""Tests for pooled scored-row RMSE and exact metric state."""

from collections.abc import Sequence
from typing import override

from numpy.typing import NDArray

import numpy as np
import pytest
import torch

from priml.baselines.wellbore_geology.data import (
    ExpandPrediction,
    PredictionRows,
)
from priml.baselines.wellbore_geology.metrics import PooledRMSE


def _metric() -> PooledRMSE:
    config = PooledRMSE.Config()
    config.expansion.prefix_rows = 0
    config.expansion.target_rows = 3
    config.expansion.downsample = 1
    config.expansion.target_stats = (0.0, 1.0)
    return config.make()


def _meta(count: int, target: list[float]) -> dict[str, object]:
    return {
        "well_id": "well",
        "suffix_len": count,
        "tvt0": 0.0,
        "submit_indices": np.arange(count),
        "suffix_z": np.zeros(count, dtype=np.float32),
        "TVT": np.asarray(target, dtype=np.float32),
    }


def test_rmse_pools_rows_instead_of_averaging_batches() -> None:
    metric = _metric()
    metric.update(
        torch.tensor([[2.0, 4.0, 2.0]]),
        meta=[_meta(3, [1.0, 2.0, 5.0])],
    )
    saved = metric.state_dict()
    metric.reset()
    with pytest.raises(ValueError, match="scored rows"):
        metric.compute()
    metric.load_state_dict(saved)
    metric.update(
        torch.tensor([[7.0, 0.0, 9.0], [9.0, 5.0, 5.0]]),
        meta=[_meta(2, [7.0, 4.0]), _meta(1, [9.0])],
    )
    expected = ((1 + 4 + 9 + 0 + 16 + 0) / 6) ** 0.5
    assert metric.compute() == {"rmse": expected}
    assert len(saved["predictions"]) == 1
    metric.reset()
    assert metric.state_dict() == {"predictions": [], "labels": []}


def test_metric_detaches_inputs_and_rejects_mismatched_wells() -> None:
    metric = _metric()
    prediction = torch.tensor([[1.0, 3.0, 0.0]], requires_grad=True)
    metric.update(prediction, meta=[_meta(2, [2.0, 1.0])])
    assert not metric.predictions[0].requires_grad
    assert metric.compute() == {"rmse": (5 / 2) ** 0.5}
    with pytest.raises(ValueError, match="zip"):
        metric.update(prediction, meta=[_meta(2, [2.0, 1.0])] * 2)


def test_metric_rejects_a_split_without_scored_rows() -> None:
    metric = _metric()
    metric.update(torch.tensor([[1.0, 2.0, 3.0]]), meta=[_meta(0, [])])
    with pytest.raises(ValueError, match="scored rows"):
        metric.compute()


def test_expansion_without_truth_is_rejected() -> None:
    class Untargeted(ExpandPrediction):
        @override
        def __call__(
            self,
            prediction: NDArray[np.float32],
            *,
            metadata: Sequence[dict[str, object]],
            include_target: bool,
        ) -> PredictionRows:
            del include_target
            return super().__call__(prediction, metadata=metadata, include_target=False)

    metric = _metric()
    metric.expansion = Untargeted(metric.expansion.config)
    with pytest.raises(ValueError, match=r"rows\.target"):
        metric.update(torch.zeros(2, 3), meta=[_meta(1, [0.0]), _meta(1, [0.0])])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
