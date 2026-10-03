"""Tests for ETTh1 validation metrics."""

import numpy as np
import pytest
import torch

from priml.baselines.etth1.metrics import ForecastMSE


def test_source_float32_reduction_and_restore() -> None:
    metric = ForecastMSE.Config().make()
    expected: list[float] = []
    for value in (1.0, 0.1, 0.03):
        prediction = torch.full((2, 3, 4), value)
        target = torch.zeros_like(prediction)
        metric.update(prediction, label=target)
        expected.append(torch.nn.functional.mse_loss(prediction, target).item())
    state = metric.state_dict()
    metric.reset()
    with pytest.raises(ValueError, match="without complete batches"):
        metric.compute()
    metric.load_state_dict(state)
    assert metric.compute()["total_loss"] == float(
        np.average(np.asarray(expected, dtype=np.float32)),
    )
