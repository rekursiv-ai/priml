from __future__ import annotations

from priml.custom_types import MetricObjective


def test_metric_objective_minimize_is_better():
    obj = MetricObjective(metric_key="eval/total_loss", direction="minimize")
    assert obj.is_better(1.0, 2.0)
    assert not obj.is_better(2.0, 1.0)
    assert not obj.is_better(1.0, 1.0)


def test_metric_objective_maximize_is_better():
    obj = MetricObjective(metric_key="eval/roc_auc", direction="maximize")
    assert obj.is_better(2.0, 1.0)
    assert not obj.is_better(1.0, 2.0)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
