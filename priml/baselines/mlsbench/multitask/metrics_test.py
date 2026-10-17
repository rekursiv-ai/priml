"""Tests for the per-head accuracy."""

from __future__ import annotations

import pytest
import torch

from priml.baselines.mlsbench.multitask.metrics import HeadAccuracy


def metric(start: int, stop: int, key: str) -> HeadAccuracy:
    """Build an accuracy over columns ``[start, stop)`` scored against ``key``."""
    config = HeadAccuracy.Config()
    config.logits_start = start
    config.logits_stop = stop
    config.label_key = key
    made = config.make()
    assert isinstance(made, HeadAccuracy)
    return made


def test_each_head_reads_its_own_columns_and_labels() -> None:
    # Columns 0-2 are the fine head, 3-4 the coarse head.
    logits = torch.tensor([[0.0, 5.0, 1.0, 2.0, 0.0], [3.0, 0.0, 1.0, 0.0, 4.0]])
    batch = {"label": torch.tensor([1, 2]), "label_coarse": torch.tensor([0, 1])}
    fine = metric(0, 3, "label")
    coarse = metric(3, 5, "label_coarse")
    fine.update(logits, **batch)
    coarse.update(logits, **batch)
    assert fine.compute() == {"final": 50.0, "best": 50.0}
    assert coarse.compute() == {"final": 100.0, "best": 100.0}


def test_an_empty_span_is_rejected() -> None:
    with pytest.raises(ValueError, match="logits_stop"):
        metric(3, 3, "label")


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
