"""Tests for the grid-accuracy metric."""

from __future__ import annotations

from torch import Tensor

import pytest
import torch
import torch.distributed as dist

from priml.baselines.sudoku.metric import GridAccuracy
from priml.lib.custom_json import ReadError


def _packed(predictions: Tensor, prefix: int = 1) -> Tensor:
    """Pack predictions behind ``prefix`` diagnostic columns."""
    lead = torch.zeros(predictions.shape[0], prefix)
    return torch.cat([lead, predictions.float()], dim=-1)


def test_one_wrong_cell_fails_the_whole_puzzle() -> None:
    """A puzzle is solved or it is not; 80 of 81 is a wrong answer."""
    metric = GridAccuracy.Config().make()
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    predictions = labels.clone()
    predictions[1, 0] = 5
    metric.update(_packed(predictions), label=labels, valid_count=len(labels))
    result = metric.compute()
    assert result["exact"] == 0.5
    assert result["cell"] == 17 / 18


def test_grid_is_read_from_the_end() -> None:
    """Leading diagnostic columns must not shift the prediction window."""
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    scores = [
        _score(_packed(labels.clone(), prefix=width), labels) for width in (1, 5, 40)
    ]
    assert scores == [1.0, 1.0, 1.0]


def _score(packed: Tensor, labels: Tensor) -> float:
    metric = GridAccuracy.Config().make()
    metric.update(packed, label=labels, valid_count=len(labels))
    return metric.compute()["exact"]


def test_padding_counts_for_neither_side() -> None:
    """Rows squaring off a short batch must not be scored as solved or failed."""
    metric = GridAccuracy.Config().make()
    labels = torch.full((4, 9), 3, dtype=torch.int64)
    labels[2:] = -100  # The padded tail.
    metric.update(_packed(labels.clone()), label=labels, valid_count=len(labels))
    assert metric.compute()["exact"] == 1.0
    puzzles = metric.puzzles
    assert puzzles == 2


def test_valid_count_truncates_before_scoring() -> None:
    metric = GridAccuracy.Config().make()
    labels = torch.full((4, 9), 3, dtype=torch.int64)
    predictions = labels.clone()
    predictions[2:] = 7  # `wrong`, but past the valid rows.
    metric.update(_packed(predictions), label=labels, valid_count=2)
    assert metric.compute()["exact"] == 1.0


def test_valid_count_is_required_because_loader_pad_rows_are_zero() -> None:
    """The loaders pad with label 0, not ``ignore_label_id``; only the count knows."""
    metric = GridAccuracy.Config().make()
    labels = torch.full((4, 2), 3, dtype=torch.int64)
    labels[2:] = 0  # The loaders' pad rows.
    with pytest.raises(KeyError, match="valid_count"):
        metric.update(_packed(labels), label=labels)
    metric.update(_packed(labels), label=labels, valid_count=2)
    assert metric.puzzles == 2
    assert metric.compute()["exact"] == 1.0


def test_counts_accumulate_across_batches() -> None:
    """Ratios are computed once at the end, not averaged per batch."""
    metric = GridAccuracy.Config().make()
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    metric.update(
        _packed(labels.clone()),
        label=labels,
        valid_count=len(labels),
    )  # Solved.
    wrong = labels.clone()
    wrong[0, -1] = 5
    metric.update(_packed(wrong), label=labels, valid_count=len(labels))  # Not solved.
    metric.update(_packed(wrong), label=labels, valid_count=len(labels))  # Not solved.
    assert metric.state_dict() == {
        "solved": 4,
        "puzzles": 6,
        "cells_correct": 52,
        "cells": 54,
    }
    assert metric.compute() == {"exact": 2 / 3, "cell": 26 / 27}


def test_update_casts_predictions_and_labels_to_integer() -> None:
    metric = GridAccuracy.Config().make()
    labels = torch.full((2, 3), 3.9)
    logits = _packed(torch.full((2, 3), 3.1))
    metric.update(logits, label=labels, valid_count=len(labels))
    assert metric.compute() == {"exact": 1.0, "cell": 1.0}


def test_update_moves_labels_to_prediction_device() -> None:
    metric = GridAccuracy.Config().make()
    labels = torch.full((2, 3), 3, dtype=torch.int64)
    with pytest.raises(RuntimeError, match=r"item.*meta"):
        metric.update(_packed(labels).to("meta"), label=labels, valid_count=len(labels))


def test_empty_metric_reports_zero_not_a_division_error() -> None:
    assert GridAccuracy.Config().make().compute() == {"exact": 0.0, "cell": 0.0}


def test_single_counted_cell_can_solve_a_puzzle() -> None:
    metric = GridAccuracy.Config().make()
    labels = torch.tensor([[4, -100, -100], [-100, -100, -100]])
    metric.update(
        _packed(labels.clone(), prefix=0),
        label=labels,
        valid_count=len(labels),
    )
    assert metric.state_dict() == {
        "solved": 1,
        "puzzles": 1,
        "cells_correct": 1,
        "cells": 1,
    }
    assert metric.compute() == {"exact": 1.0, "cell": 1.0}


def test_state_round_trips() -> None:
    metric = GridAccuracy.Config().make()
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    metric.update(_packed(labels.clone()), label=labels, valid_count=len(labels))
    state = metric.state_dict()
    assert state == {
        "solved": 2,
        "puzzles": 2,
        "cells_correct": 18,
        "cells": 18,
    }

    restored = GridAccuracy.Config().make()
    restored.load_state_dict(state)
    assert restored.compute() == metric.compute()


def test_state_defaults_and_numeric_coercion() -> None:
    metric = GridAccuracy.Config().make()
    metric.load_state_dict({})
    assert metric.state_dict() == {
        "solved": 0,
        "puzzles": 0,
        "cells_correct": 0,
        "cells": 0,
    }

    metric.load_state_dict({"solved": "2", "puzzles": 4.0})
    assert metric.state_dict() == {
        "solved": 2,
        "puzzles": 4,
        "cells_correct": 0,
        "cells": 0,
    }


@pytest.mark.parametrize("corrupt", ["two", None, [2]])
def test_corrupt_state_counts_raise(corrupt: object) -> None:
    metric = GridAccuracy.Config().make()
    with pytest.raises(ReadError):
        metric.load_state_dict({"solved": corrupt})


def test_compute_reduces_initialized_gloo_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = GridAccuracy.Config().make()
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    metric.update(_packed(labels), label=labels, valid_count=len(labels))

    def fake_all_reduce(counts: Tensor, **kwargs: object) -> None:
        assert counts.dtype == torch.float64
        assert kwargs == {"op": dist.ReduceOp.SUM}

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    assert metric.compute() == {"exact": 1.0, "cell": 1.0}


def test_reset_clears_every_count() -> None:
    metric = GridAccuracy.Config().make()
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    metric.update(_packed(labels.clone()), label=labels, valid_count=len(labels))
    metric.reset()
    assert metric.compute() == {"exact": 0.0, "cell": 0.0}


def test_compute_moves_counts_to_the_current_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metric = GridAccuracy.Config().make()
    labels = torch.full((2, 9), 3, dtype=torch.int64)
    metric.update(_packed(labels), label=labels, valid_count=len(labels))
    device_calls: list[tuple[object, ...]] = []
    original_device = torch.device

    def device(*args: object) -> torch.device:
        device_calls.append(args)
        assert args == ("cuda", 2)
        return original_device("cpu")

    def fake_all_reduce(counts: Tensor, *, op: dist.ReduceOp) -> None:
        assert counts.device == original_device("cpu")
        assert op == dist.ReduceOp.SUM

    with monkeypatch.context() as scoped:
        scoped.setattr(
            "priml.baselines.sudoku.metric.dist.is_initialized",
            lambda: True,
        )
        scoped.setattr(
            "priml.baselines.sudoku.metric.dist.get_backend",
            lambda: "nccl",
        )
        scoped.setattr(
            "priml.baselines.sudoku.metric.dist.all_reduce",
            fake_all_reduce,
        )
        scoped.setattr(
            "priml.baselines.sudoku.metric.torch.cuda.current_device",
            lambda: 2,
        )
        scoped.setattr("priml.baselines.sudoku.metric.torch.device", device)
        assert metric.compute() == {"exact": 1.0, "cell": 1.0}

    assert device_calls == [("cuda", 2)]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
