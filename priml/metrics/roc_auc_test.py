"""Verify whole-split ROC AUC, dtype preservation and accumulator ownership."""

from __future__ import annotations

import pytest
import torch

from priml.metrics.roc_auc import MacroRocAuc


def test_macro_auc_matches_official_reduction_with_tied_predictions() -> None:
    # Three findings differ from the four study rows, exposing axis mistakes.
    labels = torch.tensor([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]])
    predictions = torch.tensor(
        [[0.2, 0.1, 0.2], [0.5, 0.4, 0.2], [0.5, 0.8, 0.3], [0.3, 0.9, 0.4]],
        requires_grad=True,
    )
    metric = MacroRocAuc.Config().make()
    metric.update(predictions, label=labels)

    expected = (0.625 + 0.75 + 0.0) / 3
    assert metric.compute() == {"macro_auc": expected}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_auc_rejects_nonfinite_predictions(value: float) -> None:
    metric = MacroRocAuc.Config().make()
    metric.update(
        torch.tensor([[value, 0.1], [0.5, 0.2]]),
        label=torch.tensor([[0, 1], [1, 0]]),
    )
    with pytest.raises(ValueError, match="finite"):
        metric.compute()


def test_auc_rejects_nonbinary_and_empty_studies() -> None:
    metric = MacroRocAuc.Config().make()
    metric.update(torch.zeros(2, 3), label=torch.full((2, 3), 0.3))
    with pytest.raises(ValueError, match="binary"):
        metric.compute()
    metric.reset()
    metric.update(torch.zeros(0, 3), label=torch.zeros(0, 3))
    with pytest.raises(ValueError, match="nonempty"):
        metric.compute()


def test_macro_auc_accumulates_whole_split_before_scoring() -> None:
    labels = torch.tensor([[0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]])
    predictions = torch.tensor(
        [[0.1, 0.7, 0.8], [0.2, 0.9, 0.7], [0.3, 0.4, 0.2], [0.4, 0.2, 0.1]],
    )
    metric = MacroRocAuc.Config().make()
    metric.update(predictions[:2], label=labels[:2])
    metric.update(predictions[2:], label=labels[2:])

    assert metric.compute() == {"macro_auc": (1.0 + 1.0 + 0.0) / 3}


def test_macro_auc_checkpoints_detached_storage_and_reset() -> None:
    labels = torch.tensor([[0, 1, 0], [1, 0, 1]])
    predictions = torch.tensor([[0.1, 0.8, 0.9], [0.9, 0.2, 0.1]], requires_grad=True)
    metric = MacroRocAuc.Config().make()
    metric.update(predictions, label=labels)
    snapshot = metric.state_dict()
    restored = MacroRocAuc.Config().make()
    restored.load_state_dict(snapshot)
    metric.reset()

    assert metric.state_dict() == {"labels": [], "predictions": []}
    assert restored.compute() == {"macro_auc": (1.0 + 1.0 + 0.0) / 3}
    assert snapshot["predictions"][0].grad_fn is None
    assert snapshot["predictions"][0].device.type == "cpu"
    restored.reset()
    assert len(snapshot["labels"]) == len(snapshot["predictions"]) == 1


def test_macro_auc_rejects_an_empty_validation_pass() -> None:
    metric = MacroRocAuc.Config().make()
    with pytest.raises(ValueError, match="without study predictions"):
        metric.compute()


def test_macro_auc_owns_batches_after_model_buffers_are_reused() -> None:
    labels = torch.tensor([[0, 1, 0], [1, 0, 1]])
    predictions = torch.tensor([[0.1, 0.8, 0.9], [0.9, 0.2, 0.1]])
    metric = MacroRocAuc.Config().make()
    metric.update(predictions, label=labels)
    labels.zero_()
    predictions.zero_()

    assert metric.compute() == {"macro_auc": (1.0 + 1.0 + 0.0) / 3}


def test_macro_auc_preserves_float64_rank_distinctions() -> None:
    labels = torch.tensor([[0, 1, 0], [1, 0, 1]])
    predictions = torch.tensor(
        [[0.5, 0.500_000_001, 0.5], [0.500_000_001, 0.5, 0.500_000_001]],
        dtype=torch.float64,
    )
    metric = MacroRocAuc.Config().make()
    metric.update(predictions, label=labels)

    assert metric.compute() == {"macro_auc": 1.0}


@pytest.mark.parametrize("labels_bfloat16", [True, False])
def test_macro_auc_accepts_bfloat16_scores_and_labels(labels_bfloat16: bool) -> None:
    labels = torch.tensor([[0, 1, 0], [1, 0, 1]])
    scores = torch.tensor([[0.1, 0.9, 0.1], [0.9, 0.1, 0.9]], dtype=torch.bfloat16)
    metric = MacroRocAuc.Config().make()
    metric.update(scores, label=labels.bfloat16() if labels_bfloat16 else labels)
    assert metric.compute() == {"macro_auc": 1.0}


def test_macro_auc_requires_tensor_labels() -> None:
    metric = MacroRocAuc.Config().make()
    with pytest.raises(AssertionError):
        metric.update(torch.zeros(4, 3), label=[[0, 1, 0]])


@pytest.mark.parametrize("prediction_shape", [(4,), (4, 2), (4, 3, 2)])
def test_macro_auc_rejects_misaligned_study_predictions(
    prediction_shape: tuple[int, ...],
) -> None:
    metric = MacroRocAuc.Config().make()
    with pytest.raises(ValueError, match="same study-by-finding shape"):
        metric.update(torch.zeros(prediction_shape), label=torch.zeros(4, 3))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
