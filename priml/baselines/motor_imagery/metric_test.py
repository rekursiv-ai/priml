"""Official trial accuracy and public-reference best-epoch tie behavior."""

from __future__ import annotations

import torch

from priml.baselines.motor_imagery.metric import EEGAccuracy, pooled_accuracy


def test_window_and_trial_accuracy_have_the_correct_denominators() -> None:
    metric = EEGAccuracy.Config().make()
    assert metric.compute() == {"accuracy": 0.0, "window_accuracy": 0.0}
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 0, 1])
    trials = torch.tensor([4, 4, 5])
    metric.update(logits, label=labels, trial=trials)
    result = metric.compute()
    assert result["window_accuracy"] == 2 / 3
    assert result["accuracy"] == 1 / 2
    assert metric.best_labels.tolist() == [0, 0, 1]
    assert metric.best_correct == 2


def test_bfloat16_logits_keep_float32_trial_probability_reductions() -> None:
    metric = EEGAccuracy.Config().make()
    logits = torch.tensor([[0, 1], [2, 0]], dtype=torch.bfloat16)
    metric.update(logits, label=torch.tensor([1, 0]), trial=torch.tensor([1, 2]))
    expected = logits.float().softmax(dim=1)[:, 1]
    assert metric.probabilities[0].dtype == torch.float32
    assert torch.equal(metric.probabilities[0], expected)
    assert metric.compute() == {"window_accuracy": 1.0, "accuracy": 1.0}


def test_best_epoch_keeps_first_on_ties_and_roundtrips_state() -> None:
    metric = EEGAccuracy.Config().make()
    labels = torch.tensor([0, 1, 0])
    trials = torch.tensor([2, 3, 4])
    metric.update(
        torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]),
        label=labels,
        trial=trials,
    )
    metric.compute()
    first = metric.best_probabilities.clone()
    metric.reset()
    metric.update(
        torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]]),
        label=labels,
        trial=trials,
    )
    metric.compute()
    assert torch.equal(first, metric.best_probabilities)
    restored = EEGAccuracy.Config().make()
    restored.load_state_dict(metric.state_dict())
    assert restored.best_accuracy == 1.0
    assert restored.best_correct == 3
    assert torch.equal(restored.best_probabilities, first)
    assert torch.equal(restored.best_labels, labels)
    assert torch.equal(restored.best_trials, trials)


def test_trial_probabilities_use_the_source_threshold_tie() -> None:
    probability = torch.tensor([0.25, 0.75, 0.1])
    assert (
        pooled_accuracy(
            probability,
            labels=torch.tensor([1, 1, 0]),
            trials=torch.tensor([2, 2, 3]),
        )
        == 1.0
    )


def test_zero_accuracy_epochs_never_select_probabilities_before_improvement() -> None:
    metric = EEGAccuracy.Config().make()
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([1, 0])
    trials = torch.tensor([2, 3])
    metric.update(logits, label=labels, trial=trials)
    assert metric.compute() == {"accuracy": 0.0, "window_accuracy": 0.0}
    assert metric.best_accuracy == 0.0
    assert metric.best_correct == 0
    assert metric.best_labels.numel() == 0
    assert metric.best_trials.numel() == 0
    assert metric.best_probabilities.numel() == 0
    metric.reset()
    metric.update(logits * 2, label=labels, trial=trials)
    metric.compute()
    assert metric.best_probabilities.numel() == 0
    metric.reset()
    metric.update(-logits, label=labels, trial=trials)
    assert metric.compute() == {"accuracy": 1.0, "window_accuracy": 1.0}
    assert metric.best_accuracy == 1.0
    assert torch.equal(metric.best_labels, labels)
    assert torch.equal(metric.best_trials, trials)
    assert torch.equal(metric.best_probabilities, (-logits).softmax(dim=1)[:, 1])


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
