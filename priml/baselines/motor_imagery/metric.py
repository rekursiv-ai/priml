"""Retain source window selection and report official trial-level accuracy."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from configgle import Fig
from torch import Tensor
from torch.nn import functional
from treekle import from_plain

import numpy as np
import torch


if TYPE_CHECKING:
    from collections.abc import Mapping


class EEGAccuracy:
    """Select the first best validation epoch by source window argmax accuracy."""

    class Config(Fig["EEGAccuracy"]):
        """The official accuracy metric has no adjustable parameters."""

    def __init__(self, config: Config) -> None:
        del config
        self.best_accuracy = 0.0
        self.best_correct = 0
        self.best_probabilities = torch.empty(0, device="cpu")
        self.best_labels = torch.empty(0, dtype=torch.long, device="cpu")
        self.best_trials = torch.empty(0, dtype=torch.long, device="cpu")
        self.reset()

    def reset(self) -> None:
        """Start an evaluation without discarding the best earlier epoch."""
        self.probabilities: list[Tensor] = []
        self.predictions: list[Tensor] = []
        self.labels: list[Tensor] = []
        self.trials: list[Tensor] = []

    def update(
        self,
        logits: Tensor,
        *,
        label: object = None,
        trial: object = None,
        **kwargs: object,
    ) -> None:
        """Accumulate the reference's softmax probabilities and argmax labels.

        Args:
          logits: Class logits in window order.
          label: Window ground-truth labels.
          trial: Cue identifiers repeated over windows.
          **kwargs: Additional fields from the Priml metric bus.

        """
        del kwargs
        assert isinstance(label, Tensor)
        assert isinstance(trial, Tensor)
        self.probabilities.append(
            functional.softmax(logits, dim=1, dtype=torch.float32)[:, 1].cpu(),
        )
        self.predictions.append(logits.argmax(dim=1).cpu())
        self.labels.append(label.cpu())
        self.trials.append(trial.cpu())

    def compute(self) -> dict[str, float]:
        """Report source window accuracy and official accuracy after cue pooling.

        Returns:
          scores: Window argmax accuracy and mean-probability trial accuracy.

        """
        if not self.labels:
            return {"window_accuracy": 0.0, "accuracy": 0.0}
        probabilities = torch.cat(self.probabilities)
        labels = torch.cat(self.labels)
        trials = torch.cat(self.trials)
        correct = int((torch.cat(self.predictions) == labels).sum())
        accuracy = correct / len(labels)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_correct = correct
            self.best_probabilities = probabilities.clone()
            self.best_labels = labels.clone()
            self.best_trials = trials.clone()
        trial_accuracy = pooled_accuracy(probabilities, labels=labels, trials=trials)
        return {"window_accuracy": accuracy, "accuracy": trial_accuracy}

    def state_dict(self) -> Mapping[str, object]:
        """Return best epoch evidence for inspection or checkpoint serialization.

        Returns:
          state: Best accuracy, probabilities, labels, and cue identifiers.

        """
        return {
            "best_accuracy": self.best_accuracy,
            "best_correct": self.best_correct,
            "best_probabilities": self.best_probabilities,
            "best_labels": self.best_labels,
            "best_trials": self.best_trials,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore best epoch evidence."""
        self.best_accuracy = cast(float, state_dict["best_accuracy"])
        self.best_correct = cast(int, state_dict["best_correct"])
        self.best_probabilities = cast(Tensor, state_dict["best_probabilities"])
        self.best_labels = cast(Tensor, state_dict["best_labels"])
        self.best_trials = cast(Tensor, state_dict["best_trials"])


def pooled_accuracy(probabilities: Tensor, *, labels: Tensor, trials: Tensor) -> float:
    """Apply official binary accuracy after averaging windows of each cue.

    Args:
      probabilities: Window P(move), in original order.
      labels: Window ground truth, repeated per cue.
      trials: Cue identifiers, repeated per window.

    Returns:
      accuracy: Correct trial labels divided by the number of observed cues.

    """
    unique = sorted(set(from_plain(trials.tolist(), list[int])))
    correct = 0
    for identifier in unique:
        mask = trials == identifier
        probability = cast(float, np.mean(probabilities[mask].numpy()))
        correct += int((probability >= 0.5) == bool(labels[mask][0]))
    return correct / len(unique)
