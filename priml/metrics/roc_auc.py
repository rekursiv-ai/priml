"""Accumulate per-target ROC AUC over a complete validation pass."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

from configgle import Fig
from torch import Tensor

import numpy as np
import torch

from priml.math.stats import binary_roc_auc


if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


class MacroRocAuc:
    """Report the unweighted mean ROC AUC across target columns."""

    class Config(Fig["MacroRocAuc"]):
        """Accumulate confidence scores without transforming their ranks."""

    class StateDict(TypedDict):
        """Target labels and confidence scores accumulated across batches."""

        labels: list[Tensor]
        predictions: list[Tensor]

    def __init__(self, config: Config) -> None:
        del config
        self.labels: list[Tensor] = []
        self.predictions: list[Tensor] = []

    def update(self, logits: Tensor, **batch: object) -> None:
        """Accumulate confidence scores and the corresponding target labels.

        Args:
          logits: Confidence scores, with examples on rows and targets on columns.
          **batch: Batch containing the matching example-by-target ``label`` tensor.

        """
        labels = batch["label"]
        assert isinstance(labels, Tensor)
        if logits.ndim != 2 or logits.shape != labels.shape:
            raise ValueError(
                "Labels and predictions need the same study-by-finding shape.",
            )
        # Evaluation can reuse model output buffers on the next batch.
        self.labels.append(labels.detach().to(device="cpu", copy=True))
        self.predictions.append(logits.detach().to(device="cpu", copy=True))

    def compute(self) -> dict[str, float]:
        """Compute macro ROC AUC after collecting the complete validation set.

        Returns:
          metrics: The unweighted mean target AUC, keyed by ``macro_auc``.

        """
        if not self.predictions:
            raise ValueError("Cannot compute macro ROC AUC without study predictions.")
        accumulated_labels = torch.cat(self.labels)
        labels: NDArray[np.float64] = np.asarray(
            _numpy_compatible(accumulated_labels).numpy(),
            dtype=np.float64,
        )
        predictions: NDArray[np.float64] = np.asarray(
            _numpy_compatible(torch.cat(self.predictions)).numpy(),
            dtype=np.float64,
        )
        if not labels.size:
            raise ValueError("ROC AUC requires nonempty study predictions.")
        if np.count_nonzero(np.isfinite(predictions)) != predictions.size:
            raise ValueError("ROC AUC requires finite prediction scores.")
        if np.count_nonzero(
            np.logical_and(np.not_equal(labels, 0), np.not_equal(labels, 1)),
        ):
            raise ValueError("ROC AUC requires binary finding labels.")
        values = np.array(
            [
                float(
                    binary_roc_auc(
                        np.equal(labels[:, finding], 1),
                        scores=predictions[:, finding],
                    ),
                )
                for finding in range(accumulated_labels.shape[1])
            ],
        )
        return {"macro_auc": float(values.mean())}

    def reset(self) -> None:
        """Discard the previous validation pass."""
        self.labels.clear()
        self.predictions.clear()

    def state_dict(self) -> StateDict:
        """Snapshot accumulated labels and predictions."""
        return {"labels": list(self.labels), "predictions": list(self.predictions)}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore an accumulated validation pass."""
        state = cast(MacroRocAuc.StateDict, state_dict)
        self.labels = list(state["labels"])
        self.predictions = list(state["predictions"])


def _numpy_compatible(value: Tensor) -> Tensor:
    return value.float() if value.dtype == torch.bfloat16 else value
