"""The step of an MLS-Bench run: it owns nothing.

The score comes from the task's own scripts, run by the metric
(:class:`~priml.baselines.mlsbench.metric.MlsbenchSuite`), not from a model.
Every batch is ``metric_only``, so the loop never asks this step for a loss;
each loss entry point refuses loudly rather than return a fake one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from configgle import Fig


if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from priml.train.custom_types import TrainStepOutput


_EVAL_ONLY: Final = (
    "The mlsbench harness evaluates edits by running task scripts; run it with "
    "TrainLoop eval_only=True. Every batch is metric-only, so the loop never "
    "asks the step for a loss."
)


class NullStep:
    """A step that owns nothing: the score comes from the metric, not a model."""

    class Config(Fig["NullStep"]):
        """Nothing to configure."""

    def __init__(self, config: Config) -> None:
        """Store the empty config."""
        self.config = config

    @property
    def global_step(self) -> int:
        """Always zero: no update is ever taken."""
        return 0

    def preprocess_batch(self, batch: dict[str, object]) -> dict[str, object]:
        """Pass the batch through untouched."""
        return batch

    def train_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Refuse: there is no training objective."""
        del preprocessed_batch
        raise NotImplementedError(_EVAL_ONLY)

    def eval_loss(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Refuse: the score is not a loss."""
        del preprocessed_batch
        raise NotImplementedError(_EVAL_ONLY)

    def train_step(self, **preprocessed_batch: object) -> TrainStepOutput:
        """Refuse: nothing is trained."""
        del preprocessed_batch
        raise NotImplementedError(_EVAL_ONLY)

    def call_eval(self, **preprocessed_batch: object) -> Tensor:
        """Refuse: the metric runs scripts rather than a forward pass."""
        del preprocessed_batch
        raise NotImplementedError(_EVAL_ONLY)

    def on_epoch_end(self) -> None:
        """Do nothing."""

    def state_dict(self) -> Mapping[str, object]:
        """Return nothing: there is no state a resume could not rebuild."""
        return {}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Accept anything and keep nothing."""
        del state_dict
