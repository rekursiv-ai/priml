"""The dataset seam for a trainer whose environment generates its own data.

On-policy learning has no corpus: each epoch's data is whatever the policy does
next, and the train step owns the environments that produce it. What the
training loop needs from a dataset is a cadence -- a tick per epoch -- and, at
evaluation, what a fresh evaluation of the policy being trained played.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import itertools

from configgle import Fig

from priml.baselines.craftax.evaluation import MakesEvaluator
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from priml.train.custom_types import TrainStepProtocol


class CraftaxRollouts:
    """An endless cadence, one empty tick per training step; an eval plays one evaluation.

    Each tick is empty: the data lives in the environments the train step owns.
    The ticks never run out, since the run's length is the loop's
    ``max_steps`` (and the step's ``train_budget_steps``), so the whole run is
    one loop epoch and nothing epoch-driven interrupts it. A tick has no
    content and no position to resume.
    """

    class Config(Fig["CraftaxRollouts"]):
        """Nothing to configure: the step owns the data and its budget."""

    def __init__(self, config: Config) -> None:
        """Start the cadence."""
        self.config = config
        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the cadence; the loop ticks it when a pass runs out, which
        an endless one never does."""
        self._step: TrainStepProtocol | None = None

    def bind_step(self, step: TrainStepProtocol) -> None:
        """Receive the train step whose policy plays the evaluations.

        Args:
          step: The step that owns the environments and the policy.

        """
        self._step = step

    def train_dataloader(self) -> Iterator[dict[str, object]]:
        """Return the cadence: one empty tick per training step, without end."""
        return ({"valid_count": 1} for _ in itertools.count())

    def eval_dataloader(self) -> Iterator[dict[str, object]]:
        """Yield one batch: what a fresh evaluation of the policy played.

        One, because an evaluation starts afresh from its seeds and plays
        deterministically: a second would replay the first, and the score keeps
        the last. The evaluation is built, played and closed as the loop draws
        the batch, so the batch holds no environments.

        Returns:
          batches: One batch, whose ``played`` is an ``evaluation.Played``.

        Raises:
          TypeError: No bound step can build an evaluator.

        """
        step = self._step
        if not isinstance(step, MakesEvaluator):
            raise TypeError("evaluation requires a bound step with make_evaluator()")
        return _evaluation(step)

    class StateDict(TypedDict):
        """The pass count."""

        timer_epoch: CheckpointableStepTimer.StateDict

    def state_dict(self) -> StateDict:
        """Return the pass count.

        Returns:
          state: ``timer_epoch``.

        """
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the pass count.

        Args:
          state_dict: What :meth:`state_dict` returned.

        """
        state = cast(CraftaxRollouts.StateDict, state_dict)
        self.timer_epoch.load_state_dict(state["timer_epoch"])


def _evaluation(step: MakesEvaluator) -> Iterator[dict[str, object]]:
    """Play one fresh evaluation when the batch is drawn."""
    evaluator = step.make_evaluator()
    try:
        played = evaluator.play()
    finally:
        evaluator.close()
    yield {"valid_count": 1, "metric_only": True, "played": played}
