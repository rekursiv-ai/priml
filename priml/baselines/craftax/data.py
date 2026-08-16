"""The dataset seam for an environment that generates its own data.

On-policy learning has no corpus: the next batch is whatever the current
policy does next, so there is nothing to load and nothing to shuffle. What the
training loop needs from a dataset is a cadence -- something to iterate that
says "take another step" -- and that is all this provides.

The training step owns the environment and the rollout. This hands it the
loop's tick, and receives the step through ``bind_step`` so an evaluation pass
can score the policy that is actually training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from configgle import Fig

from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import nn

    from priml.train.custom_types import TrainStepProtocol


@runtime_checkable
class _SupportsPolicy(Protocol):
    """A training step whose network can be scored directly."""

    model: nn.Module


class CraftaxRollouts:
    """A cadence of updates, one batch per training step.

    Each yielded batch is empty: the data lives in the environment the train
    step owns. The count of batches is what sets the epoch length.
    """

    class Config(Fig["CraftaxRollouts"]):
        """How many updates make an epoch."""

        updates_per_epoch: int = 1_000
        """Training steps between epoch boundaries.

        Only the epoch-driven parts of the loop see this -- the run's real
        budget is its step count. Set it large enough that epoch bookkeeping
        does not interrupt a run."""

        eval_batches: int = 1
        """Evaluation passes per eval; each scores one fresh rollout."""

    def __init__(self, config: Config) -> None:
        """Store the cadence.

        Args:
          config: Epoch and evaluation lengths.

        Raises:
          ValueError: A count is not positive.

        """
        if config.updates_per_epoch <= 0 or config.eval_batches <= 0:
            raise ValueError("Rollout cadence must be positive")
        self.config = config
        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the cadence; ticked by the loop when it runs out."""

        self._step: TrainStepProtocol | None = None

    def bind_step(self, step: TrainStepProtocol) -> None:
        """Receive the training step whose policy generates the data.

        Called once by the training loop before the first batch is drawn.

        Args:
          step: The step that owns the environment and the policy.

        """
        self._step = step

    def train_dataloader(self) -> Iterator[dict[str, Any]]:
        """Yield one tick per update in an epoch."""
        return iter([{"valid_count": 1} for _ in range(self.config.updates_per_epoch)])

    def eval_dataloader(self) -> Iterator[dict[str, Any]]:
        """Yield one tick per evaluation pass, carrying the live policy.

        The score is a property of the policy, not of any batch of
        transitions, so the network itself travels in the batch and the metric
        plays its own episodes with it.
        """
        batch: dict[str, Any] = {"valid_count": 1}
        if isinstance(self._step, _SupportsPolicy):
            batch["policy"] = self._step.model
        return iter([dict(batch) for _ in range(self.config.eval_batches)])

    def state_dict(self) -> dict[str, Any]:
        """Return the pass count; the ENVIRONMENT carries the resumable state."""
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the pass count; there is no dataset position to restore."""
        if "timer_epoch" in state_dict:
            self.timer_epoch.load_state_dict(state_dict["timer_epoch"])
