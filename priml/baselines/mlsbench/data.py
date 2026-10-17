"""One ``metric_only`` batch per setting of an MLS-Bench task.

Nothing is loaded: a batch only names a setting, and the metric
(:class:`~priml.baselines.mlsbench.metric.MlsbenchSuite`) runs it. The task is
read from the upstream clone when the pass starts, so the batches follow the
task's own setting order.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from configgle import Fig

from priml.baselines.mlsbench.task import TaskSpec, load_task
from priml.timer import CheckpointableStepTimer


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class SettingBatches:
    """One ``metric_only`` batch per setting, in the task's order."""

    class Config(Fig["SettingBatches"]):
        """Which task, and which of its settings."""

        task: str = ""
        """Task name; required."""

        mlsbench_root: Path | str = ""
        """The read-only upstream clone; required. ``MlsbenchEval`` fills it from
        its own logical ``mlsbench_root`` at finalize."""

        settings: tuple[str, ...] = ()
        """Subset by label; ``()`` is every setting, hidden ones included."""

    def __init__(self, config: Config) -> None:
        """Store the selection; the task is read when the pass starts."""
        self.config = config
        self.timer_epoch = CheckpointableStepTimer()
        """Passes over the settings. An eval-only run makes exactly one."""

    def bind_step(self, step: object) -> None:
        """Accept the step; batches do not depend on it."""
        del step

    def labels(self, task: TaskSpec) -> tuple[str, ...]:
        """Return the selected labels in task order.

        Args:
          task: The parsed task.

        Returns:
          labels: The settings this pass runs.

        Raises:
          TaskSpecError: A selected label is not a setting of the task.

        """
        if not self.config.settings:
            return tuple(s.label for s in task.settings)
        for label in self.config.settings:
            task.setting(label)
        return tuple(s.label for s in task.settings if s.label in self.config.settings)

    def train_dataloader(self) -> Iterator[dict[str, object]]:
        """Refuse: nothing here is trained.

        Returns:
          batches: Never; it raises.

        Raises:
          NotImplementedError: Always.

        """
        raise NotImplementedError(
            "SettingBatches has no training stream; run with eval_only=True.",
        )

    def eval_dataloader(self) -> Iterator[dict[str, object]]:
        """Yield one batch per setting.

        Returns:
          batches: ``{"metric_only": True, "setting": label}`` per setting.

        """
        if not str(self.config.mlsbench_root):
            raise ValueError("SettingBatches needs mlsbench_root, the upstream clone.")
        task = load_task(self.config.task, root=Path(self.config.mlsbench_root))
        batches: list[dict[str, object]] = [
            {"valid_count": 1, "metric_only": True, "setting": label}
            for label in self.labels(task)
        ]
        return iter(batches)

    class StateDict(TypedDict):
        """Checkpointed loader state: the pass count."""

        timer_epoch: CheckpointableStepTimer.StateDict

    def state_dict(self) -> StateDict:
        """Return the pass count."""
        return {"timer_epoch": self.timer_epoch.state_dict()}

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore the pass count."""
        state = cast("SettingBatches.StateDict", state_dict)
        self.timer_epoch.load_state_dict(state["timer_epoch"])
