"""MLS-Bench harness experiments: one baseline edit, scored on every setting.

One experiment is one (task, baseline, seed): the baseline's edit is applied to
a pristine workspace and every setting of the task is run through an injected
:class:`~priml.baselines.mlsbench.executor.Executor`. Nothing is trained; the
loop runs with ``eval_only=True`` and ``metrics.json`` carries the
leaderboard's own column names (``eval/accuracy_<setting>``).

Every (task, baseline, seed) has its own function, ``expNNN`` in
``priml.baselines.mlsbench.tasks.<task_module>``, one module per task, written
by ``scripts/generate_experiments.py`` from ``registry.json`` (a committed
snapshot of the pinned clone: baseline names, seeds and setting time limits).
They all go through :func:`mlsbench_experiment`. The one function here,
``exp_smoke``, is a plumbing check on a synthetic task with a fake executor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, NotRequired, Self, TypedDict, cast, override

import functools
import json

from configgle import Makes

from priml.baselines.mlsbench.data import SettingBatches
from priml.baselines.mlsbench.executor import FakeExecutor
from priml.baselines.mlsbench.metric import MlsbenchSuite, budget_seconds
from priml.baselines.mlsbench.scripts.prepare_data import (
    DATASETS_DIR,
    SMOKE_DIR,
    STAGED_DIR,
    UPSTREAM_DIR,
)
from priml.baselines.mlsbench.testing import SYNTH_TASK
from priml.baselines.mlsbench.train_step import NullStep
from priml.paths import resolve_working_dir
from priml.runtime import SingleProcess
from priml.train.tracker import FileTracker
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    from configgle import Makeable

    from priml.baselines.mlsbench.executor import Executor

REGISTRY_PATH: Final = Path(__file__).with_name("registry.json")
"""Committed snapshot of the pinned clone: names and budgets, nothing else."""


@dataclass(frozen=True)
class TaskEntry:
    """What a config needs to know about one task, without the clone."""

    baselines: tuple[str, ...]
    """Leaderboard baseline names, in upstream order."""

    settings: Mapping[str, float]
    """Setting label to its time limit in seconds."""

    seeds: tuple[int, ...] = (42,)
    """Seeds the task declares (upstream default ``(42,)``); one experiment each."""

    hidden: tuple[str, ...] = ()
    """Labels of the settings hidden from an agent. Informational: every setting runs."""

    @property
    def max_eval_time(self) -> float:
        """Deterministic per-run budget: the limits summed, plus slack."""
        return round(budget_seconds(self.settings.values()), 3)


class _RegistryRow(TypedDict):
    """One task of ``registry.json``."""

    baselines: list[str]
    settings: dict[str, float]
    seeds: NotRequired[list[int]]
    hidden: NotRequired[list[str]]


@functools.cache
def _snapshot() -> dict[str, TaskEntry]:
    raw = cast("dict[str, _RegistryRow]", json.loads(REGISTRY_PATH.read_text()))
    return {
        name: TaskEntry(
            tuple(d["baselines"]),
            {k: float(v) for k, v in d["settings"].items()},
            tuple(int(x) for x in d.get("seeds", (42,))),
            tuple(str(x) for x in d.get("hidden", ())),
        )
        for name, d in raw.items()
    }


def registry() -> dict[str, tuple[str, ...]]:
    """Return task name to baseline names, from the committed snapshot.

    Pure and host-independent: it never reads the clone. An opt-in test checks
    it against one (``scripts/generate_experiments.py --registry`` rewrites it).
    """
    return {name: e.baselines for name, e in _snapshot().items()}


def task_entry(task: str) -> TaskEntry | None:
    """Return the snapshot entry of ``task``, or ``None`` for an unlisted task."""
    return _snapshot().get(task)


REGISTRY: Final[dict[str, tuple[str, ...]]] = registry()
"""Task to baseline names for all upstream tasks (see :func:`registry`)."""


class MlsbenchEval(
    Makes["TrainLoop"],
    TrainLoop.Config[NullStep.Config, SettingBatches.Config],
):
    """An eval-only loop with the null step and the per-setting batches in place.

    The upstream clone and the data directories are resources shared by every
    run, so they are fields of the root, resolved beneath ``base_dir`` (as the
    dataset's paths are) and handed to the dataset and the suite at finalize.
    The metrics' own ``base_dir`` is the run directory, which is why the suite
    cannot resolve them itself.
    """

    step: NullStep.Config = field(default_factory=NullStep.Config)
    """Owns nothing; the score comes from the metric."""

    dataset: SettingBatches.Config = field(default_factory=SettingBatches.Config)
    """One batch per setting of the task."""

    mlsbench_root: Path | str = UPSTREAM_DIR
    """Logical directory of the read-only upstream clone."""

    data_root: Path | str = DATASETS_DIR
    """Logical ``{data_root}`` of upstream's package configs (datasets, weights)."""

    staged_data: Path | str = STAGED_DIR
    """Logical ``scripts/prepare_data.py`` destination (each task's ``data/``)."""

    @override
    def finalize(self) -> Self:
        root = resolve_working_dir(self.base_dir, self.mlsbench_root)
        if not str(self.dataset.mlsbench_root):
            self.dataset.mlsbench_root = root
        for metric in self.metrics_eval.values():
            if not isinstance(metric, MlsbenchSuite.Config):
                continue
            if not str(metric.mlsbench_root):
                metric.mlsbench_root = root
            if not str(metric.data_root):
                metric.data_root = resolve_working_dir(self.base_dir, self.data_root)
            if not str(metric.staged_data):
                metric.staged_data = resolve_working_dir(
                    self.base_dir,
                    self.staged_data,
                )
            if str(metric.edit_path):
                metric.edit_path = resolve_working_dir(self.base_dir, metric.edit_path)
        return super().finalize()


def mlsbench_experiment(
    task: str,
    baseline: str,
    *,
    seed: int = 42,
    executor: Makeable[Executor] | None = None,
) -> MlsbenchEval:
    """Build the eval-only loop for one baseline of one task.

    ``max_eval_time`` is a pure function of the task's registry entry (the
    settings' limits summed, plus slack), never of the environment, so configs
    and goldens are identical on every host. A task outside the registry (a
    test fixture) gets no budget.

    Args:
      task: Task name.
      baseline: Baseline (leaderboard row) name.
      seed: Seed handed to the executor.
      executor: Injected executor; ``None`` keeps the suite's default,
        :class:`~priml.baselines.mlsbench.executor.LocalDocker`.

    Returns:
      cfg: The loop config.

    Raises:
      ValueError: ``task`` is in the registry but ``baseline`` is not.

    """
    entry = task_entry(task)
    if entry is not None and baseline not in entry.baselines:
        raise ValueError(
            f"{task}: unknown baseline {baseline!r}; have {entry.baselines}",
        )
    return _eval_loop(task, baseline, seed, executor, baseline=baseline)


def candidate_experiment(
    task: str,
    name: str,
    edit_path: Path | str,
    *,
    seed: int = 42,
    settings: tuple[str, ...] = (),
    executor: Makeable[Executor] | None = None,
) -> MlsbenchEval:
    """Build the eval-only loop for a new edit of a task, run like a baseline row.

    An agent's candidate is an experiment function of its own that returns
    this, so it runs, lands and is judged exactly as a published row does:
    ``<base_dir>/runs/mlsbench/<task>/<name>/seed<N>/metrics.json``, read by
    ``leaderboard --task <task> --row <row> --candidate <name>``.

    Args:
      task: Task name.
      name: The candidate's slug; it names the run directory.
      edit_path: Logical path of the edit script, beneath ``base_dir`` like the
        other resources (``/edits/<name>.edit.py``). Only the task's editable
        ranges may change; the workspace guard refuses anything else.
      seed: Seed handed to the executor; one of the task's published seeds.
      settings: A subset of setting labels for a screen; ``()`` runs every
        setting, which a comparison with a published row needs. A screen is
        its own candidate, under its own name.
      executor: Injected executor; ``None`` keeps
        :class:`~priml.baselines.mlsbench.executor.LocalDocker`.

    Returns:
      cfg: The loop config.

    Raises:
      ValueError: ``name`` is empty or a published baseline's, or ``edit_path``
        is empty.

    """
    entry = task_entry(task)
    if not name or (entry is not None and name in entry.baselines):
        raise ValueError(
            f"{task}: candidate name {name!r} must be new; baselines are "
            f"{entry.baselines if entry is not None else ()}",
        )
    if not str(edit_path):
        raise ValueError(f"{task}/{name}: edit_path is required.")
    cfg = _eval_loop(task, name, seed, executor, edit_path=edit_path)
    cfg.dataset.settings = settings
    return cfg


def _eval_loop(
    task: str,
    name: str,
    seed: int,
    executor: Makeable[Executor] | None,
    *,
    baseline: str = "",
    edit_path: Path | str = "",
) -> MlsbenchEval:
    """Build the loop shared by baseline and candidate runs; give exactly one edit."""
    entry = task_entry(task)
    cfg = MlsbenchEval()
    cfg.study_name = "mlsbench"
    cfg.experiment_name = f"{task}/{name}/seed{seed}"
    cfg.eval_only = True
    cfg.checkpointer = None
    cfg.tracker = FileTracker.Config()
    runtime = SingleProcess.Config()
    # Determinism is decided by the task's own scripts, not by this process.
    runtime.deterministic = False
    cfg.runtime = runtime

    cfg.dataset.task = task
    suite = MlsbenchSuite.Config()
    suite.task = task
    suite.baseline = baseline
    suite.edit_path = edit_path
    suite.seed = seed
    if executor is not None:
        suite.executor = executor
    cfg.metrics_eval[""] = suite

    if entry is not None:
        cfg.max_eval_time = entry.max_eval_time
    return cfg


SMOKE_LOGS: Final[dict[str, str]] = {
    "set-a": "accuracy=0.3220\navg_steps=256.0\n",
    "set-b": "pass_at_1=0.3902\navg_steps=256.0\n",
    "set-c": "gen_ppl=237.0499\nmauve=0.1124\nentropy=5.9261\nrep2=0.0254\navg_steps=224.0\n",
}
"""Canned ``key=value`` logs for the synthetic fixture task, for the fake executor."""


def exp_smoke() -> MlsbenchEval:
    """Check the plumbing end to end with canned logs; no container, no GPU.

    The fake executor replays ``key=value`` logs, which a real upstream
    ``parser.py`` need not accept, so the run reads a synthetic clone whose
    parser does: ``scripts/prepare_data.py --smoke`` builds it.

    Hypothesis:
      None: it measures nothing and is not a result. It fails if the harness
      cannot carry a setting from batch to ``metrics.json``.

    Returns:
      cfg: The ``base_a`` loop with a fake executor.

    References:
      ``testing.build_synth_task``: the synthetic task.

    Results:
      TBD.

    """
    fake = FakeExecutor.Config()
    fake.outputs = dict(SMOKE_LOGS)
    cfg = mlsbench_experiment(SYNTH_TASK, "base_a", executor=fake)
    cfg.experiment_name = "exp_smoke"
    cfg.mlsbench_root = SMOKE_DIR
    return cfg
