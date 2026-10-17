"""Run a task's settings inside ``TrainLoop(eval_only=True)`` and publish scores.

One ``metric_only`` batch per unit of work, the metric does the work in
``update()``, and ``FileTracker`` writes ``metrics.json``. Here the unit of
work is a *setting*. ``update()`` runs the setting's script through an injected
:class:`Executor`, hands the log to the task's own ``parser.py`` (loaded as-is
from the upstream clone, never copied), and remembers the result. ``compute()``
publishes ``<metric>_<setting>``, the leaderboard's own column names, for every
setting that produced a number.

A setting that timed out, crashed, or was not parseable publishes *nothing*
rather than NaN: the verdict layer reports the absent key, and an absent key
can never be mistaken for a measurement.

What a setting runs under, and how its log is parsed, is :mod:`.setting`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self, TypedDict, cast, override

import json
import shutil

from configgle import Fig, Makeable

from priml.baselines.mlsbench.executor import (
    OUTPUT_MOUNT,
    Executor,
    LocalDocker,
    Outcome,
    RunRequest,
)
from priml.baselines.mlsbench.harbor import (
    allow_internet,
    bundle_gpus,
    check_eval_scripts,
    find_bundle,
    load_bundle,
)
from priml.baselines.mlsbench.setting import (
    Mounts,
    ParseError,
    data_mounts,
    elapsed_columns,
    expected_columns,
    inputgen_dir,
    package_env,
    run_parser,
)
from priml.baselines.mlsbench.task import TaskSpec, load_task
from priml.baselines.mlsbench.workspace import (
    Workspace,
    materialize,
    workspace_changes,
)
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import Tensor


TIME_SLACK: Final = 1.10
"""``max_eval_time`` is the sum of the settings' limits times this."""


_ERR_MAX: Final = 300


@dataclass(frozen=True, kw_only=True, slots=True)
class SettingRun:
    """The record of one setting: what happened and what the parser said."""

    label: str
    outcome: Outcome
    seconds: float
    metrics: Mapping[str, float]
    """Parser output, keyed by leaderboard column."""

    error: str | None
    """``timeout`` | ``crashed:<rc>`` | ``missing_metric:<key>`` | ``parse:<msg>`` |
    ``workspace_changed:<kind:path>`` (the setting was not run) |
    ``missing_input:<what>`` (a held-out input or package config was unusable; not run)."""


class SuiteState(TypedDict):
    """What a checkpoint of the suite holds: the per-setting records."""

    runs: list[dict[str, object]]


class MlsbenchSuite:
    """Runs settings and publishes ``<metric>_<setting>`` for each success."""

    class Config(Fig["MlsbenchSuite"]):
        """Which task, which edit, and how settings are executed."""

        task: str = ""
        """Task name; required."""

        baseline: str = ""
        """Baseline edit name, or ``""`` for the agent path or a pristine run."""

        edit_path: Path | str = ""
        """Explicit edit script (a candidate); exclusive with ``baseline``.
        ``MlsbenchEval`` resolves it beneath its ``base_dir`` at finalize."""

        seed: int = 42
        """Seed handed to the executor. One seed is one experiment."""

        mlsbench_root: Path | str = ""
        """The read-only upstream clone; required. ``MlsbenchEval`` fills it, and
        ``data_root`` below, from its own logical fields at finalize."""

        data_root: Path | str = ""
        """``{data_root}`` of upstream's package configs: datasets and model weights
        the settings bind. A host directory that is absent is skipped; ``""`` binds
        nothing."""

        bundle: Path | str = ""
        """Harbor bundle directory (``harbor/tasks-docker/mls-bench__<task>``). ``""``
        derives it from the upstream root the task was read from; a checkout with no
        bundle for the task falls back to the synthetic line-diff guard, and the
        network policy to ``none``."""

        base_dir: Path | str | None = None
        """Owner directory supplied during parent finalization."""

        working_dir: Path | str = "/runs"
        """Where the workspace, logs and ``settings.json`` go."""

        executor: Makeable[Executor] = field(default_factory=LocalDocker.Config)
        """Injected: docker in production, :class:`FakeExecutor` in tests."""

        @override
        def finalize(self) -> Self:
            if not self.task:
                raise ValueError("task is required.")
            if self.baseline and str(self.edit_path):
                raise ValueError("baseline and edit_path are mutually exclusive.")
            self.working_dir = resolve_working_dir(self.base_dir, self.working_dir)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        """Build the executor; nothing runs and no task is read yet."""
        self.config = config
        self.executor = config.executor.make()
        self._task: TaskSpec | None = None
        self._workspace: Workspace | None = None
        self._bundle: Path | None = None
        self.reset()

    def reset(self) -> None:
        """Forget every setting run so far."""
        self.runs: list[SettingRun] = []

    def _work_dir(self) -> Path:
        return Path(self.config.working_dir)

    def _task_spec(self) -> TaskSpec:
        if self._task is None:
            if not str(self.config.mlsbench_root):
                raise ValueError(
                    "MlsbenchSuite needs mlsbench_root, the upstream clone.",
                )
            task = load_task(self.config.task, root=Path(self.config.mlsbench_root))
            if self.config.seed not in task.seeds:
                raise ValueError(
                    f"seed {self.config.seed} is not one of {task.name}'s seeds "
                    f"{list(task.seeds)}; each published seed is its own experiment.",
                )
            self._task = task
        return self._task

    def _edit(self, task: TaskSpec) -> Path | None:
        if self.config.baseline:
            return task.baseline(self.config.baseline).edit
        if str(self.config.edit_path):
            return Path(self.config.edit_path)
        return None

    def _find_bundle(self, task: TaskSpec) -> Path | None:
        """Return the Harbor bundle dir (explicit field, else from the clone), or None."""
        explicit = str(self.config.bundle)
        if explicit:
            return Path(explicit)
        return find_bundle(task.name, root=task.root.parent.parent)

    # The container runs upstream's native ``tasks/<t>/scripts`` (the leaderboard's
    # provenance); Harbor's ``tests/eval/scripts`` is the tamper reference. See
    # ``workspace.check_eval_scripts`` for the policy. Returns the scripts Harbor
    # adapted (recorded as ``harbor_adapted_scripts``).
    def _verify_scripts(self, task: TaskSpec) -> tuple[str, ...]:
        """Refuse edited or swapped eval scripts (before any setting runs)."""
        bundle = self._bundle if self._bundle is not None else self._find_bundle(task)
        if bundle is None:
            return ()
        return check_eval_scripts(task.name, load_bundle(bundle), task.root / "scripts")

    def _materialize(self, task: TaskSpec) -> Workspace:
        if self._workspace is None:
            self._bundle = self._find_bundle(task)
            adapted = self._verify_scripts(task)
            work = self._work_dir()
            # Harness-generated copies from an earlier pass of this same
            # experiment; materialize refuses to reuse them.
            for name in ("workspace", "pristine"):
                shutil.rmtree(work / name, ignore_errors=True)
            work.mkdir(parents=True, exist_ok=True)
            self._workspace = materialize(
                task,
                work,
                edit=self._edit(task),
                bundle=self._bundle,
                # Published baselines are trusted upstream scripts; agent edits are not.
                strict=not self.config.baseline,
            )
            (work / "workspace.json").write_text(
                json.dumps(
                    {
                        "edit": self._workspace.edit,
                        "digest": self._workspace.digest,
                        "bundle": None if self._bundle is None else str(self._bundle),
                        "warnings": list(self._workspace.warnings),
                        "harbor_adapted_scripts": list(adapted),
                        "seed": self.config.seed,
                    },
                    indent=2,
                ),
            )
        return self._workspace

    def update(self, logits: Tensor, **batch: object) -> None:
        """Run one setting and record the outcome.

        Args:
          logits: Unused; nothing is being trained.
          **batch: Must carry ``setting``, a label of the task.

        Raises:
          TypeError: The batch has no ``setting``.

        """
        del logits
        label = batch.get("setting")
        if not isinstance(label, str):
            raise TypeError("An mlsbench metric requires a 'setting' batch entry.")
        if any(r.label == label for r in self.runs):
            return  # The parser runs at most once per setting.
        task = self._task_spec()
        setting = task.setting(label)
        cmd_swapped = False
        if self.config.baseline:
            # A baseline may swap the setting's script for its own ``cmd``.
            bl = task.baseline(self.config.baseline)
            if bl.cmd is not None and (bl.labels is None or label in bl.labels):
                setting = replace(setting, script=bl.cmd)
                cmd_swapped = True
        self._verify_scripts(task)  # Cheap; also catches a swap between settings.
        workspace = self._materialize(task)
        # The digest is re-verified before EVERY setting, not only at materialize:
        # an earlier setting (agent or task code running in the mounted workspace)
        # may have rewritten the edit surface. Files a setting adds are outputs.
        changes = workspace_changes(workspace)
        if changes:
            self.runs.append(
                SettingRun(
                    label=label,
                    outcome=Outcome.CRASHED,
                    seconds=0.0,
                    metrics={},
                    error=f"workspace_changed:{changes[0]}",
                ),
            )
            self._dump()
            return
        inputgen = inputgen_dir(self._bundle)
        mounts = data_mounts(
            task,
            setting.package,
            data_root=self.config.data_root,
        )
        missing = mounts.missing
        if not missing and inputgen is None and task.ephemeral_inputs:
            missing = (
                "held-out inputs: task has ephemeral_inputs but the Harbor bundle "
                "ships no tests/eval/_inputgen/apply.py"
                if self._bundle is not None
                else "held-out inputs: task has ephemeral_inputs and no Harbor bundle "
                "was found to generate them"
            )
        if missing:
            self.runs.append(
                SettingRun(
                    label=label,
                    outcome=Outcome.MISSING_INPUT,
                    seconds=0.0,
                    metrics={},
                    error=f"missing_input:{missing}"[:_ERR_MAX],
                ),
            )
            self._dump()
            return
        baseline = task.baseline(self.config.baseline) if self.config.baseline else None
        overlay: Mapping[str, str] = (
            baseline.env
            if baseline is not None
            and (baseline.labels is None or label in baseline.labels)
            else dict[str, str]()
        )
        env = package_env(task, setting.package, self._bundle, overlay=overlay)
        request = RunRequest(
            # Settings of one task may run in different package images.
            task=replace(task, package=setting.package),
            setting=setting,
            workspace=workspace,
            seed=self.config.seed,
            log_path=self._work_dir() / "logs" / f"{label}.log",
            allow_internet=(
                None if self._bundle is None else allow_internet(self._bundle)
            ),
            gpus=None if self._bundle is None else bundle_gpus(self._bundle),
            env=env,
            binds=mounts.binds,
            inputgen=inputgen,
        )
        self._record_request(request, mounts, cmd_swapped=cmd_swapped)
        result = self.executor.run(request)
        if result.outcome is Outcome.OK:
            run = self._parse(task, label, result.seconds, result.log_path)
        else:
            if result.outcome is Outcome.TIMEOUT:
                error = "timeout"
            elif result.outcome is Outcome.MISSING_INPUT:
                error = f"missing_input:{result.detail}"[:_ERR_MAX]
            else:
                error = f"crashed:{result.returncode}"
            run = SettingRun(
                label=label,
                outcome=result.outcome,
                seconds=result.seconds,
                metrics={},
                error=error,
            )
        self.runs.append(run)
        self._dump()

    def _record_request(
        self,
        request: RunRequest,
        mounts: Mounts,
        *,
        cmd_swapped: bool = False,
    ) -> None:
        """Write what the executor was asked to provide (audit trail, no secrets read)."""
        path = self._work_dir() / "logs" / f"{request.setting.label}.request.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "seed": request.seed,
                    "SEED": str(request.seed),
                    "ENV": request.setting.label,
                    "OUTPUT_DIR": OUTPUT_MOUNT,
                    "output_dir_host": str(request.log_path.with_suffix(".out")),
                    "baseline": self.config.baseline or None,
                    "script": request.setting.script,
                    "cmd": request.setting.script if cmd_swapped else None,
                    "package": request.task.package,
                    "env": dict(request.env),
                    "binds": list(request.binds),
                    "skipped_binds": list(mounts.skipped),
                    "inputgen": request.inputgen is not None,
                },
                indent=2,
            ),
        )

    def _parse(
        self,
        task: TaskSpec,
        label: str,
        seconds: float,
        log_path: Path,
    ) -> SettingRun:
        try:
            text = log_path.read_text(errors="replace")
            meta = None if self._bundle is None else self._bundle / "tests" / "meta"
            metrics = run_parser(task, text, label, meta=meta)
        except (ParseError, OSError) as e:
            msg = f"parse:{e}"[:_ERR_MAX]
            return SettingRun(
                label=label,
                outcome=Outcome.OK,
                seconds=seconds,
                metrics={},
                error=msg,
            )
        missing = [col for col in expected_columns(task, label) if col not in metrics]
        error = f"missing_metric:{missing[0]}" if missing else None
        return SettingRun(
            label=label,
            outcome=Outcome.OK,
            seconds=seconds,
            metrics=metrics,
            error=error,
        )

    def compute(self) -> dict[str, float]:
        """Publish every metric of every setting that produced one.

        Returns:
          metrics: Leaderboard column to value (``elapsed_*`` is the runner's
            wall time). Failed settings and missing metrics are absent, never NaN.

        """
        self._dump()
        task = self._task_spec() if self.runs else None
        out: dict[str, float] = {}
        for run in self.runs:
            out.update(run.metrics)
            if run.error is not None and run.error.startswith("workspace_changed"):
                # A marker column no leaderboard row carries: the row reads unverified.
                out[f"workspace_changed_{run.label}"] = 1.0
            if task is not None and run.outcome is Outcome.OK and run.error is None:
                out.update(dict.fromkeys(elapsed_columns(task, run.label), run.seconds))
        return out

    def _dump(self) -> None:
        work = self._work_dir()
        work.mkdir(parents=True, exist_ok=True)
        records = [{**asdict(run), "outcome": str(run.outcome)} for run in self.runs]
        (work / "settings.json").write_text(json.dumps(records, indent=2))

    def state_dict(self) -> SuiteState:
        """Return the per-setting records as plain data."""
        return {
            "runs": [{**asdict(run), "outcome": str(run.outcome)} for run in self.runs],
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        """Restore records saved by :meth:`state_dict`.

        Args:
          state_dict: As returned by :meth:`state_dict`.

        """
        state = cast("SuiteState", state_dict)
        self.runs = [
            SettingRun(
                label=cast("str", r["label"]),
                outcome=Outcome(cast("str", r["outcome"])),
                seconds=cast("float", r["seconds"]),
                metrics=cast("dict[str, float]", r["metrics"]),
                error=cast("str | None", r["error"]),
            )
            for r in state["runs"]
        ]


def budget_seconds(limits: Iterable[float]) -> float:
    """Return ``sum(limits) * TIME_SLACK``: the pure rule behind ``max_eval_time``."""
    return sum(limits) * TIME_SLACK


def time_budget(task: TaskSpec, settings: tuple[str, ...] = ()) -> float:
    """Total eval budget: the settings' limits summed, plus slack.

    Args:
      task: The parsed task.
      settings: Label subset; ``()`` for all.

    Returns:
      seconds: ``sum(time_limit) * TIME_SLACK``.

    """
    return budget_seconds(
        s.time_limit for s in task.settings if not settings or s.label in settings
    )


__all__ = [
    "TIME_SLACK",
    "MlsbenchSuite",
    "SettingRun",
    "SuiteState",
    "budget_seconds",
    "time_budget",
]
