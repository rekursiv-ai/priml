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

Parser contract (read from upstream, ``mlsbench/agent/parsers.py``): each task's
``parser.py`` defines ``class Parser(OutputParser)`` whose
``parse(cmd_label, raw_output)`` returns a ``ParseResult`` with a ``metrics``
dict already keyed by the leaderboard column (``<metric>_<label>``). The
parser imports ``mlsbench`` from ``<root>/src``, so that directory is put on
``sys.path`` while it loads; nothing is copied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypedDict, cast, override

import contextlib
import importlib.util
import json
import math
import re
import shutil
import sys

from configgle import Fig, Makeable

from priml.baselines.mlsbench.executor import (
    _OUTPUT_MOUNT,
    Executor,
    LocalDocker,
    Outcome,
    RunRequest,
)
from priml.baselines.mlsbench.scripts.prepare_data import (
    StageError,
    dataset_binds,
    staged_bind,
    staged_tasks,
)
from priml.baselines.mlsbench.task import TaskSpec, load_task
from priml.baselines.mlsbench.workspace import (
    Workspace,
    materialize,
    workspace_changes,
)
from priml.baselines.mlsbench.workspace_harbor import (
    allow_internet,
    bundle_gpus,
    check_eval_scripts,
    find_bundle,
    load_bundle,
)
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from torch import Tensor


TIME_SLACK = 1.10
"""``max_eval_time`` is the sum of the settings' limits times this."""

_ERR_MAX = 300


class ParseError(RuntimeError):
    """``parser.py`` raised, or returned something that is not finite numbers."""


@dataclass(frozen=True)
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
    ``missing_input:<what>`` (a held-out or staged input was absent; not run)."""


_ENV_VAR = re.compile(r"\$(\w+)|\$\{([^}:]+)(?::-[^}]*)?\}")
"""Same pattern as Harbor's verifier: ``$X``, ``${X}``, ``${X:-d}`` (the default is
ignored, an unset name expands to empty)."""


def expand_env(value: str, base: Mapping[str, str]) -> str:
    """Expand ``$X`` templates against ``base`` like Harbor's ``_expand_env_template``."""

    def repl(m: re.Match[str]) -> str:
        return base.get(m.group(1) or m.group(2) or "", "")

    return _ENV_VAR.sub(repl, value)


def _json_env(path: Path) -> dict[str, dict[str, str]]:
    try:
        raw = cast("object", json.loads(path.read_text()))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(pkg): {str(k): str(v) for k, v in cast("dict[object, object]", env).items()}
        for pkg, env in cast("dict[object, object]", raw).items()
        if isinstance(env, dict)
    }


def package_env(
    task: TaskSpec,
    package: str,
    bundle: Path | None,
    *,
    overlay: Mapping[str, str] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Return the container environment a package's scripts run under.

    Mirrors Harbor's verifier (``score_task._eval_env``): the block of
    ``tests/meta/package_envs.json`` for the setting's package (else the task's
    default package), ``$X`` templates expanded against what is already set (the
    verifier unsets ``PYTHONPATH`` and ``LD_LIBRARY_PATH`` first, so those expand to
    empty), ``HOME`` taken literally, then ``overlay`` (a baseline's own ``env``), and
    ``DATA_ROOT=/data`` unless set. Without a bundle the native pkg config's ``env``
    (``vendor/pkg_configs/<package>/config.json``) is the source. ``SEED`` and the
    output variables are the executor's, not set here.

    Args:
      task: The parsed task (locates the clone and the default package).
      package: The setting's package.
      bundle: Harbor bundle directory, or ``None``.
      overlay: Extra variables applied after the package block.

    Returns:
      env: ``(name, value)`` pairs in application order, deduplicated.

    """
    block: dict[str, str] = {}
    envs = _json_env(bundle / "tests" / "meta" / "package_envs.json") if bundle else {}
    if envs:
        block = envs.get(package, envs.get(task.package, {}))
    else:
        cfg = task.root.parent.parent / "vendor" / "pkg_configs"
        for d in sorted(cfg.iterdir()) if cfg.is_dir() else []:
            if d.name.lower() == package.lower() and (d / "config.json").is_file():
                try:
                    raw = cast("object", json.loads((d / "config.json").read_text()))
                except (OSError, ValueError):
                    raw = {}
                env = (
                    cast("dict[str, object]", raw).get("env")
                    if isinstance(raw, dict)
                    else None
                )
                if isinstance(env, dict):
                    block = {
                        str(k): str(v)
                        for k, v in cast("dict[object, object]", env).items()
                    }
    out: dict[str, str] = {}
    for key, value in block.items():
        out[key] = value if key == "HOME" else expand_env(value, out)
    for key, value in (overlay or {}).items():
        out[str(key)] = expand_env(str(value), out)
    out.setdefault("DATA_ROOT", "/data")
    return tuple(out.items())


def inputgen_dir(bundle: Path | None) -> Path | None:
    """Return the bundle's ``tests/eval/_inputgen`` if it ships an ``apply.py``."""
    if bundle is None:
        return None
    d = bundle / "tests" / "eval" / "_inputgen"
    return d if (d / "apply.py").is_file() else None


@dataclass(frozen=True)
class Mounts:
    """Data mounts of one setting, or why they cannot be provided."""

    binds: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    """Dataset binds dropped because the host directory is absent."""

    missing: str = ""
    """Set when a required input is unavailable: the setting must not run."""


def data_mounts(
    task: TaskSpec,
    package: str,
    *,
    data_root: Path | str = "",
    staged_data: Path | str = "",
) -> Mounts:
    """Derive a setting's data mounts from the data directories and ``prepare_data``.

    ``data_root`` selects the package's dataset binds (upstream's ``data_bind`` /
    ``data_deps``; a host directory that is absent is skipped). ``staged_data``
    overlays a task's staged ``data/`` after verifying its digests; a mismatch is
    ``missing``. ``""`` disables either.
    """
    clone = task.root.parent.parent
    binds: list[str] = []
    skipped: tuple[str, ...] = ()
    if str(data_root):
        try:
            found = dataset_binds(clone, package, data_root)
        except StageError as e:
            return Mounts(missing=f"pkg config: {e}")
        binds += found.binds
        skipped = found.skipped
    staged = str(staged_data)
    if staged and task.name in staged_tasks(Path(staged)):
        try:
            binds.append(staged_bind(Path(staged), task.name))
        except StageError as e:
            return Mounts(tuple(binds), skipped, f"staged data: {e}")
    return Mounts(tuple(binds), skipped)


_NOT_PARSED = ("elapsed_",)
"""Columns the runner measures (wall time); the parser does not emit them."""


def expected_columns(task: TaskSpec, label: str) -> tuple[str, ...]:
    """Parser-emitted leaderboard columns that belong to a setting.

    A column belongs to the *longest* label it ends with, so a label that is a
    suffix of another (``x`` and ``a-x``) does not claim its sibling's columns.
    Columns that end in no label (some tasks name them otherwise) belong to no
    setting and are never demanded; ``elapsed_*`` columns are the runner's.

    Args:
      task: The parsed task.
      label: A setting label.

    Returns:
      columns: Sorted column names any published row carries.

    """
    cols = {c for row in task.leaderboard for c in row.values}
    labels = [s.label for s in task.settings]
    mine: list[str] = []
    for col in cols:
        if col.startswith(_NOT_PARSED):
            continue
        owner = max(
            (lab for lab in labels if col.endswith("_" + lab)),
            key=len,
            default=None,
        )
        if owner == label:
            mine.append(col)
    return tuple(sorted(mine))


def elapsed_columns(task: TaskSpec, label: str) -> tuple[str, ...]:
    """Return the published ``elapsed_*`` columns of a setting (runner-measured)."""
    cols = {
        c for row in task.leaderboard for c in row.values if c.startswith(_NOT_PARSED)
    }
    return tuple(sorted(c for c in cols if c.endswith("_" + label)))


@contextlib.contextmanager
def _src_on_path(src: Path) -> Generator[None]:
    entry = str(src)
    added = src.is_dir() and entry not in sys.path
    if added:
        sys.path.insert(0, entry)
    try:
        yield
    finally:
        if added and entry in sys.path:
            sys.path.remove(entry)


def _host_modules(meta: Path | None) -> tuple[str, ...]:
    """Names of the host-only modules (e.g. ``dgp``) a bundle's ``tests/meta`` ships."""
    if meta is None or not meta.is_dir():
        return ()
    return tuple(sorted(p.stem for p in meta.glob("*.py") if p.name != "parser.py"))


def _clone_src(task: TaskSpec) -> Path:
    """``<root>/src``: ``<root>/tasks/<task>/parser.py`` imports ``mlsbench`` from it."""
    return task.parser.parent.parent.parent / "src"


def _parser_path(task: TaskSpec, meta: Path | None) -> Path:
    """Return the parser file to execute: the bundle's copy when there is one.

    Harbor's ``score_task.py`` runs ``tests/meta/parser.py``, a byte-identical
    copy of the task's. Parsers that find ``dgp.py`` next to their own file
    (``optimization-variance-reduction``) need that location.
    """
    if meta is not None and (meta / "parser.py").is_file():
        return meta / "parser.py"
    return task.parser


@contextlib.contextmanager
def _loaded_parser(
    path: Path,
    src: Path,
    meta: Path | None = None,
) -> Generator[Callable[[str, str], object]]:
    """Import upstream's parser and yield its ``parse``, with its imports reachable.

    The clone's ``src`` and the bundle's ``tests/meta`` stay on ``sys.path``, and
    hosted modules stay registered, until the caller is done: upstream parsers
    also import inside ``parse`` (``optimization-variance-reduction`` loads
    ``dgp`` lazily).
    """
    name = f"mlsbench_parser_{path.parent.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ParseError(f"cannot load parser {path}")
    module = importlib.util.module_from_spec(spec)
    # Host-only scoring modules (``dgp.py``: held-out labels) ship only in the
    # bundle's tests/meta, never in the task dir; Harbor's score_task.py puts that
    # directory on sys.path before it runs the parser, and so does this. The
    # names are dropped from sys.modules afterwards: each task has its own dgp.
    hosted = _host_modules(meta)
    for mod in hosted:
        sys.modules.pop(mod, None)
    sys.modules[name] = module
    try:
        with (
            _src_on_path(src),
            _src_on_path(meta) if meta is not None else contextlib.nullcontext(),
        ):
            spec.loader.exec_module(module)
            cls = getattr(module, "Parser", None)
            if not callable(cls):
                raise ParseError(f"{path}: no class Parser")
            parse = getattr(cls(), "parse", None)
            if not callable(parse):
                raise ParseError(f"{path}: Parser has no parse(cmd_label, raw_output)")
            yield cast("Callable[[str, str], object]", parse)
    finally:
        sys.modules.pop(name, None)
        for mod in hosted:
            sys.modules.pop(mod, None)


def run_parser(
    task: TaskSpec,
    log_text: str,
    setting: str,
    *,
    meta: Path | None = None,
) -> dict[str, float]:
    """Call upstream's ``Parser().parse`` once and validate what it returns.

    Args:
      task: The parsed task; ``task.parser`` is executed as-is.
      log_text: Combined stdout and stderr of the setting.
      setting: The setting label.
      meta: The Harbor bundle's ``tests/meta``, put on ``sys.path`` for the
        host-only modules the parser imports (``dgp``); its ``parser.py``, the
        same file, runs in place of the task's, as Harbor does. ``None`` without
        a bundle.

    Returns:
      metrics: Leaderboard column (``<metric>_<setting>``) to finite float. A
        NaN or infinite value is left out, so only that column goes missing.

    Raises:
      ParseError: The parser raised, or returned non-numeric values.

    """
    try:
        with _loaded_parser(_parser_path(task, meta), _clone_src(task), meta) as parse:
            result = parse(setting, log_text)
    except ParseError:
        raise
    except Exception as e:  # upstream code: any failure is a parse failure
        raise ParseError(f"{type(e).__name__}: {e}") from e
    raw = getattr(result, "metrics", None)
    if not hasattr(raw, "items"):
        raise ParseError(
            f"parser result has no metrics mapping ({type(result).__name__})",
        )
    out: dict[str, float] = {}
    for key, value in cast("Mapping[object, object]", raw).items():
        try:
            x = float(cast("float", value))
        except (TypeError, ValueError) as e:
            raise ParseError(f"{key!r}: non-numeric {value!r}") from e
        if math.isfinite(x):  # upstream's scorer drops a non-finite cell alone
            out[str(key)] = x
    return out


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
        the two data directories below, from its own logical fields at finalize."""

        data_root: Path | str = ""
        """``{data_root}`` of upstream's package configs: datasets and model weights
        the settings bind. A host directory that is absent is skipped; ``""`` binds
        nothing."""

        staged_data: Path | str = ""
        """A ``prepare_data`` destination; a task its manifest lists gets that
        digest-checked ``data/`` laid over the clone's. ``""`` overlays nothing."""

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

    def _verify_scripts(self, task: TaskSpec) -> tuple[str, ...]:
        """Refuse edited or swapped eval scripts (before any setting runs).

        The container runs upstream's native ``tasks/<t>/scripts`` (the leaderboard's
        provenance); Harbor's ``tests/eval/scripts`` is the tamper reference. See
        ``workspace_harbor.check_eval_scripts`` for the policy. Returns the scripts
        Harbor adapted (recorded as ``harbor_adapted_scripts``).

        Raises:
          EvalScriptsError: A script differs from what the bundle vouches for.

        """
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
            return  # the parser runs at most once per setting
        task = self._task_spec()
        setting = task.setting(label)
        cmd_swapped = False
        if self.config.baseline:
            # A baseline may swap the setting's script for its own ``cmd``.
            bl = task.baseline(self.config.baseline)
            if bl.cmd is not None and (bl.labels is None or label in bl.labels):
                setting = replace(setting, script=bl.cmd)
                cmd_swapped = True
        self._verify_scripts(task)  # cheap; also catches a swap between settings
        workspace = self._materialize(task)
        # The digest is re-verified before EVERY setting, not only at materialize:
        # an earlier setting (agent or task code running in the mounted workspace)
        # may have rewritten the edit surface. Files a setting adds are outputs.
        changes = workspace_changes(workspace)
        if changes:
            self.runs.append(
                SettingRun(
                    label,
                    Outcome.CRASHED,
                    0.0,
                    {},
                    f"workspace_changed:{changes[0]}",
                ),
            )
            self._dump()
            return
        inputgen = inputgen_dir(self._bundle)
        mounts = data_mounts(
            task,
            setting.package,
            data_root=self.config.data_root,
            staged_data=self.config.staged_data,
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
                    label,
                    Outcome.MISSING_INPUT,
                    0.0,
                    {},
                    f"missing_input:{missing}"[:_ERR_MAX],
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
            run = SettingRun(label, result.outcome, result.seconds, {}, error)
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
                    "OUTPUT_DIR": _OUTPUT_MOUNT,
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
            return SettingRun(label, Outcome.OK, seconds, {}, msg)
        missing = [col for col in expected_columns(task, label) if col not in metrics]
        error = f"missing_metric:{missing[0]}" if missing else None
        return SettingRun(label, Outcome.OK, seconds, metrics, error)

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
        """Restore records saved by :meth:`state_dict`."""
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
    "ParseError",
    "SettingRun",
    "budget_seconds",
    "elapsed_columns",
    "expected_columns",
    "run_parser",
    "time_budget",
]
