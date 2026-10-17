#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Place a run's metrics back on an MLS-Bench task's published leaderboard.

Verdict layer of the mlsbench harness. For one leaderboard row it compares the
measured ``<metric>_<setting>`` values with the published cells and says, per
column and per row, whether the reproduction is ``ok``, ``diverged`` or
``unverified``. Generic: no task-specific logic. Published values
are per baseline: the upstream ``mean`` row, or the mean over seed rows, with the
seed standard deviation as spread (see :func:`priml.baselines.mlsbench.task.parse_leaderboard`).

Tolerance by metric kind (see :func:`classify`):

- ``exact``: step counts and other deterministic values; any difference diverges.
- ``accuracy``: fixed test set, +/-2 points absolute (0.02 on a [0, 1] scale).
- ``continuous``: perplexity, loss, MAUVE; only judged when the task supplies a
  per-column ``atol`` measured from a repeat run, else ``unverified``.
- ``multi_seed``: mean of several runs within the published seed spread (or the
  runs' own sample standard deviation, whichever is larger).

A missing published cell or a missing measurement is never ``ok``.

Whole benchmark: :func:`aggregate` walks every task's runs under one directory
(``<base_dir>/runs/mlsbench/<task>/<baseline>/seed<N>/metrics.json``), judges each
baseline over all its seeds (multi-seed rows by the published spread) and counts
``ok`` / ``diverged`` / ``unverified`` per task and overall. A baseline with no
run is ``unverified``; a multi-seed baseline missing some of the task's seeds can
be ``diverged`` but never ``ok`` (the missing seeds are listed).

Torch-free. Run::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.leaderboard --task T --runs DIR [--row R]
    uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.leaderboard --all --runs BASE_DIR [--include-missing]

'''
# fmt: on

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import functools
import math
import statistics

from priml.baselines.mlsbench.experiments import MlsbenchEval
from priml.baselines.mlsbench.task import load_task, task_names
from priml.paths import resolve_working_dir
from priml.train.results import read_metrics


if TYPE_CHECKING:
    from priml.baselines.mlsbench.task import TaskSpec


class Verdict(StrEnum):
    """Outcome of comparing one column, or (worst-of) one row."""

    OK = "ok"
    DIVERGED = "diverged"
    UNVERIFIED = "unverified"


class MetricKind(StrEnum):
    """How a metric's reproduction is judged."""

    EXACT = "exact"
    ACCURACY = "accuracy"
    CONTINUOUS = "continuous"
    MULTI_SEED = "multi_seed"


@dataclass(frozen=True, kw_only=True, slots=True)
class Tolerance:
    """A metric kind and its absolute tolerance."""

    kind: MetricKind
    atol: float


DEFAULT_TOLERANCES: Final[Mapping[MetricKind, float]] = {
    MetricKind.EXACT: 0.0,
    MetricKind.ACCURACY: 0.02,
}
"""Plan defaults: exact for deterministic counts, 2 points for accuracy."""


def classify(metric: str) -> MetricKind:
    """Classify a metric (or ``<metric>_<setting>`` column) by its name.

    Args:
      metric: Metric name or leaderboard column.

    Returns:
      kind: ``EXACT`` for step/count names, ``ACCURACY`` for accuracy/pass
        names, else ``CONTINUOUS`` (unknown names included).

    """
    if metric.startswith(("avg_steps", "steps", "num_")):
        return MetricKind.EXACT
    if metric.startswith(("accuracy", "pass", "acc")):
        return MetricKind.ACCURACY
    return MetricKind.CONTINUOUS


@dataclass(frozen=True, kw_only=True, slots=True)
class TaskOverrides:
    """Per-task adjustments, keyed by leaderboard column name.

    Attributes:
      continuous_atol: Column to atol for continuous metrics, measured from a
        repeat run. Absent means noise is unmeasured and the column is
        ``unverified``.
      kinds: Explicit reclassification of a column.
      scale: Column to scale (e.g. 100.0 when accuracy is published in percent);
        multiplies the accuracy tolerance.

    """

    continuous_atol: Mapping[str, float] = field(default_factory=dict[str, float])
    kinds: Mapping[str, MetricKind] = field(default_factory=dict[str, MetricKind])
    scale: Mapping[str, float] = field(default_factory=dict[str, float])


@dataclass(frozen=True, kw_only=True, slots=True)
class Comparison:
    """One column, measured against its published cell."""

    column: str
    published: float | None
    measured: float | None
    delta: float | None
    kind: MetricKind
    atol: float | None
    verdict: Verdict
    reason: str


@dataclass(frozen=True, kw_only=True, slots=True)
class RowReport:
    """All comparisons for one leaderboard row; ``verdict`` is worst-of its published columns."""

    task: str
    row: str
    seed_count: int
    comparisons: tuple[Comparison, ...]
    verdict: Verdict


class _RowLike(Protocol):
    @property
    def values(self) -> Mapping[str, float]: ...
    @property
    def spread(self) -> Mapping[str, float]: ...
    @property
    def seeds(self) -> Mapping[int, Mapping[str, float]]: ...


class _TaskLike(Protocol):
    @property
    def name(self) -> str: ...
    def row(self, name: str) -> _RowLike: ...


def read_metrics_file(run_dir: Path) -> dict[str, float]:
    """Read ``<run_dir>/metrics.json``; keep ``eval/*`` keys, prefix stripped.

    Args:
      run_dir: A run directory holding ``metrics.json``.

    Returns:
      metrics: ``<metric>_<setting>`` to value.

    """
    raw = read_metrics(run_dir / "metrics.json")
    return {k.removeprefix("eval/"): v for k, v in raw.items() if k.startswith("eval/")}


def compare(
    task: TaskSpec | _TaskLike,
    row: str,
    runs: Sequence[Mapping[str, float]],
    overrides: TaskOverrides = TaskOverrides(),  # noqa: B008 -- frozen, so a shared default is safe.
    seeds: Sequence[int] | None = None,
) -> RowReport:
    """Compare measured runs of one row with the published row.

    Args:
      task: The task; ``task.row(row)`` supplies published values and spread.
      row: Leaderboard row (baseline) name.
      runs: One metrics mapping per seed (``<metric>_<setting>`` keys).
      overrides: Per-task tolerance adjustments.
      seeds: The seed of each run. When every one has a published seed row,
        each column is judged against the mean of those seeds' published
        values, so a partial run is not held to the all-seed mean.

    Returns:
      report: Per-column comparisons, published columns first, then measured
        columns with no published cell. Row verdict is worst-of over the
        published columns; the others are informational.

    """
    published_row: _RowLike = task.row(row)
    published = _published_for(published_row, seeds)
    columns = list(published_row.values)
    extra = sorted({k for run in runs for k in run} - set(columns))
    comparisons = tuple(
        _judge(
            col,
            published.get(col),
            [run[col] for run in runs if col in run and math.isfinite(run[col])],
            published_row.spread.get(col),
            overrides,
            len(runs),
        )
        for col in [*columns, *extra]
    )
    return RowReport(
        task=task.name,
        row=row,
        seed_count=len(runs),
        comparisons=comparisons,
        # A column the row never published (a hidden-split metric, the
        # workspace marker) is shown but cannot fail or hold back the row.
        verdict=_worst([c.verdict for c in comparisons[: len(columns)]]),
    )


def headline(reports: Sequence[RowReport]) -> tuple[int, int, int]:
    """Count rows by verdict.

    Diverged rows are counted separately and never contribute to the ok count.

    Args:
      reports: Row reports.

    Returns:
      counts: ``(ok, diverged, unverified)``.

    """
    ok = sum(r.verdict == Verdict.OK for r in reports)
    diverged = sum(r.verdict == Verdict.DIVERGED for r in reports)
    return ok, diverged, len(reports) - ok - diverged


def format_report(reports: Sequence[RowReport]) -> str:
    """Render reports as aligned tables plus a headline line.

    Args:
      reports: Row reports.

    Returns:
      text: The rendered diff.

    """
    lines: list[str] = []
    for rep in reports:
        lines.append(
            f"[{rep.verdict}] {rep.task} / {rep.row} ({rep.seed_count} run(s))",
        )
        width = max((len(c.column) for c in rep.comparisons), default=6)
        lines.append(
            f"  {'column':<{width}}  {'measured':>10}  {'published':>10}  "
            f"{'delta':>10}  {'verdict':<10}  reason",
        )
        lines.extend(
            f"  {c.column:<{width}}  {_fmt(c.measured):>10}  {_fmt(c.published):>10}  "
            f"{('-' if c.delta is None else f'{c.delta:+.4f}'):>10}  "
            f"{c.verdict.value:<10}  {c.reason}"
            for c in rep.comparisons
        )
    ok, diverged, unverified = headline(reports)
    lines.append(f"headline: ok={ok} diverged={diverged} unverified={unverified}")
    return "\n".join(lines)


class _SpecLike(_TaskLike, Protocol):
    @property
    def seeds(self) -> Sequence[int]: ...
    @property
    def baselines(self) -> Sequence[_HasName]: ...


class _HasName(Protocol):
    @property
    def name(self) -> str: ...


@dataclass(frozen=True, kw_only=True, slots=True)
class TaskSummary:
    """One task's row reports and the seeds each row lacks.

    Attributes:
      task: Task name.
      reports: One report per baseline, in the task's baseline order.
      missing_seeds: Baseline to the task's seeds that have no run (only rows
        with at least one run and at least one missing seed appear).

    """

    task: str
    reports: tuple[RowReport, ...]
    missing_seeds: Mapping[str, tuple[int, ...]] = field(
        default_factory=dict[str, tuple[int, ...]],
    )

    @property
    def counts(self) -> tuple[int, int, int]:
        """``(ok, diverged, unverified)`` over this task's rows."""
        return headline(self.reports)


def find_seed_runs(runs_dir: Path, task: str, row: str) -> dict[int, dict[str, float]]:
    """Return seed to metrics for ``<runs>/<task>/<row>/seed<N>/metrics.json``.

    Also accepts the flat ``<runs>/<row>/seed<N>`` layout of :func:`main`.

    Args:
      runs_dir: ``<base_dir>/runs/mlsbench``.
      task: Task name.
      row: Leaderboard row (or candidate) name.

    Returns:
      runs: Seed to its ``<metric>_<setting>`` metrics.

    """
    for base in (runs_dir / task / row, runs_dir / row):
        found: dict[int, dict[str, float]] = {}
        for path in sorted(base.glob("seed*/metrics.json")):
            digits = path.parent.name.removeprefix("seed")
            if digits.lstrip("-").isdigit():
                found[int(digits)] = read_metrics_file(path.parent)
        if found:
            return found
    return {}


def discover_tasks(runs_dir: Path) -> list[str]:
    """Return the task names that have a ``<task>/<row>/seed*/metrics.json``."""
    return sorted({p.parts[-4] for p in runs_dir.glob("*/*/seed*/metrics.json")})


def aggregate(
    specs: Sequence[_SpecLike],
    runs_dir: Path,
    overrides: Mapping[str, TaskOverrides] | None = None,
) -> tuple[TaskSummary, ...]:
    """Judge every baseline of every task in ``specs`` against its runs.

    Args:
      specs: Parsed tasks (``TaskSpec`` or anything shaped like it).
      runs_dir: ``<base_dir>/runs/mlsbench``.
      overrides: Task name to its tolerance overrides.

    Returns:
      summaries: One per task, in ``specs`` order.

    """
    return tuple(
        _summarize(spec, runs_dir, (overrides or {}).get(spec.name)) for spec in specs
    )


def overall(summaries: Sequence[TaskSummary]) -> tuple[int, int, int]:
    """Return ``(ok, diverged, unverified)`` over every row of every task."""
    return headline([r for s in summaries for r in s.reports])


def format_summary(summaries: Sequence[TaskSummary], *, verbose: bool = False) -> str:
    """Render per-task counts, optionally each row, and the overall headline.

    Args:
      summaries: One per task.
      verbose: Also list each row's verdict.

    Returns:
      text: The rendered table.

    """
    width = max((len(s.task) for s in summaries), default=4)
    lines = [f"{'task':<{width}}  {'ok':>4}  {'diverged':>8}  {'unverified':>10}"]
    for s in summaries:
        ok, diverged, unverified = s.counts
        lines.append(f"{s.task:<{width}}  {ok:>4}  {diverged:>8}  {unverified:>10}")
        if verbose:
            for r in s.reports:
                extra = s.missing_seeds.get(r.row, ())
                note = f"  missing seeds {list(extra)}" if extra else ""
                lines.append(f"    [{r.verdict}] {r.row} ({r.seed_count} run(s)){note}")
    ok, diverged, unverified = overall(summaries)
    lines.append(
        f"overall: tasks={len(summaries)} rows={ok + diverged + unverified} "
        f"ok={ok} diverged={diverged} unverified={unverified}",
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Diff runs on disk against a task's published leaderboard.

    Args:
      argv: ``--task``, ``--runs``, optional ``--row`` and ``--root``; or
        ``--all`` (also the default without ``--task``) to aggregate every task
        under ``--runs`` (a base_dir or its ``runs/mlsbench``), with
        ``--include-missing`` to count runless clone tasks as unverified.

    Returns:
      code: 0 iff no row diverged (overall, in aggregate mode).

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("_Flags", parser.parse_args(argv))
    if flags.all or flags.task is None:
        if flags.task is not None or flags.row or flags.candidate:
            parser.error("--all takes none of --task, --row and --candidate")
        runs_root = _runs_root(Path(flags.runs))
        root = Path(flags.root)
        names = discover_tasks(runs_root)
        if flags.include_missing:
            names = task_names(root)
        specs = [load_task(n, root=root) for n in names]
        summaries = aggregate(specs, runs_root)
        print(format_summary(summaries, verbose=flags.verbose))
        return 1 if overall(summaries)[1] else 0
    if flags.candidate and not flags.row:
        parser.error("--candidate needs --row, the published row to judge it against")
    spec = load_task(flags.task, root=Path(flags.root))
    names = [flags.row] if flags.row else [b.name for b in spec.baselines]
    runs_dir = _runs_root(Path(flags.runs))
    reports = [
        _compare_on_disk(spec, runs_dir, flags.task, name, flags.candidate)
        for name in names
    ]
    print(format_report(reports))
    _, diverged, _ = headline(reports)
    return 1 if diverged else 0


def _within(delta: float, atol: float) -> bool:
    return abs(delta) <= atol * (1 + 1e-9) + 1e-12


def _judge(  # noqa: PLR0917 -- every caller passes the six facts of one column.
    column: str,
    published: float | None,
    values: list[float],
    spread: float | None,
    overrides: TaskOverrides,
    n_runs: int,
) -> Comparison:
    kind = overrides.kinds.get(column, classify(column))
    multi = kind != MetricKind.EXACT and (n_runs > 1 or spread is not None)
    if multi:
        kind = MetricKind.MULTI_SEED

    out = functools.partial(_comparison, column, published, kind)
    measured = statistics.fmean(values) if values else None
    if published is None:
        return out(Verdict.UNVERIFIED, "no published value", measured)
    if measured is None:
        return out(Verdict.UNVERIFIED, "not measured")
    delta = measured - published

    if kind == MetricKind.MULTI_SEED:
        std = statistics.stdev(values) if len(values) > 1 else None
        candidates = [x for x in (spread, std) if x is not None]
        if not candidates:
            return out(Verdict.UNVERIFIED, "no seed spread available", measured)
        tol = max(candidates)
        if _within(delta, tol):
            return out(Verdict.OK, f"within seed spread {tol:g}", measured, tol)
        return out(
            Verdict.DIVERGED,
            f"|delta|={abs(delta):g} exceeds seed spread {tol:g}",
            measured,
            tol,
        )

    if kind == MetricKind.CONTINUOUS:
        atol = overrides.continuous_atol.get(column)
        if atol is None:
            return out(Verdict.UNVERIFIED, "noise not measured", measured)
    elif kind == MetricKind.ACCURACY:
        atol = DEFAULT_TOLERANCES[MetricKind.ACCURACY] * overrides.scale.get(
            column,
            1.0,
        )
    else:
        atol = DEFAULT_TOLERANCES[MetricKind.EXACT]
    if _within(delta, atol):
        return out(Verdict.OK, f"within {atol:g}", measured, atol)
    return out(
        Verdict.DIVERGED,
        f"|delta|={abs(delta):g} exceeds {atol:g}",
        measured,
        atol,
    )


def _comparison(  # noqa: PLR0917 -- partially applied by _judge, positionally.
    column: str,
    published: float | None,
    kind: MetricKind,
    verdict: Verdict,
    reason: str,
    measured: float | None = None,
    atol: float | None = None,
) -> Comparison:
    """Build one column's comparison; ``delta`` is derived from the two values."""
    return Comparison(
        column=column,
        published=published,
        measured=measured,
        delta=None if measured is None or published is None else measured - published,
        kind=kind,
        atol=atol,
        verdict=verdict,
        reason=reason,
    )


def _worst(verdicts: Sequence[Verdict]) -> Verdict:
    if Verdict.DIVERGED in verdicts:
        return Verdict.DIVERGED
    if Verdict.UNVERIFIED in verdicts or not verdicts:
        return Verdict.UNVERIFIED
    return Verdict.OK


def _published_for(row: _RowLike, seeds: Sequence[int] | None) -> Mapping[str, float]:
    if seeds is None or len(seeds) == 0 or any(s not in row.seeds for s in seeds):
        return row.values
    out = dict(row.values)
    for col in row.values:
        cells = [row.seeds[s].get(col) for s in seeds]
        if all(c is not None for c in cells):
            out[col] = statistics.fmean(c for c in cells if c is not None)
    return out


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _find_runs(runs_dir: Path, task: str, row: str) -> list[dict[str, float]]:
    for base in (runs_dir / row, runs_dir / task / row):
        dirs = sorted(p.parent for p in base.glob("seed*/metrics.json"))
        if dirs:
            return [read_metrics_file(d) for d in dirs]
        if (base / "metrics.json").exists():
            return [read_metrics_file(base)]
    return []


def _compare_on_disk(
    spec: _TaskLike,
    runs_dir: Path,
    task: str,
    row: str,
    measured: str | None = None,
) -> RowReport:
    """Judge the runs named ``measured`` (default: ``row``'s own) against ``row``."""
    name = measured or row
    seeded = find_seed_runs(runs_dir, task, name)
    if seeded:
        return compare(
            spec,
            row,
            [seeded[k] for k in sorted(seeded)],
            seeds=sorted(seeded),
        )
    return compare(spec, row, _find_runs(runs_dir, task, name))


def _unrun(task: str, row: str) -> RowReport:
    return RowReport(
        task=task,
        row=row,
        seed_count=0,
        comparisons=(),
        verdict=Verdict.UNVERIFIED,
    )


def _summarize(
    spec: _SpecLike,
    runs_dir: Path,
    overrides: TaskOverrides | None,
) -> TaskSummary:
    reports: list[RowReport] = []
    missing: dict[str, tuple[int, ...]] = {}
    for base in spec.baselines:
        seeds = find_seed_runs(runs_dir, spec.name, base.name)
        if not seeds:
            reports.append(_unrun(spec.name, base.name))
            continue
        try:
            rep = compare(
                spec,
                base.name,
                [seeds[k] for k in sorted(seeds)],
                overrides or TaskOverrides(),
                sorted(seeds),
            )
        except (KeyError, ValueError):  # No published row to compare with.
            rep = RowReport(
                task=spec.name,
                row=base.name,
                seed_count=len(seeds),
                comparisons=(),
                verdict=Verdict.UNVERIFIED,
            )
        lacking = tuple(s for s in spec.seeds if s not in seeds)
        if lacking:
            missing[base.name] = lacking
            if rep.verdict == Verdict.OK:
                rep = replace(rep, verdict=Verdict.UNVERIFIED)
        reports.append(rep)
    return TaskSummary(task=spec.name, reports=tuple(reports), missing_seeds=missing)


def _runs_root(runs: Path) -> Path:
    """Accept a base_dir as well as the ``runs/mlsbench`` directory itself."""
    nested = runs / "runs" / "mlsbench"
    return nested if nested.is_dir() else runs


class _Flags(Protocol):
    task: str | None
    runs: str
    row: str
    candidate: str
    root: str
    all: bool
    include_missing: bool
    verbose: bool


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument("--task", default=None)
    parser.add_argument(
        "--all",
        action="store_true",
        help="aggregate every task found under --runs",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="with --all: also count every clone task that has no run as unverified",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--row", default="")
    parser.add_argument(
        "--candidate",
        default="",
        help="with --task and --row: judge this run name (a candidate_experiment) "
        "against the row's published values instead of the row's own runs",
    )
    parser.add_argument(
        "--root",
        default=str(_default_clone()),
        help="the upstream clone",
    )


def _default_clone() -> Path:
    """Return the upstream clone a default ``MlsbenchEval`` resolves."""
    cfg = MlsbenchEval()
    return resolve_working_dir(cfg.base_dir, cfg.mlsbench_root)


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
