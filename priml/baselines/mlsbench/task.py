"""Typed, strict view of one MLS-Bench task directory.

The schema below was read from the pinned upstream clone (commit ``80cf5c5c``,
all 140 tasks), not guessed. Layout read (``<root>/tasks/<name>/``)::

    config.json        strict schema below; any extra key is an error
    leaderboard.csv    long format, see below
    parser.py          ``class Parser(OutputParser)``; ``parse(label, raw) -> ParseResult``
    score_spec.py      optional
    edits/*.edit.py    one per baseline that has ``edit_ops``
    scripts/*          one per setting

``config.json`` (the task name is the directory name; there is no ``name``,
``family`` or ``package`` key: family is the name's first ``-`` token and the
package is the first setting's)::

    required
      test_cmds  [{cmd, label, group:int, compute:num, time:"H:MM:SS", package,
                   hidden?:bool, mem?:int, h200?:{compute?:num, env?:{str:str}}}]
      baselines  {name: {edit_ops?: "edits/x.edit.py", cmd?: "scripts/x.sh",
                         env?: {str:str}, labels?: [str]}}   # {} = the codebase as shipped
      files      [{filename, read:[{start,end}], edit?:[{start,end,_comment?}], _comment?}]
    optional
      seeds:[int]  allow_create:bool  rigorous_codebase:bool  use_cuda:bool
      requires_network:bool  ephemeral_inputs:bool  mid_edit:str
      guard_exclude:[str]  hidden_metrics:[str]  leaderboard:{baseline:{metric:num}}

A ``read`` range of ``-1,-1`` means the whole file. An absent ``seeds`` means
``(42,)``, as upstream's ``task_config.get("seeds") or global.get("seeds", [42])``.

``leaderboard.csv`` header is ``timestamp,model,is_final,seed,<metric columns>``;
``model`` is ``baseline:<name>``; ``seed`` is an integer or ``mean`` (an
aggregate row over the seeds). A blank cell is unpublished. Columns are named by
the task's own parser, so they do not always end in a setting label.

Editable files live in the package codebase (the task image), not in the task
directory, so their existence is not checked here.
"""

# ruff: noqa: TRY301
# (errors are raised from helpers by design)

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, cast

import csv
import itertools
import json
import math
import re
import statistics


_TIME: Final = re.compile(r"^(\d+):([0-5]\d):([0-5]\d)$")
_MEAN: Final = "mean"
_MODEL_PREFIX: Final = "baseline:"
_LB_FIXED: Final = ("timestamp", "model", "is_final", "seed")
_DEFAULT_SEEDS: Final = (42,)


class TaskSpecError(ValueError):
    """Any schema violation; the message names the file and json path."""


class UnknownFieldError(TaskSpecError):
    """An extra key anywhere in ``config.json``."""


class MissingFileError(TaskSpecError, FileNotFoundError):
    """A required file or directory is absent."""


@dataclass(frozen=True)
class EditableFile:
    """One declared file: package-relative path, editable and readable line ranges."""

    path: str
    ranges: tuple[tuple[int, int], ...]
    """Editable 1-indexed inclusive ranges; ``()`` is declared read-only.

    ``end == -1`` extends to EOF and ``(-1, -1)`` is the whole file, as upstream.
    """

    read_ranges: tuple[tuple[int, int], ...] = ()
    """Ranges shown to the agent, same conventions."""


@dataclass(frozen=True)
class H200Override:
    """A setting's override when the job runs on H200 cards."""

    compute: float | None = None
    env: Mapping[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True)
class Setting:
    """One evaluation setting: a script, its label, and its budget."""

    label: str
    script: str
    time_limit: float
    """Seconds (``time`` is ``H:MM:SS`` in ``config.json``)."""

    compute: float
    """GPUs' worth of compute the setting occupies (may exceed 1; 0 is CPU-only)."""

    group: int = 1
    package: str = ""
    hidden: bool = False
    mem: int | None = None
    h200: H200Override | None = None

    def compute_for(self, hardware: str = "H100") -> float:
        """Return the compute share on ``hardware`` (``"H100"`` or ``"H200"``)."""
        if (
            hardware == "H200"
            and self.h200 is not None
            and self.h200.compute is not None
        ):
            return self.h200.compute
        return self.compute


@dataclass(frozen=True)
class Baseline:
    """A published baseline: leaderboard row name and how it is produced."""

    name: str
    edit: Path | None = None
    """``edits/<x>.edit.py``; ``None`` for a baseline that is a ``cmd`` or the shipped code."""

    cmd: str | None = None
    """Script that replaces the settings' commands (``labels`` selects which)."""

    env: Mapping[str, str] = field(default_factory=dict[str, str])
    labels: tuple[str, ...] | None = None


@dataclass(frozen=True)
class LeaderboardRow:
    """One baseline's published values, aggregated over its published seeds."""

    name: str
    values: Mapping[str, float]
    """Column to published value: the ``mean`` row if present, else the mean of the seeds."""

    spread: Mapping[str, float] = field(default_factory=dict[str, float])
    """Column to sample standard deviation over seeds (only with two or more)."""

    seeds: Mapping[int, Mapping[str, float]] = field(
        default_factory=dict[int, Mapping[str, float]],
    )
    """Per-seed published values."""

    is_final: bool = True


@dataclass(frozen=True)
class TaskSpec:
    """A fully parsed and cross-validated task."""

    name: str
    family: str
    package: str
    root: Path
    seeds: tuple[int, ...]
    files: tuple[EditableFile, ...]
    settings: tuple[Setting, ...]
    baselines: tuple[Baseline, ...]
    leaderboard: tuple[LeaderboardRow, ...]
    parser: Path
    score_spec: Path | None
    seeds_declared: bool = True
    allow_create: bool = False
    rigorous_codebase: bool = False
    use_cuda: bool = False
    requires_network: bool = False
    ephemeral_inputs: bool = False
    mid_edit: Path | None = None
    guard_exclude: tuple[str, ...] = ()
    hidden_metrics: tuple[str, ...] = ()
    inline_leaderboard: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict[str, Mapping[str, float]],
    )

    def setting(self, label: str) -> Setting:
        """Return the setting with this label."""
        for s in self.settings:
            if s.label == label:
                return s
        raise TaskSpecError(
            f"{self.name}: no setting {label!r}; have {[s.label for s in self.settings]}",
        )

    def baseline(self, name: str) -> Baseline:
        """Baseline by name."""
        for b in self.baselines:
            if b.name == name:
                return b
        raise TaskSpecError(
            f"{self.name}: no baseline {name!r}; have {[b.name for b in self.baselines]}",
        )

    def row(self, name: str) -> LeaderboardRow:
        """Leaderboard row by name."""
        for r in self.leaderboard:
            if r.name == name:
                return r
        raise TaskSpecError(f"{self.name}: no leaderboard row {name!r}")

    def visible_settings(self) -> tuple[Setting, ...]:
        """Return the settings not hidden from the agent."""
        return tuple(s for s in self.settings if not s.hidden)

    @property
    def packages(self) -> tuple[str, ...]:
        """Distinct packages over the settings, in first-seen order."""
        return tuple(dict.fromkeys(s.package for s in self.settings))


# -- strict json helpers ---------------------------------------------------


def _obj(
    v: object,
    where: str,
    required: set[str],
    optional: frozenset[str] | set[str] = frozenset(),
) -> dict[str, object]:
    if not isinstance(v, dict):
        raise TaskSpecError(f"{where}: expected object, got {type(v).__name__}")
    d = cast("dict[str, object]", v)  # json object keys are strings
    extra = set(d) - required - set(optional)
    if extra:
        raise UnknownFieldError(f"{where}: unknown field(s) {sorted(extra)}")
    missing = required - set(d)
    if missing:
        raise TaskSpecError(f"{where}: missing field(s) {sorted(missing)}")
    return d


def _check(v: object, typ: type | tuple[type, ...], where: str) -> None:
    ok = isinstance(v, typ) and not (
        isinstance(v, bool) and bool not in (typ if isinstance(typ, tuple) else (typ,))
    )
    if not ok:
        raise TaskSpecError(f"{where}: expected {typ}, got {v!r}")


def _typed[T](v: object, typ: type[T], where: str) -> T:
    _check(v, typ, where)
    return cast("T", v)


def _str(v: object, where: str) -> str:
    s = _typed(v, str, where)
    if not s:
        raise TaskSpecError(f"{where}: empty string")
    return s


def _num(v: object, where: str) -> float:
    _check(v, (int, float), where)
    n = float(cast("float", v))
    if not math.isfinite(n):
        raise TaskSpecError(f"{where}: not finite")
    return n


def _int(v: object, where: str) -> int:
    return _typed(v, int, where)


def _bool(v: object, where: str) -> bool:
    return _typed(v, bool, where)


def _list(v: object, where: str) -> list[object]:
    _check(v, list, where)
    return cast("list[object]", v)


def _dict(v: object, where: str) -> dict[object, object]:
    _check(v, dict, where)
    return cast("dict[object, object]", v)


def _strs(v: object, where: str) -> tuple[str, ...]:
    return tuple(_str(x, f"{where}[{i}]") for i, x in enumerate(_list(v, where)))


def _env(v: object, where: str) -> dict[str, str]:
    d = _dict(v, where)
    return {
        _str(k, f"{where}.key"): _typed(x, str, f"{where}.{k}") for k, x in d.items()
    }


def _relpath(v: object, where: str) -> str:
    path = _str(v, where)
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise TaskSpecError(
            f"{where}: must be relative and inside its dir, got {path!r}",
        )
    return path


def _ranges(raw: object, where: str) -> tuple[tuple[int, int], ...]:
    out: list[tuple[int, int]] = []
    for j, r in enumerate(_list(raw, where)):
        rw = f"{where}[{j}]"
        d = _obj(r, rw, {"start", "end"}, {"_comment"})
        if "_comment" in d:
            _ = _typed(d["_comment"], str, f"{rw}._comment")
        a, b = _int(d["start"], f"{rw}.start"), _int(d["end"], f"{rw}.end")
        whole = (a, b) == (-1, -1)
        if not whole and (a < 1 or (b != -1 and a > b)):
            raise TaskSpecError(
                f"{rw}: need 1 <= start <= end (or end -1), got [{a}, {b}]",
            )
        out.append((a, b))
    if len(out) > 1 and (-1, -1) in out:
        raise TaskSpecError(f"{where}: (-1, -1) whole-file range must stand alone")
    out.sort()
    for (_, e1), (s2, _) in itertools.pairwise(out):
        if e1 == -1 or s2 <= e1:
            raise TaskSpecError(f"{where}: overlapping ranges")
    return tuple(out)


def _parse_files(raw: object, where: str) -> tuple[EditableFile, ...]:
    out: list[EditableFile] = []
    seen: set[str] = set()
    for i, item in enumerate(_list(raw, where)):
        w = f"{where}[{i}]"
        d = _obj(item, w, {"filename", "read"}, {"edit", "_comment"})
        if "_comment" in d:
            _ = _typed(d["_comment"], str, f"{w}._comment")
        path = _relpath(d["filename"], f"{w}.filename")
        if path in seen:
            raise TaskSpecError(f"{w}.filename: duplicate {path!r}")
        seen.add(path)
        out.append(
            EditableFile(
                path,
                _ranges(d.get("edit", []), f"{w}.edit"),
                _ranges(d["read"], f"{w}.read"),
            ),
        )
    return tuple(out)


def parse_time(v: object, where: str = "time") -> float:
    """Parse ``H:MM:SS`` into seconds; anything else is an error."""
    m = _TIME.match(_str(v, where))
    if m is None:
        raise TaskSpecError(f"{where}: expected H:MM:SS, got {v!r}")
    h, mi, se = (int(x) for x in m.groups())
    seconds = h * 3600 + mi * 60 + se
    if seconds <= 0:
        raise TaskSpecError(f"{where}: must be > 0")
    return float(seconds)


def _parse_settings(raw: object, where: str) -> tuple[Setting, ...]:
    out: list[Setting] = []
    labels: set[str] = set()
    items = _list(raw, where)
    if not items:
        raise TaskSpecError(f"{where}: empty")
    for i, item in enumerate(items):
        w = f"{where}[{i}]"
        d = _obj(
            item,
            w,
            {"cmd", "label", "group", "compute", "time", "package"},
            {"hidden", "h200", "mem"},
        )
        label = _str(d["label"], f"{w}.label")
        if label in labels:
            raise TaskSpecError(f"{w}.label: duplicate setting label {label!r}")
        labels.add(label)
        cp = _num(d["compute"], f"{w}.compute")
        if cp < 0:
            raise TaskSpecError(f"{w}.compute: must be >= 0 (0 is CPU-only)")
        h200: H200Override | None = None
        if "h200" in d:
            hd = _obj(d["h200"], f"{w}.h200", set(), {"compute", "env"})
            hc = _num(hd["compute"], f"{w}.h200.compute") if "compute" in hd else None
            if hc is not None and hc <= 0:
                raise TaskSpecError(f"{w}.h200.compute: must be > 0")
            h200 = H200Override(hc, _env(hd.get("env", {}), f"{w}.h200.env"))
        mem = _int(d["mem"], f"{w}.mem") if "mem" in d else None
        if mem is not None and mem <= 0:
            raise TaskSpecError(f"{w}.mem: must be > 0")
        out.append(
            Setting(
                label=label,
                script=_relpath(d["cmd"], f"{w}.cmd"),
                time_limit=parse_time(d["time"], f"{w}.time"),
                compute=cp,
                group=_int(d["group"], f"{w}.group"),
                package=_str(d["package"], f"{w}.package"),
                hidden=_bool(d.get("hidden", False), f"{w}.hidden"),
                mem=mem,
                h200=h200,
            ),
        )
    return tuple(out)


def _parse_baselines(
    raw: object,
    where: str,
    task_dir: Path,
    labels: set[str],
) -> tuple[Baseline, ...]:
    out: list[Baseline] = []
    for name, item in _dict(raw, where).items():
        bw = f"{where}.{name}"
        d = _obj(item, bw, set(), {"edit_ops", "cmd", "env", "labels"})
        edit = _relpath(d["edit_ops"], f"{bw}.edit_ops") if "edit_ops" in d else None
        cmd = _relpath(d["cmd"], f"{bw}.cmd") if "cmd" in d else None
        bl = _strs(d["labels"], f"{bw}.labels") if "labels" in d else None
        if bl is not None:
            unknown = set(bl) - labels
            if unknown:
                raise TaskSpecError(f"{bw}.labels: not settings: {sorted(unknown)}")
        out.append(
            Baseline(
                _str(name, f"{where}.<name>"),
                None if edit is None else task_dir / edit,
                cmd,
                _env(d.get("env", {}), f"{bw}.env"),
                bl,
            ),
        )
    return tuple(out)


def _cells(path: Path, n: int, header: list[str], cells: list[str]) -> dict[str, float]:
    vals: dict[str, float] = {}
    for col, cell in zip(header[4:], cells[4:], strict=True):
        text = cell.strip()
        if not text:
            continue
        try:
            x = float(text)
        except ValueError as e:
            raise TaskSpecError(
                f"{path}:{n}: column {col!r}: non-numeric {text!r}",
            ) from e
        if not math.isfinite(x):
            raise TaskSpecError(f"{path}:{n}: column {col!r}: not finite")
        vals[col] = x
    return vals


def parse_leaderboard(path: Path) -> tuple[LeaderboardRow, ...]:
    """Read a long-format ``leaderboard.csv`` into one row per baseline.

    Args:
      path: The csv file.

    Returns:
      rows: One per ``model``, in first-seen order.

    Raises:
      MissingFileError: ``path`` is absent.
      TaskSpecError: Bad header, model, seed, boolean, duplicate (model, seed) or cell.

    """
    if not path.is_file():
        raise MissingFileError(f"{path}: missing")
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, list[str]())
        if not header or tuple(header[:4]) != _LB_FIXED:
            raise TaskSpecError(f"{path}: header must start with {list(_LB_FIXED)}")
        if len(set(header)) != len(header):
            raise TaskSpecError(f"{path}: duplicate columns")
        per_seed: dict[str, dict[int, dict[str, float]]] = {}
        means: dict[str, dict[str, float]] = {}
        final: dict[str, bool] = {}
        seen: set[tuple[str, str]] = set()
        for n, cells in enumerate(reader, start=2):
            if not cells:
                continue
            if len(cells) != len(header):
                raise TaskSpecError(
                    f"{path}:{n}: expected {len(header)} cells, got {len(cells)}",
                )
            model = cells[1]
            if not model.startswith(_MODEL_PREFIX) or model == _MODEL_PREFIX:
                raise TaskSpecError(
                    f"{path}:{n}: model must be 'baseline:<name>', got {model!r}",
                )
            name = model.removeprefix(_MODEL_PREFIX)
            if cells[2] not in ("true", "false"):
                raise TaskSpecError(
                    f"{path}:{n}: is_final must be true|false, got {cells[2]!r}",
                )
            if (name, cells[3]) in seen:
                raise TaskSpecError(
                    f"{path}:{n}: duplicate row {name!r} seed {cells[3]!r}",
                )
            seen.add((name, cells[3]))
            final[name] = final.get(name, True) and cells[2] == "true"
            per_seed.setdefault(name, {})
            vals = _cells(path, n, header, cells)
            if cells[3] == _MEAN:
                means[name] = vals
            else:
                try:
                    seed = int(cells[3])
                except ValueError as e:
                    raise TaskSpecError(
                        f"{path}:{n}: seed must be an integer or 'mean', got {cells[3]!r}",
                    ) from e
                per_seed[name][seed] = vals
    rows: list[LeaderboardRow] = []
    for name, seeds in per_seed.items():
        cols = dict.fromkeys(c for v in seeds.values() for c in v)
        values = dict(means.get(name, {}))
        for c in cols:
            if c not in values:
                values[c] = statistics.fmean(v[c] for v in seeds.values() if c in v)
        spread = {
            c: statistics.stdev(xs)
            for c in cols
            if len(xs := [v[c] for v in seeds.values() if c in v]) > 1
        }
        rows.append(LeaderboardRow(name, values, spread, seeds, final[name]))
    return tuple(rows)


def _parse_inline_leaderboard(
    raw: object,
    where: str,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, metrics in _dict(raw, where).items():
        m = _dict(metrics, f"{where}.{name}")
        out[_str(name, where)] = {
            _str(k, f"{where}.{name}"): _num(v, f"{where}.{name}.{k}")
            for k, v in m.items()
        }
    return out


def task_names(root: Path) -> tuple[str, ...]:
    """Sorted names of every task directory (one holding a ``config.json``)."""
    base = Path(root)
    tasks = base / "tasks"
    if not tasks.is_dir():
        raise MissingFileError(f"{tasks}: missing")
    return tuple(
        sorted(d.name for d in tasks.iterdir() if (d / "config.json").is_file()),
    )


def load_all(root: Path) -> tuple[TaskSpec, ...]:
    """Parse every task under the upstream clone ``root``, strictly."""
    base = Path(root)
    return tuple(load_task(n, root=base) for n in task_names(base))


def load_task(name: str, *, root: Path) -> TaskSpec:
    """Parse ``<root>/tasks/<name>`` of the upstream clone ``root``."""
    base = Path(root)
    task_dir = (base / "tasks" / name).resolve()
    cfg_path = task_dir / "config.json"
    if not cfg_path.is_file():
        raise MissingFileError(f"{cfg_path}: missing")
    try:
        raw = cast("object", json.loads(cfg_path.read_text()))
    except json.JSONDecodeError as e:
        raise TaskSpecError(f"{cfg_path}: invalid json: {e}") from e
    w = "config.json"
    try:
        d = _obj(
            raw,
            w,
            {"test_cmds", "baselines", "files"},
            {
                "seeds",
                "allow_create",
                "rigorous_codebase",
                "use_cuda",
                "requires_network",
                "ephemeral_inputs",
                "mid_edit",
                "guard_exclude",
                "hidden_metrics",
                "leaderboard",
            },
        )
        seeds_declared = "seeds" in d
        seeds = (
            tuple(
                _int(s, f"{w}.seeds[{i}]")
                for i, s in enumerate(_list(d["seeds"], f"{w}.seeds"))
            )
            if seeds_declared
            else _DEFAULT_SEEDS
        )
        if not seeds or len(set(seeds)) != len(seeds):
            raise TaskSpecError(f"{w}.seeds: empty or duplicated")
        files = _parse_files(d["files"], f"{w}.files")
        settings = _parse_settings(d["test_cmds"], f"{w}.test_cmds")
        baselines = _parse_baselines(
            d["baselines"],
            f"{w}.baselines",
            task_dir,
            {s.label for s in settings},
        )
        if not baselines:
            raise TaskSpecError(f"{w}.baselines: empty")
        mid = _relpath(d["mid_edit"], f"{w}.mid_edit") if "mid_edit" in d else None
        flags = {
            k: _bool(d.get(k, False), f"{w}.{k}")
            for k in (
                "allow_create",
                "rigorous_codebase",
                "use_cuda",
                "requires_network",
                "ephemeral_inputs",
            )
        }
        guard = _strs(d.get("guard_exclude", []), f"{w}.guard_exclude")
        hidden_metrics = _strs(d.get("hidden_metrics", []), f"{w}.hidden_metrics")
        inline = _parse_inline_leaderboard(d.get("leaderboard", {}), f"{w}.leaderboard")
    except TaskSpecError as e:
        raise type(e)(f"{cfg_path}: {e}") from None

    for b in baselines:
        if b.edit is not None and not b.edit.is_file():
            raise MissingFileError(
                f"{cfg_path}: baseline {b.name!r}: edit file {b.edit} missing",
            )
        if b.cmd is not None and not (task_dir / b.cmd).is_file():
            raise MissingFileError(
                f"{cfg_path}: baseline {b.name!r}: cmd script {task_dir / b.cmd} missing",
            )
    for s in settings:
        if not (task_dir / s.script).is_file():
            raise MissingFileError(
                f"{cfg_path}: setting {s.label!r}: script {task_dir / s.script} missing",
            )
    mid_path = None if mid is None else task_dir / mid
    if mid_path is not None and not mid_path.is_file():
        raise MissingFileError(f"{cfg_path}: mid_edit {mid_path} missing")
    parser = task_dir / "parser.py"
    if not parser.is_file():
        raise MissingFileError(f"{parser}: missing")
    score = task_dir / "score_spec.py"

    lb_path = task_dir / "leaderboard.csv"
    leaderboard = parse_leaderboard(lb_path)
    rows = {r.name for r in leaderboard}
    for b in baselines:
        if b.name not in rows:
            raise TaskSpecError(
                f"{lb_path}: baseline {b.name!r} has no leaderboard row",
            )
    for extra in sorted(set(inline) - {b.name for b in baselines}):
        raise TaskSpecError(
            f"{cfg_path}: inline leaderboard row {extra!r} is no baseline",
        )

    return TaskSpec(
        name=name,
        family=name.split("-", 1)[0],
        package=settings[0].package,
        root=task_dir,
        seeds=seeds,
        files=files,
        settings=settings,
        baselines=baselines,
        leaderboard=leaderboard,
        parser=parser,
        score_spec=score if score.is_file() else None,
        seeds_declared=seeds_declared,
        mid_edit=mid_path,
        guard_exclude=guard,
        hidden_metrics=hidden_metrics,
        inline_leaderboard=inline,
        **flags,
    )
