"""What one setting runs under, and how its log becomes leaderboard columns.

A setting's container gets the package environment of the task's Harbor bundle
(:func:`package_env`), the package's dataset binds (:func:`data_mounts`) and,
when the bundle ships a generator, held-out inputs (:func:`inputgen_dir`). Its
log goes to the task's own ``parser.py``, loaded as-is from the upstream clone
(:func:`run_parser`).

Parser contract (read from upstream, ``mlsbench/agent/parsers.py``): each task's
``parser.py`` defines ``class Parser(OutputParser)`` whose
``parse(cmd_label, raw_output)`` returns a ``ParseResult`` with a ``metrics``
dict already keyed by the leaderboard column (``<metric>_<label>``). The
parser imports ``mlsbench`` from ``<root>/src``, so that directory is put on
``sys.path`` while it loads; nothing is copied.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import util
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import contextlib
import math
import re
import sys

from treekle import ReadError, from_plain, json_loads


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

    from priml.baselines.mlsbench.task import TaskSpec


class ParseError(RuntimeError):
    """``parser.py`` raised, or returned something that is not finite numbers."""


_ENV_VAR = re.compile(r"\$(\w+)|\$\{([^}:]+)(?::-[^}]*)?\}")
"""Same pattern as Harbor's verifier: ``$X``, ``${X}``, ``${X:-d}`` (the default is
ignored, an unset name expands to empty)."""


def expand_env(value: str, base: Mapping[str, str]) -> str:
    """Expand ``$X`` templates against ``base`` like Harbor's ``_expand_env_template``."""

    def repl(m: re.Match[str]) -> str:
        return base.get(m.group(1) or m.group(2) or "", "")

    return _ENV_VAR.sub(repl, value)


class PackageConfigError(RuntimeError):
    """A package config under ``vendor/pkg_configs`` exists but is unreadable."""


def pkg_config(root: Path, package: str) -> dict[str, object]:
    """Return ``vendor/pkg_configs/<package>/config.json`` of a clone, or ``{}``.

    The package name is matched case-insensitively, as upstream spells it both ways.

    Args:
      root: The upstream clone.
      package: Package name as the task config spells it.

    Returns:
      config: The parsed object, or ``{}`` when there is none.

    Raises:
      PackageConfigError: The config exists but is not a JSON object.

    """
    base = root / "vendor" / "pkg_configs"
    want = package.lower()
    for d in sorted(base.iterdir()) if base.is_dir() else []:
        if d.name.lower() == want and (d / "config.json").is_file():
            try:
                return json_loads((d / "config.json").read_text(), dict[str, object])
            except ReadError as e:
                raise PackageConfigError(
                    f"{d.name}: pkg config is not an object",
                ) from e
            except (OSError, ValueError) as e:
                raise PackageConfigError(
                    f"{d.name}: unreadable pkg config ({e})",
                ) from e
    return {}


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
        env = pkg_config(task.root.parent.parent, package).get("env")
        raw = from_plain(env, dict[str, object], default={})
        block = {k: str(v) for k, v in raw.items()}
    out: dict[str, str] = {}
    for key, value in block.items():
        out[key] = value if key == "HOME" else expand_env(value, out)
    for key, value in (overlay or {}).items():
        out[key] = expand_env(value, out)
    out.setdefault("DATA_ROOT", "/data")
    return tuple(out.items())


def inputgen_dir(bundle: Path | None) -> Path | None:
    """Return the bundle's ``tests/eval/_inputgen`` if it ships an ``apply.py``."""
    if bundle is None:
        return None
    d = bundle / "tests" / "eval" / "_inputgen"
    return d if (d / "apply.py").is_file() else None


@dataclass(frozen=True, kw_only=True, slots=True)
class Mounts:
    """Data mounts of one setting, or why they cannot be provided."""

    binds: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    """Dataset binds dropped because the host directory is absent."""

    missing: str = ""
    """Set when a required input is unavailable: the setting must not run."""


def data_mounts(task: TaskSpec, package: str, *, data_root: Path | str = "") -> Mounts:
    """Resolve a package's ``data_bind`` and ``data_deps`` into ``-v`` specs.

    Follows upstream's native runner: ``{data_root}`` and ``{project_root}`` are
    expanded, an explicit ``data_bind`` comes first, and each ``data_deps`` entry
    adds ``host:container`` unless already present. A host directory that is
    absent is skipped, never created: the image may bake that data in and an empty
    bind would mask it (upstream's own rule). ``""`` binds nothing.

    Args:
      task: The parsed task; locates the clone.
      package: The setting's package.
      data_root: Host directory of datasets and model weights.

    Returns:
      mounts: The binds, the skipped ones, and why the inputs are missing, if they are.

    """
    if not str(data_root):
        return Mounts()
    clone = task.root.parent.parent
    try:
        cfg = pkg_config(clone, package)
    except PackageConfigError as e:
        return Mounts(missing=f"pkg config: {e}")

    raw = cfg.get("data_bind")
    entries = (
        cast("list[object]", raw) if isinstance(raw, list) else [raw] if raw else []
    )
    roots = {
        "{project_root}": str(clone),
        "{data_root}": str(Path(data_root).expanduser()),
    }
    specs = [_expand(e, roots) for e in entries if isinstance(e, str)]
    deps = cfg.get("data_deps")
    for dep in cast("list[object]", deps) if isinstance(deps, list) else []:
        fields = cast("dict[str, object]", dep) if isinstance(dep, dict) else {}
        host, cont = fields.get("host_path"), fields.get("container_path")
        if isinstance(host, str) and isinstance(cont, str) and host and cont:
            spec = f"{_expand(host, roots)}:{cont}"
            if spec not in specs:
                specs.append(spec)
    keep = tuple(s for s in specs if Path(s.split(":", 1)[0]).exists())
    return Mounts(binds=keep, skipped=tuple(s for s in specs if s not in keep))


_NOT_PARSED: Final = ("elapsed_",)
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
    except Exception as e:  # Upstream code: any failure is a parse failure.
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
        if math.isfinite(x):  # upstream's scorer drops a non-finite cell alone.
            out[str(key)] = x
    return out


def _json_env(path: Path) -> dict[str, dict[str, str]]:
    try:
        raw = json_loads(path.read_text(), dict[str, dict[str, object]])
    except (OSError, ValueError):
        return {}
    return {pkg: {k: str(v) for k, v in env.items()} for pkg, env in raw.items()}


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


# Harbor's ``score_task.py`` runs ``tests/meta/parser.py``, a byte-identical copy of the
# task's. Parsers that find ``dgp.py`` next to their own file (``optimization-variance-
# reduction``) need that location.
def _parser_path(task: TaskSpec, meta: Path | None) -> Path:
    """Return the parser file to execute: the bundle's copy when there is one."""
    if meta is not None and (meta / "parser.py").is_file():
        return meta / "parser.py"
    return task.parser


# The clone's ``src`` and the bundle's ``tests/meta`` stay on ``sys.path``, and hosted
# modules stay registered, until the caller is done: upstream parsers also import inside
# ``parse`` (``optimization-variance-reduction`` loads ``dgp`` lazily).
@contextlib.contextmanager
def _loaded_parser(
    path: Path,
    src: Path,
    meta: Path | None = None,
) -> Generator[Callable[[str, str], object]]:
    """Import upstream's parser and yield its ``parse``, with its imports reachable."""
    name = f"mlsbench_parser_{path.parent.name.replace('-', '_')}"
    spec = util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ParseError(f"cannot load parser {path}")
    module = util.module_from_spec(spec)
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


def _expand(text: str, roots: Mapping[str, str]) -> str:
    """Substitute each ``{name}`` placeholder of ``roots`` in ``text``."""
    for key, value in roots.items():
        text = text.replace(key, value)
    return text
