"""Reader and verifier for Harbor bundles (read at runtime, never copied).

A Harbor bundle (``<clone>/harbor/tasks-docker/mls-bench__<task>/``) carries
under ``tests/meta/``:

- ``config.json``: ``files[]`` with ``filename`` and ``edit`` ranges (``start``/``end``,
  1-indexed inclusive; ``-1`` means whole file / to EOF; empty means read-only),
  optional ``guard_exclude`` prefixes;
- ``pristine_manifest.json``: ``{relpath: sha256}`` of every file under a guarded prefix;
- ``pristine/<relpath>``: pristine bytes of each declared file (for the segment diff).

The verifier semantics implemented here (independently, from the documented behaviour):

1. *Fail closed*: a missing, malformed or empty manifest refuses to verify.
2. *Byte-segment diff*: the pristine file is cut into fixed segments between the editable
   ranges; each fixed segment must survive verbatim and in order in the edited file
   (first/last segment anchored at the file ends, intermediate ones at their rightmost
   feasible occurrence). Whatever lies between is the edit, of any length. Unlike upstream's
   ``read_text`` this works on raw bytes, so CRLF or non-UTF-8 changes are not normalised away.
3. *Manifest hash check* for undeclared files under guarded prefixes (modification and,
   when the whole tree is present, deletion). New files are not violations.
4. *Sanitised environment*: fixed ``PATH``, every interpreter-influencing variable dropped.

Extra strictness over upstream (documented deviation): symlinks at or above a guarded path
are violations rather than being followed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final, NotRequired, TypedDict, cast

import difflib
import filecmp
import functools
import hashlib
import re
import tomllib

from treekle import json_loads


if TYPE_CHECKING:
    from collections.abc import Mapping


_CWD: Final = Path(__file__).resolve().parent

SAFE_PATH: Final = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


_SCRUB: Final = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONIOENCODING",
    "PYTHONHASHSEED",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
)


_EXEMPT_TOP: Final = frozenset({"_task"})


class BundleError(RuntimeError):
    """The Harbor bundle is missing or malformed (the guard fails closed)."""


@dataclass(frozen=True, kw_only=True, slots=True)
class HarborViolation:
    """One breach found by :func:`guard`; ``kind`` is a stable slug."""

    path: str
    kind: str
    detail: str = ""


@dataclass(frozen=True, kw_only=True, slots=True)
class HarborBundle:
    """Parsed ``tests/meta`` of one Harbor bundle."""

    root: Path
    manifest: dict[str, str]
    editable: dict[str, tuple[tuple[int, int], ...]]  # () = declared read-only.
    guard_exclude: tuple[str, ...]
    pristine_root: Path

    @property
    def guarded_prefixes(self) -> frozenset[str]:
        """Top-level directories the guard protects."""
        names = list(self.editable) + list(self.manifest)
        return frozenset(PurePosixPath(n).parts[0] for n in names if n)


def sanitised_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for verifier/edit subprocesses: fixed PATH, interpreter vars dropped."""
    env = {k: v for k, v in (base or {}).items() if k not in _SCRUB}
    env["PATH"] = SAFE_PATH
    env["PYTHONNOUSERSITE"] = "1"
    return env


def bundle_dir(task: str, *, root: Path, flavor: str = "docker") -> Path:
    """Locate ``harbor/tasks-<flavor>/mls-bench__<task>`` under the upstream checkout."""
    base = Path(root) / "harbor" / f"tasks-{flavor}"
    path = base / f"mls-bench__{task}"
    if not (path / "tests" / "meta").is_dir():
        raise BundleError(f"{path}: no Harbor bundle (tests/meta missing)")
    return path


def load_bundle(bundle: Path) -> HarborBundle:
    """Read ``tests/meta`` from a bundle directory; raise :class:`BundleError` if unusable.

    Args:
      bundle: The bundle directory.

    Returns:
      hb: Its manifest, editable ranges, guard excludes and pristine root.

    """
    meta = Path(bundle) / "tests" / "meta"
    try:
        raw = json_loads(
            (meta / "pristine_manifest.json").read_text(),
            dict[str, str],
        )
    except FileNotFoundError as e:
        raise BundleError("pristine_manifest.json missing: refusing to verify") from e
    except ValueError as e:
        raise BundleError("pristine_manifest.json malformed") from e
    if not raw:
        raise BundleError("pristine_manifest.json empty: refusing to verify")
    if not all(len(v) == 64 for v in raw.values()):
        raise BundleError("pristine_manifest.json: values must be sha256 hex")
    manifest = {_safe_rel(k): v for k, v in raw.items()}
    try:
        cfg = json_loads((meta / "config.json").read_text(), _BundleConfig)
    except (FileNotFoundError, ValueError) as e:
        raise BundleError(f"config.json unusable: {e}") from e
    editable = {
        _safe_rel(f["filename"]): tuple(
            (r["start"], r["end"]) for r in f.get("edit") or []
        )
        for f in cfg.get("files", [])
    }
    excl = tuple(_safe_rel(e) for e in cfg.get("guard_exclude") or [])
    return HarborBundle(
        root=Path(bundle),
        manifest=manifest,
        editable=editable,
        guard_exclude=excl,
        pristine_root=meta / "pristine",
    )


def find_bundle(task: str, *, root: Path) -> Path | None:
    """Return the Harbor bundle of ``task`` under the upstream clone ``root``, or None.

    ``None`` means the checkout has no bundle for it (a synthetic fixture, or a
    clone without ``harbor/``); it is not an error here, the caller decides.

    Args:
      task: Task name.
      root: The upstream clone.

    Returns:
      bundle: ``harbor/tasks-docker/mls-bench__<task>``, or ``None``.

    """
    try:
        return bundle_dir(task, root=root)
    except BundleError:
        return None


def pristine_source(hb: HarborBundle, rel: str) -> Path | None:
    """Locate the pristine bytes of a declared file inside the bundle.

    Order: ``tests/meta/pristine/<rel>``, then the image scaffold
    ``environment/_scaffold/<rel>`` (whole-file editable files that a task ships
    empty are only in the scaffold). The caller checks the manifest hash.

    Args:
      hb: The parsed bundle.
      rel: Path of the declared file, relative to the workspace.

    Returns:
      path: The pristine copy, or ``None`` when the bundle has none.

    """
    for base in (hb.pristine_root, hb.root / "environment" / "_scaffold"):
        p = base / rel
        if p.is_file() and not p.is_symlink():
            return p
    return None


def bundle_gpus(bundle: Path) -> int | None:
    """Read ``[environment] gpus`` from the bundle's ``task.toml``.

    Args:
      bundle: The bundle directory.

    Returns:
      gpus: The declared GPU count; ``0`` for a CPU-only task and, as in
        Harbor, when ``[environment]`` omits ``gpus``. ``None`` when there is no
        readable ``task.toml`` or ``[environment]`` table, or the value is not
        an integer.

    """
    try:
        doc = tomllib.loads((Path(bundle) / "task.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    env = doc.get("environment")
    table = cast("dict[str, object]", env) if isinstance(env, dict) else None
    value = table.get("gpus", 0) if table is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def allow_internet(bundle: Path) -> bool | None:
    """Read ``[environment] allow_internet`` from the bundle's ``task.toml``.

    Args:
      bundle: The bundle directory.

    Returns:
      allow: The declared value, or ``None`` when absent or unreadable (callers
        must treat that as *no network*).

    """
    try:
        doc = tomllib.loads((Path(bundle) / "task.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    env = doc.get("environment")
    table = cast("dict[str, object]", env) if isinstance(env, dict) else None
    value = table.get("allow_internet") if table is not None else None
    return value if isinstance(value, bool) else None


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes, streamed."""
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check_segments(
    pristine: bytes,
    current: bytes,
    ranges: tuple[tuple[int, int], ...],
) -> str | None:
    """Return None if ``current`` differs from ``pristine`` only inside ``ranges``, else why not.

    ``ranges`` are 1-indexed inclusive pristine lines; ``(-1, -1)`` is whole-file editable,
    an ``end`` of -1 means to EOF, and an empty tuple means read-only.

    Args:
      pristine: The pristine file's bytes.
      current: The edited file's bytes.
      ranges: Editable ranges, as described above.

    Returns:
      why: ``None`` when the edit stays in range, else the first violation.

    """
    if pristine == current:
        return None
    if any(r == (-1, -1) for r in ranges):
        return None
    if not ranges:
        return "modified but file is declared read-only"
    lines = pristine.splitlines(keepends=True)
    n = len(lines)
    norm = sorted((max(s, 1), n if e == -1 else min(e, n)) for s, e in ranges)
    merged: list[tuple[int, int]] = []
    for s, e in norm:
        if merged and s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    segs: list[bytes] = []
    cur = 0
    for s, e in merged:
        if s - 1 > cur:
            segs.append(b"".join(lines[cur : s - 1]))
        cur = max(cur, e)
    if cur < n:
        segs.append(b"".join(lines[cur:]))
    lead = merged[0][0] > 1
    trail = merged[-1][1] < n
    span: list[tuple[int, int] | None] = [None] * len(segs)
    if segs and lead:
        if not current.startswith(segs[0]):
            return "leading fixed segment changed"
        span[0] = (0, len(segs[0]))
    if segs and trail:
        if not current.endswith(segs[-1]):
            return "trailing fixed segment changed"
        span[-1] = (len(current) - len(segs[-1]), len(current))
    limit = len(current)
    found: list[tuple[int, int]] = []
    for i in range(len(segs) - 1, -1, -1):
        got = span[i]
        if got is None:
            idx = current.rfind(segs[i], 0, limit)
            if idx < 0:
                return f"fixed segment #{i + 1} missing or out of order"
            got = (idx, idx + len(segs[i]))
        found.append(got)
        limit = got[0]
    found.reverse()
    prev = 0
    for i, (start, end) in enumerate(found):
        if start < prev:
            return f"fixed segment #{i + 1} overlaps an earlier one"
        prev = end
    if len(merged) == 1:
        return None
    editable = {ln for s, e in merged for ln in range(s, e + 1)}
    sm = difflib.SequenceMatcher(
        None,
        lines,
        current.splitlines(keepends=True),
        autojunk=False,
    )
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag in {"equal", "insert"}:
            continue
        bad = [ln for ln in range(i1 + 1, i2 + 1) if ln not in editable]
        if bad:
            return f"fixed line(s) {bad[:5]} changed"
    return None


def guard(
    hb: HarborBundle,
    workspace: Path,
    *,
    check_deletions: bool = True,
) -> tuple[HarborViolation, ...]:
    """Verify ``workspace`` against a Harbor bundle; empty result means clean.

    ``check_deletions=False`` is for partial workspaces that only hold the declared files.

    Args:
      hb: The parsed bundle.
      workspace: The materialized workspace.
      check_deletions: Report manifest files missing from ``workspace``.

    Returns:
      violations: One per breach, sorted by path within each check.

    """
    ws = Path(workspace)
    out: list[HarborViolation] = []
    guarded = hb.guarded_prefixes

    for rel in sorted(set(hb.manifest) | set(hb.editable)):
        if (
            _excluded(rel, hb.guard_exclude)
            or PurePosixPath(rel).parts[0] not in guarded
        ):
            continue
        if _symlinked(ws, rel):
            out.append(
                HarborViolation(
                    path=rel,
                    kind="symlink",
                    detail="guarded path is or sits under a symlink",
                ),
            )

    for rel, ranges in sorted(hb.editable.items()):
        p = ws / rel
        if _symlinked(ws, rel):
            continue
        pri = hb.pristine_root / rel
        if not p.is_file():
            out.append(
                HarborViolation(
                    path=rel,
                    kind="deleted",
                    detail="declared file missing",
                ),
            )
            continue
        if not pri.is_file():
            if not any(r == (-1, -1) for r in ranges):
                out.append(
                    HarborViolation(
                        path=rel,
                        kind="pristine_missing",
                        detail="no pristine snapshot",
                    ),
                )
            continue
        why = check_segments(pri.read_bytes(), p.read_bytes(), ranges)
        if why is not None:
            kind = "outside_range" if ranges else "read_only_modified"
            out.append(HarborViolation(path=rel, kind=kind, detail=why))

    for rel, sha in sorted(hb.manifest.items()):
        if rel in hb.editable or _excluded(rel, hb.guard_exclude):
            continue
        if PurePosixPath(rel).parts[0] in _EXEMPT_TOP or _symlinked(ws, rel):
            continue
        p = ws / rel
        if not p.is_file():
            if check_deletions:
                out.append(
                    HarborViolation(
                        path=rel,
                        kind="deleted",
                        detail="manifest file missing",
                    ),
                )
        elif sha256_file(p) != sha:
            out.append(
                HarborViolation(
                    path=rel,
                    kind="modified_unlisted",
                    detail="hash differs from manifest",
                ),
            )
    return tuple(out)


# --- Which eval-script set does the container run? -----------------------------------
#
# Two script sets exist upstream. ``tasks/<t>/scripts`` is what native MLS-Bench
# ran (task dir bind-mounted, no /tests mount) and so what produced every row of
# ``leaderboard.csv``. ``harbor/.../tests/eval/scripts`` is what Harbor's
# ``test.sh`` -> ``score_task.py run-evals --eval-root /tests/eval`` executes; it
# is the native set plus Harbor scaffolding. Our container has no /tests mount
# (no ``_inputgen/apply.py``, no holdout inputs) and no /workspace/_task links,
# so the Harbor-adapted scripts cannot run in it, while the native ones are the
# leaderboard's provenance. Decision: the container runs the NATIVE
# ``tasks/<t>/scripts``, and Harbor's copy is the tamper reference: an executed
# script that is not byte-identical to Harbor's is refused unless upstream
# itself ships that exact file as adapted (``harbor_adapted.json``, measured
# against the pinned commit; see ``real_clone_test``). Adapted tasks are still
# logged as ``harbor_adapted_scripts`` in workspace.json.
#
# 32 tasks differ on some file (29 on ``*.sh``; the 3 quant-* tasks differ only in
# ``_label_guard.py``). In 12 the difference is only the documented Harbor
# additions (``cd /workspace[/pkg]`` and the ``_inputgen/apply.py`` call, plus
# blank lines): for those the executed script must equal Harbor's after
# normalising exactly that. The other 20 have substantive Harbor patches
# (runtime races, ephemeral inputs, label guards): Harbor's copy cannot vouch for
# them, so the file *set* that differs is pinned AND the sha256 of every native
# ``tasks/<t>/scripts`` file is pinned in ``eval_script_pins.json`` (see
# ``eval_script_pins``); the executed native bytes must match the pin.
ADAPTED_PATH: Final = _CWD / "harbor_adapted.json"
"""``adapted``: task to the scripts Harbor ships patched; ``normalised_only``: the
tasks whose patches are only the documented additions (``_HARBOR_ADDITIONS``)."""


class AdaptedScripts(TypedDict):
    """Which tasks Harbor adapts, measured against the pinned commit."""

    adapted: dict[str, tuple[str, ...]]
    normalised_only: frozenset[str]


@functools.cache
def adapted_scripts() -> AdaptedScripts:
    """Return the committed table of Harbor-adapted eval scripts.

    Returns:
      table: Read once from ``harbor_adapted.json``.

    """
    return json_loads(ADAPTED_PATH.read_text(), AdaptedScripts)


_HARBOR_ADDITIONS: Final = (
    re.compile(r"^cd /workspace(/[\w.\-]+)?\s*$"),
    re.compile(r'^python "/tests/eval/_inputgen/apply\.py" "[\w\-]+" /workspace\s*$'),
)


PINS_PATH: Final = _CWD / "eval_script_pins.json"


@functools.cache
def eval_script_pins() -> Mapping[str, Mapping[str, str]]:
    """Return the native ``tasks/<t>/scripts`` digests of the substantively adapted tasks.

    Returns:
      pins: Task to ``{relative path: sha256}``, read once from ``eval_script_pins.json``.

    """
    return json_loads(PINS_PATH.read_text(), dict[str, dict[str, str]])


def pinned_tasks(table: AdaptedScripts | None = None) -> frozenset[str]:
    """Return the tasks whose native scripts are byte-pinned.

    Args:
      table: Adapted-script table; ``None`` is :func:`adapted_scripts`.

    Returns:
      tasks: Adapted and not normalised-only.

    """
    t = adapted_scripts() if table is None else table
    return frozenset(t["adapted"]) - t["normalised_only"]


def generate_pins(
    root: Path,
    table: AdaptedScripts | None = None,
) -> dict[str, dict[str, str]]:
    """Hash every native script file of the pinned tasks under a clone ``root``.

    Regenerate after a deliberate upstream bump (review the diff first)::

        uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.pin_eval_scripts [--root DIR] [--out FILE]  # noqa: E501 -- one line for copy-paste.

    Args:
      root: The upstream clone.
      table: Adapted-script table; ``None`` is :func:`adapted_scripts`.

    Returns:
      pins: Task to ``{relative path: sha256}`` of its native scripts.

    """
    out: dict[str, dict[str, str]] = {}
    for task in sorted(pinned_tasks(table)):
        d = Path(root) / "tasks" / task / "scripts"
        out[task] = {
            p.relative_to(d).as_posix(): sha256_file(p)
            for p in sorted(x for x in d.rglob("*") if x.is_file())
            if "__pycache__" not in p.parts
        }
    return out


def check_native_pins(
    task: str,
    scripts_dir: Path,
    pins: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[HarborViolation, ...]:
    """Verify a native scripts tree byte-for-byte against the pinned digests.

    Args:
      task: Task name.
      scripts_dir: The task's native ``scripts`` directory.
      pins: Digest table; ``None`` is :func:`eval_script_pins`.

    Returns:
      violations: Missing, mismatched and unpinned files; empty when intact.

    """
    table = eval_script_pins() if pins is None else pins
    want = table.get(task, {})
    if not want:
        return (HarborViolation(path="eval_script_pins", kind="eval_script_unpinned"),)
    base = Path(scripts_dir)
    out: list[HarborViolation] = []
    for rel, digest in sorted(want.items()):
        q = base / rel
        if q.is_symlink() or not q.is_file():
            out.append(HarborViolation(path=rel, kind="eval_script_missing"))
        elif sha256_file(q) != digest:
            out.append(HarborViolation(path=rel, kind="eval_script_pin_mismatch"))
    out.extend(
        HarborViolation(path=rel, kind="eval_script_unpinned")
        for p in sorted(base.rglob("*"))
        if (p.is_file() or p.is_symlink())
        and "__pycache__" not in p.parts
        and (rel := p.relative_to(base).as_posix()) not in want
    )
    return tuple(out)


class EvalScriptsError(BundleError):
    """The eval scripts about to run are not the ones the Harbor bundle vouches for."""

    def __init__(self, task: str, violations: tuple[HarborViolation, ...]) -> None:
        self.violations = violations
        super().__init__(
            f"{task}: eval scripts refused: "
            + "; ".join(f"{v.kind} {v.path}" for v in violations),
        )


class MissingBundleError(BundleError, FileNotFoundError):
    """A real-clone task was materialized without its Harbor bundle."""


def normalise_harbor_script(text: str) -> list[str]:
    """Strip only the documented Harbor additions and blank lines; return the rest."""
    return [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not any(p.match(line) for p in _HARBOR_ADDITIONS)
    ]


def eval_scripts_intact(
    hb: HarborBundle,
    scripts_dir: Path,
) -> tuple[HarborViolation, ...]:
    """Compare a directory of eval scripts against the bundle's ``tests/eval/scripts``.

    Every regular file of the reference is compared byte for byte (not only ``*.sh``:
    helper modules such as ``_label_guard.py`` are executed too).

    Args:
      hb: The parsed bundle.
      scripts_dir: The scripts about to run.

    Returns:
      violations: One per missing or modified file; empty when identical.

    """
    ref = hb.root / "tests" / "eval" / "scripts"
    out: list[HarborViolation] = []
    for p in sorted(x for x in ref.rglob("*") if x.is_file()):
        rel = p.relative_to(ref).as_posix()
        q = Path(scripts_dir) / rel
        if q.is_symlink() or not q.is_file():
            out.append(HarborViolation(path=rel, kind="eval_script_missing"))
        elif not filecmp.cmp(q, p, shallow=False):
            out.append(HarborViolation(path=rel, kind="eval_script_modified"))
    return tuple(out)


def check_eval_scripts(
    task: str,
    hb: HarborBundle,
    scripts_dir: Path,
    pins: Mapping[str, Mapping[str, str]] | None = None,
    table: AdaptedScripts | None = None,
) -> tuple[str, ...]:
    """Refuse an edited or swapped eval script; return the names of adapted ones.

    Policy (see the block comment above): a task Harbor does not adapt must match
    byte for byte; a normalised-only task must match after :func:`normalise_harbor_script`
    and differ on exactly the pinned files; a substantively adapted task must differ
    on exactly the pinned files and match the pinned sha256 of every native file
    (:func:`eval_script_pins`).

    Args:
      task: Task name.
      hb: The parsed bundle.
      scripts_dir: The scripts about to run.
      pins: Digest table; ``None`` is :func:`eval_script_pins`.
      table: Adapted-script table; ``None`` is :func:`adapted_scripts`.

    Returns:
      adapted: Sorted names of scripts that differ from Harbor's, all expected.

    Raises:
      EvalScriptsError: Any deviation from the policy.

    """
    t = adapted_scripts() if table is None else table
    found = eval_scripts_intact(hb, scripts_dir)
    names = tuple(sorted(v.path for v in found))
    expected = t["adapted"].get(task, ())
    bad: list[HarborViolation] = [v for v in found if v.path not in expected]
    if task in t["adapted"]:
        # A pinned file that now matches Harbor's also signals drift, or a swap.
        bad += [
            HarborViolation(path=n, kind="eval_script_expected_adapted")
            for n in expected
            if n not in names
        ]
    if task in t["normalised_only"]:
        ref = hb.root / "tests" / "eval" / "scripts"
        for v in found:
            q = Path(scripts_dir) / v.path
            if v.kind == "eval_script_modified" and normalise_harbor_script(
                (ref / v.path).read_text(errors="replace"),
            ) != normalise_harbor_script(q.read_text(errors="replace")):
                bad.append(HarborViolation(path=v.path, kind="eval_script_modified"))
    if task in pinned_tasks(t):
        bad += check_native_pins(task, scripts_dir, pins)
    if bad:
        raise EvalScriptsError(task, tuple(bad))
    return names


def _safe_rel(rel: object) -> str:
    if not isinstance(rel, str) or not rel or "\\" in rel:
        raise BundleError(f"unsafe path {rel!r}")
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts:
        raise BundleError(f"unsafe path {rel!r}")
    return p.as_posix()


def _excluded(rel: str, excl: tuple[str, ...]) -> bool:
    return any(rel == e or rel.startswith(e + "/") for e in excl)


def _symlinked(root: Path, rel: str) -> bool:
    """Report whether ``rel`` or any parent under ``root`` is a symlink."""
    p = root
    for part in PurePosixPath(rel).parts:
        p = p / part
        if p.is_symlink():
            return True
    return False


class _BundleRange(TypedDict):
    start: int
    end: int


class _BundleFile(TypedDict):
    filename: str
    edit: NotRequired[list[_BundleRange] | None]


class _BundleConfig(TypedDict):
    """The fields of ``tests/meta/config.json`` the guard reads; others pass."""

    files: NotRequired[list[_BundleFile]]
    guard_exclude: NotRequired[list[str] | None]
