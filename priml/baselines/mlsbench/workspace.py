"""Materialize a workspace (pristine files + one edit) and guard the edit surface.

``materialize`` copies the task's editable files into ``<run_dir>/workspace``
(and a reference copy into ``<run_dir>/pristine``), runs an edit script
against the workspace in a sanitised subprocess, then runs :func:`check_diff`.

The diff guard works in *pristine* line coordinates: a hunk of the edited file
is legal iff the pristine lines it replaces (or, for a pure insertion, the gap
it is inserted into) lie within one allowed range. Working in pristine
coordinates is equivalent to re-anchoring later ranges after earlier length
changes: a range grows or shrinks with its own edits and never shifts into a
neighbour. An insertion is legal at the range boundaries (directly before its
first line or directly after its last).
"""

# ruff: noqa: S603, N818

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

import difflib
import hashlib
import json
import shutil
import subprocess
import sys

from priml.baselines.mlsbench.task import EditableFile, TaskSpec
from priml.baselines.mlsbench.workspace_harbor import (
    BundleError,
    HarborBundle,
    MissingBundleError,
    find_bundle,
    guard,
    load_bundle,
    pristine_source,
    sanitised_env,
    sha256_file,
)


_SAFE_PATH: Final = "/usr/bin:/bin"


class EditError(RuntimeError):
    """The edit script exited non-zero or timed out."""


ViolationKind = Literal[
    "outside_range",
    "unlisted_file_changed",
    "file_deleted",
    "file_added",
    # Harbor-bundle guard kinds (workspace_harbor.HarborViolation.kind)
    "symlink",
    "deleted",
    "pristine_missing",
    "read_only_modified",
    "modified_unlisted",
]


@dataclass(frozen=True)
class Violation:
    """One breach of the edit surface."""

    path: str
    kind: ViolationKind
    line: int | None = None  # 1-indexed pristine line; None for whole-file kinds


class DiffGuardViolation(RuntimeError):
    """Raised by ``materialize`` when the edit leaves the allowed line ranges."""

    def __init__(self, violations: tuple[Violation, ...]) -> None:
        self.violations = violations
        super().__init__(
            "diff guard: "
            + "; ".join(
                f"{v.kind} {v.path}" + (f":{v.line}" if v.line is not None else "")
                for v in violations
            ),
        )


@dataclass(frozen=True)
class Workspace:
    """A materialized (optionally edited) workspace and its content digest."""

    root: Path
    task: TaskSpec
    edit: str | None
    digest: str
    warnings: tuple[str, ...] = ()
    """Non-fatal findings (a manifest hash upstream itself ignores)."""
    files: tuple[tuple[str, str], ...] = ()
    """``(relative path, hash)`` snapshot behind ``digest``, for :func:`workspace_changes`."""


class WorkspaceChangedError(RuntimeError):
    """The materialized workspace no longer matches its recorded digest."""

    def __init__(self, changes: tuple[str, ...]) -> None:
        self.changes = changes
        super().__init__("workspace changed since materialize: " + "; ".join(changes))


def _sha(path: Path) -> str:
    if path.is_symlink():
        return "symlink:" + str(path.readlink())
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root: Path) -> dict[str, str]:
    """Relative path -> sha256 of every file under ``root`` (symlinks are not followed)."""
    return {
        p.relative_to(root).as_posix(): _sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_symlink() or p.is_file()
    }


def _digest(m: dict[str, str]) -> str:
    h = hashlib.sha256()
    for k in sorted(m):
        h.update(f"{k}\0{m[k]}\n".encode())
    return h.hexdigest()


def workspace_changes(ws: Workspace) -> tuple[str, ...]:
    """Return ``kind:path`` for every recorded file that changed since materialize.

    Files the run *adds* (outputs written by a setting into the mounted workspace) are
    not changes; a recorded file that was modified, deleted or replaced by a symlink is.
    A workspace built without a snapshot falls back to the whole-tree digest.
    """
    if not ws.files:
        now = _digest(manifest(ws.root))
        return () if now == ws.digest else ("digest:<tree>",)
    out: list[str] = []
    for rel, sha in ws.files:
        p = ws.root / rel
        if not (p.is_symlink() or p.is_file()):
            out.append(f"deleted:{rel}")
        elif _sha(p) != sha:
            out.append(f"modified:{rel}")
    return tuple(out)


def _lines(path: Path) -> list[str]:
    return (
        path.read_bytes()
        .decode("utf-8", errors="surrogateescape")
        .splitlines(keepends=True)
    )


def _hunk_legal(i1: int, i2: int, ef: EditableFile) -> bool:
    """Report whether pristine span ``[i1, i2)`` (0-indexed, half-open) fits one range.

    ``end == -1`` extends to EOF and ``(-1, -1)`` is the whole file (see ``task.py``).
    """
    for s, end in ef.ranges:  # 1-indexed inclusive -> 0-indexed half-open [s-1, e)
        if (s, end) == (-1, -1):
            return True
        e = sys.maxsize if end == -1 else end
        if i1 == i2:
            if s - 1 <= i1 <= e:
                return True
        elif s - 1 <= i1 and i2 <= e:
            return True
    return False


def check_diff(task: TaskSpec, pristine: Path, edited: Path) -> tuple[Violation, ...]:
    """Return violations of the edit surface between two trees (empty when clean)."""
    editable = {f.path: f for f in task.files}
    pm, em = manifest(pristine), manifest(edited)
    out: list[Violation] = []
    for path in sorted(set(pm) | set(em)):
        if path not in em:
            out.append(Violation(path, "file_deleted", None))
        elif path not in pm:
            out.append(Violation(path, "file_added", None))
        elif pm[path] != em[path]:
            ef = editable.get(path)
            if (
                ef is None
                or pm[path].startswith("symlink:")
                or em[path].startswith("symlink:")
            ):
                out.append(Violation(path, "unlisted_file_changed", None))
                continue
            a, b = _lines(pristine / path), _lines(edited / path)
            sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
            for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
                if tag != "equal" and not _hunk_legal(i1, i2, ef):
                    out.append(Violation(path, "outside_range", i1 + 1))
    return tuple(out)


def check_bundle(hb: HarborBundle, workspace: Path) -> tuple[Violation, ...]:
    """Run the Harbor byte-segment / manifest guard on a partial (declared-files) workspace."""
    return tuple(
        Violation(v.path, cast(ViolationKind, v.kind), None)
        for v in guard(hb, workspace, check_deletions=False)
    )


def _missing_bundle(task: TaskSpec, rel: str, src: Path) -> FileNotFoundError:
    """Explain a missing editable file when no Harbor bundle was supplied.

    Real upstream tasks do not ship their editable files under ``tasks/<t>/``; the
    pristine bytes live in the Harbor bundle. A synthetic task simply lacks the file.
    """
    root = task.root.parent.parent
    has_harbor = (root / "harbor").is_dir()
    found = find_bundle(task.name, root=root)
    if found is None and not has_harbor:
        return FileNotFoundError(f"{src}: editable file {rel} not found")
    expected = root / "harbor" / "tasks-docker" / f"mls-bench__{task.name}"
    what = (
        f"the bundle exists at {found} but was not passed; call with bundle={found}"
        if found is not None
        else f"the clone has no Harbor bundle at {expected}; point mlsbench_root at "
        "a full upstream clone that includes harbor/tasks-docker"
    )
    return MissingBundleError(
        f"{task.name}: editable file {rel} is not in {task.root} and no Harbor bundle "
        f"was given (bundle=None): {what}.",
    )


def _check_bundle_matches_task(task: TaskSpec, hb: HarborBundle) -> None:
    """Refuse a bundle that declares other files than the task, or other ranges.

    The task declares ranges in upstream's source coordinates while the bundle's
    pristine snapshot already has the package ``pre_edit`` and the task's
    ``mid_edit`` applied (its ranges are the live-shifted ones), so the file set
    must agree and each range may only have moved (same length), or the task has
    a ``mid_edit``, which may also resize a range.
    """
    mine = {f.path: tuple((int(r[0]), int(r[1])) for r in f.ranges) for f in task.files}
    if mine == hb.editable:
        return
    if set(mine) == set(hb.editable) and (
        task.mid_edit is not None
        or all(_shifted(mine[k], hb.editable[k]) for k in mine)
    ):
        return
    raise BundleError(
        f"{task.name}: task files/ranges {mine} disagree with bundle {hb.editable}",
    )


def _shifted(a: tuple[tuple[int, int], ...], b: tuple[tuple[int, int], ...]) -> bool:
    """Report whether ``b`` is ``a`` moved down by package pre_edit ops (same lengths)."""
    return len(a) == len(b) and all(
        e1 - s1 == e2 - s2 and (s1, e1) != (-1, -1)
        for (s1, e1), (s2, e2) in zip(a, b, strict=True)
    )


_RUNNER = """\
import json, runpy, sys
edit, ws, out = sys.argv[1:4]
sys.argv = [edit, ws]
ns = runpy.run_path(edit, run_name="__main__")
ops = ns.get("OPS")
if ops is not None:
    with open(out, "w") as f:
        json.dump(ops, f)
"""
"""Loads an edit script in a sanitised interpreter: legacy scripts edit ``argv[1]``
themselves; upstream ``*.edit.py`` files only define ``OPS`` and hand it back as JSON."""

_OP_KEYS = frozenset({"replace", "insert", "delete", "create"})


def _target(ws: Path, rel: object) -> Path:
    if not isinstance(rel, str) or not rel:
        raise EditError(f"op has no file: {rel!r}")
    parts = Path(rel).parts
    if Path(rel).is_absolute() or ".." in parts:
        raise EditError(f"unsafe op path: {rel}")
    path = ws / rel
    for i in range(len(parts)):
        if (ws.joinpath(*parts[: i + 1])).is_symlink():
            raise EditError(f"op path crosses a symlink: {rel}")
    return path


def apply_ops(ws: Path, ops: object) -> int:
    """Apply upstream edit ops (``replace``/``insert``/``delete``/``create``) in order.

    Semantics follow the upstream applier: 1-indexed lines, ``replace`` and
    ``delete`` are inclusive, ``insert`` goes after ``after_line``, content gets a
    trailing newline, ops run top to bottom against the file as the previous op
    left it (upstream orders them bottom-to-top).

    Returns:
      count: Number of ops applied.

    Raises:
      EditError: Malformed op, unknown op, a missing target or an unsafe path.

    """
    if not isinstance(ops, list) or not ops:
        raise EditError("edit script defines an empty or non-list OPS")
    items = cast("list[object]", ops)
    for n, raw in enumerate(items):
        op = cast("dict[str, object]", raw) if isinstance(raw, dict) else None
        if op is None or op.get("op") not in _OP_KEYS:
            raise EditError(f"OPS[{n}]: unknown or malformed op {raw!r:.200}")
        path = _target(ws, op.get("file"))
        kind = op["op"]
        try:
            if kind == "create":
                content = str(op["content"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content if content.endswith("\n") else content + "\n")
                continue
            if not path.is_file():
                raise EditError(f"OPS[{n}]: target {op['file']} missing in workspace")
            lines = path.read_text().splitlines(keepends=True)
            if kind == "replace":
                content = str(op["content"])
                new = content if content.endswith("\n") else content + "\n"
                lines[int(str(op["start_line"])) - 1 : int(str(op["end_line"]))] = (
                    new.splitlines(keepends=True)
                )
            elif kind == "insert":
                content = str(op["content"])
                new = content if content.endswith("\n") else content + "\n"
                i = int(str(op["after_line"]))
                lines[i:i] = new.splitlines(keepends=True)
            else:
                start = int(str(op.get("start_line", op.get("line", 0))))
                end = int(str(op.get("end_line", start)))
                del lines[start - 1 : end]
            path.write_text("".join(lines))
        except (KeyError, ValueError, TypeError) as e:
            raise EditError(f"OPS[{n}]: {type(e).__name__}: {e}") from e
    return len(items)


def materialize(
    task: TaskSpec,
    run_dir: Path,
    *,
    edit: Path | None,
    edit_timeout: float = 60.0,
    bundle: Path | None = None,
    strict: bool = True,
) -> Workspace:
    """Copy pristine files, apply ``edit`` (if any), enforce the diff guard.

    With ``bundle`` (a Harbor bundle directory, read at runtime from the upstream checkout)
    the pristine bytes come from ``tests/meta/pristine`` (falling back to the task tree),
    are verified against ``pristine_manifest.json``, and the guard is Harbor's byte-segment
    diff plus manifest hashing. Without it the synthetic line-diff guard is used.

    ``strict=False`` is for *published baseline* edits only (trusted upstream scripts
    that upstream itself never runs through its verifier): guard violations are
    recorded in ``Workspace.warnings`` instead of raising. Agent edits stay strict.
    """
    run_dir = Path(run_dir)
    ws, ref = run_dir / "workspace", run_dir / "pristine"
    if ws.exists() or ref.exists():
        raise FileExistsError(f"{run_dir}: workspace already materialized")
    hb = load_bundle(bundle) if bundle is not None else None
    warnings: list[str] = []
    if hb is not None:
        _check_bundle_matches_task(task, hb)
    for f in task.files:
        if hb is not None:
            src = pristine_source(hb, f.path)
            if src is None:
                raise FileNotFoundError(
                    f"{f.path}: not in the bundle's pristine/ or environment/_scaffold/",
                )
        else:
            src = task.root / f.path
            if not src.is_file() and not src.is_symlink():
                raise _missing_bundle(task, f.path, src)
        if src.is_symlink() or not src.is_file():
            raise FileNotFoundError(f"{src}: not a regular file")
        if (
            hb is not None
            and f.path in hb.manifest
            and sha256_file(src) != hb.manifest[f.path]
        ):
            msg = f"{f.path}: source differs from pristine_manifest.json"
            if hb.editable.get(f.path, ()):
                raise BundleError(msg)
            # Upstream's verifier never consults the manifest for declared files
            # (it diffs against pristine/); a stale hash on a read-only file is
            # an upstream data glitch, so it is reported, not fatal.
            warnings.append(msg)
        for base in (ws, ref):
            dst = base / f.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst, follow_symlinks=False)
    name: str | None = None
    if edit is not None:
        edit = Path(edit).resolve()
        if not edit.is_file():
            raise FileNotFoundError(f"{edit}: edit script missing")
        name = next(
            (
                b.name
                for b in task.baselines
                if b.edit is not None and b.edit.resolve() == edit
            ),
            "agent",
        )
        try:
            ops_file = run_dir / "edit_ops.json"
            ops_file.unlink(missing_ok=True)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    _RUNNER,
                    str(edit),
                    str(ws),
                    str(ops_file),
                ],
                cwd=ws,
                env=sanitised_env() if hb is not None else {"PATH": _SAFE_PATH},
                capture_output=True,
                text=True,
                timeout=edit_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise EditError(f"{edit.name}: timed out after {edit_timeout}s") from e
        if proc.returncode != 0:
            raise EditError(
                f"{edit.name}: exit {proc.returncode}: {proc.stderr.strip()[-2000:]}",
            )
        if ops_file.is_file():
            ops = cast("object", json.loads(ops_file.read_text()))
            apply_ops(ws, ops)
    violations = check_bundle(hb, ws) if hb is not None else check_diff(task, ref, ws)
    if violations and strict:
        raise DiffGuardViolation(violations)
    warnings.extend(
        f"guard(baseline) {v.kind} {v.path}"
        + (f":{v.line}" if v.line is not None else "")
        for v in violations
    )
    snapshot = manifest(ws)
    return Workspace(
        root=ws,
        task=task,
        edit=name,
        digest=_digest(snapshot),
        warnings=tuple(warnings),
        files=tuple(sorted(snapshot.items())),
    )
