"""Stage the data an MLS-Bench task ships in ``tasks/<task>/data/``.

Upstream mounts a task's directory into the container as ``/workspace/_task``,
so the files a script reads from ``_task/data`` are the task's own bytes. This
script copies those trees to a host directory an experiment can point at,
digest-checks every copy against its source, and writes ``manifest.json``.
Data the images bake in, and data fetched at run time (Hugging Face caches,
downloads), are not staged: tasks whose ``config.json`` sets
``requires_network`` or ``ephemeral_inputs`` are reported as such and left
alone, since nothing here can fetch them offline or vouch for their bytes.

Stdlib only, never called from a config. Idempotent: a file already staged
with the source digest is left untouched.

The clone is read from ``/opt/scratch/mlsbench`` and staged into
``/opt/scratch/datasets/mlsbench_staged`` unless ``--root`` / ``--dest`` say
otherwise; those are where an experiment looks by default (:data:`UPSTREAM_DIR`
and :data:`STAGED_DIR` beneath ``base_dir``).

Examples::

    uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.prepare_data --list
    uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.prepare_data --task llm-dllm-demask-strategy
    uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.prepare_data --all
    uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.prepare_data --all --verify
    uv --quiet run --frozen python -m priml.baselines.mlsbench.scripts.prepare_data --smoke
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import argparse
import hashlib
import json
import shutil
import sys


if TYPE_CHECKING:
    from collections.abc import Sequence


MANIFEST: Final = "manifest.json"
_CHUNK: Final = 1 << 20

BASE_DIR: Final = "/opt/scratch"
"""Default resource root (``TrainLoop.Config.base_dir``'s default)."""

UPSTREAM_DIR: Final = "/mlsbench"
"""Logical location of the read-only upstream clone, beneath ``base_dir``."""

DATASETS_DIR: Final = "/datasets/mlsbench"
"""Logical ``{data_root}`` of upstream's package configs (datasets, weights)."""

STAGED_DIR: Final = "/datasets/mlsbench_staged"
"""Logical location this script stages each task's ``data/`` to."""

SMOKE_DIR: Final = "/mlsbench_smoke"
"""Logical location of the synthetic clone ``exp_smoke`` runs against."""


def default_path(logical: str) -> Path:
    """Return a logical directory beneath the default :data:`BASE_DIR`."""
    return Path(BASE_DIR) / logical.lstrip("/")


class StageError(RuntimeError):
    """The source is missing or a staged file's digest disagrees with it."""


@dataclass(frozen=True)
class Staged:
    """One task's staged tree."""

    task: str
    files: dict[str, str]
    """Path relative to ``data/`` to sha256 hex digest."""

    copied: int
    """How many files were written this run (0 when already staged)."""


def sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file, streamed."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def digest_tree(root: Path) -> dict[str, str]:
    """Return ``relative posix path -> sha256`` for every file under ``root``."""
    return {
        p.relative_to(root).as_posix(): sha256(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and not p.is_symlink() and "__pycache__" not in p.parts
    }


def resolve_root(root: Path) -> Path:
    """Return the clone root, checked to hold a ``tasks/`` directory."""
    path = Path(root).expanduser().resolve()
    if not (path / "tasks").is_dir():
        raise StageError(
            f"{path} has no tasks/ directory; pass --root a read-only MLS-Bench clone",
        )
    return path


def _config(task_dir: Path) -> dict[str, object]:
    try:
        raw = cast("object", json.loads((task_dir / "config.json").read_text()))
    except (OSError, ValueError) as e:
        raise StageError(f"{task_dir.name}: unreadable config.json ({e})") from e
    if not isinstance(raw, dict):
        raise StageError(f"{task_dir.name}: config.json is not an object")
    return {str(k): v for k, v in cast("dict[object, object]", raw).items()}


def survey(root: Path) -> dict[str, dict[str, object]]:
    """Describe every task: whether it ships data, and whether it needs the network.

    Returns:
      info: Task name to ``{"ships_data": bool, "files": int, "external": bool}``.

    """
    out: dict[str, dict[str, object]] = {}
    for d in sorted((root / "tasks").iterdir()):
        if not (d / "config.json").is_file():
            continue
        cfg = _config(d)
        data = d / "data"
        n = sum(1 for p in data.rglob("*") if p.is_file()) if data.is_dir() else 0
        out[d.name] = {
            "ships_data": n > 0,
            "files": n,
            "external": bool(
                cfg.get("requires_network") or cfg.get("ephemeral_inputs"),
            ),
        }
    return out


def stage_task(
    root: Path,
    task: str,
    dest: Path,
    *,
    verify_only: bool = False,
) -> Staged:
    """Copy one task's ``data/`` to ``<dest>/<task>/data`` and check every digest.

    Args:
      root: The upstream clone (read only).
      task: Task directory name.
      dest: Destination root.
      verify_only: Compare digests, write nothing.

    Returns:
      staged: The digests and how many files were written.

    Raises:
      StageError: No ``data/`` in the task, a file is missing or differs (in
        ``verify_only``), or a copy does not hash like its source.

    """
    src = root / "tasks" / task / "data"
    if not src.is_dir():
        raise StageError(f"{task}: ships no data/ directory")
    want = digest_tree(src)
    out = dest / task / "data"
    copied = 0
    for rel, digest in want.items():
        target = out / rel
        if target.is_file() and sha256(target) == digest:
            continue
        if verify_only:
            state = "differs" if target.is_file() else "is missing"
            raise StageError(f"{task}/{rel}: staged file {state}")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        _ = shutil.copyfile(src / rel, tmp)
        if sha256(tmp) != digest:
            tmp.unlink()
            raise StageError(f"{task}/{rel}: copy does not match the source digest")
        _ = tmp.replace(target)
        copied += 1
    if not verify_only:
        _write_manifest(dest, task, want)
    return Staged(task, want, copied)


def _write_manifest(dest: Path, task: str, files: dict[str, str]) -> None:
    path = dest / MANIFEST
    doc: dict[str, dict[str, str]] = (
        cast("dict[str, dict[str, str]]", json.loads(path.read_text()))
        if path.is_file()
        else {}
    )
    doc[task] = files
    dest.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def verify_manifest(root: Path, dest: Path) -> list[str]:
    """Return problems found comparing ``manifest.json`` to disk and to the clone."""
    path = dest / MANIFEST
    if not path.is_file():
        return [f"no {MANIFEST} in {dest}"]
    problems: list[str] = []
    doc = cast("dict[str, dict[str, str]]", json.loads(path.read_text()))
    for task, files in doc.items():
        src = (
            digest_tree(root / "tasks" / task / "data")
            if (root / "tasks" / task / "data").is_dir()
            else {}
        )
        if src != files:
            problems.append(f"{task}: manifest disagrees with the clone")
        for rel, digest in files.items():
            f = dest / task / "data" / rel
            if not f.is_file():
                problems.append(f"{task}/{rel}: missing")
            elif sha256(f) != digest:
                problems.append(f"{task}/{rel}: digest mismatch")
    return problems


TASK_DATA_MOUNT: Final = "/workspace/_task/data"
"""Where scripts read a task's own ``data/`` (the task dir is mounted at ``_task``)."""


@dataclass(frozen=True)
class DatasetBinds:
    """``docker -v`` specs for a package's dataset and model directories."""

    binds: tuple[str, ...]
    skipped: tuple[str, ...]
    """Specs whose host directory is absent. Skipped, never created: the image may
    bake that data in and an empty bind would mask it (upstream's own rule)."""


def _pkg_config(root: Path, package: str) -> dict[str, object]:
    base = root / "vendor" / "pkg_configs"
    if not base.is_dir():
        return {}
    want = package.lower()
    for d in sorted(base.iterdir()):
        if d.name.lower() == want and (d / "config.json").is_file():
            try:
                raw = cast("object", json.loads((d / "config.json").read_text()))
            except (OSError, ValueError) as e:
                raise StageError(f"{d.name}: unreadable pkg config ({e})") from e
            if not isinstance(raw, dict):
                return {}
            return {str(k): v for k, v in cast("dict[object, object]", raw).items()}
    return {}


def dataset_binds(root: Path, package: str, data_root: Path | str) -> DatasetBinds:
    """Resolve a package's ``data_bind`` and ``data_deps`` into ``-v`` specs.

    Follows upstream's native runner: ``{data_root}`` and ``{project_root}`` are
    expanded, an explicit ``data_bind`` comes first, and each ``data_deps`` entry adds
    ``host:container`` unless already present. The pkg config is read from the clone
    at call time (never copied). With no ``data_root`` nothing can be resolved and the
    result is empty.

    Args:
      root: The upstream clone (read only).
      package: Package name as the task config spells it (case-insensitive).
      data_root: Host directory that holds the datasets and model weights.

    Returns:
      binds: Specs to mount, and the ones skipped for a missing host directory.

    """
    if not str(data_root):
        return DatasetBinds((), ())
    cfg = _pkg_config(root, package)

    def expand(text: str) -> str:
        return text.replace("{project_root}", str(root)).replace(
            "{data_root}",
            str(Path(data_root).expanduser()),
        )

    specs: list[str] = []
    raw = cfg.get("data_bind")
    entries = (
        cast("list[object]", raw) if isinstance(raw, list) else [raw] if raw else []
    )
    for entry in entries:
        specs += [expand(entry)] if isinstance(entry, str) else []
    deps = cfg.get("data_deps")
    for dep in cast("list[object]", deps) if isinstance(deps, list) else []:
        if not isinstance(dep, dict):
            continue
        fields = cast("dict[str, object]", dep)
        host, cont = fields.get("host_path"), fields.get("container_path")
        if isinstance(host, str) and isinstance(cont, str) and host and cont:
            spec = f"{expand(host)}:{cont}"
            if spec not in specs:
                specs.append(spec)
    keep = tuple(s for s in specs if Path(s.split(":", 1)[0]).exists())
    return DatasetBinds(keep, tuple(s for s in specs if s not in keep))


def staged_bind(dest: Path, task: str) -> str:
    """Return the ``-v`` spec that overlays a task's staged ``data/`` (read-only).

    Every staged file is re-digested against ``manifest.json`` first, so a bind is
    only ever offered for bytes the manifest vouches for.

    Raises:
      StageError: The task is not in the manifest, or a staged file is missing or
        differs from its recorded digest.

    """
    path = dest / MANIFEST
    if not path.is_file():
        raise StageError(f"no {MANIFEST} in {dest}")
    doc = cast("dict[str, dict[str, str]]", json.loads(path.read_text()))
    if task not in doc:
        raise StageError(f"{task}: not staged in {dest}")
    data = dest / task / "data"
    for rel, digest in doc[task].items():
        f = data / rel
        if not f.is_file():
            raise StageError(f"{task}/{rel}: staged file is missing")
        if sha256(f) != digest:
            raise StageError(f"{task}/{rel}: staged file differs from the manifest")
    return f"{data.resolve()}:{TASK_DATA_MOUNT}:ro"


def staged_tasks(dest: Path) -> frozenset[str]:
    """Return the tasks ``manifest.json`` lists (empty when there is none)."""
    path = dest / MANIFEST
    if not path.is_file():
        return frozenset[str]()
    listed = cast("dict[str, object]", json.loads(path.read_text()))
    names: set[str] = {str(name) for name in listed}
    return frozenset(names)


class _Flags:
    root: str
    dest: str
    task: list[str]
    all: bool
    list: bool
    verify: bool
    smoke: str

    def __init__(self) -> None:
        self.root = ""
        self.dest = ""
        self.task = []
        self.all = False
        self.list = False
        self.verify = False
        self.smoke = ""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI; returns the process exit code."""
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    _ = p.add_argument("--root", default=str(default_path(UPSTREAM_DIR)))
    _ = p.add_argument("--dest", default=str(default_path(STAGED_DIR)))
    _ = p.add_argument("--task", action="append", default=[])
    _ = p.add_argument("--all", action="store_true", help="every task that ships data")
    _ = p.add_argument("--list", action="store_true")
    _ = p.add_argument(
        "--verify",
        action="store_true",
        help="check digests, write nothing",
    )
    _ = p.add_argument(
        "--smoke",
        nargs="?",
        const=str(default_path(SMOKE_DIR)),
        default="",
        help="build the synthetic clone exp_smoke reads, then stop",
    )
    ns = p.parse_args(argv, namespace=_Flags())
    try:
        return _run(p, ns)
    except StageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _run(p: argparse.ArgumentParser, ns: _Flags) -> int:
    if ns.smoke:
        from priml.baselines.mlsbench.testing import build_synth_task  # noqa: PLC0415

        dest = Path(ns.smoke).expanduser()
        shutil.rmtree(dest / "tasks", ignore_errors=True)
        print(f"synthetic clone: {build_synth_task(dest)}")
        return 0
    root = resolve_root(Path(ns.root))
    info = survey(root)
    if ns.list:
        for name, i in info.items():
            if i["ships_data"] or i["external"]:
                note = (
                    "external/run-time inputs, not stageable" if i["external"] else ""
                )
                print(f"{name}\t{i['files']} files\t{note}".rstrip())
        return 0
    dest = Path(ns.dest).expanduser()
    names = ns.task or (
        [n for n, i in info.items() if i["ships_data"]] if ns.all else []
    )
    if not names and not ns.verify:
        p.error("name a --task, or pass --all or --list")
    if ns.verify and not names:
        problems = verify_manifest(root, dest)
        for line in problems:
            print(line, file=sys.stderr)
        return 1 if problems else 0
    for name in names:
        if name not in info:
            raise StageError(f"unknown task {name!r}")
        if info[name]["external"]:
            print(f"{name}: skipped (external or run-time inputs)")
            continue
        s = stage_task(root, name, dest, verify_only=ns.verify)
        print(f"{name}: {len(s.files)} files ok, {s.copied} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
