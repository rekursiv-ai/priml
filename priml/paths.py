"""Filesystem paths a run composes and writes.

Two functions, split by WHEN they run:

- :func:`resolve_working_dir` composes two config fields at config time. Pure;
  no filesystem access.
- :func:`validated_output_path` validates a destination at write time. Touches the disk.

Deliberately torch-free, so config-time path resolution stays available to
processes that never import torch (``priml.runtime`` costs seconds to import).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import os


__all__ = [
    "resolve_working_dir",
    "validated_output_path",
]


def resolve_working_dir(
    base_dir: Path | str | None,
    working_dir: Path | str,
) -> Path:
    """Resolve a Config's ``working_dir`` against an optional ``base_dir``.

    The one path-composition rule every path-owning Config shares: when
    ``base_dir`` is ``None`` the ``working_dir`` is its own absolute logical
    root; otherwise ``working_dir`` is made relative (its leading slash
    stripped) and joined beneath ``base_dir``. Stripping the slash is required
    because ``Path("/a") / "/b" == Path("/b")`` -- an absolute right operand
    discards the base (POSIX semantics), so a logical root like
    ``"/checkpoints"`` would otherwise ignore its owner's ``base_dir``.

    Args:
      base_dir: Owner-supplied root, or ``None`` when the Config is its own root.
      working_dir: The Config's opinionated logical location.

    Returns:
      resolved: ``working_dir`` as a ``Path``, beneath ``base_dir`` when given.

    """
    if base_dir is None:
        return Path(working_dir)
    return Path(base_dir) / str(working_dir).lstrip("/")


def validated_output_path(
    path: Path | str,
    *,
    protected: Iterable[Path] | Callable[[], Iterable[Path]] = (),
) -> Path:
    """Return the canonical destination to write, or raise if it is unusable.

    Write to the path this returns rather than to the argument, so the checks
    below and the write itself name the same file. A caller that keeps its own
    spelling reintroduces the very gap this function closes: ``~/x.json``
    passes every check here and then writes a literal ``~`` directory.

    When ``protected`` is given, the destination must not name one of those
    input artifacts. A job that writes a report attesting the digests of files
    beside it can be handed one of them as ``--output``; the write then
    destroys the evidence the report describes, and the report still claims a
    digest for bytes that no longer exist.

    Resolved-path equality alone is insufficient: a hard link resolves to a
    distinct name yet shares the inode, so writing would still destroy the
    artifact.

    ``protected`` may be a callable, evaluated only after the destination is
    validated. Deriving the protected set usually means READING those inputs,
    and an unusable destination should fail before a run touches them.

    This does NOT inspect version control. An earlier revision walked ancestors
    looking for ``.git`` and refused paths inside a checkout; that is a
    repository-hygiene rule, enforced by a ``.gitignore`` (an ignored file
    cannot be staged, so the harm is prevented before a commit rather than
    detected after a job has already written the bytes). A published library
    has no business inspecting a user's version control, and the check refused
    legitimate work for anyone whose workspace sits inside a repository.

    Args:
      path: Intended output destination, absolute or relative, ``~`` allowed.
      protected: Input artifacts the write must never touch, or a callable
        returning them. Entries need not exist; a missing one cannot be
        aliased. Defaults to none.

    Returns:
      destination: The expanded, absolute, validated destination.

    Raises:
      ValueError: The path is not normalized, it names the filesystem root
        either literally or after resolving symlinks, or it resolves to (or
        shares an inode with) a protected input.

    """
    destination = Path(path).expanduser().absolute()
    if destination == Path(destination.anchor):
        raise ValueError(f"runtime output path must not be root: {path}")
    if Path(os.path.normpath(destination)) != destination:
        raise ValueError(f"runtime output path must be normalized: {path}")
    resolved = destination.resolve(strict=False)
    if resolved == Path(resolved.anchor):
        raise ValueError(f"runtime output path must not be root: {path}")
    entries = protected if isinstance(protected, Iterable) else protected()
    guarded = {entry.expanduser().resolve() for entry in entries}
    aliases_by_inode = destination.exists() and any(
        entry.exists() and destination.samefile(entry) for entry in guarded
    )
    if resolved in guarded or aliases_by_inode:
        raise ValueError(f"output path aliases protected input artifact: {destination}")
    return destination
