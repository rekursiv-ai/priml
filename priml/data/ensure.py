"""Resumable, idempotent data-ensure primitive.

``ensure_data(spec)`` brings a target directory into agreement with a
declared manifest of expected files. It runs at job launch (experiment
side), not in the scheduler, and is safe to call repeatedly:

- All manifest files present and matching -> :data:`EnsureResult.PRESENT`,
  no fetch.
- Any file missing or wrong size -> the per-file ``fetch`` closure is
  invoked, then integrity is re-verified against the full manifest.
- A file that is the right size but the wrong sha256 (genuine corruption)
  -> the whole target directory is archived to ``<dir>.corrupt.<utc_ts>``
  and rebuilt from scratch, so a half-state never later reads as present.

A common-case resumable HTTP helper, :func:`resumable_http_download`, is
provided: it streams to a ``<dest>.part`` file, uses an HTTP ``Range``
request to resume an interrupted transfer, and renames into place only on
completion.

Resume capability is per-transport. The primitive's own HTTP helper
supports true byte-range resume. A fetch that cannot resume (for example a
``git clone`` or a plain directory copy) simply re-runs from clean; the
archive-and-restart path guarantees that is always safe.

Cross-job safety (independent jobs sharing one scratch path):

- A POSIX advisory lock (``fcntl.flock``) on ``<target_dir>.lock``
  serializes the build across processes and nodes, so a second job blocks
  until the first finishes its build, then re-checks (and typically finds
  the data already present).
- A completion marker, ``<target_dir>/.ensure_complete``, is written only
  after the full manifest verifies. The data is accepted as
  :data:`EnsureResult.PRESENT` only when the marker exists, so a
  mid-build tree (files present but unverified -- the failure mode of an
  existence-only manifest) is never mistaken for complete.

The marker fast-path is checked before the lock, so a steady-state
present dataset never contends for the lock. ``flock`` requires
filesystem support; the cluster's Weka scratch FS supports it across
nodes. A filesystem without ``flock`` support degrades to the prior
single-job behavior (the marker still rejects partial trees within a
single node).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import contextlib
import datetime
import enum
import fcntl
import hashlib
import http.client
import logging
import shutil
import sys
import urllib.request

from tqdm import tqdm


if TYPE_CHECKING:
    from collections.abc import Generator


logger = logging.getLogger(__name__)


@dataclass(slots=True, kw_only=True, frozen=True)
class _EnsureSettings:
    """Tunable knobs for :func:`ensure_data`, grouped into one named scope."""

    chunk_bytes: int = 1 << 20
    """Streaming read/write chunk size for hashing and HTTP download."""

    marker_name: str = ".ensure_complete"
    """Sentinel file written under a target dir once its manifest fully verifies."""


_SETTINGS = _EnsureSettings()


class EnsureResult(enum.Enum):
    """Outcome of :func:`ensure_data`."""

    PRESENT = "present"
    """Every manifest file was already present and matching; no fetch ran."""

    DOWNLOADED = "downloaded"
    """At least one file was fetched; the manifest verified afterwards."""


class Fetch(Protocol):
    """Closure that materializes a single manifest file.

    Implementations download (or copy, or clone) ``rel_path`` from the
    upstream source into ``dest``, creating parent directories as needed.
    They may resume from a partial ``dest`` if the transport supports it;
    otherwise they should overwrite. A failure must raise.
    """

    def __call__(self, *, rel_path: str, dest: Path) -> None:
        """Apply to the input."""
        ...


@dataclass(slots=True, kw_only=True, frozen=True)
class FileSpec:
    """One expected file, relative to the target directory.

    Attributes:
      rel_path: Path relative to ``DataSpec.target_dir``.
      size: Expected byte size, or ``None`` to skip the size check.
      sha256: Expected lowercase hex sha256, or ``None`` to skip it.

    """

    rel_path: str
    size: int | None = None
    sha256: str | None = None


@dataclass(slots=True, kw_only=True, frozen=True)
class DataSpec:
    """Declarative description of a dataset to ensure on local disk.

    Attributes:
      target_dir: Directory the manifest paths are resolved against.
      manifest: Expected files with optional size/sha256 integrity data.
      fetch: Closure invoked per missing/partial file to materialize it.

    """

    target_dir: Path
    manifest: list[FileSpec] = field(default_factory=list)
    fetch: Fetch


def ensure_data(spec: DataSpec) -> EnsureResult:
    """Ensure ``spec.target_dir`` matches ``spec.manifest`` on local disk.

    Idempotent: a call whose manifest already verifies returns
    :data:`EnsureResult.PRESENT` without invoking ``spec.fetch``. Integrity
    is always re-checked against the full manifest after any fetch before
    the result is declared.

    Args:
      spec: The dataset specification (target, manifest, fetch closure).

    Returns:
      result: :data:`EnsureResult.PRESENT` if nothing was fetched, else
        :data:`EnsureResult.DOWNLOADED`.

    Raises:
      RuntimeError: The manifest still fails verification after fetching.

    """
    if _is_complete(spec):
        logger.info(
            "ensure_data: %s present (%d files)",
            spec.target_dir,
            len(spec.manifest),
        )
        return EnsureResult.PRESENT

    with _build_lock(spec.target_dir):
        return _ensure_locked(spec)


def resumable_http_download(*, url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``, resuming an interrupted prior transfer.

    Streams into ``<dest>.part`` and renames onto ``dest`` only after the
    body is fully received, so ``dest`` never exists in a half-written
    state. If ``<dest>.part`` already exists, its length is sent as an HTTP
    ``Range`` header and the server's bytes are appended; servers that
    ignore ranges (HTTP 200) cause a clean restart.

    Args:
      url: Source URL.
      dest: Destination path. A sibling ``<dest>.part`` is used while
        streaming.

    Raises:
      urllib.error.URLError: The request could not be completed.

    """
    part = dest.with_suffix(dest.suffix + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0

    request = urllib.request.Request(url)  # noqa: S310 -- Dataset builders download only configured public source URLs supplied by the caller.
    if have:
        request.add_header("Range", f"bytes={have}-")

    response = cast(
        http.client.HTTPResponse,
        urllib.request.urlopen(request),  # noqa: S310 -- Dataset builders download only configured public source URLs.
    )
    with response:
        resumed = response.status == 206
        mode = "ab" if resumed else "wb"
        if not resumed:
            have = 0
        total = _content_length(response, already=have, resumed=resumed)
        with (
            part.open(mode) as out,
            tqdm(
                total=total,
                initial=have,
                desc=dest.name,
                unit="B",
                unit_scale=True,
                disable=not sys.stdout.isatty(),
            ) as bar,
        ):
            while chunk := response.read(_SETTINGS.chunk_bytes):
                out.write(chunk)
                bar.update(len(chunk))

    part.replace(dest)


def _content_length(
    response: http.client.HTTPResponse,
    *,
    already: int,
    resumed: bool,
) -> int | None:
    """Best-effort total size in bytes for the progress bar, or ``None``."""
    raw = response.headers.get("Content-Length")
    if raw is None:
        return None
    length = int(raw)
    return already + length if resumed else length


def _unsatisfied_files(spec: DataSpec) -> list[FileSpec]:
    """Manifest entries that are missing or fail their size check."""
    return [f for f in spec.manifest if not _file_ok(spec.target_dir / f.rel_path, f)]


def _corrupt_files(spec: DataSpec) -> list[FileSpec]:
    """Present, right-size, but wrong-sha256 entries (genuine corruption)."""
    bad: list[FileSpec] = []
    for f in spec.manifest:
        path = spec.target_dir / f.rel_path
        if f.sha256 is None or not path.is_file():
            continue
        if f.size is not None and path.stat().st_size != f.size:
            continue  # Wrong size is partial, not corrupt; handled by refetch.
        if _sha256(path) != f.sha256:
            bad.append(f)
    return bad


def _file_ok(path: Path, spec: FileSpec) -> bool:
    """Report whether ``path`` satisfies ``spec`` (existence, size, sha256)."""
    if not path.is_file():
        return False
    if spec.size is not None and path.stat().st_size != spec.size:
        return False
    return spec.sha256 is None or _sha256(path) == spec.sha256


def _sha256(path: Path) -> str:
    """Streaming sha256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_SETTINGS.chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_dir(target_dir: Path) -> None:
    """Move a corrupt target aside to ``<dir>.corrupt.<utc_ts>``."""
    if not target_dir.exists():
        return
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    archived = target_dir.with_name(f"{target_dir.name}.corrupt.{stamp}")
    logger.warning("ensure_data: archiving corrupt %s -> %s", target_dir, archived)
    shutil.move(str(target_dir), str(archived))


# Another job's build may have completed while we waited for the lock, so the
# completeness check is repeated here before any fetch. The completion marker is cleared
# before fetching and re-written only after the full manifest verifies, so a tree is
# never marked complete while partial.
def _ensure_locked(spec: DataSpec) -> EnsureResult:
    """Re-check under the build lock, then build if still incomplete."""
    if _is_complete(spec):
        logger.info(
            "ensure_data: %s present (%d files)",
            spec.target_dir,
            len(spec.manifest),
        )
        return EnsureResult.PRESENT

    corrupt = _corrupt_files(spec)
    if corrupt:
        _archive_dir(spec.target_dir)

    missing = _unsatisfied_files(spec)
    if not missing:
        logger.info(
            "ensure_data: %s adopting present tree (%d files)",
            spec.target_dir,
            len(spec.manifest),
        )
        _write_marker(spec.target_dir)
        return EnsureResult.PRESENT

    _clear_marker(spec.target_dir)
    logger.info(
        "ensure_data: fetching %d/%d files into %s",
        len(missing),
        len(spec.manifest),
        spec.target_dir,
    )
    for file_spec in missing:
        dest = spec.target_dir / file_spec.rel_path
        logger.info("ensure_data: fetch %s", file_spec.rel_path)
        spec.fetch(rel_path=file_spec.rel_path, dest=dest)

    still_bad = _unsatisfied_files(spec)
    if still_bad:
        raise RuntimeError(
            f"ensure_data: manifest still unsatisfied after fetch: "
            f"{[f.rel_path for f in still_bad]}",
        )
    _write_marker(spec.target_dir)
    logger.info("ensure_data: %s complete", spec.target_dir)
    return EnsureResult.DOWNLOADED


# Serializes builders across processes and nodes on a shared filesystem that supports
# POSIX advisory locks (the cluster's Weka scratch does). The lock file lives beside the
# target so it survives the archive-and-rebuild of a corrupt target.
@contextlib.contextmanager
def _build_lock(target_dir: Path) -> Generator[None]:
    """Hold an exclusive ``flock`` on ``<target_dir>.lock`` for the build."""
    lock_path = target_dir.with_name(target_dir.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _is_complete(spec: DataSpec) -> bool:
    """Report whether the completion marker exists and the manifest still verifies."""
    return (
        spec.target_dir / _SETTINGS.marker_name
    ).is_file() and not _unsatisfied_files(spec)


def _write_marker(target_dir: Path) -> None:
    """Stamp the completion marker after the manifest has fully verified."""
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / _SETTINGS.marker_name).write_text(
        datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ") + "\n",
    )


def _clear_marker(target_dir: Path) -> None:
    """Remove the completion marker before a (re)build makes the tree partial."""
    (target_dir / _SETTINGS.marker_name).unlink(missing_ok=True)
