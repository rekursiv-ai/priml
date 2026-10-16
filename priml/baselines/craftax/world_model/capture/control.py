"""Files through which capture jobs coordinate: the archive halt and launch markers.

Every capture worker and the replay verifier share one archive root. A replay
mismatch or a damaged shard writes ``HALT.json`` at that root, and every worker
of every arm stops at its next poll. A worker of a launch writes
``started/{launch}/arm{arm}-w{worker}.json`` when its job starts and
``complete/{launch}/arm{arm}-w{worker}.json`` once it met its budget; the
verifier of that launch counts a worker as silent only between the two, and
stops only once all of its workers are complete, or once the archive is
halted. Markers are counted by launch id, so no two launches share one.
"""

from pathlib import Path

import json
import os


class CaptureHaltedError(RuntimeError):
    """Capture of the whole archive was halted by a failed verification."""


def halt(root: Path, *, shard: str, reason: str) -> None:
    """Halt every capture worker under ``root``; the first halt's reason is kept.

    Args:
      root: Archive root.
      shard: Shard that failed, relative to ``root``.
      reason: What failed.

    """
    path = root / "HALT.json"
    if path.exists():
        return
    _write_atomic(path, {"shard": shard, "reason": reason})


def check_halt(root: Path) -> None:
    """Raise when capture under ``root`` has been halted.

    Args:
      root: Archive root.

    Raises:
      CaptureHaltedError: ``root`` holds ``HALT.json``.

    """
    path = root / "HALT.json"
    if path.exists():
        raise CaptureHaltedError(f"{path}: {path.read_text().strip()}")


def mark_started(root: Path, *, launch: str, arm: int, worker: int) -> None:
    """Record that a worker of ``launch`` left the queue and is running.

    Args:
      root: Archive root.
      launch: Launch that started the worker.
      arm: The worker's arm.
      worker: The worker's index.

    """
    directory = root / "started" / launch
    directory.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        directory / f"arm{arm}-w{worker}.json",
        {"arm": arm, "worker": worker},
    )


def started(root: Path, *, launch: str) -> int:
    """Return how many workers of ``launch`` have started."""
    return len(list((root / "started" / launch).glob("*.json")))


def mark_complete(
    root: Path,
    *,
    launch: str,
    arm: int,
    worker: int,
    decisions: int,
) -> None:
    """Record that a worker of ``launch`` met its budget and published everything.

    Args:
      root: Archive root.
      launch: Launch that started the worker.
      arm: The worker's arm.
      worker: The worker's index.
      decisions: Decisions the worker has published over all its runs.

    """
    directory = root / "complete" / launch
    directory.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        directory / f"arm{arm}-w{worker}.json",
        {"arm": arm, "worker": worker, "decisions": decisions},
    )


def completed(root: Path, *, launch: str) -> int:
    """Return how many workers of ``launch`` have marked themselves complete."""
    return len(list((root / "complete" / launch).glob("*.json")))


def _write_atomic(path: Path, record: dict[str, object]) -> None:
    """Write one JSON line to ``path`` through a fsynced temporary file."""
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.rename(path)
