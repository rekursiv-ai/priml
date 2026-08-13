"""Structured job progress: the ``progress.json`` writer (jobber spec §8).

A job may periodically call :func:`write_progress`; ``jobber status``/``wait``
surface the file verbatim, so agents poll structured progress instead of
grepping log prose. Deliberately tiny and dependency-free: any script the
scheduler launches can afford it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import json
import os

from priml.paths import validated_output_path


def write_progress(
    step: int,
    total: int,
    *,
    working_dir: Path | str,
    metrics: dict[str, float] | None = None,
) -> Path:
    """Atomically write ``<working_dir>/progress.json`` for the scheduler.

    Atomic replacement ensures a concurrent scheduler read never sees a torn
    file.

    Args:
        step: Steps completed so far.
        total: Total steps planned (0 when unknown).
        working_dir: Explicit job directory containing ``progress.json``.
        metrics: Optional scalar metrics worth surfacing (e.g. loss).

    Returns:
        path: The written progress file.

    Raises:
        ValueError: ``working_dir`` is empty.

    """
    if not working_dir:
        raise ValueError("working_dir must not be empty")
    path = validated_output_path(Path(working_dir) / "progress.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "total": total,
        "metrics": dict(metrics or {}),
        "updated_at": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # Per-writer tmp name: torchrun launches N ranks of the same script, and
    # a shared tmp path would let one rank publish another's half-written
    # file (or crash on the vanished tmp after a concurrent rename).
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    _ = tmp.write_text(json.dumps(payload))
    _ = tmp.replace(path)
    return path
