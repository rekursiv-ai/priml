"""Read a run's ``FileTracker`` metrics file and print a one-line summary.

A scored run writes its ``eval/*`` scalars to a JSON metrics file via
``FileTracker``. This is the shared reader every task's ``results.py`` delegates
to: read the file, format a
one-line ``key=value`` summary, and expose a ``summarize`` CLI. Torch-free --
importing it (and a task's ``results.py`` shim) costs nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import argparse
import json
import sys

from priml.lib.custom_json import DictCodec, FloatCodec


def read_metrics(path: Path) -> dict[str, float]:
    """Read a metrics JSON as a flat ``{key: value}`` mapping.

    Args:
      path: Path to the metrics JSON written by the run.

    Returns:
      metrics: The decoded ``eval/*`` scalar mapping.

    Raises:
      FileNotFoundError: If ``path`` does not exist.

    """
    decoded = json.loads(path.read_text())
    return {k: FloatCodec.coerce(v) for k, v in DictCodec.coerce(decoded).items()}


def format_summary(
    metrics: dict[str, float],
    keys: Sequence[str] | None = None,
) -> str:
    """Format metrics as a one-line ``key=value`` summary (``eval/`` stripped).

    Args:
      metrics: The decoded metrics mapping.
      keys: Ordered keys to show (missing ones skipped); ``None`` shows every
        key, sorted.

    Returns:
      summary: The one-line summary, or a "no eval metrics found" placeholder.

    """
    selected = list(keys) if keys is not None else sorted(metrics)
    parts = [
        f"{key.removeprefix('eval/')}={metrics[key]}"
        for key in selected
        if key in metrics
    ]
    return " ".join(parts) if parts else "no eval metrics found"


def summarize(
    argv: Sequence[str] | None = None,
    *,
    keys: Sequence[str] | None = None,
    note: Callable[[dict[str, float]], str] | None = None,
) -> int:
    """Read the metrics file at ``--path`` and print a one-line summary.

    The shared entry point for per-task ``results.py`` shims.

    Args:
      argv: Command-line args (defaults to ``sys.argv``).
      keys: Ordered metric keys to show; ``None`` shows all.
      note: Optional hook returning extra text to append (e.g. a task-specific
        sentinel note); receives the metrics mapping.

    Returns:
      code: Process exit code (0 on success, 1 if the file is missing).

    """
    parser = argparse.ArgumentParser(description="Read a run's eval metrics.")
    parser.add_argument(
        "--path",
        default="metrics.json",
        help="Metrics JSON path (default: metrics.json).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    path = Path(args.path)
    if not path.exists():
        print(f"No metrics file at {path}.", file=sys.stderr)  # noqa: T201
        return 1
    metrics = read_metrics(path)
    summary = format_summary(metrics, keys)
    if note is not None:
        summary += note(metrics)
    print(summary)  # noqa: T201
    return 0
