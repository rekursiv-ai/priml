#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Build the synthetic MLS-Bench clone that ``exp_smoke`` reads.

Real tasks need no preparation beyond a clone at the pinned commit: a setting's
container mounts the task directory read-only at ``/workspace/_task``, so the
data a task ships in ``tasks/<task>/data`` is already in place. This script
only writes the synthetic three-setting task ``exp_smoke`` runs against.
Idempotent: the synthetic ``tasks/`` is rebuilt on every call.

Examples:
  python -m priml.baselines.mlsbench.scripts.prepare_data
  python -m priml.baselines.mlsbench.scripts.prepare_data --directory /data/mlsbench_smoke

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import shutil

from priml.baselines.mlsbench.experiments import SMOKE_DIR, MlsbenchEval
from priml.baselines.mlsbench.testing import build_synth_task
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Build the synthetic clone; return the process exit code.

    Args:
      argv: Command-line arguments; ``None`` is ``sys.argv``.

    Returns:
      code: 0 on success.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast("Flags", parser.parse_args(argv))
    shutil.rmtree(flags.directory / "tasks", ignore_errors=True)
    print(f"synthetic clone: {build_synth_task(flags.directory)}")
    return 0


def default_directory() -> Path:
    """Return the synthetic clone ``exp_smoke`` resolves under a default ``base_dir``."""
    return resolve_working_dir(MlsbenchEval().base_dir, SMOKE_DIR)


class Flags(Protocol):
    """Parsed command-line flags."""

    directory: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--directory",
        type=Path,
        default=default_directory(),
        help="Destination of the synthetic clone.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
