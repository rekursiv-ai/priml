#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Regenerate ``eval_script_pins.json`` from an upstream clone.

Hashes every native ``tasks/<task>/scripts`` file of the substantively adapted
tasks. Run after a deliberate upstream bump and review the JSON diff before
committing it.

Examples:
  python -m priml.baselines.mlsbench.scripts.pin_eval_scripts
  python -m priml.baselines.mlsbench.scripts.pin_eval_scripts --root /clones/mls-bench

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import json

from priml.baselines.mlsbench.experiments import MlsbenchEval
from priml.baselines.mlsbench.harbor import PINS_PATH, generate_pins
from priml.paths import resolve_working_dir


if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Write the pins; return the process exit code.

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
    pins = generate_pins(flags.root)
    flags.out.write_text(json.dumps(pins, indent=1, sort_keys=True) + "\n")
    print(f"wrote {sum(map(len, pins.values()))} digests for {len(pins)} tasks")
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    root: Path
    out: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_clone(),
        help="The upstream clone.",
    )
    parser.add_argument("--out", type=Path, default=PINS_PATH, help="Pins file.")


def _default_clone() -> Path:
    """Return the upstream clone a default ``MlsbenchEval`` resolves."""
    cfg = MlsbenchEval()
    return resolve_working_dir(cfg.base_dir, cfg.mlsbench_root)


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
