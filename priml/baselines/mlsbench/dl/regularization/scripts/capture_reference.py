#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the dl-regularization script's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``dl-regularization`` script and each baseline's
edit of it from a clone of the benchmark, runs the port on the same inputs
under the same pins, and compares with exact equality: each regularizer's value
and gradients on a probe network, the setting's batch size, the rate of all 200
epochs, the optimizer, ResNet-56's initial state and forward, where the
generator stands after building it, every pixel the loaders produce, and every
parameter after each update of a short run per regularizer. ``main`` is traced
with the arguments of the setting's launch script. Any mismatch is printed and
the exit code is 1; otherwise the values the default test replays are written
under ``--out``.

The script is imported from the clone, not vendored. Nothing runs on a GPU.

Examples:
  capture_reference.py --root /opt/scratch/mlsbench --out testdata

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import argparse
import tempfile

from priml.baselines.mlsbench.dl.regularization import reference, upstream
from priml.baselines.mlsbench.dl.vision import (
    reference as vision,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.testing.golden import mismatches


def main() -> int:
    """Capture and compare; return the process exit code.

    Returns:
      code: 0 when every comparison is exact and the golden was written, 1 when
        any comparison differs.

    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(_Flags, parser.parse_args())
    with tempfile.TemporaryDirectory(prefix="capture-reference-") as scratch:
        captured = capture(flags.root, Path(scratch))
    if captured.report:
        print("\n".join(captured.report))
        print(f"{len(captured.report)} mismatches; nothing written.")
        return 1
    capture_reference.write_records(flags.out, {"reference.pt": captured.record})
    print(f"0 mismatches; wrote {len(captured.record)} values to {flags.out}.")
    return 0


def capture(root: Path, scratch: Path) -> vision.Captured:
    """Run both sides and compare them.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts and the loader split go to.

    Returns:
      captured: The script's values, keyed as the port's are.

    """
    report = check_pristine(root)
    scripts = upstream.variants(root, scratch)
    argv = upstream.setting_argv(root)
    with vision_upstream.hidden_accelerators():
        trace = upstream.trace_setting(scripts["zero"], argv)
        lockstep = {
            name: upstream.upstream_lockstep(script, name, trace)
            for name, script in scripts.items()
        }
        record = upstream.upstream_record(
            trace,
            upstream.upstream_regularizers(scripts),
            lockstep,
        )
        record["setting/batch_size"] = reference.setting_batch_size(argv)
        batches = vision_upstream.upstream_batches(scripts["zero"])
        report += capture_reference.compare_wiring(trace)
        report += capture_reference.compare_full_size(scripts["zero"], trace)
    port = reference.port_record()
    report += mismatches(record, port)
    report += capture_reference.compare_lockstep(
        lockstep,
        run=reference.lockstep_record,
    )
    report += capture_reference.compare_loaders(batches, vision.loader_batches(scratch))
    return vision.Captured(record=record, sites={}, report=report)


def check_pristine(root: Path) -> list[str]:
    """Report a pristine regularizer that does not raise, as the port assumes it does.

    The port's ``exp000`` adds zero because the published body raises and its
    docstring names zero as "no regularization".

    Args:
      root: Clone of the benchmark.

    Returns:
      report: A line if the pristine regularizer returns, else nothing.

    """
    return [] if upstream.pristine_raises(root) else ["pristine: does not raise"]


class _Flags(Protocol):
    """Parsed command-line flags."""

    root: Path
    out: Path


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Clone of the benchmark at the pinned commit.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="The testdata directory the golden is written to.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
