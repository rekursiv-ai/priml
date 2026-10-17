#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the dl-normalization script's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``dl-normalization`` script and each baseline's
edit of it from a clone of the benchmark, runs the port on the same inputs
under the same pins, and compares with exact equality: each normalization
layer's output and gradients on fixed features, ResNet-56's initial state and
forward with every normalization, where the generator stands after building
it, the rate of all 200 epochs, the optimizer, the reported accuracy, every
pixel the loaders produce, and every parameter after each update of a short
run per normalization. Any mismatch is printed and the exit code is 1;
otherwise the values the default test replays are written under ``--out``.

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

from priml.baselines.mlsbench.normalization import reference, upstream
from priml.baselines.mlsbench.pytorch_vision import (
    reference as vision,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.pytorch_vision.scripts import (
    capture_reference as vision_capture,
)
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
    vision_capture.write_records(flags.out, {"reference.pt": captured.record})
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
    scripts = upstream.variants(root, scratch)
    with vision_upstream.hidden_accelerators():
        traces = {
            name: vision_upstream.trace_main(script) for name, script in scripts.items()
        }
        lockstep = {
            name: upstream.upstream_lockstep(script, name, traces["batch_norm"])
            for name, script in scripts.items()
        }
        record = upstream.upstream_record(scripts, traces, lockstep)
        batches = vision_upstream.upstream_batches(scripts["batch_norm"])
        report = [
            f"full/{name}: {line}"
            for name, factory in reference.NORMS.items()
            for line in vision_capture.compare_full_size(
                scripts[name],
                traces[name],
                factory,
            )
        ]
    report += mismatches(record, reference.port_record())
    report += vision_capture.compare_lockstep(lockstep, run=reference.lockstep_record)
    report += vision_capture.compare_loaders(batches, vision.loader_batches(scratch))
    return vision.Captured(record=record, sites={}, report=report)


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
