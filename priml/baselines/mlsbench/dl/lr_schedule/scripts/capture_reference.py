#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the dl-lr-schedule script's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``dl-lr-schedule`` script and each baseline's edit
of it from a clone of the benchmark, runs the port on the same inputs under the
same pins, and compares with exact equality: the rate of all 200 epochs of
every schedule, the optimizer, ResNet-56's initial state and forward, where
the generator stands after building it, every pixel the loaders produce, and
every parameter after each update of a short run per schedule. Any mismatch is
printed and the exit code is 1; otherwise the values the default test replays
are written under ``--out``.

The script is imported from the clone, not vendored. Nothing runs on a GPU.

Examples:
  capture_reference.py --root /opt/scratch/mlsbench --out testdata

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import tempfile

from priml.baselines.mlsbench.dl.lr_schedule import reference, upstream
from priml.baselines.mlsbench.dl.lr_schedule.experiments import exp000
from priml.baselines.mlsbench.dl.vision import (
    reference as vision,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.scripts import (
    capture_reference as vision_capture,
)
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from torch import Tensor


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
            name: upstream.upstream_lockstep(script, name, traces["constant"])
            for name, script in scripts.items()
        }
        record = upstream.upstream_record(traces, lockstep)
        record["draws/exp000"] = traces["constant"].draws
        batches = vision_upstream.upstream_batches(scripts["constant"])
        report = vision_capture.compare_full_size(
            scripts["constant"],
            traces["constant"],
            exp000,
        )
    port = reference.port_record() | vision.draws_record(exp000)
    report += mismatches(record, port)
    report += compare_lockstep(lockstep)
    report += vision_capture.compare_loaders(batches, vision.loader_batches(scratch))
    return vision.Captured(record=record, sites={}, report=report)


def compare_lockstep(lockstep: dict[str, vision_upstream.Lockstep]) -> list[str]:
    """Compare every parameter and buffer after each update of every schedule.

    Args:
      lockstep: Each schedule's :func:`upstream.upstream_lockstep`.

    Returns:
      report: One line per tensor that differs after any update.

    """
    report: list[str] = []
    for name, (_, theirs) in lockstep.items():
        ours: list[dict[str, Tensor]] = []
        _ = reference.lockstep_record(name, states=ours)
        for index, (want, got) in enumerate(zip(theirs, ours, strict=True), start=1):
            report += [
                f"lockstep/{name}/{index}: {line}"
                for line in vision_capture.compare_states(want, got)
            ]
    return report


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
