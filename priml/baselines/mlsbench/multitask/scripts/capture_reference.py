#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the cv-multitask-loss script's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``cv-multitask-loss`` script and each baseline's
edit of it from a clone of the benchmark, runs the port on the same inputs
under the same pins, and compares with exact equality: each combination's
value and gradients on fixed losses, the superclass map, the 200 learning
rates, the optimizer and the size and order of every tensor it updates, the
reported accuracy, the two-head ResNet-56's initial state and forward, where
the generator stands after building each experiment, every pixel and label of
the loaders, and every parameter after two updates per combination. Any
mismatch is printed and the exit code is 1; otherwise the values the default
test replays are written under ``--out``.

The ``pcgrad`` edit's graph walk skips a heap-dependent subset of the
parameters, so its instances are pinned to the port's full set and its own
walk is checked to be a subsequence of it.

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

from priml.baselines.mlsbench.multitask import reference, upstream
from priml.baselines.mlsbench.pytorch_vision import (
    reference as vision,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.pytorch_vision.scripts import (
    capture_reference as vision_capture,
)
from priml.testing.bfb import host_agnostic_numerics
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
        traces = {name: upstream.trace_main(s) for name, s in scripts.items()}
        lockstep = {
            name: upstream.upstream_lockstep(script, name, traces["sum"])
            for name, script in scripts.items()
        }
        record = upstream.upstream_record(scripts, traces, lockstep)
        batches = upstream.upstream_batches(scripts["sum"])
        report = compare_wiring(traces) + compare_full_size(scripts, traces["sum"])
    report += mismatches(record, reference.port_record())
    report += vision_capture.compare_lockstep(lockstep, run=reference.lockstep_record)
    report += vision_capture.compare_loaders(
        batches,
        reference.loader_batches(scratch),
    )
    return vision.Captured(record=record, sites={}, report=report)


def compare_wiring(traces: dict[str, upstream.Trace]) -> list[str]:
    """Check what every variant's ``main`` told its training epochs.

    Args:
      traces: A run of each variant's ``main``.

    Returns:
      report: A line per variant whose epochs are not zero-based ``e`` of 200.

    """
    expected = [(epoch, 200) for epoch in range(200)]
    return [
        f"main/{name}: epoch arguments differ"
        for name, trace in traces.items()
        if trace.epochs != expected
    ]


def compare_full_size(
    scripts: dict[str, upstream.Script],
    trace: upstream.Trace,
) -> list[str]:
    """Compare the two-head ResNet-56's initial state and forward with the script's.

    Also checks the ``pcgrad`` walk on the full network's fine loss.

    Args:
      scripts: Each variant's script.
      trace: The pristine ``main``'s initial state, taken after its seeding.

    Returns:
      report: One line per tensor that differs, and any walk finding.

    """
    pristine = scripts["sum"]
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        ours = reference.full_step(reference.COMBINATIONS["sum"]).model
        theirs = pristine.ResNet(pristine.BasicBlock, [9, 9, 9])
        theirs.load_state_dict(trace.state)
        report = vision_capture.compare_states(trace.state, ours.state_dict())
        media = vision.fixed_media(0)
        for mode in ("train", "eval"):
            getattr(ours, mode)()
            getattr(theirs, mode)()
            want = cast("tuple[Tensor, Tensor]", theirs(media))
            got = cast("tuple[Tensor, Tensor]", ours(media))
            report += mismatches(
                {f"forward/{mode}/{i}": t.detach() for i, t in enumerate(want)},
                {f"forward/{mode}/{i}": t.detach() for i, t in enumerate(got)},
            )
    theirs.train()
    return report + upstream.check_walk(scripts["pcgrad"], theirs)


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
