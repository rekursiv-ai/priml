#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the dl-activation-function script's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``dl-activation-function`` script and each
baseline's edit of it from a clone of the benchmark, runs the port on the same
inputs under the same pins, and compares with exact equality: every activation
and its gradient on fixed inputs, the rate of all 200 epochs, the optimizer,
the initial state and forward of ResNet-20 (the setting's) and ResNet-56,
where the generator stands after building the network, every pixel the
CIFAR-10 loaders produce, and every parameter after each update of a short run
per activation. Any mismatch is printed and the exit code is 1; otherwise the
values the default test replays are written under ``--out``.

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

from priml.baselines.mlsbench.dl.activation import reference, upstream
from priml.baselines.mlsbench.dl.activation.experiments import exp000
from priml.baselines.mlsbench.dl.vision import (
    reference as vision,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.testing.bfb import host_agnostic_numerics
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
    scripts = upstream.variants(root, scratch)
    pristine = scripts["relu"]
    with vision_upstream.hidden_accelerators():
        trace = upstream.trace_setting(pristine)
        lockstep = {
            name: upstream.upstream_lockstep(script, name, trace)
            for name, script in scripts.items()
        }
        record = upstream.upstream_record(scripts, trace, lockstep)
        batches = vision_upstream.upstream_batches(pristine, dataset=upstream.DATASET)
        report = compare_setting(pristine, trace)
        report += capture_reference.compare_full_size(
            pristine,
            vision_upstream.trace_main(pristine),
        )
    report += mismatches(record, reference.port_record())
    report += capture_reference.compare_lockstep(
        lockstep,
        run=reference.lockstep_record,
    )
    report += capture_reference.compare_loaders(
        batches,
        reference.loader_batches(scratch),
    )
    return vision.Captured(record=record, sites={}, report=report)


def compare_setting(
    script: vision_upstream.Upstream,
    trace: vision_upstream.MainTrace,
) -> list[str]:
    """Compare ResNet-20's initial state and forward with the script's.

    Args:
      script: The pristine script.
      trace: Its ``main``'s initial state at the ``resnet20-cifar10`` setting.

    Returns:
      report: One line per tensor that differs.

    """
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        ours = vision.full_step(exp000).model
        theirs = script.ResNet(script.BasicBlock, [3, 3, 3], upstream.CLASSES)
        theirs.load_state_dict(trace.state)
        report = capture_reference.compare_states(trace.state, ours.state_dict())
        media = vision.fixed_media(0)
        for mode in ("train", "eval"):
            ours.train(mode == "train")
            theirs.train(mode == "train")
            report += mismatches(
                {f"setting/{mode}": vision.logits_of(theirs, media).detach()},
                {f"setting/{mode}": vision.logits_of(ours, media).detach()},
            )
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
