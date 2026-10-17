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
from typing import TYPE_CHECKING, Protocol, cast

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
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches, write_tensors


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
    flags.out.mkdir(parents=True, exist_ok=True)
    write_tensors(flags.out / "reference.pt", captured.record)
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
        report = vision_capture.compare_full_size(
            scripts["batch_norm"],
            traces["batch_norm"],
        )
        for name in upstream.EDITED:
            report += compare_full_size(scripts[name], traces[name], name)
    report += mismatches(record, reference.port_record())
    report += compare_lockstep(lockstep)
    report += vision_capture.compare_loaders(batches, vision.loader_batches(scratch))
    return vision.Captured(record=record, sites={}, report=report)


def compare_full_size(
    script: vision_upstream.Upstream,
    trace: vision_upstream.MainTrace,
    name: str,
) -> list[str]:
    """Compare ResNet-56's initial state and forward under one baseline's norm.

    Args:
      script: The script carrying the normalization.
      trace: Its ``main``'s initial state, taken after its own seeding.
      name: A key of :data:`reference.NORMS`.

    Returns:
      report: One line per tensor that differs, prefixed with ``name``.

    """
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        ours = reference.full_step(reference.NORMS[name]).model
        theirs = script.ResNet(script.BasicBlock, [9, 9, 9], 100)
        theirs.load_state_dict(trace.state)
        report = vision_capture.compare_states(trace.state, ours.state_dict())
        media = vision.fixed_media(0)
        for mode in ("train", "eval"):
            _ = ours.train(mode == "train")
            _ = theirs.train(mode == "train")
            report += mismatches(
                {f"forward/{mode}": vision.logits_of(theirs, media).detach()},
                {f"forward/{mode}": vision.logits_of(ours, media).detach()},
            )
    return [f"full/{name}: {line}" for line in report]


def compare_lockstep(lockstep: dict[str, vision_upstream.Lockstep]) -> list[str]:
    """Compare every parameter and buffer after each update of every normalization.

    Args:
      lockstep: Each normalization's :func:`upstream.upstream_lockstep`.

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
