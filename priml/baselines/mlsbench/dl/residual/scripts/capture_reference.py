#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the dl-residual-connection script's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``dl-residual-connection`` script and each
baseline's edit of it from a clone of the benchmark, runs the port on the same
inputs under the same pins, and compares with exact equality: every block alone
on fixed features, identity and projected, over four training forwards with
their gradients and random draws and one evaluation forward; the learning rate
of all 200 epochs, the optimizer and the reported accuracy; ResNet-56's initial
state and its training and evaluation forwards with every block; where the
generator stands after building each; every pixel the loaders produce; and
every parameter after each update of a short run per block. Any mismatch is
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

import torch

from priml.baselines.mlsbench.dl.residual import reference, upstream
from priml.baselines.mlsbench.dl.vision import (
    reference as vision,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.scripts import (
    capture_reference as vision_capture,
)
from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches
from priml.train.parallelism import NoParallel


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
            name: upstream.upstream_lockstep(script, name, traces["basic"])
            for name, script in scripts.items()
        }
        record = upstream.upstream_record(scripts, traces, lockstep)
        batches = vision_upstream.upstream_batches(scripts["basic"])
        report = vision_capture.compare_full_size(scripts["basic"], traces["basic"])
        for name in upstream.EDITED:
            report += compare_full_network(scripts[name], name, traces[name])
    report += mismatches(record, reference.port_record())
    report += vision_capture.compare_lockstep(lockstep, run=reference.lockstep_record)
    report += vision_capture.compare_loaders(batches, vision.loader_batches(scratch))
    report += check_branches(record)
    return vision.Captured(record=record, sites={}, report=report)


def check_branches(record: dict[str, Tensor]) -> list[str]:
    """Report if the stochastic-depth block runs never kept, or never dropped, a branch.

    Args:
      record: The reference's values.

    Returns:
      report: A line if the draws of every standalone stochastic-depth run
        together fall on one side of 0.5, the survival of each such block.

    """
    draws = torch.cat(
        [
            value
            for key, value in record.items()
            if key.startswith("block/stochastic_depth/") and key.endswith("/draws")
        ],
    )
    if bool((draws < 0.5).any()) and bool((draws >= 0.5).any()):
        return []
    return ["block/stochastic_depth: the draws reach only one branch"]


def compare_full_network(
    script: vision_upstream.Upstream,
    name: str,
    trace: vision_upstream.MainTrace,
) -> list[str]:
    """Compare ResNet-56 with one variant's block: initial state and both forwards.

    Each side's training forward runs from the task's seed, so a block that
    draws in its forward draws the same numbers on both sides.

    Args:
      script: The script carrying the block.
      name: A key of :data:`reference.VARIANTS`.
      trace: That script's ``main``, whose initial state is compared.

    Returns:
      report: One line per tensor that differs, prefixed with ``name``.

    """
    config = reference.VARIANTS[name]().step
    config.parallelism = NoParallel.Config(device="cpu")
    task = cast(upstream.Script, cast(object, script))
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        step = config.make()
        # Built after the port, so a block that reads a class-wide counter in
        # its forward sees this network's count.
        theirs = task.ResNet(task.CustomBlock, [9, 9, 9], 100)
    assert isinstance(step, VisionTrainStep)
    ours = step.model
    theirs.load_state_dict(trace.state)
    report = vision_capture.compare_states(trace.state, ours.state_dict())
    media = vision.fixed_media(0)
    for mode in ("train", "eval"):
        outputs: list[dict[str, Tensor]] = []
        for model in (theirs, ours):
            model.train(mode == "train")
            with vision.seeded(vision.SEED), host_agnostic_numerics():
                outputs.append(
                    {f"forward/{mode}": vision.logits_of(model, media).detach()},
                )
        report += mismatches(*outputs)
    return [f"full/{name}: {line}" for line in report]


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
