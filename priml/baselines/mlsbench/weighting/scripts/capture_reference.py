#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the cv-sample-weighting script's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``cv-sample-weighting`` script and each baseline's
edit of it from a clone of the benchmark, runs the port on the same inputs under
the same pins, and compares with exact equality: every weighting's weights and
weighted loss with its gradient, the class counts and weights ``main`` builds
for the ``resnet32-cifar100lt`` setting, the positions its long-tail cut keeps
of CIFAR-100's training split (real when ``--cifar100`` holds it, else of the
label layout alone), the rate of all 200 epochs, the optimizer, ResNet-32's
initial state and forward, where the generator stands after building it, every
pixel the long-tailed loaders produce, and every parameter after each update of
a short run per weighting. Any mismatch is printed and the exit code is 1;
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

from priml.baselines.mlsbench.pytorch_vision import (
    reference as vision,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.pytorch_vision.scripts import (
    capture_reference as vision_capture,
)
from priml.baselines.mlsbench.weighting import reference, upstream
from priml.baselines.mlsbench.weighting.data import long_tail_indices
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
        captured = capture(flags.root, Path(scratch), cifar100=flags.cifar100)
    if captured.report:
        print("\n".join(captured.report))
        print(f"{len(captured.report)} mismatches; nothing written.")
        return 1
    flags.out.mkdir(parents=True, exist_ok=True)
    write_tensors(flags.out / "reference.pt", captured.record)
    print(f"0 mismatches; wrote {len(captured.record)} values to {flags.out}.")
    return 0


def capture(root: Path, scratch: Path, *, cifar100: Path | None) -> vision.Captured:
    """Run both sides and compare them.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts and the loader split go to.
      cifar100: Directory holding CIFAR-100's raw ``train``, ``test`` and
        ``meta`` files, whose training split is cut on both sides; ``None``
        skips that comparison.

    Returns:
      captured: The script's values, keyed as the port's are.

    """
    scripts = upstream.variants(root, scratch)
    pristine = scripts["uniform"]
    with vision_upstream.hidden_accelerators():
        trace = upstream.trace_main(pristine)
        lockstep = {
            name: upstream.upstream_lockstep(script, name, trace)
            for name, script in scripts.items()
        }
        batches = upstream.upstream_batches(pristine)
        record = upstream.upstream_record(
            trace,
            upstream.upstream_weights(scripts),
            lockstep,
            upstream.upstream_subset(pristine),
            batches[0],
        )
        report = compare_full_size(pristine, trace)
    _, indices = reference.counts_record()
    port = reference.port_record(scratch)
    report += mismatches(record, port)
    report += mismatches({"main/indices": trace.indices}, {"main/indices": indices})
    report += compare_lockstep(lockstep)
    report += vision_capture.compare_loaders(batches, reference.loader_batches(scratch))
    if cifar100 is not None:
        report += compare_real(pristine, cifar100, scratch)
    return vision.Captured(record=record, sites={}, report=report)


def compare_full_size(script: upstream.Script, trace: upstream.Trace) -> list[str]:
    """Compare ResNet-32's initial state and forward with the script's.

    Args:
      script: The pristine script.
      trace: Its ``main``'s initial state, taken after its own seeding.

    Returns:
      report: One line per tensor that differs.

    """
    state = trace.main.state
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        ours = reference.full_step().model
        theirs = script.ResNet(script.BasicBlock, [5, 5, 5], reference.CLASSES)
        theirs.load_state_dict(state)
        report = vision_capture.compare_states(state, ours.state_dict())
        media = vision.fixed_media(0)
        for mode in ("train", "eval"):
            ours.train(mode == "train")
            theirs.train(mode == "train")
            report += mismatches(
                {f"forward/{mode}": vision.logits_of(theirs, media).detach()},
                {f"forward/{mode}": vision.logits_of(ours, media).detach()},
            )
    return report


def compare_real(script: upstream.Script, cifar100: Path, scratch: Path) -> list[str]:
    """Cut the real training split on both sides and compare positions and counts.

    Args:
      script: The pristine script.
      cifar100: Directory holding the raw ``train``, ``test`` and ``meta`` files.
      scratch: Directory the root torchvision expects is linked under.

    Returns:
      report: One line per tensor that differs.

    """
    data_root = scratch / "cifar"
    data_root.mkdir()
    (data_root / "cifar-100-python").symlink_to(cifar100.resolve())
    indices, counts, label = upstream.real_subset(script, data_root)
    ours, our_counts = long_tail_indices(
        label,
        imbalance_ratio=reference.RATIO,
        num_classes=reference.CLASSES,
        seed=vision.SEED,
    )
    return mismatches(
        {"real/indices": indices, "real/counts": counts},
        {"real/indices": ours, "real/counts": our_counts},
    )


def compare_lockstep(lockstep: dict[str, vision_upstream.Lockstep]) -> list[str]:
    """Compare every parameter and buffer after each update of every weighting.

    Args:
      lockstep: Each weighting's :func:`upstream.upstream_lockstep`.

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
    cifar100: Path


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
    parser.add_argument(
        "--cifar100",
        type=Path,
        default=Path("/opt/scratch/datasets/cifar100"),
        help="CIFAR-100's raw train, test and meta files, for the real cut.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
