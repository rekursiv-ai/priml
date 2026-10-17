#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the reference's values and prove the port reproduces them.

Runs the pristine MLS-Bench vision code from a clone of the benchmark, runs the
port on the same inputs under the same pins, and compares every component with
exact equality: the initial state and forward of the full ResNet-56, the four
losses, the learning rates of all 200 epochs, the optimizer, the image
pipelines, every pixel the loaders produce, and every parameter after each
update of a short run per loss. It also counts the random draws on both sides.
Any mismatch is printed and the exit code is 1; otherwise the values the
default test replays are written under ``--out``.

The reference is imported from the clone, not vendored. Nothing runs on a GPU.

Examples:
  capture_reference.py --root /opt/scratch/mlsbench --out testdata

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import io
import tempfile

import torch

from priml.baselines.mlsbench.dl.vision import reference, upstream
from priml.baselines.mlsbench.dl.vision.experiments import exp000
from priml.baselines.mlsbench.dl.vision.upstream import (
    PRISTINE,
    VARIANTS,
    Batches,
    Lockstep,
    MainTrace,
    Upstream,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches, pack


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torch import Tensor

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


CEILING: Final = 32_768
"""Largest binary golden, in bytes."""

REQUIRED_UPSTREAM: Final = (
    "uniform_@nn/init.py",
    "normal_@nn/init.py",
    "random_@data/dataloader.py",
    "random_@data/sampler.py",
    "randperm@data/sampler.py",
    "randint@transforms/transforms.py",
    "rand@transforms/transforms.py",
)
"""Random call sites of the reference's stack that a capture run must reach."""

REQUIRED_PORT: Final = (
    "uniform_@nn/init.py",
    "normal_@nn/init.py",
    "random_@vision/data.py",
    "randperm@vision/data.py",
    "randint@vision/data.py",
    "rand@vision/data.py",
)
"""Random call sites of the port that a capture run must reach."""


def main() -> int:
    """Capture and compare; return the process exit code.

    Returns:
      code: 0 when every comparison is exact and the goldens were written, 1
        when any comparison differs.

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
    write_goldens(flags.out, captured)
    print(f"0 mismatches; wrote {len(captured.record)} values to {flags.out}.")
    return 0


def capture(root: Path, scratch: Path) -> reference.Captured:
    """Run both sides and compare them.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts and the loader splits go to.

    Returns:
      captured: The reference's values, keyed as the port's are.

    """
    pristine = upstream.load_script(root / PRISTINE)
    variants = {"cross_entropy": pristine} | {
        name: upstream.edited_script(root, file, scratch)
        for name, file in VARIANTS.items()
    }
    with upstream.hidden_accelerators():
        trace = upstream.trace_main(pristine)
        lockstep = {
            name: upstream.upstream_lockstep(script, name, trace)
            for name, script in variants.items()
        }
        batches = upstream.upstream_batches(pristine)
        record = upstream.upstream_record(variants, trace, lockstep, batches[0])
        sites = upstream.upstream_sites(pristine)
        report = compare_wiring(trace) + compare_full_size(pristine, trace)
    port = reference.port_record(scratch)
    port_sites = reference.draw_sites(scratch)
    report += mismatches(record, port)
    report += compare_lockstep(lockstep)
    report += compare_loaders(batches, reference.loader_batches(scratch))
    report += check_sites(sites, port_sites)
    return reference.Captured(
        record=record,
        sites={"reference": sites, "port": port_sites},
        report=report,
    )


def write_goldens(out: Path, captured: reference.Captured) -> None:
    """Write the reference's values under ``out``, within the size ceiling.

    The two image pipelines are stored apart because each holds float32 images
    that already fill most of a file.

    Args:
      out: The ``testdata`` directory.
      captured: The result of :func:`capture`.

    Raises:
      ValueError: A golden exceeds :data:`CEILING`.

    """
    record = captured.record
    write_records(
        out,
        {
            "reference.pt": {
                k: v for k, v in record.items() if not k.startswith("transform/")
            },
            "reference_train_pipeline.pt": {
                "transform/train": record["transform/train"],
            },
            "reference_test_pipeline.pt": {"transform/test": record["transform/test"]},
        },
    )
    lines = [
        f"{side} {key} {count}"
        for side, counts in captured.sites.items()
        for key, count in sorted(counts.items())
    ]
    (out / "reference_sites.txt").write_text("\n".join(lines) + "\n")


def write_records(out: Path, files: Mapping[str, Mapping[str, Tensor]]) -> None:
    """Write each record as a tensor golden, or none of them if any is too large.

    Every file is serialized in memory and measured before the first is
    written, so a refused capture leaves the previous goldens in place.

    Args:
      out: The ``testdata`` directory.
      files: File name to the record it holds.

    Raises:
      ValueError: A golden exceeds :data:`CEILING`.

    """
    encoded: dict[str, bytes] = {}
    for name, values in files.items():
        buffer = io.BytesIO()
        torch.save(pack(values), buffer)
        encoded[name] = buffer.getvalue()
        if len(encoded[name]) > CEILING:
            raise ValueError(
                f"{name} is {len(encoded[name])} bytes; the ceiling is {CEILING}.",
            )
    out.mkdir(parents=True, exist_ok=True)
    for name, data in encoded.items():
        (out / name).write_bytes(data)


def check_sites(
    upstream_sites: dict[str, int],
    port_sites: dict[str, int],
) -> list[str]:
    """Report every required random call site that a run did not reach.

    Args:
      upstream_sites: The reference's draw counts.
      port_sites: The port's draw counts.

    Returns:
      report: One line per required site a side never drew from.

    """
    return [
        f"{side}: random call site {site} was never reached"
        for side, counts, required in (
            ("reference", upstream_sites, REQUIRED_UPSTREAM),
            ("port", port_sites, REQUIRED_PORT),
        )
        for site in required
        if not any(key.startswith(site) and count for key, count in counts.items())
    ]


def compare_wiring(trace: MainTrace) -> list[str]:
    """Check what the reference's ``main`` told its loss in every epoch.

    Args:
      trace: A run of the reference's ``main``.

    Returns:
      report: A line if any epoch's configuration is not zero-based epoch
        ``e`` of 200 with 100 classes.

    """
    expected = [
        {"num_classes": 100, "epoch": epoch, "total_epochs": 200}
        for epoch in range(200)
    ]
    return [] if trace.configs == expected else ["main: loss configuration differs"]


def compare_full_size(
    script: Upstream,
    trace: MainTrace,
    factory: Callable[[], VisionTrainLoop] = exp000,
) -> list[str]:
    """Compare ResNet-56's initial state and forward with the reference's.

    Args:
      script: The pristine reference script.
      trace: Its ``main``'s initial state, taken after its own seeding.
      factory: The port's experiment; ``exp000`` by default.

    Returns:
      report: One line per tensor that differs.

    """
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        ours = reference.full_step(factory).model
        theirs = script.ResNet(script.BasicBlock, [9, 9, 9], 100)
        theirs.load_state_dict(trace.state)
        report = compare_states(trace.state, ours.state_dict())
        media = reference.fixed_media(0)
        ours.train()
        theirs.train()
        report += mismatches(
            {"forward/train": reference.logits_of(theirs, media).detach()},
            {"forward/train": reference.logits_of(ours, media).detach()},
        )
        ours.eval()
        theirs.eval()
        report += mismatches(
            {"forward/eval": reference.logits_of(theirs, media).detach()},
            {"forward/eval": reference.logits_of(ours, media).detach()},
        )
    return report


# The two networks name their layers differently; the order is the same, and is itself
# part of what is compared.
def compare_states(
    theirs: dict[str, Tensor],
    ours: dict[str, Tensor],
) -> list[str]:
    """Compare two states tensor by tensor in registration order.

    Args:
      theirs: The reference network's state.
      ours: The port network's state.

    Returns:
      report: A line for a differing count, else one per differing tensor.

    """
    if len(theirs) != len(ours):
        return [f"state: {len(ours)} tensors against {len(theirs)}"]
    report: list[str] = []
    for (name, want), got in zip(theirs.items(), ours.values(), strict=True):
        report += mismatches({name: want}, {name: got})
    return report


class LockstepRun(Protocol):
    """A port's lockstep run: a variant's record, its states appended to ``states``."""

    def __call__(
        self,
        name: str,
        *,
        states: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Run variant ``name``."""
        raise NotImplementedError


def compare_lockstep(
    lockstep: Mapping[str, Lockstep],
    *,
    run: LockstepRun = reference.lockstep_record,
) -> list[str]:
    """Compare every parameter and buffer after each update of every variant.

    Args:
      lockstep: Each variant's upstream lockstep run.
      run: The port's lockstep run; this task's by default.

    Returns:
      report: One line per tensor that differs after any update.

    """
    report: list[str] = []
    for name, (_, theirs) in lockstep.items():
        ours: list[dict[str, Tensor]] = []
        _ = run(name, states=ours)
        for index, (want, got) in enumerate(zip(theirs, ours, strict=True), start=1):
            report += [
                f"lockstep/{name}/{index}: {line}" for line in compare_states(want, got)
            ]
    return report


def compare_loaders(
    theirs: tuple[Batches, Batches],
    ours: tuple[Batches, Batches],
) -> list[str]:
    """Compare every pixel and label the two loaders produce, pass by pass.

    Args:
      theirs: The reference's training and test passes.
      ours: The port's training and test passes.

    Returns:
      report: One line per tensor that differs.

    """
    report: list[str] = []
    for kind, want_passes, got_passes in zip(
        ("train", "test"),
        theirs,
        ours,
        strict=True,
    ):
        for index, (want, got) in enumerate(zip(want_passes, got_passes, strict=True)):
            report += [
                f"loader/{kind}/{index}: {line}"
                for line in mismatches(_flat(want), _flat(got))
            ]
    return report


def _flat(batches: Sequence[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Key every tensor of every batch by its position."""
    return {
        f"{index}/{key}": value
        for index, batch in enumerate(batches)
        for key, value in batch.items()
    }


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
        help="The testdata directory the goldens are written to.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
