#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the cv-data-augmentation script's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``cv-data-augmentation`` script and each baseline's
edit of it from a clone of the benchmark, runs the port on the same inputs
under the same pins, and compares with exact equality: every pixel of each
variant's training pipeline over many images and of the fixed test pipeline,
every pixel and label each variant's loaders produce over two passes,
ResNet-56's initial state and forward, where the generator stands after
building it, the 200 learning rates, the optimizer, and every parameter after
each update of a short run per variant. It also checks that every policy
operation, and each signed one with both signs, was reached on both sides. Any mismatch is printed
and the exit code is 1; otherwise the values the default test replays are
written under ``--out``.

The script is imported from the clone, not vendored. Nothing runs on a GPU.

Examples:
  capture_reference.py --root /opt/scratch/mlsbench --out testdata

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import tempfile

import torch

from priml.baselines.mlsbench.dl.augmentation import reference, upstream
from priml.baselines.mlsbench.dl.vision import (
    reference as vision,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.data import normalize
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.mlsbench.dl.vision.upstream import Lockstep, Upstream

POLICIES: Final = ("randaugment", "trivialaugment")
"""The variants whose operations the census counts."""


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
    try:
        capture_reference.write_records(flags.out, {"reference.pt": captured.record})
    except ValueError as error:
        print(error)
        return 1
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
        trace = vision_upstream.trace_main(scripts["pad_crop_flip"])
        lockstep = {
            name: upstream.upstream_lockstep(script, name, trace)
            for name, script in scripts.items()
        }
        batches = {
            name: upstream.upstream_batches(script) for name, script in scripts.items()
        }
        report = capture_reference.compare_full_size(scripts["pad_crop_flip"], trace)
    pipelines, census, test = compare_pipelines(scripts)
    report += pipelines
    record = {
        "schedule/lr": torch.tensor(trace.rates, dtype=torch.float64),
        "optimizer/settings": torch.tensor(
            [trace.settings[name] for name in vision.HYPERPARAMETERS],
            dtype=torch.float64,
        ),
        "draws/exp000": trace.draws,
    } | test
    for name, script in scripts.items():
        train, _ = upstream.pipelines(script)
        media = upstream.upstream_pipeline(
            train,
            reference.pipeline_images(reference.GOLDEN_IMAGES),
            seed=vision.SEED,
        )
        record[f"augment/{name}/logits"] = reference.classified(media)
        record |= reference.stored(lockstep[name][0])
    report += mismatches(record, reference.port_record())
    report += compare_lockstep(lockstep)
    for name in scripts:
        report += [
            f"{name}: {line}"
            for line in capture_reference.compare_loaders(
                batches[name],
                reference.loader_batches(name, scratch),
            )
        ]
    report += census
    return vision.Captured(record=record, sites={}, report=report)


def compare_pipelines(
    scripts: dict[str, Upstream],
) -> tuple[list[str], list[str], dict[str, Tensor]]:
    """Compare every variant's training pipeline and the test pipeline, pixel by pixel.

    Args:
      scripts: Each variant of the script, keyed as :data:`reference.AUGMENTS`.

    Returns:
      report: One line per pipeline whose pixels differ.
      census: One line per policy operation and sign a side never reached, and
        one if the two sides applied different operations.
      test: ``augment/test/logits``, the classified test pipeline's output.

    """
    images = reference.pipeline_images(reference.PIPELINE_IMAGES)
    report: list[str] = []
    census: list[str] = []
    for name, script in scripts.items():
        train, _ = upstream.pipelines(script)
        with upstream.op_census() as theirs:
            want = upstream.upstream_pipeline(train, images, seed=vision.SEED)
        with reference.op_census() as ours:
            got = reference.augmented(name, images, seed=vision.SEED)
        report += mismatches({f"pipeline/{name}": want}, {f"pipeline/{name}": got})
        if name in POLICIES:
            census += [
                f"reference {name}: {k} never reached"
                for k in upstream.census_gaps(theirs)
            ]
            census += [
                f"port {name}: {k} never reached" for k in upstream.census_gaps(ours)
            ]
            census += [] if theirs == ours else [f"{name}: operation counts differ"]
    _, test = upstream.pipelines(scripts["pad_crop_flip"])
    config = reference.AUGMENTS["pad_crop_flip"]().dataset
    want = upstream.upstream_pipeline(test, images, seed=vision.SEED)
    got = normalize(
        images,
        mean=torch.tensor(config.mean),
        std=torch.tensor(config.std),
    )
    report += mismatches({"pipeline/test": want}, {"pipeline/test": got})
    return (
        report,
        census,
        {"augment/test/logits": reference.classified(want[: reference.GOLDEN_IMAGES])},
    )


def compare_lockstep(lockstep: dict[str, Lockstep]) -> list[str]:
    """Compare each update's batch and every parameter and buffer after it.

    Args:
      lockstep: Each variant's :func:`upstream.upstream_lockstep`.

    Returns:
      report: One line per tensor that differs.

    """
    run = _Recorded()
    report = capture_reference.compare_lockstep(lockstep, run=run)
    for name, (theirs, _) in lockstep.items():
        key = f"lockstep/{name}/media"
        report += mismatches({key: theirs[key]}, {key: run.record[key]})
    return report


class _Recorded:
    """The port's lockstep run, keeping every record it returned."""

    def __init__(self) -> None:
        self.record: dict[str, Tensor] = {}

    def __call__(
        self,
        name: str,
        *,
        states: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Run ``name``'s lockstep and keep its record."""
        record = reference.lockstep_record(name, states=states)
        self.record |= record
        return record


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
