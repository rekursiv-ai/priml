#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the cv-pooling-aggregation script's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``cv-pooling-aggregation`` script and each
baseline's edit of it from a clone of the benchmark, runs the port on the same
inputs under the same pins, and compares with exact equality: every pool on
fixed features with its gradients, the learning rate of all 200 epochs, the
optimizer, the reported accuracy, ResNet-56's initial state and forward, where
the generator stands after building it with each pool, every pixel the loaders
produce, and every parameter after each update of a short run per pool. At
the ``vgg16bn-cifar100`` and ``mobilenetv2-fmnist`` settings it compares each
pool's fork: two updates of the setting's shallowest network with every
parameter after each, the full network's initial state and draws, the
shallowest network's forwards through the pool, and the Fashion-MNIST loaders' every pixel. Any
mismatch is
printed and the exit code is 1; otherwise the values the default
test replays are written under ``--out``.

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

from priml.baselines.mlsbench.dl.pooling import reference, upstream
from priml.baselines.mlsbench.dl.vision import (
    reference as vision,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
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
    record = captured.record
    capture_reference.write_records(
        flags.out,
        {
            "reference.pt": {
                k: v for k, v in record.items() if not k.startswith("settings/")
            },
            "reference_settings.pt": {
                k: v for k, v in record.items() if k.startswith("settings/")
            },
        },
    )
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
            name: upstream.upstream_lockstep(script, name, traces["avg"])
            for name, script in scripts.items()
        }
        record = upstream.upstream_record(scripts, traces, lockstep)
        batches = vision_upstream.upstream_batches(scripts["avg"])
        report = capture_reference.compare_full_size(scripts["avg"], traces["avg"])
        # Every setting's script runs the same SGD flags, so the resnet56
        # trace's optimizer settings are each setting's.
        setting_lockstep = {
            f"{setting}/{name}": upstream.upstream_lockstep(
                script,
                name,
                traces["avg"],
                arch=vision.SETTINGS[setting].arch,
                prefix=f"settings/{setting}/",
            )
            for setting in reference.SETTING_POOLS
            for name, script in scripts.items()
        }
    report += mismatches(record, reference.port_record())
    report += capture_reference.compare_lockstep(
        lockstep,
        run=reference.lockstep_record,
    )
    lockstep_values: dict[str, Tensor] = {}
    for values, _ in setting_lockstep.values():
        lockstep_values |= values
    report += mismatches(lockstep_values, reference.settings_lockstep_record())
    report += capture_reference.compare_lockstep(
        setting_lockstep,
        run=reference.setting_lockstep_record,
    )
    record |= lockstep_values
    report += capture_reference.compare_loaders(batches, vision.loader_batches(scratch))
    settings, settings_report = capture_reference.compare_task_settings(
        scripts,
        reference.SETTING_POOLS,
        scratch,
        pristine="avg",
    )
    report += settings_report
    return vision.Captured(record=record | settings, sites={}, report=report)


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
