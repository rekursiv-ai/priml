#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Capture the cv-classification-loss script's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``cv-classification-loss`` script and each
baseline's edit of it from a clone of the benchmark, runs the port on the same
inputs under the same pins, and compares with exact equality: every loss on
fixed logits with its gradient, and per loss two updates of the shallowest
network, their losses, logits and probes, what the loss was told, and every
parameter after each. At the ``vgg16bn-cifar100`` and ``mobilenetv2-fmnist``
settings it also compares each loss's fork: the loss and its gradient at the
setting's class count, two updates of the setting's shallowest network with
every parameter after each, the full network's initial state, forwards and
draws, and the Fashion-MNIST loaders' every pixel. Any mismatch is
printed and the exit code is 1; otherwise the values the default test replays
are written under ``--out``.
The network, data, schedule and optimizer are the shared recipe's, proven by
the vision package's capture.

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

from priml.baselines.mlsbench.dl.classification_loss import reference, upstream
from priml.baselines.mlsbench.dl.vision import (
    reference as vision_reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
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


def capture(root: Path, scratch: Path) -> vision_reference.Captured:
    """Run both sides and compare them.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts go to.

    Returns:
      captured: The script's values, keyed as the port's are.

    """
    scripts = upstream.variants(root, scratch)
    with vision_upstream.hidden_accelerators():
        trace = vision_upstream.trace_main(scripts["cross_entropy"])
        lockstep = {
            name: upstream.upstream_lockstep(script, name, trace)
            for name, script in scripts.items()
        }
        record = upstream.upstream_record(scripts, lockstep)
        losses = upstream.upstream_setting_losses(scripts, reference.SETTING_LOSSES)
        # Every setting's script runs the same SGD flags, so the resnet56
        # trace's optimizer settings are each setting's.
        setting_lockstep = {
            f"{setting}/{name}": upstream.upstream_lockstep(
                script,
                name,
                trace,
                arch=vision_reference.SETTINGS[setting].arch,
                prefix=f"settings/{setting}/",
            )
            for setting in reference.SETTING_LOSSES
            for name, script in scripts.items()
        }
    for values, _ in setting_lockstep.values():
        losses |= values
    report = mismatches(record, reference.port_record())
    report += capture_reference.compare_lockstep(
        lockstep,
        run=reference.lockstep_record,
    )
    report += mismatches(losses, reference.settings_port_record())
    report += capture_reference.compare_lockstep(
        setting_lockstep,
        run=reference.setting_lockstep_record,
    )
    settings, settings_report = capture_reference.compare_task_settings(
        scripts,
        reference.SETTING_LOSSES,
        scratch,
        pristine="cross_entropy",
    )
    report += settings_report
    return vision_reference.Captured(
        record=record | losses | settings,
        sites={},
        report=report,
    )


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
