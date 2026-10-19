#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the dl-regularization script's values and prove the port reproduces them.

Runs the pristine MLS-Bench ``dl-regularization`` script and each baseline's
edit of it from a clone of the benchmark, runs the port on the same inputs
under the same pins, and compares with exact equality: each regularizer's value
and gradients on a probe network, the setting's batch size, the rate of all 200
epochs, the optimizer, ResNet-56's initial state and forward, where the
generator stands after building it, every pixel the loaders produce, and every
parameter after each update of a short run per regularizer. ``main`` is traced
with the arguments of the setting's launch script. At the ``vgg16bn-cifar100``
and ``mobilenetv2-fmnist`` settings, ``main`` is traced once with each
setting's launch script and compared: the full network's initial state,
forwards and draws against each regularizer's fork, the rates (VGG's warmup
included), optimizer and batch size, each regularizer's value and gradients
on the setting's shallowest network, two updates of that network per
regularizer, and every Fashion-MNIST loader pixel. Any mismatch is printed and
the exit code is 1; otherwise the values the default test replays are written
under ``--out``.

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

from priml.baselines.mlsbench.dl.regularization import reference, upstream
from priml.baselines.mlsbench.dl.vision import (
    reference as vision_reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.mlsbench.dl.vision.upstream import Lockstep, Upstream


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
    capture_reference.write_records(
        flags.out,
        {
            "reference.pt": {
                k: v
                for k, v in captured.record.items()
                if not k.startswith("settings/")
            },
            "reference_settings.pt": {
                k: v
                for k, v in captured.record.items()
                # The script hardcodes 512- and 1280-wide layers that even the
                # shallowest network keeps; their gradients are compared in
                # ``capture`` and would fill the golden a thousand times over.
                if k.startswith("settings/") and not k.endswith("/parameters_grad")
            },
        },
    )
    print(f"0 mismatches; wrote {len(captured.record)} values to {flags.out}.")
    return 0


def capture(root: Path, scratch: Path) -> vision_reference.Captured:
    """Run both sides and compare them.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts and the loader split go to.

    Returns:
      captured: The script's values, keyed as the port's are.

    """
    report = check_pristine(root)
    scripts = upstream.variants(root, scratch)
    argv = upstream.setting_argv(root)
    with vision_upstream.hidden_accelerators():
        trace = upstream.trace_setting(scripts["zero"], argv)
        short = upstream.trace_setting(
            scripts["zero"],
            upstream.with_epochs(argv, vision_reference.EPOCHS),
        )
        lockstep = {
            name: upstream.upstream_lockstep(script, name, short)
            for name, script in scripts.items()
        }
        record = upstream.upstream_record(
            trace,
            upstream.upstream_regularizers(scripts),
            lockstep,
        )
        record["setting/batch_size"] = reference.setting_batch_size(argv)
        batches = vision_upstream.upstream_batches(scripts["zero"])
        report += capture_reference.compare_wiring(trace)
        report += capture_reference.compare_full_size(scripts["zero"], trace)
    port = reference.port_record()
    report += mismatches(record, port)
    report += capture_reference.compare_lockstep(
        lockstep,
        run=reference.lockstep_record,
    )
    report += capture_reference.compare_loaders(
        batches,
        vision_reference.loader_batches(scratch),
    )
    settings, settings_report = capture_settings(root, scripts, scratch)
    return vision_reference.Captured(
        record=record | settings,
        sites={},
        report=report + settings_report,
    )


def capture_settings(
    root: Path,
    scripts: dict[str, Upstream],
    scratch: Path,
) -> tuple[dict[str, Tensor], list[str]]:
    """Compare every regularizer at the task's other settings with its script.

    ``main`` is traced once per setting, with its launch arguments: what it
    builds before the first epoch does not depend on the regularizer, so one
    trace's network is every variant's.

    Args:
      root: Clone of the benchmark, for the settings' launch scripts.
      scripts: Each variant's script, keyed as :func:`upstream.variants`.
      scratch: Directory the Fashion-MNIST loader split goes to.

    Returns:
      record: The scripts' values, keyed as :func:`reference.settings_record`
        and the vision ``task_settings_record``; :func:`main` stores all but
        the parameter gradients.
      report: One line per mismatch.

    """
    record: dict[str, Tensor] = {}
    report: list[str] = []
    lockstep: dict[str, Lockstep] = {}
    pristine = scripts["zero"]
    with vision_upstream.hidden_accelerators():
        for setting, variants in reference.SETTING_REGULARIZERS.items():
            spec = vision_reference.SETTINGS[setting]
            argv = upstream.setting_argv(root, upstream.SETTING_SCRIPTS[setting])
            trace = upstream.trace_setting(pristine, argv, classes=spec.classes)
            report += [
                f"settings/{setting}/{line}"
                for line in capture_reference.compare_wiring(
                    trace,
                    classes=spec.classes,
                )
            ]
            record |= upstream.upstream_setting_recipe(trace, setting, argv)
            forward = vision_upstream.tiny_setting_record(pristine, spec.arch, setting)
            for name, factory in variants.items():
                key = f"{setting}/{name}"
                report += capture_reference.compare_setting(
                    trace,
                    setting,
                    factory,
                    key=key,
                )
                record |= {
                    k.replace(f"settings/{setting}/", f"settings/{key}/"): v
                    for k, v in forward.items()
                }
            tiny = upstream.tiny_network(pristine, spec.arch)
            report += [
                f"settings/{setting}/tiny/{line}"
                for line in capture_reference.compare_states(
                    tiny.state_dict(),
                    reference.tiny_network(setting).state_dict(),
                )
            ]
            record |= upstream.upstream_setting_regularizers(scripts, setting, tiny)
            short = upstream.trace_setting(
                pristine,
                upstream.with_epochs(argv, vision_reference.EPOCHS),
                classes=spec.classes,
            )
            for name, script in scripts.items():
                key = f"{setting}/{name}"
                lockstep[key] = upstream.upstream_lockstep(
                    script,
                    name,
                    short,
                    arch=spec.arch,
                    prefix=f"settings/{setting}/lockstep/{name}/",
                )
                record |= lockstep[key][0]
            if spec.dataset != "cifar100":
                batches = vision_upstream.upstream_batches(
                    pristine,
                    dataset=spec.dataset,
                )
                decoded = vision_reference.decode_batches(
                    batches[0],
                    config=spec.experiment().dataset,
                    source=spec.split()[0],
                )
                record |= {f"settings/{setting}/{k}": v for k, v in decoded.items()}
                report += [
                    f"settings/{setting}/{line}"
                    for line in capture_reference.compare_loaders(
                        batches,
                        vision_reference.setting_batches(scratch, setting),
                    )
                ]
    port = reference.settings_record() | vision_reference.task_settings_record(
        reference.SETTING_REGULARIZERS,
        scratch,
    )
    report += mismatches(record, port)
    report += capture_reference.compare_lockstep(
        lockstep,
        run=reference.setting_lockstep_record,
    )
    return record, report


def check_pristine(root: Path) -> list[str]:
    """Report a pristine regularizer that does not raise, as the port assumes it does.

    The port's ``exp000`` adds zero because the published body raises and its
    docstring names zero as "no regularization".

    Args:
      root: Clone of the benchmark.

    Returns:
      report: A line if the pristine regularizer returns, else nothing.

    """
    return [] if upstream.pristine_raises(root) else ["pristine: does not raise"]


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
