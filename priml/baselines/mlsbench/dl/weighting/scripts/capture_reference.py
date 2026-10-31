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
of CIFAR-100's label layout (and of the real training split, when
``--cifar100`` names a root ``prepare_data`` has filled), the rate of all 200
epochs, the optimizer, ResNet-32's
initial state and forward, where the generator stands after building it, every
pixel the long-tailed loaders produce, and every parameter after each update of
a short run per weighting. At the ``resnet32-cifar10lt`` and
``vgg16bn-cifar100lt`` settings it compares, per weighting, the full network's
initial state, the setting's counts, weights and kept positions, the rates,
the optimizer, the generator's position and the forwards; every pixel of the
long-tailed CIFAR-10 loader; and for VGG every parameter after two updates of
the shallowest VGG the script's own table builds. Any mismatch is printed and the exit code is 1;
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

from priml.baselines.mlsbench.dl.vision import (
    reference as vision_reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.baselines.mlsbench.dl.weighting import reference, upstream
from priml.baselines.mlsbench.dl.weighting.data import long_tail_indices
from priml.baselines.mlsbench.dl.weighting.experiments import exp000
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches, rng_fingerprint


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.mlsbench.dl.vision.upstream import Lockstep


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
    capture_reference.write_records(
        flags.out,
        {
            "reference.pt": {
                k: v
                for k, v in captured.record.items()
                if not k.startswith("settings/")
            },
            "reference_settings.pt": {
                k: v for k, v in captured.record.items() if k.startswith("settings/")
            },
        },
    )
    print(f"0 mismatches; wrote {len(captured.record)} values to {flags.out}.")
    return 0


def capture(
    root: Path,
    scratch: Path,
    *,
    cifar100: Path | None,
) -> vision_reference.Captured:
    """Run both sides and compare them.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts and the loader split go to.
      cifar100: torchvision's CIFAR-100 root, holding ``cifar-100-python`` as
        ``prepare_data`` leaves it, whose training split is cut on both sides;
        ``None`` skips that comparison.

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
    report += capture_reference.compare_lockstep(
        lockstep,
        run=reference.lockstep_record,
    )
    report += capture_reference.compare_loaders(
        batches,
        reference.loader_batches(scratch),
    )
    if cifar100 is not None:
        report += compare_real(pristine, cifar100)
    settings, settings_report = capture_settings(scripts, scratch)
    return vision_reference.Captured(
        record=record | settings,
        sites={},
        report=report + settings_report,
    )


def capture_settings(
    scripts: dict[str, upstream.Script],
    scratch: Path,
) -> tuple[dict[str, Tensor], list[str]]:
    """Compare every weighting at the task's other settings with its script.

    Args:
      scripts: Each variant's script, keyed as :func:`upstream.variants`.
      scratch: Directory the CIFAR-10 loader's split goes to.

    Returns:
      record: The scripts' values, keyed as :func:`reference.settings_record`,
        and each loader setting's decoded indices.
      report: One line per mismatch.

    """
    record: dict[str, Tensor] = {}
    report: list[str] = []
    lockstep: dict[str, Lockstep] = {}
    port: dict[str, Tensor] = {}
    with vision_upstream.hidden_accelerators():
        for name, setting in reference.SETTINGS.items():
            traces, values, steps = upstream.upstream_setting(scripts, name)
            record |= values
            lockstep |= steps
            _, indices = reference.setting_counts_record(name)
            for variant, trace in traces.items():
                with (
                    vision_reference.seeded(vision_reference.SEED),
                    host_agnostic_numerics(),
                ):
                    ours = vision_reference.full_step(
                        setting.experiments[variant],
                    ).model
                    draws = rng_fingerprint()
                report += [
                    f"settings/{name}/{variant}/state: {line}"
                    for line in capture_reference.compare_states(
                        trace.main.state,
                        ours.state_dict(),
                    )
                ]
                report += [
                    f"settings/{name}/{variant}/{line}"
                    for line in mismatches(
                        {"indices": trace.indices, "full/draws": trace.main.draws},
                        {"indices": indices, "full/draws": draws},
                    )
                ]
            if setting.loader:
                uniform = setting.experiments["uniform"]
                theirs = upstream.upstream_batches(
                    scripts["uniform"],
                    setting,
                    imbalance_ratio=reference.SPLIT_RATIO_10,
                )
                ours_batches = reference.loader_batches(
                    scratch,
                    uniform,
                    imbalance_ratio=reference.SPLIT_RATIO_10,
                )
                key = f"settings/{name}/loader/index"
                dataset = uniform().dataset
                record[key] = reference.decode_indices(theirs[0], dataset)[
                    "loader/index"
                ]
                port[key] = reference.decode_indices(ours_batches[0], dataset)[
                    "loader/index"
                ]
                report += [
                    f"settings/{name}/{line}"
                    for line in capture_reference.compare_loaders(theirs, ours_batches)
                ]
    report += mismatches(record, reference.settings_record() | port)
    report += capture_reference.compare_lockstep(
        lockstep,
        run=reference.setting_lockstep_record,
    )
    return record, report


def compare_full_size(script: upstream.Script, trace: upstream.Trace) -> list[str]:
    """Compare ResNet-32's initial state and forward with the script's.

    Args:
      script: The pristine script.
      trace: Its ``main``'s initial state, taken after its own seeding.

    Returns:
      report: One line per tensor that differs.

    """
    state = trace.main.state
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        ours = vision_reference.full_step(exp000).model
        theirs = script.ResNet(script.BasicBlock, [5, 5, 5], reference.CLASSES)
        theirs.load_state_dict(state)
        report = capture_reference.compare_states(state, ours.state_dict())
        media = vision_reference.fixed_media(0)
        for mode in ("train", "eval"):
            ours.train(mode == "train")
            theirs.train(mode == "train")
            report += mismatches(
                {
                    f"forward/{mode}": vision_reference.model_logits(
                        theirs,
                        media,
                    ).detach(),
                },
                {
                    f"forward/{mode}": vision_reference.model_logits(
                        ours,
                        media,
                    ).detach(),
                },
            )
    return report


def compare_real(script: upstream.Script, cifar100: Path) -> list[str]:
    """Cut the real training split on both sides and compare positions and counts.

    Args:
      script: The pristine script.
      cifar100: torchvision's CIFAR-100 root, holding ``cifar-100-python``.

    Returns:
      report: One line per tensor that differs.

    """
    indices, counts, label = upstream.real_subset(script, cifar100)
    ours, our_counts = long_tail_indices(
        label,
        imbalance_ratio=reference.RATIO,
        num_classes=reference.CLASSES,
        seed=vision_reference.SEED,
    )
    return mismatches(
        {"real/indices": indices, "real/counts": counts},
        {"real/indices": ours, "real/counts": our_counts},
    )


class _Flags(Protocol):
    """Parsed command-line flags."""

    root: Path
    out: Path
    cifar100: Path | None


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
        default=None,
        help=(
            "torchvision's CIFAR-100 root (holding cifar-100-python, as "
            "prepare_data leaves it), to compare the real cut too."
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
