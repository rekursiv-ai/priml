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
operation, and each signed one with both signs, was reached on both sides. At
the ``resnet20-cifar10`` and hidden ``mobilenetv2-fmnist`` settings it repeats,
per variant, the full network, the training pipeline, the loaders and the test
pipeline, Fashion-MNIST's on grayscale images. Any mismatch is printed
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
    from collections.abc import Callable

    from PIL import Image
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
                    if k.startswith("settings/")
                },
            },
        )
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
    settings, settings_report = capture_settings(scripts, scratch)
    report += settings_report
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
    return vision.Captured(record=record | settings, sites={}, report=report)


def capture_settings(
    scripts: dict[str, Upstream],
    scratch: Path,
) -> tuple[dict[str, Tensor], list[str]]:
    """Compare every variant at each other setting with its script.

    Args:
      scripts: Each variant of the script, keyed as :data:`reference.AUGMENTS`.
      scratch: Directory the loader splits go to.

    Returns:
      record: The script's values, keyed as :func:`reference.settings_record`.
      report: One line per mismatch: each full network, every pixel of each
        variant's training pipeline and loaders and of the test pipeline, and
        every policy operation either side never reached on the setting.

    """
    record, report = capture_reference.compare_task_settings(
        scripts,
        reference.SETTINGS,
        scratch,
        pristine="pad_crop_flip",
    )
    with vision_upstream.hidden_accelerators():
        for setting, variants in reference.SETTINGS.items():
            dataset = vision.SETTINGS[setting].dataset
            for name in variants:
                key = f"settings/{setting}/{name}"
                train, _ = upstream.pipelines(scripts[name], dataset=dataset)
                pixels, census = compare_pipeline(setting, name, train)
                report += pixels + census
                media = upstream.upstream_pipeline(
                    train,
                    reference.source_images(setting, reference.GOLDEN_IMAGES),
                    seed=vision.SEED,
                )
                record[f"{key}/augment"] = reference.classified(media)
                report += [
                    f"{key}/{line}"
                    for line in capture_reference.compare_loaders(
                        upstream.upstream_batches(scripts[name], dataset=dataset),
                        reference.setting_loader_batches(setting, name, scratch),
                    )
                ]
                report += mismatches(
                    {f"{key}/augment": record[f"{key}/augment"]},
                    reference.setting_pipeline_record(setting, name),
                )
            report += compare_test_pipeline(scripts["pad_crop_flip"], setting)
            known = vision.SETTINGS[setting]
            trace = vision_upstream.trace_main(
                scripts["pad_crop_flip"],
                arch=known.arch,
                dataset=known.dataset,
                classes=known.classes,
            )
            lockstep = {
                name: upstream.upstream_lockstep(
                    scripts[name],
                    name,
                    trace,
                    setting=setting,
                    images=reference.lockstep_images(setting),
                    key=reference.lockstep_key(name, setting=setting),
                )
                for name in variants
            }
            for values, _ in lockstep.values():
                record |= reference.stored(values)
            report += compare_lockstep(lockstep, setting=setting)
    return record, report


def compare_pipeline(
    setting: str,
    name: str,
    train: Callable[[Image.Image], Tensor],
) -> tuple[list[str], list[str]]:
    """Compare one variant's training pipeline at a setting, pixel by pixel.

    Args:
      setting: A key of :data:`reference.SETTINGS`.
      name: A key of :data:`reference.AUGMENTS`.
      train: The script's training pipeline at the setting.

    Returns:
      report: A line if the pixels differ.
      census: One line per policy operation and sign a side never reached, and
        one if the two sides applied different operations.

    """
    key = f"settings/{setting}/{name}/pipeline"
    images = reference.source_images(setting, reference.PIPELINE_IMAGES)
    with upstream.op_census() as theirs:
        want = upstream.upstream_pipeline(train, images, seed=vision.SEED)
    with reference.op_census() as ours:
        got = reference.augmented(
            name,
            reference.prepared(setting, images),
            seed=vision.SEED,
            factory=reference.SETTINGS[setting][name],
        )
    census: list[str] = []
    if name in POLICIES:
        census += [
            f"reference {key}: {k} never reached" for k in upstream.census_gaps(theirs)
        ]
        census += [f"port {key}: {k} never reached" for k in upstream.census_gaps(ours)]
        census += [] if theirs == ours else [f"{key}: operation counts differ"]
    return mismatches({key: want}, {key: got}), census


def compare_test_pipeline(script: Upstream, setting: str) -> list[str]:
    """Compare the script's fixed test pipeline at a setting with normalization.

    Args:
      script: The pristine script.
      setting: A key of :data:`reference.SETTINGS`.

    Returns:
      report: A line if the pixels differ.

    """
    _, test = upstream.pipelines(script, dataset=vision.SETTINGS[setting].dataset)
    images = reference.source_images(setting, reference.PIPELINE_IMAGES)
    config = reference.SETTINGS[setting]["pad_crop_flip"]().dataset
    key = f"settings/{setting}/test/pipeline"
    return mismatches(
        {key: upstream.upstream_pipeline(test, images, seed=vision.SEED)},
        {
            key: normalize(
                reference.prepared(setting, images),
                mean=torch.tensor(config.mean),
                std=torch.tensor(config.std),
            ),
        },
    )


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


def compare_lockstep(lockstep: dict[str, Lockstep], *, setting: str = "") -> list[str]:
    """Compare each update's batch and every parameter and buffer after it.

    Args:
      lockstep: Each variant's :func:`upstream.upstream_lockstep`.
      setting: A key of :data:`reference.SETTINGS`, or empty for ``exp000``'s.

    Returns:
      report: One line per tensor that differs, prefixed with the setting.

    """
    run = _Recorded(setting)
    report = capture_reference.compare_lockstep(lockstep, run=run)
    for name, (theirs, _) in lockstep.items():
        key = f"{reference.lockstep_key(name, setting=setting)}/media"
        report += mismatches({key: theirs[key]}, {key: run.record[key]})
    prefix = f"settings/{setting}/" if setting else ""
    return [f"{prefix}{line}" for line in report]


class _Recorded:
    """The port's lockstep run at one setting, keeping every record it returned."""

    def __init__(self, setting: str) -> None:
        self.setting = setting
        self.record: dict[str, Tensor] = {}

    def __call__(
        self,
        name: str,
        *,
        states: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Run ``name``'s lockstep and keep its record."""
        record = reference.lockstep_record(name, states=states, setting=self.setting)
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
