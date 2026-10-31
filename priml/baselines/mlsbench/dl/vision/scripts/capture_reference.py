#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Capture the reference's values and prove the port reproduces them.

Runs the pristine MLS-Bench vision code from a clone of the benchmark, runs the
shared port on the same inputs under the same pins, and compares every
component with exact equality: the initial state and forward of the full
ResNet-56, what ``main`` tells its loss each epoch, the learning rates of all
200 epochs, the optimizer, the reported accuracy, the image pipelines and every
pixel the loaders produce. It also counts the random draws on both sides. Any
mismatch is printed and the exit code is 1; otherwise the values the default
test replays are written under ``--out``. Each task package's own capture
proves its slot and the updates.

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

from priml.baselines.mlsbench.dl.vision import reference, upstream
from priml.baselines.mlsbench.dl.vision.experiments import exp000
from priml.baselines.mlsbench.dl.vision.upstream import (
    PRISTINE,
    Batches,
    Lockstep,
    MainTrace,
    Upstream,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches, pack, rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torch import Tensor, nn

    import torch

    from priml.baselines.mlsbench.dl.vision.reference import VisionLoop
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


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
      scratch: Directory the loader splits go to.

    Returns:
      captured: The reference's values, keyed as the port's are.

    """
    pristine = upstream.load_script(root / PRISTINE)
    with upstream.hidden_accelerators():
        trace = upstream.trace_main(pristine)
        batches = upstream.upstream_batches(pristine)
        record = upstream.upstream_record(pristine, trace, batches[0])
        sites = upstream.upstream_sites(pristine)
        fmnist_sites = upstream.upstream_fmnist_sites(pristine)
        report = compare_wiring(trace) + compare_full_size(pristine, trace)
        report += compare_tiny_settings(pristine)
        settings, settings_report = capture_settings(pristine, scratch)
    port = reference.port_record(scratch)
    port_sites = reference.draw_sites(scratch)
    port_fmnist_sites = reference.fmnist_draw_sites(scratch)
    report += mismatches(record, port)
    report += compare_loaders(batches, reference.loader_batches(scratch))
    report += check_sites(sites, port_sites)
    report += [
        f"fmnist {line}" for line in check_sites(fmnist_sites, port_fmnist_sites)
    ]
    report += settings_report
    return reference.Captured(
        record=record | settings,
        sites={
            "reference": sites,
            "port": port_sites,
            "reference-fmnist": fmnist_sites,
            "port-fmnist": port_fmnist_sites,
        },
        report=report,
    )


def capture_settings(
    script: Upstream,
    scratch: Path,
) -> tuple[dict[str, Tensor], list[str]]:
    """Compare every one of :data:`reference.SETTINGS` with the reference.

    Args:
      script: The pristine reference script.
      scratch: Directory the loader splits go to.

    Returns:
      record: The reference's values, keyed as :func:`reference.settings_record`.
      report: One line per mismatch: the full-size state of each network, its
        forwards and draws, the loss wiring, and every loader pixel.

    """
    traces, record = upstream.upstream_settings(script)
    report = mismatches(record, reference.settings_record(scratch))
    for name in reference.CAPTURED:
        setting = reference.SETTINGS[name]
        trace = traces[name]
        report += compare_wiring(trace, classes=setting.classes)
        report += compare_setting(trace, name)
        report += [
            f"settings/{name}/{line}"
            for line in compare_loaders(
                upstream.upstream_batches(script, dataset=setting.dataset),
                reference.setting_batches(scratch, name),
            )
        ]
    return record, report


def compare_tiny_settings(script: Upstream) -> list[str]:
    """Compare each setting's shallowest network, built and run, with the reference's.

    Args:
      script: The pristine reference script.

    Returns:
      report: One line per differing tensor of the state, in registration
        order, the generator after the build, or a forward.

    """
    report: list[str] = []
    media = reference.fixed_media(reference.STEPS)
    for name in ("vgg16bn-cifar100", "mobilenetv2-fmnist"):
        setting = reference.SETTINGS[name]
        with reference.seeded(reference.SEED), host_agnostic_numerics():
            theirs = upstream.tiny_model(script, setting.arch, reference.CLASSES)
            script.initialize_weights(theirs)
            want = {"draws": rng_fingerprint()}
        with reference.seeded(reference.SEED), host_agnostic_numerics():
            ours = reference.tiny_step(
                setting.experiment,
                blocks=(1, 1, 1),
                classes=reference.CLASSES,
                epochs=reference.EPOCHS,
            ).model
            got = {"draws": rng_fingerprint()}
        lines = compare_states(theirs.state_dict(), ours.state_dict())
        for mode in ("train", "eval"):
            for model, record in ((theirs, want), (ours, got)):
                model.train(mode == "train")
                with reference.seeded(reference.SEED), host_agnostic_numerics():
                    record[mode] = reference.model_logits(model, media).detach()
        lines += mismatches(want, got)
        report += [f"tiny/{name}: {line}" for line in lines]
    return report


def compare_task_settings(
    scripts: Mapping[str, Upstream],
    experiments: Mapping[str, Mapping[str, Callable[[], VisionLoop]]],
    scratch: Path,
    *,
    pristine: str,
    build: Callable[[Upstream, str], nn.Module] = upstream.initialized_tiny,
) -> tuple[dict[str, Tensor], list[str]]:
    """Compare every variant of a task at each setting with the variant's script.

    Args:
      scripts: Each variant's script, pristine or edited.
      experiments: As :func:`reference.task_settings_record`; the variant
        names key ``scripts``.
      scratch: Directory the loader splits go to.
      pristine: The variant whose script runs the loaders.
      build: As :func:`upstream.upstream_setting`.

    Returns:
      record: The scripts' values, keyed as :func:`reference.task_settings_record`.
      report: One line per mismatch: each full network's state, its forwards
        and draws, and every pixel of each non-CIFAR-100 setting's loaders.

    """
    record: dict[str, Tensor] = {}
    report: list[str] = []
    with upstream.hidden_accelerators():
        for setting, variants in experiments.items():
            for variant, factory in variants.items():
                key = f"{setting}/{variant}"
                trace, network = upstream.upstream_setting(
                    scripts[variant],
                    setting,
                    key=key,
                    build=build,
                )
                record |= network
                report += compare_setting(trace, setting, factory, key=key)
            spec = reference.SETTINGS[setting]
            if spec.dataset == "cifar100":
                continue
            batches = upstream.upstream_batches(
                scripts[pristine],
                dataset=spec.dataset,
            )
            decoded = reference.decode_batches(
                batches[0],
                config=spec.experiment().dataset,
                source=spec.split()[0],
            )
            record |= {f"settings/{setting}/{k}": v for k, v in decoded.items()}
            report += [
                f"settings/{setting}/{line}"
                for line in compare_loaders(
                    batches,
                    reference.setting_batches(scratch, setting),
                )
            ]
    report += mismatches(record, reference.task_settings_record(experiments, scratch))
    return record, report


def compare_setting(
    trace: MainTrace,
    name: str,
    factory: Callable[[], VisionLoop] | None = None,
    *,
    key: str | None = None,
) -> list[str]:
    """Compare a setting's full network, as ``main`` initialized it, with the port's.

    The full-size network is compared here, live, and never stored: its
    golden would be a slow test. The stored values are the shallowest network's.

    Args:
      trace: The reference's ``main`` at the setting, after its own seeding.
      name: A key of :data:`reference.SETTINGS`.
      factory: The port's experiment; the setting's pristine one by default.
      key: Names the report's lines; ``name`` by default.

    Returns:
      report: One line per tensor that differs, in registration order, then
        one if the generator stands elsewhere after the build.

    """
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        ours = reference.full_step(factory or reference.SETTINGS[name].experiment).model
        draws = rng_fingerprint()
    return [
        f"settings/{key or name}/state: {line}"
        for line in compare_states(trace.state, ours.state_dict())
    ] + [
        f"settings/{key or name}/full/{line}"
        for line in mismatches({"draws": trace.draws}, {"draws": draws})
    ]


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
                k: v
                for k, v in record.items()
                if not k.startswith(("transform/", "settings/"))
            },
            "reference_train_pipeline.pt": {
                "transform/train": record["transform/train"],
            },
            "reference_test_pipeline.pt": {"transform/test": record["transform/test"]},
            "reference_settings.pt": {
                k: v for k, v in record.items() if k.startswith("settings/")
            },
            "reference_fmnist_train_pipeline.pt": {
                "transform/fmnist/train": record["transform/fmnist/train"],
            },
            "reference_fmnist_test_pipeline.pt": {
                "transform/fmnist/test": record["transform/fmnist/test"],
            },
        },
    )
    write_sites(out, captured.sites)


def shared_scores(
    record: Mapping[str, Tensor],
    *,
    pairs: Mapping[str, str],
) -> list[str]:
    """Report each pair of score sets a rank AUC is taken over that share a value.

    An AUC ranked by NumPy's default sort orders a value in both sets by the
    host's SIMD kernel, so a golden of it holds on some hosts only.

    Args:
      record: A capture's values.
      pairs: Key of each member score set to the key of its non-member set.

    Returns:
      report: One line per pair with a shared value.

    """
    report: list[str] = []
    for member, nonmember in pairs.items():
        shared = set(record[member].tolist()) & set(record[nonmember].tolist())
        if shared:
            report.append(
                f"{member}: {len(shared)} scores also in {nonmember}; "
                "their order is the host's",
            )
    return report


def write_sites(out: Path, sites: Mapping[str, Mapping[str, int]]) -> None:
    """Write each side's draw counts to ``reference_sites.txt``, one site a line.

    Args:
      out: The ``testdata`` directory.
      sites: Side (``reference``, ``port``, ...) to its draw counts.

    """
    lines = [
        f"{side} {key} {count}"
        for side, counts in sites.items()
        for key, count in sorted(counts.items())
    ]
    (out / "reference_sites.txt").write_text("\n".join(lines) + "\n")


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


def read_sites(directory: Path) -> dict[str, dict[str, int]]:
    """Read the ``reference_sites.txt`` :func:`write_sites` wrote into ``directory``.

    Args:
      directory: The ``testdata`` directory.

    Returns:
      sites: Side to its draw counts.

    """
    sites: dict[str, dict[str, int]] = {}
    for line in (directory / "reference_sites.txt").read_text().splitlines():
        side, key, count = line.split()
        sites.setdefault(side, {})[key] = int(count)
    return sites


def assert_sites_reached(
    directory: Path,
    *,
    port: Callable[[], dict[str, int]],
    check: Callable[[dict[str, int], dict[str, int]], list[str]] = check_sites,
) -> None:
    """Require both stored sides to reach every site, and the port to replay its side.

    Args:
      directory: The ``testdata`` directory holding ``reference_sites.txt``.
      port: Runs the port and returns its draw counts.
      check: The task's ``check_sites``, over the reference's and the port's.

    Raises:
      AssertionError: A stored side misses a required site, or the port's
        draws differ from its stored side.

    """
    sides = read_sites(directory)
    report = check(sides["reference"], sides["port"])
    if report:
        raise AssertionError("\n".join(report))
    drawn = port()
    if drawn != sides["port"]:
        raise AssertionError(
            f"the port replays {drawn}, not its stored {sides['port']}",
        )


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


def compare_wiring(trace: MainTrace, *, classes: int = 100) -> list[str]:
    """Check what the reference's ``main`` told its loss in every epoch.

    Args:
      trace: A run of the reference's ``main``.
      classes: The setting's class count.

    Returns:
      report: A line if any epoch's configuration is not zero-based epoch
        ``e`` of 200 with ``classes`` classes.

    """
    expected = [
        {"num_classes": classes, "epoch": epoch, "total_epochs": 200}
        for epoch in range(200)
    ]
    return [] if trace.configs == expected else ["main: loss configuration differs"]


def compare_full_size(
    script: Upstream,
    trace: MainTrace,
    factory: Callable[[], VisionLoop] = exp000,
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
            {"forward/train": reference.model_logits(theirs, media).detach()},
            {"forward/train": reference.model_logits(ours, media).detach()},
        )
        ours.eval()
        theirs.eval()
        report += mismatches(
            {"forward/eval": reference.model_logits(theirs, media).detach()},
            {"forward/eval": reference.model_logits(ours, media).detach()},
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
    run: LockstepRun,
) -> list[str]:
    """Compare every parameter and buffer after each update of every variant.

    Args:
      lockstep: Each variant's upstream lockstep run.
      run: The task's port-side lockstep run.

    Returns:
      report: One line per variant whose runs made another number of
        updates, and per tensor that differs after any update both made.

    """
    report: list[str] = []
    for name, (_, theirs) in lockstep.items():
        ours: list[dict[str, Tensor]] = []
        _ = run(name, states=ours)
        if len(ours) != len(theirs):
            report.append(
                f"lockstep/{name}: {len(ours)} updates against {len(theirs)}",
            )
        # Not strict: a count mismatch is reported above, and the updates both
        # runs made are still compared.
        for index, (want, got) in enumerate(zip(theirs, ours, strict=False), start=1):
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
