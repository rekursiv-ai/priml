#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Capture the security-poison-robust-learning driver's values and prove the port reproduces them.

Runs the pristine MLS-Bench driver with its pristine loss and with each
baseline's edit of it from a clone of the benchmark, runs the port on the same
inputs under the same pins, and compares with exact equality: every loss and
its gradient on fixed logits, the label flip of a fixed split and of CIFAR-10's
50,000 and Fashion-MNIST's 60,000 training labels with every poisoned item,
both evaluations on fixed logits, the rate of all 100 epochs and the
optimizer, each setting's initial state, generator position and forward, and
per loss a whole run of ``main`` on the shallowest network with its four
loader workers: every loss, logit, label, epoch and rate, every parameter
after each update, every pixel and logit of both evaluations, the published
metrics, and Python's and NumPy's generators left untouched. It counts the
random draws on both sides. Any mismatch is printed and the exit code is 1;
otherwise the values the default test replays are written under ``--out``.

The driver is imported from the clone, not vendored. Nothing runs on a GPU.

Examples:
  capture_reference.py --root /opt/scratch/mlsbench --out testdata

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import argparse
import tempfile

import torch

from priml.baselines.mlsbench.dl.vision import (
    reference as vision_reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.draws import recorded_draws
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.baselines.mlsbench.tl.poison_robust_learning import (
    reference,
    upstream,
)
from priml.baselines.mlsbench.tl.poison_robust_learning.data import (
    flipped_labels,
)
from priml.baselines.mlsbench.tl.poison_robust_learning.experiments import (
    exp000,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from priml.baselines.mlsbench.tl.poison_robust_learning.experiments import (
        PoisonTrainLoop,
    )


TRAINING_IMAGES: Final = {"cifar10": 50_000, "cifar100": 50_000, "fmnist": 60_000}
"""Images in each dataset's training split, which the full-size flip runs over."""

REQUIRED_UPSTREAM: Final = (
    "uniform_@nn/init.py",
    "normal_@nn/init.py",
    "random_@data/dataloader.py",
    "random_@data/sampler.py",
    "randperm@data/sampler.py",
    "randperm@poison/run_poison_robust.py",
)
"""Random call sites of the driver that a lockstep run must reach in this process."""

REQUIRED_PORT: Final = (
    "uniform_@nn/init.py",
    "normal_@nn/init.py",
    "random_@vision/data.py",
    "randperm@vision/data.py",
    "randint@vision/data.py",
    "rand@vision/data.py",
    "random_@poison_robust_learning/data.py",
    "randperm@poison_robust_learning/data.py",
)
"""Random call sites of the port that a lockstep run must reach."""


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
    capture_reference.write_records(flags.out, {"reference.pt": captured.record})
    capture_reference.write_sites(flags.out, captured.sites)
    print(f"0 mismatches; wrote {len(captured.record)} values to {flags.out}.")
    return 0


def capture(root: Path, scratch: Path) -> vision_reference.Captured:
    """Run both sides and compare them.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited losses and the lockstep split go to.

    Returns:
      captured: The driver's values, keyed as the port's are.

    """
    scripts = upstream.variants(root, scratch)
    pristine = scripts["pristine"]
    recipe = upstream.trace_main(
        pristine,
        upstream.recipe_argv("resnet20-cifar10"),
        build=upstream.linear_model,
        split=vision_reference.synthetic_split(),
        serial=True,
    )
    record = upstream_record(scripts, recipe)
    report = compare_recipe(recipe)
    for key, (setting, factory) in reference.SETTINGS.items():
        initial = upstream.trace_main(
            pristine,
            upstream.recipe_argv(setting),
            build=upstream.full_model,
            split=_labels_only(TRAINING_IMAGES[upstream.SETTINGS[setting][1]]),
            serial=True,
            stop=True,
        )
        record |= vision_upstream.tiny_setting_record(
            cast(vision_upstream.Upstream, pristine),
            vision_reference.SETTINGS[setting].arch,
            key,
        )
        report += compare_initial(initial, key=key, factory=factory)
    lockstep = {
        name: upstream.upstream_lockstep(script, name)
        for name, script in scripts.items()
    }
    for values, _ in lockstep.values():
        record |= values
    report += mismatches(record, reference.port_record(scratch))
    report += capture_reference.compare_lockstep(lockstep, run=PortLockstep(scratch))
    report += compare_pristine_run(pristine, scratch)
    sites = {"reference": upstream_sites(pristine), "port": port_sites(scratch)}
    report += check_sites(sites["reference"], sites["port"])
    return vision_reference.Captured(record=record, sites=sites, report=report)


def upstream_record(
    scripts: dict[str, upstream.Script],
    recipe: upstream.Trace,
) -> dict[str, Tensor]:
    """Return the driver's value of every fixed-input quantity the default test replays.

    Args:
      scripts: Each variant, keyed as :func:`upstream.variants`.
      recipe: The pristine ``main`` at the recipe's own arguments.

    Returns:
      record: Keyed as the port's records, but for the networks and the
        lockstep runs.

    """
    record: dict[str, Tensor] = {}
    for name, script in scripts.items():
        for case, (logits, label) in reference.loss_cases().items():
            value, grad = upstream.loss_value(script, logits, label, epoch=0)
            record[f"loss/{name}/{case}/value"] = value
            record[f"loss/{name}/{case}/grad"] = grad
    pristine = scripts["pristine"]
    dataset = upstream.poisoned(
        pristine,
        reference.flip_labels(),
        classes=reference.FLIP_CLASSES,
        fraction=reference.LOCKSTEP_FRACTION,
    )
    record["flip/labels"] = dataset.labels
    record["flip/index"] = dataset.poison_idx
    record["flip/mask"] = dataset.is_poison
    record["flip/items"] = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    record |= upstream_evaluations(pristine)
    record["recipe/lr"] = torch.tensor(recipe.rates, dtype=torch.float64)
    record["recipe/optimizer"] = torch.tensor(
        [recipe.settings[name] for name in vision_reference.HYPERPARAMETERS],
        dtype=torch.float64,
    )
    return record


def upstream_evaluations(script: upstream.Script) -> dict[str, Tensor]:
    """Score the port's fixed evaluation logits with the driver's own evaluations.

    Each image of the poisoned split is its own index, and the network a lookup
    of that index's fixed logits, so the driver's loader, batching and masking
    run unchanged.

    Args:
      script: The pristine driver.

    Returns:
      record: Keyed as :func:`reference.evaluate_record`.

    """
    logits, clean = reference.evaluate_logits()
    model = _Lookup(logits)
    batch = reference.LOCKSTEP_BATCH
    index = torch.arange(len(clean))
    test = [
        (index[start : start + batch], clean[start : start + batch])
        for start in range(0, len(clean), batch)
    ]
    dataset = upstream.poisoned(
        script,
        clean,
        classes=reference.FLIP_CLASSES,
        fraction=reference.LOCKSTEP_FRACTION,
    )
    cpu = torch.device("cpu")
    with (
        host_agnostic_numerics(),
        vision_reference.script_workers(),
        vision_upstream.forked_workers(),
        vision_upstream.hidden_accelerators(),
    ):
        test_acc = script.evaluate_test_acc(model, test, cpu)
        poison_fit = script.evaluate_poison_fit(
            model,
            dataset,
            cpu,
            reference.LOCKSTEP_BATCH,
        )
    return {
        "evaluate/test_acc": torch.tensor(test_acc, dtype=torch.float64),
        "evaluate/poison_fit": torch.tensor(poison_fit, dtype=torch.float64),
    }


def compare_recipe(trace: upstream.Trace) -> list[str]:
    """Check what the 100-epoch run told the loss, its horizon and the loaders it built.

    Over the seven-image split at batch 128 every epoch is one update.

    Args:
      trace: The pristine ``main`` at the recipe's own arguments.

    Returns:
      report: A line per difference from the port's recipe and loaders.

    """
    report: list[str] = []
    if trace.epochs != list(range(reference.EPOCHS)):
        report.append("recipe: the loss was told other epochs")
    if trace.t_max != reference.EPOCHS:
        report.append(f"recipe: the schedule's horizon is {trace.t_max}")
    config = exp000().dataset
    shuffles = [loader["shuffle"] for loader in trace.loaders]
    if shuffles != [True, False, False]:
        report.append(f"recipe: loaders shuffle as {shuffles}")
    workers = [loader["num_workers"] for loader in trace.loaders]
    if workers != [config.num_streams] * 3:
        report.append(f"recipe: loaders run {workers} workers")
    sizes = {loader["batch_size"] for loader in trace.loaders}
    if sizes != {config.batch_size} or config.eval_batch_size != config.batch_size:
        report.append(f"recipe: loaders batch {sizes}")
    return report


def compare_initial(
    trace: upstream.Trace,
    *,
    key: str,
    factory: Callable[[], PoisonTrainLoop],
) -> list[str]:
    """Compare a setting's full-size label flip and its network's initial state.

    Args:
      trace: The pristine ``main`` at the setting over a labels-only split of
        the dataset's size, up to its first update.
      key: Names the report's lines: the setting's pristine experiment.
      factory: That experiment.

    Returns:
      report: One line per tensor that differs.

    """
    config = factory().copy_tree().finalize()
    (dataset,) = trace.poisoned
    clean = _labels_only(len(dataset))[1]
    label, index = flipped_labels(
        clean,
        num_classes=config.dataset.num_classes,
        fraction=config.dataset.poison_fraction,
        seed=config.dataset.poison_seed,
    )
    report = mismatches(
        {
            f"flip/{key}/labels": dataset.labels,
            f"flip/{key}/index": dataset.poison_idx,
        },
        {f"flip/{key}/labels": label, f"flip/{key}/index": index},
    )
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        ours = vision_reference.full_step(factory).model
    return report + [
        f"settings/{key}/state: {line}"
        for line in capture_reference.compare_states(trace.initial, ours.state_dict())
    ]


def compare_pristine_run(script: upstream.Script, scratch: Path) -> list[str]:
    """Check the pristine lockstep run's evaluation pixels and its other generators.

    Args:
      script: The pristine driver.
      scratch: Directory the port's split is written to.

    Returns:
      report: One line per evaluation batch whose pixels differ, and per
        generator of Python's or NumPy's the driver drew from in this process.

    """
    trace = upstream.trace_main(
        script,
        upstream.lockstep_argv(),
        build=upstream.tiny_model,
        split=vision_reference.synthetic_split(),
    )
    ours: list[Tensor] = []
    _ = reference.lockstep_record("pristine", scratch, media=ours)
    report = mismatches(
        {f"evaluate/media/{i}": media for i, (media, _) in enumerate(trace.evaluated)},
        {f"evaluate/media/{i}": media for i, media in enumerate(ours)},
    )
    seeded = upstream.seeded_generator_states()
    after = trace.generators
    if after is None:
        raise ValueError("Expected after is not None.")
    if after.python != seeded.python:
        report.append("generators: python random was drawn from")
    if after.numpy != seeded.numpy:
        report.append("generators: numpy random was drawn from")
    return report


def upstream_sites(script: upstream.Script) -> dict[str, int]:
    """Count the driver's random draws over one lockstep run, loaders in this process.

    Args:
      script: The pristine driver.

    Returns:
      counts: As :func:`~priml.baselines.mlsbench.dl.vision.draws.recorded_draws`.

    """
    with recorded_draws() as counts:
        _ = upstream.trace_main(
            script,
            upstream.lockstep_argv(),
            build=upstream.tiny_model,
            split=vision_reference.synthetic_split(),
            serial=True,
        )
    return counts


def port_sites(scratch: Path) -> dict[str, int]:
    """Count the port's random draws over one lockstep run.

    Args:
      scratch: Directory the split is written to.

    Returns:
      counts: As :func:`~priml.baselines.mlsbench.dl.vision.draws.recorded_draws`.

    """
    with recorded_draws() as counts:
        _ = reference.lockstep_record("pristine", scratch)
    return counts


def check_sites(
    upstream_sites: dict[str, int],
    port_sites: dict[str, int],
) -> list[str]:
    """Report every required random call site a lockstep run did not reach.

    Args:
      upstream_sites: The driver's draw counts.
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


class PortLockstep:
    """The port's lockstep run, in the shape ``compare_lockstep`` calls.

    Args:
      scratch: Directory the lockstep split is written to.

    """

    def __init__(self, scratch: Path) -> None:
        self.scratch = scratch

    def __call__(
        self,
        name: str,
        *,
        states: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        """Run variant ``name``, appending its states."""
        return reference.lockstep_record(name, self.scratch, states=states)


def _labels_only(count: int) -> tuple[Tensor, Tensor]:
    """Return ``count`` one-pixel images and labels cycling through ten classes."""
    return torch.zeros(count, 1, 1, 1, dtype=torch.uint8), torch.arange(count) * 7 % 10


class _Lookup(torch.nn.Module):
    """Returns the fixed logits of each image, whose pixels are its own index."""

    def __init__(self, logits: Tensor) -> None:
        super().__init__()
        self.logits = logits

    @override
    def forward(self, images: Tensor) -> Tensor:
        """Return the rows of ``logits`` the images name."""
        return self.logits[images.reshape(len(images), -1)[:, 0].long()]


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
