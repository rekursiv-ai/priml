#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 -P "$0" "$@"
Capture the security-membership-inference-defense driver's values and prove the port reproduces them.

Runs the pristine MLS-Bench driver with its pristine defense and with each
baseline's edit of it from a clone of the benchmark, runs the port on the same
inputs under the same pins, and compares with exact equality: every defense
and its gradient on fixed logits in both RelaxLoss phases, the attack's scores,
accuracies and AUC on fixed logits, both training pipelines (with and without
``--augment``) and the test pipeline, the rate of all 300 epochs and the
optimizer, the member split of CIFAR-10's 50,000 images, ResNet-20's initial
state, generator position and forward, and per defense a whole run of
``main`` on the shallowest network: every loss, logit, label, epoch and rate,
every parameter after each update, and the attack's outputs. It counts the
random draws on both sides. Any mismatch is printed and the exit code is 1;
otherwise the values the default test replays are written under ``--out``.

The driver is imported from the clone, not vendored. Nothing runs on a GPU.

Examples:
  capture_reference.py --root /opt/scratch/mlsbench --out testdata

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import argparse
import tempfile

from priml.baselines.mlsbench.dl.vision import (
    reference as vision_reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.draws import recorded_draws
from priml.baselines.mlsbench.dl.vision.scripts import capture_reference
from priml.baselines.mlsbench.tl.membership_inference_defense import (
    reference,
    upstream,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.data import (
    member_split,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.experiments import (
    exp000,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    import numpy as np
    import torch

    from priml.baselines.mlsbench.tl.membership_inference_defense.experiments import (
        MembershipTrainLoop,
    )
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


TRAINING_IMAGES: Final = 50_000
"""Images in CIFAR-10's training split, which the member split halves."""

REQUIRED_UPSTREAM: Final = (
    "uniform_@nn/init.py",
    "normal_@nn/init.py",
    "random_@data/dataloader.py",
    "random_@data/sampler.py",
    "randperm@data/sampler.py",
)
"""Random call sites of the driver that a lockstep run must reach."""

REQUIRED_PORT: Final = (
    "uniform_@nn/init.py",
    "normal_@nn/init.py",
    "random_@vision/data.py",
    "randperm@vision/data.py",
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
      scratch: Directory the edited defenses and the lockstep split go to.

    Returns:
      captured: The driver's values, keyed as the port's are.

    """
    scripts = upstream.variants(root, scratch)
    pristine = scripts["pristine"]
    with vision_reference.seeded(vision_reference.SEED):
        recipe = upstream.trace_main(
            pristine,
            upstream.recipe_argv("resnet20-cifar10"),
            build=upstream.linear_model,
            split=vision_reference.synthetic_split(),
        )
    record = upstream_record(scripts, recipe)
    report = compare_recipe(recipe)
    for key, (setting, factory) in reference.SETTINGS.items():
        initial, _ = upstream.initial_main(
            pristine,
            setting,
            count=TRAINING_IMAGES,
            build=upstream.full_model,
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
    report += mismatches(upstream_pipelines(pristine), reference.pipeline_record())
    report += mismatches(record, reference.port_record(scratch))
    report += capture_reference.compare_lockstep(lockstep, run=_PortLockstep(scratch))
    sites = {
        "reference": upstream_sites(pristine),
        "port": port_sites(scratch),
    }
    report += check_sites(sites["reference"], sites["port"])
    report += capture_reference.shared_scores(
        record,
        pairs={
            f"{prefix}/scores/member": f"{prefix}/scores/nonmember"
            for prefix in ("attack", *(f"lockstep/{name}" for name in lockstep))
        },
    )
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
      record: Keyed as the port's records, but for the network and the
        lockstep runs.

    """
    record: dict[str, Tensor] = {}
    for name, script in scripts.items():
        for case, (logits, label) in reference.defense_cases().items():
            for epoch in (0, 1):
                value, grad = upstream.defense_loss(script, logits, label, epoch=epoch)
                key = f"defense/{name}/{case}/{epoch}"
                record[f"{key}/value"] = value
                record[f"{key}/grad"] = grad
    record |= upstream_attack(scripts["pristine"])
    record |= reference.augment_record(
        upstream_pipelines(scripts["pristine"])["pipeline/cifar10/augment"],
    )
    record["recipe/lr"] = torch.tensor(recipe.rates, dtype=torch.float64)
    record["recipe/optimizer"] = torch.tensor(
        [recipe.settings[name] for name in vision_reference.HYPERPARAMETERS],
        dtype=torch.float64,
    )
    member, _, nonmember = recipe.subsets
    count = vision_reference.LOADER_IMAGES
    record[f"split/{count}/member"] = torch.tensor(member)
    record[f"split/{count}/nonmember"] = torch.tensor(nonmember)
    return record


def upstream_pipelines(script: upstream.Script) -> dict[str, Tensor]:
    """Run the driver's CIFAR-10 pipelines, with and without ``--augment``.

    Args:
      script: The pristine driver.

    Returns:
      record: Keyed as :func:`reference.pipeline_record`.

    """
    images = reference.pipeline_images()
    train, test = upstream.pipelines(script, images, dataset="cifar10", augment=False)
    augment, _ = upstream.pipelines(script, images, dataset="cifar10", augment=True)
    return {
        "pipeline/cifar10/train": train,
        "pipeline/cifar10/augment": augment,
        "pipeline/cifar10/test": test,
    }


def upstream_attack(script: upstream.Script) -> dict[str, Tensor]:
    """Score the port's fixed attack logits with the driver's own functions.

    Args:
      script: The pristine driver.

    Returns:
      record: Keyed as :func:`reference.attack_record`.

    """
    record: dict[str, Tensor] = {}
    accuracy: list[float] = []
    scores: dict[str, np.ndarray] = {}
    for split, (logits, label) in reference.attack_logits().items():
        loader = [(logits, label)]
        model = torch.nn.Identity()
        with host_agnostic_numerics():
            scores[split], acc = script.confidence_scores(
                model,
                loader,
                torch.device("cpu"),
            )
        record[f"attack/scores/{split}"] = torch.from_numpy(scores[split])
        accuracy.append(acc)
    record["attack/accuracy"] = torch.tensor(accuracy, dtype=torch.float64)
    auc = script.auc_from_scores(scores["member"], scores["nonmember"])
    record["attack/auc"] = torch.tensor(auc, dtype=torch.float64)
    return record


def compare_recipe(trace: upstream.Trace) -> list[str]:
    """Check what the 300-epoch run told the defense, and the loaders it built.

    Over the seven-image split, three members at batch 128 make one update
    per epoch.

    Args:
      trace: The pristine ``main`` at the recipe's own arguments.

    Returns:
      report: A line per difference from one update per epoch, zero-based,
        and from the port's loaders.

    """
    report: list[str] = []
    if trace.epochs != list(range(reference.EPOCHS)):
        report.append("recipe: the defense was told other epochs")
    config = exp000().dataset
    shuffles = [loader["shuffle"] for loader in trace.loaders]
    if shuffles != [True, False, False, False]:
        report.append(f"recipe: loaders shuffle as {shuffles}")
    sizes = {loader["batch_size"] for loader in trace.loaders}
    if sizes != {config.batch_size} or config.eval_batch_size != config.batch_size:
        report.append(f"recipe: loaders batch {sizes}")
    return report


def compare_initial(
    trace: upstream.Trace,
    *,
    key: str,
    factory: Callable[[], MembershipTrainLoop],
) -> list[str]:
    """Compare a setting's member split of 50,000 images and its network's initial state.

    Args:
      trace: The pristine ``main`` at the setting over 50,000 images, up to
        its first update.
      key: Names the report's lines: the setting's pristine experiment.
      factory: That experiment.

    Returns:
      report: One line per tensor that differs.

    """
    seed = factory().dataset.split_seed
    member, nonmember = member_split(TRAINING_IMAGES, seed=seed)
    train, scored_member, scored_nonmember = trace.subsets
    report = mismatches(
        {
            f"split/{key}/train": torch.tensor(train),
            f"split/{key}/member": torch.tensor(scored_member),
            f"split/{key}/nonmember": torch.tensor(scored_nonmember),
        },
        {
            f"split/{key}/train": member,
            f"split/{key}/member": member,
            f"split/{key}/nonmember": nonmember,
        },
    )
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        ours = vision_reference.full_step(factory).model
    return report + [
        f"settings/{key}/state: {line}"
        for line in capture_reference.compare_states(trace.initial, ours.state_dict())
    ]


def upstream_sites(script: upstream.Script) -> dict[str, int]:
    """Count the driver's random draws over one lockstep run.

    Args:
      script: The pristine driver.

    Returns:
      counts: As :func:`~priml.baselines.mlsbench.dl.vision.draws.recorded_draws`.

    """
    with recorded_draws() as counts:
        _ = upstream.upstream_lockstep(script, "pristine")
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


class _PortLockstep:
    """The port's lockstep run, in the shape ``compare_lockstep`` calls."""

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
