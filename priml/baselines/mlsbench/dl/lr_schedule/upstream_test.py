"""Tests for the runs of the ``dl-lr-schedule`` script, against a stand-in clone.

The stand-in has the script's interface with a tiny network and a constant
``get_lr``; its baselines' edits replace that one function. ``write_clone`` is
shared with the capture's tests.

The script it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c/harbor/tasks-docker/mls-bench__dl-lr-schedule/tests/meta/pristine/pytorch-vision/custom_schedule.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch

from priml.baselines.mlsbench.dl.lr_schedule import upstream
from priml.baselines.mlsbench.dl.vision import (
    reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.upstream_test import SCRIPT as LOSS_SCRIPT


if TYPE_CHECKING:
    from pathlib import Path

GET_LR: Final = (
    "def get_lr(epoch, total_epochs, base_lr, config):\n    return base_lr\n"
)
"""The two lines of the stand-in's schedule, which the baselines' edits replace."""

SCRIPT: Final = (
    LOSS_SCRIPT.replace(
        "def compute_loss(logits, targets, config):\n"
        "    return F.cross_entropy(logits, targets)\n",
        GET_LR,
    )
    .replace(
        "def train_epoch(model, loader, optimizer, device, config):",
        "def train_epoch(model, loader, criterion, optimizer, device):",
    )
    .replace("compute_loss(model(x), y, config)", "criterion(model(x), y)")
    .replace("def evaluate(model, loader, device, config):", "def evaluate(*args):")
    .replace(
        "    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)\n"
        '    config = {"num_classes": classes, "epoch": 0, "total_epochs": args.epochs}\n',
        "    criterion = nn.CrossEntropyLoss()\n",
    )
    .replace(
        '        config["epoch"] = epoch\n'
        "        train_epoch(model, train_loader, optimizer, None, config)\n"
        "        best = max(best, evaluate(model, test_loader, None, config)[1])\n"
        "        scheduler.step()\n",
        "        for group in optimizer.param_groups:\n"
        '            group["lr"] = get_lr(epoch, args.epochs, 0.1, {})\n'
        "        train_epoch(model, train_loader, criterion, optimizer, None)\n"
        "        best = max(best, evaluate(model, test_loader, criterion, None)[1])\n",
    )
)
"""A stand-in for the ``dl-lr-schedule`` script."""

BODIES: Final = {
    "cosine": "base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))",
    "warmup_cosine": "base_lr / 2",
    "one_cycle": "base_lr * (epoch + 1)",
}
"""The return expression each baseline's edit puts in ``get_lr``."""


def write_clone(root: Path) -> None:
    """Write the stand-in script and the three baselines' edits under ``root``.

    Args:
      root: Directory to hold the clone.

    """
    script = root / upstream.PRISTINE
    script.parent.mkdir(parents=True)
    script.write_text("import math\n" + SCRIPT)
    edits = root / upstream.EDITS
    edits.mkdir(parents=True)
    first = ("import math\n" + SCRIPT).split(GET_LR)[0].count("\n") + 1
    for name, body in BODIES.items():
        content = GET_LR.replace("return base_lr", f"return {body}")
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': {first}, 'end_line': {first + 1}, "
            f"'content': {content!r}}}]\n",
        )


def scripts(tmp_path: Path) -> dict[str, vision_upstream.Upstream]:
    """Write the stand-in clone and import every variant of its script."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return upstream.variants(root, scratch)


def test_each_variant_carries_its_baselines_schedule(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == ["constant", *upstream.EDITED]
    traces = {
        name: vision_upstream.trace_main(script) for name, script in variants.items()
    }
    assert traces["constant"].rates == [0.1] * 200
    assert traces["warmup_cosine"].rates == [0.05] * 200
    assert traces["one_cycle"].rates[:3] == [0.1, 0.2, 0.30000000000000004]
    assert "initial_lr" not in traces["constant"].settings


def test_lockstep_sets_the_rate_before_each_update(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    trace = vision_upstream.trace_main(variants["constant"])
    record, states = upstream.upstream_lockstep(
        variants["one_cycle"],
        "one_cycle",
        trace,
    )
    assert record["lockstep/one_cycle/lr"].tolist() == [0.1, 0.2]
    assert record["lockstep/one_cycle/loss"].shape == (reference.STEPS,)
    assert record["lockstep/one_cycle/probe"].shape == (
        reference.STEPS,
        2,
        reference.CLASSES,
    )
    assert len(states) == reference.STEPS
    assert not torch.equal(states[0]["fc.weight"], states[1]["fc.weight"])


def test_lockstep_builds_its_optimizer_from_mains_settings(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    trace = vision_upstream.trace_main(variants["constant"])
    record, _ = upstream.upstream_lockstep(variants["constant"], "constant", trace)
    trace.settings["momentum"] = 0.0
    changed, _ = upstream.upstream_lockstep(variants["constant"], "constant", trace)
    assert torch.equal(
        record["lockstep/constant/loss"][:1],
        changed["lockstep/constant/loss"][:1],
    )
    assert not torch.equal(
        record["lockstep/constant/probe"],
        changed["lockstep/constant/probe"],
    )


def test_a_setting_lockstep_tells_get_lr_the_setting_and_keys_under_it(
    tmp_path: Path,
) -> None:
    variants = scripts(tmp_path)
    trace = vision_upstream.trace_main(variants["constant"])
    record, states = upstream.upstream_lockstep(
        variants["one_cycle"],
        "one_cycle",
        trace,
        setting="resnet20-cifar10",
    )
    prefix = "settings/resnet20-cifar10/lockstep/one_cycle/"
    assert set(record) == {f"{prefix}{k}" for k in ("lr", "loss", "logits", "probe")}
    assert record[f"{prefix}lr"].tolist() == [0.1, 0.2]
    assert len(states) == reference.STEPS


def test_each_settings_schedules_are_traced_per_variant(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    record = upstream.upstream_setting_schedules(variants, ["resnet20-cifar10"])
    assert set(record) == {
        f"settings/resnet20-cifar10/schedule/{name}" for name in variants
    }
    assert record["settings/resnet20-cifar10/schedule/warmup_cosine"].tolist() == (
        [0.05] * 200
    )
    assert record["settings/resnet20-cifar10/schedule/constant"].dtype == torch.float64


def test_the_record_holds_every_schedule_and_the_optimizer(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    traces = {
        name: vision_upstream.trace_main(script) for name, script in variants.items()
    }
    lockstep = {
        name: upstream.upstream_lockstep(script, name, traces["constant"])
        for name, script in variants.items()
    }
    record = upstream.upstream_record(traces, lockstep)
    assert {k for k in record if k.startswith("schedule/")} == {
        f"schedule/{name}" for name in variants
    }
    assert record["optimizer/settings"].shape == (len(upstream.SETTINGS),)
    assert "lockstep/cosine/lr" in record


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
