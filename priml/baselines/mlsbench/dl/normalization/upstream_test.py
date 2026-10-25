"""Tests for the runs of the ``dl-normalization`` script, against a stand-in clone.

The stand-in has the script's interface with a tiny network whose one norm is a
``CustomNorm`` wrapping batch norm; its baselines' edits replace that class.
``write_clone`` is shared with the capture's tests.

The script it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/b873944969c53b01e4ee6e843c77767b8c702485/harbor/tasks-docker/mls-bench__dl-normalization/tests/meta/pristine/pytorch-vision/custom_norm.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch

from priml.baselines.mlsbench.dl.normalization import upstream
from priml.baselines.mlsbench.dl.vision import (
    reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.upstream_test import SETTINGS_SCRIPT


if TYPE_CHECKING:
    from pathlib import Path

CUSTOM_NORM: Final = (
    "class CustomNorm(nn.Module):\n"
    "    def __init__(self, num_features):\n"
    "        super().__init__()\n"
    "        self.norm = nn.BatchNorm2d(num_features)\n"
    "\n"
    "    def forward(self, x):\n"
    "        return self.norm(x)\n"
)
"""The stand-in's normalization class, which the baselines' edits replace."""

SCRIPT: Final = (
    SETTINGS_SCRIPT.replace(
        "def compute_loss(logits, targets, config):\n"
        "    return F.cross_entropy(logits, targets)\n",
        CUSTOM_NORM,
    )
    .replace(
        "        self.fc = nn.Linear(4, num_classes)\n",
        "        self.bn = CustomNorm(4)\n        self.fc = nn.Linear(4, num_classes)\n",
    )
    .replace("F.relu(self.conv(x))", "F.relu(self.bn(self.conv(x)))")
    .replace(
        "def train_epoch(model, loader, optimizer, device, config):",
        "def train_epoch(model, loader, criterion, optimizer, device):",
    )
    .replace("compute_loss(model(x), y, config)", "criterion(model(x), y)")
    .replace("def evaluate(model, loader, device, config):", "def evaluate(*args):")
    .replace(
        '    config = {"num_classes": classes, "epoch": 0, "total_epochs": args.epochs}\n',
        "    criterion = nn.CrossEntropyLoss()\n",
    )
    .replace(
        '        config["epoch"] = epoch\n'
        "        train_epoch(model, train_loader, optimizer, None, config)\n"
        "        best = max(best, evaluate(model, test_loader, None, config)[1])\n",
        "        train_epoch(model, train_loader, criterion, optimizer, None)\n"
        "        best = max(best, evaluate(model, test_loader, criterion, None)[1])\n",
    )
    + """

class VGG(nn.Module):
    VGG16_CFG = [64, "M"]


class MobileNetV2(nn.Module):
    CFG = [[1, 16, 1, 1]]
"""
)
"""A stand-in for the ``dl-normalization`` script.

Its ``build_model`` builds the stand-in ResNet for every ``--arch``; the
architecture tables exist so the shallowest-network build can patch them."""

BODIES: Final = {
    "group_norm": "nn.GroupNorm(1, num_features)",
    "batch_instance_norm": "nn.InstanceNorm2d(num_features, affine=True)",
    "switchable_norm": "nn.BatchNorm2d(num_features, eps=0.5)",
}
"""The layer each baseline's edit wraps in ``CustomNorm``."""


def write_clone(root: Path) -> None:
    """Write the stand-in script and the three baselines' edits under ``root``.

    Args:
      root: Directory to hold the clone.

    """
    script = root / upstream.PRISTINE
    script.parent.mkdir(parents=True)
    script.write_text(SCRIPT)
    edits = root / upstream.EDITS
    edits.mkdir(parents=True)
    first = SCRIPT.split(CUSTOM_NORM)[0].count("\n") + 1
    last = first + CUSTOM_NORM.count("\n") - 1
    for name, body in BODIES.items():
        content = CUSTOM_NORM.replace("nn.BatchNorm2d(num_features)", body)
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': {first}, 'end_line': {last}, "
            f"'content': {content!r}}}]\n",
        )


def scripts(tmp_path: Path) -> dict[str, vision_upstream.Upstream]:
    """Write the stand-in clone and import every variant of its script."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return upstream.variants(root, scratch)


def test_each_variant_carries_its_baselines_norm(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == ["batch_norm", *upstream.EDITED]
    traces = {
        name: vision_upstream.trace_main(script) for name, script in variants.items()
    }
    assert "bn.norm.running_mean" in traces["batch_norm"].state
    assert "bn.norm.running_mean" not in traces["group_norm"].state
    assert upstream.upstream_groups(variants["group_norm"])[
        "norm/group_norm/groups"
    ].tolist() == [1] * len(upstream.GROUP_CHANNELS)


def test_the_norm_run_records_output_and_every_gradient(tmp_path: Path) -> None:
    record = upstream.upstream_norm(scripts(tmp_path)["batch_norm"], "batch_norm")
    assert record["norm/batch_norm/train"].shape == upstream.FEATURES_SHAPE
    assert record["norm/batch_norm/grad_input"].shape == upstream.FEATURES_SHAPE
    assert record["norm/batch_norm/grad_params"].shape == (
        2 * upstream.FEATURES_SHAPE[1],
    )
    assert not torch.equal(
        record["norm/batch_norm/train"],
        record["norm/batch_norm/eval"],
    )


def test_lockstep_records_two_updates(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    trace = vision_upstream.trace_main(variants["batch_norm"])
    record, states = upstream.upstream_lockstep(variants["group_norm"], "g", trace)
    assert record["lockstep/g/loss"].shape == (reference.STEPS,)
    assert record["lockstep/g/probe"].shape == (reference.STEPS, 2, reference.CLASSES)
    assert len(states) == reference.STEPS
    assert not torch.equal(states[0]["fc.weight"], states[1]["fc.weight"])


def test_the_setting_lockstep_runs_every_norm_at_every_setting(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    trace = vision_upstream.trace_main(variants["batch_norm"])
    lockstep = upstream.upstream_setting_lockstep(
        variants,
        ["mobilenetv2-fmnist"],
        trace,
    )
    assert set(lockstep) == {f"mobilenetv2-fmnist/{name}" for name in variants}
    record, states = lockstep["mobilenetv2-fmnist/group_norm"]
    assert set(record) == {
        f"settings/mobilenetv2-fmnist/lockstep/group_norm/{part}"
        for part in ("loss", "logits", "probe")
    }
    assert len(states) == reference.STEPS


def test_the_record_holds_every_norm_and_the_recipe(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    traces = {
        name: vision_upstream.trace_main(script) for name, script in variants.items()
    }
    lockstep = {
        name: upstream.upstream_lockstep(script, name, traces["batch_norm"])
        for name, script in variants.items()
    }
    record = upstream.upstream_record(variants, traces, lockstep)
    assert {k.split("/")[1] for k in record if k.startswith("draws/")} == set(variants)
    assert record["schedule/lr"].shape == (200,)
    assert record["optimizer/settings"].shape == (len(reference.HYPERPARAMETERS),)
    assert "lockstep/switchable_norm/loss" in record


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
