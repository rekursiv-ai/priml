"""Tests for the runs of the ``cv-pooling-aggregation`` script, against a stand-in clone.

The stand-in has the script's interface with a tiny network ending in
``CustomPool``; its baselines' edits replace that one class. ``write_clone`` is
shared with the capture's tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

import torch

from priml.baselines.mlsbench.pooling import upstream
from priml.baselines.mlsbench.pooling.upstream import Script
from priml.baselines.mlsbench.pytorch_vision import (
    reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.pytorch_vision.upstream_test import SCRIPT as LOSS_SCRIPT


if TYPE_CHECKING:
    from pathlib import Path

POOL: Final = (
    "class CustomPool(nn.Module):\n"
    "    def forward(self, x):\n"
    "        return F.adaptive_avg_pool2d(x, 1).flatten(1)\n"
)
"""The three lines of the stand-in's pool, which the baselines' edits replace."""

SCRIPT: Final = (
    LOSS_SCRIPT.replace(
        "def compute_loss(logits, targets, config):\n"
        "    return F.cross_entropy(logits, targets)\n",
        "",
    )
    .replace("class BasicBlock(nn.Module):", POOL + "\n\nclass BasicBlock(nn.Module):")
    .replace(
        "        self.fc = nn.Linear(4, num_classes)\n",
        "        self.pool = CustomPool()\n        self.fc = nn.Linear(4, num_classes)\n",
    )
    .replace(
        "F.adaptive_avg_pool2d(F.relu(self.conv(x)), 1).flatten(1)",
        "self.pool(F.relu(self.conv(x)))",
    )
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
)
"""A stand-in for the ``cv-pooling-aggregation`` script."""

CLASSES: Final = {
    "global_max": POOL.replace("adaptive_avg_pool2d", "adaptive_max_pool2d"),
    "gem": (
        "class CustomPool(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.p = nn.Parameter(torch.ones(1) * 2.0)\n"
        "    def forward(self, x):\n"
        "        return (x * self.p).mean((2, 3))\n"
    ),
    "avg_max": POOL.replace(".flatten(1)", ".flatten(1) * 2"),
}
"""The class each baseline's edit puts in place of the pool."""


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
    first = SCRIPT[: SCRIPT.index(POOL)].count("\n") + 1
    for name, content in CLASSES.items():
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': {first}, 'end_line': {first + 2}, "
            f"'content': {content!r}}}]\n",
        )


def scripts(tmp_path: Path) -> dict[str, vision_upstream.Upstream]:
    """Write the stand-in clone and import every variant of its script."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return upstream.variants(root, scratch)


def test_fixed_inputs_are_distinct_and_signed() -> None:
    features = upstream.fixed_features()
    assert features.shape == upstream.FEATURES
    assert features.flatten(2).unique(dim=-1).shape == features.flatten(2).shape
    assert features.min() < 0 < features.max()
    assert upstream.fixed_cotangent().unique().numel() == 6


def test_each_variant_carries_its_baselines_pool(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == ["avg", *upstream.EDITED]
    pristine = upstream.pooled(
        cast(Script, variants["avg"]).CustomPool(),
        "avg",
    )
    doubled = upstream.pooled(
        cast(Script, variants["avg_max"]).CustomPool(),
        "avg_max",
    )
    assert torch.equal(doubled["pool/avg_max/value"], pristine["pool/avg/value"] * 2)
    assert "pool/avg/param_grad" not in pristine


def test_a_pool_with_a_parameter_records_its_gradient(tmp_path: Path) -> None:
    script = cast(Script, scripts(tmp_path)["gem"])
    record = upstream.pooled(script.CustomPool(), "gem")
    features, cotangent = upstream.fixed_features(), upstream.fixed_cotangent()
    expected = (features.mean((2, 3)) * cotangent).sum()
    assert record["pool/gem/param_grad"].shape == (1,)
    torch.testing.assert_close(record["pool/gem/param_grad"][0], expected)


def test_lockstep_records_two_updates_with_the_scripts_pool(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    trace = vision_upstream.trace_main(variants["gem"])
    record, states = upstream.upstream_lockstep(variants["gem"], "gem", trace)
    assert record["lockstep/gem/loss"].shape == (reference.STEPS,)
    assert record["lockstep/gem/probe"].shape == (
        reference.STEPS,
        2,
        reference.CLASSES,
    )
    assert len(states) == reference.STEPS
    assert "pool.p" in states[0]
    assert not torch.equal(states[0]["pool.p"], states[1]["pool.p"])


def test_the_record_holds_every_pool_and_the_pristine_recipe(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    traces = {
        name: vision_upstream.trace_main(script) for name, script in variants.items()
    }
    lockstep = {
        name: upstream.upstream_lockstep(script, name, traces["avg"])
        for name, script in variants.items()
    }
    record = upstream.upstream_record(variants, traces, lockstep)
    assert {k.split("/")[1] for k in record if k.startswith("pool/")} == set(variants)
    assert {k for k in record if k.startswith("draws/")} == {
        f"draws/{name}" for name in variants
    }
    assert record["schedule/lr"].shape == (200,)
    assert record["optimizer/settings"].shape == (len(reference.HYPERPARAMETERS),)
    assert "lockstep/avg_max/loss" in record


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
