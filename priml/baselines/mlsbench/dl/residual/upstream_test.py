"""Tests for the runs of the ``dl-residual-connection`` script, against a stand-in clone.

The stand-in has the script's interface with a tiny network built around one
``CustomBlock``; its baselines' edits replace that one class. ``write_clone`` is
shared with the capture's tests.

The script it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c/harbor/tasks-docker/mls-bench__dl-residual-connection/tests/meta/pristine/pytorch-vision/custom_residual.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast, override
from unittest import mock

import torch

from priml.baselines.mlsbench.dl.residual import upstream
from priml.baselines.mlsbench.dl.residual.upstream import Script
from priml.baselines.mlsbench.dl.vision import (
    reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.upstream_test import SCRIPT as LOSS_SCRIPT


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

BLOCK: Final = (
    "class CustomBlock(nn.Module):\n"
    "    def __init__(self, in_planes, planes, stride=1):\n"
    "        super().__init__()\n"
    "        self.conv = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1)\n"
    "    def forward(self, x):\n"
    "        return F.relu(self.conv(x))\n"
)
"""The six lines of the stand-in's block, which the baselines' edits replace."""

SCRIPT: Final = (
    LOSS_SCRIPT.replace(
        "def compute_loss(logits, targets, config):\n"
        "    return F.cross_entropy(logits, targets)\n",
        "",
    )
    .replace("class BasicBlock(nn.Module):\n    pass\n", BLOCK)
    .replace(
        "        self.fc = nn.Linear(4, num_classes)\n",
        "        self.block = block(4, 4, 1)\n        self.fc = nn.Linear(4, num_classes)\n",
    )
    .replace(
        "F.adaptive_avg_pool2d(F.relu(self.conv(x)), 1)",
        "F.adaptive_avg_pool2d(self.block(F.relu(self.conv(x))), 1)",
    )
    .replace(
        "ResNet(BasicBlock, [1, 1, 1], classes)",
        "ResNet(CustomBlock, [1, 1, 1], classes)",
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
"""A stand-in for the ``dl-residual-connection`` script."""

CLASSES: Final = {
    "pre_activation": BLOCK.replace("F.relu(self.conv(x))", "self.conv(F.relu(x))"),
    "gated_residual": BLOCK.replace(
        "        super().__init__()\n",
        "        super().__init__()\n        self.alpha = nn.Parameter(torch.ones(1))\n",
    ).replace("F.relu(self.conv(x))", "self.alpha * F.relu(self.conv(x))"),
    "stochastic_depth": BLOCK.replace(
        "    def __init__",
        "    _block_counter = 5\n    def __init__",
    ).replace(
        "        return F.relu(self.conv(x))\n",
        "        if self.training and torch.rand(1).item() < 0.5:\n"
        "            return F.relu(self.conv(x)) * 0\n"
        "        return F.relu(self.conv(x))\n",
    ),
}
"""The class each baseline's edit puts in place of the block."""


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
    first = SCRIPT[: SCRIPT.index(BLOCK)].count("\n") + 1
    for name, content in CLASSES.items():
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': {first}, 'end_line': {first + 5}, "
            f"'content': {content!r}}}]\n",
        )


def scripts(tmp_path: Path) -> dict[str, vision_upstream.Upstream]:
    """Write the stand-in clone and import every variant of its script."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return upstream.variants(root, scratch)


def test_every_variant_names_its_block_as_the_shared_capture_expects(
    tmp_path: Path,
) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == ["basic", *upstream.EDITED]
    for script in variants.values():
        assert script.BasicBlock is cast(Script, script).CustomBlock


def test_drawn_records_every_rand_in_order() -> None:
    with upstream.drawn() as draws:
        first = torch.rand(2)
        second = torch.rand(1)
    assert torch.equal(torch.cat(draws), torch.cat([first, second]))


def test_a_block_run_records_draws_and_counts_an_unreached_parameter_zero(
    tmp_path: Path,
) -> None:
    script = cast(Script, scripts(tmp_path)["stochastic_depth"])
    record = upstream.block_run(
        lambda: script.CustomBlock(upstream.FEATURES[1], 4, 2),
        "b",
    )
    assert record["b/train"].shape == (upstream.FORWARDS, 2, 4, 2, 3)
    assert record["b/draws"].shape == (upstream.FORWARDS,)
    assert record["b/eval"].shape == (2, 4, 2, 3)
    assert record["b/grad"].shape == upstream.FEATURES
    unused = upstream.block_run(lambda: torch.nn.Linear(upstream.FEATURES[-1], 3), "u")
    assert unused["u/draws"].numel() == 0
    assert unused["u/param_grad"].shape == (upstream.FEATURES[-1] * 3 + 3,)
    dropped = upstream.block_run(_Unused, "d")
    assert torch.equal(dropped["d/param_grad"], torch.zeros(3))


def test_a_block_counter_starts_at_zero_for_each_run_and_is_restored(
    tmp_path: Path,
) -> None:
    script = scripts(tmp_path)["stochastic_depth"]
    block = cast(Script, script).CustomBlock
    seen: list[object] = []
    original = upstream.block_run

    def spying(
        make: Callable[[], torch.nn.Module],
        key: str,
    ) -> dict[str, torch.Tensor]:
        seen.append(cast(object, vars(block)["_block_counter"]))
        return original(make, key)

    with mock.patch.object(upstream, "block_run", spying):
        _ = upstream.upstream_blocks(script, "s")
    assert seen == [0] * len(upstream.BLOCKS)
    assert vars(block)["_block_counter"] == 5


def test_each_variant_runs_its_baselines_block(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    basic = upstream.upstream_blocks(variants["basic"], "basic")
    gated = upstream.upstream_blocks(variants["gated_residual"], "gated_residual")
    assert {k.split("/")[2] for k in basic} == {"identity", "projected"}
    assert torch.equal(
        basic["block/basic/identity/eval"],
        gated["block/gated_residual/identity/eval"],
    )
    assert gated["block/gated_residual/identity/param_grad"].numel() == (
        basic["block/basic/identity/param_grad"].numel() + 1
    )


def test_lockstep_records_two_updates_with_the_scripts_block(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    trace = vision_upstream.trace_main(variants["gated_residual"])
    record, states = upstream.upstream_lockstep(
        variants["gated_residual"],
        "gated_residual",
        trace,
    )
    assert record["lockstep/gated_residual/loss"].shape == (reference.STEPS,)
    assert record["lockstep/gated_residual/probe"].shape == (
        reference.STEPS,
        2,
        reference.CLASSES,
    )
    assert len(states) == reference.STEPS
    assert not torch.equal(states[0]["block.alpha"], states[1]["block.alpha"])


def test_the_record_holds_every_block_and_the_pristine_recipe(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    traces = {
        name: vision_upstream.trace_main(script) for name, script in variants.items()
    }
    lockstep = {
        name: upstream.upstream_lockstep(script, name, traces["basic"])
        for name, script in variants.items()
    }
    record = upstream.upstream_record(variants, traces, lockstep)
    assert {k.split("/")[1] for k in record if k.startswith("block/")} == set(variants)
    assert {k for k in record if k.startswith("draws/")} == {
        f"draws/{name}" for name in variants
    }
    assert record["schedule/lr"].shape == (200,)
    assert record["optimizer/settings"].shape == (len(reference.HYPERPARAMETERS),)
    assert "lockstep/stochastic_depth/loss" in record


class _Unused(torch.nn.Module):
    """Holds a parameter its forward never reaches."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
