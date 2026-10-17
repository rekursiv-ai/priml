"""Tests for the runs of the ``dl-weight-initialization`` script, against a stand-in clone.

The stand-in has the script's interface with a tiny network and a Kaiming
``initialize_weights``; its baselines' edits replace that one function.
``write_clone`` is shared with the capture's tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import torch

from priml.baselines.mlsbench.initialization import upstream
from priml.baselines.mlsbench.pytorch_vision import (
    reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.pytorch_vision.upstream_test import SCRIPT as LOSS_SCRIPT


if TYPE_CHECKING:
    from pathlib import Path

INIT: Final = (
    "def initialize_weights(model):\n"
    "    for m in model.modules():\n"
    "        if isinstance(m, (nn.Conv2d, nn.Linear)):\n"
    "            nn.init.kaiming_normal_(m.weight)\n"
    "            nn.init.zeros_(m.bias)\n"
)
"""The stand-in's initialization, which the baselines' edits replace."""

SCRIPT: Final = (
    LOSS_SCRIPT.replace(
        "def compute_loss(logits, targets, config):\n"
        "    return F.cross_entropy(logits, targets)\n",
        "",
    )
    .replace(INIT, INIT.replace("(model)", "(model, config)"))
    .replace("initialize_weights(model)\n", "initialize_weights(model, {})\n")
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
    .replace('        config["epoch"] = epoch\n', "")
    .replace(
        "train_epoch(model, train_loader, optimizer, None, config)",
        "train_epoch(model, train_loader, criterion, optimizer, None)",
    )
    .replace(
        "evaluate(model, test_loader, None, config)",
        "evaluate(model, test_loader, criterion, None)",
    )
)
"""A stand-in for the ``dl-weight-initialization`` script."""

DRAWS: Final = {
    "kaiming_normal": "nn.init.normal_(m.weight, 0, 0.5)",
    "fixup": "nn.init.ones_(m.weight)",
    "orthogonal": "nn.init.orthogonal_(m.weight)",
}
"""The weight draw each baseline's edit puts in ``initialize_weights``."""


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
    pristine = INIT.replace("(model)", "(model, config)")
    first = SCRIPT[: SCRIPT.index(pristine)].count("\n") + 1
    for name, draw in DRAWS.items():
        content = pristine.replace("nn.init.kaiming_normal_(m.weight)", draw)
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': {first}, 'end_line': {first + 4}, "
            f"'content': {content!r}}}]\n",
        )


def scripts(tmp_path: Path) -> dict[str, vision_upstream.Upstream]:
    """Write the stand-in clone and import every variant of its script."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return upstream.variants(root, scratch)


def test_each_variant_carries_its_baselines_initialization(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == ["normal", *upstream.EDITED]
    fc = {
        name: upstream.initialized(script).state_dict()["fc.weight"]
        for name, script in variants.items()
    }
    assert torch.equal(fc["fixup"], torch.ones(reference.CLASSES, 4))
    assert not torch.equal(fc["normal"], fc["kaiming_normal"])


def test_the_network_run_records_its_draws_logits_and_state(tmp_path: Path) -> None:
    script = scripts(tmp_path)["fixup"]
    record, state = upstream.upstream_network(script, "fixup")
    assert set(record) == {
        "network/fixup/draws",
        "network/fixup/train",
        "network/fixup/eval",
    }
    assert record["network/fixup/train"].shape == (2, reference.CLASSES)
    assert torch.equal(state["fc.weight"], torch.ones(reference.CLASSES, 4))


def test_lockstep_records_two_updates_through_mains_criterion(tmp_path: Path) -> None:
    script = scripts(tmp_path)["normal"]
    trace = vision_upstream.trace_main(script)
    record, states = upstream.upstream_lockstep(script, "normal", trace)
    assert record["lockstep/normal/loss"].shape == (reference.STEPS,)
    assert record["lockstep/normal/logits"].shape == (
        reference.STEPS,
        2,
        reference.CLASSES,
    )
    assert record["lockstep/normal/probe"].shape == (
        reference.STEPS,
        2,
        reference.CLASSES,
    )
    assert len(states) == reference.STEPS
    assert not torch.equal(states[0]["fc.weight"], states[1]["fc.weight"])


def test_the_record_holds_every_variant_and_the_recipe(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    traces = {
        name: vision_upstream.trace_main(script) for name, script in variants.items()
    }
    networks = {
        name: upstream.upstream_network(script, name)[0]
        for name, script in variants.items()
    }
    lockstep = {
        name: upstream.upstream_lockstep(script, name, traces["normal"])
        for name, script in variants.items()
    }
    record = upstream.upstream_record(traces, networks, lockstep)
    assert {k for k in record if k.startswith("draws/")} == {
        f"draws/{name}" for name in variants
    }
    assert record["schedule/lr"].shape == (200,)
    assert record["optimizer/settings"].shape == (len(reference.HYPERPARAMETERS),)
    assert "network/orthogonal/eval" in record
    assert "lockstep/fixup/probe" in record


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
