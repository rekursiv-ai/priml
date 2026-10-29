"""Tests for the runs of the ``dl-activation-function`` script, against a stand-in clone.

The stand-in has the script's interface with a tiny network that applies its
``CustomActivation`` twice; its baselines' edits replace that class.
``write_clone`` is shared with the capture's tests.

The script it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/b873944969c53b01e4ee6e843c77767b8c702485/harbor/tasks-docker/mls-bench__dl-activation-function/tests/meta/pristine/pytorch-vision/custom_activation.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import torch

from priml.baselines.mlsbench.dl.activation import reference, upstream
from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.upstream_test import SETTINGS_SCRIPT
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.upstream import Upstream

_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD / "testdata"

ACTIVATION: Final = (
    "class CustomActivation(nn.Module):\n"
    "    def forward(self, x):\n"
    "        return F.relu(x)\n"
)
"""The three lines of the stand-in's activation, which the baselines' edits replace."""

SCRIPT: Final = (
    SETTINGS_SCRIPT.replace(
        "def compute_loss(logits, targets, config):\n"
        "    return F.cross_entropy(logits, targets)\n",
        "",
    )
    .replace(
        "class BasicBlock(nn.Module):",
        ACTIVATION + "\n\nclass BasicBlock(nn.Module):",
    )
    .replace(
        "        self.fc = nn.Linear(4, num_classes)\n",
        "        self.act = CustomActivation()\n        self.fc = nn.Linear(4, num_classes)\n",
    )
    .replace("F.relu(self.conv(x))", "self.act(self.act(self.conv(x)))")
    .replace("datasets.CIFAR100(", "datasets.CIFAR10(")
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
"""A stand-in for the ``dl-activation-function`` script.

Its ``build_model`` builds the stand-in ResNet for every ``--arch``; the
architecture tables exist so the shallowest-network build can patch them."""

BODIES: Final = {
    "gelu": "F.gelu(x)",
    "silu": "F.relu(x) * 2",
    "mish": "x * torch.tanh(F.softplus(x))",
}
"""The return expression each baseline's edit puts in ``CustomActivation``."""


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
    first = SCRIPT[: SCRIPT.index(ACTIVATION)].count("\n") + 1
    for name, body in BODIES.items():
        content = ACTIVATION.replace("F.relu(x)", body)
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': {first}, 'end_line': {first + 2}, "
            f"'content': {content!r}}}]\n",
        )


def scripts(tmp_path: Path) -> dict[str, Upstream]:
    """Write the stand-in clone and import every variant of its script."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return upstream.variants(root, scratch)


def test_each_variant_carries_its_baselines_activation(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == list(reference.ACTIVATIONS)
    record = upstream.upstream_activation(variants["silu"], "silu")
    x = reference.fixed_inputs()
    assert torch.equal(record["activation/silu/value"], torch.relu(x) * 2)
    assert torch.equal(record["activation/silu/grad"], (x > 0).float() * 2)


def test_main_is_traced_at_the_resnet20_cifar10_setting(tmp_path: Path) -> None:
    trace = upstream.trace_setting(scripts(tmp_path)["relu"])
    assert len(trace.rates) == 200
    assert trace.settings["momentum"] == 0.9
    assert list(trace.state) == ["conv.weight", "conv.bias", "fc.weight", "fc.bias"]


def test_lockstep_records_two_updates(tmp_path: Path) -> None:
    script = scripts(tmp_path)["gelu"]
    trace = upstream.trace_setting(script)
    record, states = upstream.upstream_lockstep(script, "gelu", trace)
    assert record["lockstep/gelu/loss"].shape == (vision_reference.STEPS,)
    assert record["lockstep/gelu/probe"].shape == (
        vision_reference.STEPS,
        2,
        vision_reference.CLASSES,
    )
    assert len(states) == vision_reference.STEPS
    assert not torch.equal(states[0]["fc.weight"], states[1]["fc.weight"])


def test_the_setting_lockstep_runs_every_activation_at_every_setting(
    tmp_path: Path,
) -> None:
    variants = scripts(tmp_path)
    trace = upstream.trace_setting(variants["relu"])
    lockstep = upstream.upstream_setting_lockstep(variants, ["vgg16bn-cifar100"], trace)
    assert set(lockstep) == {f"vgg16bn-cifar100/{name}" for name in variants}
    record, states = lockstep["vgg16bn-cifar100/mish"]
    assert set(record) == {
        f"settings/vgg16bn-cifar100/lockstep/mish/{part}"
        for part in ("loss", "logits", "probe")
    }
    assert len(states) == vision_reference.STEPS


def test_the_record_holds_every_key_the_golden_does(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    trace = upstream.trace_setting(variants["relu"])
    lockstep = {
        name: upstream.upstream_lockstep(script, name, trace)
        for name, script in variants.items()
    }
    record = upstream.upstream_record(variants, trace, lockstep)
    assert record.keys() == read_tensors(_DATA / "reference.pt").keys()
    assert torch.equal(
        record["schedule/lr"],
        torch.tensor(trace.rates, dtype=torch.float64),
    )
    assert torch.equal(
        record["optimizer/settings"],
        torch.tensor(
            [trace.settings[name] for name in vision_reference.HYPERPARAMETERS],
            dtype=torch.float64,
        ),
    )
    assert record["draws/exp000"] is trace.draws
    assert {
        record[key].dtype
        for key in ("schedule/lr", "optimizer/settings", "best/accuracy")
    } == {torch.float64}
    assert torch.equal(
        record["best/accuracy"],
        torch.tensor(trace.best, dtype=torch.float64),
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
