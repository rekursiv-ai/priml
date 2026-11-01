"""Tests for the runs of the ``dl-regularization`` script, against a stand-in clone.

The stand-in has the script's interface with a tiny network, a regularizer that
raises as the pristine one does, and the setting's launch script; its
baselines' edits replace that one function. ``write_clone`` is shared with the
capture's tests.

The script it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/b873944969c53b01e4ee6e843c77767b8c702485/harbor/tasks-docker/mls-bench__dl-regularization/tests/meta/pristine/pytorch-vision/custom_reg.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
import torch

from priml.baselines.mlsbench.dl.regularization import upstream
from priml.baselines.mlsbench.dl.regularization.reference import PROBE_EPOCHS
from priml.baselines.mlsbench.dl.vision import (
    reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.upstream_test import SETTINGS_SCRIPT


if TYPE_CHECKING:
    from pathlib import Path

REGULARIZATION: Final = (
    "def compute_regularization(model, inputs, outputs, targets, config):\n"
    '    raise NotImplementedError("Implement your regularization strategy")\n'
)
"""The two lines of the stand-in's regularizer, which the baselines' edits replace."""

SCRIPT: Final = (
    SETTINGS_SCRIPT.replace(
        "def compute_loss(logits, targets, config):\n"
        "    return F.cross_entropy(logits, targets)\n",
        REGULARIZATION,
    )
    .replace(
        "def train_epoch(model, loader, optimizer, device, config):",
        "def train_epoch(model, loader, criterion, optimizer, device, config):",
    )
    .replace(
        "compute_loss(model(x), y, config).backward()",
        "outputs = model(x)\n"
        "        loss = criterion(outputs, y) + compute_regularization(\n"
        "            model, x, outputs, y, config,\n"
        "        )\n"
        "        loss.backward()",
    )
    .replace("def evaluate(model, loader, device, config):", "def evaluate(*args):")
    .replace(
        '    parser.add_argument("--seed", type=int)\n',
        '    parser.add_argument("--seed", type=int)\n'
        '    parser.add_argument("--data-root")\n'
        '    parser.add_argument("--batch-size", type=int, default=128)\n'
        '    parser.add_argument("--lr", type=float, default=0.1)\n'
        '    parser.add_argument("--momentum", type=float, default=0.9)\n'
        '    parser.add_argument("--weight-decay", type=float, default=0)\n'
        '    parser.add_argument("--output-dir")\n',
    )
    .replace(
        "lr=0.1, momentum=0.9, weight_decay=5e-4",
        "lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay",
    )
    .replace(
        "    config = {",
        "    criterion = nn.CrossEntropyLoss()\n    config = {",
    )
    .replace(
        "train_epoch(model, train_loader, optimizer, None, config)",
        "train_epoch(model, train_loader, criterion, optimizer, None, config)",
    )
    .replace(
        "evaluate(model, test_loader, None, config)",
        "evaluate(model, test_loader, criterion, None)",
    )
    + """

class VGG(nn.Module):
    VGG16_CFG = [64, "M"]


class MobileNetV2(nn.Module):
    CFG = [[1, 16, 1, 1]]
"""
)
"""A stand-in for the ``dl-regularization`` script.

Its ``build_model`` builds the stand-in ResNet for every ``--arch``; the
architecture tables exist so the shallowest-network build can patch them."""

SETTING: Final = """\
#!/bin/bash
cd /workspace
python pytorch-vision/custom_reg.py \\
    --arch ARCH --dataset DATASET \\
    --data-root /data/cifar \\
    --epochs 200 --batch-size 128 \\
    --lr 0.2 --momentum 0.8 --weight-decay 5e-4 \\
    --seed ${SEED:-42} \\
    --output-dir ${OUTPUT_DIR:-./output}
"""
"""A stand-in launch script; its rate and momentum differ from the port's.

``ARCH`` and ``DATASET`` name each setting's own."""

BODIES: Final = {
    "dropblock": "torch.tensor(0.0, device=outputs.device)",
    "confidence_penalty": "outputs.pow(2).mean()",
    "orthogonal_reg": "1e-4 * sum(p.pow(2).sum() for p in model.parameters())",
}
"""The return expression each baseline's edit puts in the regularizer."""


def write_clone(root: Path) -> None:
    """Write the stand-in script, its launch script and the baselines' edits.

    Args:
      root: Directory to hold the clone.

    """
    script = root / upstream.PRISTINE
    script.parent.mkdir(parents=True)
    script.write_text(SCRIPT)
    setting = root / upstream.SETTING
    setting.parent.mkdir(parents=True)
    setting.write_text(
        SETTING.replace("ARCH", "resnet56").replace("DATASET", "cifar100"),
    )
    for name, launch in upstream.SETTING_SCRIPTS.items():
        spec = reference.SETTINGS[name]
        (root / launch).write_text(
            SETTING.replace("ARCH", spec.arch).replace("DATASET", spec.dataset),
        )
    edits = root / upstream.EDITS
    edits.mkdir(parents=True)
    first = SCRIPT[: SCRIPT.index(REGULARIZATION)].count("\n") + 1
    for name, body in BODIES.items():
        content = REGULARIZATION.splitlines(keepends=True)[0] + f"    return {body}\n"
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': {first}, 'end_line': {first + 1}, "
            f"'content': {content!r}}}]\n",
        )


def clone(tmp_path: Path) -> Path:
    """Write the stand-in clone and return its root."""
    root = tmp_path / "clone"
    write_clone(root)
    return root


def scripts(root: Path) -> dict[str, vision_upstream.Upstream]:
    """Import every variant of the stand-in clone's script."""
    scratch = root.parent / "scratch"
    scratch.mkdir(exist_ok=True)
    return upstream.variants(root, scratch)


def traced(root: Path) -> vision_upstream.MainTrace:
    """Trace the stand-in's ``main`` as its setting launches it."""
    return upstream.trace_setting(scripts(root)["zero"], upstream.setting_argv(root))


def test_the_pristine_regularizer_raises_and_the_zero_variant_adds_nothing(
    tmp_path: Path,
) -> None:
    root = clone(tmp_path)
    assert upstream.pristine_raises(root)
    variants = scripts(root)
    assert list(variants) == ["zero", *upstream.EDITED]
    zero = upstream.no_regularization(
        torch.nn.Linear(2, 3),
        torch.ones(2, 3),
        torch.ones(2, 3),
        torch.zeros(2),
        {},
    )
    assert torch.equal(zero, torch.tensor(0.0))


def test_a_regularizer_that_returns_is_reported(tmp_path: Path) -> None:
    root = clone(tmp_path)
    script = root / upstream.PRISTINE
    script.write_text(script.read_text().replace("raise NotImplementedError(", "("))
    assert not upstream.pristine_raises(root)


def test_the_setting_argv_resolves_shell_defaults(tmp_path: Path) -> None:
    argv = upstream.setting_argv(clone(tmp_path))
    assert argv[:3] == ["script.py", "--arch", "resnet56"]
    assert argv[argv.index("--seed") + 1] == "42"
    assert argv[argv.index("--output-dir") + 1] == "./output"
    assert argv[argv.index("--weight-decay") + 1] == "5e-4"


def test_main_is_traced_with_the_settings_arguments(tmp_path: Path) -> None:
    trace = traced(clone(tmp_path))
    assert trace.settings["lr"] == 0.2
    assert trace.settings["momentum"] == 0.8
    assert trace.settings["weight_decay"] == 5e-4
    assert len(trace.rates) == 200


def test_the_scripts_default_weight_decay_is_zero(tmp_path: Path) -> None:
    root = clone(tmp_path)
    argv = upstream.setting_argv(root)
    index = argv.index("--weight-decay")
    trace = upstream.trace_setting(
        scripts(root)["zero"],
        argv[:index] + argv[index + 2 :],
    )
    assert trace.settings["weight_decay"] == 0


def test_each_regularizer_is_run_at_every_probe_epoch(tmp_path: Path) -> None:
    record = upstream.upstream_regularizers(scripts(clone(tmp_path)))
    assert {k.split("/")[1] for k in record} == set(upstream.EDITED)
    assert record["regularizer/dropblock/value"].tolist() == [0.0] * len(PROBE_EPOCHS)
    grads = record["regularizer/orthogonal_reg/parameters_grad"]
    assert grads.shape[0] == len(PROBE_EPOCHS)
    assert grads.abs().sum() > 0
    assert record["regularizer/confidence_penalty/logits_grad"].abs().sum() > 0


def test_the_scripts_regularizer_is_called_with_its_own_argument_order() -> None:
    seen: list[tuple[object, ...]] = []

    def regularizer(*arguments: object) -> torch.Tensor:
        seen.append(arguments)
        return torch.zeros(())

    model = torch.nn.Linear(4, 3)
    media, logits, label = torch.ones(2, 4), torch.zeros(2, 3), torch.tensor([0, 2])
    config = {"num_classes": 3, "epoch": 1, "total_epochs": 4}
    upstream._ScriptCall(regularizer)(
        model,
        media=media,
        logits=logits,
        label=label,
        config=config,
    )
    assert seen == [(model, media, logits, label, config)]


@pytest.mark.parametrize("name", ["zero", "confidence_penalty"])
def test_lockstep_adds_the_term_to_cross_entropy(tmp_path: Path, name: str) -> None:
    root = clone(tmp_path)
    variants = scripts(root)
    trace = traced(root)
    record, states = upstream.upstream_lockstep(variants[name], name, trace)
    regularization = record[f"lockstep/{name}/regularization"]
    assert record[f"lockstep/{name}/loss"].shape == (reference.STEPS,)
    assert (regularization == 0).all() == (name == "zero")
    assert record[f"lockstep/{name}/context"].tolist() == [
        [reference.CLASSES, 0, reference.EPOCHS],
        [reference.CLASSES, 1, reference.EPOCHS],
    ]
    assert len(states) == reference.STEPS
    assert not torch.equal(states[0]["fc.weight"], states[1]["fc.weight"])


def test_with_epochs_replaces_only_the_horizon() -> None:
    argv = ["script.py", "--epochs", "200", "--seed", "42"]
    assert upstream.with_epochs(argv, 2) == [
        "script.py",
        "--epochs",
        "2",
        "--seed",
        "42",
    ]
    assert argv[2] == "200"


def test_a_settings_recipe_is_keyed_under_it_as_its_launch_traced_it(
    tmp_path: Path,
) -> None:
    root = clone(tmp_path)
    argv = upstream.setting_argv(root, upstream.SETTING_SCRIPTS["vgg16bn-cifar100"])
    trace = upstream.trace_setting(scripts(root)["zero"], argv)
    record = upstream.upstream_setting_recipe(trace, "vgg16bn-cifar100", argv)
    prefix = "settings/vgg16bn-cifar100/"
    assert set(record) == {
        f"{prefix}schedule/lr",
        f"{prefix}optimizer/settings",
        f"{prefix}setting/batch_size",
    }
    assert record[f"{prefix}schedule/lr"].shape == (200,)
    assert record[f"{prefix}setting/batch_size"].tolist() == [128, 128]


def test_each_regularizer_runs_on_a_settings_tiny_network(tmp_path: Path) -> None:
    variants = scripts(clone(tmp_path))
    model = upstream.tiny_network(variants["zero"], "vgg16bn")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    record = upstream.upstream_setting_regularizers(variants, "s", model)
    assert {k.split("/")[3] for k in record} == set(upstream.EDITED)
    assert all(k.startswith("settings/s/regularizer/") for k in record)
    assert record["settings/s/regularizer/orthogonal_reg/parameters_grad"].abs().sum()
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_the_record_holds_every_regularizer_and_the_recipe(tmp_path: Path) -> None:
    root = clone(tmp_path)
    variants = scripts(root)
    trace = traced(root)
    lockstep = {
        name: upstream.upstream_lockstep(script, name, trace)
        for name, script in variants.items()
    }
    record = upstream.upstream_record(
        trace,
        upstream.upstream_regularizers(variants),
        lockstep,
    )
    assert record["schedule/lr"].shape == (200,)
    assert record["optimizer/settings"].shape == (len(reference.HYPERPARAMETERS),)
    assert {"draws/exp000", "best/accuracy", "lockstep/zero/probe"} <= record.keys()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
