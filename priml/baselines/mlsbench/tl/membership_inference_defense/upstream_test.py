"""Tests for the runs of the ``security-membership-inference-defense`` driver, against a stand-in clone.

The stand-in has the driver's interface with a tiny network and the same
loop, split, scoring and AUC; its baselines' edits replace the defense class.
``write_clone`` is shared with the capture's tests.

The driver it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c/harbor/tasks-docker/mls-bench__security-membership-inference-defense/tests/meta/pristine/pytorch-vision/run_membership_defense.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import random
import sys
import types

import torch

from priml.baselines.mlsbench.dl.vision import (
    reference as vision,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.tl.membership_inference_defense import (
    reference,
    upstream,
)
from priml.testing.golden import read_tensors


_CWD: Final = Path(__file__).resolve().parent

DATA: Final = _CWD / "testdata"

DRIVER: Final = '''\
"""A stand-in for the membership-inference driver."""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from custom_membership_defense import MembershipDefense


class BasicBlock(nn.Module):
    pass


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x)).mean((2, 3)))


def build_model(arch, num_classes):
    if arch not in ("resnet20", "vgg16bn", "mobilenetv2"):
        raise ValueError(f"Unknown architecture: {arch}")
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)


def get_dataset(dataset, data_root, augment=False):
    steps = [transforms.RandomCrop(32, padding=4)] if augment else []
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = transforms.Compose(
        [*steps, transforms.ToTensor(), transforms.Normalize(*stats)]
    )
    train = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=transform
    )
    test = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=transform
    )
    return train, test, 10


def auc_from_scores(member_scores, nonmember_scores):
    return float(member_scores.mean() > nonmember_scores.mean())


@torch.no_grad()
def confidence_scores(model, loader, device):
    scores = [torch.softmax(model(x), 1).max(1).values.numpy() for x, _ in loader]
    return np.concatenate(scores), 0.5


def main():
    parser = argparse.ArgumentParser()
    for name in ("--arch", "--dataset", "--data-root"):
        parser.add_argument(name)
    for name in ("--epochs", "--batch-size", "--seed"):
        parser.add_argument(name, type=int)
    for name in ("--lr", "--momentum", "--weight-decay", "--schedule-gamma"):
        parser.add_argument(name, type=float)
    parser.add_argument("--schedule-milestones", type=int, nargs="+")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train_set, test_set, classes = get_dataset(args.dataset, args.data_root)
    indices = list(range(len(train_set)))
    random.Random(args.seed).shuffle(indices)
    half = len(indices) // 2
    member = torch.utils.data.Subset(train_set, indices[:half])
    nonmember = torch.utils.data.Subset(train_set, indices[half:])
    loader = torch.utils.data.DataLoader
    train = loader(member, batch_size=args.batch_size, shuffle=True, num_workers=4)
    member_eval = loader(member, batch_size=args.batch_size, shuffle=False)
    nonmember_eval = loader(nonmember, batch_size=args.batch_size, shuffle=False)
    test = loader(test_set, batch_size=args.batch_size, shuffle=False)
    model = build_model(args.arch, classes)
    initialize_weights(model)
    defense = MembershipDefense()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.schedule_milestones, gamma=args.schedule_gamma
    )
    for epoch in range(args.epochs):
        for x, y in train:
            optimizer.zero_grad()
            defense.compute_loss(model(x), y, epoch).backward()
            optimizer.step()
        scheduler.step()
    confidence_scores(model, test, None)
    m, _ = confidence_scores(model, member_eval, None)
    n, _ = confidence_scores(model, nonmember_eval, None)
    print(f"TEST_METRICS mia_auc={auc_from_scores(m, n):.4f}")
'''
"""A stand-in for the driver."""

DEFENSE: Final = (
    "import torch.nn.functional as F\n"
    "\n"
    "class MembershipDefense:\n"
    "    def compute_loss(self, logits, labels, epoch):\n"
    "        return F.cross_entropy(logits, labels)\n"
)
"""A stand-in for the pristine defense; its class spans lines 3 to 5."""

EDITED: Final = {
    name: DEFENSE.replace("F.cross_entropy(logits, labels)", body)
    for name, body in {
        "erm": "F.cross_entropy(logits, labels)",
        "label_smoothing": "F.cross_entropy(logits, labels, label_smoothing=0.1)",
        "confidence_penalty": "F.cross_entropy(logits, labels) * 2",
        "relaxloss": "(F.cross_entropy(logits, labels) - 1).abs()",
    }.items()
}
"""Each stand-in baseline's defense module."""


def write_clone(root: Path) -> None:
    """Write the stand-in driver, defense and the four baselines' edits under ``root``.

    Args:
      root: Directory to hold the clone.

    """
    (root / upstream.PRISTINE).mkdir(parents=True)
    (root / upstream.DRIVER).write_text(DRIVER)
    (root / upstream.DEFENSE).write_text(DEFENSE)
    edits = root / upstream.EDITS
    edits.mkdir(parents=True)
    for name, module in EDITED.items():
        content = "".join(module.splitlines(keepends=True)[2:])
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': 3, 'end_line': 5, 'content': {content!r}}}]\n",
        )


def scripts(tmp_path: Path) -> dict[str, upstream.Script]:
    """Write the stand-in clone and import every variant of its driver."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return upstream.variants(root, scratch)


def test_each_variant_imports_its_own_defense(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == ["pristine", *upstream.EDITED]
    logits, label = reference.defense_cases()["low"]
    pristine, _ = upstream.defense_loss(variants["pristine"], logits, label, epoch=0)
    doubled, _ = upstream.defense_loss(
        variants["confidence_penalty"],
        logits,
        label,
        epoch=0,
    )
    assert torch.equal(doubled, pristine * 2)
    assert "custom_membership_defense" not in sys.modules


def test_main_is_traced_update_by_update(tmp_path: Path) -> None:
    script = scripts(tmp_path)["pristine"]
    with vision.seeded(vision.SEED):
        trace = upstream.trace_main(
            script,
            upstream.lockstep_argv("resnet20-cifar10"),
            build=upstream.linear_model,
            split=vision.synthetic_split(),
        )
    # Three members at batch three: one update in each of two epochs.
    assert trace.epochs == [0, 1]
    assert trace.rates == [0.1, 0.1 * 0.1]
    assert len(trace.states) == len(trace.losses) == 2
    indices = list(range(vision.LOADER_IMAGES))
    random.Random(vision.SEED).shuffle(indices)  # noqa: S311 -- The driver's shuffle.
    assert trace.subsets == [indices[:3], indices[:3], indices[3:]]
    assert [len(s) for s, _ in trace.scores] == [7, 3, 4]
    assert trace.loaders[0]["num_workers"] == 4
    assert "TEST_METRICS" in trace.printed


def test_the_initial_run_stops_before_training(tmp_path: Path) -> None:
    script = scripts(tmp_path)["pristine"]
    trace, model = upstream.initial_main(
        script,
        "resnet20-cifar10",
        count=10,
        build=upstream.full_model,
    )
    assert not trace.losses
    assert len(trace.subsets[0]) == 5
    assert trace.settings["weight_decay"] == 1e-4
    assert list(trace.initial) == list(model.state_dict())


def test_the_lockstep_record_has_the_ports_keys(tmp_path: Path) -> None:
    record, states = upstream.upstream_lockstep(
        scripts(tmp_path)["relaxloss"],
        "relaxloss",
    )
    stored = read_tensors(DATA / "reference.pt")
    assert record.keys() == {k for k in stored if k.startswith("lockstep/relaxloss/")}
    assert len(states) == 2


def test_the_pipelines_honor_the_augment_flag(tmp_path: Path) -> None:
    script = scripts(tmp_path)["pristine"]
    images = reference.pipeline_images()
    plain, test = upstream.pipelines(script, images, dataset="cifar10", augment=False)
    assert torch.equal(plain, test)
    cropped, _ = upstream.pipelines(script, images, dataset="cifar10", augment=True)
    assert not torch.equal(cropped, plain)


def test_an_image_reaches_pil_in_its_own_mode() -> None:
    # One channel: upstream._pil squeezes it to the 2-D array PIL.Image.fromarray
    # reads as mode L, Fashion-MNIST's; three channels are fromarray's RGB.
    gray = torch.arange(8, dtype=torch.uint8).reshape(1, 2, 4)
    color = torch.arange(24, dtype=torch.uint8).reshape(3, 2, 4)
    as_gray, as_color = upstream._pil(gray), upstream._pil(color)
    assert (as_gray.mode, as_gray.size) == ("L", (4, 2))
    assert (as_color.mode, as_color.size) == ("RGB", (4, 2))
    assert as_color.getpixel((1, 0)) == tuple(color[:, 0, 1].tolist())


def test_importing_a_driver_restores_a_defense_module_already_present(
    tmp_path: Path,
) -> None:
    root = tmp_path / "clone"
    write_clone(root)
    present = types.ModuleType("custom_membership_defense")
    sys.modules["custom_membership_defense"] = present
    try:
        _ = upstream.driver(root, vision_upstream.load_script(root / upstream.DEFENSE))
        assert sys.modules["custom_membership_defense"] is present
    finally:
        del sys.modules["custom_membership_defense"]


def test_the_recipe_arguments_are_the_settings_scripts() -> None:
    argv = upstream.recipe_argv("vgg16bn-cifar100")
    assert argv[:4] == ["--arch", "vgg16bn", "--dataset", "cifar100"]
    shrunk = upstream.lockstep_argv("resnet20-cifar10")
    assert shrunk[shrunk.index("--epochs") + 1] == str(reference.LOCKSTEP_EPOCHS)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
