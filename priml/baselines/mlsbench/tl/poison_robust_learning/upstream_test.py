"""Tests for the runs of the ``security-poison-robust-learning`` driver, against a stand-in clone.

The stand-in has the driver's interface, poisoning and evaluations with a tiny
network; its baselines' edits replace the loss class. ``write_clone`` is shared
with the capture's tests.

The driver it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/b873944969c53b01e4ee6e843c77767b8c702485/harbor/tasks-docker/mls-bench__security-poison-robust-learning/tests/meta/pristine/pytorch-vision/bench/poison/run_poison_robust.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import random
import sys
import types

import numpy as np
import torch

from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.tl.poison_robust_learning import (
    reference,
    upstream,
)
from priml.baselines.mlsbench.tl.poison_robust_learning.data import (
    flipped_labels,
)
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from PIL import Image
    from torch import Tensor

    import pytest

_CWD: Final = Path(__file__).resolve().parent

DATA: Final = _CWD / "testdata"

DRIVER: Final = '''\
"""A stand-in for the poison-robust driver."""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from custom_robust_loss import RobustLoss


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
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)


def get_datasets(dataset, data_root):
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    train = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=train_transform
    )
    test = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=test_transform
    )
    return train, test, 10


class PoisonedDataset(Dataset):
    def __init__(self, base_dataset, num_classes, poison_fraction, seed):
        self.base_dataset = base_dataset
        self.labels = torch.tensor(base_dataset.targets, dtype=torch.long)
        g = torch.Generator().manual_seed(seed)
        n = len(base_dataset)
        self.poison_idx = torch.randperm(n, generator=g)[: int(n * poison_fraction)]
        self.labels[self.poison_idx] = (self.labels[self.poison_idx] + 1) % num_classes
        self.is_poison = torch.zeros(n, dtype=torch.bool)
        self.is_poison[self.poison_idx] = True

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        return img, self.labels[idx].item()


@torch.no_grad()
def evaluate_test_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        correct += model(images).argmax(dim=1).eq(labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_poison_fit(model, poisoned_dataset, device, batch_size):
    model.eval()
    loader = DataLoader(poisoned_dataset, batch_size=batch_size, shuffle=False)
    hit, total, offset = 0, 0, 0
    for images, labels in loader:
        mask = poisoned_dataset.is_poison[offset:offset + images.size(0)]
        offset += images.size(0)
        if not mask.any():
            continue
        hit += model(images).argmax(dim=1)[mask].eq(labels[mask]).sum().item()
        total += int(mask.sum().item())
    return hit / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    for name in ("--arch", "--dataset", "--data-root"):
        parser.add_argument(name)
    for name in ("--epochs", "--batch-size", "--seed"):
        parser.add_argument(name, type=int)
    for name in ("--lr", "--momentum", "--weight-decay", "--poison-fraction"):
        parser.add_argument(name, type=float)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_set, test_set, classes = get_datasets(args.dataset, args.data_root)
    poisoned = PoisonedDataset(train_set, classes, args.poison_fraction, args.seed)
    train = DataLoader(poisoned, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = build_model(args.arch, classes)
    initialize_weights(model)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    robust_loss = RobustLoss()
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train:
            loss = robust_loss.compute_loss(model(images), labels, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    test_acc = evaluate_test_acc(model, test, None)
    poison_fit = evaluate_poison_fit(model, poisoned, None, args.batch_size)
    print(f"TEST_METRICS test_acc={test_acc:.4f} poison_fit={poison_fit:.4f}")
'''
"""A stand-in for the driver."""

LOSS: Final = (
    "import torch\n"
    "import torch.nn.functional as F\n"
    "\n"
    "class RobustLoss:\n"
    "    def compute_loss(self, logits, labels, epoch):\n"
    "        return F.cross_entropy(logits, labels)\n"
)
"""A stand-in for the pristine loss; its class spans lines 4 to 6."""

EDITED: Final = {
    name: LOSS.replace("F.cross_entropy(logits, labels)", body)
    for name, body in {
        "cross_entropy": "F.cross_entropy(logits, labels)",
        "generalized_ce": "F.cross_entropy(logits, labels) * 2",
        "symmetric_ce": "F.cross_entropy(logits, labels) + 1",
        "bootstrap": "F.cross_entropy(logits, labels, label_smoothing=0.1)",
    }.items()
}
"""Each stand-in baseline's loss module."""


def write_clone(root: Path) -> None:
    """Write the stand-in driver, loss and the four baselines' edits under ``root``.

    Args:
      root: Directory to hold the clone.

    """
    (root / upstream.PRISTINE).mkdir(parents=True)
    (root / upstream.DRIVER).write_text(DRIVER)
    (root / upstream.LOSS).write_text(LOSS)
    edits = root / upstream.EDITS
    edits.mkdir(parents=True)
    for name, module in EDITED.items():
        content = "".join(module.splitlines(keepends=True)[3:])
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': 4, 'end_line': 6, 'content': {content!r}}}]\n",
        )


def scripts(tmp_path: Path) -> dict[str, upstream.Script]:
    """Write the stand-in clone and import every variant, its loaders worker-free.

    The driver asks for four workers; forking them per run would cost a test far
    more than its run. A serial trace still records the four it asked for.
    """
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    variants = upstream.variants(root, scratch)
    for script in variants.values():
        vars(script)["DataLoader"] = workerless
    return variants


def workerless(
    dataset: torch.utils.data.Dataset[tuple[Tensor, Tensor]],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader[tuple[Tensor, Tensor]]:
    """Build the loader the driver asks for, without its worker processes."""
    del num_workers
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def test_each_variant_imports_its_own_loss(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == ["pristine", *upstream.EDITED]
    logits, label = vision_reference.fixed_logits()
    plain, _ = upstream.loss_value(variants["pristine"], logits, label, epoch=0)
    doubled, grad = upstream.loss_value(
        variants["generalized_ce"],
        logits,
        label,
        epoch=0,
    )
    assert torch.equal(doubled, plain * 2)
    assert grad.shape == logits.shape
    assert "custom_robust_loss" not in sys.modules


def test_importing_a_driver_puts_back_a_loss_module_already_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = types.ModuleType("custom_robust_loss")
    monkeypatch.setitem(sys.modules, "custom_robust_loss", held)
    variants = scripts(tmp_path)
    assert sys.modules["custom_robust_loss"] is held
    assert variants["pristine"].RobustLoss is not getattr(held, "RobustLoss", None)


def test_a_served_split_runs_its_transform_on_each_pil_image() -> None:
    media, label = vision_reference.synthetic_split()
    served = upstream.Served((media, label), _Pixels())
    image, target = served[3]
    assert isinstance(image, torch.Tensor)
    assert torch.equal(image, media[3])
    assert target == 3
    assert len(served) == vision_reference.LOADER_IMAGES


def test_the_drivers_poisoned_dataset_flips_as_the_port_does(tmp_path: Path) -> None:
    script = scripts(tmp_path)["pristine"]
    clean = reference.flip_labels()
    dataset = upstream.poisoned(script, clean, classes=4, fraction=0.5)
    label, index = flipped_labels(
        clean,
        num_classes=4,
        fraction=0.5,
        seed=vision_reference.SEED,
    )
    assert torch.equal(dataset.labels, label)
    assert torch.equal(dataset.poison_idx, index)
    assert [dataset[i][1] for i in range(len(dataset))] == label.tolist()


def test_main_is_traced_update_by_update(tmp_path: Path) -> None:
    script = scripts(tmp_path)["pristine"]
    trace = upstream.trace_main(
        script,
        upstream.lockstep_argv(),
        build=upstream.linear_model,
        split=vision_reference.synthetic_split(),
    )
    # The whole split is one batch: one update in each of two epochs.
    assert trace.epochs == [0, 1]
    assert trace.rates[0] == 0.1
    assert trace.rates[1] < 0.1
    assert trace.t_max == upstream.LOCKSTEP_EPOCHS
    assert len(trace.states) == len(trace.losses) == 2
    assert len(trace.results) == 2
    (dataset,) = trace.poisoned
    assert int(dataset.is_poison.sum()) == 3
    assert trace.evaluated
    assert "TEST_METRICS" in trace.printed


def test_a_run_restores_every_generator_and_leaves_python_and_numpy_unused(
    tmp_path: Path,
) -> None:
    script = scripts(tmp_path)["pristine"]
    torch_state = torch.get_rng_state()
    before = upstream.generator_states()
    trace = upstream.trace_main(
        script,
        upstream.lockstep_argv(),
        build=upstream.linear_model,
        split=vision_reference.synthetic_split(),
    )
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert upstream.generator_states() == before
    assert trace.generators == upstream.seeded_generator_states()
    random.seed(0)
    np.random.seed(0)  # noqa: NPY002 -- Moves the generator the comparison reads.
    assert upstream.generator_states() != upstream.seeded_generator_states()


def test_a_stopped_run_records_the_initial_state_and_no_update(tmp_path: Path) -> None:
    script = scripts(tmp_path)["pristine"]
    trace = upstream.trace_main(
        script,
        upstream.recipe_argv("resnet20-cifar10"),
        build=upstream.full_model,
        split=vision_reference.synthetic_split(),
        serial=True,
        stop=True,
    )
    assert not trace.losses
    assert trace.settings["weight_decay"] == 5e-4
    assert list(trace.initial) == ["conv.weight", "conv.bias", "fc.weight", "fc.bias"]
    assert [loader["num_workers"] for loader in trace.loaders] == [4, 4]


def test_the_lockstep_record_has_the_ports_keys(tmp_path: Path) -> None:
    record, states = upstream.upstream_lockstep(
        scripts(tmp_path)["bootstrap"],
        "bootstrap",
    )
    stored = read_tensors(DATA / "reference.pt")
    assert record.keys() == {k for k in stored if k.startswith("lockstep/bootstrap/")}
    assert len(states) == 2


def test_the_recipe_arguments_are_the_settings_scripts() -> None:
    argv = upstream.recipe_argv("mobilenetv2-fmnist")
    assert argv[:4] == ["--arch", "mobilenetv2", "--dataset", "fmnist"]
    assert argv[argv.index("--poison-fraction") + 1] == "0.15"
    shrunk = upstream.lockstep_argv()
    assert shrunk[shrunk.index("--epochs") + 1] == str(upstream.LOCKSTEP_EPOCHS)


class _Pixels:
    """A transform that returns a PIL RGB image's pixels channel first."""

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Return the ``(3, H, W)`` uint8 pixels, refusing any other mode."""
        if image.mode != "RGB":
            raise ValueError(f"Expected an RGB image; got {image.mode}.")
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
