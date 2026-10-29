"""Tests for the runs of the driver, against a stand-in clone.

The stand-in has the driver's interface with a tiny network; its defense module
holds the pristine scorer, and one baseline's edit replaces it. ``write_clone``
is shared with the capture's tests.

The driver it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/b873944969c53b01e4ee6e843c77767b8c702485/harbor/tasks-docker/mls-bench__security-backdoor-defense/tests/meta/pristine/pytorch-vision/bench/backdoor/run_backdoor_defense.py
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Final

import sys

import numpy as np
import torch

from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.tl.backdoor_defense import reference, upstream


if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from priml.baselines.mlsbench.dl.vision.upstream import Upstream

DEFENSE: Final = '''\
"""A stand-in defense module."""

import numpy as np
import torch

class BackdoorDefense:
    def fit(self, features, labels, poison_fraction, **kwargs):
        self.labels = np.asarray(labels)

    def score_samples(self, features, logits):
        probs = torch.softmax(torch.as_tensor(logits), dim=1).cpu().numpy()
        return probs.max(axis=1)
'''
"""The stand-in's pristine defense: top softmax probability, as upstream's."""

EDITED: Final = (
    "class BackdoorDefense:\n"
    "    def fit(self, features, labels, poison_fraction, **kwargs):\n"
    "        pass\n"
    "    def score_samples(self, features, logits):\n"
    "        return -np.asarray(features)[:, 0]\n"
)
"""The class each baseline's edit puts in place of the defense."""

DRIVER: Final = '''\
"""A stand-in for the backdoor-defense driver."""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from custom_backdoor_defense import BackdoorDefense


class BasicBlock(nn.Module):
    pass


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4, num_classes)

    def forward_features(self, x):
        return F.adaptive_avg_pool2d(F.relu(self.conv(x)), 1).flatten(1)

    def forward(self, x):
        return self.fc(self.forward_features(x))


class VGG(ResNet):
    VGG16_CFG = [64, "M"]


class MobileNetV2(ResNet):
    CFG = [[1, 16, 1, 1]]


def build_model(arch, num_classes):
    return ResNet(BasicBlock, [1, 1, 1], num_classes)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)


def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=train_transform
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=test_transform
    )
    return (
        train_set,
        test_set,
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=batch_size),
        10,
    )


class PoisonedDataset(Dataset):
    def __init__(self, base_dataset, poison_indices, target_class, trigger, mean, std):
        self.base_dataset = base_dataset
        self.poison_set = set(poison_indices)
        self.target_class = target_class
        self.trigger = trigger
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.is_poison = [i in self.poison_set for i in range(len(base_dataset))]

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if idx in self.poison_set:
            return self._apply_trigger(image), self.target_class
        return image, label

    def _apply_trigger(self, image):
        img = image * self.std + self.mean
        img[:, -1:, -1:] = 1.0
        return (img.clamp(0.0, 1.0) - self.mean) / self.std


def select_poison_indices(dataset, poison_fraction, target_class):
    eligible = [i for i in range(len(dataset)) if dataset[i][1] != target_class]
    random.shuffle(eligible)
    return eligible[:max(1, int(len(dataset) * poison_fraction))]


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    return 0.0, 0.0


def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            correct += model(x).max(1)[1].eq(y).sum().item()
            total += x.size(0)
    return 0.0, 100.0 * correct / total


@torch.no_grad()
def collect_features_logits(model, loader, device):
    model.eval()
    features, logits, labels = [], [], []
    for x, y in loader:
        f = model.forward_features(x)
        features.append(f.numpy())
        logits.append(model.fc(f).numpy())
        labels.append(y.numpy())
    return np.concatenate(features), np.concatenate(logits), np.concatenate(labels)


@torch.no_grad()
def evaluate_asr(model, test_set, device, batch_size, target_class, trigger, mean, std):
    model.eval()
    hits, total = 0, 0
    for i in range(len(test_set)):
        image, label = test_set[i]
        if label == target_class:
            continue
        hits += int(model(image[None]).argmax(1).item() == target_class)
        total += 1
    return hits / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    for flag in ("--arch", "--dataset", "--trigger"):
        parser.add_argument(flag)
    parser.add_argument("--poison-fraction", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_set, test_set, _, test_loader, classes = get_dataloaders(args.dataset, "x", 4)
    indices = select_poison_indices(train_set, args.poison_fraction, 0)
    poisoned = PoisonedDataset(train_set, indices, 0, args.trigger, (0.5,) * 3, (0.25,) * 3)
    loader = DataLoader(poisoned, batch_size=4, shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    model = build_model(args.arch, classes)
    initialize_weights(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(args.epochs):
        train_epoch(model, loader, criterion, optimizer, None)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            evaluate(model, test_loader, criterion, None)
    analysis = DataLoader(poisoned, batch_size=4, shuffle=False, num_workers=0)
    features, logits, labels = collect_features_logits(model, analysis, None)
    defense = BackdoorDefense()
    defense.fit(features, labels, args.poison_fraction)
    scores = np.asarray(defense.score_samples(features, logits), dtype=np.float64)
    n_remove = max(1, int(len(scores) * args.poison_fraction * 1.5))
    remove_idx = np.argsort(scores)[-n_remove:]
    is_poison = np.array(poisoned.is_poison, dtype=bool)
    recall = float(is_poison[remove_idx].sum()) / max(float(is_poison.sum()), 1.0)
    print(f"Defense removed {int(is_poison[remove_idx].sum())}/{int(is_poison.sum())} "
          f"poison samples out of {n_remove} flagged "
          f"(recall={recall:.4f}, filter_ratio=1.5*eps)")
    keep = np.ones(len(scores), dtype=bool)
    keep[remove_idx] = False

    class FilteredDataset(Dataset):
        def __init__(self, base_dataset, indices):
            self.base_dataset = base_dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.base_dataset[self.indices[idx]]

    filtered = FilteredDataset(poisoned, np.nonzero(keep)[0].tolist())
    retrained = build_model(args.arch, classes)
    initialize_weights(retrained)
    optimizer = optim.SGD(retrained.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    train_epoch(retrained, DataLoader(filtered, batch_size=4), criterion, optimizer, None)
'''
"""A stand-in for the driver: the interface the capture calls, a tiny network."""


def write_clone(root: Path) -> None:
    """Write the stand-in driver, its defense module and one baseline's edit under ``root``.

    Args:
      root: Directory to hold the clone.

    """
    driver = root / upstream.DRIVER
    driver.parent.mkdir(parents=True)
    driver.write_text(DRIVER)
    (root / upstream.DEFENSE).write_text(DEFENSE)
    edits = root / upstream.EDITS
    edits.mkdir(parents=True)
    first = DEFENSE[: DEFENSE.index("class BackdoorDefense")].count("\n") + 1
    last = DEFENSE.count("\n")
    for name in reference.VARIANTS:
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': {first}, 'end_line': {last}, "
            f"'content': {EDITED!r}}}]\n",
        )


def scripts(tmp_path: Path) -> dict[str, Upstream]:
    """Write the stand-in clone and import every variant of its driver."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return upstream.variants(root, scratch)


def test_each_variant_imports_the_driver_with_its_own_defense(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == list(reference.VARIANTS)
    features, logits, labels = reference.defense_inputs()
    pristine = upstream.driver(variants["pristine"]).BackdoorDefense()
    edited = upstream.driver(variants["zscore_outlier"]).BackdoorDefense()
    pristine.fit(features, labels, 0.1)
    edited.fit(features, labels, 0.1)
    expected = torch.softmax(torch.as_tensor(logits), dim=1).numpy().max(axis=1)
    np.testing.assert_array_equal(pristine.score_samples(features, logits), expected)
    np.testing.assert_array_equal(
        edited.score_samples(features, logits),
        -features[:, 0],
    )


def test_main_is_traced_through_poisoning_filtering_and_retraining(
    tmp_path: Path,
) -> None:
    script = scripts(tmp_path)["pristine"]
    trace = upstream.trace_driver(script, reference.FIRST)
    victim, retrained = trace.networks
    assert len(victim.rates) == reference.EPOCHS_MAIN
    assert victim.evaluated == [1, 10, 20]
    assert victim.settings["momentum"] == 0.9
    assert sum(trace.is_poison) == 2
    assert victim.indices == list(range(reference.MAIN_IMAGES))
    # int(40 * 0.05 * 1.5) = 3 removed.
    assert len(retrained.indices) == reference.MAIN_IMAGES - 3
    assert trace.features.shape == (reference.MAIN_IMAGES, 4)
    assert trace.scores.dtype == torch.float64
    assert 0.0 <= trace.recall <= 1.0
    record = upstream.main_record(trace, reference.FIRST)
    assert record[f"main/{reference.FIRST}/is_poison"].sum() == 2
    assert upstream.filter_record({"pristine": trace})[
        "filter/pristine/kept"
    ].shape == (reference.MAIN_IMAGES - 3,)
    inputs = upstream.filter_inputs(trace)
    port = reference.filter_record(inputs)
    assert torch.equal(port["filter/pristine/scores"], trace.scores)
    assert port["filter/pristine/kept"].tolist() == retrained.indices


def test_main_reads_fashion_mnist_from_the_grayscale_split(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    write_clone(root)
    driver = root / upstream.DRIVER
    driver.write_text(
        DRIVER.replace(
            "torchvision.datasets.CIFAR10(",
            "torchvision.datasets.FashionMNIST(",
        ).replace(
            "transforms.ToTensor()",
            "transforms.Grayscale(3), transforms.ToTensor()",
        ),
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    script = upstream.variants(root, scratch)["pristine"]
    trace = upstream.trace_driver(script, "mobilenetv2-fmnist-badnets")
    # int(40 * 0.08) = 3 poisoned.
    assert sum(trace.is_poison) == 3


def test_every_defense_of_the_driver_is_scored_and_but_clustering_on_a_prefix(
    tmp_path: Path,
) -> None:
    record = upstream.defense_record(scripts(tmp_path))
    assert "defense/activation_clustering/subset" not in record
    assert record["defense/pristine/subset"].shape == (reference.DEFENSE_SUBSET,)
    assert record["defense/pristine/scores"].dtype == torch.float64


def test_a_setting_builds_its_full_network_and_runs_its_shallowest(
    tmp_path: Path,
) -> None:
    model, draws, record = upstream.setting_record(
        scripts(tmp_path)["pristine"],
        reference.FIRST,
    )
    assert record.keys() == {
        f"settings/{reference.FIRST}/{key}" for key in ("draws", "train", "eval")
    }
    assert draws.shape == (8,)
    assert not torch.equal(draws, record[f"settings/{reference.FIRST}/draws"])
    assert all(not torch.isnan(p).any() for p in model.parameters())


def test_the_draws_are_counted_in_this_process(tmp_path: Path) -> None:
    counts = upstream.upstream_sites(scripts(tmp_path)["pristine"])
    assert counts["normal_@nn/init.py:kaiming_normal_"] == 2
    assert any(key.startswith("randperm@data/sampler.py") for key in counts)


def test_importing_a_driver_restores_a_defense_module_already_imported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = ModuleType("custom_backdoor_defense")
    monkeypatch.setitem(sys.modules, "custom_backdoor_defense", sentinel)
    _ = scripts(tmp_path)
    assert sys.modules["custom_backdoor_defense"] is sentinel


def test_poisoning_and_triggers_are_run_through_the_driver(tmp_path: Path) -> None:
    script = scripts(tmp_path)["pristine"]
    poison = upstream.poison_record(script)
    assert poison[f"poison/{reference.FIRST}/indices"].shape == (2,)
    triggers = upstream.trigger_record(script)
    assert triggers["trigger/badnets"].shape == reference.trigger_media().shape


def test_the_tiny_record_has_the_ports_keys(tmp_path: Path) -> None:
    record = upstream.tiny_record(scripts(tmp_path)["pristine"], reference.FIRST)
    assert record.keys() == {
        f"tiny/{reference.FIRST}/{key}"
        for key in ("draws", "features", "logits", "labels", "clean_acc", "asr")
    }


def test_the_poisoned_loaders_trigger_the_picked_images(tmp_path: Path) -> None:
    train, analysis = upstream.poisoned_batches(
        scripts(tmp_path)["pristine"],
        reference.FIRST,
        num_workers=0,
    )
    assert len(train) == 2
    assert [len(b["label"]) for b in analysis[0]] == [2, 2, 2, 1]
    labels = torch.cat([b["label"] for b in analysis[0]])
    assert (labels == 0).sum() == 3


def test_the_lockstep_records_two_updates(tmp_path: Path) -> None:
    script = scripts(tmp_path)["pristine"]
    trace = upstream.trace_driver(script, reference.FIRST)
    record, states = upstream.lockstep(script, trace)
    assert record["lockstep/loss"].shape == (vision_reference.STEPS,)
    assert len(states) == vision_reference.STEPS


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
