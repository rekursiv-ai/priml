"""Tests for the runs of the ``cv-data-augmentation`` script, against a stand-in clone.

The stand-in has the script's interface with a tiny network. Its pristine
``build_train_transform`` is the real one; its baselines' edits each replace
that one function with a different pipeline, so a capture can tell them apart.
``write_clone`` is shared with the capture's tests.

The script it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c/harbor/tasks-docker/mls-bench__cv-data-augmentation/tests/meta/pristine/pytorch-vision/custom_augment.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from PIL import Image
from torch.utils import data
from torchvision.transforms import autoaugment

import torch

from priml.baselines.mlsbench.dl.augmentation import upstream
from priml.baselines.mlsbench.dl.vision import (
    reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.upstream_test import SCRIPT as LOSS_SCRIPT


if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from priml.baselines.mlsbench.dl.vision.upstream import Batches

TRANSFORM: Final = """\
def build_train_transform(config):
    return transforms.Compose([
        transforms.RandomCrop(config['img_size'], padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config['mean'], config['std']),
    ])
"""
"""The stand-in's editable function, which the baselines' edits replace."""

LOADERS: Final = """\
class MobileNetV2(nn.Module):
    CFG = [[1, 16, 1, 1]]

    def __init__(self, num_classes=10):
        super().__init__()
        width = self.CFG[-1][1]
        self.conv = nn.Conv2d(3, width, 3, padding=1)
        self.fc = nn.Linear(width, num_classes)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.conv(x)), 1).flatten(1))


def build_model(arch, num_classes):
    if arch == 'mobilenetv2':
        return MobileNetV2(num_classes)
    return ResNet(BasicBlock, [1, 1, 1], num_classes)


STATS = {
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), 'CIFAR10'),
    'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), 'CIFAR100'),
    'fmnist': ((0.2860,) * 3, (0.3530,) * 3, 'FashionMNIST'),
}


def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
    mean, std, name = STATS[dataset]
    config = {'img_size': 32, 'mean': mean, 'std': std, 'dataset': dataset}
    train_transform = build_train_transform(config)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    if dataset == 'fmnist':
        repeat = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        steps = [transforms.Resize(32), *train_transform.transforms]
        at = next(i for i, t in enumerate(steps) if isinstance(t, transforms.ToTensor))
        steps.insert(at + 1, repeat)
        train_transform = transforms.Compose(steps)
        test_transform = transforms.Compose([
            transforms.Resize(32), transforms.ToTensor(), repeat,
            transforms.Normalize(mean, std),
        ])
    Dataset = vars(torchvision.datasets)[name]
    train = Dataset(
        root=data_root, train=True, download=False, transform=train_transform
    )
    test = Dataset(
        root=data_root, train=False, download=False, transform=test_transform
    )
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
        100,
    )
"""

SCRIPT: Final = (
    LOSS_SCRIPT.replace(
        "def compute_loss(logits, targets, config):\n"
        "    return F.cross_entropy(logits, targets)\n",
        TRANSFORM,
    )
    .replace(
        LOSS_SCRIPT[
            LOSS_SCRIPT.index("def get_dataloaders") : LOSS_SCRIPT.index(
                "def train_epoch",
            )
        ],
        LOADERS + "\n\n",
    )
    .replace(
        "def train_epoch(model, loader, optimizer, device, config):",
        "def train_epoch(model, loader, criterion, optimizer, device):",
    )
    .replace(
        "model = ResNet(BasicBlock, [1, 1, 1], classes)",
        "model = build_model(args.arch, classes)",
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
"""A stand-in for the ``cv-data-augmentation`` script."""

BODIES: Final = {
    "cutout": "transforms.RandomErasing(p=1.0, scale=(0.1, 0.1), ratio=(1.0, 1.0))",
    "randaugment": "transforms.RandomInvert(p=1.0)",
    "trivialaugment": "transforms.RandomPosterize(bits=2, p=1.0)",
}
"""What each baseline's edit adds to the stand-in pipeline, at its own position."""


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
    first = SCRIPT[: SCRIPT.index(TRANSFORM)].count("\n") + 1
    last = first + TRANSFORM.count("\n") - 1
    for name, body in BODIES.items():
        anchor = (
            "        transforms.Normalize("
            if name == "cutout"
            else "        transforms.RandomCrop("
        )
        content = TRANSFORM.replace(anchor, f"        {body},\n{anchor}")
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': {first}, 'end_line': {last}, "
            f"'content': {content!r}}}]\n",
        )


def steps(pipeline: object) -> list[str]:
    """Name the transforms of one of the script's ``Compose`` pipelines, in order."""
    return [
        type(t).__name__ for t in cast("list[object]", vars(pipeline)["transforms"])
    ]


def scripts(tmp_path: Path) -> dict[str, vision_upstream.Upstream]:
    """Write the stand-in clone and import every variant of its script."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return upstream.variants(root, scratch)


def test_each_variant_builds_its_baselines_pipeline(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == ["pad_crop_flip", *upstream.EDITED]
    names = {
        name: steps(upstream.pipelines(script)[0]) for name, script in variants.items()
    }
    assert names["randaugment"][0] == "RandomInvert"
    assert names["cutout"][3] == "RandomErasing"
    assert steps(upstream.pipelines(variants["cutout"])[1]) == ["ToTensor", "Normalize"]


def test_a_pipeline_draws_from_the_seed_it_is_given(tmp_path: Path) -> None:
    train, _ = upstream.pipelines(scripts(tmp_path)["pad_crop_flip"])
    images = reference.fixed_images(3)
    first = upstream.upstream_pipeline(train, images, seed=1)
    assert first.shape == (3, 3, 32, 32)
    assert torch.equal(first, upstream.upstream_pipeline(train, images, seed=1))
    assert not torch.equal(first, upstream.upstream_pipeline(train, images, seed=2))


def test_the_census_counts_torchvisions_operations_by_sign() -> None:
    policy = autoaugment.TrivialAugmentWide()
    image = Image.new("RGB", (4, 3))
    with reference.seeded(0), upstream.op_census() as counts:
        for _ in range(200):
            policy(image)
    assert sum(counts.values()) == 200
    assert upstream.census_gaps(counts) == []
    assert upstream.census_gaps({}) == [
        *(f"{op}+" for op in upstream.OPS.values()),
        *(f"{op}-" for op in upstream.SIGNED),
    ]


def test_loaders_alternate_training_and_test_passes(tmp_path: Path) -> None:
    train, evaluation = upstream.upstream_batches(
        scripts(tmp_path)["cutout"],
        num_workers=0,
    )
    assert len(train) == len(evaluation) == reference.LOADER_PASSES
    labels = torch.cat([batch["label"] for batch in train[0]])
    assert labels.sort().values.tolist() == list(range(reference.LOADER_IMAGES))


def test_loaders_fork_their_workers_and_read_the_settings_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _LoaderCalls()
    original = data.DataLoader
    monkeypatch.setattr(vision_upstream, "upstream_batches", calls)
    script = cast("vision_upstream.Upstream", object())
    assert upstream.upstream_batches(script, dataset="fmnist") == ([], [])
    upstream.upstream_batches(script, num_workers=0)
    (loader, dataset, four), (_, default, zero) = calls.seen
    assert (dataset, default, four, zero) == ("fmnist", "cifar100", 4, 0)
    assert isinstance(loader, vision_upstream.ForkedLoader)
    assert loader.loader is original
    assert data.DataLoader is original


def test_lockstep_trains_on_the_augmented_batch(tmp_path: Path) -> None:
    script = scripts(tmp_path)["randaugment"]
    trace = vision_upstream.trace_main(script)
    record, states = upstream.upstream_lockstep(script, "randaugment", trace)
    assert record["lockstep/randaugment/media"].shape == (reference.STEPS, 2, 3, 32, 32)
    assert record["lockstep/randaugment/loss"].shape == (reference.STEPS,)
    assert record["lockstep/randaugment/probe"].shape == (
        reference.STEPS,
        2,
        reference.CLASSES,
    )
    assert len(states) == reference.STEPS
    assert not torch.equal(states[0]["fc.weight"], states[1]["fc.weight"])


class _LoaderCalls:
    """Stands in for the vision loaders; keeps the loader class, dataset and workers."""

    def __init__(self) -> None:
        self.seen: list[tuple[object, str, int]] = []

    def __call__(
        self,
        script: object,
        *,
        dataset: str,
        num_workers: int,
    ) -> tuple[Batches, Batches]:
        del script
        self.seen.append((data.DataLoader, dataset, num_workers))
        return [], []


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
