"""Tests for the runs of the reference script.

The script is a stand-in with the reference's interface and a different, tiny
network. These tests pin what the runs observe of it: the recorded rates, the
configurations and the batches. ``write_clone`` is shared with the capture's
tests, which need the same clone, and ``SCRIPT`` with the task packages, whose
stand-ins are edits of it.

The script it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c/harbor/tasks-docker/mls-bench__cv-classification-loss/tests/meta/pristine/pytorch-vision/custom_loss.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final
from unittest import mock

import torch

from priml.baselines.mlsbench.dl.vision import reference, upstream
from priml.baselines.mlsbench.dl.vision.data import (
    Cifar100Data,
    normalize,
)
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor
    from torch.utils.data import DataLoader


_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD / "testdata"

SCRIPT: Final = '''\
"""A stand-in for the reference script."""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


class BasicBlock(nn.Module):
    pass


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.conv(x)), 1).flatten(1))


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)


def compute_loss(logits, targets, config):
    return F.cross_entropy(logits, targets)


def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    train = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=False, transform=train_transform
    )
    test = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=False, transform=test_transform
    )
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
        100,
    )


def train_epoch(model, loader, optimizer, device, config):
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        compute_loss(model(x), y, config).backward()
        optimizer.step()
    return 0.0, 0.0


def evaluate(model, loader, device, config):
    return 0.0, 50.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch")
    parser.add_argument("--dataset")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train_loader, test_loader, classes = get_dataloaders(args.dataset, "unused")
    model = ResNet(BasicBlock, [1, 1, 1], classes)
    initialize_weights(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    config = {"num_classes": classes, "epoch": 0, "total_epochs": args.epochs}
    best = 0.0
    for epoch in range(args.epochs):
        config["epoch"] = epoch
        train_epoch(model, train_loader, optimizer, None, config)
        best = max(best, evaluate(model, test_loader, None, config)[1])
        scheduler.step()
    print(f"TEST_METRICS: test_acc={best:.2f}")
'''


def write_clone(root: Path) -> None:
    """Write the stand-in pristine script under ``root``.

    Args:
      root: Directory to hold the clone.

    """
    script = root / upstream.PRISTINE
    script.parent.mkdir(parents=True)
    script.write_text(SCRIPT)


def clone(tmp_path: Path) -> Path:
    """Write the stand-in clone and return its root."""
    root = tmp_path / "clone"
    write_clone(root)
    return root


def script(tmp_path: Path) -> upstream.Upstream:
    """Return the stand-in pristine script."""
    return upstream.load_script(clone(tmp_path) / upstream.PRISTINE)


class RecordingLoaders:
    """Forwards to a script's ``get_dataloaders`` and keeps each ``num_workers``.

    The stand-in builds its loaders without workers, so the count a run asks
    for is visible only here.
    """

    def __init__(self, script: upstream.Upstream) -> None:
        self.get_dataloaders = script.get_dataloaders
        self.workers: list[int] = []

    def __call__(
        self,
        dataset: str,
        data_root: str,
        *,
        batch_size: int,
        num_workers: int,
    ) -> tuple[
        DataLoader[tuple[Tensor, Tensor]],
        DataLoader[tuple[Tensor, Tensor]],
        int,
    ]:
        """Note ``num_workers`` and return the script's loaders."""
        self.workers.append(num_workers)
        return self.get_dataloaders(
            dataset,
            data_root,
            batch_size=batch_size,
            num_workers=num_workers,
        )


def requested_workers(
    run: Callable[[upstream.Upstream], object],
    tmp_path: Path,
) -> list[int]:
    """Return the ``num_workers`` of each loader ``run`` asks the stand-in for."""
    pristine = script(tmp_path)
    recording = RecordingLoaders(pristine)
    with mock.patch.object(pristine, "get_dataloaders", recording):
        _ = run(pristine)
    return recording.workers


def test_an_edit_replaces_exactly_the_lines_it_names(tmp_path: Path) -> None:
    root = clone(tmp_path)
    line = "    return F.cross_entropy(logits, targets)\n"
    doubled = line.replace(")\n", ") * 2\n")
    first = SCRIPT[: SCRIPT.index(line)].count("\n") + 1
    edits = Path("edits")
    (root / edits).mkdir()
    (root / edits / "doubled.edit.py").write_text(
        f"OPS = [{{'start_line': {first}, 'end_line': {first}, "
        f"'content': {doubled!r}}}]\n",
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    edited = upstream.edited_script(
        root,
        "doubled",
        scratch,
        pristine=upstream.PRISTINE,
        edits=edits,
    )
    logits, targets = reference.fixed_logits()
    config = {"num_classes": 4, "epoch": 0, "total_epochs": 1}
    plain = upstream.load_script(root / upstream.PRISTINE).compute_loss(
        logits,
        targets,
        config,
    )
    assert edited.compute_loss(logits, targets, config) == plain * 2
    assert (scratch / "doubled.py").read_text() == SCRIPT.replace(line, doubled, 1)


def test_the_reference_sees_a_cpu_only_machine() -> None:
    with upstream.hidden_accelerators():
        assert not torch.cuda.is_available()
        assert not torch.accelerator.is_available()


def test_main_is_traced_for_every_epoch(tmp_path: Path) -> None:
    trace = upstream.trace_main(script(tmp_path))
    assert len(trace.rates) == len(trace.configs) == 200
    assert trace.rates[0] == 0.1
    assert torch.equal(
        torch.tensor(trace.rates, dtype=torch.float64),
        reference.schedule_record()["schedule/lr"],
    )
    assert trace.configs[199] == {"num_classes": 100, "epoch": 199, "total_epochs": 200}
    assert trace.settings["momentum"] == 0.9
    assert trace.settings["weight_decay"] == 5e-4
    assert trace.settings["initial_lr"] == 0.1
    assert list(trace.state) == [
        "conv.weight",
        "conv.bias",
        "fc.weight",
        "fc.bias",
    ]
    assert trace.draws.shape == (8,)


def test_main_is_traced_for_any_setting(tmp_path: Path) -> None:
    trace = upstream.trace_main(
        script(tmp_path),
        arch="resnet20",
        dataset="cifar10",
        classes=10,
    )
    assert trace.configs[0] == {"num_classes": 10, "epoch": 0, "total_epochs": 200}


def test_batches_of_cifar10_read_the_synthetic_split(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    write_clone(root)
    path = root / upstream.PRISTINE
    path.write_text(SCRIPT.replace("datasets.CIFAR100(", "datasets.CIFAR10("))
    train, evaluation = upstream.upstream_batches(
        upstream.load_script(path),
        dataset="cifar10",
    )
    assert len(train) == len(evaluation) == reference.LOADER_PASSES
    labels = torch.cat([batch["label"] for batch in evaluation[0]])
    assert labels.tolist() == list(range(reference.LOADER_IMAGES))


def test_the_reported_accuracy_is_the_best_scripted_one(tmp_path: Path) -> None:
    trace = upstream.trace_main(script(tmp_path))
    best = 100.0 * max(reference.SCRIPTED_CORRECT) / reference.SCRIPTED_TOTAL
    assert trace.best == float(f"{best:.2f}")
    assert (
        trace.best != 100.0 * reference.SCRIPTED_CORRECT[-1] / reference.SCRIPTED_TOTAL
    )


def test_the_network_run_has_the_shape_the_port_gives_its_own(tmp_path: Path) -> None:
    record = upstream.upstream_network(script(tmp_path))
    assert record.keys() == reference.network_record().keys()
    assert record["network/train"].shape == (2, reference.CLASSES)
    assert record["network/draws"].shape == (8,)


def test_batches_alternate_training_and_test_passes(tmp_path: Path) -> None:
    train, evaluation = upstream.upstream_batches(script(tmp_path))
    config = Cifar100Data.Config()
    images = normalize(
        reference.synthetic_split()[0],
        mean=torch.tensor(config.mean),
        std=torch.tensor(config.std),
    )
    assert len(train) == len(evaluation) == reference.LOADER_PASSES
    for training, testing in zip(train, evaluation, strict=True):
        assert [len(batch["label"]) for batch in training] == [2, 2, 2, 1]
        labels = torch.cat([batch["label"] for batch in training])
        assert labels.sort().values.tolist() == list(range(reference.LOADER_IMAGES))
        assert torch.cat([batch["label"] for batch in testing]).tolist() == list(
            range(reference.LOADER_IMAGES),
        )
        assert torch.equal(torch.cat([batch["media"] for batch in testing]), images)


def test_batches_come_from_the_scripts_four_worker_loaders(tmp_path: Path) -> None:
    # The reference's own count: each worker seeds its own transforms, so the
    # crops a pass draws depend on it.
    assert requested_workers(upstream.upstream_batches, tmp_path) == [4]


def test_draws_are_counted_with_the_loaders_in_this_process(tmp_path: Path) -> None:
    # A worker process's draws never reach the counter, so the pipelines'
    # loaders and the counted passes both run without workers.
    assert requested_workers(upstream.upstream_sites, tmp_path) == [0, 0]


def test_pipelines_normalize_two_fixed_images(tmp_path: Path) -> None:
    record = upstream.upstream_pipelines(script(tmp_path))
    config = Cifar100Data.Config()
    expected = normalize(
        reference.fixed_images(2),
        mean=torch.tensor(config.mean),
        std=torch.tensor(config.std),
    )
    assert torch.equal(record["transform/test"], expected)
    assert record["transform/train"].shape == (2, 3, 32, 32)


def test_the_record_has_every_key_the_goldens_hold(tmp_path: Path) -> None:
    pristine = script(tmp_path)
    trace = upstream.trace_main(pristine)
    batches = upstream.upstream_batches(pristine)
    record = upstream.upstream_record(pristine, trace, batches[0])
    stored = read_tensors(_DATA / "reference.pt").keys() | {
        "transform/train",
        "transform/test",
    }
    assert record.keys() == stored
    assert record["schedule/lr"].shape == (200,)


def test_the_stand_in_draws_from_the_samplers_it_uses(tmp_path: Path) -> None:
    counts = upstream.upstream_sites(script(tmp_path))
    # The network's two layers: torch's default init, then the script's own.
    assert counts["uniform_@nn/init.py:kaiming_uniform_"] == 2
    assert counts["normal_@nn/init.py:kaiming_normal_"] == 2
    assert counts["random_@data/dataloader.py:__init__"] == 4
    # One permutation per pass and one more, sliced to nothing, for the remainder.
    assert counts["randperm@data/sampler.py:__iter__"] == 2 * reference.LOADER_PASSES
    assert counts["randint@transforms/transforms.py:get_params"] == 2 * (
        reference.LOADER_PASSES * reference.LOADER_IMAGES + 2
    )


def test_the_criterion_spy_is_cross_entropy_and_keeps_what_it_saw() -> None:
    spy = upstream.CriterionSpy()
    logits, targets = reference.fixed_logits()
    loss = spy(logits, targets)
    assert torch.equal(loss, torch.nn.functional.cross_entropy(logits, targets))
    weighted = upstream.CriterionSpy(
        torch.nn.CrossEntropyLoss(weight=torch.ones(4) * 2),
    )
    assert torch.equal(weighted(logits, targets), loss)
    assert [x.shape for x in spy.logits] == [logits.shape]
    assert spy.losses[0].requires_grad is False


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
