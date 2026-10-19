"""Tests for the runs of the ``optimization-gradient-compression`` script, against a stand-in.

The stand-in has the script's interface: its own tiny network, loaders over
torchvision datasets (cropping with less padding than the script), a warmup cosine, an editable ``Compressor`` and a
``train`` that evaluates on the first, every tenth and the last epoch. Its
baselines' edits replace the ``Compressor`` class. ``write_clone`` is shared
with the capture's tests.

The script it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c/harbor/tasks-docker/mls-bench__optimization-gradient-compression/tests/meta/pristine/pytorch-vision/custom_compressor.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import functools

from configgle import PartialConfig

import torch

from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.opt.gradient_compression import (
    reference,
    upstream,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from torch import Tensor
    from torch.utils.data import DataLoader

    import pytest

    from priml.baselines.mlsbench.opt.gradient_compression.train_loop import (
        CompressionTrainLoop,
    )

COMPRESSOR: Final = """\
class Compressor:
    def __init__(self, compress_ratio=0.01):
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        return [tensor.clone()], tensor.shape

    def decompress(self, compressed_tensors, ctx):
        return compressed_tensors[0].view(ctx)
"""
"""The stand-in's pristine compressor, which the baselines' edits replace."""

SCRIPT: Final = (
    '''\
"""A stand-in for the optimization-gradient-compression script."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class BasicBlock(nn.Module):
    pass


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.linear = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.linear(F.adaptive_avg_pool2d(F.relu(self.conv(x)), 1).flatten(1))


def build_model(model_name, num_classes, device):
    return ResNet(BasicBlock, [1, 1, 1], num_classes).to(device)


def get_dataloaders(dataset_name, batch_size, num_workers=2):
    Dataset = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
    stats = (
        ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        if dataset_name == "cifar10"
        else ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    )
    train_set = Dataset("/unused", train=True, download=False, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]))
    test_set = Dataset("/unused", train=False, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ]))
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=256, shuffle=False),
        10 if dataset_name == "cifar10" else 100,
    )


'''
    + COMPRESSOR
    + """

def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs, base_lr, min_lr=0.0):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / (warmup_epochs + 1)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def evaluate(model, test_loader, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            total_loss += F.cross_entropy(outputs, labels, reduction="sum").item()
            correct += outputs.max(1)[1].eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total, total_loss / total


def train(args):
    torch.manual_seed(args.seed)
    train_loader, test_loader, num_classes = get_dataloaders(args.dataset, args.batch_size)
    model = build_model(args.model, num_classes, torch.device("cpu"))
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    compressor = Compressor(compress_ratio=args.compress_ratio)
    best_acc = 0.0
    for epoch in range(args.epochs):
        cosine_lr(optimizer, epoch, args.epochs, args.warmup_epochs, args.lr,
                  min_lr=args.lr * 0.01)
        model.train()
        assert len(train_loader) == 1
        for images, labels in train_loader:
            optimizer.zero_grad()
            F.cross_entropy(model(images), labels).backward()
            for name, param in model.named_parameters():
                compressed, ctx = compressor.compress(param.grad.data, name)
                param.grad.data = compressor.decompress(compressed, ctx)
            optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            best_acc = max(best_acc, evaluate(model, test_loader, None)[0])
    test_acc, test_loss = evaluate(model, test_loader, None)
    best_acc = max(best_acc, test_acc)
    print(f"TEST_METRICS test_acc={test_acc:.2f} best_acc={best_acc:.2f}")
"""
)
"""A stand-in for the ``optimization-gradient-compression`` script."""

CLASSES: Final = {
    "topk_ef": COMPRESSOR.replace(
        "        self.compress_ratio = compress_ratio\n",
        "        self.compress_ratio = compress_ratio\n        self.residuals = {}\n",
    ).replace(
        "        return [tensor.clone()], tensor.shape\n",
        "        self.residuals[name] = tensor / 2\n"
        "        return [tensor / 2], tensor.shape\n",
    ),
    "qsgd": COMPRESSOR.replace(
        "        return [tensor.clone()], tensor.shape\n",
        "        return [tensor * torch.rand_like(tensor)], tensor.shape\n",
    ),
    "signsgd": COMPRESSOR.replace("tensor.clone()", "tensor.sign()"),
}
"""The class each baseline's edit puts in place of the compressor."""


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
    first = SCRIPT[: SCRIPT.index(COMPRESSOR)].count("\n") + 1
    last = first + COMPRESSOR.count("\n") - 1
    for name, content in CLASSES.items():
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': {first}, 'end_line': {last}, "
            f"'content': {content!r}}}]\n",
        )


def scripts(tmp_path: Path) -> dict[str, upstream.Script]:
    """Write the stand-in clone and import every variant of its script."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return upstream.variants(root, scratch)


def test_each_variant_carries_its_baselines_compressor(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == ["none", *upstream.EDITED]
    none = upstream.upstream_compressor(variants["none"], "none")
    signs = upstream.upstream_compressor(variants["signsgd"], "signsgd")
    assert torch.equal(
        none["compressor/none/weight"][0],
        reference.fixed_gradients(0)["weight"],
    )
    assert torch.equal(
        signs["compressor/signsgd/bias"],
        none["compressor/none/bias"].sign(),
    )
    assert not any("residual" in key for key in none)


def test_a_stateful_compressor_records_its_residuals(tmp_path: Path) -> None:
    record = upstream.upstream_compressor(scripts(tmp_path)["topk_ef"], "topk_ef")
    last = reference.COMPRESSION_STEPS - 1
    assert torch.equal(
        record["compressor/topk_ef/residual/weight"],
        reference.fixed_gradients(last)["weight"] / 2,
    )


def test_a_random_compressor_draws_from_the_seeded_generator(tmp_path: Path) -> None:
    script = scripts(tmp_path)["qsgd"]
    first = upstream.upstream_compressor(script, "qsgd")
    again = upstream.upstream_compressor(script, "qsgd")
    assert torch.equal(first["compressor/qsgd/weight"], again["compressor/qsgd/weight"])


def test_lockstep_records_one_update_per_epoch(tmp_path: Path) -> None:
    record, states = upstream.upstream_lockstep(scripts(tmp_path)["none"], "none")
    epochs = reference.LOCKSTEP_EPOCHS
    assert record["lockstep/none/loss"].shape == (epochs,)
    assert record["lockstep/none/probe"].shape == (epochs, 2, vision_reference.CLASSES)
    assert len(states) == epochs
    assert not torch.equal(states[0]["conv.weight"], states[1]["conv.weight"])


def test_the_trace_sees_the_schedule_cadence_and_best_accuracy(tmp_path: Path) -> None:
    # Twelve epochs reach warmup's end, one periodic evaluation and the last.
    record = upstream.upstream_trace(scripts(tmp_path)["none"], epochs=12)
    assert record["schedule/exp000"][0].item() == 0.1 / 6
    assert record["schedule/exp004"][5].item() == 0.05
    assert record["eval/epochs"].tolist() == [1, 10, 12]
    evaluations = len(record["eval/epochs"])
    best = (
        100.0
        * max(vision_reference.SCRIPTED_CORRECT[:evaluations])
        / vision_reference.SCRIPTED_TOTAL
    )
    assert record["best/accuracy"].item() == float(f"{best:.2f}")
    assert record["optimizer/exp000"].shape == (len(reference.SGD_SETTINGS),)


def test_the_trace_hands_the_driver_each_settings_own_warmup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warmups: list[int] = []
    monkeypatch.setattr(
        upstream,
        "run_train",
        functools.partial(_record_warmup, warmups=warmups, run=upstream.run_train),
    )
    monkeypatch.setitem(reference.SETTINGS, "exp004", _three_warmup_epochs)
    upstream.upstream_trace(scripts(tmp_path)["none"], epochs=6)
    assert warmups == [5, 3, 5]


def _three_warmup_epochs() -> CompressionTrainLoop.Config:
    config = reference.SETTINGS["exp000"]()
    schedule = config.step.schedule
    assert isinstance(schedule, PartialConfig)
    schedule.warmup_epochs = 3
    return config


def _record_warmup(
    script: upstream.Script,
    name: str,
    *,
    warmups: list[int],
    run: Callable[..., upstream.Run],
    **kwargs: object,
) -> upstream.Run:
    warmup = kwargs["warmup"]
    assert isinstance(warmup, int)
    warmups.append(warmup)
    return run(script, name, **kwargs)


def test_metrics_run_the_scripts_evaluate(tmp_path: Path) -> None:
    record = upstream.upstream_metrics(scripts(tmp_path)["none"])
    assert record.keys() == {"metric/accuracy", "metric/loss"}
    assert record["metric/loss"].item() > 0


def test_the_network_record_matches_the_ports_keys(tmp_path: Path) -> None:
    record, state = upstream.upstream_network(scripts(tmp_path)["none"], "exp004")
    assert set(record) == {
        "draws/exp004",
        "forward/exp004/train",
        "forward/exp004/eval",
    }
    assert list(state) == ["conv.weight", "conv.bias", "linear.weight", "linear.bias"]


def test_loaders_and_transforms_read_the_synthetic_split(tmp_path: Path) -> None:
    script = scripts(tmp_path)["none"]
    train, evaluation = upstream.upstream_batches(script, "cifar100", num_workers=0)
    assert len(train) == len(evaluation) == vision_reference.LOADER_PASSES
    labels = torch.cat([batch["label"] for batch in evaluation[0]])
    assert labels.tolist() == list(range(vision_reference.LOADER_IMAGES))
    record = upstream.upstream_transforms(script)
    assert record["transform/exp000"].shape == reference.TEST_IMAGE_SHAPE


def test_batches_come_from_the_scripts_two_worker_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The script's own count: each worker seeds its own transforms, so the crops
    # a pass draws depend on it. Recording the request avoids spawning workers.
    script = scripts(tmp_path)["none"]
    requested: list[int] = []
    get_dataloaders = script.get_dataloaders

    def recording(
        dataset_name: str,
        batch_size: int,
        num_workers: int = 2,
    ) -> tuple[
        DataLoader[tuple[Tensor, Tensor]],
        DataLoader[tuple[Tensor, Tensor]],
        int,
    ]:
        requested.append(num_workers)
        return get_dataloaders(dataset_name, batch_size, num_workers=0)

    monkeypatch.setattr(script, "get_dataloaders", recording)
    train, evaluation = upstream.upstream_batches(script, "cifar10")
    assert requested == [2]
    # Four training batches of two; the test loader's batch of 256 takes all seven.
    assert [set(batch) for batch in train[0]] == [{"media", "label"}] * 4
    assert [set(batch) for batch in evaluation[0]] == [{"media", "label"}]


def test_the_sites_name_the_edit_not_its_scratch_directory(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    counts = upstream.upstream_sites(variants["none"], variants["qsgd"])
    # The stand-in draws for every parameter, the zero one included.
    draws = len(reference.GRADIENT_SHAPES) * reference.COMPRESSION_STEPS
    assert counts["rand_like@edits/qsgd.py:compress"] == draws
    assert (
        counts["randperm@data/sampler.py:__iter__"]
        == 2 * vision_reference.LOADER_PASSES
    )


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
