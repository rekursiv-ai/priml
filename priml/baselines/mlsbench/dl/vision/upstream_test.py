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
from typing import TYPE_CHECKING, Final, cast
from unittest import mock

import functools

from torch.utils.data import DataLoader
from torchvision.transforms.functional import pil_to_tensor

import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference, upstream
from priml.baselines.mlsbench.dl.vision.data import (
    Cifar100Data,
    normalize,
)
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import ModuleType

    from torch import Tensor
    from torch.utils.data import Dataset

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop
    from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep


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


SETTINGS_SCRIPT: Final = (
    SCRIPT
    + """

class VGG(nn.Module):
    VGG16_CFG = []


class MobileNetV2(nn.Module):
    CFG = []


def build_model(arch, num_classes):
    return ResNet(BasicBlock, [1, 1, 1], num_classes)


_cifar_dataloaders = get_dataloaders


def get_dataloaders(dataset, data_root, batch_size=128, num_workers=4):
    if dataset != "fmnist":
        return _cifar_dataloaders(dataset, data_root, batch_size, num_workers)
    stats = ((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530))
    repeat = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        repeat,
        transforms.Normalize(*stats),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(32), transforms.ToTensor(), repeat, transforms.Normalize(*stats)
    ])
    train = torchvision.datasets.FashionMNIST(
        root=data_root, train=True, download=False, transform=train_transform
    )
    test = torchvision.datasets.FashionMNIST(
        root=data_root, train=False, download=False, transform=test_transform
    )
    return (
        torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=batch_size),
        10,
    )
"""
)
"""The stand-in with the reference's ``build_model`` and its Fashion-MNIST pipelines."""


def write_clone(root: Path) -> None:
    """Write the stand-in pristine script, with its settings, under ``root``.

    Args:
      root: Directory to hold the clone.

    """
    script = root / upstream.PRISTINE
    script.parent.mkdir(parents=True)
    script.write_text(SETTINGS_SCRIPT)


def stored_record(port: ModuleType) -> dict[str, Tensor]:
    """Read the golden a task's ``reference_test`` replays its port against.

    Args:
      port: The task's ``reference`` module.

    Returns:
      record: ``testdata/reference.pt`` beside it.

    """
    assert isinstance(port.__file__, str)
    return read_tensors(Path(port.__file__).parent / "testdata" / "reference.pt")


class StoredPart:
    """A port-side run answered from a golden: its keys under one prefix.

    ``prefix`` is formatted with the run's positional arguments, so
    ``"lockstep/{}/"`` answers ``lockstep_record(name)``; a string argument is
    split on ``/`` first, so ``"settings/{}/lockstep/{}/"`` answers
    ``setting_lockstep_record("<setting>/<variant>")``. With ``count``, a
    ``states`` list the caller passes receives that many placeholder states.
    """

    def __init__(
        self,
        golden: Mapping[str, Tensor],
        prefix: str,
        *,
        count: int = 0,
    ) -> None:
        self.golden = golden
        self.prefix = prefix
        self.count = count

    def __call__(
        self,
        *args: object,
        states: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Return the golden's keys under the formatted prefix."""
        if states is not None:
            states += [{"stored": torch.zeros(1)} for _ in range(self.count)]
        parts = [p for a in args for p in (a.split("/") if isinstance(a, str) else [a])]
        prefix = self.prefix.format(*parts)
        return {k: v for k, v in self.golden.items() if k.startswith(prefix)}


def stand_in_port(port: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer a task's port-side runs from its stored record, for a capture test.

    Each task's ``reference_test`` pins those runs against the record, so a
    capture test of a stand-in clone checks only what the capture compares:
    ``port_record`` returns the record, a lockstep or network run its stored
    values and placeholder states, and the full-size network is built at one
    block per stage, which names the stand-in's other shape as nine blocks do.

    Args:
      port: The task's ``reference`` module.
      monkeypatch: The test's.

    """
    stored = stored_record(port)
    lockstep = StoredPart(stored, "lockstep/{}/", count=reference.STEPS)
    monkeypatch.setattr(port, "port_record", lambda: dict(stored))
    monkeypatch.setattr(port, "lockstep_record", lockstep)
    if hasattr(port, "network_record"):
        network = StoredPart(stored, "network/{}/", count=1)
        monkeypatch.setattr(port, "network_record", network)
    shallow = functools.partial(_shallow_step, full_step=reference.full_step)
    monkeypatch.setattr(reference, "full_step", shallow)


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
    assert (scratch / "doubled.py").read_text() == SETTINGS_SCRIPT.replace(
        line,
        doubled,
        1,
    )


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


def test_the_fmnist_loaders_serve_the_grayscale_split_resized_into_three_channels(
    tmp_path: Path,
) -> None:
    train, evaluation = upstream.upstream_batches(script(tmp_path), dataset="fmnist")
    media, label = reference.prepared_gray_split()
    expected = normalize(
        media,
        mean=torch.tensor((0.2860,) * 3),
        std=torch.tensor((0.3530,) * 3),
    )
    assert torch.cat([b["label"] for b in evaluation[0]]).tolist() == label.tolist()
    assert torch.equal(torch.cat([b["media"] for b in evaluation[0]]), expected)
    assert train[0][0]["media"].shape == (2, 3, 32, 32)


def test_the_fmnist_pipelines_and_their_draws_are_seen(tmp_path: Path) -> None:
    pristine = script(tmp_path)
    record = upstream.upstream_fmnist_pipelines(pristine)
    assert record["transform/fmnist/test"].shape == (2, 3, 32, 32)
    counts = upstream.upstream_fmnist_sites(pristine)
    assert counts["randint@transforms/transforms.py:get_params"] == 2 * (
        reference.LOADER_PASSES * reference.LOADER_IMAGES + 2
    )


TINY_SCRIPT: Final = (
    SETTINGS_SCRIPT
    + """

class VGG(nn.Module):
    VGG16_CFG = [64, "M", 64, "M", 64, "M"]

    def __init__(self, num_classes=100):
        super().__init__()
        self.seen = list(self.VGG16_CFG)


class MobileNetV2(nn.Module):
    CFG = [[1, 16, 1, 1]]

    def __init__(self, num_classes=10):
        super().__init__()
        self.seen = [list(row) for row in self.CFG]


def build_model(arch, num_classes):
    if arch == "vgg16bn":
        return VGG(num_classes)
    if arch == "mobilenetv2":
        return MobileNetV2(num_classes)
    return ResNet(BasicBlock, [9, 9, 9], num_classes)
"""
)
"""The stand-in with architecture tables that record what they were built from."""


def test_the_tiny_model_patches_the_architecture_table_only_while_building(
    tmp_path: Path,
) -> None:
    path = tmp_path / "script.py"
    path.write_text(TINY_SCRIPT)
    pristine = upstream.load_script(path)
    vgg = upstream.tiny_model(pristine, "vgg16bn", reference.CLASSES)
    mobilenet = upstream.tiny_model(pristine, "mobilenetv2", reference.CLASSES)
    assert vars(vgg)["seen"] == reference.TINY_ARCHS["vgg16bn"]
    assert vars(mobilenet)["seen"] == reference.TINY_ARCHS["mobilenetv2"]
    assert vars(pristine.VGG)["VGG16_CFG"] == [64, "M", 64, "M", 64, "M"]
    resnet = upstream.tiny_model(pristine, "resnet56", reference.CLASSES)
    assert [name for name, _ in resnet.named_parameters()][:2] == [
        "conv.weight",
        "conv.bias",
    ]


@pytest.mark.parametrize(
    ("dataset", "fake", "split"),
    [
        ("cifar100", upstream.FakeCifar100, reference.synthetic_split),
        ("cifar10", upstream.FakeCifar100, reference.synthetic_split),
        ("fmnist", upstream.FakeFashionMNIST, reference.synthetic_gray_split),
    ],
)
def test_synthetic_datasets_stand_in_for_the_scripts_torchvision_class(
    tmp_path: Path,
    dataset: str,
    fake: type[upstream.FakeCifar100],
    split: Callable[[], tuple[Tensor, Tensor]],
) -> None:
    pristine = script(tmp_path)
    module = cast(dict[str, object], vars(pristine.torchvision.datasets))
    name = upstream.DATASETS[dataset]
    original = module[name]
    with upstream.synthetic_datasets(pristine, dataset):
        stand_in = module[name]
    assert stand_in is fake
    assert module[name] is original
    built = fake(root="unused", train=True, download=False, transform=pil_to_tensor)
    assert torch.equal(built.media, split()[0])
    image, label = built[1]
    assert torch.equal(image, split()[0][1])
    assert label == 1


def test_upstream_settings_traces_each_captured_setting_and_decodes_its_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The stand-in's CIFAR-10 loaders read CIFAR-100's class, so only the
    # Fashion-MNIST setting, whose pipeline is the reference's, runs here.
    monkeypatch.setattr(reference, "CAPTURED", ("mobilenetv2-fmnist",))
    traces, record = upstream.upstream_settings(script(tmp_path))
    assert list(traces) == ["mobilenetv2-fmnist"]
    prefix = "settings/mobilenetv2-fmnist/"
    loader = {k: v for k, v in record.items() if k.startswith(f"{prefix}loader/")}
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    assert set(loader) == {
        f"{prefix}loader/{k}" for k in ("label", "top", "left", "flip")
    }
    ours = reference.setting_loader_record(scratch, "mobilenetv2-fmnist")
    assert loader.keys() == ours.keys()
    assert (
        loader[f"{prefix}loader/label"].sort().values.tolist()
        == [list(range(reference.LOADER_IMAGES))] * reference.LOADER_PASSES
    )
    assert set(record) - set(loader) == {
        "transform/fmnist/train",
        "transform/fmnist/test",
        f"{prefix}draws",
        f"{prefix}train",
        f"{prefix}eval",
    }


def test_a_setting_is_traced_at_its_arch_dataset_and_classes(tmp_path: Path) -> None:
    trace, record = upstream.upstream_setting(script(tmp_path), "mobilenetv2-fmnist")
    assert trace.configs[0] == {"num_classes": 10, "epoch": 0, "total_epochs": 200}
    assert set(record) == {
        "settings/mobilenetv2-fmnist/draws",
        "settings/mobilenetv2-fmnist/train",
        "settings/mobilenetv2-fmnist/eval",
    }
    assert trace.state["fc.weight"].shape == (10, 4)
    assert record["settings/mobilenetv2-fmnist/train"].shape == (
        reference.PROBE_SHAPE[0],
        reference.CLASSES,
    )


LITERAL_SCRIPT: Final = """\
import torch.nn as nn


class CustomNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.width = num_features


def build():
    return [
        nn.Conv2d(3, 32, 3, padding=1, bias=False),
        nn.Conv2d(320, 1280, 1, bias=False),
        nn.BatchNorm2d(1280),
        nn.Linear(512, 512),
        nn.Linear(in_features=512, out_features=10),
        CustomNorm(32),
        nn.Conv2d(24, 40, 3),
    ]
"""
"""A stand-in whose layers carry the reference's hardcoded widths."""


def test_narrowed_widths_shrinks_each_hardcoded_width_while_building(
    tmp_path: Path,
) -> None:
    path = tmp_path / "literal.py"
    path.write_text(LITERAL_SCRIPT)
    module = upstream.load_script(path)
    build = cast("Callable[[], list[torch.nn.Module]]", vars(module)["build"])
    with upstream.narrowed_widths(module):
        built = build()
    stem, last, norm, hidden, head, custom, kept = built
    widths = reference.TINY_WIDTHS
    assert isinstance(stem, torch.nn.Conv2d)
    assert isinstance(last, torch.nn.Conv2d)
    assert isinstance(norm, torch.nn.BatchNorm2d)
    assert isinstance(hidden, torch.nn.Linear)
    assert isinstance(head, torch.nn.Linear)
    assert isinstance(kept, torch.nn.Conv2d)
    assert stem.weight.shape == (widths[32], 3, 3, 3)
    assert last.weight.shape == (widths[1280], widths[320], 1, 1)
    assert norm.num_features == widths[1280]
    assert hidden.weight.shape == (widths[512], widths[512])
    assert head.weight.shape == (10, widths[512])
    assert vars(custom)["width"] == widths[32]
    assert kept.weight.shape == (40, 24, 3, 3)
    after = build()[0]
    assert isinstance(after, torch.nn.Conv2d)
    assert after.out_channels == 32


def test_a_forked_loader_forks_only_a_loader_with_workers() -> None:
    built: list[dict[str, object]] = []

    def loader(
        dataset: Dataset[tuple[Tensor, Tensor]],
        **kwargs: object,
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        built.append(kwargs)
        return DataLoader(dataset)

    forked = upstream.ForkedLoader(loader)
    _ = forked(cast("Dataset[tuple[Tensor, Tensor]]", []), batch_size=2, num_workers=4)
    _ = forked(cast("Dataset[tuple[Tensor, Tensor]]", []), shuffle=True)
    assert built == [
        {
            "batch_size": 2,
            "shuffle": False,
            "num_workers": 4,
            "pin_memory": False,
            "multiprocessing_context": "fork",
        },
        {
            "batch_size": 1,
            "shuffle": True,
            "num_workers": 0,
            "pin_memory": False,
            "multiprocessing_context": None,
        },
    ]


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


def _shallow_step(
    factory: Callable[[], VisionTrainLoop],
    *,
    full_step: Callable[[Callable[[], VisionTrainLoop]], VisionTrainStep],
) -> VisionTrainStep:
    """Build ``factory``'s full-size step at one block per stage."""
    return full_step(functools.partial(_shallow, factory))


def _shallow(factory: Callable[[], VisionTrainLoop]) -> VisionTrainLoop:
    """Return ``factory``'s experiment at one block per stage."""
    config = factory()
    assert isinstance(config.step.model, ResNet.Config)
    config.step.model.blocks_per_stage = (1, 1, 1)
    return config


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
