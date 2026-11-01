"""Tests for the runs of the ``cv-sample-weighting`` script, against a stand-in clone.

The stand-in has the script's interface with a tiny network, the script's own
long-tail cut and loaders, and uniform weights; its baselines' edits replace
that one function. ``write_clone`` is shared with the capture's tests.

The script it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/b873944969c53b01e4ee6e843c77767b8c702485/harbor/tasks-docker/mls-bench__cv-sample-weighting/tests/meta/pristine/pytorch-vision/custom_weighting.py
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from unittest import mock

import pickle
import warnings

from torch import nn
from torchvision.datasets import cifar

import numpy as np
import torch

from priml.baselines.mlsbench.dl.vision import (
    reference as vision_reference,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.vision.upstream_test import (
    SCRIPT as VISION_SCRIPT,
)
from priml.baselines.mlsbench.dl.weighting import reference, upstream
from priml.testing.golden import read_tensors


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import Tensor

    import pytest

_CWD: Final = Path(__file__).resolve().parent

_DATA: Final = _CWD / "testdata"

SGD_SETTINGS: Final = ("lr", "momentum", "weight_decay")
"""The optimizer settings ``main`` passes to SGD."""

WEIGHTS: Final = (
    "def compute_class_weights(class_counts, num_classes, config):\n"
    "    return torch.ones(num_classes)\n"
)
"""The two lines of the stand-in's weighting, which the baselines' edits replace."""

LOADERS: Final = """\
def create_imbalanced_cifar(dataset, imbalance_ratio, num_classes, seed=42):
    targets = np.array(dataset.targets)
    n_max = np.sum(targets == 0)
    counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        mu = (1.0 / imbalance_ratio) ** (c / (num_classes - 1))
        counts[c] = max(int(n_max * mu), 1)
    selected = []
    rng = np.random.RandomState(seed)
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        rng.shuffle(class_indices)
        selected.extend(class_indices[:counts[c]])
    return Subset(dataset, selected), torch.tensor(counts, dtype=torch.float32)


def get_dataloaders(dataset_name, data_root, imbalance_ratio, batch_size=128, num_workers=4, seed=42):
    stats = (
        ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        if dataset_name == "cifar10"
        else ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    )
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    Dataset, classes = (
        (torchvision.datasets.CIFAR10, 10)
        if dataset_name == "cifar10"
        else (torchvision.datasets.CIFAR100, 100)
    )
    full = Dataset(root=data_root, train=True, download=False, transform=train_transform)
    test = Dataset(root=data_root, train=False, download=False, transform=test_transform)
    train, counts = create_imbalanced_cifar(full, imbalance_ratio, classes, seed)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
        classes,
        counts,
    )


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    return 0.0, 0.0


def evaluate(model, loader, criterion, device):
    return 0.0, 50.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch")
    parser.add_argument("--dataset")
    parser.add_argument("--imbalance-ratio", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train_loader, test_loader, classes, counts = get_dataloaders(
        args.dataset, "unused", args.imbalance_ratio, 128, seed=args.seed
    )
    model = build_model(args.arch, classes)
    initialize_weights(model)
    weights = compute_class_weights(counts, classes, {})
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = 0.0
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, criterion, optimizer, None)
        best = max(best, evaluate(model, test_loader, criterion, None)[1])
        scheduler.step()
    print(f"TEST_METRICS: test_acc={best:.2f}")
"""
"""The stand-in's cut, loaders, epoch functions and ``main``, the script's in shape."""

SCRIPT: Final = (
    "import numpy as np\nfrom torch.utils.data import DataLoader, Subset\n"
    + VISION_SCRIPT.split("def compute_loss")[0]
    + "class VGG(nn.Module):\n    VGG16_CFG = [64, 'M']\n\n\n"
    + "def build_model(arch, num_classes):\n"
    + "    return ResNet(BasicBlock, [1, 1, 1], num_classes)\n\n\n"
    + WEIGHTS
    + "\n\n"
    + LOADERS
)
"""A stand-in for the ``cv-sample-weighting`` script."""

BODIES: Final = {
    "inverse_freq": "class_counts / class_counts.sum()",
    "effective_number": "torch.ones(num_classes) * 2",
    "balanced_softmax": "torch.ones(num_classes) / 2",
}
"""The return expression each baseline's edit puts in the weighting."""


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
    first = SCRIPT.split(WEIGHTS)[0].count("\n") + 1
    for name, body in BODIES.items():
        content = WEIGHTS.replace("torch.ones(num_classes)", body)
        (edits / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': {first}, 'end_line': {first + 1}, "
            f"'content': {content!r}}}]\n",
        )


def scripts(tmp_path: Path) -> dict[str, upstream.Script]:
    """Write the stand-in clone and import every variant of its script."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return upstream.variants(root, scratch)


def write_cifar100(directory: Path) -> None:
    """Write raw CIFAR-100 pickles of six images, as torchvision reads them."""
    folder = directory / "cifar-100-python"
    folder.mkdir(parents=True)
    labels = [2, 0, 1, 0, 1, 0]
    data = np.zeros((len(labels), 3 * 32 * 32), dtype=np.uint8)
    for name in ("train", "test"):
        with (folder / name).open("wb") as file:
            pickle.dump({"data": data, "fine_labels": labels}, file)
    with (folder / "meta").open("wb") as file:
        pickle.dump({"fine_label_names": [str(c) for c in range(100)]}, file)


def intact(*_arguments: object) -> bool:
    """Report any file as matching its checksum."""
    return True


def test_main_reports_its_cut_weights_and_epochs(tmp_path: Path) -> None:
    trace = upstream.trace_main(scripts(tmp_path)["uniform"])
    counts, indices = reference.counts_record()
    assert torch.equal(trace.counts, counts["counts/cifar100lt"])
    assert torch.equal(trace.indices, indices)
    assert torch.equal(trace.weight, torch.ones(reference.CLASSES))
    assert len(trace.main.rates) == 200
    assert trace.main.settings["initial_lr"] == 0.1
    assert trace.main.draws.shape == (8,)
    best = (
        100.0 * max(vision_reference.SCRIPTED_CORRECT) / vision_reference.SCRIPTED_TOTAL
    )
    assert trace.main.best == float(f"{best:.2f}")


def test_each_variant_carries_its_baselines_weighting(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == ["uniform", *upstream.EDITED]
    record = upstream.upstream_weights(variants)
    counts = torch.tensor(reference.COUNTS)
    assert torch.equal(record["weights/uniform"], torch.ones(4))
    assert torch.equal(record["weights/inverse_freq"], counts / counts.sum())
    assert torch.equal(record["weights/effective_number/flat"], torch.full((4,), 2.0))
    assert record["loss/balanced_softmax/grad"].shape == (3, 4)


def test_lockstep_trains_with_the_weighted_criterion(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    trace = upstream.trace_main(variants["uniform"])
    record, states = upstream.upstream_lockstep(variants["uniform"], "uniform", trace)
    assert record["lockstep/uniform/loss"].shape == (vision_reference.STEPS,)
    assert record["lockstep/uniform/probe"].shape == (
        vision_reference.STEPS,
        2,
        vision_reference.CLASSES,
    )
    assert len(states) == vision_reference.STEPS
    assert not torch.equal(states[0]["fc.weight"], states[1]["fc.weight"])


def test_lockstep_builds_the_scored_settings_resnet_by_default(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    trace = upstream.trace_main(variants["uniform"])
    seen: list[tuple[str, int]] = []
    build = vision_upstream.tiny_model

    def tiny(script: vision_upstream.Upstream, arch: str, classes: int) -> nn.Module:
        seen.append((arch, classes))
        return build(script, arch, classes)

    with mock.patch.object(vision_upstream, "tiny_model", tiny):
        upstream.upstream_lockstep(variants["uniform"], "uniform", trace)
    assert seen == [("resnet32", vision_reference.CLASSES)]


def test_lockstep_builds_weighs_and_steps_as_main_does(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    trace = upstream.trace_main(variants["uniform"])
    script = variants["inverse_freq"]
    seen: dict[str, object] = {}
    build, weigh, train = (
        script.ResNet,
        script.compute_class_weights,
        script.train_epoch,
    )

    def resnet(block: type[nn.Module], blocks: list[int], classes: int) -> nn.Module:
        seen["resnet"] = (block, blocks, classes)
        return build(block, blocks, classes)

    def weights(counts: Tensor, classes: int, config: dict[str, object]) -> Tensor:
        seen["config"] = config
        return weigh(counts, classes, config)

    def epoch(
        model: nn.Module,
        loader: Iterable[tuple[Tensor, Tensor]],
        criterion: Callable[[Tensor, Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[float, float]:
        group = cast("dict[str, float]", optimizer.param_groups[0])
        seen.setdefault("settings", {k: group[k] for k in SGD_SETTINGS})
        seen["device"] = device
        return train(model, loader, criterion, optimizer, device)

    with (
        mock.patch.object(script, "ResNet", resnet),
        mock.patch.object(script, "compute_class_weights", weights),
        mock.patch.object(script, "train_epoch", epoch),
    ):
        record, _ = upstream.upstream_lockstep(script, "inverse_freq", trace)
    assert seen["resnet"] == (script.BasicBlock, [1, 1, 1], vision_reference.CLASSES)
    assert seen["config"] == upstream.CONFIG
    assert seen["settings"] == {k: trace.main.settings[k] for k in SGD_SETTINGS}
    assert seen["device"] == torch.device("cpu")
    counts = torch.tensor(reference.LOCKSTEP_COUNTS)
    for index in range(vision_reference.STEPS):
        assert torch.equal(
            record["lockstep/inverse_freq/loss"][index],
            nn.functional.cross_entropy(
                record["lockstep/inverse_freq/logits"][index],
                vision_reference.fixed_label(index),
                weight=counts / counts.sum(),
            ),
        )


def test_the_synthetic_cut_matches_the_port(tmp_path: Path) -> None:
    pristine = scripts(tmp_path)["uniform"]
    subset = upstream.upstream_subset(pristine)
    assert subset["subset/indices"].tolist() == (
        reference.subset_record()["subset/indices"].tolist()
    )
    train, evaluation = upstream.upstream_batches(pristine)
    assert [len(batch["label"]) for batch in train[0]] == [2] * 6
    assert len(evaluation[0]) == len(reference.long_tail_split()[1]) // 2
    decoded = reference.decode_indices(train)["loader/index"]
    assert torch.equal(
        decoded[0].sort().values,
        subset["subset/indices"].sort().values,
    )


def test_the_real_cut_reads_torchvision_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_cifar100(tmp_path / "cifar")
    # Torchvision checks each file against the published checksums of the real
    # split, which six blank images cannot match.
    monkeypatch.setattr(cifar, "check_integrity", intact)
    pristine = scripts(tmp_path)["uniform"]
    indices, counts, label = upstream.real_subset(pristine, tmp_path / "cifar")
    assert label.tolist() == [2, 0, 1, 0, 1, 0]
    assert counts[:3].tolist() == [3.0, 2.0, 2.0]
    assert len(indices) == 3 + 2 + 1


def test_a_vgg_setting_is_traced_per_variant_and_trained_two_updates(
    tmp_path: Path,
) -> None:
    variants = scripts(tmp_path)
    traces, record, lockstep = upstream.upstream_setting(
        variants,
        "vgg16bn-cifar100lt",
    )
    assert list(traces) == list(variants)
    prefix = "settings/vgg16bn-cifar100lt/"
    assert record[f"{prefix}schedule/lr"].shape == (200,)
    assert torch.equal(record[f"{prefix}weight/uniform"], torch.ones(reference.CLASSES))
    assert list(lockstep) == [f"vgg16bn-cifar100lt/{name}" for name in variants]
    assert record[f"{prefix}lockstep/uniform/loss"].shape == (vision_reference.STEPS,)
    # Ratio 50 over 100 classes: 500 images of class 0 down to 10 of class 99.
    assert record[f"{prefix}counts"][[0, -1]].tolist() == [500.0, 10.0]


def test_a_cifar10_setting_reads_cifar10_and_runs_no_lockstep(tmp_path: Path) -> None:
    traces, record, lockstep = upstream.upstream_setting(
        scripts(tmp_path),
        "resnet32-cifar10lt",
    )
    assert lockstep == {}
    assert record["settings/resnet32-cifar10lt/counts"].shape == (10,)
    assert traces["uniform"].indices.shape == (
        int(record["settings/resnet32-cifar10lt/counts"].sum()),
    )


def test_main_runs_without_a_warning(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        # The script's scheduler warns if it steps before the optimizer has.
        warnings.simplefilter("error")
        trace = upstream.trace_main(scripts(tmp_path)["uniform"])
    assert len(trace.main.rates) == 200


def test_the_record_holds_every_key_the_port_gives(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    pristine = variants["uniform"]
    trace = upstream.trace_main(pristine)
    lockstep = {
        name: upstream.upstream_lockstep(script, name, trace)
        for name, script in variants.items()
    }
    batches = upstream.upstream_batches(pristine)
    record = upstream.upstream_record(
        trace,
        upstream.upstream_weights(variants),
        lockstep,
        upstream.upstream_subset(pristine),
        batches[0],
    )
    assert record.keys() == read_tensors(_DATA / "reference.pt").keys()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
