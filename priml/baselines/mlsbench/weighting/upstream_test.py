"""Tests for the runs of the ``cv-sample-weighting`` script, against a stand-in clone.

The stand-in has the script's interface with a tiny network, the script's own
long-tail cut and loaders, and uniform weights; its baselines' edits replace
that one function. ``write_clone`` is shared with the capture's tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pickle

from torchvision.datasets import cifar

import numpy as np
import pytest
import torch

from priml.baselines.mlsbench.pytorch_vision import reference as vision
from priml.baselines.mlsbench.pytorch_vision.upstream_test import SCRIPT as LOSS_SCRIPT
from priml.baselines.mlsbench.weighting import reference, upstream


if TYPE_CHECKING:
    from pathlib import Path

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
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    full = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=False, transform=train_transform
    )
    test = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=False, transform=test_transform
    )
    train, counts = create_imbalanced_cifar(full, imbalance_ratio, 100, seed)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
        100,
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
    model = ResNet(BasicBlock, [1, 1, 1], classes)
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
    + LOSS_SCRIPT.split("def compute_loss")[0]
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
    best = 100.0 * max(vision.SCRIPTED_CORRECT) / vision.SCRIPTED_TOTAL
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
    assert record["lockstep/uniform/loss"].shape == (vision.STEPS,)
    assert record["lockstep/uniform/probe"].shape == (vision.STEPS, 2, vision.CLASSES)
    assert len(states) == vision.STEPS
    assert not torch.equal(states[0]["fc.weight"], states[1]["fc.weight"])


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
    # torchvision checks each file against the published checksums of the real
    # split, which six blank images cannot match.
    monkeypatch.setattr(cifar, "check_integrity", intact)
    pristine = scripts(tmp_path)["uniform"]
    indices, counts, label = upstream.real_subset(pristine, tmp_path / "cifar")
    assert label.tolist() == [2, 0, 1, 0, 1, 0]
    assert counts[:3].tolist() == [3.0, 2.0, 2.0]
    assert len(indices) == 3 + 2 + 1


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
    assert record.keys() == reference.port_record(tmp_path).keys()


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
