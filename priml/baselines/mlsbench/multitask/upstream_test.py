"""Tests for the runs of the ``cv-multitask-loss`` script, against a stand-in clone.

The stand-in has the script's interface with a tiny two-head network and a
summing ``MultiTaskLoss``; its baselines' edits replace that class. The
``pcgrad`` stand-in has the real edit's walk interface and finds no leaves.
``write_clone`` is shared with the capture's tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from torch import Tensor

import torch

from priml.baselines.mlsbench.multitask import reference, upstream
from priml.baselines.mlsbench.pytorch_vision import reference as vision


if TYPE_CHECKING:
    from pathlib import Path

SCRIPT: Final = '''\
"""A stand-in for the cv-multitask-loss script."""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

CIFAR100_COARSE_MAP = [c % 20 for c in range(100)]


class BasicBlock(nn.Module):
    pass


class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.fc_fine = nn.Linear(4, 100)
        self.fc_coarse = nn.Linear(4, 20)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(F.relu(self.conv(x)), 1).flatten(1)
        return self.fc_fine(out), self.fc_coarse(out)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)


class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        return fine_loss + coarse_loss


class CIFAR100MultiTask(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.dataset = torchvision.datasets.CIFAR100(
            root=root, train=train, transform=transform, download=download,
        )
        self.coarse_map = torch.tensor(CIFAR100_COARSE_MAP, dtype=torch.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, fine_label = self.dataset[idx]
        return image, fine_label, self.coarse_map[fine_label].item()


def get_dataloaders(data_root, batch_size=128, num_workers=4):
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats),
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])
    train = CIFAR100MultiTask(data_root, train=True, transform=train_transform)
    test = CIFAR100MultiTask(data_root, train=False, transform=test_transform)
    return (
        torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        ),
        torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=num_workers),
    )


def train_epoch(model, loader, mtl_loss, optimizer, device, epoch, total_epochs):
    model.train()
    mtl_loss.train()
    for x, y, z in loader:
        optimizer.zero_grad()
        fine, coarse = model(x)
        loss = mtl_loss(F.cross_entropy(fine, y), F.cross_entropy(coarse, z), epoch, total_epochs)
        loss.backward()
        optimizer.step()
    return 0.0, 0.0


def evaluate(model, loader, device):
    return 50.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train_loader, test_loader = get_dataloaders("unused")
    model = ResNet(BasicBlock, [1, 1, 1])
    initialize_weights(model)
    mtl_loss = MultiTaskLoss(num_tasks=2)
    all_params = list(model.parameters()) + list(mtl_loss.parameters())
    optimizer = optim.SGD(all_params, lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = 0.0
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, mtl_loss, optimizer, None, epoch, args.epochs)
        best = max(best, evaluate(model, test_loader, None))
        scheduler.step()
    print(f"TEST_METRICS: test_acc={best:.2f}")
'''

LOSS: Final = """\
class MultiTaskLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()

    def forward(self, fine_loss, coarse_loss, epoch, total_epochs):
        return fine_loss + coarse_loss
"""
"""The stand-in's combination, which the baselines' edits replace."""

EDITED: Final = {
    "uncertainty": LOSS.replace(
        "super().__init__()\n",
        "super().__init__()\n        self.log_vars = nn.Parameter(torch.zeros(2))\n",
    ).replace(
        "return fine_loss + coarse_loss",
        "return fine_loss * torch.exp(-self.log_vars[0]) + coarse_loss",
    ),
    "dwa": LOSS.replace("fine_loss + coarse_loss", "fine_loss + 2 * coarse_loss"),
    "pcgrad": LOSS.replace(
        "super().__init__()\n",
        "super().__init__()\n        self._shared_params = None\n",
    )
    .replace(
        "    def forward",
        "    def _get_shared_params(self, loss):\n"
        "        if self._shared_params is None:\n"
        "            self._shared_params = []\n"
        "        return self._shared_params\n\n"
        "    def forward",
    )
    .replace(
        "return fine_loss + coarse_loss",
        "return len(self._get_shared_params(fine_loss)) * fine_loss + coarse_loss",
    ),
}
"""Each baseline's stand-in combination."""


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
    first = SCRIPT[: SCRIPT.index(LOSS)].count("\n") + 1
    last = first + LOSS.count("\n") - 1
    for name, content in EDITED.items():
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


def test_each_variant_carries_its_baselines_combination(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    assert list(variants) == ["sum", *upstream.EDITED]
    combination = variants["dwa"].MultiTaskLoss(2)
    loss = cast(Tensor, combination(torch.tensor(1.0), torch.tensor(2.0), 0, 1))
    assert loss.tolist() == 5.0


def test_main_is_traced_for_every_epoch_with_the_combinations_parameters(
    tmp_path: Path,
) -> None:
    variants = scripts(tmp_path)
    plain = upstream.trace_main(variants["sum"])
    learned = upstream.trace_main(variants["uncertainty"])
    assert plain.epochs == [(e, 200) for e in range(200)]
    assert plain.rates[0] == 0.1
    best = 100.0 * max(vision.SCRIPTED_CORRECT) / vision.SCRIPTED_TOTAL
    assert plain.best == float(f"{best:.2f}")
    assert plain.numels == [4 * 3 * 3 * 3, 4, 400, 100, 80, 20]
    assert learned.numels == [*plain.numels, 2]
    assert plain.settings["initial_lr"] == 0.1


def test_pin_shared_pins_only_a_walking_combination(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    w = torch.ones(2, requires_grad=True)
    loss = (w * 3).sum()
    plain = variants["sum"].MultiTaskLoss(2)
    assert upstream.pin_shared(plain, loss) == []
    pcgrad = variants["pcgrad"].MultiTaskLoss(2)
    assert upstream.pin_shared(pcgrad, loss) == []
    assert isinstance(pcgrad, upstream.Walker)
    assert pcgrad._shared_params == [w]


def test_a_walk_that_finds_more_than_the_ports_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    write_clone(root)
    # The edit's walk returns a leaf that is not in the loss's graph.
    edit = root / upstream.EDITS / "pcgrad.edit.py"
    edit.write_text(
        edit.read_text().replace(
            "self._shared_params = []",
            "self._shared_params = [torch.ones(1, requires_grad=True)]",
        ),
    )
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pcgrad = upstream.variants(root, scratch)["pcgrad"].MultiTaskLoss(2)
    report = upstream.pin_shared(pcgrad, (torch.ones(2, requires_grad=True) * 2).sum())
    assert report == ["pcgrad: the edit's walk is not a subsequence of the port's"]


def test_check_walk_reports_parameters_the_port_misses(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    model = variants["sum"].ResNet(variants["sum"].BasicBlock, [1, 1, 1])
    assert upstream.check_walk(variants["pcgrad"], model) == []
    model.add_module("extra", torch.nn.Linear(2, 3))
    (line,) = upstream.check_walk(variants["pcgrad"], model)
    assert "extra.weight" in line


def test_lockstep_records_two_updates_per_variant(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    trace = upstream.trace_main(variants["sum"])
    record, states = upstream.upstream_lockstep(variants["uncertainty"], "u", trace)
    assert record["lockstep/u/epochs"].tolist() == [
        [0, vision.EPOCHS],
        [1, vision.EPOCHS],
    ]
    assert record["lockstep/u/probe"].shape == (vision.STEPS, 2, 120)
    assert record["lockstep/u/mtl_params"].shape == (2,)
    assert len(states) == vision.STEPS
    assert not torch.equal(states[0]["fc_fine.weight"], states[1]["fc_fine.weight"])


def test_the_record_holds_every_key_the_port_writes(tmp_path: Path) -> None:
    variants = scripts(tmp_path)
    traces = {name: upstream.trace_main(s) for name, s in variants.items()}
    lockstep = {
        name: upstream.upstream_lockstep(s, name, traces["sum"])
        for name, s in variants.items()
    }
    record = upstream.upstream_record(variants, traces, lockstep)
    port = {k for k in reference.port_record() if not k.startswith("lockstep/")}
    assert {k for k in record if not k.startswith("lockstep/")} == port
    assert record["data/coarse_of_fine"].tolist()[:3] == [0, 1, 2]


def test_the_synthetic_split_serves_pixels_and_labels() -> None:
    dataset = upstream._FakeCifar100(
        root="unused",
        train=True,
        transform=lambda image: torch.tensor(image.size),
        download=False,
    )
    assert len(dataset) == vision.LOADER_IMAGES
    size, label = dataset[3]
    assert (size.tolist(), label) == ([32, 32], 3)


def test_batches_carry_the_superclass_of_each_label(tmp_path: Path) -> None:
    train, evaluation = upstream.upstream_batches(scripts(tmp_path)["sum"])
    assert len(train) == len(evaluation) == vision.LOADER_PASSES
    for batch in train[0] + evaluation[0]:
        assert torch.equal(batch["label_coarse"], batch["label"] % 20)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
