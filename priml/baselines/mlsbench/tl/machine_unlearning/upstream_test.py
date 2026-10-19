"""Tests for the runs of the ``security-machine-unlearning`` harness, against a stand-in clone.

The stand-in harness has the real one's interface, loaders included, with a
two-layer network; its method and each baseline's edit replace one region of
the method file as the real edits do. ``write_clone`` is shared with the
capture's tests.

The harness it stands in for, whose repository publishes no license:
https://github.com/Imbernoulli/MLS-Bench/blob/80cf5c5cc8f70fb1b09f185e59ebf5eca04f610c/harbor/tasks-docker/mls-bench__security-machine-unlearning/tests/meta/pristine/pytorch-vision/bench/unlearning/run_unlearning.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final
from unittest import mock

import numpy as np
import pytest
import torch

from priml.baselines.mlsbench.dl.vision import reference as vision_reference
from priml.baselines.mlsbench.dl.vision.upstream import hidden_accelerators
from priml.baselines.mlsbench.tl.machine_unlearning import reference, upstream


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


DRIVER: Final = '''\
"""A stand-in for the harness."""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from custom_unlearning import UnlearningMethod


class BasicBlock(nn.Module):
    pass


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(F.relu(self.bn(self.conv(x))), 1).flatten(1))


class VGG(nn.Module):
    VGG16_CFG = []


class MobileNetV2(nn.Module):
    CFG = []


def build_model(arch, num_classes):
    return ResNet(BasicBlock, [1, 1, 1], num_classes)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)


def split_by_class(dataset, forget_class):
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    retain = [i for i, t in enumerate(targets) if int(t) != forget_class]
    forget = [i for i, t in enumerate(targets) if int(t) == forget_class]
    return Subset(dataset, retain), Subset(dataset, forget)


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def pretrain(model, loader, device, epochs, lr, momentum, weight_decay):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            F.cross_entropy(model(x), y).backward()
            optimizer.step()


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total, confs = 0, 0, []
    for x, y in loader:
        probs = torch.softmax(model(x), dim=1)
        correct += probs.argmax(1).eq(y).sum().item()
        total += y.size(0)
        confs.append(probs.max(1).values.numpy())
    return correct / max(total, 1), np.concatenate(confs) if confs else np.array([])


def auc_from_scores(member_scores, nonmember_scores):
    return float(np.mean(member_scores) > np.mean(nonmember_scores))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch")
    parser.add_argument("--dataset")
    parser.add_argument("--data-root")
    parser.add_argument("--forget-class", type=int, default=0)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--unlearn-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    train = torchvision.datasets.CIFAR10(root="", train=True, download=False, transform=transform)
    test = torchvision.datasets.CIFAR10(root="", train=False, download=False, transform=transform)
    full = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    retain_train, forget_train = split_by_class(train, args.forget_class)
    retain_test, forget_test = split_by_class(test, args.forget_class)
    retain = DataLoader(retain_train, batch_size=args.batch_size, shuffle=True)
    forget = DataLoader(forget_train, batch_size=args.batch_size, shuffle=True)
    model = build_model(args.arch, 10)
    initialize_weights(model)
    pretrain(model, full, None, args.pretrain_epochs, args.lr, args.momentum, args.weight_decay)
    method = UnlearningMethod()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    retain_iter, forget_iter = cycle_loader(retain), cycle_loader(forget)
    for epoch in range(args.unlearn_epochs):
        model.train()
        for step in range(max(1, min(len(retain), len(forget)))):
            method.unlearn_step(model, next(retain_iter), next(forget_iter), optimizer, step, epoch)
    retain_acc, _ = evaluate_accuracy(model, DataLoader(retain_test, batch_size=args.batch_size), None)
    forget_acc, test_scores = evaluate_accuracy(model, DataLoader(forget_test, batch_size=args.batch_size), None)
    _, train_scores = evaluate_accuracy(model, DataLoader(forget_train, batch_size=args.batch_size), None)
    print(f"TEST_METRICS auc={auc_from_scores(train_scores, test_scores)}")
'''
"""A stand-in for ``run_unlearning.py``."""

METHOD: Final = '''\
"""A stand-in method file."""

import torch
import torch.nn.functional as F

class UnlearningMethod:
    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        loss = F.cross_entropy(model(retain_batch[0]), retain_batch[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
'''
"""A stand-in for ``custom_unlearning.py``; its lines 6 to 12 are the editable class."""

TEACHER: Final = """\
import copy

class UnlearningMethod:
    def __init__(self):
        self.teacher = None

    def unlearn_step(self, model, retain_batch, forget_batch, optimizer, step, epoch):
        if self.teacher is None:
            self.teacher = copy.deepcopy(model)
        loss = -F.cross_entropy(model(forget_batch[0]), forget_batch[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
"""
"""What every stand-in edit puts in place of the class: a forget-ascent with a teacher."""

LAUNCH: Final = (
    "#!/bin/bash\ncd /workspace\npython pytorch-vision/bench/unlearning/run_unlearning.py \\\n"
    "    --arch resnet20 --dataset cifar10 \\\n    --data-root /data/cifar \\\n"
    "    --forget-class 0 \\\n    --pretrain-epochs 80 --unlearn-epochs 20 \\\n"
    "    --batch-size 128 \\\n    --lr 0.1 --momentum 0.9 --weight-decay 5e-4 \\\n"
    "    --seed ${SEED:-42}\n"
)
"""The setting's launch script, as the clone ships it."""


def write_clone(root: Path) -> None:
    """Write the stand-in harness, method, edits and launch script under ``root``.

    Args:
      root: Directory to hold the clone.

    """
    (root / upstream.ROOT).mkdir(parents=True)
    (root / upstream.DRIVER).write_text(DRIVER)
    (root / upstream.METHOD).write_text(METHOD)
    (root / upstream.EDITS).mkdir(parents=True)
    for name in upstream.EDITED:
        (root / upstream.EDITS / f"{name}.edit.py").write_text(
            f"OPS = [{{'start_line': 6, 'end_line': 12, 'content': {TEACHER!r}}}]\n",
        )
    (root / upstream.LAUNCH).parent.mkdir(parents=True)
    (root / upstream.LAUNCH).write_text(LAUNCH)


def loaded(tmp_path: Path) -> tuple[upstream.Driver, dict[str, upstream.MethodModule]]:
    """Write the stand-in clone and import its harness and every method variant."""
    root = tmp_path / "clone"
    write_clone(root)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return upstream.driver(root), upstream.variants(root, scratch)


def test_the_recipe_is_parsed_from_the_launch_script(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    write_clone(root)
    record = upstream.upstream_recipe(upstream.driver(root), root)
    assert torch.equal(
        record["recipe/values"],
        reference.recipe_record()["recipe/values"],
    )


def test_the_split_reads_targets_held_as_a_list_or_a_tensor(tmp_path: Path) -> None:
    script, _ = loaded(tmp_path)
    assert upstream.upstream_split(script).keys() == reference.split_record().keys()
    record = upstream.upstream_split(script)
    assert torch.equal(record["split/list/forget"], record["split/tensor/forget"])


def test_every_variant_is_loaded_and_the_edits_replace_the_class(
    tmp_path: Path,
) -> None:
    script, modules = loaded(tmp_path)
    assert list(modules) == list(reference.VARIANTS)
    _, pristine = upstream.upstream_method_lockstep(script, modules["pristine"], "p")
    record, edited = upstream.upstream_method_lockstep(script, modules["scrub"], "s")
    assert len(pristine) == len(reference.SCHEDULE)
    # The edit holds a teacher, recorded after the student's states.
    assert len(edited) == len(reference.SCHEDULE) + 1
    assert record["methods/s/probe"].shape == (
        len(reference.SCHEDULE),
        2,
        vision_reference.CLASSES,
    )


def test_main_runs_end_to_end_and_records_every_step(tmp_path: Path) -> None:
    script, modules = loaded(tmp_path)
    with hidden_accelerators():
        trace, untouched = upstream.upstream_main(script, modules["scrub"])
    assert untouched
    assert trace.initial.keys() == trace.pretrained.keys()
    assert not torch.equal(trace.initial["fc.weight"], trace.pretrained["fc.weight"])
    assert trace.pretrain_settings == (
        reference.TINY_PRETRAIN_EPOCHS,
        0.1,
        0.9,
        5e-4,
    )
    # Three forget images in a batch of four: one step per epoch.
    assert len(trace.steps) == reference.TINY_UNLEARN_EPOCHS
    assert [s.position for s in trace.steps] == [(0, 0), (1, 0)]
    assert all(s.batches[3].eq(0).all() for s in trace.steps)
    assert trace.steps[0].optimizer == (0.001, 0.9, 0.999, 1e-8, 0.0, 0.0)
    assert len(trace.confidence[0]) == reference.TRAIN_LABELS.count(0)
    assert len(trace.confidence[1]) == reference.TEST_LABELS.count(0)
    assert trace.auc in {0.0, 1.0}
    assert list(trace.teachers) == ["teacher"]
    assert trace.draws.shape == (8,)
    assert any(key.startswith("random_@data/sampler.py") for key in trace.sites)


def test_main_notices_a_drawn_numpy_generator(tmp_path: Path) -> None:
    script, modules = loaded(tmp_path)
    initialize = script.initialize_weights

    def drawing(model: torch.nn.Module) -> None:
        initialize(model)
        _ = np.random.random()  # noqa: NPY002 -- The legacy generator is what the check watches.

    with (
        hidden_accelerators(),
        mock.patch.object(script, "initialize_weights", drawing),
    ):
        _, untouched = upstream.upstream_main(script, modules["pristine"])
    assert not untouched


class _MaxStepsMethod:
    """A SCRUB-shaped method reporting its max-step epochs as its loss."""

    def __init__(self) -> None:
        self.msteps = 5

    def unlearn_step(  # noqa: PLR0917 -- The harness passes these six positionally.
        self,
        model: torch.nn.Module,
        retain_batch: tuple[torch.Tensor, torch.Tensor],
        forget_batch: tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
    ) -> dict[str, float]:
        """Report the max-step epochs in force as the loss."""
        del model, retain_batch, forget_batch, optimizer, step, epoch
        return {"loss": float(self.msteps)}


class _MaxStepsModule:
    """A method file whose class carries SCRUB's ``msteps``."""

    def __init__(self) -> None:
        self.UnlearningMethod: Callable[[], upstream.Method] = _MaxStepsMethod


def test_main_shortens_scrubs_max_steps_to_the_tiny_run(tmp_path: Path) -> None:
    script, _ = loaded(tmp_path)
    with hidden_accelerators():
        trace, _ = upstream.upstream_main(script, _MaxStepsModule())
    assert [s.loss for s in trace.steps] == [
        float(reference.TINY_SCRUB_MAX_EPOCHS),
    ] * reference.TINY_UNLEARN_EPOCHS


def test_evaluation_scores_the_fixed_network_and_scores(tmp_path: Path) -> None:
    script, _ = loaded(tmp_path)
    record = upstream.upstream_evaluation(script)
    assert record.keys() == reference.evaluation_record().keys()
    assert record["evaluation/confidence"].shape == (4,)
    assert record["evaluation/empty_confidence"].dtype == torch.float64
    # The stand-in's AUC compares mean scores: eighths 24/5 against 11/4.
    assert float(record["evaluation/auc"]) == 1.0


def test_the_full_network_is_built_by_the_harness_from_the_seed(tmp_path: Path) -> None:
    script, _ = loaded(tmp_path)
    key = "resnet20-cifar10-class0"
    record, state = upstream.upstream_full_size(script, key)
    assert set(record) == {
        f"full/{key}/draws",
        f"settings/{key}/train",
        f"settings/{key}/eval",
    }
    assert record[f"settings/{key}/train"].shape == (
        vision_reference.PROBE_SHAPE[0],
        vision_reference.CLASSES,
    )
    assert list(state) == [
        "conv.weight",
        "conv.bias",
        "bn.weight",
        "bn.bias",
        "bn.running_mean",
        "bn.running_var",
        "bn.num_batches_tracked",
        "fc.weight",
        "fc.bias",
    ]


def test_a_launch_script_for_another_setting_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "clone"
    write_clone(root)
    (root / upstream.LAUNCH).write_text(LAUNCH.replace("resnet20", "vgg16bn"))
    with pytest.raises(ValueError, match="vgg16bn on cifar10"):
        upstream.upstream_recipe(upstream.driver(root), root)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
