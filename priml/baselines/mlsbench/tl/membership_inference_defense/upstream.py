"""Runs of the pristine ``security-membership-inference-defense`` script, under the capture's pins.

Nothing is vendored: the driver and each baseline's edit of the defense are
imported from a clone of the benchmark. The driver imports its defense by
module name, so each variant's driver is imported with that variant's defense
in its place.

``main`` runs as written over a synthetic split. Recorders wrap what it calls:
the network builder, the loaders, the scheduler, the defense, the scoring and
the AUC. The loaders run without worker processes: an iterator takes its seed,
and its sampler takes another, from the global generator in the same order
either way, and with augmentation off a worker draws nothing. Nothing runs on
a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast
from unittest import mock

import contextlib
import io
import sys

from PIL import Image
from torch import Tensor, nn

import numpy as np
import torch

from priml.baselines.mlsbench.dl.vision import reference, upstream
from priml.baselines.mlsbench.tl.membership_inference_defense import (
    reference as port,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.data import SPLITS
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Sequence
    from types import ModuleType

    from torch.utils.data import DataLoader, Dataset

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__security-membership-inference-defense/tests"
    "/meta/pristine/pytorch-vision",
)
"""Directory of the unedited driver and defense, relative to the clone."""

DRIVER: Final = PRISTINE / "run_membership_defense.py"
"""The fixed driver: networks, data, training loop and attack."""

DEFENSE: Final = PRISTINE / "custom_membership_defense.py"
"""The editable defense."""

EDITS: Final = Path("tasks/security-membership-inference-defense/edits")
"""Directory of the task's baseline edits, relative to the clone."""

EDITED: Final = ("erm", "label_smoothing", "confidence_penalty", "relaxloss")
"""The published baselines, each named as its edit file and as its port's variant."""

DATASETS: Final = {
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "fmnist": "FashionMNIST",
}
"""The torchvision class each ``--dataset`` reads."""

SETTINGS: Final = {
    "resnet20-cifar10": ("resnet20", "cifar10"),
    "vgg16bn-cifar100": ("vgg16bn", "cifar100"),
    "mobilenetv2-fmnist": ("mobilenetv2", "fmnist"),
}
"""Each setting's ``--arch`` and ``--dataset``, as its ``scripts/*.sh`` passes them."""


class Defense(Protocol):
    """The editable ``MembershipDefense``."""

    def compute_loss(self, logits: Tensor, labels: Tensor, epoch: int) -> Tensor:
        """Return the scalar training loss."""
        raise NotImplementedError


class _Torchvision(Protocol):
    """The part of torchvision the driver reaches through its own import."""

    datasets: ModuleType


class Script(Protocol):
    """The part of the driver the capture uses."""

    MembershipDefense: Callable[[], Defense]
    torchvision: _Torchvision

    def build_model(self, arch: str, num_classes: int) -> nn.Module:
        """Build the network an ``--arch`` names."""
        raise NotImplementedError

    def initialize_weights(self, model: nn.Module) -> None:
        """Initialize in place."""

    def get_dataset(
        self,
        dataset: str,
        data_root: str,
        augment: bool = ...,
    ) -> tuple[Served, Served, int]:
        """Build the training and test splits with their transforms."""
        raise NotImplementedError

    def auc_from_scores(
        self,
        member_scores: np.ndarray,
        nonmember_scores: np.ndarray,
    ) -> float:
        """Return the Mann-Whitney AUC."""
        raise NotImplementedError

    def confidence_scores(
        self,
        model: nn.Module,
        loader: Iterable[tuple[Tensor, Tensor]],
        device: torch.device,
    ) -> tuple[np.ndarray, float]:
        """Return the largest softmax probability per image and the accuracy."""
        raise NotImplementedError

    def main(self) -> None:
        """Run the whole script."""


@dataclass(slots=True, kw_only=True)
class Trace:
    """What one run of the driver's ``main`` did, as its recorders saw it."""

    rates: list[float] = field(default_factory=list[float])
    """Learning rate of each update."""

    settings: dict[str, float] = field(default_factory=dict[str, float])
    """The optimizer's settings when the scheduler was built."""

    initial: dict[str, Tensor] = field(default_factory=dict[str, Tensor])
    """The network's whole state before the first update."""

    states: list[dict[str, Tensor]] = field(default_factory=list[dict[str, Tensor]])
    """The network's whole state after each update."""

    draws: Tensor = field(default_factory=lambda: torch.empty(0))
    """Where the generator stood before the first update."""

    losses: list[Tensor] = field(default_factory=list[Tensor])
    """Every update's loss."""

    logits: list[Tensor] = field(default_factory=list[Tensor])
    """Every update's logits."""

    labels: list[Tensor] = field(default_factory=list[Tensor])
    """Every update's targets: on the synthetic split, the images' indices."""

    epochs: list[int] = field(default_factory=list[int])
    """The epoch each update told the defense."""

    loaders: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    """Each loader's keyword arguments, in construction order."""

    subsets: list[list[int]] = field(default_factory=list[list[int]])
    """Each loader's subset of the training split, in construction order."""

    scores: list[tuple[np.ndarray, float]] = field(
        default_factory=list[tuple[np.ndarray, float]],
    )
    """Each scoring's scores and accuracy: test, members, non-members."""

    auc: float = 0.0
    """The attack's AUC."""

    printed: str = ""
    """Everything ``main`` printed."""


class Builder(Protocol):
    """Builds a network in place of the driver's ``build_model``, which it is handed."""

    def __call__(
        self,
        script: Script,
        build_model: Callable[[str, int], nn.Module],
        arch: str,
        classes: int,
        /,
    ) -> nn.Module:
        """Return the network, uninitialized."""
        raise NotImplementedError


def variants(root: Path, scratch: Path) -> dict[str, Script]:
    """Import the driver with the pristine defense and with each baseline's edit.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited defenses are written to.

    Returns:
      scripts: ``pristine`` and each of :data:`EDITED`.

    """
    defenses = {"pristine": upstream.load_script(root / DEFENSE)}
    for name in EDITED:
        defenses[name] = upstream.edited_script(
            root,
            name,
            scratch,
            pristine=DEFENSE,
            edits=EDITS,
        )
    return {name: driver(root, module) for name, module in defenses.items()}


def driver(root: Path, defense: object) -> Script:
    """Import the driver with ``defense`` as its ``custom_membership_defense``.

    Args:
      root: Clone of the benchmark.
      defense: The defense module.

    Returns:
      script: The driver, its ``MembershipDefense`` the one ``defense`` defines.

    """
    # Not ``mock.patch.dict(sys.modules)``: restoring the whole table would evict
    # torchvision, whose second import fails registering its kernels again.
    name = "custom_membership_defense"
    previous = sys.modules.get(name)
    sys.modules[name] = cast("ModuleType", defense)
    try:
        return cast(Script, cast(object, upstream.load_script(root / DRIVER)))
    finally:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous


def trace_main(
    script: Script,
    argv: Sequence[str],
    *,
    build: Builder,
    split: tuple[Tensor, Tensor],
) -> Trace:
    """Run ``main`` with ``argv`` over a synthetic split, recording what it did.

    Args:
      script: A driver from :func:`variants`.
      argv: The arguments after the script's name.
      build: Builds the network in place of ``build_model``.
      split: Uint8 images and labels both of the driver's splits read.

    Returns:
      trace: What the recorders saw.

    """
    trace = Trace()
    models: list[nn.Module] = []
    printed = io.StringIO()
    with (
        mock.patch.object(sys, "argv", ["script.py", *argv]),
        _recorded(script, trace, models, build=build, split=split),
        contextlib.redirect_stdout(printed),
        host_agnostic_numerics(),
    ):
        script.main()
    trace.printed = printed.getvalue()
    return trace


def initial_main(
    script: Script,
    name: str,
    *,
    count: int,
    build: Builder,
) -> tuple[Trace, nn.Module]:
    """Run ``main`` at a setting until its scheduler is built, before any image is read.

    Args:
      script: A driver.
      name: A key of :data:`SETTINGS`.
      count: Images in the training split, such as CIFAR-10's 50,000.
      build: Builds the network in place of ``build_model``.

    Returns:
      trace: The loaders' subsets of the training split, the optimizer's
        settings, the network's initial state and the generator's position.
      model: The network, as ``main`` initialized it.

    """
    trace = Trace()
    models: list[nn.Module] = []
    split = (torch.zeros(count, 1, 1, 1, dtype=torch.uint8), torch.zeros(count))
    with (
        mock.patch.object(sys, "argv", ["script.py", *recipe_argv(name)]),
        _recorded(script, trace, models, build=build, split=split, stop=True),
        contextlib.suppress(_Stopped),
        host_agnostic_numerics(),
    ):
        script.main()
    return trace, models[0]


def upstream_lockstep(
    script: Script,
    name: str,
) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
    """Run ``main`` on the shallowest ResNet over the synthetic split, recording it.

    Args:
      script: The driver carrying the defense under test.
      name: Which defense, for the record's keys.

    Returns:
      record: Keyed as the port's ``lockstep_record``.
      states: The network's whole state after each update.

    """
    with reference.seeded(reference.SEED):
        trace = trace_main(
            script,
            lockstep_argv("resnet20-cifar10"),
            build=tiny_model,
            split=reference.synthetic_split(),
        )
        draws = rng_fingerprint()
    (test, test_acc), (member, member_acc), (nonmember, nonmember_acc) = trace.scores
    gap = float(member.mean() - nonmember.mean())
    key = f"lockstep/{name}"
    record = {
        f"{key}/scores/{split}": torch.from_numpy(scores)
        for split, scores in zip(SPLITS, (test, member, nonmember), strict=True)
    }
    record |= {
        f"{key}/accuracy": torch.tensor(
            [test_acc, member_acc, nonmember_acc],
            dtype=torch.float64,
        ),
        f"{key}/metrics": torch.tensor(
            [test_acc, trace.auc, gap, test_acc - max(trace.auc - 0.5, 0.0)],
            dtype=torch.float64,
        ),
        f"{key}/loss": torch.stack(trace.losses),
        f"{key}/logits": torch.cat([x.reshape(-1) for x in trace.logits]),
        f"{key}/label": torch.cat(trace.labels),
        f"{key}/epoch": torch.tensor(trace.epochs),
        f"{key}/lr": torch.tensor(trace.rates, dtype=torch.float64),
        f"{key}/draws": draws,
    }
    return record, trace.states


def recipe_argv(name: str, *, epochs: int = 300) -> list[str]:
    """Return a setting's arguments as its ``scripts/*.sh`` passes them.

    Args:
      name: A key of :data:`SETTINGS`.
      epochs: The ``--epochs``.

    Returns:
      argv: The arguments after the script's name.

    """
    arch, dataset = SETTINGS[name]
    return [
        *("--arch", arch, "--dataset", dataset, "--data-root", "unused"),
        *("--epochs", str(epochs), "--batch-size", "128", "--lr", "0.1"),
        *("--momentum", "0.9", "--weight-decay", "1e-4"),
        *("--schedule-milestones", "150", "225", "--schedule-gamma", "0.1"),
        *("--seed", str(reference.SEED)),
    ]


def lockstep_argv(name: str) -> list[str]:
    """Return a setting's arguments shrunk to a lockstep run's horizon and batch.

    Args:
      name: A key of :data:`SETTINGS`.

    Returns:
      argv: :func:`recipe_argv` with the port's lockstep epochs, batch size
        and milestones.

    """
    argv = recipe_argv(name, epochs=port.LOCKSTEP_EPOCHS)
    argv[argv.index("--batch-size") + 1] = str(port.LOCKSTEP_BATCH)
    start = argv.index("--schedule-milestones") + 1
    argv[start : start + 2] = [str(m) for m in port.LOCKSTEP_MILESTONES]
    return argv


def linear_model(
    script: Script,
    build_model: Callable[[str, int], nn.Module],
    arch: str,
    classes: int,
    /,
) -> nn.Module:
    """Return a linear classifier of 32x32 RGB images: cheap to train for 300 epochs.

    Args:
      script: Unused.
      build_model: Unused.
      arch: Unused.
      classes: Logits.

    Returns:
      model: A flatten and a linear layer.

    """
    del script, build_model, arch
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, classes))


def tiny_model(
    script: Script,
    build_model: Callable[[str, int], nn.Module],
    arch: str,
    classes: int,
    /,
) -> nn.Module:
    """Build the driver's shallowest network of ``arch``, as the vision capture does.

    Args:
      script: A driver.
      build_model: The driver's own builder, which the vision capture shrinks.
      arch: The ``--arch``.
      classes: Logits.

    Returns:
      model: The network, as torch's constructors initialized it.

    """
    with mock.patch.object(script, "build_model", build_model):
        return upstream.tiny_model(cast(upstream.Upstream, script), arch, classes)


def full_model(
    script: Script,
    build_model: Callable[[str, int], nn.Module],
    arch: str,
    classes: int,
    /,
) -> nn.Module:
    """Build the driver's own network of ``arch``."""
    del script
    return build_model(arch, classes)


def pipelines(
    script: Script,
    images: Tensor,
    *,
    dataset: str,
    augment: bool,
) -> tuple[Tensor, Tensor]:
    """Run the driver's training pipeline, seeded, and its test pipeline on ``images``.

    Args:
      script: A driver.
      images: ``(B, C, H, W)`` uint8 images, one or three channels.
      dataset: The ``--dataset`` whose pipelines run.
      augment: The driver's ``augment`` flag.

    Returns:
      train: The training pipeline's ``(B, 3, 32, 32)`` images, its draws
        from the task's seed.
      test: The test pipeline's.

    """
    split = (images, torch.arange(len(images)))
    with mock.patch.object(
        script.torchvision.datasets,
        DATASETS[dataset],
        _Splits(split),
    ):
        train_set, test_set, _ = script.get_dataset(dataset, "unused", augment=augment)
    with reference.seeded(reference.SEED):
        train = torch.stack([train_set.transform(_pil(image)) for image in images])
    test = torch.stack([test_set.transform(_pil(image)) for image in images])
    return train, test


def defense_loss(
    script: Script,
    logits: Tensor,
    labels: Tensor,
    *,
    epoch: int,
) -> tuple[Tensor, Tensor]:
    """Run the driver's defense on ``logits``, forward and backward.

    Args:
      script: A driver.
      logits: ``(B, C)`` class scores.
      labels: ``(B,)`` targets.
      epoch: The epoch the defense is told.

    Returns:
      loss: The scalar.
      grad: Its gradient with respect to ``logits``.

    """
    leaf = logits.clone().requires_grad_()
    with host_agnostic_numerics():
        loss = script.MembershipDefense().compute_loss(leaf, labels, epoch)
        loss.backward()
    assert isinstance(leaf.grad, Tensor)
    return loss.detach(), leaf.grad


class Served:
    """A synthetic split, in the shape the driver expects of a torchvision dataset."""

    def __init__(
        self,
        split: tuple[Tensor, Tensor],
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        self.media, self.label = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self.transform(_pil(self.media[index])), int(self.label[index])


class _Splits:
    """Stands in for a torchvision dataset class: serves one split as both."""

    def __init__(self, split: tuple[Tensor, Tensor]) -> None:
        self.split = split

    def __call__(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: Callable[[Image.Image], Tensor],
    ) -> Served:
        del root, train, download
        return Served(self.split, transform)


class _Loaders:
    """Builds the driver's loaders without workers, noting what it asked for."""

    def __init__(
        self,
        trace: Trace,
        loader: type[DataLoader[tuple[Tensor, Tensor]]],
    ) -> None:
        self.trace = trace
        self.loader = loader

    def __call__(
        self,
        dataset: Dataset[tuple[Tensor, Tensor]],
        **kwargs: object,
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        self.trace.loaders.append(dict(kwargs))
        if isinstance(dataset, torch.utils.data.Subset):
            self.trace.subsets.append([int(i) for i in dataset.indices])
        return self.loader(
            dataset,
            batch_size=cast(int, kwargs["batch_size"]),
            shuffle=cast(bool, kwargs["shuffle"]),
        )


class _Schedulers:
    """Builds the driver's scheduler, noting the optimizer and the network then."""

    def __init__(
        self,
        trace: Trace,
        models: list[nn.Module],
        scheduler: type[torch.optim.lr_scheduler.MultiStepLR],
        *,
        stop: bool,
    ) -> None:
        self.trace = trace
        self.models = models
        self.scheduler = scheduler
        self.stop = stop

    def __call__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        milestones: list[int],
        gamma: float,
    ) -> torch.optim.lr_scheduler.MultiStepLR:
        scheduler = self.scheduler(optimizer, milestones=milestones, gamma=gamma)
        group = cast(dict[str, object], optimizer.param_groups[0])
        self.trace.settings = {
            name: float(cast(float, group[name])) for name in reference.HYPERPARAMETERS
        }
        self.trace.initial = {
            k: v.clone() for k, v in self.models[0].state_dict().items()
        }
        self.trace.draws = rng_fingerprint()
        if self.stop:
            raise _Stopped
        _ = optimizer.register_step_post_hook(_UpdateHook(self.trace, self.models[0]))
        return scheduler


class _UpdateHook:
    """Notes every update's rate and the network's state after it."""

    def __init__(self, trace: Trace, model: nn.Module) -> None:
        self.trace = trace
        self.model = model

    def __call__(self, optimizer: torch.optim.Optimizer, *args: object) -> None:
        del args
        group = cast(dict[str, object], optimizer.param_groups[0])
        self.trace.rates.append(float(cast(float, group["lr"])))
        self.trace.states.append(
            {k: v.clone() for k, v in self.model.state_dict().items()},
        )


class _Defenses:
    """Builds the driver's defense wrapped in a recorder."""

    def __init__(self, trace: Trace, defense: Callable[[], Defense]) -> None:
        self.trace = trace
        self.defense = defense

    def __call__(self) -> _RecordedDefense:
        return _RecordedDefense(self.trace, self.defense())


class _RecordedDefense:
    """Forwards to a defense and keeps what each update gave and got."""

    def __init__(self, trace: Trace, defense: Defense) -> None:
        self.trace = trace
        self.defense = defense

    def compute_loss(self, logits: Tensor, labels: Tensor, epoch: int) -> Tensor:
        loss = self.defense.compute_loss(logits, labels, epoch)
        self.trace.losses.append(loss.detach().clone())
        self.trace.logits.append(logits.detach().clone())
        self.trace.labels.append(labels.clone())
        self.trace.epochs.append(epoch)
        return loss


class _Builds:
    """Builds the network in place of ``build_model`` and keeps it."""

    def __init__(self, script: Script, build: Builder, models: list[nn.Module]) -> None:
        self.script = script
        self.build_model = script.build_model
        self.build = build
        self.models = models

    def __call__(self, arch: str, num_classes: int) -> nn.Module:
        model = self.build(self.script, self.build_model, arch, num_classes)
        self.models.append(model)
        return model


class _Scores:
    """Forwards to the driver's scoring and keeps each result."""

    def __init__(
        self,
        trace: Trace,
        scores: Callable[..., tuple[np.ndarray, float]],
    ) -> None:
        self.trace = trace
        self.scores = scores

    def __call__(
        self,
        model: nn.Module,
        loader: Iterable[tuple[Tensor, Tensor]],
        device: torch.device,
    ) -> tuple[np.ndarray, float]:
        result = self.scores(model, loader, device)
        self.trace.scores.append(result)
        return result


class _Auc:
    """Forwards to the driver's AUC and keeps it."""

    def __init__(
        self,
        trace: Trace,
        auc: Callable[[np.ndarray, np.ndarray], float],
    ) -> None:
        self.trace = trace
        self.auc = auc

    def __call__(self, member: np.ndarray, nonmember: np.ndarray) -> float:
        self.trace.auc = self.auc(member, nonmember)
        return self.trace.auc


class _Stopped(Exception):  # noqa: N818 -- A control-flow signal, not an error.
    """Ends a run of ``main`` once its scheduler exists."""


@contextlib.contextmanager
def _recorded(
    script: Script,
    trace: Trace,
    models: list[nn.Module],
    *,
    build: Builder,
    split: tuple[Tensor, Tensor],
    stop: bool = False,
) -> Generator[None]:
    """Wrap what the driver's ``main`` calls in recorders, over ``split``."""
    splits = _Splits(split)
    datasets = script.torchvision.datasets
    with (
        mock.patch.object(datasets, "CIFAR10", splits),
        mock.patch.object(datasets, "CIFAR100", splits),
        mock.patch.object(datasets, "FashionMNIST", splits),
        mock.patch.object(
            torch.utils.data,
            "DataLoader",
            _Loaders(trace, torch.utils.data.DataLoader),
        ),
        mock.patch.object(
            torch.optim.lr_scheduler,
            "MultiStepLR",
            _Schedulers(
                trace,
                models,
                torch.optim.lr_scheduler.MultiStepLR,
                stop=stop,
            ),
        ),
        mock.patch.object(script, "build_model", _Builds(script, build, models)),
        mock.patch.object(
            script,
            "MembershipDefense",
            _Defenses(trace, script.MembershipDefense),
        ),
        mock.patch.object(
            script,
            "confidence_scores",
            _Scores(trace, script.confidence_scores),
        ),
        mock.patch.object(
            script,
            "auc_from_scores",
            _Auc(trace, script.auc_from_scores),
        ),
        upstream.hidden_accelerators(),
    ):
        yield


def _pil(image: Tensor) -> Image.Image:
    """Return a ``(C, H, W)`` uint8 image as PIL holds it: RGB, or ``L`` for one channel."""
    return Image.fromarray(image.permute(1, 2, 0).squeeze(-1).contiguous().numpy())
