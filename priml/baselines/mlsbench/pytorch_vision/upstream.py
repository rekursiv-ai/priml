"""Runs of the pristine MLS-Bench vision script, under the capture's pins.

Nothing here is vendored: the script is imported from a clone of the benchmark
and driven through its own functions. ``main`` runs as written with its data
loaders and two epoch functions swapped for recorders, so its seeding, model
build, initialization, optimizer, scheduler, loss configuration and
best-accuracy bookkeeping are observed rather than re-implemented. The data
pipeline runs through the script's own ``get_dataloaders`` over a synthetic
split. Nothing runs on a GPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib import machinery, util
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast
from unittest import mock

import contextlib
import io
import sys
import warnings

from PIL import Image
from torch import Tensor, nn
from treekle import from_plain

import torch

from priml.baselines.mlsbench.pytorch_vision import reference
from priml.baselines.mlsbench.pytorch_vision.data import Cifar100Data
from priml.baselines.mlsbench.pytorch_vision.draws import recorded_draws
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable
    from types import ModuleType

    from torch.utils.data import DataLoader

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__cv-classification-loss/tests/meta/pristine"
    "/pytorch-vision/custom_loss.py",
)
"""The unedited training script, relative to the clone."""

EDITS: Final = Path("tasks/cv-classification-loss/edits")
"""Directory of the task's baseline edits, relative to the clone."""

VARIANTS: Final = {
    "label_smoothing": "label_smoothing",
    "focal": "focal_loss",
    "poly": "poly_loss",
}
"""Maps each non-default loss of :data:`reference.LOSSES` to its edit file."""

type Batches = list[list[dict[str, Tensor]]]
"""Per pass, the batches of one loader in order."""

type Lockstep = tuple[dict[str, Tensor], list[dict[str, Tensor]]]
"""A lockstep run's record and the model's whole state after each update."""


class _Torchvision(Protocol):
    """The part of torchvision the reference script reaches through its own import."""

    datasets: ModuleType


class Upstream(Protocol):
    """The part of the reference script the capture uses."""

    BasicBlock: type[nn.Module]
    torchvision: _Torchvision

    ResNet: Callable[[type[nn.Module], list[int], int], nn.Module]

    def initialize_weights(self, model: nn.Module) -> None:
        """Initialize in place."""

    def compute_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        config: dict[str, int],
    ) -> Tensor:
        """Return the scalar training loss."""
        raise NotImplementedError

    def get_dataloaders(
        self,
        dataset: str,
        data_root: str,
        batch_size: int = ...,
        num_workers: int = ...,
    ) -> tuple[
        DataLoader[tuple[Tensor, Tensor]],
        DataLoader[tuple[Tensor, Tensor]],
        int,
    ]:
        """Build the loaders."""
        raise NotImplementedError

    def train_epoch(
        self,
        model: nn.Module,
        loader: Iterable[tuple[Tensor, Tensor]],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: dict[str, int],
    ) -> tuple[float, float]:
        """Train for one pass over ``loader``."""
        raise NotImplementedError

    def main(self) -> None:
        """Run the whole script."""


@dataclass(slots=True, kw_only=True)
class MainTrace:
    """What the reference's ``main`` did, as its replaced calls saw it."""

    rates: list[float] = field(default_factory=list[float])
    """Learning rate of each epoch."""

    configs: list[dict[str, int]] = field(default_factory=list[dict[str, int]])
    """What each epoch told the loss."""

    settings: dict[str, float] = field(default_factory=dict[str, float])
    """The optimizer's settings in the first epoch."""

    state: dict[str, Tensor] = field(default_factory=dict[str, Tensor])
    """The network's whole state before the first epoch."""

    draws: Tensor = field(default_factory=lambda: torch.empty(0))
    """Where the generator stood before the first epoch."""

    best: float = 0.0
    """The accuracy ``main`` reported."""


def load_script(path: Path) -> Upstream:
    """Import a reference training script from its file.

    Args:
      path: A ``.py`` file of the clone.

    Returns:
      script: The imported module, typed as the part this capture uses.

    """
    return cast(Upstream, cast(object, _load_module(path)))


def edited_script(root: Path, name: str, scratch: Path) -> Upstream:
    """Apply a baseline's edit to the pristine script and import the result.

    Args:
      root: Clone of the benchmark.
      name: The edit's file stem, under :data:`EDITS`.
      scratch: Directory the edited script is written to.

    Returns:
      script: The pristine script with the edit's lines replaced.

    """
    ops = cast(
        list[dict[str, object]],
        cast(object, vars(_load_module(root / EDITS / f"{name}.edit.py"))["OPS"]),
    )
    (op,) = ops
    lines = (root / PRISTINE).read_text().splitlines(keepends=True)
    lines[cast(int, op["start_line"]) - 1 : cast(int, op["end_line"])] = [
        cast(str, op["content"]),
    ]
    path = scratch / f"{name}.py"
    path.write_text("".join(lines))
    return load_script(path)


@contextmanager
def hidden_accelerators() -> Generator[None]:
    """Make the reference see a CPU-only machine, whatever the host has.

    Yields:
      nothing: Used as a context manager.

    """
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=False),
        mock.patch.object(torch.accelerator, "is_available", return_value=False),
        warnings.catch_warnings(),
    ):
        # The reference asks its loaders for pinned memory, which a CPU-only
        # machine cannot give; the warning says nothing about the comparison.
        warnings.filterwarnings("ignore", message=".*pin_memory.*")
        yield


def trace_main(script: Upstream) -> MainTrace:
    """Run the reference's ``main`` for 200 epochs of ResNet-56 on CIFAR-100.

    Only the data loaders and the two epoch functions are replaced.

    Args:
      script: The pristine reference script.

    Returns:
      trace: What the replaced calls saw.

    """
    trace = MainTrace()
    spy = _MainSpy(trace)
    argv = ["custom_loss.py", "--arch", "resnet56", "--dataset", "cifar100"]
    argv += ["--epochs", "200", "--seed", str(reference.SEED)]
    printed = io.StringIO()
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(script, "get_dataloaders", spy.get_dataloaders),
        mock.patch.object(script, "train_epoch", spy.train_epoch),
        mock.patch.object(script, "evaluate", spy.evaluate),
        contextlib.redirect_stdout(printed),
        host_agnostic_numerics(),
        warnings.catch_warnings(),
    ):
        # With the epoch functions replaced, no optimizer step precedes the
        # scheduler's, which torch warns about; the order it checks is moot here.
        warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
        script.main()
    (reported,) = (
        line.removeprefix("TEST_METRICS: test_acc=")
        for line in printed.getvalue().splitlines()
        if line.startswith("TEST_METRICS: test_acc=")
    )
    trace.best = float(reported)
    return trace


def upstream_record(
    variants: dict[str, Upstream],
    trace: MainTrace,
    lockstep: dict[str, Lockstep],
    train_batches: Batches,
) -> dict[str, Tensor]:
    """Return the reference's value of every quantity the default test replays.

    Args:
      variants: The script carrying each loss, keyed as :data:`reference.LOSSES`.
      trace: A run of ``main``.
      lockstep: Each loss's :func:`upstream_lockstep`.
      train_batches: The training passes of :func:`upstream_batches`.

    Returns:
      record: Keyed as the port's records are.

    """
    record: dict[str, Tensor] = {}
    logits, label = reference.fixed_logits()
    with host_agnostic_numerics():
        for name, script in variants.items():
            leaf = logits.clone().requires_grad_()
            config = {"num_classes": 4, "epoch": 0, "total_epochs": 1}
            loss = script.compute_loss(leaf, label, config)
            loss.backward()
            record[f"loss/{name}/value"] = loss.detach()
            record[f"loss/{name}/grad"] = cast(Tensor, leaf.grad)
    record["schedule/lr"] = torch.tensor(trace.rates, dtype=torch.float64)
    record["optimizer/settings"] = torch.tensor(
        [trace.settings[name] for name in reference.HYPERPARAMETERS],
        dtype=torch.float64,
    )
    record["draws/exp000"] = trace.draws
    record["best/accuracy"] = torch.tensor(trace.best, dtype=torch.float64)
    record |= upstream_network(variants["cross_entropy"])
    for values, _ in lockstep.values():
        record |= values
    record |= upstream_pipelines(variants["cross_entropy"])
    record |= reference.decode_batches(
        train_batches,
        translate_pad=Cifar100Data.Config().translate_pad,
    )
    return record


def upstream_network(script: Upstream) -> dict[str, Tensor]:
    """Initialize the shallowest network at the reference's widths and run it.

    Args:
      script: The pristine reference script.

    Returns:
      record: Keyed as :func:`reference.network_record`.

    """
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = script.ResNet(script.BasicBlock, [1, 1, 1], reference.CLASSES)
        script.initialize_weights(model)
        draws = rng_fingerprint()
        media = reference.fixed_media(reference.STEPS)
        model.train()
        train = reference.logits_of(model, media)
        model.eval()
        evaluated = reference.logits_of(model, media)
    return {
        "network/draws": draws,
        "network/train": train.detach(),
        "network/eval": evaluated.detach(),
    }


def upstream_lockstep(script: Upstream, name: str, trace: MainTrace) -> Lockstep:
    """Train the shallowest network for two updates, one per epoch.

    Args:
      script: The reference script carrying the loss under test.
      name: Which loss, for the record's keys.
      trace: Source of the optimizer settings ``main`` used.

    Returns:
      lockstep: Per update the loss, its logits, a probe's logits afterwards
        and the configuration the loss was given; and the model's whole state
        after each update.

    """
    spy = _LossSpy(script)
    probes: list[Tensor] = []
    states: list[dict[str, Tensor]] = []
    with (
        reference.seeded(reference.SEED),
        host_agnostic_numerics(),
        mock.patch.object(script, "compute_loss", spy),
    ):
        model = script.ResNet(script.BasicBlock, [1, 1, 1], reference.CLASSES)
        script.initialize_weights(model)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=trace.settings["lr"],
            momentum=trace.settings["momentum"],
            dampening=trace.settings["dampening"],
            weight_decay=trace.settings["weight_decay"],
            nesterov=bool(trace.settings["nesterov"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=reference.EPOCHS,
        )
        config = {
            "num_classes": reference.CLASSES,
            "epoch": 0,
            "total_epochs": reference.EPOCHS,
        }
        for index in range(reference.STEPS):
            config["epoch"] = index
            batch = [(reference.fixed_media(index), reference.fixed_label(index))]
            script.train_epoch(model, batch, optimizer, torch.device("cpu"), config)
            model.eval()
            with torch.no_grad():
                probes.append(
                    reference.logits_of(model, reference.fixed_media(reference.STEPS)),
                )
            states.append({k: v.clone() for k, v in model.state_dict().items()})
            scheduler.step()
    record = {
        f"lockstep/{name}/loss": torch.stack(spy.losses),
        f"lockstep/{name}/logits": torch.stack(spy.logits),
        f"lockstep/{name}/probe": torch.stack(probes),
        f"lockstep/{name}/context": torch.tensor(
            [[c["num_classes"], c["epoch"], c["total_epochs"]] for c in spy.configs],
        ),
    }
    return record, states


def upstream_batches(script: Upstream) -> tuple[Batches, Batches]:
    """Run the reference's loaders, alternating training and test passes.

    Args:
      script: The pristine reference script.

    Returns:
      passes: The training passes, then the test passes; the loaders use the
        reference's four worker processes.

    """
    train: Batches = []
    evaluation: Batches = []
    with reference.seeded(reference.SEED):
        train_loader, test_loader = _loaders(script, num_workers=4)
        for _ in range(reference.LOADER_PASSES):
            train.append([{"media": x, "label": y} for x, y in train_loader])
            evaluation.append([{"media": x, "label": y} for x, y in test_loader])
    return train, evaluation


def upstream_pipelines(script: Upstream) -> dict[str, Tensor]:
    """Run both of the reference's image pipelines on two fixed images.

    Args:
      script: The pristine reference script.

    Returns:
      record: Keyed as :func:`reference.transform_record`.

    """
    train_loader, test_loader = _loaders(script, num_workers=0)
    pixels = [
        image.permute(1, 2, 0).contiguous().numpy()
        for image in reference.fixed_images(2)
    ]
    train_transform = _transform(train_loader)
    test_transform = _transform(test_loader)
    with reference.seeded(reference.SEED):
        train = torch.stack([train_transform(Image.fromarray(p)) for p in pixels])
    test = torch.stack([test_transform(Image.fromarray(p)) for p in pixels])
    return {"transform/train": train, "transform/test": test}


def upstream_sites(script: Upstream) -> dict[str, int]:
    """Count the random draws of the reference's network, transforms and loaders.

    Args:
      script: The pristine reference script.

    Returns:
      counts: As :func:`reference.draw_sites`; the loaders run in this process
        so the transforms' draws are seen.

    """
    with recorded_draws() as counts:
        _ = upstream_network(script)
        _ = upstream_pipelines(script)
        train_loader, test_loader = _loaders(script, num_workers=0)
        for _ in range(reference.LOADER_PASSES):
            _ = list(train_loader)
            _ = list(test_loader)
    return counts


class _MainSpy:
    """Stands in for the reference's heavy calls while its ``main`` runs."""

    def __init__(self, trace: MainTrace) -> None:
        self.trace = trace
        self.scripted = iter(reference.SCRIPTED_CORRECT)

    def get_dataloaders(
        self,
        dataset: str,
        data_root: str,
        batch_size: int = 0,
        num_workers: int = 0,
    ) -> tuple[list[object], list[object], int]:
        """Return no data and the class count of CIFAR-100."""
        del dataset, data_root, batch_size, num_workers
        return [], [], 100

    def train_epoch(
        self,
        model: nn.Module,
        loader: object,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: dict[str, int],
    ) -> tuple[float, float]:
        """Record the state before the first epoch and the rate of every one.

        Args:
          model: The network, initialized.
          loader: Unused.
          optimizer: The optimizer whose rate and settings are recorded.
          device: Unused.
          config: What this epoch tells the loss.

        Returns:
          result: Zero loss and zero accuracy.

        """
        del loader, device
        group = cast(dict[str, object], optimizer.param_groups[0])
        if not self.trace.rates:
            self.trace.state = {k: v.clone() for k, v in model.state_dict().items()}
            self.trace.draws = rng_fingerprint()
            self.trace.settings = {
                name: float(cast(float, group[name]))
                for name in reference.HYPERPARAMETERS
            }
        self.trace.rates.append(from_plain(group["lr"], float))
        self.trace.configs.append(dict(config))
        return 0.0, 0.0

    def evaluate(
        self,
        model: nn.Module,
        loader: object,
        device: torch.device,
        config: dict[str, int],
    ) -> tuple[float, float]:
        """Report the next scripted accuracy."""
        del model, loader, device, config
        return 0.0, 100.0 * next(self.scripted) / reference.SCRIPTED_TOTAL


class _LossSpy:
    """Forwards to the reference's loss and keeps what it saw."""

    def __init__(self, script: Upstream) -> None:
        self.compute_loss = script.compute_loss
        self.losses: list[Tensor] = []
        self.logits: list[Tensor] = []
        self.configs: list[dict[str, int]] = []

    def __call__(
        self,
        logits: Tensor,
        targets: Tensor,
        config: dict[str, int],
    ) -> Tensor:
        """Return the reference's loss for the batch."""
        loss = self.compute_loss(logits, targets, config)
        self.losses.append(loss.detach())
        self.logits.append(logits.detach())
        self.configs.append(dict(config))
        return loss


class _FakeCifar100:
    """The synthetic split, in the shape the reference expects of a dataset."""

    def __init__(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        del root, train, download
        self.media, self.label = reference.synthetic_split()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.media)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        pixels = self.media[index].permute(1, 2, 0).contiguous().numpy()
        return self.transform(Image.fromarray(pixels)), int(self.label[index])


def _load_module(path: Path) -> ModuleType:
    """Import a script from its file."""
    loader = machinery.SourceFileLoader(path.stem, str(path))
    spec = machinery.ModuleSpec(path.stem, loader, origin=str(path))
    spec.has_location = True
    module = util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _loaders(
    script: Upstream,
    *,
    num_workers: int,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Build the reference's loaders over the synthetic split."""
    with mock.patch.object(script.torchvision.datasets, "CIFAR100", _FakeCifar100):
        train, test, _ = script.get_dataloaders(
            "cifar100",
            "unused",
            batch_size=reference.LOADER_BATCH,
            num_workers=num_workers,
        )
    return train, test


def _transform(
    loader: DataLoader[tuple[Tensor, Tensor]],
) -> Callable[[Image.Image], Tensor]:
    """Return the transform a loader's dataset applies."""
    return cast(_FakeCifar100, loader.dataset).transform
