"""Runs of the pristine ``cv-multitask-loss`` script, under the capture's pins.

The script is imported from a clone of the benchmark, not vendored. Its
``main`` runs as written with the loaders and two epoch functions swapped for
recorders, so its seeding, two-head build, initialization, optimizer over
``all_params``, scheduler and best-accuracy bookkeeping are observed rather
than re-implemented. Each variant's ``MultiTaskLoss`` runs alone on fixed
inputs, the loaders run over the vision capture's synthetic split, and the
lockstep run repeats ``main``'s epoch loop on the shallowest network. Nothing
runs on a GPU.

The ``pcgrad`` edit finds its parameters by walking the graph and deduping
nodes by the ``id`` of short-lived Python wrappers, whose addresses are
reused, so the set it finds depends on the heap. Every run here pins its
instance to the port's full set (:func:`~priml.baselines.mlsbench.dl.multitask.losses.graph_leaves`)
before the first call and checks that the edit's own walk is a subsequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, runtime_checkable
from unittest import mock

import contextlib
import functools
import io
import sys
import warnings

from PIL import Image
from torch import Tensor, nn
from torch.utils import data
from treekle import from_plain

import torch

from priml.baselines.mlsbench.dl.multitask import reference
from priml.baselines.mlsbench.dl.multitask.losses import graph_leaves
from priml.baselines.mlsbench.dl.vision import (
    reference as vision,
    upstream,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from types import ModuleType

    from torch.utils.data import DataLoader

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__cv-multitask-loss/tests/meta/pristine"
    "/pytorch-vision/custom_mtl.py",
)
"""The unedited training script, relative to the clone."""

EDITS: Final = Path("tasks/cv-multitask-loss/edits")
"""Directory of the task's baseline edits, relative to the clone."""

EDITED: Final = ("uncertainty", "dwa", "pcgrad")
"""The published baselines, each named as its edit file and as its port's variant."""

type Lockstep = tuple[dict[str, Tensor], list[dict[str, Tensor]]]
"""A lockstep run's record and the model's whole state after each update."""


class _Torchvision(Protocol):
    """The part of torchvision the script reaches through its own import."""

    datasets: ModuleType


class Script(Protocol):
    """The part of the ``cv-multitask-loss`` script the capture uses."""

    BasicBlock: type[nn.Module]
    torchvision: _Torchvision
    CIFAR100_COARSE_MAP: list[int]

    ResNet: Callable[[type[nn.Module], list[int]], nn.Module]

    MultiTaskLoss: Callable[[int], nn.Module]

    def initialize_weights(self, model: nn.Module) -> None:
        """Initialize in place."""

    def get_dataloaders(
        self,
        data_root: str,
        batch_size: int = ...,
        num_workers: int = ...,
    ) -> tuple[
        DataLoader[tuple[Tensor, Tensor, Tensor]],
        DataLoader[tuple[Tensor, Tensor, Tensor]],
    ]:
        """Build the loaders."""
        raise NotImplementedError

    def train_epoch(  # noqa: PLR0917 -- The script's own signature.
        self,
        model: nn.Module,
        loader: Iterable[tuple[Tensor, Tensor, Tensor]],
        mtl_loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        total_epochs: int,
    ) -> tuple[float, float]:
        """Train for one pass over ``loader``."""
        raise NotImplementedError

    def main(self) -> None:
        """Run the whole script."""


@dataclass(slots=True, kw_only=True)
class Trace:
    """What the script's ``main`` did, as its replaced calls saw it."""

    rates: list[float] = field(default_factory=list[float])
    """Learning rate of each epoch."""

    epochs: list[tuple[int, int]] = field(default_factory=list[tuple[int, int]])
    """The epoch and total epochs each ``train_epoch`` was given."""

    settings: dict[str, float] = field(default_factory=dict[str, float])
    """The optimizer's settings in the first epoch."""

    numels: list[int] = field(default_factory=list[int])
    """Size of every optimized tensor, in the optimizer's order."""

    state: dict[str, Tensor] = field(default_factory=dict[str, Tensor])
    """The network's whole state before the first epoch."""

    draws: Tensor = field(default_factory=lambda: torch.empty(0))
    """Where the generator stood before the first epoch."""

    best: float = 0.0
    """The accuracy ``main`` reported."""


def variants(root: Path, scratch: Path) -> dict[str, Script]:
    """Import the pristine script and each baseline's edit of it.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts are written to.

    Returns:
      scripts: ``sum`` (the pristine script) and each of :data:`EDITED`.

    """
    scripts = {"sum": upstream.load_script(root / PRISTINE)} | {
        name: upstream.edited_script(
            root,
            name,
            scratch,
            pristine=PRISTINE,
            edits=EDITS,
        )
        for name in EDITED
    }
    return {name: cast(Script, cast(object, s)) for name, s in scripts.items()}


def trace_main(script: Script) -> Trace:
    """Run the script's ``main`` for 200 epochs of the ``resnet56`` setting.

    Args:
      script: The pristine script or an edit of it.

    Returns:
      trace: What the replaced calls saw.

    """
    trace = Trace()
    spy = _MainSpy(trace)
    argv = ["script.py", "--arch", "resnet56", "--epochs", "200"]
    argv += ["--seed", str(vision.SEED)]
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


def pin_shared(module: nn.Module, loss_fine: Tensor) -> list[str]:
    """Pin a ``pcgrad`` instance to the full set of shared leaves before its first call.

    Args:
      module: A script's ``MultiTaskLoss``; one that does not walk the graph
        is left alone.
      loss_fine: The fine loss the instance is about to be called with.

    Returns:
      report: A line if the edit's own walk, run on a fresh instance, is not a
        subsequence of :func:`graph_leaves`.

    """
    if not isinstance(module, Walker):
        return []
    ours = graph_leaves(loss_fine)
    theirs = type(module)()._get_shared_params(loss_fine)  # noqa: SLF001 -- The edit's own walk is what is checked.
    module._shared_params = ours  # noqa: SLF001 -- Pinned on the instance; no file is edited.
    remaining = iter(id(p) for p in ours)
    if all(id(p) in remaining for p in theirs):
        return []
    return ["pcgrad: the edit's walk is not a subsequence of the port's"]


def combination_record(make: Callable[[], nn.Module], name: str) -> dict[str, Tensor]:
    """Run a combination on the fixed calls of :func:`reference.run_combination`.

    Args:
      make: Builds the combination.
      name: Which combination, for the record's keys.

    Returns:
      record: Keyed as :func:`reference.combination_record`.

    """
    module = make()
    report: list[str] = []
    record = reference.run_combination(
        module,
        name,
        before_first=lambda loss_fine: report.extend(pin_shared(module, loss_fine)),
    )
    if report:
        raise ValueError(report)
    return record


def upstream_combinations(scripts: dict[str, Script]) -> dict[str, Tensor]:
    """Run every variant's ``MultiTaskLoss`` on the fixed calls.

    Args:
      scripts: Each variant's script, keyed as :func:`variants`.

    Returns:
      record: The union of each variant's :func:`combination_record`.

    """
    record: dict[str, Tensor] = {}
    for name, script in scripts.items():
        record |= combination_record(lambda s=script: s.MultiTaskLoss(2), name)
    return record


def upstream_lockstep(script: Script, name: str, trace: Trace) -> Lockstep:
    """Repeat ``main``'s epoch loop on the shallowest network for two epochs.

    Args:
      script: The script carrying the combination under test.
      name: Which combination, for the record's keys.
      trace: Source of the optimizer settings ``main`` used.

    Returns:
      lockstep: Keyed as :func:`reference.lockstep_record`; and the network's
        whole state after each update.

    """
    probes: list[Tensor] = []
    states: list[dict[str, Tensor]] = []
    report: list[str] = []
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        model = script.ResNet(script.BasicBlock, [1, 1, 1])
        script.initialize_weights(model)
        inner = script.MultiTaskLoss(2)
        spy = reference.CombinationSpy(
            inner,
            before_first=lambda loss_fine: report.extend(pin_shared(inner, loss_fine)),
        )
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(inner.parameters()),
            lr=trace.settings["lr"],
            momentum=trace.settings["momentum"],
            weight_decay=trace.settings["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=vision.EPOCHS,
        )
        for index in range(vision.STEPS):
            batch = [
                (
                    vision.fixed_media(index),
                    vision.fixed_label(index),
                    reference.coarse_label(index),
                ),
            ]
            script.train_epoch(
                model,
                batch,
                spy,
                optimizer,
                torch.device("cpu"),
                index,
                vision.EPOCHS,
            )
            model.eval()
            with torch.no_grad():
                fine, coarse = cast(
                    tuple[Tensor, Tensor],
                    model(vision.fixed_media(vision.STEPS)),
                )
            probes.append(torch.cat([fine, coarse], dim=-1))
            states.append({k: v.clone() for k, v in model.state_dict().items()})
            scheduler.step()
    if report:
        raise ValueError(report)
    return reference.spy_record(name, spy, probes, inner), states


def upstream_record(
    scripts: dict[str, Script],
    traces: dict[str, Trace],
    lockstep: dict[str, Lockstep],
) -> dict[str, Tensor]:
    """Return the script's value of every quantity the default test replays.

    Args:
      scripts: Each variant's script, keyed as :func:`variants`.
      traces: A run of each variant's ``main``, keyed alike.
      lockstep: Each variant's :func:`upstream_lockstep`.

    Returns:
      record: Keyed as :func:`reference.port_record`.

    """
    pristine = traces["sum"]
    record = upstream_combinations(scripts)
    record["data/coarse_of_fine"] = torch.tensor(scripts["sum"].CIFAR100_COARSE_MAP)
    record["schedule/lr"] = torch.tensor(pristine.rates, dtype=torch.float64)
    record["optimizer/settings"] = torch.tensor(
        [pristine.settings[name] for name in vision.HYPERPARAMETERS],
        dtype=torch.float64,
    )
    record["best/accuracy"] = torch.tensor(pristine.best, dtype=torch.float64)
    for name, trace in traces.items():
        record[f"draws/{name}"] = trace.draws
        record[f"optimizer/{name}/numels"] = torch.tensor(trace.numels)
    for values, _ in lockstep.values():
        record |= values
    return record


def upstream_batches(script: Script) -> tuple[reference.Batches, reference.Batches]:
    """Run the script's loaders over the synthetic split, alternating the splits.

    Args:
      script: The pristine script.

    Returns:
      passes: The training passes, then the test passes; the loaders use the
        script's four worker processes.

    """
    train: reference.Batches = []
    evaluation: reference.Batches = []
    # The script's dataset class lives in a module imported from a file, which a
    # forkserver worker cannot import by name; a forked worker inherits it. The
    # start method changes no draw: each worker's seed comes from the parent.
    forked = functools.partial(data.DataLoader, multiprocessing_context="fork")
    with (
        vision.seeded(vision.SEED),
        vision.script_workers(),
        mock.patch.object(script.torchvision.datasets, "CIFAR100", _FakeCifar100),
        mock.patch.object(data, "DataLoader", forked),
    ):
        train_loader, test_loader = script.get_dataloaders(
            "unused",
            batch_size=vision.LOADER_BATCH,
            num_workers=4,
        )
        for _ in range(vision.LOADER_PASSES):
            train.append([_batch(*b) for b in train_loader])
            evaluation.append([_batch(*b) for b in test_loader])
    return train, evaluation


def check_walk(pcgrad: Script, model: nn.Module) -> list[str]:
    """Check the ``pcgrad`` walk against the port's on a network's fine loss.

    Args:
      pcgrad: The script carrying the ``pcgrad`` edit.
      model: A two-head network of that script.

    Returns:
      report: A line if the edit's walk is not a subsequence of the port's,
        or if the port's misses any parameter besides the coarse head's.

    """
    with host_agnostic_numerics():
        fine, _ = cast(tuple[Tensor, Tensor], model(vision.fixed_media(0)))
        loss_fine = torch.nn.functional.cross_entropy(fine, vision.fixed_label(0))
        instance = pcgrad.MultiTaskLoss(2)
        report = pin_shared(instance, loss_fine)
    assert isinstance(instance, Walker)
    found = {id(p) for p in instance._shared_params or ()}  # noqa: SLF001 -- What pin_shared set.
    missed = [
        n
        for n, p in model.named_parameters()
        if id(p) not in found and not n.startswith("fc_coarse.")
    ]
    if missed:
        report.append(f"pcgrad: the port's walk missed {missed}")
    return report


@runtime_checkable
class Walker(Protocol):
    """The ``pcgrad`` edit's graph walk and the attribute that caches it."""

    _shared_params: list[Tensor] | None

    def _get_shared_params(self, loss: Tensor) -> list[Tensor]:
        """Return the shared leaves of ``loss``."""
        raise NotImplementedError


class _MainSpy:
    """Stands in for the script's heavy calls while its ``main`` runs."""

    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self.scripted = iter(vision.SCRIPTED_CORRECT)

    def get_dataloaders(
        self,
        data_root: str,
        batch_size: int = 0,
        num_workers: int = 0,
    ) -> tuple[list[object], list[object]]:
        """Return no data."""
        del data_root, batch_size, num_workers
        return [], []

    def train_epoch(  # noqa: PLR0917 -- The script calls it positionally.
        self,
        model: nn.Module,
        loader: object,
        mtl_loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: object,
        epoch: int,
        total_epochs: int,
    ) -> tuple[float, float]:
        """Record the state before the first epoch and the rate of every one.

        Args:
          model: The script's network, whose state is recorded once.
          loader: Unused; no epoch runs.
          mtl_loss: Unused; no epoch runs.
          optimizer: The script's optimizer, whose settings and rate are recorded.
          device: Unused; no epoch runs.
          epoch: The script's epoch index.
          total_epochs: The script's epoch count.

        Returns:
          metrics: Zero loss and zero accuracy.

        """
        del loader, mtl_loss, device
        group = cast(dict[str, object], optimizer.param_groups[0])
        groups = cast(list[dict[str, list[Tensor]]], optimizer.param_groups)
        if not self.trace.rates:
            self.trace.state = {k: v.clone() for k, v in model.state_dict().items()}
            self.trace.draws = rng_fingerprint()
            self.trace.settings = {
                name: float(cast(float, group[name])) for name in vision.HYPERPARAMETERS
            }
            self.trace.numels = [p.numel() for g in groups for p in g["params"]]
        self.trace.rates.append(from_plain(group["lr"], float))
        self.trace.epochs.append((epoch, total_epochs))
        return 0.0, 0.0

    def evaluate(self, model: object, loader: object, device: object) -> float:
        """Report the next scripted accuracy."""
        del model, loader, device
        return 100.0 * next(self.scripted) / vision.SCRIPTED_TOTAL


class _FakeCifar100:
    """The synthetic split, in the shape the script expects of a dataset."""

    def __init__(
        self,
        *,
        root: str,
        train: bool,
        transform: Callable[[Image.Image], Tensor],
        download: bool,
    ) -> None:
        del root, train, download
        self.media, self.label = vision.synthetic_split()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.media)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        pixels = self.media[index].permute(1, 2, 0).contiguous().numpy()
        return self.transform(Image.fromarray(pixels)), int(self.label[index])


def _batch(media: Tensor, label: Tensor, label_coarse: Tensor) -> dict[str, Tensor]:
    """Key one of the script's batches as the port's are."""
    return {"media": media, "label": label, "label_coarse": label_coarse}
