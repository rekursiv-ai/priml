"""Runs of the pristine ``cv-sample-weighting`` script, under the capture's pins.

The script is imported from a clone of the benchmark, not vendored. Its
``main`` runs as written over a dataset carrying CIFAR-100's label layout, with
its two epoch functions swapped for recorders, so its own long-tail cut, class
counts, class weights, model build, optimizer and scheduler are observed. Its
loaders run over a synthetic split, and over the real training split when one
is given. Nothing runs on a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

from priml.baselines.mlsbench.dl.vision import (
    reference as vision,
    upstream as vision_upstream,
)
from priml.baselines.mlsbench.dl.weighting import reference
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch.utils.data import DataLoader, Subset

    from priml.baselines.mlsbench.dl.vision.upstream import (
        Batches,
        Lockstep,
        Upstream,
    )

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__cv-sample-weighting/tests/meta/pristine"
    "/pytorch-vision/custom_weighting.py",
)
"""The unedited training script, relative to the clone."""

EDITS: Final = Path("tasks/cv-sample-weighting/edits")
"""Directory of the task's baseline edits, relative to the clone."""

EDITED: Final = ("inverse_freq", "effective_number", "balanced_softmax")
"""The published baselines, each named as its edit file and as its port's variant."""

CONFIG: Final = {
    "imbalance_ratio": reference.RATIO,
    "dataset": "cifar100",
    "arch": "resnet32",
    "total_samples": 0,
}
"""What ``main`` tells ``compute_class_weights``; no baseline reads it."""


class Script(Protocol):
    """The part of the ``cv-sample-weighting`` script the capture uses."""

    BasicBlock: type[nn.Module]

    ResNet: Callable[[type[nn.Module], list[int], int], nn.Module]

    def initialize_weights(self, model: nn.Module) -> None:
        """Initialize in place."""

    def compute_class_weights(
        self,
        class_counts: Tensor,
        num_classes: int,
        config: dict[str, object],
    ) -> Tensor:
        """Return the per-class weights."""
        raise NotImplementedError

    def get_dataloaders(
        self,
        dataset_name: str,
        data_root: str,
        imbalance_ratio: float,
        *,
        batch_size: int = ...,
        num_workers: int = ...,
        seed: int = ...,
    ) -> tuple[
        DataLoader[tuple[Tensor, Tensor]],
        DataLoader[tuple[Tensor, Tensor]],
        int,
        Tensor,
    ]:
        """Build the long-tailed loaders."""
        raise NotImplementedError

    def train_epoch(
        self,
        model: nn.Module,
        loader: Iterable[tuple[Tensor, Tensor]],
        criterion: Callable[[Tensor, Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[float, float]:
        """Train for one pass over ``loader``."""
        raise NotImplementedError

    def main(self) -> None:
        """Run the whole script."""


@dataclass(slots=True, kw_only=True)
class Trace:
    """What the script's ``main`` did, as its replaced calls saw it."""

    main: vision_upstream.MainTrace = field(default_factory=vision_upstream.MainTrace)
    """Rates, optimizer settings, initial state, draws and reported accuracy."""

    weight: Tensor = field(default_factory=lambda: torch.empty(0))
    """The class weights its criterion was built with."""

    counts: Tensor = field(default_factory=lambda: torch.empty(0))
    """The class counts its loaders reported."""

    indices: Tensor = field(default_factory=lambda: torch.empty(0))
    """The training positions its subset kept, in order."""


def script_of(variant: Upstream) -> Script:
    """Type an imported variant as this task's script."""
    return cast(Script, cast(object, variant))


def variants(root: Path, scratch: Path) -> dict[str, Script]:
    """Import the pristine script and each baseline's edit of it.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts are written to.

    Returns:
      scripts: ``uniform`` (the pristine script) and each of :data:`EDITED`.

    """
    pristine = vision_upstream.load_script(root / PRISTINE)
    edited = {
        name: vision_upstream.edited_script(
            root,
            name,
            scratch,
            pristine=PRISTINE,
            edits=EDITS,
        )
        for name in EDITED
    }
    return {name: script_of(s) for name, s in ({"uniform": pristine} | edited).items()}


def trace_main(script: Script) -> Trace:
    """Run ``main`` for 200 epochs of the ``resnet32-cifar100lt`` setting.

    The training split is :func:`reference.balanced_label` with no pixels: the
    cut reads only the labels, and no epoch iterates a loader.

    Args:
      script: The pristine script or an edit of it.

    Returns:
      trace: What the replaced calls saw.

    """
    trace = Trace()
    spy = _MainSpy(trace)
    argv = ["script.py", "--arch", "resnet32", "--dataset", "cifar100"]
    argv += ["--imbalance-ratio", str(reference.RATIO)]
    argv += ["--epochs", "200", "--seed", str(vision.SEED)]
    printed = io.StringIO()
    with (
        _datasets(script, _LabelsOnly),
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(script, "train_epoch", spy.train_epoch),
        mock.patch.object(script, "evaluate", spy.evaluate),
        mock.patch.object(script, "get_dataloaders", spy.wrap(script.get_dataloaders)),
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
    trace.main.best = float(reported)
    return trace


def upstream_weights(scripts: dict[str, Script]) -> dict[str, Tensor]:
    """Run every variant's weights and weighted loss on fixed inputs.

    Args:
      scripts: Keyed as :func:`variants`.

    Returns:
      record: Keyed as :func:`reference.weights_record`.

    """
    logits, label = vision.fixed_logits()
    record: dict[str, Tensor] = {}
    with host_agnostic_numerics():
        for name, script in scripts.items():
            flat = torch.tensor(reference.FLAT_COUNTS)
            record[f"weights/{name}/flat"] = script.compute_class_weights(
                flat,
                len(flat),
                dict(CONFIG),
            )
            counts = torch.tensor(reference.COUNTS)
            weight = script.compute_class_weights(counts, len(counts), dict(CONFIG))
            record[f"weights/{name}"] = weight
            leaf = logits.clone().requires_grad_()
            loss = nn.CrossEntropyLoss(weight=weight)(leaf, label)
            loss.backward()
            record[f"loss/{name}/value"] = loss.detach()
            record[f"loss/{name}/grad"] = cast(Tensor, leaf.grad)
    return record


def upstream_lockstep(script: Script, name: str, trace: Trace) -> Lockstep:
    """Repeat ``main``'s epoch loop on the shallowest network for two epochs.

    Args:
      script: The script carrying the weighting under test.
      name: Which weighting, for the record's keys.
      trace: Source of the optimizer settings ``main`` used.

    Returns:
      lockstep: Per update the loss, its logits and a probe's logits
        afterwards; and the model's whole state after each update.

    """
    settings = trace.main.settings
    probes: list[Tensor] = []
    states: list[dict[str, Tensor]] = []
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        model = script.ResNet(script.BasicBlock, [1, 1, 1], vision.CLASSES)
        script.initialize_weights(model)
        counts = torch.tensor(reference.LOCKSTEP_COUNTS)
        weight = script.compute_class_weights(counts, len(counts), dict(CONFIG))
        criterion = vision_upstream.CriterionSpy(nn.CrossEntropyLoss(weight=weight))
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=settings["lr"],
            momentum=settings["momentum"],
            weight_decay=settings["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=vision.EPOCHS,
        )
        for index in range(vision.STEPS):
            batch = [(vision.fixed_media(index), vision.fixed_label(index))]
            script.train_epoch(model, batch, criterion, optimizer, torch.device("cpu"))
            model.eval()
            with torch.no_grad():
                probes.append(
                    vision.logits_of(model, vision.fixed_media(vision.STEPS)),
                )
            states.append({k: v.clone() for k, v in model.state_dict().items()})
            scheduler.step()
    record = {
        f"lockstep/{name}/loss": torch.stack(criterion.losses),
        f"lockstep/{name}/logits": torch.stack(criterion.logits),
        f"lockstep/{name}/probe": torch.stack(probes),
    }
    return record, states


def upstream_batches(script: Script) -> tuple[Batches, Batches]:
    """Run the script's loaders over the synthetic split, alternating the splits.

    Args:
      script: The pristine script.

    Returns:
      passes: The training passes, then the test passes, through the script's
        four worker processes.

    """
    train: Batches = []
    evaluation: Batches = []
    with vision.seeded(vision.SEED), _datasets(script, _SyntheticSplit):
        train_loader, test_loader, _, _ = script.get_dataloaders(
            "cifar100",
            "unused",
            reference.SPLIT_RATIO,
            batch_size=vision.LOADER_BATCH,
            seed=vision.SEED,
        )
        for _ in range(vision.LOADER_PASSES):
            train.append([{"media": x, "label": y} for x, y in train_loader])
            evaluation.append([{"media": x, "label": y} for x, y in test_loader])
    return train, evaluation


def upstream_subset(script: Script) -> dict[str, Tensor]:
    """Cut the synthetic split with the script's own loaders.

    Args:
      script: The pristine script.

    Returns:
      record: Keyed as :func:`reference.subset_record`.

    """
    with _datasets(script, _SyntheticSplit):
        train, _, _, counts = script.get_dataloaders(
            "cifar100",
            "unused",
            reference.SPLIT_RATIO,
            num_workers=0,
            seed=vision.SEED,
        )
    return {"subset/indices": _kept(train), "subset/counts": counts}


def real_subset(script: Script, data_root: Path) -> tuple[Tensor, Tensor, Tensor]:
    """Cut the real CIFAR-100 training split with the script's own loaders.

    Args:
      script: The pristine script.
      data_root: Directory holding ``cifar-100-python``, as torchvision reads it.

    Returns:
      indices: The kept positions, in order.
      counts: The 100 class counts.
      label: Every training label, as torchvision reads it.

    """
    train, _, _, counts = script.get_dataloaders(
        "cifar100",
        str(data_root),
        reference.RATIO,
        num_workers=0,
        seed=vision.SEED,
    )
    dataset = cast("Subset[object]", train.dataset).dataset
    label = torch.as_tensor(cast(_HasTargets, dataset).targets)
    return _kept(train), counts, label


def upstream_record(
    trace: Trace,
    weights: dict[str, Tensor],
    lockstep: dict[str, Lockstep],
    subset: dict[str, Tensor],
    train_batches: Batches,
) -> dict[str, Tensor]:
    """Return the script's value of every quantity the default test replays.

    Args:
      trace: A run of the pristine ``main``.
      weights: :func:`upstream_weights`.
      lockstep: Each weighting's :func:`upstream_lockstep`.
      subset: :func:`upstream_subset`.
      train_batches: The training passes of :func:`upstream_batches`.

    Returns:
      record: Keyed as :func:`reference.port_record`.

    """
    record = dict(weights)
    record["counts/cifar100lt"] = trace.counts
    record["main/weight"] = trace.weight
    record |= subset
    record["schedule/lr"] = torch.tensor(trace.main.rates, dtype=torch.float64)
    record["optimizer/settings"] = torch.tensor(
        [trace.main.settings[name] for name in vision.HYPERPARAMETERS],
        dtype=torch.float64,
    )
    record["draws/exp000"] = trace.main.draws
    record["best/accuracy"] = torch.tensor(trace.main.best, dtype=torch.float64)
    record |= reference.decode_indices(train_batches)
    for values, _ in lockstep.values():
        record |= values
    return record


class _HasTargets(Protocol):
    """A torchvision dataset's labels."""

    targets: list[int]


class _MainSpy:
    """Stands in for the script's epoch functions and observes its loaders."""

    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self.scripted = iter(vision.SCRIPTED_CORRECT)

    def wrap(
        self,
        get_dataloaders: Callable[..., tuple[DataLoader[object], object, int, Tensor]],
    ) -> Callable[..., tuple[DataLoader[object], object, int, Tensor]]:
        """Return ``get_dataloaders``, recording the counts and kept positions."""

        def recorded(
            *args: object,
            **kwargs: object,
        ) -> tuple[DataLoader[object], object, int, Tensor]:
            loaders = get_dataloaders(*args, **kwargs)
            self.trace.counts = loaders[3].clone()
            self.trace.indices = _kept(loaders[0])
            return loaders

        return recorded

    def train_epoch(
        self,
        model: nn.Module,
        loader: object,
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
        device: object,
    ) -> tuple[float, float]:
        """Record the state before the first epoch and the rate of every one.

        Args:
          model: The network, initialized.
          loader: Unused.
          criterion: The weighted loss, whose weights are recorded.
          optimizer: Its rate and settings are recorded.
          device: Unused.

        Returns:
          result: Zero loss and zero accuracy.

        """
        del loader, device
        main = self.trace.main
        group = cast(dict[str, object], optimizer.param_groups[0])
        if not main.rates:
            main.state = {k: v.clone() for k, v in model.state_dict().items()}
            main.draws = rng_fingerprint()
            main.settings = {
                name: float(cast(float, group[name])) for name in vision.HYPERPARAMETERS
            }
            self.trace.weight = cast(Tensor, criterion.weight).clone()
        main.rates.append(from_plain(group["lr"], float))
        return 0.0, 0.0

    def evaluate(self, *args: object) -> tuple[float, float]:
        """Report the next scripted accuracy, whatever the script passes."""
        del args
        return 0.0, 100.0 * next(self.scripted) / vision.SCRIPTED_TOTAL


class _LabelsOnly:
    """CIFAR-100's label layout, without pixels; both splits alike."""

    def __init__(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: object,
    ) -> None:
        del root, train, download, transform
        self.targets = reference.balanced_label().tolist()


class _SyntheticSplit:
    """The synthetic long-tailed split, as the script reads a dataset."""

    def __init__(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        del root, train, download
        self.media, label = reference.long_tail_split()
        self.targets = from_plain(label.tolist(), list[int])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.media)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        pixels = self.media[index].permute(1, 2, 0).contiguous().numpy()
        return self.transform(Image.fromarray(pixels)), self.targets[index]


def _datasets(
    script: Script,
    dataset: type,
) -> contextlib.AbstractContextManager[object]:
    """Make the script's ``torchvision.datasets.CIFAR100`` read ``dataset``."""
    vision_script = cast(vision_upstream.Upstream, cast(object, script))
    return mock.patch.object(vision_script.torchvision.datasets, "CIFAR100", dataset)


def _kept(loader: DataLoader[object]) -> Tensor:
    """Return the positions a loader's ``Subset`` kept, in order."""
    indices = cast("Subset[object]", loader.dataset).indices
    return torch.tensor([int(i) for i in indices], dtype=torch.int64)
