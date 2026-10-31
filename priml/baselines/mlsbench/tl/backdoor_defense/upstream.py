"""Runs of the pristine ``security-backdoor-defense`` driver, under the capture's pins.

The driver and each baseline's edit of its defense module are imported from a
clone of the benchmark, not vendored. Its ``main`` runs whole on a small split
with its epoch function replaced, so its poisoning, feature extraction,
filtering and evaluation cadence are observed rather than re-implemented; its
other functions run alone on fixed inputs. Nothing runs on a GPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable
from unittest import mock

import contextlib
import io
import random
import sys
import warnings

from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from treekle import from_plain

import numpy as np
import torch

from priml.baselines.mlsbench.dl.vision import (
    reference as vision_reference,
    upstream,
)
from priml.baselines.mlsbench.dl.vision.draws import recorded_draws
from priml.baselines.mlsbench.tl.backdoor_defense import reference
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping

    from numpy.typing import NDArray
    from PIL import Image

    from priml.baselines.mlsbench.dl.vision.upstream import (
        Batches,
        Lockstep,
        Upstream,
    )

DRIVER: Final = Path(
    "harbor/tasks-docker/mls-bench__security-backdoor-defense/tests/meta/pristine"
    "/pytorch-vision/bench/backdoor/run_backdoor_defense.py",
)
"""The fixed driver, relative to the clone."""

DEFENSE: Final = DRIVER.with_name("custom_backdoor_defense.py")
"""The editable defense module the driver imports, relative to the clone."""

EDITS: Final = Path("tasks/security-backdoor-defense/edits")
"""Directory of the task's baseline edits, relative to the clone."""

STATS: Final = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "fmnist": ((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
}
"""The channel statistics the driver's ``main`` hands its trigger, per ``--dataset``."""


class Defense(Protocol):
    """The driver's ``BackdoorDefense``."""

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        poison_fraction: float,
    ) -> None:
        """Fit on the training features."""

    def score_samples(
        self,
        features: NDArray[np.float32],
        logits: NDArray[np.float32],
    ) -> NDArray[np.floating]:
        """Score each image."""
        raise NotImplementedError


@runtime_checkable
class Driver(Protocol):
    """The part of the driver the capture calls."""

    BackdoorDefense: Callable[[], Defense]
    PoisonedDataset: Callable[..., object]
    ResNet: Callable[[type[nn.Module], list[int], int], nn.Module]
    BasicBlock: type[nn.Module]

    def build_model(self, arch: str, num_classes: int) -> nn.Module:
        """Build the network an ``--arch`` names."""
        raise NotImplementedError

    def initialize_weights(self, model: nn.Module) -> None:
        """Initialize in place."""

    def select_poison_indices(
        self,
        dataset: object,
        poison_fraction: float,
        target_class: int,
    ) -> list[int]:
        """Pick the positions to poison."""
        raise NotImplementedError

    def collect_features_logits(
        self,
        model: nn.Module,
        loader: Iterable[tuple[Tensor, Tensor]],
        device: torch.device,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64]]:
        """Return features, logits and labels."""
        raise NotImplementedError

    def evaluate(
        self,
        model: nn.Module,
        loader: object,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[float, float]:
        """Return the mean loss and the accuracy in percent."""
        raise NotImplementedError

    def evaluate_asr(
        self,
        model: nn.Module,
        test_set: object,
        device: torch.device,
        batch_size: int,
        *,
        target_class: int,
        trigger: str,
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> float:
        """Return the attack success rate."""
        raise NotImplementedError

    def train_epoch(
        self,
        model: nn.Module,
        loader: Iterable[tuple[Tensor, Tensor]],
        criterion: Callable[[Tensor, Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[float, float]:
        """Train for one pass."""
        raise NotImplementedError

    def get_dataloaders(
        self,
        dataset: str,
        data_root: str,
        batch_size: int = ...,
        num_workers: int = ...,
    ) -> tuple[object, object, object, object, int]:
        """Return the datasets, their loaders and the class count."""
        ...

    def main(self) -> None:
        """Run the whole driver."""


@dataclass(slots=True, kw_only=True)
class Network:
    """One network ``main`` trained, as its epoch function saw it."""

    rates: list[float] = field(default_factory=list[float])
    """Learning rate of each epoch."""

    settings: dict[str, float] = field(default_factory=dict[str, float])
    """The optimizer's settings in the first epoch."""

    indices: list[int] = field(default_factory=list[int])
    """Positions of the poisoned split it trained on."""

    evaluated: list[int] = field(default_factory=list[int])
    """The epochs, from 1, after which the test split was evaluated."""


@dataclass(slots=True, kw_only=True)
class DriverTrace:
    """What the driver's ``main`` did, as its replaced and wrapped calls saw it."""

    networks: list[Network] = field(default_factory=list[Network])
    """The victim, then the network retrained on the kept images."""

    is_poison: list[bool] = field(default_factory=list[bool])
    """Which training images ``main`` poisoned."""

    features: Tensor = field(default_factory=lambda: torch.empty(0))
    """The features the defense was fit on and scored."""

    logits: Tensor = field(default_factory=lambda: torch.empty(0))
    """The logits the defense scored."""

    labels: Tensor = field(default_factory=lambda: torch.empty(0))
    """The labels the defense was fit on."""

    scores: Tensor = field(default_factory=lambda: torch.empty(0))
    """The defense's scores, as float64 as ``main`` takes them."""

    recall: float = 0.0
    """The poison recall ``main`` printed, at its four decimals."""

    python: Tensor = field(default_factory=lambda: torch.empty(0))
    """Python's generator after ``main``."""

    numpy: Tensor = field(default_factory=lambda: torch.empty(0))
    """NumPy's global generator after ``main``."""


def variants(root: Path, scratch: Path) -> dict[str, Upstream]:
    """Import the driver once per defense: the pristine one and each baseline's edit.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited defense modules are written to.

    Returns:
      scripts: Keyed as :data:`reference.VARIANTS`.

    """
    scripts: dict[str, Upstream] = {}
    for name in reference.VARIANTS:
        defense = (
            upstream.load_script(root / DEFENSE)
            if name == "pristine"
            else upstream.edited_script(
                root,
                name,
                scratch,
                pristine=DEFENSE,
                edits=EDITS,
            )
        )
        with _defense_module(defense):
            scripts[name] = upstream.load_script(root / DRIVER)
    return scripts


def driver(script: object) -> Driver:
    """Narrow an imported driver to the members the capture calls."""
    return upstream.narrowed(script, Driver)


def trace_driver(script: Upstream, setting: str) -> DriverTrace:
    """Run ``main`` at a setting over :func:`reference.main_split`, training skipped.

    The epoch function records and returns; everything else -- poisoning,
    the network builds, feature extraction, the defense, the filter, every
    evaluation -- runs as written, its loaders with four forked workers.

    Args:
      script: The driver, carrying the defense under test.
      setting: A key of :data:`reference.SETTINGS`.

    Returns:
      trace: What the replaced and wrapped calls saw.

    """
    task = driver(script)
    spec = reference.SETTINGS[setting]
    trace = DriverTrace()
    spy = _EpochSpy(trace, evaluate=task.evaluate)
    printed = io.StringIO()
    argv = ["driver.py", "--arch", spec.arch, "--dataset", spec.dataset]
    argv += ["--trigger", spec.trigger, "--poison-fraction", str(spec.fraction)]
    argv += [
        "--epochs",
        str(reference.EPOCHS_MAIN),
        "--seed",
        str(vision_reference.SEED),
    ]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(task, "train_epoch", spy.train_epoch),
        mock.patch.object(task, "evaluate", spy.evaluate),
        mock.patch.object(task, "BackdoorDefense", _Recorded(task, trace)),
        _main_split(script, spec.dataset),
        vision_reference.script_workers(),
        upstream.forked_workers(),
        contextlib.redirect_stdout(printed),
        warnings.catch_warnings(),
    ):
        # With the epoch function replaced no optimizer step precedes the
        # scheduler's, which torch warns about; the order is moot here.
        warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*")
        task.main()
    (recall,) = (
        line.split("recall=")[1].split(",")[0]
        for line in printed.getvalue().splitlines()
        if line.startswith("Defense removed")
    )
    trace.recall = float(recall)
    trace.python = reference.python_fingerprint(random.getstate())
    legacy = np.random.RandomState()
    legacy.set_state(np.random.get_state())  # noqa: NPY002 -- The driver seeds NumPy's legacy global generator; its state is what is checked.
    trace.numpy = reference.numpy_fingerprint(legacy)
    return trace


def main_record(trace: DriverTrace, setting: str) -> dict[str, Tensor]:
    """Return what ``main`` decided around training, keyed as :func:`reference.main_record`.

    Args:
      trace: :func:`trace_driver` of the pristine driver at ``setting``.
      setting: A key of :data:`reference.SETTINGS`.

    Returns:
      record: Under ``main/<setting>/``.

    """
    victim = trace.networks[0]
    key = f"main/{setting}"
    return {
        f"{key}/is_poison": torch.tensor(trace.is_poison, dtype=torch.uint8),
        f"{key}/evaluated": torch.tensor(victim.evaluated),
        f"{key}/rates": torch.tensor(victim.rates, dtype=torch.float64),
        f"{key}/settings": torch.tensor(
            [victim.settings[name] for name in vision_reference.HYPERPARAMETERS],
            dtype=torch.float64,
        ),
        f"{key}/python": trace.python,
        f"{key}/numpy": trace.numpy,
    }


def filter_record(traces: dict[str, DriverTrace]) -> dict[str, Tensor]:
    """Return what each variant's ``main`` kept, keyed as :func:`reference.filter_record`.

    Args:
      traces: Each variant's :func:`trace_driver` at :data:`reference.FIRST`.

    Returns:
      record: ``filter/<variant>/{scores,kept,recall}``, the recall at the
        four decimals ``main`` prints.

    """
    record: dict[str, Tensor] = {}
    for name, trace in traces.items():
        record[f"filter/{name}/scores"] = trace.scores
        record[f"filter/{name}/kept"] = torch.tensor(trace.networks[1].indices)
        record[f"filter/{name}/recall"] = torch.tensor(
            trace.recall,
            dtype=torch.float64,
        )
    return record


def filter_inputs(trace: DriverTrace) -> dict[str, Tensor]:
    """Return the features, logits and labels ``main`` gave its defense.

    Args:
      trace: :func:`trace_driver` of the pristine driver at :data:`reference.FIRST`.

    Returns:
      record: ``main/<FIRST>/{features,logits,labels,is_poison}``.

    """
    key = f"main/{reference.FIRST}"
    return {
        f"{key}/features": trace.features,
        f"{key}/logits": trace.logits,
        f"{key}/labels": trace.labels,
        f"{key}/is_poison": torch.tensor(trace.is_poison, dtype=torch.uint8),
    }


def poison_record(script: Upstream) -> dict[str, Tensor]:
    """Pick each setting's poisoned positions of :func:`reference.poison_labels`.

    Args:
      script: The pristine driver.

    Returns:
      record: Keyed as :func:`reference.poison_record`.

    """
    task = driver(script)
    dataset = [(None, label) for label in reference.poison_labels()]
    record: dict[str, Tensor] = {}
    for name, spec in reference.SETTINGS.items():
        random.seed(vision_reference.SEED)
        indices = task.select_poison_indices(dataset, spec.fraction, spec.target)
        record[f"poison/{name}/indices"] = torch.tensor(indices)
        record[f"poison/{name}/python"] = reference.python_fingerprint(
            random.getstate(),
        )
    return record


def trigger_record(script: Upstream) -> dict[str, Tensor]:
    """Stamp each trigger into :func:`reference.trigger_media`, one image at a time.

    Args:
      script: The pristine driver.

    Returns:
      record: Keyed as :func:`reference.trigger_record`.

    """
    task = driver(script)
    mean, std = reference.STATS
    record: dict[str, Tensor] = {}
    media = reference.trigger_media()
    images = [(image, 1) for image in media]
    for trigger in reference.TRIGGERS:
        # Every image poisoned, so indexing the dataset stamps each one.
        dataset = task.PoisonedDataset(images, range(len(media)), 0, trigger, mean, std)
        assert isinstance(dataset, Dataset)
        record[f"trigger/{trigger}"] = torch.stack(
            [_pair(dataset[index])[0] for index in range(len(media))],
        )
    return record


def defense_record(scripts: dict[str, Upstream]) -> dict[str, Tensor]:
    """Fit every variant's defense on the fixed inputs, score them, then a prefix.

    Args:
      scripts: Each variant, keyed as :func:`variants`.

    Returns:
      record: Keyed as :func:`reference.defense_record`.

    """
    record: dict[str, Tensor] = {}
    for name, script in scripts.items():
        features, logits, labels = reference.defense_inputs(name)
        defense = driver(script).BackdoorDefense()
        with host_agnostic_numerics():
            defense.fit(features, labels, reference.DEFENSE_FRACTION)
            record[f"defense/{name}/scores"] = _float64(
                defense.score_samples(features, logits),
            )
            if name != "activation_clustering":
                record[f"defense/{name}/subset"] = _float64(
                    defense.score_samples(
                        features[: reference.DEFENSE_SUBSET],
                        logits[: reference.DEFENSE_SUBSET],
                    ),
                )
    return record


def tiny_network(script: Upstream, setting: str) -> nn.Module:
    """Build and initialize a setting's shallowest network from the task's seed."""
    spec = reference.SETTINGS[setting]
    model = upstream.tiny_model(script, spec.arch, spec.classes)
    driver(script).initialize_weights(model)
    return model


def tiny_record(script: Upstream, setting: str) -> dict[str, Tensor]:
    """Build a setting's shallowest network and run every measurement function on it.

    Args:
      script: The pristine driver.
      setting: A key of :data:`reference.SETTINGS`.

    Returns:
      record: Keyed as :func:`reference.tiny_record`.

    """
    task = driver(script)
    spec = reference.SETTINGS[setting]
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        model = tiny_network(script, setting)
        draws = rng_fingerprint()
    with upstream.synthetic_datasets(script, spec.dataset):
        _, test_set, _, test_loader, _ = task.get_dataloaders(
            spec.dataset,
            "unused",
            vision_reference.LOADER_BATCH,
            num_workers=0,
        )
    (batch,) = reference.fixed_batches()
    cpu = torch.device("cpu")
    mean, std = STATS[spec.dataset]
    with host_agnostic_numerics():
        features, logits, labels = task.collect_features_logits(
            model,
            [(batch["media"], batch["label"])],
            cpu,
        )
        _, accuracy = task.evaluate(model, test_loader, nn.CrossEntropyLoss(), cpu)
        asr = task.evaluate_asr(
            model,
            test_set,
            cpu,
            vision_reference.LOADER_BATCH,
            target_class=spec.target,
            trigger=spec.trigger,
            mean=mean,
            std=std,
        )
    key = f"tiny/{setting}"
    return {
        f"{key}/draws": draws,
        f"{key}/features": torch.from_numpy(features),
        f"{key}/logits": torch.from_numpy(logits),
        f"{key}/labels": torch.from_numpy(labels),
        f"{key}/clean_acc": torch.tensor(accuracy / 100.0, dtype=torch.float64),
        f"{key}/asr": torch.tensor(asr, dtype=torch.float64),
    }


def setting_record(
    script: Upstream,
    setting: str,
) -> tuple[nn.Module, Tensor, dict[str, Tensor]]:
    """Build a setting's full network from the task's seed, and run its shallowest.

    Args:
      script: The pristine driver.
      setting: A key of :data:`reference.SETTINGS`.

    Returns:
      model: The initialized full network, compared live by the capture.
      draws: Where the generator stands after the full build.
      record: Keyed as the vision port's ``setting_record`` under ``setting``.

    """
    task = driver(script)
    spec = reference.SETTINGS[setting]
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        model = task.build_model(spec.arch, spec.classes)
        task.initialize_weights(model)
        draws = rng_fingerprint()
    record = upstream.tiny_setting_record(
        script,
        spec.arch,
        setting,
        build=_initialized_tiny,
    )
    return model, draws, record


def lockstep(script: Upstream, trace: DriverTrace) -> Lockstep:
    """Train the shallowest ResNet for two epochs with the driver's ``train_epoch``.

    Args:
      script: The pristine driver.
      trace: Source of the optimizer settings ``main`` used.

    Returns:
      lockstep: Per update the loss, its logits and a probe's logits
        afterwards; and the network's whole state after each update.

    """
    task = driver(script)
    settings = trace.networks[0].settings
    criterion = upstream.CriterionSpy()
    probes: list[Tensor] = []
    states: list[dict[str, Tensor]] = []
    with vision_reference.seeded(vision_reference.SEED), host_agnostic_numerics():
        with upstream.narrowed_widths(script):
            model = task.ResNet(task.BasicBlock, [1, 1, 1], vision_reference.CLASSES)
        task.initialize_weights(model)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=settings["lr"],
            momentum=settings["momentum"],
            dampening=settings["dampening"],
            weight_decay=settings["weight_decay"],
            nesterov=bool(settings["nesterov"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=vision_reference.EPOCHS,
        )
        for index in range(vision_reference.STEPS):
            batch = [
                (
                    vision_reference.fixed_media(index),
                    vision_reference.fixed_label(index),
                ),
            ]
            task.train_epoch(model, batch, criterion, optimizer, torch.device("cpu"))
            model.eval()
            with torch.no_grad():
                probes.append(
                    vision_reference.model_logits(
                        model,
                        vision_reference.fixed_media(vision_reference.STEPS),
                    ),
                )
            states.append({k: v.clone() for k, v in model.state_dict().items()})
            scheduler.step()
    record = {
        "lockstep/loss": torch.stack(criterion.losses),
        "lockstep/logits": torch.stack(criterion.logits),
        "lockstep/probe": torch.stack(probes),
    }
    return record, states


def poisoned_batches(
    script: Upstream,
    setting: str,
    *,
    num_workers: int = 4,
) -> tuple[Batches, Batches]:
    """Run the driver's poisoned training loader twice, then its analysis loader.

    The loaders are built as ``main`` builds them, over the synthetic split
    poisoned at :data:`reference.LOADER_FRACTION`.

    Args:
      script: The pristine driver.
      setting: A key of :data:`reference.SETTINGS`.
      num_workers: The driver's four; zero keeps the transforms' draws in this
        process, where they can be counted.

    Returns:
      passes: The two shuffled passes, then the one in order.

    """
    task = driver(script)
    spec = reference.SETTINGS[setting]
    mean, std = STATS[spec.dataset]
    with upstream.synthetic_datasets(script, spec.dataset):
        train_set, _, _, _, _ = task.get_dataloaders(
            spec.dataset,
            "unused",
            vision_reference.LOADER_BATCH,
            num_workers=0,
        )
    assert isinstance(train_set, upstream.FakeCifar100)
    labels = from_plain(train_set.label.tolist(), list[int])
    random.seed(vision_reference.SEED)
    indices = task.select_poison_indices(
        [(None, label) for label in labels],
        reference.LOADER_FRACTION,
        spec.target,
    )
    poisoned = task.PoisonedDataset(
        train_set,
        indices,
        spec.target,
        spec.trigger,
        mean,
        std,
    )
    assert isinstance(poisoned, Dataset)
    train: Batches = []
    analysis: Batches = []
    with (
        vision_reference.seeded(vision_reference.SEED),
        vision_reference.script_workers(),
        upstream.forked_workers(),
    ):
        for shuffle, passes in ((True, train), (True, train), (False, analysis)):
            loader = DataLoader(
                poisoned,
                batch_size=vision_reference.LOADER_BATCH,
                shuffle=shuffle,
                num_workers=num_workers,
            )
            passes.append(
                [
                    {"media": media, "label": label}
                    for media, label in map(_pair, loader)
                ],
            )
    return train, analysis


def upstream_sites(script: Upstream) -> dict[str, int]:
    """Count the torch draws of a tiny network and a poisoned loader run, in this process.

    Args:
      script: The pristine driver.

    Returns:
      counts: As :func:`reference.draw_sites`.

    """
    with recorded_draws() as counts:
        _ = tiny_record(script, reference.FIRST)
        _ = poisoned_batches(script, reference.FIRST, num_workers=0)
    return counts


class _EpochSpy:
    """Stands in for the driver's ``train_epoch`` and wraps its ``evaluate``."""

    def __init__(
        self,
        trace: DriverTrace,
        *,
        evaluate: Callable[..., tuple[float, float]],
    ) -> None:
        self.trace = trace
        self.model: nn.Module | None = None
        self.evaluate_fn = evaluate

    def train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader[tuple[Tensor, int]],
        criterion: object,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[float, float]:
        """Record a new network's data and optimizer, and every epoch's rate.

        Args:
          model: The network ``main`` trains; a new one starts a new record.
          loader: Its loader, whose dataset names the images trained on.
          criterion: Unused.
          optimizer: Its optimizer, whose settings and rate are recorded.
          device: Unused.

        Returns:
          result: Zero loss and zero accuracy, as no training runs.

        """
        del criterion, device
        group = reference.optimizer_group(optimizer.param_groups)
        if model is not self.model:
            self.model = model
            dataset = _attributes(loader.dataset)
            if not self.trace.is_poison:
                self.trace.is_poison = from_plain(dataset["is_poison"], list[bool])
            everything = list(range(len(self.trace.is_poison)))
            self.trace.networks.append(
                Network(
                    settings={
                        name: _number(group[name])
                        for name in vision_reference.HYPERPARAMETERS
                        if name in group
                    },
                    indices=from_plain(dataset.get("indices", everything), list[int]),
                ),
            )
        self.trace.networks[-1].rates.append(from_plain(group["lr"], float))
        return 0.0, 0.0

    def evaluate(self, *args: object) -> tuple[float, float]:
        """Evaluate, noting the epoch of the network being trained."""
        network = self.trace.networks[-1]
        epoch = len(network.rates)
        if epoch not in network.evaluated:
            network.evaluated.append(epoch)
        return self.evaluate_fn(*args)


class _Recorded:
    """Builds the driver's defense, wrapped to record what it is given and returns."""

    def __init__(self, task: Driver, trace: DriverTrace) -> None:
        self.build = task.BackdoorDefense
        self.trace = trace

    def __call__(self) -> _RecordingDefense:
        """Return a fresh recording defense."""
        return _RecordingDefense(self.build(), self.trace)


class _RecordingDefense:
    """Forwards to a defense and records its inputs and scores."""

    def __init__(self, defense: Defense, trace: DriverTrace) -> None:
        self.defense = defense
        self.trace = trace

    def fit(
        self,
        features: NDArray[np.float32],
        labels: NDArray[np.int64],
        poison_fraction: float,
    ) -> None:
        """Record the features and labels, then fit as the port's comparison does."""
        self.trace.features = torch.from_numpy(features.copy())
        self.trace.labels = torch.from_numpy(labels.copy())
        with host_agnostic_numerics():
            self.defense.fit(features, labels, poison_fraction)

    def score_samples(
        self,
        features: NDArray[np.float32],
        logits: NDArray[np.float32],
    ) -> NDArray[np.floating]:
        """Score as the port's comparison does, and record the logits and the scores."""
        with host_agnostic_numerics():
            scores = self.defense.score_samples(features, logits)
        self.trace.logits = torch.from_numpy(logits.copy())
        self.trace.scores = _float64(scores)
        return scores


class _MainSplit(upstream.FakeCifar100):
    """:func:`reference.main_split` in color, served as torchvision serves CIFAR."""

    def __init__(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.media, self.label = reference.main_split("cifar10")


class _MainGraySplit(upstream.FakeCifar100):
    """:func:`reference.main_split` in grayscale, served as torchvision serves Fashion-MNIST."""

    def __init__(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.media, self.label = reference.main_split("fmnist")


def _initialized_tiny(script: Upstream, arch: str) -> nn.Module:
    """Build an ``--arch``'s shallowest network and initialize it as the driver does."""
    model = upstream.tiny_model(script, arch, vision_reference.CLASSES)
    driver(script).initialize_weights(model)
    return model


@contextmanager
def _main_split(script: Upstream, dataset: str) -> Generator[None]:
    """Make the driver's torchvision class for ``dataset`` serve :func:`reference.main_split`."""
    fake = _MainGraySplit if dataset == "fmnist" else _MainSplit
    name = upstream.DATASETS[dataset]
    with mock.patch.object(script.torchvision.datasets, name, fake):
        yield


def _float64(scores: NDArray[np.floating]) -> Tensor:
    """Return scores as the driver takes them: a float64 tensor."""
    return torch.from_numpy(np.asarray(scores, dtype=np.float64))


def _number(value: object) -> float:
    """Return an optimizer setting, a number or a flag, as a float."""
    assert isinstance(value, (int, float))
    return float(value)


def _pair(item: object) -> tuple[Tensor, Tensor]:
    """Return an image-and-label item of the driver's dataset or loader as tensors."""
    media, label = from_plain(item, tuple[Tensor, Tensor | int])
    return media, torch.as_tensor(label)


def _attributes(value: object) -> Mapping[str, object]:
    """Return an object's instance attributes, each ours to narrow."""
    attributes: Mapping[str, object] = vars(value)
    return attributes


# ``mock.patch.dict(sys.modules)`` would restore the whole table and drop the
# torchvision submodules the driver's import loads, whose re-import then fails to
# register its operators twice; only the defense's own key is swapped.
@contextmanager
def _defense_module(module: object) -> Generator[None]:
    """Make ``import custom_backdoor_defense`` inside the block return ``module``."""
    assert isinstance(module, ModuleType)
    previous = sys.modules.get("custom_backdoor_defense")
    sys.modules["custom_backdoor_defense"] = module
    try:
        yield
    finally:
        if previous is None:
            del sys.modules["custom_backdoor_defense"]
        else:
            sys.modules["custom_backdoor_defense"] = previous
