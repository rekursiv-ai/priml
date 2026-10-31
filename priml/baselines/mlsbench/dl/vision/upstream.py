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
import inspect
import io
import multiprocessing
import sys
import warnings

from PIL import Image
from torch import Tensor, nn
from treekle import from_plain

import torch

from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.data import Cifar100Data
from priml.baselines.mlsbench.dl.vision.draws import recorded_draws
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable
    from types import ModuleType

    from torch.utils.data import DataLoader, Dataset

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__cv-classification-loss/tests/meta/pristine"
    "/pytorch-vision/custom_loss.py",
)
"""The unedited ``cv-classification-loss`` script, relative to the clone."""

DATASETS: Final = {
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "fmnist": "FashionMNIST",
}
"""The torchvision class each ``--dataset`` the loaders are run with reads."""


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
    VGG: type[nn.Module]
    MobileNetV2: type[nn.Module]
    torchvision: _Torchvision

    ResNet: Callable[[type[nn.Module], list[int], int], nn.Module]

    def build_model(self, arch: str, num_classes: int) -> nn.Module:
        """Build the network an ``--arch`` names."""
        raise NotImplementedError

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
        ...

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
    return cast(Upstream, _load_module(path))


def narrowed[T](script: object, protocol: type[T]) -> T:
    """Return an imported script as the task's ``protocol``, checking it has its members.

    Args:
      script: A script :func:`load_script` or :func:`edited_script` imported.
      protocol: A ``runtime_checkable`` Protocol of the members the task calls.

    Returns:
      script: The same module, typed as ``protocol``.

    Raises:
      TypeError: The script lacks a member of ``protocol``.

    """
    if not isinstance(script, protocol):
        msg = f"{script!r} lacks a member of {protocol.__name__}."
        raise TypeError(msg)
    return script


def edited_script(
    root: Path,
    name: str,
    scratch: Path,
    *,
    pristine: Path,
    edits: Path,
) -> Upstream:
    """Apply a baseline's edit to the pristine script and import the result.

    Args:
      root: Clone of the benchmark.
      name: The edit's file stem, under ``edits``.
      scratch: Directory the edited script is written to.
      pristine: The task's unedited script, relative to the clone.
      edits: The task's edit directory, relative to the clone.

    Returns:
      script: The pristine script with the edit's lines replaced.

    """
    ops = cast(
        list[dict[str, object]],
        cast(object, vars(_load_module(root / edits / f"{name}.edit.py"))["OPS"]),
    )
    (op,) = ops
    lines = (root / pristine).read_text().splitlines(keepends=True)
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


class ForkedLoader:
    """Builds a ``DataLoader`` whose workers, when it has any, are forked.

    A script whose dataset or transform class is defined in a file it was
    imported from, or inside a function, cannot be pickled into a
    ``forkserver`` worker; a forked worker inherits it. The start method changes
    no draw: each worker's seed comes from the parent.
    """

    def __init__(
        self,
        loader: Callable[..., DataLoader[tuple[Tensor, Tensor]]],
    ) -> None:
        self.loader = loader

    def __call__(
        self,
        dataset: Dataset[tuple[Tensor, Tensor]],
        *,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader[tuple[Tensor, Tensor]]:
        """Build the loader, forked when ``num_workers`` asks for workers."""
        return self.loader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context="fork" if num_workers else None,
        )


@contextmanager
def forked_workers() -> Generator[None]:
    """Start a script's loader workers by forking, as Linux did before Python 3.14.

    The reference's Fashion-MNIST pipeline holds a lambda, which the
    ``forkserver`` default of Python 3.14 cannot pickle into a worker. A forked
    worker inherits it instead, and its draws are the same: each worker seeds
    itself from the loader's base seed and its index, whatever started it.

    Yields:
      nothing: Used as a context manager.

    """
    previous = multiprocessing.get_start_method()
    multiprocessing.set_start_method("fork", force=True)
    try:
        with warnings.catch_warnings():
            # Python warns that forking a multi-threaded process may deadlock;
            # the workers only index the dataset and run its transforms.
            warnings.filterwarnings("ignore", message=".*use of fork\\(\\) may lead.*")
            yield
    finally:
        multiprocessing.set_start_method(previous, force=True)


def trace_main(
    script: Upstream,
    *,
    arch: str = "resnet56",
    dataset: str = "cifar100",
    classes: int = 100,
) -> MainTrace:
    """Run the reference's ``main`` for 200 epochs of one setting.

    Only the data loaders and the two epoch functions are replaced.

    Args:
      script: A pytorch-vision task script, pristine or edited.
      arch: The setting's ``--arch``.
      dataset: The setting's ``--dataset``.
      classes: What the replaced loaders report for ``dataset``.

    Returns:
      trace: What the replaced calls saw.

    """
    trace = MainTrace()
    spy = _MainSpy(trace, classes=classes)
    argv = ["script.py", "--arch", arch, "--dataset", dataset]
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
    script: Upstream,
    trace: MainTrace,
    train_batches: Batches,
) -> dict[str, Tensor]:
    """Return the reference's value of every quantity the default test replays.

    Args:
      script: The pristine reference script.
      trace: A run of its ``main``.
      train_batches: The training passes of :func:`upstream_batches`.

    Returns:
      record: Keyed as the port's records are.

    """
    record: dict[str, Tensor] = {}
    record["schedule/lr"] = torch.tensor(trace.rates, dtype=torch.float64)
    record["optimizer/settings"] = torch.tensor(
        [trace.settings[name] for name in reference.HYPERPARAMETERS],
        dtype=torch.float64,
    )
    record["draws/exp000"] = trace.draws
    record["best/accuracy"] = torch.tensor(trace.best, dtype=torch.float64)
    record |= upstream_network(script)
    record |= upstream_pipelines(script)
    record |= reference.decode_batches(
        train_batches,
        config=Cifar100Data.Config(),
        source=reference.synthetic_split()[0],
    )
    return record


def initialized_tiny(script: Upstream, arch: str) -> nn.Module:
    """Build the reference's shallowest network of an ``--arch`` and initialize it."""
    model = tiny_model(script, arch, reference.CLASSES)
    script.initialize_weights(model)
    return model


def upstream_setting(
    script: Upstream,
    name: str,
    *,
    key: str | None = None,
    build: Callable[[Upstream, str], nn.Module] = initialized_tiny,
) -> tuple[MainTrace, dict[str, Tensor]]:
    """Trace ``main`` at a setting, and initialize and run its shallowest network.

    Args:
      script: A pytorch-vision task script, pristine or edited.
      name: A key of :data:`reference.SETTINGS`.
      key: Names the record; ``name`` by default.
      build: Builds and initializes the shallowest network of an ``--arch``,
        as the task's script initializes.

    Returns:
      trace: The run of ``main``; its state is the full network's after
        initialization.
      record: Keyed as :func:`reference.setting_record`.

    """
    setting = reference.SETTINGS[name]
    trace = trace_main(
        script,
        arch=setting.arch,
        dataset=setting.dataset,
        classes=setting.classes,
    )
    return trace, tiny_setting_record(script, setting.arch, key or name, build=build)


def tiny_setting_record(
    script: Upstream,
    arch: str,
    key: str,
    *,
    build: Callable[[Upstream, str], nn.Module] = initialized_tiny,
) -> dict[str, Tensor]:
    """Build an ``--arch``'s shallowest network from the task's seed and run it.

    Args:
      script: A task script, pristine or edited, or a driver.
      arch: The setting's ``--arch``.
      key: Names the record.
      build: As :func:`upstream_setting`.

    Returns:
      record: Keyed as :func:`reference.setting_record` under ``key``.

    """
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = build(script, arch)
        draws = rng_fingerprint()
    return {f"settings/{key}/draws": draws} | reference.setting_forward(model, key)


def tiny_model(script: Upstream, arch: str, classes: int) -> nn.Module:
    """Build the reference's shallowest network of an ``--arch``, uninitialized.

    A VGG or MobileNetV2 is built from the table of
    :data:`reference.TINY_ARCHS`, patched in only for the build; a ResNet has
    one block per stage. Every hardcoded width is narrowed by
    :func:`narrowed_widths`. The caller runs ``initialize_weights``.

    Args:
      script: A pytorch-vision task script, pristine or edited.
      arch: The reference's ``--arch``.
      classes: Logits.

    Returns:
      model: The network, as torch's constructors initialized it.

    """
    if arch == "vgg16bn":
        table, field = script.VGG, "VGG16_CFG"
    elif arch == "mobilenetv2":
        table, field = script.MobileNetV2, "CFG"
    else:
        with narrowed_widths(script):
            return script.ResNet(script.BasicBlock, [1, 1, 1], classes)
    with (
        mock.patch.object(table, field, reference.TINY_ARCHS[arch]),
        narrowed_widths(script),
    ):
        return script.build_model(arch, classes)


@contextmanager
def narrowed_widths(script: object) -> Generator[None]:
    """Build every layer with the widths of :data:`reference.TINY_WIDTHS`.

    The reference writes ResNet's stage widths, MobileNetV2's stem and last
    convolution and VGG's classifier as literals, so no table reaches them.
    Each layer is built through ``nn`` or, in ``dl-normalization``, the
    script's ``CustomNorm``; while patched, each narrows any argument equal to
    a hardcoded width. ``nn``'s layers are not classes while patched, so only
    construction may run inside: an ``isinstance`` against them raises.

    Args:
      script: The script being built from.

    Yields:
      nothing: Used as a context manager.

    """
    with contextlib.ExitStack() as stack:
        for name in ("Conv2d", "BatchNorm2d", "Linear"):
            layer = cast("Callable[..., nn.Module]", getattr(nn, name))
            stack.enter_context(mock.patch.object(nn, name, _Narrowed(layer)))
        norm = cast(object, vars(script).get("CustomNorm"))
        if isinstance(norm, type):
            narrowed = _Narrowed(cast("Callable[..., nn.Module]", norm))
            stack.enter_context(mock.patch.object(script, "CustomNorm", narrowed))
        yield


def upstream_settings(
    script: Upstream,
) -> tuple[dict[str, MainTrace], dict[str, Tensor]]:
    """Run every setting's network and loaders, and the Fashion-MNIST pipelines.

    Args:
      script: The pristine reference script.

    Returns:
      traces: Each setting's run of ``main``.
      record: Keyed as :func:`reference.settings_record`.

    """
    traces: dict[str, MainTrace] = {}
    record = upstream_fmnist_pipelines(script)
    for name in reference.CAPTURED:
        setting = reference.SETTINGS[name]
        traces[name], network = upstream_setting(script, name)
        record |= network
        train, _ = upstream_batches(script, dataset=setting.dataset)
        decoded = reference.decode_batches(
            train,
            config=setting.experiment().dataset,
            source=setting.split()[0],
        )
        record |= {f"settings/{name}/{key}": value for key, value in decoded.items()}
    return traces, record


def upstream_network(script: Upstream) -> dict[str, Tensor]:
    """Initialize the shallowest network at ``reference.TINY_WIDTHS`` and run it.

    Args:
      script: The pristine reference script.

    Returns:
      record: Keyed as :func:`reference.network_record`.

    """
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = tiny_model(script, "resnet56", reference.CLASSES)
        script.initialize_weights(model)
        draws = rng_fingerprint()
        media = reference.fixed_media(reference.STEPS)
        model.train()
        train = reference.model_logits(model, media)
        model.eval()
        evaluated = reference.model_logits(model, media)
    return {
        "network/draws": draws,
        "network/train": train.detach(),
        "network/eval": evaluated.detach(),
    }


def default_workers(get_dataloaders: Callable[..., object]) -> int:
    """Return the ``num_workers`` a script's ``get_dataloaders`` uses by default.

    Each worker seeds its own transforms, so the draws a pass makes depend on
    the count; it is read from the script, never restated.

    Args:
      get_dataloaders: The script's loader factory.

    Returns:
      workers: Its ``num_workers`` default.

    """
    parameter = inspect.signature(get_dataloaders).parameters["num_workers"]
    return from_plain(cast(object, parameter.default), int)


def upstream_batches(
    script: Upstream,
    *,
    dataset: str = "cifar100",
    num_workers: int | None = None,
) -> tuple[Batches, Batches]:
    """Run the reference's loaders, alternating training and test passes.

    Args:
      script: The pristine reference script.
      dataset: The setting's ``--dataset``; its loaders read the synthetic split.
      num_workers: Loader workers; None takes the script's own default, whose
        seeds the draws depend on.

    Returns:
      passes: The training passes, then the test passes.

    """
    train: Batches = []
    evaluation: Batches = []
    with (
        reference.seeded(reference.SEED),
        reference.script_workers(),
        forked_workers(),
    ):
        train_loader, test_loader = _loaders(
            script,
            num_workers=(
                default_workers(script.get_dataloaders)
                if num_workers is None
                else num_workers
            ),
            dataset=dataset,
        )
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
    train, test = _pipelines(script, reference.fixed_images(2), dataset="cifar100")
    return {"transform/train": train, "transform/test": test}


def upstream_fmnist_pipelines(script: Upstream) -> dict[str, Tensor]:
    """Run both of the reference's Fashion-MNIST pipelines on two fixed 28x28 images.

    Args:
      script: The pristine reference script.

    Returns:
      record: Keyed as :func:`reference.fmnist_transform_record`.

    """
    images = reference.fixed_images(2, channels=1, side=28)
    train, test = _pipelines(script, images, dataset="fmnist")
    return {"transform/fmnist/train": train, "transform/fmnist/test": test}


def upstream_fmnist_sites(script: Upstream) -> dict[str, int]:
    """Count the draws of building MobileNetV2 and of the Fashion-MNIST data path.

    Args:
      script: The pristine reference script.

    Returns:
      counts: As :func:`reference.fmnist_draw_sites`; the loaders run in this
        process so the transforms' draws are seen.

    """
    setting = reference.SETTINGS["mobilenetv2-fmnist"]
    with recorded_draws() as counts:
        with reference.seeded(reference.SEED):
            script.initialize_weights(script.build_model(setting.arch, setting.classes))
        _ = upstream_fmnist_pipelines(script)
        train_loader, test_loader = _loaders(script, num_workers=0, dataset="fmnist")
        for _ in range(reference.LOADER_PASSES):
            _ = list(train_loader)
            _ = list(test_loader)
    return counts


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


class CriterionSpy:
    """A script's criterion, unweighted cross-entropy by default, keeping what it saw."""

    def __init__(self, criterion: nn.CrossEntropyLoss | None = None) -> None:
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.losses: list[Tensor] = []
        self.logits: list[Tensor] = []

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Return the batch's loss."""
        loss = self.criterion(logits, targets)
        self.losses.append(loss.detach())
        self.logits.append(logits.detach())
        return loss


class FakeCifar100:
    """The synthetic split, standing in for torchvision's CIFAR classes.

    Takes the keywords the reference's ``get_dataloaders`` passes and serves
    :func:`reference.synthetic_split` through the script's transform, as a PIL
    image, the way torchvision's class does.

    Args:
      root: Ignored.
      train: Ignored; both splits are the synthetic split.
      download: Ignored.
      transform: The script's pipeline for one PIL image.

    """

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
        """Return the images in the split."""
        return len(self.media)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Return image ``index`` through the transform, and its label."""
        return self.transform(_pil(self.media[index])), int(self.label[index])


class FakeFashionMNIST(FakeCifar100):
    """The grayscale synthetic split, served as torchvision serves Fashion-MNIST.

    Each image is a 28x28 ``L``-mode PIL image of
    :func:`reference.synthetic_gray_split`.
    """

    def __init__(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        super().__init__(root=root, train=train, download=download, transform=transform)
        self.media, self.label = reference.synthetic_gray_split()


@contextmanager
def synthetic_datasets(script: Upstream, dataset: str) -> Generator[None]:
    """Make a script's torchvision class for ``dataset`` serve the synthetic split.

    Inside the block, the class :data:`DATASETS` names on the script's own
    ``torchvision.datasets`` is :class:`FakeFashionMNIST` for ``fmnist`` and
    :class:`FakeCifar100` otherwise; it is restored on exit.

    Args:
      script: A pytorch-vision task script.
      dataset: The setting's ``--dataset``.

    Yields:
      nothing: Used as a context manager.

    """
    fake = FakeFashionMNIST if dataset == "fmnist" else FakeCifar100
    with mock.patch.object(script.torchvision.datasets, DATASETS[dataset], fake):
        yield


def _pil(image: Tensor) -> Image.Image:
    """Return a ``(C, H, W)`` uint8 image as PIL holds it: RGB, or ``L`` for one channel."""
    return Image.fromarray(image.permute(1, 2, 0).squeeze(-1).contiguous().numpy())


def _pipelines(
    script: Upstream,
    images: Tensor,
    *,
    dataset: str,
) -> tuple[Tensor, Tensor]:
    """Run a dataset's training pipeline, seeded, and its test pipeline on ``images``."""
    train_loader, test_loader = _loaders(script, num_workers=0, dataset=dataset)
    train_transform = _transform(train_loader)
    test_transform = _transform(test_loader)
    with reference.seeded(reference.SEED):
        train = torch.stack([train_transform(_pil(image)) for image in images])
    test = torch.stack([test_transform(_pil(image)) for image in images])
    return train, test


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
    dataset: str = "cifar100",
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Build the reference's loaders over the synthetic split."""
    with synthetic_datasets(script, dataset):
        train, test, _ = script.get_dataloaders(
            dataset,
            "unused",
            batch_size=reference.LOADER_BATCH,
            num_workers=num_workers,
        )
    return train, test


def _transform(
    loader: DataLoader[tuple[Tensor, Tensor]],
) -> Callable[[Image.Image], Tensor]:
    """Return the transform a loader's dataset applies."""
    return cast(FakeCifar100, loader.dataset).transform


class _MainSpy:
    """Stands in for the reference's heavy calls while its ``main`` runs."""

    def __init__(self, trace: MainTrace, *, classes: int) -> None:
        self.trace = trace
        self.classes = classes
        self.scripted = iter(reference.SCRIPTED_CORRECT)

    def get_dataloaders(
        self,
        dataset: str,
        data_root: str,
        batch_size: int = 0,
        num_workers: int = 0,
    ) -> tuple[list[object], list[object], int]:
        """Return no data and the setting's class count."""
        del dataset, data_root, batch_size, num_workers
        return [], [], self.classes

    def train_epoch(
        self,
        model: nn.Module,
        loader: object,
        *rest: object,
    ) -> tuple[float, float]:
        """Record the state before the first epoch and the rate of every one.

        The tasks' scripts order the remaining arguments differently, so the
        optimizer and the loss configuration are found by type.

        Args:
          model: The network, initialized.
          loader: Unused.
          *rest: The script's other arguments: the optimizer whose rate and
            settings are recorded, and the dict told to the loss, if any.

        Returns:
          result: Zero loss and zero accuracy.

        """
        del loader
        (optimizer,) = (a for a in rest if isinstance(a, torch.optim.Optimizer))
        configs = [cast(dict[str, int], a) for a in rest if isinstance(a, dict)]
        config = configs[0] if configs else {}
        group = cast(dict[str, object], optimizer.param_groups[0])
        if not self.trace.rates:
            self.trace.state = {k: v.clone() for k, v in model.state_dict().items()}
            self.trace.draws = rng_fingerprint()
            self.trace.settings = {
                name: float(cast(float, group[name]))
                for name in reference.HYPERPARAMETERS
                if name in group
            }
        self.trace.rates.append(from_plain(group["lr"], float))
        self.trace.configs.append(dict(config))
        return 0.0, 0.0

    def evaluate(self, *args: object) -> tuple[float, float]:
        """Report the next scripted accuracy, whatever the script passes."""
        del args
        return 0.0, 100.0 * next(self.scripted) / reference.SCRIPTED_TOTAL


class _Narrowed:
    """Builds a layer with each hardcoded width among its arguments narrowed."""

    def __init__(self, layer: Callable[..., nn.Module]) -> None:
        self.layer = layer

    def __call__(self, *args: object, **kwargs: object) -> nn.Module:
        """Build the layer from the narrowed arguments."""
        return self.layer(
            *map(_narrow, args),
            **{key: _narrow(value) for key, value in kwargs.items()},
        )


def _narrow(value: object) -> object:
    """Return a hardcoded width's tiny width; any other argument as it is."""
    if isinstance(value, int):
        return reference.TINY_WIDTHS.get(value, value)
    return value
