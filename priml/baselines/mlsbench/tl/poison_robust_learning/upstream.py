"""Runs of the pristine ``security-poison-robust-learning`` driver, under the capture's pins.

Nothing is vendored: the driver and each baseline's edit of its loss are
imported from a clone of the benchmark. The driver imports ``RobustLoss`` by
module name, so each variant's driver is imported with that variant's loss in
its place.

``main`` runs as written over a synthetic split, its loaders with their four
worker processes, forked. Recorders wrap what it calls: the network builder,
the loss, the scheduler (which hooks every update), the network's evaluation
forwards and both evaluations. Nothing runs on a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast
from unittest import mock

import contextlib
import io
import random
import sys

from PIL import Image
from torch import Tensor, nn

import numpy as np
import torch

from priml.baselines.mlsbench.dl.vision import reference, upstream
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Sequence
    from types import ModuleType

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__security-poison-robust-learning/tests/meta"
    "/pristine/pytorch-vision/bench/poison",
)
"""Directory of the unedited driver and loss, relative to the clone."""

DRIVER: Final = PRISTINE / "run_poison_robust.py"
"""The fixed driver: networks, data, poisoning, training loop and evaluation."""

LOSS: Final = PRISTINE / "custom_robust_loss.py"
"""The editable loss."""

EDITS: Final = Path("tasks/security-poison-robust-learning/edits")
"""Directory of the task's baseline edits, relative to the clone."""

EDITED: Final = ("cross_entropy", "generalized_ce", "symmetric_ce", "bootstrap")
"""The published baselines, each named as its edit file and as its port's variant."""

SETTINGS: Final = {
    "resnet20-cifar10": ("resnet20", "cifar10", "0.10"),
    "vgg16bn-cifar100": ("vgg16bn", "cifar100", "0.10"),
    "mobilenetv2-fmnist": ("mobilenetv2", "fmnist", "0.15"),
}
"""Each setting's ``--arch``, ``--dataset`` and ``--poison-fraction``, as its script passes them."""

LOCKSTEP_EPOCHS: Final = 2
"""Epochs of a lockstep run, so the second runs at an annealed rate."""

LOCKSTEP_BATCH: Final = reference.LOADER_IMAGES
"""Batch size of a lockstep run: four batches per pass over the synthetic split."""

LOCKSTEP_FRACTION: Final = "0.5"
"""Poison fraction of a lockstep run: three of the synthetic split's seven images."""


class Loss(Protocol):
    """The editable ``RobustLoss``."""

    def compute_loss(self, logits: Tensor, labels: Tensor, epoch: int) -> Tensor:
        """Return the scalar training loss."""
        raise NotImplementedError


class Poisoned(Protocol):
    """The driver's ``PoisonedDataset``."""

    labels: Tensor
    poison_idx: Tensor
    is_poison: Tensor

    def __len__(self) -> int:
        """Return the images of the wrapped split."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[object, int]:
        """Return an image and its possibly flipped label."""
        raise NotImplementedError


class _Torchvision(Protocol):
    """The part of torchvision the driver reaches through its own import."""

    datasets: ModuleType


class Script(Protocol):
    """The part of the driver the capture uses."""

    RobustLoss: Callable[[], Loss]
    torchvision: _Torchvision
    DataLoader: type[torch.utils.data.DataLoader[tuple[Tensor, Tensor]]]

    def PoisonedDataset(  # noqa: N802 -- The driver's class, called by its name.
        self,
        base_dataset: object,
        num_classes: int,
        poison_fraction: float,
        seed: int,
    ) -> Poisoned:
        """Wrap a split with its labels flipped."""
        raise NotImplementedError

    def build_model(self, arch: str, num_classes: int) -> nn.Module:
        """Build the network an ``--arch`` names."""
        raise NotImplementedError

    def evaluate_test_acc(
        self,
        model: nn.Module,
        loader: Iterable[tuple[Tensor, Tensor]],
        device: torch.device,
    ) -> float:
        """Return the accuracy on ``loader``."""
        raise NotImplementedError

    def evaluate_poison_fit(
        self,
        model: nn.Module,
        poisoned_dataset: object,
        device: torch.device,
        batch_size: int,
    ) -> float:
        """Return the fraction of flipped images predicted as their flipped label."""
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
    """Every update's targets."""

    epochs: list[int] = field(default_factory=list[int])
    """The epoch each update told the loss."""

    t_max: int = 0
    """The scheduler's horizon."""

    poisoned: list[Poisoned] = field(default_factory=list[Poisoned])
    """Every ``PoisonedDataset`` built, in construction order."""

    loaders: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    """Each loader's keyword arguments, when loaders are recorded."""

    evaluated: list[tuple[Tensor, Tensor]] = field(
        default_factory=list[tuple[Tensor, Tensor]],
    )
    """Every evaluation-mode forward's images and logits, in order."""

    results: list[float] = field(default_factory=list[float])
    """The test accuracy, then the poison fit."""

    generators: Generators | None = None
    """Python's and NumPy's generator states after ``main``."""

    end: Tensor = field(default_factory=lambda: torch.empty(0))
    """Where the global generator stood after ``main``."""

    models: list[nn.Module] = field(default_factory=list[nn.Module])
    """The network ``main`` built."""

    printed: str = ""
    """Everything ``main`` printed."""


class Builder(Protocol):
    """Builds a network in place of the driver's ``build_model``."""

    def __call__(self, script: Script, arch: str, classes: int, /) -> nn.Module:
        """Return the network, uninitialized."""
        raise NotImplementedError


def variants(root: Path, scratch: Path) -> dict[str, Script]:
    """Import the driver with the pristine loss and with each baseline's edit.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited losses are written to.

    Returns:
      scripts: ``pristine`` and each of :data:`EDITED`.

    """
    losses = {"pristine": upstream.load_script(root / LOSS)}
    for name in EDITED:
        losses[name] = upstream.edited_script(
            root,
            name,
            scratch,
            pristine=LOSS,
            edits=EDITS,
        )
    return {name: driver(root, module) for name, module in losses.items()}


def driver(root: Path, loss: object) -> Script:
    """Import the driver with ``loss`` as its ``custom_robust_loss``.

    Args:
      root: Clone of the benchmark.
      loss: The loss module.

    Returns:
      script: The driver, its ``RobustLoss`` the one ``loss`` defines.

    """
    # Not ``mock.patch.dict(sys.modules)``: restoring the whole table would evict
    # torchvision, whose second import fails registering its kernels again.
    name = "custom_robust_loss"
    previous = sys.modules.get(name)
    sys.modules[name] = cast("ModuleType", loss)
    try:
        return cast(Script, cast(object, upstream.load_script(root / DRIVER)))
    finally:
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous


def recipe_argv(
    name: str,
    *,
    epochs: int = 100,
    batch_size: int = 128,
    fraction: str = "",
) -> list[str]:
    """Return a setting's arguments as its ``scripts/*.sh`` passes them.

    Args:
      name: A key of :data:`SETTINGS`.
      epochs: The ``--epochs``.
      batch_size: The ``--batch-size``.
      fraction: The ``--poison-fraction``; the setting's own when empty.

    Returns:
      argv: The arguments after the script's name.

    """
    arch, dataset, own = SETTINGS[name]
    return [
        *("--arch", arch, "--dataset", dataset, "--data-root", "unused"),
        *("--poison-fraction", fraction or own, "--epochs", str(epochs)),
        *("--batch-size", str(batch_size), "--lr", "0.1", "--momentum", "0.9"),
        *("--weight-decay", "5e-4", "--seed", str(reference.SEED)),
    ]


def trace_main(
    script: Script,
    argv: Sequence[str],
    *,
    build: Builder,
    split: tuple[Tensor, Tensor],
    serial: bool = False,
    stop: bool = False,
) -> Trace:
    """Run ``main`` with ``argv`` over a synthetic split, recording what it did.

    The global generator starts from the task's seed, as the driver's own
    seeding would set it; it, and Python's and NumPy's, are restored
    afterwards.

    Args:
      script: A driver from :func:`variants`.
      argv: The arguments after the script's name.
      build: Builds the network in place of ``build_model``.
      split: Uint8 images and labels both of the driver's splits read.
      serial: Build every loader without workers, noting what it asked for;
        for runs whose pixels are not compared.
      stop: End the run once the scheduler is built, before any image is read.

    Returns:
      trace: What the recorders saw.

    """
    trace = Trace()
    printed = io.StringIO()
    before = generator_states()
    try:
        with (
            reference.seeded(reference.SEED),
            mock.patch.object(sys, "argv", ["script.py", *argv]),
            _recorded(
                script,
                trace,
                build=build,
                split=split,
                serial=serial,
                stop=stop,
            ),
            contextlib.redirect_stdout(printed),
            contextlib.suppress(_Stopped),
            host_agnostic_numerics(),
        ):
            script.main()
            trace.generators = generator_states()
            trace.end = rng_fingerprint()
    finally:
        _restore(before)
    trace.printed = printed.getvalue()
    return trace


@dataclass(frozen=True, slots=True, kw_only=True)
class Generators:
    """Python's and NumPy's legacy generator states, comparable with ``==``."""

    python: tuple[object, ...]
    """``random.getstate()``."""

    numpy: tuple[str, bytes, int, int, float]
    """NumPy's legacy state, its key array as bytes."""


def generator_states() -> Generators:
    """Return Python's and NumPy's legacy generator states.

    Returns:
      states: Both, comparable with ``==``.

    """
    name, keys, position, has_gauss, cached = np.random.get_state(legacy=True)  # noqa: NPY002 -- The driver seeds the legacy generator.
    return Generators(
        python=random.getstate(),
        numpy=(name, keys.tobytes(), position, has_gauss, cached),
    )


def seeded_generator_states() -> Generators:
    """Return :func:`generator_states` right after the driver's ``set_seed``.

    Returns:
      states: What a run that draws from neither generator leaves them at.

    """
    before = generator_states()
    try:
        random.seed(reference.SEED)
        np.random.seed(reference.SEED)  # noqa: NPY002 -- The driver's own seeding.
        return generator_states()
    finally:
        _restore(before)


def lockstep_argv() -> list[str]:
    """Return the ``resnet20-cifar10`` arguments shrunk to a lockstep run.

    Returns:
      argv: :func:`recipe_argv` at :data:`LOCKSTEP_EPOCHS`,
        :data:`LOCKSTEP_BATCH` and :data:`LOCKSTEP_FRACTION`.

    """
    return recipe_argv(
        "resnet20-cifar10",
        epochs=LOCKSTEP_EPOCHS,
        batch_size=LOCKSTEP_BATCH,
        fraction=LOCKSTEP_FRACTION,
    )


def upstream_lockstep(
    script: Script,
    name: str,
) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
    """Run ``main`` on the shallowest ResNet over the synthetic split, recording it.

    Args:
      script: The driver carrying the loss under test.
      name: Which loss, for the record's keys.

    Returns:
      record: Keyed as the port's ``lockstep_record``.
      states: The network's whole state after each update.

    """
    trace = trace_main(
        script,
        lockstep_argv(),
        build=tiny_model,
        split=reference.synthetic_split(),
    )
    test_acc, poison_fit = trace.results
    key = f"lockstep/{name}"
    record = {
        f"{key}/loss": torch.stack(trace.losses),
        f"{key}/logits": torch.cat([x.reshape(-1) for x in trace.logits]),
        f"{key}/label": torch.cat(trace.labels),
        f"{key}/epoch": torch.tensor(trace.epochs),
        f"{key}/lr": torch.tensor(trace.rates, dtype=torch.float64),
        f"{key}/eval": torch.cat([logits.reshape(-1) for _, logits in trace.evaluated]),
        f"{key}/metrics": torch.tensor(
            [test_acc, poison_fit, (test_acc + (1.0 - poison_fit)) / 2.0],
            dtype=torch.float64,
        ),
        f"{key}/draws": trace.end,
    }
    return record, trace.states


def tiny_model(script: Script, arch: str, classes: int, /) -> nn.Module:
    """Build the driver's shallowest network of ``arch``, as the vision capture does."""
    return upstream.tiny_model(cast(upstream.Upstream, script), arch, classes)


def full_model(script: Script, arch: str, classes: int, /) -> nn.Module:
    """Build the driver's own network of ``arch``.

    Called while ``build_model`` is patched, so the driver's own function is
    looked up past the recorder.

    Args:
      script: A driver.
      arch: The ``--arch``.
      classes: Logits.

    Returns:
      model: The network, as torch's constructors initialized it.

    """
    build = script.build_model
    original = build.original if isinstance(build, _Builds) else build
    return original(arch, classes)


def linear_model(script: Script, arch: str, classes: int, /) -> nn.Module:
    """Return a linear classifier of 32x32 RGB images: cheap to train for 100 epochs.

    Args:
      script: Unused.
      arch: Unused.
      classes: Logits.

    Returns:
      model: A flatten and a linear layer.

    """
    del script, arch
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, classes))


def loss_value(
    script: Script,
    logits: Tensor,
    labels: Tensor,
    *,
    epoch: int,
) -> tuple[Tensor, Tensor]:
    """Run the driver's loss on ``logits``, forward and backward.

    Args:
      script: A driver.
      logits: ``(B, C)`` class scores.
      labels: ``(B,)`` targets.
      epoch: The epoch the loss is told.

    Returns:
      loss: The scalar.
      grad: Its gradient with respect to ``logits``.

    """
    leaf = logits.clone().requires_grad_()
    with host_agnostic_numerics():
        loss = script.RobustLoss().compute_loss(leaf, labels, epoch)
        loss.backward()
    assert isinstance(leaf.grad, Tensor)
    return loss.detach(), leaf.grad


def poisoned(
    script: Script,
    labels: Tensor,
    *,
    classes: int,
    fraction: float,
) -> Poisoned:
    """Build the driver's ``PoisonedDataset`` over images-free ``labels``.

    Args:
      script: A driver.
      labels: ``(N,)`` integer labels, served as a torchvision dataset's
        ``targets`` list.
      classes: Classes a flipped label wraps around.
      fraction: The ``--poison-fraction``.

    Returns:
      dataset: The wrapped split, its items ``(index, label)``.

    """
    split = (torch.arange(len(labels)), labels)
    return script.PoisonedDataset(
        Served(split, None),
        classes,
        fraction,
        reference.SEED,
    )


class Served:
    """A synthetic split, in the shape the driver expects of a torchvision dataset.

    Args:
      split: Uint8 ``(N, C, H, W)`` images and ``(N,)`` labels; ``None`` as the
        transform serves the images as they are.
      transform: The dataset's transform.

    """

    def __init__(
        self,
        split: tuple[Tensor, Tensor],
        transform: Callable[[Image.Image], Tensor] | None,
    ) -> None:
        self.media, label = split
        self.targets = [int(value) for value in label]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[object, int]:
        image = self.media[index]
        if self.transform is None:
            return image, self.targets[index]
        return self.transform(
            Image.fromarray(image.permute(1, 2, 0).squeeze(-1).contiguous().numpy()),
        ), self.targets[index]


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
        loader: type[torch.utils.data.DataLoader[tuple[Tensor, Tensor]]],
    ) -> None:
        self.trace = trace
        self.loader = loader

    def __call__(
        self,
        dataset: torch.utils.data.Dataset[tuple[Tensor, Tensor]],
        **kwargs: object,
    ) -> torch.utils.data.DataLoader[tuple[Tensor, Tensor]]:
        self.trace.loaders.append(dict(kwargs))
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
        scheduler: type[torch.optim.lr_scheduler.CosineAnnealingLR],
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
        T_max: int,  # noqa: N803 -- The driver's keyword.
    ) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        scheduler = self.scheduler(optimizer, T_max=T_max)
        self.trace.t_max = T_max
        group = cast(dict[str, object], optimizer.param_groups[0])
        self.trace.settings = {
            name: float(cast(float, group[name])) for name in reference.HYPERPARAMETERS
        }
        model = self.models[0]
        self.trace.initial = {k: v.clone() for k, v in model.state_dict().items()}
        self.trace.draws = rng_fingerprint()
        if self.stop:
            raise _Stopped
        _ = optimizer.register_step_post_hook(_UpdateHook(self.trace, model))
        _ = model.register_forward_hook(_EvalHook(self.trace))
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


class _EvalHook:
    """Notes the images and logits of every evaluation-mode forward."""

    def __init__(self, trace: Trace) -> None:
        self.trace = trace

    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[Tensor],
        output: Tensor,
    ) -> None:
        if not module.training:
            self.trace.evaluated.append((inputs[0].clone(), output.detach().clone()))


class _Losses:
    """Builds the driver's loss wrapped in a recorder."""

    def __init__(self, trace: Trace, loss: Callable[[], Loss]) -> None:
        self.trace = trace
        self.loss = loss

    def __call__(self) -> _RecordedLoss:
        return _RecordedLoss(self.trace, self.loss())


class _RecordedLoss:
    """Forwards to a loss and keeps what each update gave and got."""

    def __init__(self, trace: Trace, loss: Loss) -> None:
        self.trace = trace
        self.loss = loss

    def compute_loss(self, logits: Tensor, labels: Tensor, epoch: int) -> Tensor:
        loss = self.loss.compute_loss(logits, labels, epoch)
        self.trace.losses.append(loss.detach().clone())
        self.trace.logits.append(logits.detach().clone())
        self.trace.labels.append(labels.clone())
        self.trace.epochs.append(epoch)
        return loss


class _Builds:
    """Builds the network in place of ``build_model`` and keeps it."""

    def __init__(self, script: Script, build: Builder, models: list[nn.Module]) -> None:
        self.script = script
        self.build = build
        self.models = models
        self.original: Callable[[str, int], nn.Module] = script.build_model

    def __call__(self, arch: str, num_classes: int) -> nn.Module:
        self.models.append(self.build(self.script, arch, num_classes))
        return self.models[-1]


class _PoisonedDatasets:
    """Builds the driver's ``PoisonedDataset`` and keeps it."""

    def __init__(self, trace: Trace, dataset: Callable[..., Poisoned]) -> None:
        self.trace = trace
        self.dataset = dataset

    def __call__(self, *args: object) -> Poisoned:
        self.trace.poisoned.append(self.dataset(*args))
        return self.trace.poisoned[-1]


class _Results:
    """Forwards to one of the driver's evaluations and keeps its result."""

    def __init__(self, trace: Trace, evaluate: Callable[..., float]) -> None:
        self.trace = trace
        self.evaluate = evaluate

    def __call__(self, *args: object) -> float:
        result = self.evaluate(*args)
        self.trace.results.append(result)
        return result


class _Stopped(Exception):  # noqa: N818 -- A control-flow signal, not an error.
    """Ends a run of ``main`` once its scheduler exists."""


@contextlib.contextmanager
def _recorded(
    script: Script,
    trace: Trace,
    *,
    build: Builder,
    split: tuple[Tensor, Tensor],
    serial: bool,
    stop: bool,
) -> Generator[None]:
    """Wrap what the driver's ``main`` calls in recorders, over ``split``."""
    splits = _Splits(split)
    datasets = script.torchvision.datasets
    models = trace.models
    # The driver binds ``DataLoader`` by name at import, so its own name is patched.
    loaders = (
        mock.patch.object(
            script,
            "DataLoader",
            _Loaders(trace, torch.utils.data.DataLoader),
        )
        if serial
        else contextlib.nullcontext()
    )
    with (
        mock.patch.object(datasets, "CIFAR10", splits),
        mock.patch.object(datasets, "CIFAR100", splits),
        mock.patch.object(datasets, "FashionMNIST", splits),
        loaders,
        mock.patch.object(
            torch.optim.lr_scheduler,
            "CosineAnnealingLR",
            _Schedulers(
                trace,
                models,
                torch.optim.lr_scheduler.CosineAnnealingLR,
                stop=stop,
            ),
        ),
        mock.patch.object(script, "build_model", _Builds(script, build, models)),
        mock.patch.object(script, "RobustLoss", _Losses(trace, script.RobustLoss)),
        mock.patch.object(
            script,
            "PoisonedDataset",
            _PoisonedDatasets(trace, script.PoisonedDataset),
        ),
        mock.patch.object(
            script,
            "evaluate_test_acc",
            _Results(trace, script.evaluate_test_acc),
        ),
        mock.patch.object(
            script,
            "evaluate_poison_fit",
            _Results(trace, script.evaluate_poison_fit),
        ),
        upstream.hidden_accelerators(),
        reference.script_workers(),
        upstream.forked_workers(),
    ):
        yield


def _restore(states: Generators) -> None:
    """Put Python's and NumPy's generators back where :func:`generator_states` found them."""
    random.setstate(states.python)
    name, keys, position, has_gauss, cached = states.numpy
    np.random.set_state(  # noqa: NPY002 -- Restores what the driver seeded.
        (
            name,
            np.frombuffer(keys, dtype=np.uint32).copy(),
            position,
            has_gauss,
            cached,
        ),
    )
