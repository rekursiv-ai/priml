"""Runs of the pristine ``security-machine-unlearning`` harness, under the capture's pins.

The harness and its editable method are imported from a clone of the
benchmark, not vendored, and driven through their own functions on the inputs
of :mod:`reference`. :func:`upstream_main` runs the harness's ``main`` end to
end with its four-worker loaders, its network shrunk to the shallowest ResNet
and CIFAR-10 replaced by a synthetic dataset; every other call is the
harness's own, recorded on the way. Nothing runs on a GPU.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Final, Protocol, cast, runtime_checkable
from unittest import mock

import contextlib
import io
import random
import sys

from torch import Tensor
from treekle import from_plain

import numpy as np
import torch

from priml.baselines.mlsbench.dl.vision import reference as vision
from priml.baselines.mlsbench.dl.vision.draws import recorded_draws
from priml.baselines.mlsbench.dl.vision.upstream import (
    Upstream,
    edited_script,
    forked_workers,
    load_script,
    narrowed_widths,
    tiny_model,
)
from priml.baselines.mlsbench.tl.machine_unlearning import reference
from priml.baselines.mlsbench.tl.machine_unlearning.reference import (
    MainTrace,
    Step,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from numpy.typing import NDArray
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Subset
else:
    from wrapt import lazy_import

    Image = lazy_import("PIL.Image")


ROOT: Final = Path(
    "harbor/tasks-docker/mls-bench__security-machine-unlearning/tests/meta/pristine"
    "/pytorch-vision/bench/unlearning",
)
"""Directory of the unedited harness and method, relative to the clone."""

DRIVER: Final = ROOT / "run_unlearning.py"
"""The fixed harness."""

METHOD: Final = ROOT / "custom_unlearning.py"
"""The editable method, pristine."""

EDITS: Final = Path("tasks/security-machine-unlearning/edits")
"""Directory of the task's baseline edits, relative to the clone."""

EDITED: Final = ("retain_finetune", "negative_gradient", "bad_teacher", "scrub")
"""The published baselines, each named as its edit file and its port variant."""

LAUNCH: Final = Path(
    "tasks/security-machine-unlearning/scripts/resnet20_cifar10_class0.sh",
)
"""The setting's launch script, relative to the clone."""


class Method(Protocol):
    """An instance of a variant's ``UnlearningMethod``."""

    def unlearn_step(  # noqa: PLR0917 -- The harness passes these six positionally.
        self,
        model: nn.Module,
        retain_batch: tuple[Tensor, Tensor],
        forget_batch: tuple[Tensor, Tensor],
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
    ) -> dict[str, float]:
        """Update ``model`` in place and report its loss."""
        raise NotImplementedError


@runtime_checkable
class ScrubMethod(Protocol):
    """The SCRUB edit's method: its max steps run for ``msteps`` epochs."""

    msteps: int


@runtime_checkable
class MethodModule(Protocol):
    """A variant of ``custom_unlearning.py``."""

    UnlearningMethod: Callable[[], Method]


class _Datasets(Protocol):
    """The torchvision datasets the harness reaches through its own import."""

    CIFAR10: type


class _Torchvision(Protocol):
    """The part of torchvision the harness reaches through its own import."""

    datasets: _Datasets


class Flags(Protocol):
    """The harness's parsed flags the capture reads."""

    arch: str
    dataset: str
    forget_class: int
    pretrain_epochs: int
    unlearn_epochs: int
    batch_size: int
    lr: float
    momentum: float
    weight_decay: float
    seed: int


@runtime_checkable
class Driver(Protocol):
    """The part of ``run_unlearning.py`` the capture uses."""

    BasicBlock: type[nn.Module]
    torchvision: _Torchvision
    ResNet: Callable[[type[nn.Module], list[int], int], nn.Module]
    UnlearningMethod: Callable[[], Method]

    def build_model(self, arch: str, num_classes: int) -> nn.Module:
        """Build the network an ``--arch`` names."""
        raise NotImplementedError

    def initialize_weights(self, model: nn.Module) -> None:
        """Initialize in place."""

    def split_by_class(
        self,
        dataset: object,
        forget_class: int,
    ) -> tuple[Subset[object], Subset[object]]:
        """Cut ``dataset`` into retained and forget subsets."""
        raise NotImplementedError

    def evaluate_accuracy(
        self,
        model: nn.Module,
        loader: Iterable[tuple[Tensor, Tensor]],
        device: torch.device,
    ) -> tuple[float, NDArray[np.floating]]:
        """Return the accuracy and every image's top probability."""
        raise NotImplementedError

    def auc_from_scores(
        self,
        member_scores: NDArray[np.floating],
        nonmember_scores: NDArray[np.floating],
    ) -> float:
        """Return the Mann-Whitney AUC."""
        raise NotImplementedError

    def pretrain(  # noqa: PLR0917 -- The harness's own signature; main passes all seven positionally.
        self,
        model: nn.Module,
        loader: DataLoader[tuple[Tensor, Tensor]],
        device: torch.device,
        epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        """Train ``model`` on every class."""

    def parse_args(self) -> Flags:
        """Parse ``sys.argv``."""
        raise NotImplementedError

    def main(self) -> None:
        """Run the whole harness."""


def driver(root: Path) -> Driver:
    """Import the harness with its pristine method in place.

    Args:
      root: Clone of the benchmark.

    Returns:
      driver: ``run_unlearning.py``.

    """
    method: object = load_script(root / METHOD)
    assert isinstance(method, ModuleType)
    # One key set and removed by hand: ``mock.patch.dict(sys.modules)`` restores
    # the whole table, evicting torchvision, whose next import then refuses to
    # register its kernels a second time.
    sys.modules["custom_unlearning"] = method
    try:
        script: object = load_script(root / DRIVER)
    finally:
        del sys.modules["custom_unlearning"]
    assert isinstance(script, Driver)
    return script


def variants(root: Path, scratch: Path) -> dict[str, MethodModule]:
    """Import the pristine method and each baseline's edit of it.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited files are written to.

    Returns:
      modules: ``pristine`` and each of :data:`EDITED`, as :data:`reference.VARIANTS`.

    """
    modules: dict[str, object] = {"pristine": load_script(root / METHOD)}
    for name in EDITED:
        modules[name] = edited_script(root, name, scratch, pristine=METHOD, edits=EDITS)
    narrowed: dict[str, MethodModule] = {}
    for name, module in modules.items():
        assert isinstance(module, MethodModule)
        narrowed[name] = module
    return narrowed


def upstream_recipe(script: Driver, root: Path) -> dict[str, Tensor]:
    """Parse the setting's launch script with the harness's own parser.

    Args:
      script: The harness.
      root: Clone of the benchmark.

    Returns:
      record: Keyed as :func:`reference.recipe_record`.

    Raises:
      ValueError: The launch script names another architecture or dataset.

    """
    command = (root / LAUNCH).read_text().split("run_unlearning.py", 1)[1]
    flags = command.replace("\\\n", " ").replace("${SEED:-42}", "42").split()
    with mock.patch.object(sys, "argv", ["run_unlearning.py", *flags]):
        args = script.parse_args()
    if (args.arch, args.dataset) != ("resnet20", "cifar10"):
        raise ValueError(f"The launch script runs {args.arch} on {args.dataset}.")
    return reference.recipe_values(
        forget_class=args.forget_class,
        pretrain_epochs=args.pretrain_epochs,
        unlearn_epochs=args.unlearn_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        classes=10,
    )


def upstream_split(script: Driver) -> dict[str, Tensor]:
    """Cut the fixed labels with the harness, held as a list and as a tensor.

    Args:
      script: The harness.

    Returns:
      record: Keyed as :func:`reference.split_record`.

    """
    labels = reference.split_labels()
    record: dict[str, Tensor] = {}
    listed = from_plain(labels.tolist(), list[int])
    for kind, targets in (("list", listed), ("tensor", labels)):
        retain, forget = script.split_by_class(
            _Targets(targets),
            reference.FORGET_CLASS,
        )
        record[f"split/{kind}/retain"] = torch.tensor(retain.indices)
        record[f"split/{kind}/forget"] = torch.tensor(forget.indices)
    return record


def upstream_method_lockstep(
    script: Driver,
    module: MethodModule,
    name: str,
) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
    """Run a variant's method for the updates of :data:`reference.SCHEDULE`.

    Args:
      script: The harness, whose ResNet and initialization build the network.
      module: The variant.
      name: The variant's key, for the record.

    Returns:
      record: Keyed as :func:`reference.method_lockstep`.
      states: The student after each update, then each teacher after the last.

    """
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        with narrowed_widths(script):
            model = script.ResNet(script.BasicBlock, [1, 1, 1], vision.CLASSES)
        script.initialize_weights(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        method = module.UnlearningMethod()
        steps: list[Step] = []
        for index, (epoch, step) in enumerate(reference.SCHEDULE):
            retain, forget = reference.fixed_batches(index)
            model.train()
            result = method.unlearn_step(model, retain, forget, optimizer, step, epoch)
            steps.append(
                Step(
                    loss=float(result["loss"]),
                    state=reference.cloned(model),
                    probe=reference.probe(model),
                ),
            )
    states = [s.state for s in steps] + list(reference.teachers(method).values())
    return reference.steps_record(f"methods/{name}", steps), states


def upstream_evaluation(script: Driver) -> dict[str, Tensor]:
    """Score the fixed network and the fixed scores with the harness's functions.

    Args:
      script: The harness.

    Returns:
      record: Keyed as :func:`reference.evaluation_record`.

    """
    cpu = torch.device("cpu")
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        with narrowed_widths(script):
            model = script.ResNet(script.BasicBlock, [1, 1, 1], vision.CLASSES)
        script.initialize_weights(model)
        accuracy, confidence = script.evaluate_accuracy(
            model,
            reference.evaluation_batches(),
            cpu,
        )
        empty, none = script.evaluate_accuracy(model, [], cpu)
    return reference.evaluation_values(
        accuracy=accuracy,
        confidence=torch.from_numpy(confidence.copy()),
        empty=(empty, torch.from_numpy(none.copy())),
        auc=script.auc_from_scores(*reference.fixed_scores()),
    )


def upstream_full_network(script: Driver, key: str) -> tuple[nn.Module, Tensor]:
    """Build a setting's full network with the harness from the task's seed.

    Args:
      script: The harness.
      key: A key of :data:`reference.SETTINGS`.

    Returns:
      model: The network, initialized.
      draws: Where the generator stood afterwards.

    """
    arch, classes, _ = reference.SETTINGS[key]
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        model = script.build_model(arch, classes)
        script.initialize_weights(model)
        return model, rng_fingerprint()


def upstream_full_size(
    script: Driver,
    key: str,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Build a setting's full network with the harness, and run its shallowest.

    Args:
      script: The harness.
      key: A key of :data:`reference.SETTINGS`.

    Returns:
      record: Keyed as :func:`reference.full_size_record`.
      state: The full network's whole state after initialization.

    """
    model, draws = upstream_full_network(script, key)
    arch, _, _ = reference.SETTINGS[key]
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        tiny = tiny_model(cast(Upstream, cast(object, script)), arch, vision.CLASSES)
        script.initialize_weights(tiny)
    record = {f"full/{key}/draws": draws} | vision.setting_forward(tiny, key)
    return record, reference.cloned(model)


def upstream_main(script: Driver, module: MethodModule) -> tuple[MainTrace, bool]:
    """Run the harness's ``main`` end to end at the capture's size.

    Args:
      script: The harness.
      module: The variant whose method ``main`` builds.

    Returns:
      trace: Every state, step and result the run produced.
      untouched: Whether Python's and NumPy's generators still stand where
        ``main``'s seeding put them, so torch's stream alone decided the run.

    """
    trace = MainTrace()
    spy = _MainSpy(script, module, trace)
    argv = ["run_unlearning.py", "--arch", "resnet20", "--dataset", "cifar10"]
    with (
        mock.patch.object(sys, "argv", [*argv, *reference.tiny_flags()]),
        mock.patch.object(script, "build_model", spy.build_model),
        mock.patch.object(script, "initialize_weights", spy.initialize_weights),
        mock.patch.object(script, "pretrain", spy.pretrain),
        mock.patch.object(script, "UnlearningMethod", spy.method),
        mock.patch.object(script, "evaluate_accuracy", spy.evaluate_accuracy),
        mock.patch.object(script, "auc_from_scores", spy.auc_from_scores),
        mock.patch.object(script.torchvision.datasets, "CIFAR10", _FakeCifar10),
        contextlib.redirect_stdout(io.StringIO()),
        vision.seeded(vision.SEED),
        vision.script_workers(),
        forked_workers(),
        recorded_draws() as sites,
        host_agnostic_numerics(),
    ):
        script.main()
        trace.draws = rng_fingerprint()
    seeded = random.Random(vision.SEED).getstate()  # noqa: S311 -- A reproducible reference state, not a secret.
    untouched = random.getstate() == seeded and _numpy_seeded(vision.SEED)
    trace.sites = dict(sites)
    trace.teachers = reference.teachers(spy.instance)
    return trace, untouched


class _Targets:
    """A dataset reduced to the one attribute ``split_by_class`` reads."""

    def __init__(self, targets: list[int] | Tensor) -> None:
        self.targets = targets


class _FakeCifar10:
    """The synthetic dataset, in the shape the harness expects of torchvision's."""

    def __init__(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: Callable[[Image.Image], Tensor],
    ) -> None:
        del root, download
        self.media, label = reference.synthetic_dataset(train=train)
        self.targets = from_plain(label.tolist(), list[int])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.media)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        pixels = self.media[index].permute(1, 2, 0).contiguous().numpy()
        return self.transform(Image.fromarray(pixels)), self.targets[index]


class _Recorded:
    """A variant's method that records each update as the harness made it."""

    def __init__(self, method: Method, steps: list[Step]) -> None:
        self.method = method
        self.steps = steps

    def unlearn_step(  # noqa: PLR0917 -- The harness passes these six positionally.
        self,
        model: nn.Module,
        retain_batch: tuple[Tensor, Tensor],
        forget_batch: tuple[Tensor, Tensor],
        optimizer: torch.optim.Optimizer,
        step: int,
        epoch: int,
    ) -> dict[str, float]:
        """Run the variant's update, then record its inputs and the network.

        Args:
          model: The student, in training mode.
          retain_batch: Retained images and labels.
          forget_batch: Forget images and labels.
          optimizer: Steps ``model``'s parameters.
          step: Index of the update within its epoch.
          epoch: Zero-based unlearning epoch.

        Returns:
          result: What the variant's method returned.

        """
        result = self.method.unlearn_step(
            model,
            retain_batch,
            forget_batch,
            optimizer,
            step,
            epoch,
        )
        self.steps.append(
            Step(
                loss=float(result["loss"]),
                state=reference.cloned(model),
                batches=(*retain_batch, *forget_batch),
                position=(epoch, step),
                optimizer=reference.adam_settings(optimizer),
            ),
        )
        return result


class _MainSpy:
    """Records the harness's calls while its ``main`` runs; changes only the build."""

    def __init__(self, script: Driver, module: MethodModule, trace: MainTrace) -> None:
        self.script = script
        self.module = module
        self.trace = trace
        self.initialize = script.initialize_weights
        self.train = script.pretrain
        self.evaluate = script.evaluate_accuracy
        self.auc = script.auc_from_scores
        self.instance: Method | None = None
        self.evaluations: list[tuple[float, Tensor]] = []

    def build_model(self, arch: str, num_classes: int) -> nn.Module:
        """Build the shallowest ResNet at the tiny widths."""
        del arch
        with narrowed_widths(self.script):
            return self.script.ResNet(self.script.BasicBlock, [1, 1, 1], num_classes)

    def initialize_weights(self, model: nn.Module) -> None:
        """Initialize as the harness does, then record the state."""
        self.initialize(model)
        self.trace.initial = reference.cloned(model)

    def pretrain(  # noqa: PLR0917 -- Stands in for the harness's pretrain, which main passes all seven positionally.
        self,
        model: nn.Module,
        loader: DataLoader[tuple[Tensor, Tensor]],
        device: torch.device,
        epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        """Pretrain as the harness does, then record its settings and the state."""
        self.train(model, loader, device, epochs, lr, momentum, weight_decay)
        self.trace.pretrain_settings = (epochs, lr, momentum, weight_decay)
        self.trace.pretrained = reference.cloned(model)

    def method(self) -> _Recorded:
        """Build the variant's method, wrapped to record every update."""
        self.instance = self.module.UnlearningMethod()
        # The end-to-end run's two epochs: SCRUB's max step in the first only.
        if isinstance(self.instance, ScrubMethod):
            self.instance.msteps = reference.TINY_SCRUB_MAX_EPOCHS
        return _Recorded(self.instance, self.trace.steps)

    def evaluate_accuracy(
        self,
        model: nn.Module,
        loader: Iterable[tuple[Tensor, Tensor]],
        device: torch.device,
    ) -> tuple[float, NDArray[np.floating]]:
        """Evaluate as the harness does; record the accuracies and the attack's scores.

        ``main`` scores the retained test images, the forget test images, then
        the forget training images.

        Args:
          model: The unlearned network.
          loader: The images to score.
          device: Where the network runs.

        Returns:
          accuracy: The harness's accuracy.
          confidence: The harness's scores.

        """
        accuracy, confidence = self.evaluate(model, loader, device)
        self.evaluations.append((accuracy, torch.from_numpy(confidence.copy())))
        if len(self.evaluations) == 3:
            (retain, _), (forget, test), (_, train) = self.evaluations
            self.trace.accuracy = (retain, forget)
            self.trace.confidence = (train, test)
        return accuracy, confidence

    def auc_from_scores(
        self,
        member_scores: NDArray[np.floating],
        nonmember_scores: NDArray[np.floating],
    ) -> float:
        """Score as the harness does and record the AUC."""
        self.trace.auc = self.auc(member_scores, nonmember_scores)
        return self.trace.auc


def _numpy_seeded(seed: int) -> bool:
    """Return whether NumPy's legacy generator stands exactly where ``seed`` puts it."""
    _, live_key, *live_rest = np.random.get_state()  # noqa: NPY002 -- The harness seeds NumPy's legacy generator; its state is what is checked.
    _, fresh_key, *fresh_rest = np.random.RandomState(seed).get_state()
    return bool(np.array_equal(live_key, fresh_key)) and live_rest == fresh_rest
