"""Inputs and port-side runs the ``security-machine-unlearning`` capture and test share.

``scripts/capture_reference.py`` runs the pristine harness, each variant of its
method, and these runs on the same inputs under the same pins, compares them
bit for bit, and stores the harness's values under ``testdata/``;
``reference_test.py`` replays these runs against them.

Five comparisons, each reduced to a record of tensors both sides build with
the same function here:

- ``recipe/``: the ``resnet20-cifar10-class0`` launch flags against ``exp000``.
- ``split/``: the forget split of fixed labels.
- ``methods/<variant>/``: three updates of every method on fixed batches of
  the shallowest ResNet, the network after each, teachers included.
- ``evaluation/``: accuracy and confidence of a fixed network on fixed
  batches, and the membership AUC of fixed scores with ties.
- ``main/<variant>/``: the whole harness end to end on a synthetic dataset:
  pretraining, every unlearning step and its batches, the evaluation, and
  where the generator stands afterwards.

Every input is built from integers, so it is exact in float32 and the same on
every host. Every run that computes does so under
:func:`~priml.testing.bfb.host_agnostic_numerics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from torch import Tensor, nn
from treekle import from_plain

import numpy as np
import torch

from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.draws import recorded_draws
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.baselines.mlsbench.tl.machine_unlearning.data import split_by_class
from priml.baselines.mlsbench.tl.machine_unlearning.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
)
from priml.baselines.mlsbench.tl.machine_unlearning.methods import Scrub
from priml.baselines.mlsbench.tl.machine_unlearning.metrics import (
    auc_from_scores,
    evaluate_accuracy,
)
from priml.baselines.mlsbench.tl.machine_unlearning.unlearning import (
    MachineUnlearning,
    seed_like_the_reference,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import NDArray

    from priml.baselines.mlsbench.tl.machine_unlearning.methods import (
        Batch,
        UnlearningMethod,
    )


VARIANTS: Final[dict[str, Callable[[], MachineUnlearning.Config]]] = {
    "pristine": exp000,
    "retain_finetune": exp001,
    "negative_gradient": exp002,
    "bad_teacher": exp003,
    "scrub": exp004,
}
"""The experiment carrying each variant of the method, keyed as the capture's variants."""

SETTINGS: Final[dict[str, tuple[str, int, Callable[[], MachineUnlearning.Config]]]] = {
    "resnet20-cifar10-class0": ("resnet20", 10, exp000),
    "vgg16bn-cifar100-class0": ("vgg16bn", 100, exp005),
    "mobilenetv2-fmnist-class0": ("mobilenetv2", 10, exp006),
}
"""Each setting's ``--arch``, class count, and the experiment that runs it."""

FORGET_CLASS: Final = 0
"""The class every comparison forgets, the setting's own."""

SCHEDULE: Final = ((0, 0), (1, 0), (2, 1))
"""``(epoch, step)`` of each method update: SCRUB's max steps run in the first two."""

RECIPE: Final = (
    "forget_class",
    "pretrain_epochs",
    "unlearn_epochs",
    "batch_size",
    "lr",
    "momentum",
    "weight_decay",
    "seed",
    "classes",
)
"""The recipe's values, in the order ``recipe/values`` holds them."""

TRAIN_LABELS: Final = (0, 3, 1, 0)
"""Labels of the synthetic training images: two of the forget class, two others."""

TEST_LABELS: Final = (1, 0, 4, 0)
"""Labels of the synthetic test images: two of the forget class, two others."""

TINY_BATCH: Final = 4
"""Batch size of the end-to-end run: every split is one batch, none of one image."""

TINY_PRETRAIN_EPOCHS: Final = 1
"""Pretraining epochs of the end-to-end run."""

TINY_UNLEARN_EPOCHS: Final = 2
"""Unlearning epochs of the end-to-end run: one with SCRUB's max step, one without."""

TINY_SCRUB_MAX_EPOCHS: Final = 1
"""SCRUB's max-step epochs in the end-to-end run, so its two epochs reach both phases."""

ADAM: Final = ("lr", "beta1", "beta2", "eps", "weight_decay", "amsgrad")
"""Adam settings ``main/<variant>/adam`` holds, in order."""


@dataclass(slots=True, kw_only=True)
class Step:
    """One unlearning update, as the network and the harness saw it."""

    loss: float
    """What the method reported."""

    state: dict[str, Tensor]
    """The student's whole state afterwards."""

    probe: Tensor = field(default_factory=lambda: torch.empty(0))
    """Evaluation-mode logits of a fixed batch afterwards, when probed."""

    batches: tuple[Tensor, ...] = ()
    """The retained images and labels, then the forget images and labels."""

    position: tuple[int, int] = (0, 0)
    """The ``(epoch, step)`` the harness passed."""

    optimizer: tuple[float, ...] = ()
    """The optimizer's settings, as :data:`ADAM`."""


@dataclass(slots=True, kw_only=True)
class MainTrace:
    """What one end-to-end run did, as its recorded calls saw it."""

    initial: dict[str, Tensor] = field(default_factory=dict[str, Tensor])
    """The network's state after initialization."""

    pretrained: dict[str, Tensor] = field(default_factory=dict[str, Tensor])
    """The network's state after pretraining."""

    pretrain_settings: tuple[float, ...] = ()
    """Pretraining's epochs, rate, momentum and weight decay."""

    steps: list[Step] = field(default_factory=list[Step])
    """Every unlearning step, in order."""

    teachers: dict[str, dict[str, Tensor]] = field(
        default_factory=dict[str, dict[str, Tensor]],
    )
    """Each network the method holds besides the student, after the last step."""

    accuracy: tuple[float, float] = (0.0, 0.0)
    """Accuracy on the retained and on the forget test images."""

    confidence: tuple[Tensor, Tensor] = (torch.empty(0), torch.empty(0))
    """Top probabilities of the forget training and the forget test images."""

    auc: float = 0.0
    """The membership AUC."""

    draws: Tensor = field(default_factory=lambda: torch.empty(0))
    """Where torch's generator stood when the run returned."""

    sites: dict[str, int] = field(default_factory=dict[str, int])
    """Torch draws made in the main process, by API and caller."""


class Recorded:
    """Wraps a method so each update is recorded as the harness made it."""

    def __init__(self, method: UnlearningMethod, steps: list[Step]) -> None:
        self.method = method
        self.steps = steps

    def unlearn_step(
        self,
        model: nn.Module,
        retain_batch: Batch,
        forget_batch: Batch,
        optimizer: torch.optim.Optimizer,
        *,
        step: int,
        epoch: int,
    ) -> Tensor:
        """Run the method's update and record its inputs and the network after it.

        Args:
          model: The student, in training mode.
          retain_batch: Retained images and labels.
          forget_batch: Forget images and labels.
          optimizer: Steps ``model``'s parameters.
          step: Index of the update within its epoch.
          epoch: Zero-based unlearning epoch.

        Returns:
          loss: What the wrapped method reported.

        """
        loss = self.method.unlearn_step(
            model,
            retain_batch,
            forget_batch,
            optimizer,
            step=step,
            epoch=epoch,
        )
        self.steps.append(
            Step(
                loss=float(loss),
                state=cloned(model),
                batches=(*retain_batch, *forget_batch),
                position=(epoch, step),
                optimizer=adam_settings(optimizer),
            ),
        )
        return loss


def cloned(model: nn.Module) -> dict[str, Tensor]:
    """Return a copy of ``model``'s whole state, in registration order."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def teachers(method: object) -> dict[str, dict[str, Tensor]]:
    """Return the state of every network a method holds, by attribute name.

    Args:
      method: A method instance, the port's or a script's.

    Returns:
      states: Attribute name to the network's whole state.

    """
    attributes: dict[str, object] = dict(vars(method))
    return {
        name: cloned(value)
        for name, value in sorted(attributes.items())
        if isinstance(value, nn.Module)
    }


def adam_settings(optimizer: torch.optim.Optimizer) -> tuple[float, ...]:
    """Return an Adam optimizer's settings in the order of :data:`ADAM`.

    Args:
      optimizer: An Adam optimizer, the port's or a script's.

    Returns:
      settings: Its first group's rate, betas, epsilon, weight decay and
        ``amsgrad`` flag, as floats.

    """
    group = from_plain(optimizer.param_groups[0], dict[str, object])
    beta1, beta2 = from_plain(group["betas"], tuple[float, float])
    return (
        from_plain(group["lr"], float),
        beta1,
        beta2,
        from_plain(group["eps"], float),
        from_plain(group["weight_decay"], float),
        float(from_plain(group["amsgrad"], bool)),
    )


def recipe_values(**values: float) -> dict[str, Tensor]:
    """Return the recipe's values as ``recipe/values``, in the order of :data:`RECIPE`.

    Args:
      **values: Each name of :data:`RECIPE` and its value.

    Returns:
      record: ``recipe/values``, as float64.

    """
    return {
        "recipe/values": torch.tensor(
            [float(values[name]) for name in RECIPE],
            dtype=torch.float64,
        ),
    }


def recipe_record() -> dict[str, Tensor]:
    """Return ``exp000``'s values of the setting's launch flags.

    Returns:
      record: ``recipe/values``.

    """
    cfg = exp000()
    model = cfg.pretrain.model
    assert isinstance(model, ResNet.Config)
    sgd = cfg.pretrain.optimizer.make()(nn.Linear(2, 3).parameters())
    group = from_plain(sgd.param_groups[0], dict[str, object])
    return recipe_values(
        forget_class=cfg.dataset.forget_class,
        pretrain_epochs=cfg.pretrain.train_budget_epochs,
        unlearn_epochs=cfg.unlearn_epochs,
        batch_size=cfg.dataset.batch_size,
        lr=from_plain(group["lr"], float),
        momentum=from_plain(group["momentum"], float),
        weight_decay=from_plain(group["weight_decay"], float),
        seed=cfg.seed,
        classes=model.channels_out,
    )


def split_labels() -> Tensor:
    """Return the synthetic training labels, the forget split's input."""
    return torch.tensor(TRAIN_LABELS)


def split_record() -> dict[str, Tensor]:
    """Cut :func:`split_labels` with the port, as the harness cuts a list and a tensor.

    Returns:
      record: ``split/<kind>/retain`` and ``split/<kind>/forget`` indices.

    """
    retain, forget = split_by_class(split_labels(), forget_class=FORGET_CLASS)
    return {
        f"split/{kind}/{part}": indices
        for kind in ("list", "tensor")
        for part, indices in (("retain", retain), ("forget", forget))
    }


def fixed_batches(index: int) -> tuple[Batch, Batch]:
    """Return update ``index``'s retained batch of two and forget batch of three.

    Args:
      index: Which update of :data:`SCHEDULE`.

    Returns:
      retain: Two images of :data:`reference.PROBE_SHAPE` and their classes.
      forget: Three images and the forget class.

    """
    retain = (reference.fixed_media(3 * index), reference.fixed_label(index))
    forget_media = torch.cat(
        [reference.fixed_media(3 * index + 1), reference.fixed_media(3 * index + 2)],
    )[:3]
    forget = (forget_media, torch.full((3,), FORGET_CLASS))
    return retain, forget


def probe(model: nn.Module) -> Tensor:
    """Return evaluation-mode logits of a fixed batch, then restore training mode.

    Args:
      model: The student, in training mode.

    Returns:
      logits: ``(2, CLASSES)``, outside inference mode.

    """
    model.eval()
    with torch.no_grad():
        logits = reference.model_logits(model, reference.fixed_media(len(SCHEDULE) * 3))
    model.train()
    return logits


def steps_record(prefix: str, steps: list[Step]) -> dict[str, Tensor]:
    """Return each update's loss and probe under ``prefix``.

    Args:
      prefix: Key prefix.
      steps: The updates.

    Returns:
      record: ``<prefix>/loss`` (float64) and ``<prefix>/probe``.

    """
    return {
        f"{prefix}/loss": torch.tensor([s.loss for s in steps], dtype=torch.float64),
        f"{prefix}/probe": torch.stack([s.probe for s in steps]),
    }


def tiny_model(factory: Callable[[], MachineUnlearning.Config]) -> nn.Module:
    """Build ``factory``'s network shrunk as the vision ``tiny_model`` shrinks it, at three classes.

    Args:
      factory: An experiment; its initializers are kept.

    Returns:
      model: Initialized on the CPU, as the global generator stands.

    """
    config = factory().pretrain
    reference.tiny_model(config.model)
    assert isinstance(config.model, (ResNet.Config, VGG16BN.Config, MobileNetV2.Config))
    config.model.channels_out = reference.CLASSES
    config.parallelism = NoParallel.Config(device="cpu")
    step = config.make()
    assert isinstance(step, VisionTrainStep)
    return step.model


def method_lockstep(name: str) -> tuple[dict[str, Tensor], list[dict[str, Tensor]]]:
    """Run variant ``name``'s method for the updates of :data:`SCHEDULE`.

    Args:
      name: A key of :data:`VARIANTS`.

    Returns:
      record: ``methods/<name>/loss`` and ``methods/<name>/probe``.
      states: The student after each update, then each teacher after the last.

    """
    factory = VARIANTS[name]
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = tiny_model(factory)
        optimizer = factory().unlearn_optimizer.make()(model.parameters())
        method = factory().method.make()
        steps: list[Step] = []
        for index, (epoch, step) in enumerate(SCHEDULE):
            retain, forget = fixed_batches(index)
            model.train()
            loss = method.unlearn_step(
                model,
                retain,
                forget,
                optimizer,
                step=step,
                epoch=epoch,
            )
            steps.append(
                Step(loss=float(loss), state=cloned(model), probe=probe(model)),
            )
    states = [s.state for s in steps] + list(teachers(method).values())
    return steps_record(f"methods/{name}", steps), states


def method_record() -> dict[str, Tensor]:
    """Return :func:`method_lockstep`'s record for every variant."""
    record: dict[str, Tensor] = {}
    for name in VARIANTS:
        record |= method_lockstep(name)[0]
    return record


def evaluation_batches() -> list[tuple[Tensor, Tensor]]:
    """Return two fixed batches whose labels some predictions match."""
    return [
        (reference.fixed_media(index), torch.tensor([index % 3, (2 * index + 1) % 3]))
        for index in range(2)
    ]


def fixed_scores() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return member and non-member scores, eighths with ties within each set.

    No score is in both sets: NumPy's unstable sort orders a tie across them
    by the host's SIMD kernel, so such a tie would make the AUC the host's.

    Returns:
      member: Five float32 scores.
      nonmember: Four float32 scores.

    """
    eighth = np.float32(1 / 8)
    return (
        np.array([5, 3, 7, 3, 6], dtype=np.float32) * eighth,
        np.array([2, 4, 4, 1], dtype=np.float32) * eighth,
    )


def evaluation_values(
    *,
    accuracy: float,
    confidence: Tensor,
    empty: tuple[float, Tensor],
    auc: float,
) -> dict[str, Tensor]:
    """Return an evaluation's results as the ``evaluation/`` record.

    Args:
      accuracy: Accuracy on :func:`evaluation_batches`.
      confidence: Top probability of each of their images.
      empty: Accuracy and confidences of an evaluation of no images.
      auc: AUC of :func:`fixed_scores`.

    Returns:
      record: Floats as float64, confidences as they were returned.

    """
    return {
        "evaluation/accuracy": torch.tensor(accuracy, dtype=torch.float64),
        "evaluation/confidence": confidence,
        "evaluation/empty_accuracy": torch.tensor(empty[0], dtype=torch.float64),
        "evaluation/empty_confidence": empty[1],
        "evaluation/auc": torch.tensor(auc, dtype=torch.float64),
    }


def evaluation_record() -> dict[str, Tensor]:
    """Score a fixed network on fixed batches and the fixed scores with the port.

    Returns:
      record: Keyed as :func:`evaluation_values`.

    """
    cpu = torch.device("cpu")
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = tiny_model(exp000)
        accuracy, confidence = evaluate_accuracy(
            model,
            evaluation_batches(),
            device=cpu,
        )
        empty, none = evaluate_accuracy(model, [], device=cpu)
    return evaluation_values(
        accuracy=accuracy,
        confidence=torch.from_numpy(confidence.copy()),
        empty=(empty, torch.from_numpy(none.copy())),
        auc=auc_from_scores(*fixed_scores()),
    )


def full_size_values(
    model: nn.Module,
    draws: Tensor,
    *,
    key: str,
) -> dict[str, Tensor]:
    """Return a full network's draws and its training- and evaluation-mode logits.

    Each forward starts from the task's seed, so a training-mode dropout draws
    the same masks on both sides.

    Args:
      model: The full network, freshly initialized.
      draws: Where the generator stood after its initialization.
      key: The setting, a key of :data:`SETTINGS`.

    Returns:
      record: ``full/<key>/draws``, ``full/<key>/train`` and ``full/<key>/eval``.

    """
    record = {f"full/{key}/draws": draws}
    media = reference.full_size_media()
    for mode in ("train", "eval"):
        model.train(mode == "train")
        with (
            reference.seeded(reference.SEED),
            torch.no_grad(),
            host_agnostic_numerics(),
        ):
            record[f"full/{key}/{mode}"] = reference.model_logits(model, media)
    return record


def full_network(key: str) -> tuple[nn.Module, Tensor]:
    """Build a setting's full network from the task's seed.

    Args:
      key: A key of :data:`SETTINGS`.

    Returns:
      model: The network, initialized.
      draws: Where the generator stood afterwards.

    """
    config = SETTINGS[key][2]().pretrain
    config.parallelism = NoParallel.Config(device="cpu")
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = config.make()
        assert isinstance(step, VisionTrainStep)
        return step.model, rng_fingerprint()


def full_size_record(key: str) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Build a setting's full network from the task's seed, and run its shallowest.

    The full network's forward is compared by the capture alone: at full
    width it is too slow for the default test.

    Args:
      key: A key of :data:`SETTINGS`.

    Returns:
      record: ``full/<key>/draws``, and the shallowest network's
        ``settings/<key>/train`` and ``settings/<key>/eval`` logits.
      state: The full network's whole state after initialization.

    """
    model, draws = full_network(key)
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        tiny = tiny_model(SETTINGS[key][2])
    record = {f"full/{key}/draws": draws} | reference.setting_forward(tiny, key)
    return record, cloned(model)


def synthetic_dataset(*, train: bool) -> tuple[Tensor, Tensor]:
    """Return the end-to-end run's 32x32 uint8 images and labels.

    Every pixel differs from its neighbours and every image from the others, so
    a wrong crop, mirror or order changes the batches.

    Args:
      train: The training split, else the test split.

    Returns:
      media: ``(N, 3, 32, 32)`` uint8 images.
      label: :data:`TRAIN_LABELS` or :data:`TEST_LABELS`.

    """
    labels = TRAIN_LABELS if train else TEST_LABELS
    offset = 0 if train else 53
    position = torch.arange(3 * 32 * 32)
    media = torch.stack(
        [
            ((position * 37 + 101 * (index + offset)) % 251).reshape(3, 32, 32)
            for index in range(len(labels))
        ],
    ).to(torch.uint8)
    return media, torch.tensor(labels)


def tiny_flags() -> list[str]:
    """Return the harness flags of the end-to-end run; the rest stay its defaults.

    Returns:
      flags: ``--forget-class``, the epochs, the batch size and the seed.

    """
    return [
        "--forget-class",
        str(FORGET_CLASS),
        "--pretrain-epochs",
        str(TINY_PRETRAIN_EPOCHS),
        "--unlearn-epochs",
        str(TINY_UNLEARN_EPOCHS),
        "--batch-size",
        str(TINY_BATCH),
        "--seed",
        str(reference.SEED),
    ]


def tiny_config(
    factory: Callable[[], MachineUnlearning.Config],
    directory: Path,
) -> MachineUnlearning.Config:
    """Shrink ``factory`` in depth, epochs and batch to the end-to-end run's size.

    Args:
      factory: An experiment; its recipe and method are kept.
      directory: Holds the synthetic ``train.pt`` and ``test.pt``.

    Returns:
      config: A CPU run of the shallowest ResNet over the synthetic dataset.

    """
    cfg = factory()
    cfg.device = "cpu"
    assert isinstance(cfg.pretrain.model, ResNet.Config)
    reference.tiny_model(cfg.pretrain.model)
    cfg.pretrain.train_budget_epochs = TINY_PRETRAIN_EPOCHS
    cfg.unlearn_epochs = TINY_UNLEARN_EPOCHS
    if isinstance(cfg.method, Scrub.Config):
        cfg.method.max_epochs = TINY_SCRUB_MAX_EPOCHS
    cfg.base_dir = None
    cfg.dataset.working_dir = directory
    cfg.dataset.batch_size = cfg.dataset.eval_batch_size = TINY_BATCH
    return cfg


def write_synthetic(directory: Path) -> None:
    """Write :func:`synthetic_dataset` in the prepared format under ``directory``."""
    for split, train in (("train", True), ("test", False)):
        media, label = synthetic_dataset(train=train)
        torch.save({"media": media, "label": label}, directory / f"{split}.pt")


def port_main(name: str, directory: Path) -> MainTrace:
    """Run variant ``name`` end to end as :meth:`MachineUnlearning.unlearn` does.

    Args:
      name: A key of :data:`VARIANTS`.
      directory: Scratch directory the synthetic dataset is written to.

    Returns:
      trace: Every state, step and result the run produced.

    """
    write_synthetic(directory)
    job = tiny_config(VARIANTS[name], directory).make()
    trace = MainTrace()
    with (
        reference.seeded(reference.SEED),
        recorded_draws() as sites,
        host_agnostic_numerics(),
    ):
        seed_like_the_reference(job.config.seed)
        data = job.config.dataset.make()
        step = job.build()
        trace.initial = cloned(step.model)
        job.pretrain(step, data.full_loader())
        trace.pretrained = cloned(step.model)
        group = from_plain(step.optimizer.param_groups[0], dict[str, object])
        trace.pretrain_settings = (
            step.config.train_budget_epochs,
            *(
                from_plain(group[name], float)
                for name in ("initial_lr", "momentum", "weight_decay")
            ),
        )
        method = job.config.method.make()
        job.forget(step.model, data, method=Recorded(method, trace.steps))
        report = job.evaluate(step.model, data)
        trace.draws = rng_fingerprint()
    trace.sites = dict(sites)
    trace.teachers = teachers(method)
    trace.accuracy = (report.retain_acc, report.forget_acc)
    trace.confidence = (
        torch.from_numpy(report.forget_train_confidence.copy()),
        torch.from_numpy(report.forget_test_confidence.copy()),
    )
    trace.auc = report.forget_mia_auc
    return trace


def main_record(name: str, trace: MainTrace) -> dict[str, Tensor]:
    """Reduce an end-to-end run to the values a golden holds.

    Args:
      name: The variant, for the keys.
      trace: A run of either side.

    Returns:
      record: Per step the loss, labels and position; the optimizers'
        settings; the evaluation; and where the generator stands.

    """
    prefix = f"main/{name}"
    steps = trace.steps
    return {
        f"{prefix}/loss": torch.tensor([s.loss for s in steps], dtype=torch.float64),
        f"{prefix}/position": torch.tensor([s.position for s in steps]),
        f"{prefix}/retain_label": torch.cat([s.batches[1] for s in steps]),
        f"{prefix}/forget_label": torch.cat([s.batches[3] for s in steps]),
        f"{prefix}/adam": torch.tensor(
            [s.optimizer for s in steps],
            dtype=torch.float64,
        ),
        f"{prefix}/pretrain": torch.tensor(
            trace.pretrain_settings,
            dtype=torch.float64,
        ),
        f"{prefix}/metrics": torch.tensor(
            [*trace.accuracy, trace.auc],
            dtype=torch.float64,
        ),
        f"{prefix}/forget_train_confidence": trace.confidence[0],
        f"{prefix}/forget_test_confidence": trace.confidence[1],
        f"{prefix}/draws": trace.draws,
    }


def port_record(
    directory: Path,
) -> tuple[dict[str, Tensor], dict[str, dict[str, Tensor]]]:
    """Run every port-side comparison and return all their values.

    Args:
      directory: Scratch directory the end-to-end runs write their dataset to.

    Returns:
      record: The union of the records above, keyed as the stored golden is.
      states: Each setting's full network after initialization.

    """
    record = recipe_record() | split_record() | evaluation_record()
    states: dict[str, dict[str, Tensor]] = {}
    for key in SETTINGS:
        full, states[key] = full_size_record(key)
        record |= full
    record |= method_record()
    for name in VARIANTS:
        record |= main_record(name, port_main(name, directory))
    return record, states
