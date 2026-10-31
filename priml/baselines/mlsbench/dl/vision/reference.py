"""Inputs and port-side runs the reference captures and their tests share.

``scripts/capture_reference.py`` runs the pristine reference code and this
module's port-side runs on the same inputs under the same pins, compares them
bit for bit, and stores the reference's values under ``testdata/``.
``reference_test.py`` replays the port-side runs against those stored values.
Each task package's capture builds its own runs from the inputs, pins and steps
here.

Every input is rebuilt from integers and powers of two, so it is exact in
float32 and the same on every host. Every run that computes does so under
:func:`~priml.testing.bfb.host_agnostic_numerics`, except the data
pipeline, whose operations are all correctly rounded and so host independent
without it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, cast, runtime_checkable

import warnings

from torch import Tensor, nn
from treekle import from_plain

import torch

from priml.baselines.mlsbench.dl.vision.data import (
    Cifar100Data,
    normalize,
    pad_crop_flip,
    resized,
)
from priml.baselines.mlsbench.dl.vision.draws import recorded_draws
from priml.baselines.mlsbench.dl.vision.experiments import (
    VisionTrainLoop,
    exp000,
)
from priml.baselines.mlsbench.dl.vision.metrics import BestAccuracy
from priml.baselines.mlsbench.dl.vision.mobilenet import MobileNetV2
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.baselines.mlsbench.dl.vision.settings import (
    mobilenetv2_fmnist,
    resnet20_cifar10,
    resnet110_cifar100,
    vgg16bn_cifar100,
)
from priml.baselines.mlsbench.dl.vision.train_step import VisionTrainStep
from priml.baselines.mlsbench.dl.vision.vgg import VGG16BN
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint
from priml.train.parallelism import NoParallel
from priml.train.train_loop import TrainLoop


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from configgle import Makeable

type VisionLoop = TrainLoop.Config[VisionTrainStep.Config, Cifar100Data.Config]
"""Any loop config holding the vision step and a CIFAR-style dataset.

The runs here read only ``step``, ``dataset`` and ``experiment_name``, so a
task's own loop config -- with a narrowed step or dataset -- is accepted."""

SEED: Final = 42
"""Seed of every pinned run, the task's own."""

CLASSES: Final = 3
"""Logits of the tiny network; its widths are :data:`TINY_WIDTHS`."""

PROBE_SHAPE: Final = (2, 3, 8, 10)
"""Batch, channels, height and width of the tiny network's inputs."""

EPOCHS: Final = 5
"""Schedule horizon of the lockstep runs."""

STEPS: Final = 2
"""Updates of a lockstep run, one per epoch."""

LOADER_IMAGES: Final = 7
"""Images in the synthetic split, enough batches to reach every stream."""

LOADER_BATCH: Final = 2
"""Batch size of the loader runs: four batches over the synthetic split."""

LOADER_PASSES: Final = 2
"""Training passes of a loader run, so the second follows an evaluation pass."""

TINY_WIDTHS: Final = {16: 4, 32: 5, 64: 9, 320: 6, 1280: 7, 512: 8}
"""Each width the reference hardcodes, and the tiny one for it.

ResNet's 16/32/64 stages, MobileNetV2's 32-channel stem and ``320 -> 1280``
last convolution, and VGG's 512-wide classifier are literals in the reference;
the tiny build narrows them on both sides. Every tiny width differs from every
other and from the classes."""

TINY_ARCHS: Final = {
    "vgg16bn": [4, "M", TINY_WIDTHS[512], "M"],
    "mobilenetv2": [[1, 4, 1, 2], [1, TINY_WIDTHS[320], 1, 2]],
}
"""The reference's architecture tables for the smallest networks :func:`tiny_step` builds.

``VGG.VGG16_CFG`` and ``MobileNetV2.CFG`` rows, cut to the least. Each table
ends in the tiny width of the literal its next layer reads."""

HYPERPARAMETERS: Final = (
    "lr",
    "initial_lr",
    "momentum",
    "dampening",
    "weight_decay",
    "nesterov",
    "maximize",
    "differentiable",
)
"""Parameter-group entries compared between the two optimizers."""

SCRIPTED_CORRECT: Final = tuple((2 * epoch**2 + epoch) % 7 for epoch in range(200))
"""Correct predictions out of ``SCRIPTED_TOTAL`` that a scripted evaluation reports."""

SCRIPTED_TOTAL: Final = 7
"""Test images of a scripted evaluation."""


@dataclass(slots=True, kw_only=True)
class Captured:
    """The reference's values, the random draws of both sides, and every mismatch."""

    record: dict[str, Tensor]
    sites: dict[str, dict[str, int]]
    report: list[str]


@runtime_checkable
class WidthKeyed(Protocol):
    """A block config whose behavior keys on one of the reference's widths."""

    reset_width: int


@contextmanager
def seeded(seed: int) -> Generator[None]:
    """Seed the global CPU generator and restore its state afterwards.

    Args:
      seed: Value the generator starts from, as ``torch.manual_seed`` sets it.

    Yields:
      nothing: Used as a context manager.

    """
    state = torch.get_rng_state()
    generator = torch.Generator().manual_seed(seed)
    torch.set_rng_state(generator.get_state())
    try:
        yield
    finally:
        torch.set_rng_state(state)


@contextmanager
def script_workers() -> Generator[None]:
    """Let a script's four loader workers run on a host with fewer CPUs.

    Yields:
      nothing: Used as a context manager.

    """
    with warnings.catch_warnings():
        # `torch` warns when a loader asks for more workers than the host has
        # CPUs. The count is the script's, and a worker's draws do not
        # depend on the CPUs, so the warning says nothing about the comparison.
        warnings.filterwarnings("ignore", message=".*worker processes in total.*")
        yield


def fixed_logits() -> tuple[Tensor, Tensor]:
    """Return a ``(3, 4)`` logit batch and its labels.

    Returns:
      logits: Multiples of one half.
      label: One target class per row.

    """
    logits = ((torch.arange(12) * 5 % 11 - 5) / 2).reshape(3, 4)
    return logits, torch.tensor([1, 3, 0])


def fixed_media(step: int) -> Tensor:
    """Return the tiny network's input for ``step``: multiples of one eighth.

    Args:
      step: Index that shifts the pattern, so each update sees new images.

    Returns:
      media: Float32 images of shape ``PROBE_SHAPE``.

    """
    count = PROBE_SHAPE[0] * PROBE_SHAPE[1] * PROBE_SHAPE[2] * PROBE_SHAPE[3]
    return ((torch.arange(count) * (3 + 2 * step) % 17 - 8) / 8).reshape(PROBE_SHAPE)


def fixed_label(step: int) -> Tensor:
    """Return the tiny network's targets for ``step``.

    Args:
      step: Index that shifts the classes.

    Returns:
      label: One class per image of ``fixed_media``.

    """
    return torch.tensor([step % CLASSES, (step + 1) % CLASSES])


def fixed_images(count: int, *, channels: int = 3, side: int = 32) -> Tensor:
    """Return uint8 square images in a fixed pattern.

    Args:
      count: Images to build.
      channels: Channels of each image.
      side: Height and width of each image.

    Returns:
      images: ``(count, channels, side, side)`` pixels.

    """
    pixels = torch.arange(count * channels * side * side) * 37 % 251
    return pixels.reshape(count, channels, side, side).to(torch.uint8)


def synthetic_split() -> tuple[Tensor, Tensor]:
    """Return images whose pixels name their own position, and their labels.

    Channel 0 holds ``row + 1``, channel 1 ``column + 1`` and channel 2
    ``index + 1``, so a crop and a mirror of one is recognizable by its values.

    Returns:
      media: ``(LOADER_IMAGES, 3, 32, 32)`` uint8 images.
      label: The index of each image.

    """
    row, column = torch.meshgrid(
        torch.arange(1, 33),
        torch.arange(1, 33),
        indexing="ij",
    )
    media = torch.stack(
        [
            torch.stack([row, column, torch.full((32, 32), index + 1)])
            for index in range(LOADER_IMAGES)
        ],
    ).to(torch.uint8)
    return media, torch.arange(LOADER_IMAGES)


def full_size_media() -> Tensor:
    """Return a ``(2, 3, 32, 32)`` batch of multiples of one eighth.

    Five 2x2 pools leave VGG-16 one position, so the full-size networks are run
    on images of the reference's size rather than ``PROBE_SHAPE``.

    Returns:
      media: Float32 images.

    """
    return ((torch.arange(2 * 3 * 32 * 32) * 5 % 17 - 8) / 8).reshape(2, 3, 32, 32)


def synthetic_gray_split() -> tuple[Tensor, Tensor]:
    """Return ``LOADER_IMAGES`` grayscale 28x28 images, Fashion-MNIST's shape, and labels.

    Every pixel differs from its neighbours, so a crop and a mirror of one are
    recognizable even after the resize to 32x32 blends them.

    Returns:
      media: ``(LOADER_IMAGES, 1, 28, 28)`` uint8 images.
      label: The index of each image.

    """
    position = torch.arange(28 * 28)
    media = torch.stack(
        [(position * 37 + 101 * index) % 251 for index in range(LOADER_IMAGES)],
    )
    return media.reshape(LOADER_IMAGES, 1, 28, 28).to(torch.uint8), torch.arange(
        LOADER_IMAGES,
    )


def prepared_gray_split() -> tuple[Tensor, Tensor]:
    """Return :func:`synthetic_gray_split` as ``prepare`` caches it: resized to 32x32."""
    media, label = synthetic_gray_split()
    return resized(media, size=32), label


@dataclass(frozen=True, slots=True, kw_only=True)
class Setting:
    """One of the reference script's ``--arch``/``--dataset`` settings, and the port's."""

    arch: str
    """The reference's ``--arch``."""

    dataset: str
    """The reference's ``--dataset``."""

    classes: int
    """Classes of the dataset, as the reference's loaders report them."""

    apply: Callable[[VisionTrainLoop], VisionTrainLoop]
    """Switches a recipe to this setting; see ``settings.py``."""

    split: Callable[[], tuple[Tensor, Tensor]]
    """The synthetic split as the port's loader reads it, prepared."""

    def experiment(self) -> VisionTrainLoop:
        """Return ``exp000`` switched to this setting."""
        return self.apply(exp000())


SETTINGS: Final = {
    "vgg16bn-cifar100": Setting(
        arch="vgg16bn",
        dataset="cifar100",
        classes=100,
        apply=vgg16bn_cifar100,
        split=synthetic_split,
    ),
    "mobilenetv2-fmnist": Setting(
        arch="mobilenetv2",
        dataset="fmnist",
        classes=10,
        apply=mobilenetv2_fmnist,
        split=prepared_gray_split,
    ),
    "resnet110-cifar100": Setting(
        arch="resnet110",
        dataset="cifar100",
        classes=100,
        apply=resnet110_cifar100,
        split=synthetic_split,
    ),
    "resnet20-cifar10": Setting(
        arch="resnet20",
        dataset="cifar10",
        classes=10,
        apply=resnet20_cifar10,
        split=synthetic_split,
    ),
}
"""The reference's other settings, keyed by the task's own labels."""

CAPTURED: Final = ("vgg16bn-cifar100", "mobilenetv2-fmnist", "resnet20-cifar10")
"""The settings of :data:`SETTINGS` the ``cv-classification-loss`` script runs.

Its ``--arch`` has no ``resnet110``; that setting is proven by the captures of
the tasks whose scripts run it, against those scripts."""


def model_logits(model: nn.Module, media: Tensor) -> Tensor:
    """Run ``model`` on ``media``, typed as the images-to-logits map it is.

    Args:
      model: A classifier.
      media: A batch of images.

    Returns:
      logits: The class scores.

    """
    return cast(Tensor, model(media))


def tiny_step(
    factory: Callable[[], VisionLoop],
    *,
    blocks: tuple[int, ...],
    classes: int,
    epochs: int,
) -> VisionTrainStep:
    """Build the step of ``factory``, shrunk in depth, width and classes only.

    Args:
      factory: The experiment whose recipe is kept.
      blocks: Blocks per stage of a ResNet; see :func:`tiny_model`.
      classes: Logits.
      epochs: Schedule horizon.

    Returns:
      step: A CPU step with the experiment's optimizer, loss and schedule.

    """
    config = factory().step
    tiny_model(config.model, blocks=blocks)
    assert isinstance(config.model, (ResNet.Config, VGG16BN.Config, MobileNetV2.Config))
    config.model.channels_out = classes
    config.parallelism = NoParallel.Config(device="cpu")
    config.train_budget_epochs = epochs
    return _built(config)


def tiny_model(
    model: Makeable[nn.Module],
    *,
    blocks: tuple[int, ...] = (1, 1, 1),
) -> None:
    """Shrink a network config in place to the shallowest the reference can build.

    A ResNet takes ``blocks`` and the stage widths of :data:`TINY_WIDTHS`. A
    VGG16BN or MobileNetV2 is cut to the stages of :data:`TINY_ARCHS`, and its
    stem, last convolution and classifier to :data:`TINY_WIDTHS`. Every slot,
    and the class count, is left as it was.

    Args:
      model: A ResNet, VGG16BN or MobileNetV2 config.
      blocks: Blocks per stage of a ResNet.

    Raises:
      TypeError: ``model`` is none of the three.

    """
    if isinstance(model, VGG16BN.Config):
        model.channels_hidden = ((4,), (TINY_WIDTHS[512],))
        model.channels_hidden_head = TINY_WIDTHS[512]
    elif isinstance(model, MobileNetV2.Config):
        model.channels_stem = TINY_WIDTHS[32]
        model.channels_hidden = (4, TINY_WIDTHS[320])
        model.channels_last = TINY_WIDTHS[1280]
        model.blocks_per_stage = (1, 1)
        model.expand_ratios = (1, 1)
        model.strides = (2, 2)
    elif isinstance(model, ResNet.Config):
        model.channels_hidden = tuple(TINY_WIDTHS[w] for w in model.channels_hidden)
        model.blocks_per_stage = blocks
        # A block that keys on a width, as stochastic depth's reset does, keys on
        # the narrowed one; upstream's literal compares the unnarrowed arguments.
        block = model.block
        if isinstance(block, WidthKeyed):
            block.reset_width = TINY_WIDTHS[block.reset_width]
    else:
        raise TypeError(
            "tiny_model shrinks a ResNet, VGG16BN or MobileNetV2 config; got "
            f"{model!r}.",
        )


def full_step(factory: Callable[[], VisionLoop] = exp000) -> VisionTrainStep:
    """Build an experiment's step at full size on the CPU.

    Args:
      factory: The experiment; ``exp000`` by default.

    Returns:
      step: Its network, at the experiment's depth and width.

    """
    config = factory().step
    config.parallelism = NoParallel.Config(device="cpu")
    return _built(config)


def schedule_record(
    factory: Callable[[], VisionLoop] = exp000,
) -> dict[str, Tensor]:
    """Return the learning rate of each of the 200 epochs of an experiment.

    Args:
      factory: The experiment; ``exp000`` by default.

    Returns:
      record: The 200 rates, as float64.

    """
    step = tiny_step(factory, blocks=(1, 1, 1), classes=CLASSES, epochs=200)
    base_lr = from_plain(cast(object, step.optimizer.param_groups[0]["lr"]), float)
    rates = [
        step.schedule(epoch, base_lr=base_lr, total_epochs=200) for epoch in range(200)
    ]
    return {"schedule/lr": torch.tensor(rates, dtype=torch.float64)}


def optimizer_record(
    factory: Callable[[], VisionLoop] = exp000,
) -> dict[str, Tensor]:
    """Return the settings of the optimizer an experiment builds.

    Args:
      factory: The experiment; ``exp000`` by default.

    Returns:
      record: The entries of :data:`HYPERPARAMETERS`, as float64.

    """
    step = tiny_step(factory, blocks=(1, 1, 1), classes=CLASSES, epochs=200)
    group = cast(Mapping[str, object], step.optimizer.param_groups[0])
    values = [float(cast(float, group[name])) for name in HYPERPARAMETERS]
    return {"optimizer/settings": torch.tensor(values, dtype=torch.float64)}


def network_record() -> dict[str, Tensor]:
    """Initialize the shallowest network at :data:`TINY_WIDTHS` and run it.

    Returns:
      record: Where the generator stands afterwards, and the logits of a
        training-mode and then an evaluation-mode forward.

    """
    with seeded(SEED), host_agnostic_numerics():
        step = tiny_step(exp000, blocks=(1, 1, 1), classes=CLASSES, epochs=EPOCHS)
        model = step.model
        draws = rng_fingerprint()
        media = fixed_media(STEPS)
        model.train()
        train = model_logits(model, media)
        model.eval()
        evaluated = model_logits(model, media)
    return {
        "network/draws": draws,
        "network/train": train.detach(),
        "network/eval": evaluated.detach(),
    }


def draws_record(
    factory: Callable[[], VisionLoop] = exp000,
) -> dict[str, Tensor]:
    """Return where the generator stands after building an experiment's full network.

    Args:
      factory: The experiment; ``exp000`` by default.

    Returns:
      record: The next eight draws, under ``draws/<experiment_name>``.

    """
    key = f"draws/{factory().experiment_name}"
    with seeded(SEED), host_agnostic_numerics():
        _ = full_step(factory)
        return {key: rng_fingerprint()}


def transform_record() -> dict[str, Tensor]:
    """Run both image pipelines on two fixed images.

    Returns:
      record: The training pipeline's crops and mirrors, and the test
        pipeline's output, both normalized.

    """
    config = Cifar100Data.Config()
    mean, std = torch.tensor(config.mean), torch.tensor(config.std)
    images = fixed_images(2)
    stream = torch.Generator().manual_seed(SEED)
    return {
        "transform/train": normalize(
            pad_crop_flip(images, stream, pad=config.translate_pad),
            mean=mean,
            std=std,
        ),
        "transform/test": normalize(images, mean=mean, std=std),
    }


def best_record() -> dict[str, Tensor]:
    """Report 200 scripted evaluations to the accuracy metric.

    Returns:
      record: The best accuracy, in percent, rounded to the two places the
        reference prints.

    """
    metric = BestAccuracy.Config().make()
    label = torch.zeros(SCRIPTED_TOTAL, dtype=torch.int64)
    hit, miss = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    for correct in SCRIPTED_CORRECT:
        metric.reset()
        metric.update(
            torch.cat(
                [hit.expand(correct, 2), miss.expand(SCRIPTED_TOTAL - correct, 2)],
            ),
            label=label,
        )
        _ = metric.compute()
    return {
        "best/accuracy": torch.tensor(
            float(f"{metric.compute()['best']:.2f}"),
            dtype=torch.float64,
        ),
    }


def loader_batches(
    directory: Path,
) -> tuple[list[list[dict[str, Tensor]]], list[list[dict[str, Tensor]]]]:
    """Run the loaders over the synthetic split, alternating the two splits.

    Args:
      directory: Scratch directory the split is written to.

    Returns:
      passes: Per training pass the batches in order, then per evaluation pass
        the batches in order.

    """
    return dataset_batches(Cifar100Data.Config(), directory, split=synthetic_split())


def loader_record(directory: Path) -> dict[str, Tensor]:
    """Return what the training loader drew over the synthetic split.

    Args:
      directory: Scratch directory the split is written to.

    Returns:
      record: Per training pass the label of every image in order, and the
        crop offsets and mirror each was given.

    """
    train, _ = loader_batches(directory)
    return decode_batches(
        train,
        config=Cifar100Data.Config(),
        source=synthetic_split()[0],
    )


def setting_record(
    name: str,
    factory: Callable[[], VisionLoop] | None = None,
    *,
    key: str | None = None,
) -> dict[str, Tensor]:
    """Build a setting's shallowest network, then run it in training and in evaluation mode.

    The full-size network is compared live by the capture; the golden holds
    the shallowest one, the same recipe at :func:`tiny_model`'s depth.

    Args:
      name: A key of :data:`SETTINGS`.
      factory: The experiment to build; the setting's pristine one by default.
        A task passes its own fork at the setting.
      key: Names the record; ``name`` by default.

    Returns:
      record: Where the generator stands after the build, and the logits of
        :func:`setting_forward`, under ``settings/<key>/``.

    """
    key = key or name
    with seeded(SEED), host_agnostic_numerics():
        model = tiny_step(
            factory or SETTINGS[name].experiment,
            blocks=(1, 1, 1),
            classes=CLASSES,
            epochs=EPOCHS,
        ).model
        draws = rng_fingerprint()
    return {f"settings/{key}/draws": draws} | setting_forward(model, key)


def setting_forward(model: nn.Module, key: str) -> dict[str, Tensor]:
    """Run a setting's shallowest network on :func:`fixed_media`, training then evaluation.

    Each forward starts from the task's seed: a training-mode dropout draws
    from the global generator, and both sides must draw the same masks.

    Args:
      model: The setting's network, the port's or the reference's.
      key: Names the record; a key of :data:`SETTINGS` or a task's own.

    Returns:
      record: ``settings/<key>/train`` and ``settings/<key>/eval`` logits.

    """
    record: dict[str, Tensor] = {}
    media = fixed_media(STEPS)
    for mode in ("train", "eval"):
        model.train(mode == "train")
        with seeded(SEED), host_agnostic_numerics():
            record[f"settings/{key}/{mode}"] = model_logits(model, media).detach()
    return record


def setting_batches(
    directory: Path,
    name: str,
) -> tuple[list[list[dict[str, Tensor]]], list[list[dict[str, Tensor]]]]:
    """Run a setting's loaders over its synthetic split, alternating the splits.

    Args:
      directory: Scratch directory the split is written to.
      name: A key of :data:`SETTINGS`.

    Returns:
      passes: Per training pass the batches in order, then per evaluation pass.

    """
    setting = SETTINGS[name]
    return dataset_batches(
        setting.experiment().dataset,
        directory,
        split=setting.split(),
    )


def dataset_batches(
    config: Cifar100Data.Config,
    directory: Path,
    *,
    split: tuple[Tensor, Tensor],
) -> tuple[list[list[dict[str, Tensor]]], list[list[dict[str, Tensor]]]]:
    """Run any dataset config's loaders over ``split``, alternating the two splits.

    For a task fork whose dataset is no entry of :data:`SETTINGS`, such as a
    changed ``augment`` or another dataset's statistics.

    Args:
      config: The dataset; its directory and batch sizes are overwritten.
      directory: Scratch directory ``split`` is written to, as both splits.
      split: Uint8 images and labels, as ``prepare`` caches them.

    Returns:
      passes: Per training pass the batches in order, then per evaluation pass;
        ``LOADER_PASSES`` of each at batch ``LOADER_BATCH``, under seed ``SEED``.

    """
    media, label = split
    for name in ("train", "test"):
        torch.save({"media": media, "label": label}, directory / f"{name}.pt")
    config.base_dir = None
    config.working_dir = directory
    config.batch_size = config.eval_batch_size = LOADER_BATCH
    train: list[list[dict[str, Tensor]]] = []
    evaluation: list[list[dict[str, Tensor]]] = []
    with seeded(SEED):
        data = config.make()
        for _ in range(LOADER_PASSES):
            train.append(list(data.train_dataloader()))
            evaluation.append(list(data.eval_dataloader()))
    return train, evaluation


def setting_loader_record(directory: Path, name: str) -> dict[str, Tensor]:
    """Return what a setting's training loader drew over its synthetic split.

    Args:
      directory: Scratch directory the split is written to.
      name: A key of :data:`SETTINGS`.

    Returns:
      record: As :func:`loader_record`, keyed under ``settings/<name>/``.

    """
    setting = SETTINGS[name]
    train, _ = setting_batches(directory, name)
    record = decode_batches(
        train,
        config=setting.experiment().dataset,
        source=setting.split()[0],
    )
    return {f"settings/{name}/{key}": value for key, value in record.items()}


def fmnist_transform_record() -> dict[str, Tensor]:
    """Run both Fashion-MNIST pipelines on two fixed 28x28 grayscale images.

    Returns:
      record: ``transform/fmnist/train`` and ``transform/fmnist/test``, each
        ``(2, 3, 32, 32)``: resized, then cropped and mirrored for training,
        then normalized into three channels.

    """
    config = SETTINGS["mobilenetv2-fmnist"].experiment().dataset
    mean, std = torch.tensor(config.mean), torch.tensor(config.std)
    images = resized(fixed_images(2, channels=1, side=28), size=32)
    stream = torch.Generator().manual_seed(SEED)
    return {
        "transform/fmnist/train": normalize(
            pad_crop_flip(images, stream, pad=config.translate_pad),
            mean=mean,
            std=std,
        ),
        "transform/fmnist/test": normalize(images, mean=mean, std=std),
    }


def settings_record(directory: Path) -> dict[str, Tensor]:
    """Run every setting's comparisons and return all their values.

    Args:
      directory: Scratch directory the loader runs write their splits to.

    Returns:
      record: Each setting's network and loader records, and the Fashion-MNIST
        pipelines.

    """
    record = fmnist_transform_record()
    for name in CAPTURED:
        record |= setting_record(name) | setting_loader_record(directory, name)
    return record


def task_settings_record(
    experiments: Mapping[str, Mapping[str, Callable[[], VisionLoop]]],
    directory: Path,
) -> dict[str, Tensor]:
    """Build every variant of a task at each setting, and run each setting's loader.

    Args:
      experiments: Per key of :data:`SETTINGS`, the task's experiment of each
        variant (one per baseline) at that setting.
      directory: Scratch directory the loader runs write their splits to.

    Returns:
      record: ``settings/<setting>/<variant>/{draws,train,eval}`` per
        experiment, and ``settings/<setting>/loader/*`` for each setting whose
        dataset is not ``exp000``'s CIFAR-100, whose loader is proven already.

    """
    record: dict[str, Tensor] = {}
    for setting, variants in experiments.items():
        for variant, factory in variants.items():
            record |= setting_record(setting, factory, key=f"{setting}/{variant}")
        if SETTINGS[setting].dataset != "cifar100":
            record |= setting_loader_record(directory, setting)
    return record


def fmnist_draw_sites(directory: Path) -> dict[str, int]:
    """Count the draws of building MobileNetV2 and of the Fashion-MNIST data path.

    Args:
      directory: Scratch directory the loader run writes its split to.

    Returns:
      counts: As :func:`draw_sites`.

    """
    with recorded_draws() as counts:
        with seeded(SEED):
            _ = full_step(SETTINGS["mobilenetv2-fmnist"].experiment)
        _ = fmnist_transform_record()
        _ = setting_loader_record(directory, "mobilenetv2-fmnist")
    return counts


def port_record(directory: Path) -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Args:
      directory: Scratch directory the loader run writes its split to.

    Returns:
      record: The union of the records above, keyed as the stored goldens are.

    """
    return (
        schedule_record()
        | optimizer_record()
        | network_record()
        | draws_record()
        | best_record()
        | transform_record()
        | loader_record(directory)
    )


def decode_batches(
    passes: list[list[dict[str, Tensor]]],
    *,
    config: Cifar100Data.Config,
    source: Tensor,
) -> dict[str, Tensor]:
    """Recover each image's crop offsets and mirror from a loader's batches.

    Args:
      passes: Per training pass, the batches in order.
      config: The loader's dataset: its padding and channel statistics.
      source: The uint8 images the loader read, indexed by label.

    Returns:
      record: ``loader/label``, ``loader/top``, ``loader/left`` and
        ``loader/flip``, each of shape ``(passes, images)``.

    """
    mean, std = torch.tensor(config.mean), torch.tensor(config.std)
    padded = torch.nn.functional.pad(source, (config.translate_pad,) * 4)
    found: dict[str, list[list[int]]] = {"label": [], "top": [], "left": [], "flip": []}
    for batches in passes:
        for values in found.values():
            values.append([])
        for batch in batches:
            labels = from_plain(batch["label"].tolist(), list[int])
            for image, label in zip(batch["media"], labels, strict=True):
                crop = _decode(image, padded[label], mean=mean, std=std)
                for key, value in zip(found, (label, *crop), strict=True):
                    found[key][-1].append(value)
    return {f"loader/{key}": torch.tensor(values) for key, values in found.items()}


def draw_sites(directory: Path) -> dict[str, int]:
    """Count the random draws of a network, a transform and a loader run.

    Args:
      directory: Scratch directory the loader run writes its split to.

    Returns:
      counts: Maps ``"<api>@<file>:<function>"`` to the number of calls.

    """
    with recorded_draws() as counts:
        _ = network_record()
        _ = transform_record()
        _ = loader_record(directory)
    return counts


def _built(config: VisionTrainStep.Config) -> VisionTrainStep:
    """Make the step, narrowing what ``make`` returns."""
    step = config.make()
    assert isinstance(step, VisionTrainStep)
    return step


def _decode(
    image: Tensor,
    padded: Tensor,
    *,
    mean: Tensor,
    std: Tensor,
) -> tuple[int, int, int]:
    """Find the offsets and mirror that turn ``padded`` into ``image``."""
    size = image.shape[-1]
    offsets = padded.shape[-1] - size + 1
    for top in range(offsets):
        for left in range(offsets):
            crop = padded[:, top : top + size, left : left + size]
            for flip in (0, 1):
                candidate = crop.flip(-1) if flip else crop
                if torch.equal(
                    image,
                    normalize(candidate[None], mean=mean, std=std)[0],
                ):
                    return top, left, flip
    raise ValueError("The image is not a crop of its source at any offset.")
