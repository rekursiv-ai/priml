"""Inputs and port-side runs the reference capture and its test share.

``scripts/capture_reference.py`` runs the pristine reference code and this
module's port-side runs on the same inputs under the same pins, compares them
bit for bit, and stores the reference's values under ``testdata/``.
``reference_test.py`` replays the port-side runs against those stored values.

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
from typing import TYPE_CHECKING, Final, cast

from torch import Tensor, nn
from treekle import from_plain

import torch

from priml.baselines.mlsbench.pytorch_vision.data import (
    Cifar100Data,
    normalize,
    pad_crop_flip,
)
from priml.baselines.mlsbench.pytorch_vision.draws import recorded_draws
from priml.baselines.mlsbench.pytorch_vision.experiments import (
    VisionTrainLoop,
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.mlsbench.pytorch_vision.metrics import BestAccuracy
from priml.baselines.mlsbench.pytorch_vision.model import ResNet
from priml.baselines.mlsbench.pytorch_vision.train_step import (
    LossContext,
    VisionTrainStep,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from priml.loss.custom_types import LossOutput

SEED: Final = 42
"""Seed of every pinned run, the task's own."""

CLASSES: Final = 3
"""Logits of the tiny network; its widths are the reference's."""

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

LOSSES: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "cross_entropy": exp000,
    "label_smoothing": exp001,
    "focal": exp002,
    "poly": exp003,
}
"""The experiment that carries each loss, so its own settings are compared."""

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


def fixed_images(count: int) -> Tensor:
    """Return uint8 images of 32x32 pixels in a fixed pattern.

    Args:
      count: Images to build.

    Returns:
      images: ``(count, 3, 32, 32)`` pixels.

    """
    pixels = torch.arange(count * 3 * 32 * 32) * 37 % 251
    return pixels.reshape(count, 3, 32, 32).to(torch.uint8)


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


def logits_of(model: nn.Module, media: Tensor) -> Tensor:
    """Run ``model`` on ``media``, typed as the images-to-logits map it is.

    Args:
      model: A classifier.
      media: A batch of images.

    Returns:
      logits: The class scores.

    """
    return cast(Tensor, model(media))


def tiny_step(
    factory: Callable[[], VisionTrainLoop],
    *,
    blocks: tuple[int, ...],
    classes: int,
    epochs: int,
) -> VisionTrainStep:
    """Build the step of ``factory``, shrunk in depth and classes only.

    Args:
      factory: The experiment whose recipe is kept.
      blocks: Blocks per stage; the widths stay the reference's.
      classes: Logits.
      epochs: Schedule horizon.

    Returns:
      step: A CPU step with the experiment's optimizer, loss and schedule.

    """
    config = factory().step
    assert isinstance(config.model, ResNet.Config)
    config.model.blocks_per_stage = blocks
    config.model.channels_out = classes
    config.parallelism = NoParallel.Config(device="cpu")
    config.train_budget_epochs = epochs
    return _built(config)


def full_step(factory: Callable[[], VisionTrainLoop] = exp000) -> VisionTrainStep:
    """Build an experiment's step at full size on the CPU.

    Args:
      factory: The experiment; ``exp000`` by default.

    Returns:
      step: Its network, at the experiment's depth and width.

    """
    config = factory().step
    config.parallelism = NoParallel.Config(device="cpu")
    return _built(config)


def loss_record() -> dict[str, Tensor]:
    """Run every loss on fixed logits, forward and backward.

    Returns:
      record: Each loss's value and its gradient with respect to the logits.

    """
    logits, label = fixed_logits()
    record: dict[str, Tensor] = {}
    with host_agnostic_numerics():
        for name, factory in LOSSES.items():
            leaf = logits.clone().requires_grad_()
            context = LossContext(num_classes=4, epoch=0, total_epochs=1)
            loss = factory().step.loss.make()(leaf, label=label, context=context)
            loss["loss"].backward()
            record[f"loss/{name}/value"] = loss["loss"].detach()
            record[f"loss/{name}/grad"] = cast(Tensor, leaf.grad)
    return record


def schedule_record(
    factory: Callable[[], VisionTrainLoop] = exp000,
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
    factory: Callable[[], VisionTrainLoop] = exp000,
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
    """Initialize the shallowest network at the reference's widths and run it.

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
        train = logits_of(model, media)
        model.eval()
        evaluated = logits_of(model, media)
    return {
        "network/draws": draws,
        "network/train": train.detach(),
        "network/eval": evaluated.detach(),
    }


def draws_record(
    factory: Callable[[], VisionTrainLoop] = exp000,
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


def lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train the shallowest network for ``STEPS`` updates, one per epoch.

    Args:
      name: A key of :data:`LOSSES`.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: Per update the loss, the logits it came from, the logits of a
        fixed probe in evaluation mode afterwards, and what the loss was given.

    """
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    probes: list[Tensor] = []
    contexts: list[LossContext] = []
    with seeded(SEED), host_agnostic_numerics():
        step = tiny_step(LOSSES[name], blocks=(1, 1, 1), classes=CLASSES, epochs=EPOCHS)
        step.loss = _ContextSpy(step.loss, contexts)
        for index in range(STEPS):
            step.timer_epoch.global_count = index
            result = step.train_step(
                media=fixed_media(index),
                label=fixed_label(index),
            )
            losses.append(result["loss"])
            logits.append(result["model"])
            probe = step.call_eval(media=fixed_media(STEPS))
            assert isinstance(probe, Tensor)
            probes.append(probe)
            if states is not None:
                states.append(
                    {k: v.clone() for k, v in step.model.state_dict().items()},
                )
    return {
        f"lockstep/{name}/loss": torch.stack(losses),
        f"lockstep/{name}/logits": torch.stack(logits),
        f"lockstep/{name}/probe": torch.stack(probes),
        f"lockstep/{name}/context": torch.tensor(
            [[c.num_classes, c.epoch, c.total_epochs] for c in contexts],
        ),
    }


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
    media, label = synthetic_split()
    for split in ("train", "test"):
        torch.save({"media": media, "label": label}, directory / f"{split}.pt")
    config = Cifar100Data.Config()
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


def loader_record(directory: Path) -> dict[str, Tensor]:
    """Return what the training loader drew over the synthetic split.

    Args:
      directory: Scratch directory the split is written to.

    Returns:
      record: Per training pass the label of every image in order, and the
        crop offsets and mirror each was given.

    """
    train, _ = loader_batches(directory)
    return decode_batches(train, translate_pad=Cifar100Data.Config().translate_pad)


def port_record(directory: Path) -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Args:
      directory: Scratch directory the loader run writes its split to.

    Returns:
      record: The union of the records above, keyed as the stored goldens are.

    """
    record = (
        loss_record()
        | schedule_record()
        | optimizer_record()
        | network_record()
        | draws_record()
        | best_record()
        | transform_record()
        | loader_record(directory)
    )
    for name in LOSSES:
        record |= lockstep_record(name)
    return record


def decode_batches(
    passes: list[list[dict[str, Tensor]]],
    *,
    translate_pad: int,
) -> dict[str, Tensor]:
    """Recover each image's crop offsets and mirror from a loader's batches.

    Args:
      passes: Per training pass, the batches in order.
      translate_pad: Padding the loader added before cropping.

    Returns:
      record: ``loader/label``, ``loader/top``, ``loader/left`` and
        ``loader/flip``, each of shape ``(passes, images)``.

    """
    source, _ = synthetic_split()
    config = Cifar100Data.Config()
    mean, std = torch.tensor(config.mean), torch.tensor(config.std)
    padded = torch.nn.functional.pad(source, (translate_pad,) * 4)
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


class _ContextSpy:
    """Forwards to a loss and keeps what it was told about the run."""

    def __init__(
        self,
        loss: Callable[..., LossOutput],
        contexts: list[LossContext],
    ) -> None:
        self.loss = loss
        self.contexts = contexts

    def __call__(
        self,
        logits: Tensor,
        *,
        label: Tensor,
        context: LossContext,
    ) -> LossOutput:
        """Record ``context`` and return the wrapped loss."""
        self.contexts.append(context)
        return self.loss(logits, label=label, context=context)


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
