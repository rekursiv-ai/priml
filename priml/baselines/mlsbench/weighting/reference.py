"""Inputs and port-side runs the ``cv-sample-weighting`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script, each baseline's edit
of it, and these runs on the same inputs under the same pins, compares them
bit for bit, and stores the script's values under ``testdata/``;
``reference_test.py`` replays these runs against them. The network's widths,
the augmentation, the schedule and the optimizer are the
``cv-classification-loss`` port's; this task adds the long-tailed subset, the
class weights and the weighted loss, at ResNet-32.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, cast

from torch import Tensor
from treekle import from_plain

import torch

from priml.baselines.mlsbench.pytorch_vision import reference as vision
from priml.baselines.mlsbench.pytorch_vision.data import Cifar100Data, normalize
from priml.baselines.mlsbench.pytorch_vision.train_step import (
    LossContext,
    VisionTrainStep,
)
from priml.baselines.mlsbench.weighting.data import (
    LongTailCifar100Data,
    long_tail_indices,
)
from priml.baselines.mlsbench.weighting.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.mlsbench.weighting.loss import ClassWeightedCrossEntropy
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from pathlib import Path

    from priml.baselines.mlsbench.pytorch_vision.experiments import VisionTrainLoop
    from priml.baselines.mlsbench.pytorch_vision.upstream import Batches

WEIGHTS: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "uniform": exp000,
    "inverse_freq": exp001,
    "effective_number": exp002,
    "balanced_softmax": exp003,
}
"""The experiment carrying each weighting, keyed as the script's variants."""

COUNTS: Final = (12.0, 5.0, 3.0, 1.0)
"""Per-class counts the weightings are compared on, one per logit of ``fixed_logits``."""

FLAT_COUNTS: Final = (3.0, 3.0, 3.0, 3.0)
"""Equal counts, where a weighting by count gaps has no gap to scale."""

LOCKSTEP_COUNTS: Final = (8.0, 3.0, 1.0)
"""Per-class counts of the lockstep runs, one per logit of the tiny network."""

RATIO: Final = 100.0
"""Imbalance ratio of the ``resnet32-cifar100lt`` setting."""

CLASSES: Final = 100
"""Classes of CIFAR-100."""

PER_CLASS: Final = 500
"""Training images per class of the balanced CIFAR-100 split."""

SPLIT_CLASSES: Final = 4
"""Classes holding images in the synthetic long-tailed split."""

SPLIT_PER_CLASS: Final = 6
"""Images per class of the synthetic split before the cut."""

SPLIT_RATIO: Final = 4.0**33
"""Imbalance ratio over 100 classes that keeps 6, 3, 2 and 1 of the four held."""


def balanced_label() -> Tensor:
    """Return CIFAR-100's training labels in shape: each class 500 times, interleaved.

    Returns:
      label: ``(50_000,)`` int64, image ``i`` of class ``i % 100``.

    """
    return torch.arange(CLASSES * PER_CLASS) % CLASSES


def long_tail_split() -> tuple[Tensor, Tensor]:
    """Return images that name their own index, in four classes of six.

    Channel 0 holds ``row + 1``, channel 1 ``column + 1`` and channel 2
    ``index + 1``, so any crop and mirror of an image still names it.

    Returns:
      media: ``(24, 3, 32, 32)`` uint8 images.
      label: Image ``i`` is class ``(5 i mod 24) // 6``, so classes interleave.

    """
    count = SPLIT_CLASSES * SPLIT_PER_CLASS
    row, column = torch.meshgrid(
        torch.arange(1, 33),
        torch.arange(1, 33),
        indexing="ij",
    )
    media = torch.stack(
        [
            torch.stack([row, column, torch.full((32, 32), index + 1)])
            for index in range(count)
        ],
    ).to(torch.uint8)
    return media, torch.arange(count) * 5 % count // SPLIT_PER_CLASS


def weights_record() -> dict[str, Tensor]:
    """Return each weighting's weights and its weighted loss on fixed inputs.

    Returns:
      record: Per weighting, the weights of :data:`COUNTS` and of
        :data:`FLAT_COUNTS`, and the loss on ``fixed_logits`` with its gradient.

    """
    logits, label = vision.fixed_logits()
    record: dict[str, Tensor] = {}
    with host_agnostic_numerics():
        for name, factory in WEIGHTS.items():
            loss = _loss(factory)
            loss.bind_class_counts(torch.tensor(FLAT_COUNTS))
            record[f"weights/{name}/flat"] = cast(Tensor, loss.weight)
            loss.bind_class_counts(torch.tensor(COUNTS))
            record[f"weights/{name}"] = cast(Tensor, loss.weight)
            leaf = logits.clone().requires_grad_()
            context = LossContext(num_classes=4, epoch=0, total_epochs=1)
            value = loss(leaf, label=label, context=context)["loss"]
            value.backward()
            record[f"loss/{name}/value"] = value.detach()
            record[f"loss/{name}/grad"] = cast(Tensor, leaf.grad)
    return record


def counts_record() -> tuple[dict[str, Tensor], Tensor]:
    """Cut CIFAR-100's label layout to the setting's long tail.

    Returns:
      record: The 100 class counts, and the weights ``exp000``'s loss takes
        from them.
      indices: The kept positions, compared by the capture and too large to
        store.

    """
    indices, counts = long_tail_indices(
        balanced_label(),
        imbalance_ratio=RATIO,
        num_classes=CLASSES,
        seed=vision.SEED,
    )
    loss = _loss(exp000)
    loss.bind_class_counts(counts)
    return {"counts/cifar100lt": counts, "main/weight": cast(Tensor, loss.weight)}, (
        indices
    )


def subset_record() -> dict[str, Tensor]:
    """Cut the synthetic split as the loader comparison does.

    Returns:
      record: The kept positions in order, and the 100 class counts.

    """
    _, label = long_tail_split()
    indices, counts = long_tail_indices(
        label,
        imbalance_ratio=SPLIT_RATIO,
        num_classes=CLASSES,
        seed=vision.SEED,
    )
    return {"subset/indices": indices, "subset/counts": counts}


def schedule_record() -> dict[str, Tensor]:
    """Return the learning rate of each of the 200 epochs of ``exp000``.

    Returns:
      record: The 200 rates, as float64.

    """
    step = _tiny(exp000, epochs=200)
    base_lr = from_plain(cast(object, step.optimizer.param_groups[0]["lr"]), float)
    rates = [
        step.schedule(epoch, base_lr=base_lr, total_epochs=200) for epoch in range(200)
    ]
    return {"schedule/lr": torch.tensor(rates, dtype=torch.float64)}


def optimizer_record() -> dict[str, Tensor]:
    """Return the settings of the optimizer ``exp000`` builds.

    Returns:
      record: The entries of ``HYPERPARAMETERS``, as float64.

    """
    group = cast(
        Mapping[str, object],
        _tiny(exp000, epochs=200).optimizer.param_groups[0],
    )
    values = [float(cast(float, group[name])) for name in vision.HYPERPARAMETERS]
    return {"optimizer/settings": torch.tensor(values, dtype=torch.float64)}


def full_step() -> VisionTrainStep:
    """Build the step of ``exp000`` at full size on the CPU.

    Returns:
      step: ResNet-32 with 100 classes.

    """
    config = exp000().step
    config.parallelism = NoParallel.Config(device="cpu")
    step = config.make()
    assert isinstance(step, VisionTrainStep)
    return step


def draws_record() -> dict[str, Tensor]:
    """Return where the generator stands after building the full ``exp000`` network.

    Returns:
      record: The next eight draws.

    """
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        _ = full_step()
        return {"draws/exp000": rng_fingerprint()}


def lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train the shallowest network for two updates, one per epoch.

    Args:
      name: A key of :data:`WEIGHTS`.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: Per update the loss, the logits it came from and the logits of a
        fixed probe in evaluation mode afterwards.

    """
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    probes: list[Tensor] = []
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        step = _tiny(WEIGHTS[name], epochs=vision.EPOCHS)
        assert isinstance(step.loss, ClassWeightedCrossEntropy)
        step.loss.bind_class_counts(torch.tensor(LOCKSTEP_COUNTS))
        for index in range(vision.STEPS):
            step.timer_epoch.global_count = index
            result = step.train_step(
                media=vision.fixed_media(index),
                label=vision.fixed_label(index),
            )
            losses.append(result["loss"])
            logits.append(result["model"])
            probe = step.call_eval(media=vision.fixed_media(vision.STEPS))
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
    }


def loader_batches(directory: Path) -> tuple[Batches, Batches]:
    """Run the long-tailed loaders over the synthetic split, alternating the splits.

    Args:
      directory: Scratch directory the split is written to.

    Returns:
      passes: Per training pass the batches in order, then per evaluation pass
        the batches in order.

    """
    media, label = long_tail_split()
    for split in ("train", "test"):
        torch.save({"media": media, "label": label}, directory / f"{split}.pt")
    config = LongTailCifar100Data.Config()
    config.base_dir = None
    config.working_dir = directory
    config.imbalance_ratio = SPLIT_RATIO
    config.batch_size = config.eval_batch_size = vision.LOADER_BATCH
    train: Batches = []
    evaluation: Batches = []
    with vision.seeded(vision.SEED):
        data = config.make()
        for _ in range(vision.LOADER_PASSES):
            train.append(list(data.train_dataloader()))
            evaluation.append(list(data.eval_dataloader()))
    return train, evaluation


def decode_indices(passes: Batches) -> dict[str, Tensor]:
    """Name the source image of every training image a loader produced.

    The centre pixel lies inside the image under every crop and mirror, and
    its third channel holds the image's index plus one.

    Args:
      passes: Per training pass, the batches in order.

    Returns:
      record: ``loader/index``, of shape ``(passes, images)``.

    """
    config = Cifar100Data.Config()
    media, _ = long_tail_split()
    codes = normalize(
        media,
        mean=torch.tensor(config.mean),
        std=torch.tensor(config.std),
    )
    table = codes[:, 2, 16, 16]
    found = [
        [
            int((table == image[2, 16, 16]).nonzero().item())
            for batch in batches
            for image in batch["media"]
        ]
        for batches in passes
    ]
    return {"loader/index": torch.tensor(found)}


def port_record(directory: Path) -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Args:
      directory: Scratch directory the loader run writes its split to.

    Returns:
      record: The union of the records above, keyed as the stored golden is.

    """
    counts, _ = counts_record()
    train, _ = loader_batches(directory)
    record = (
        weights_record()
        | counts
        | subset_record()
        | schedule_record()
        | optimizer_record()
        | draws_record()
        | vision.best_record()
        | decode_indices(train)
    )
    for name in WEIGHTS:
        record |= lockstep_record(name)
    return record


def _loss(factory: Callable[[], VisionTrainLoop]) -> ClassWeightedCrossEntropy:
    """Make the loss of ``factory``'s step."""
    loss = factory().step.loss.make()
    assert isinstance(loss, ClassWeightedCrossEntropy)
    return loss


def _tiny(factory: Callable[[], VisionTrainLoop], *, epochs: int) -> VisionTrainStep:
    """Build ``factory``'s step at the shallowest depth with three classes."""
    return vision.tiny_step(
        factory,
        blocks=(1, 1, 1),
        classes=vision.CLASSES,
        epochs=epochs,
    )
