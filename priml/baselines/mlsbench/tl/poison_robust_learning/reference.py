"""Inputs and port-side runs the ``security-poison-robust-learning`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine driver, each baseline's edit
of its loss, and these runs on the same inputs under the same pins, compares
them bit for bit, and stores the driver's values under ``testdata/``;
``reference_test.py`` replays these runs against them. The networks, the data
pipeline, the schedule and the optimizer are the vision scaffold's; this task
adds the label flip, the robust losses and the two evaluations.

Evaluation logits are cloned before they are scored or stored: an
inference-mode tensor escapes ``host_agnostic_numerics``' exact rounding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from torch import Tensor

import torch

from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.train_step import (
    LossContext,
)
from priml.baselines.mlsbench.tl.poison_robust_learning.data import (
    flipped_labels,
)
from priml.baselines.mlsbench.tl.poison_robust_learning.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp010,
)
from priml.baselines.mlsbench.tl.poison_robust_learning.metrics import (
    PoisonRobustness,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import joined, rng_fingerprint
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from priml.baselines.mlsbench.tl.poison_robust_learning.experiments import (
        PoisonTrainLoop,
    )
    from priml.loss.custom_types import LossOutput

LOSSES: Final[dict[str, Callable[[], PoisonTrainLoop]]] = {
    "pristine": exp000,
    "cross_entropy": exp001,
    "generalized_ce": exp002,
    "symmetric_ce": exp003,
    "bootstrap": exp004,
}
"""Per variant of the driver, its experiment at the ``resnet20-cifar10`` setting."""

SETTINGS: Final[dict[str, tuple[str, Callable[[], PoisonTrainLoop]]]] = {
    "exp000": ("resnet20-cifar10", exp000),
    "exp005": ("vgg16bn-cifar100", exp005),
    "exp010": ("mobilenetv2-fmnist", exp010),
}
"""Each setting's pristine experiment, keyed by its name, with the setting's label."""

EPOCHS: Final = 100
"""Epochs of the recipe."""

LOCKSTEP_EPOCHS: Final = 2
"""Epochs of a lockstep run, so the second runs at an annealed rate."""

LOCKSTEP_BATCH: Final = reference.LOADER_IMAGES
"""Batch size of a lockstep run: one batch per pass over the synthetic split."""

LOCKSTEP_FRACTION: Final = 0.5
"""Poison fraction of a lockstep run: three of the synthetic split's seven images."""

FLIP_CLASSES: Final = 4
"""Classes of :func:`flip_labels`."""


def loss_cases() -> dict[str, tuple[Tensor, Tensor]]:
    """Return fixed logits and labels that drive every branch of every loss.

    Returns:
      cases: ``fixed``, the scaffold's ``(3, 4)`` multiples of one half;
        ``saturated``, ``(2, 5)`` scores whose true class's probability, and
        others, fall below every loss's probability floor.

    """
    saturated = torch.tensor(
        [[40.0, -40.0, 0.0, 1.0, 2.0], [-30.0, 0.0, 30.0, 5.0, -5.0]],
    )
    return {
        "fixed": reference.fixed_logits(),
        "saturated": (saturated, torch.tensor([1, 0])),
    }


def loss_record() -> dict[str, Tensor]:
    """Run every variant's loss on every case, forward and backward.

    Returns:
      record: ``loss/<variant>/<case>/{value,grad}``.

    """
    record: dict[str, Tensor] = {}
    for name, factory in LOSSES.items():
        loss = factory().step.loss.make()
        for case, (logits, label) in loss_cases().items():
            leaf = logits.clone().requires_grad_()
            context = LossContext(
                num_classes=logits.shape[-1],
                epoch=0,
                total_epochs=EPOCHS,
            )
            with host_agnostic_numerics():
                value = loss(leaf, label=label, context=context)["loss"]
                value.backward()
            grad = leaf.grad
            assert isinstance(grad, Tensor)
            record[f"loss/{name}/{case}/value"] = value.detach()
            record[f"loss/{name}/{case}/grad"] = grad
    return record


def flip_labels() -> Tensor:
    """Return ``(LOADER_IMAGES,)`` labels below :data:`FLIP_CLASSES`, the last class among them."""
    return torch.arange(reference.LOADER_IMAGES) * 3 % FLIP_CLASSES


def flip_record() -> dict[str, Tensor]:
    """Flip half of :func:`flip_labels` with the recipe's seed.

    Returns:
      record: ``flip/labels`` the training labels, ``flip/index`` the flipped
        positions in the permutation's order, ``flip/mask`` which are flipped.

    """
    label, index = flipped_labels(
        flip_labels(),
        num_classes=FLIP_CLASSES,
        fraction=LOCKSTEP_FRACTION,
        seed=exp000().dataset.poison_seed,
    )
    mask = torch.zeros(len(label), dtype=torch.bool)
    mask[index] = True
    # A training item's label is its flipped one, which the driver's
    # ``__getitem__`` reads back; the capture checks it against these.
    return {
        "flip/labels": label,
        "flip/index": index,
        "flip/mask": mask,
        "flip/items": label,
    }


def evaluate_logits() -> tuple[Tensor, Tensor]:
    """Return ``(LOADER_IMAGES, 4)`` logits, with ties, and their labels for both evaluations.

    Returns:
      logits: Multiples of one half; the third row ties its two largest.
      label: :func:`flip_labels`, the clean labels the poisoned split flips.

    """
    logits = ((torch.arange(28) * 7 % 9 - 4) / 2).reshape(reference.LOADER_IMAGES, 4)
    logits[2] = torch.tensor([1.0, 1.0, 0.0, -1.0])
    return logits, flip_labels()


def evaluate_record() -> dict[str, Tensor]:
    """Score :func:`evaluate_logits` as both of the driver's evaluations do, in batches of two.

    Returns:
      record: ``evaluate/test_acc`` and ``evaluate/poison_fit``, float64.

    """
    logits, clean = evaluate_logits()
    label, index = flipped_labels(
        clean,
        num_classes=FLIP_CLASSES,
        fraction=LOCKSTEP_FRACTION,
        seed=exp000().dataset.poison_seed,
    )
    poison = torch.zeros(len(label), dtype=torch.bool)
    poison[index] = True
    metric = PoisonRobustness.Config().make()
    for start in range(0, len(logits), LOCKSTEP_BATCH):
        rows = slice(start, start + LOCKSTEP_BATCH)
        metric.update(logits[rows], label=clean[rows], split=torch.tensor(0))
        metric.update(
            logits[rows],
            label=label[rows],
            poison=poison[rows],
            split=torch.tensor(1),
        )
    metrics = metric.compute()
    return {
        f"evaluate/{key}": torch.tensor(metrics[key], dtype=torch.float64)
        for key in ("test_acc", "poison_fit")
    }


def recipe_record() -> dict[str, Tensor]:
    """Return ``exp000``'s rate in every epoch and its optimizer's settings.

    Returns:
      record: ``recipe/lr`` (100 rates, float64) and ``recipe/optimizer`` (as
        :data:`~priml.baselines.mlsbench.dl.vision.reference.HYPERPARAMETERS`).

    """
    step = reference.tiny_step(
        exp000,
        blocks=(1, 1, 1),
        classes=reference.CLASSES,
        epochs=EPOCHS,
    )
    base_lr = float(cast(float, step.optimizer.param_groups[0]["lr"]))
    rates = [
        step.schedule(epoch, base_lr=base_lr, total_epochs=EPOCHS)
        for epoch in range(EPOCHS)
    ]
    return {
        "recipe/lr": torch.tensor(rates, dtype=torch.float64),
        "recipe/optimizer": reference.optimizer_record(exp000)["optimizer/settings"],
    }


def network_record() -> dict[str, Tensor]:
    """Build each setting's full network, then run it in training and evaluation mode.

    Returns:
      record: ``settings/<experiment>/{draws,train,eval}`` per :data:`SETTINGS`.

    """
    record: dict[str, Tensor] = {}
    for key, (setting, factory) in SETTINGS.items():
        record |= reference.setting_record(setting, factory, key=key)
    return record


def lockstep_record(
    name: str,
    directory: Path,
    *,
    states: list[dict[str, Tensor]] | None = None,
    media: list[Tensor] | None = None,
) -> dict[str, Tensor]:
    """Train the shallowest ResNet on the poisoned synthetic split, then evaluate it.

    The run is the ``resnet20-cifar10`` recipe shrunk to :data:`LOCKSTEP_EPOCHS`
    epochs of batches of :data:`LOCKSTEP_BATCH` over the seven-image synthetic
    split, :data:`LOCKSTEP_FRACTION` of it flipped, from the task's seed.

    Args:
      name: A key of :data:`LOSSES`.
      directory: Scratch directory the split is written to.
      states: When given, receives a copy of the model's whole state after
        each update.
      media: When given, receives the images of every evaluation forward.

    Returns:
      record: Under ``lockstep/<name>/``, per update the loss, the logits and
        labels joined, the epoch the loss was told and the rate; the logits of
        every evaluation forward the driver makes, joined; the published
        metrics; and where the generator stands.

    """
    config = _lockstep_config(name, directory)
    contexts: list[LossContext] = []
    trained: dict[str, list[Tensor]] = {"loss": [], "logits": [], "label": []}
    rates: list[float] = []
    evaluated: list[Tensor] = []
    metric = PoisonRobustness.Config().make()
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = config.step.make()
        data = config.dataset.make()
        step.loss = _ContextSpy(step.loss, contexts)
        for epoch in range(LOCKSTEP_EPOCHS):
            step.timer_epoch.global_count = epoch
            for batch in data.train_dataloader():
                result = step.train_step(**batch)
                rates.append(float(cast(float, step.optimizer.param_groups[0]["lr"])))
                trained["loss"].append(result["loss"])
                trained["logits"].append(result["model"])
                trained["label"].append(batch["label"])
                if states is not None:
                    states.append(
                        {k: v.clone() for k, v in step.model.state_dict().items()},
                    )
        for batch in data.eval_dataloader():
            images = batch["media"]
            logits = step.call_eval(media=images)
            assert isinstance(logits, Tensor)
            logits = logits.clone()
            metric.update(logits, **batch)
            poison = batch.get("poison")
            # The driver skips the forward of a poisoned batch holding no flip.
            if not isinstance(poison, Tensor) or bool(poison.any()):
                evaluated.append(logits)
                if media is not None:
                    media.append(images)
        metrics = metric.compute()
        draws = rng_fingerprint()
    key = f"lockstep/{name}"
    return {
        f"{key}/loss": torch.stack(trained["loss"]),
        f"{key}/logits": joined(trained["logits"]),
        f"{key}/label": joined(trained["label"]),
        f"{key}/epoch": torch.tensor([c.epoch for c in contexts]),
        f"{key}/lr": torch.tensor(rates, dtype=torch.float64),
        f"{key}/eval": joined(evaluated),
        f"{key}/metrics": torch.tensor(
            [metrics[k] for k in ("test_acc", "poison_fit", "robust_score")],
            dtype=torch.float64,
        ),
        f"{key}/draws": draws,
    }


def port_record(directory: Path) -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Args:
      directory: Scratch directory the lockstep runs write their split to.

    Returns:
      record: The union of the records above, keyed as the stored golden is.

    """
    record = (
        loss_record()
        | flip_record()
        | evaluate_record()
        | recipe_record()
        | network_record()
    )
    for name in LOSSES:
        record |= lockstep_record(name, directory)
    return record


def _lockstep_config(name: str, directory: Path) -> PoisonTrainLoop:
    """Shrink a variant's ``resnet20-cifar10`` experiment to a lockstep run, finalized."""
    media, label = reference.synthetic_split()
    for split in ("train", "test"):
        torch.save({"media": media, "label": label}, directory / f"{split}.pt")
    config = LOSSES[name]()
    config.base_dir = None
    config.working_dir = directory
    config.dataset.working_dir = directory
    config.dataset.batch_size = config.dataset.eval_batch_size = LOCKSTEP_BATCH
    config.dataset.poison_fraction = LOCKSTEP_FRACTION
    config.max_epochs = config.step.train_budget_epochs = LOCKSTEP_EPOCHS
    config.step.parallelism = NoParallel.Config(device="cpu")
    reference.tiny_model(config.step.model)
    return config.copy_tree().finalize()


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
