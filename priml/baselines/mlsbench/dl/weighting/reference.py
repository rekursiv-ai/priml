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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, cast

from torch import Tensor

import torch

from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.data import Cifar100Data, normalize
from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop
from priml.baselines.mlsbench.dl.vision.train_step import (
    LossContext,
)
from priml.baselines.mlsbench.dl.weighting.data import (
    LongTailCifar100Data,
    long_tail_indices,
)
from priml.baselines.mlsbench.dl.weighting.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
    exp007,
    exp008,
    exp009,
    exp010,
    exp011,
)
from priml.baselines.mlsbench.dl.weighting.loss import ClassWeightedCrossEntropy
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from pathlib import Path

    from priml.baselines.mlsbench.dl.vision.upstream import Batches

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

SPLIT_RATIO_10: Final = (5 / 3) ** 9
"""Imbalance ratio over 10 classes that keeps 6, 3, 2 and 1 of the four held."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Setting:
    """One of the task's other launch scripts, and the port's experiments for it."""

    arch: str
    """The script's ``--arch``."""

    dataset: str
    """The script's ``--dataset``."""

    imbalance_ratio: float
    """The script's ``--imbalance-ratio``."""

    classes: int
    """Classes of the dataset."""

    experiments: dict[str, Callable[[], VisionTrainLoop]]
    """Each weighting's experiment, keyed as :data:`WEIGHTS`."""

    lockstep: bool
    """Whether its shallowest network differs from ``resnet32-cifar100lt``'s.

    Cut to one block per stage, a ResNet-32 over ten classes is the lockstep
    network ``lockstep/`` already proves, up to its class count, which the
    lockstep runs fix at ``reference.CLASSES``."""

    loader: bool
    """Whether its loader reads another dataset class than CIFAR-100's."""


SETTINGS: Final = {
    "resnet32-cifar10lt": Setting(
        arch="resnet32",
        dataset="cifar10",
        imbalance_ratio=100.0,
        classes=10,
        experiments={
            "uniform": exp004,
            "inverse_freq": exp005,
            "effective_number": exp006,
            "balanced_softmax": exp007,
        },
        lockstep=False,
        loader=True,
    ),
    "vgg16bn-cifar100lt": Setting(
        arch="vgg16bn",
        dataset="cifar100",
        imbalance_ratio=50.0,
        classes=100,
        experiments={
            "uniform": exp008,
            "inverse_freq": exp009,
            "effective_number": exp010,
            "balanced_softmax": exp011,
        },
        lockstep=True,
        loader=False,
    ),
}
"""The task's other settings, keyed by the task's labels."""


def balanced_label(classes: int = CLASSES) -> Tensor:
    """Return a CIFAR training split's labels in shape: interleaved, 50,000 in all.

    Args:
      classes: 100 for CIFAR-100, 10 for CIFAR-10.

    Returns:
      label: ``(50_000,)`` int64, image ``i`` of class ``i % classes``.

    """
    return torch.arange(CLASSES * PER_CLASS) % classes


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
    logits, label = reference.fixed_logits()
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
        seed=reference.SEED,
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
        seed=reference.SEED,
    )
    return {"subset/indices": indices, "subset/counts": counts}


def setting_counts_record(name: str) -> tuple[dict[str, Tensor], Tensor]:
    """Cut a setting's label layout to its long tail, and weigh it per weighting.

    Args:
      name: A key of :data:`SETTINGS`.

    Returns:
      record: Under ``settings/<name>/``: the class counts and, per weighting,
        the weights its experiment's loss takes from them.
      indices: The kept positions, compared by the capture and too large to
        store.

    """
    setting = SETTINGS[name]
    indices, counts = long_tail_indices(
        balanced_label(setting.classes),
        imbalance_ratio=setting.imbalance_ratio,
        num_classes=setting.classes,
        seed=reference.SEED,
    )
    record = {f"settings/{name}/counts": counts}
    # Under the upcast, as the script's ``main`` computes its weights.
    with host_agnostic_numerics():
        for weighting, factory in setting.experiments.items():
            loss = _loss(factory)
            loss.bind_class_counts(counts)
            record[f"settings/{name}/weight/{weighting}"] = cast(Tensor, loss.weight)
    return record, indices


def setting_record(name: str, *, include_lockstep: bool = True) -> dict[str, Tensor]:
    """Run every port-side comparison of a setting whose values fit a golden.

    Args:
      name: A key of :data:`SETTINGS`.
      include_lockstep: Whether to run the setting's two-update locksteps. The
        per-weighting setting lockstep tests compare those records separately.

    Returns:
      record: Under ``settings/<name>/``: the 200 rates and the optimizer,
        the shallowest network's draws and forwards, and the counts and
        weights. When ``include_lockstep`` is true and the setting's network
        differs, it also includes each weighting's lockstep.

    """
    setting = SETTINGS[name]
    uniform = setting.experiments["uniform"]
    prefix = f"settings/{name}/"
    record = {
        f"{prefix}{k}": v
        for k, v in (
            reference.schedule_record(uniform) | reference.optimizer_record(uniform)
        ).items()
    }
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = reference.tiny_step(
            uniform,
            blocks=(1, 1, 1),
            classes=reference.CLASSES,
            epochs=reference.EPOCHS,
        ).model
        record[f"{prefix}draws"] = rng_fingerprint()
    record |= reference.setting_forward(model, name)
    record |= setting_counts_record(name)[0]
    if setting.lockstep and include_lockstep:
        for weighting in setting.experiments:
            record |= setting_lockstep_record(f"{name}/{weighting}")
    return record


def settings_record() -> dict[str, Tensor]:
    """Return :func:`setting_record` of every one of :data:`SETTINGS`."""
    record: dict[str, Tensor] = {}
    for name in SETTINGS:
        record |= setting_record(name)
    return record


def draws_record() -> dict[str, Tensor]:
    """Return where the generator stands after building the full ``exp000`` network.

    Returns:
      record: The next eight draws.

    """
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        _ = reference.full_step(exp000)
        return {"draws/exp000": rng_fingerprint()}


def lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train the shallowest ResNet-32 for two updates, one per epoch.

    Args:
      name: A key of :data:`WEIGHTS`.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: As :func:`lockstep_run`, under ``lockstep/<name>/``.

    """
    return lockstep_run(WEIGHTS[name], prefix=f"lockstep/{name}/", states=states)


def setting_lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train a setting's shallowest network for two updates, one per epoch.

    Args:
      name: ``<setting>/<weighting>``, a key of :data:`SETTINGS` and one of
        its experiments.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: As :func:`lockstep_run`, under ``settings/<setting>/lockstep/<weighting>/``.

    """
    setting, weighting = name.split("/")
    return lockstep_run(
        SETTINGS[setting].experiments[weighting],
        prefix=f"settings/{setting}/lockstep/{weighting}/",
        states=states,
    )


def lockstep_run(
    factory: Callable[[], VisionTrainLoop],
    *,
    prefix: str,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train an experiment's shallowest network for two updates, one per epoch.

    Args:
      factory: The experiment; its network is cut as ``tiny_step`` cuts it.
      prefix: Put before every key.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: Per update the loss, the logits it came from and the logits of a
        fixed probe in evaluation mode afterwards.

    """
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    probes: list[Tensor] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = reference.tiny_step(
            factory,
            blocks=(1, 1, 1),
            classes=reference.CLASSES,
            epochs=reference.EPOCHS,
        )
        assert isinstance(step.loss, ClassWeightedCrossEntropy)
        step.loss.bind_class_counts(torch.tensor(LOCKSTEP_COUNTS))
        for index in range(reference.STEPS):
            step.timer_epoch.global_count = index
            result = step.train_step(
                media=reference.fixed_media(index),
                label=reference.fixed_label(index),
            )
            losses.append(result["loss"])
            logits.append(result["model"])
            # Outside inference mode, as the script's own probe is: an
            # inference tensor escapes the float64 upcast, so a 512-wide layer
            # would round by host.
            step.model.eval()
            with torch.no_grad():
                probes.append(
                    reference.logits_of(
                        step.model,
                        reference.fixed_media(reference.STEPS),
                    ),
                )
            step.model.train()
            if states is not None:
                states.append(
                    {k: v.clone() for k, v in step.model.state_dict().items()},
                )
    return {
        f"{prefix}loss": torch.stack(losses),
        f"{prefix}logits": torch.stack(logits),
        f"{prefix}probe": torch.stack(probes),
    }


def loader_batches(
    directory: Path,
    factory: Callable[[], VisionTrainLoop] = exp000,
    *,
    imbalance_ratio: float = SPLIT_RATIO,
) -> tuple[Batches, Batches]:
    """Run an experiment's long-tailed loaders over the synthetic split.

    Args:
      directory: Scratch directory the split is written to.
      factory: The experiment whose dataset is run; ``exp000`` by default.
      imbalance_ratio: Cut of the four held classes; :data:`SPLIT_RATIO`, which
        keeps 6, 3, 2 and 1 of them over 100 classes, by default.

    Returns:
      passes: Per training pass the batches in order, then per evaluation pass
        the batches in order.

    """
    media, label = long_tail_split()
    for split in ("train", "test"):
        torch.save({"media": media, "label": label}, directory / f"{split}.pt")
    config = factory().dataset
    assert isinstance(config, LongTailCifar100Data.Config)
    config.base_dir = None
    config.working_dir = directory
    config.imbalance_ratio = imbalance_ratio
    config.batch_size = config.eval_batch_size = reference.LOADER_BATCH
    train: Batches = []
    evaluation: Batches = []
    with reference.seeded(reference.SEED):
        data = config.make()
        for _ in range(reference.LOADER_PASSES):
            train.append(list(data.train_dataloader()))
            evaluation.append(list(data.eval_dataloader()))
    return train, evaluation


def decode_indices(
    passes: Batches,
    config: Cifar100Data.Config | None = None,
) -> dict[str, Tensor]:
    """Name the source image of every training image a loader produced.

    The centre pixel lies inside the image under every crop and mirror, and
    its third channel holds the image's index plus one.

    Args:
      passes: Per training pass, the batches in order.
      config: The loader's dataset, for its channel statistics; CIFAR-100's
        by default.

    Returns:
      record: ``loader/index``, of shape ``(passes, images)``.

    """
    config = config or Cifar100Data.Config()
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
        | reference.schedule_record(exp000)
        | reference.optimizer_record(exp000)
        | draws_record()
        | reference.best_record()
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
