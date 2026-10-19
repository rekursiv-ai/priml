"""Inputs and port-side runs the ``security-membership-inference-defense`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine driver, each baseline's edit
of its defense, and these runs on the same inputs under the same pins,
compares them bit for bit, and stores the driver's values under ``testdata/``;
``reference_test.py`` replays these runs against them. The network, the data
pipeline and the optimizer are the vision scaffold's; this task adds the
member split, the defenses, the step decay and the membership attack.

The attack scores a clone of each evaluation's logits: an inference-mode
tensor escapes ``host_agnostic_numerics``' exact rounding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from configgle import PartialConfig
from torch import Tensor

import torch

from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.data import (
    normalize,
    pad_crop_flip_normalize,
)
from priml.baselines.mlsbench.dl.vision.train_step import (
    LossContext,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.data import (
    SPLITS,
    member_split,
)
from priml.baselines.mlsbench.tl.membership_inference_defense.experiments import (
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
)
from priml.baselines.mlsbench.tl.membership_inference_defense.metrics import (
    MembershipInference,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import joined, rng_fingerprint
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from priml.baselines.mlsbench.tl.membership_inference_defense.experiments import (
        MembershipTrainLoop,
    )
    from priml.loss.custom_types import LossOutput

DEFENSES: Final[dict[str, tuple[Callable[[], MembershipTrainLoop], ...]]] = {
    "pristine": (exp000, exp005),
    "erm": (exp001, exp006),
    "label_smoothing": (exp002, exp007),
    "confidence_penalty": (exp003, exp008),
    "relaxloss": (exp004, exp009),
}
"""Per variant of the driver, its experiment at ten classes and at a hundred."""

EPOCHS: Final = 300
"""Epochs of the recipe."""

LOCKSTEP_EPOCHS: Final = 2
"""Epochs of a lockstep run: one of each RelaxLoss phase."""

LOCKSTEP_MILESTONES: Final = (1,)
"""Decay epochs of a lockstep run, so its second epoch runs at a decayed rate."""

LOCKSTEP_BATCH: Final = 3
"""Batch size of a lockstep run: the three members in one batch per epoch."""

METRICS: Final = ("test_acc", "mia_auc", "privacy_gap", "privacy_score")
"""The four metrics the driver publishes, in its order."""


def defense_cases() -> dict[str, tuple[Tensor, Tensor]]:
    """Return fixed logits and labels that drive every branch of every defense.

    Returns:
      cases: ``low``, four classes with mean cross-entropy below RelaxLoss's
        level of 1 and one example wrong; ``high``, the same mirrored, above
        it and all wrong; ``wide``, a hundred classes below the level of 0.5,
        the last example wrong, with probabilities below the confidence
        penalty's floor.

    """
    low = torch.tensor(
        [[3.0, 0.0, -1.0, 0.5], [0.0, 2.5, 0.0, -2.0], [1.5, 0.0, 2.0, 0.0]],
    )
    base = ((torch.arange(300) * 37 % 101 - 50) / 8).reshape(3, 100)
    wide = base * torch.tensor([[32.0], [32.0], [4.0]])
    return {
        "low": (low, torch.tensor([0, 1, 0])),
        "high": (low.flip(1), torch.tensor([0, 1, 0])),
        "wide": (wide, torch.tensor([30, 31, 62])),
    }


def defense_record() -> dict[str, Tensor]:
    """Run every variant's defense on every case in both RelaxLoss phases.

    Returns:
      record: ``defense/<variant>/<case>/<epoch>/{value,grad}``.

    """
    record: dict[str, Tensor] = {}
    for name, (ten, hundred) in DEFENSES.items():
        for case, (logits, label) in defense_cases().items():
            classes = logits.shape[-1]
            loss = (hundred if classes == 100 else ten)().step.loss.make()
            for epoch in (0, 1):
                leaf = logits.clone().requires_grad_()
                context = LossContext(
                    num_classes=classes,
                    epoch=epoch,
                    total_epochs=EPOCHS,
                )
                with host_agnostic_numerics():
                    value = loss(leaf, label=label, context=context)["loss"]
                    value.backward()
                key = f"defense/{name}/{case}/{epoch}"
                record[f"{key}/value"] = value.detach()
                record[f"{key}/grad"] = cast(Tensor, leaf.grad)
    return record


def attack_logits() -> dict[str, tuple[Tensor, Tensor]]:
    """Return fixed logits and labels of the three splits, with tied scores.

    Returns:
      logits: Per split of :data:`SPLITS`, ``(N, 4)`` logits in multiples of
        a quarter or an eighth, and ``(N,)`` labels. Many rows share their
        largest probability within a half, but no score is in both: NumPy's
        unstable sort orders a tie across the halves by the host's SIMD
        kernel, so such a tie would make the AUC the host's.

    """
    member = ((torch.arange(600) * 13 % 29 - 14) / 4).reshape(150, 4)
    nonmember = ((torch.arange(680) * 11 % 23 - 11) / 8).reshape(170, 4)
    test = ((torch.arange(36) * 5 % 17 - 8) / 4).reshape(9, 4)
    return {
        "test": (test, torch.arange(9) % 4),
        "member": (member, torch.arange(150) % 4),
        "nonmember": (nonmember, torch.arange(170) * 3 % 4),
    }


def attack_record() -> dict[str, Tensor]:
    """Run the attack on fixed logits of the three splits.

    Returns:
      record: ``attack/scores/<split>``, ``attack/accuracy`` per split and
        ``attack/auc``, the last two float64.

    """
    metric = MembershipInference.Config().make()
    with host_agnostic_numerics():
        for index, (logits, label) in enumerate(attack_logits().values()):
            metric.update(logits, label=label, split=torch.tensor(index))
        metrics = metric.compute()
    return scored("attack", metric) | {
        "attack/auc": torch.tensor(metrics["mia_auc"], dtype=torch.float64),
    }


def scored(key: str, metric: MembershipInference) -> dict[str, Tensor]:
    """Return an evaluation's scores and accuracy per split, under ``key``.

    Args:
      key: Prefix of the record's keys.
      metric: The attack after an evaluation.

    Returns:
      record: ``<key>/scores/<split>`` float32 and ``<key>/accuracy`` float64.

    """
    record = {
        f"{key}/scores/{split}": torch.from_numpy(metric.split_scores(split))
        for split in SPLITS
    }
    record[f"{key}/accuracy"] = torch.tensor(
        [metric.correct[split] / max(metric.total[split], 1) for split in SPLITS],
        dtype=torch.float64,
    )
    return record


def pipeline_images() -> Tensor:
    """Return the first two images of the synthetic split, whose crops are recognizable."""
    return reference.synthetic_split()[0][:2]


def pipeline_record() -> dict[str, Tensor]:
    """Run ``exp000``'s training and test pipelines, and the augmenting one, on two images.

    The capture compares these pixels live; they are not stored. The first two
    feed every lockstep run, whose stored values imply them, and the third is
    stored as :func:`augment_record`.

    Returns:
      record: ``pipeline/cifar10/{train,augment,test}``: training as
        ``exp000`` runs it, unaugmented; the scaffold's crop and mirror from
        the task's seed, which the driver's ``--augment`` selects; and the
        test pipeline.

    """
    config = exp000().dataset
    mean, std = torch.tensor(config.mean), torch.tensor(config.std)
    images = pipeline_images()
    augment = config.augment.make()
    return {
        "pipeline/cifar10/train": augment(
            images,
            torch.Generator(),
            pad=config.translate_pad,
            mean=mean,
            std=std,
        ),
        "pipeline/cifar10/augment": pad_crop_flip_normalize(
            images,
            torch.Generator().manual_seed(reference.SEED),
            pad=config.translate_pad,
            mean=mean,
            std=std,
        ),
        "pipeline/cifar10/test": normalize(images, mean=mean, std=std),
    }


def augment_record(augmented: Tensor) -> dict[str, Tensor]:
    """Recover each augmented image's crop offsets and mirror.

    Args:
      augmented: The augmenting pipeline's output on :func:`pipeline_images`.

    Returns:
      record: ``augment/{label,top,left,flip}``, each ``(1, 2)``.

    """
    batch = {"media": augmented, "label": torch.arange(len(augmented))}
    decoded = reference.decode_batches(
        [[batch]],
        config=exp000().dataset,
        source=reference.synthetic_split()[0],
    )
    return {
        f"augment/{key.removeprefix('loader/')}": value
        for key, value in decoded.items()
    }


def recipe_record() -> dict[str, Tensor]:
    """Return ``exp000``'s rate in every epoch and its optimizer's settings.

    Returns:
      record: ``recipe/lr`` (300 rates, float64) and ``recipe/optimizer`` (as
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


def split_record(count: int) -> dict[str, Tensor]:
    """Cut a training split of ``count`` images into members and non-members.

    Args:
      count: Images in the split.

    Returns:
      record: ``split/<count>/member`` and ``split/<count>/nonmember``.

    """
    member, nonmember = member_split(count, seed=exp000().dataset.split_seed)
    return {f"split/{count}/member": member, f"split/{count}/nonmember": nonmember}


SETTINGS: Final[dict[str, tuple[str, Callable[[], MembershipTrainLoop]]]] = {
    "exp000": ("resnet20-cifar10", exp000),
    "exp005": ("vgg16bn-cifar100", exp005),
    "exp010": ("mobilenetv2-fmnist", exp010),
}
"""Each setting's pristine experiment, keyed by its name, with the setting's label."""


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
) -> dict[str, Tensor]:
    """Train the shallowest network on the synthetic split, then attack it.

    The run is the recipe shrunk to :data:`LOCKSTEP_EPOCHS` epochs of batches
    of :data:`LOCKSTEP_BATCH`, decayed at :data:`LOCKSTEP_MILESTONES`, over
    the seven-image synthetic split and its three members, from the task's
    seed.

    Args:
      name: A key of :data:`DEFENSES`.
      directory: Scratch directory the split is written to.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: Under ``lockstep/<name>/``, per update the loss, the logits and
        labels joined, the epoch the defense was told and the rate; then the
        attack's scores and accuracy (:func:`scored`), the published metrics
        as :data:`METRICS`, and where the generator stands.

    """
    config = _lockstep_config(name, directory)
    contexts: list[LossContext] = []
    trained: dict[str, list[Tensor]] = {"loss": [], "logits": [], "label": []}
    rates: list[float] = []
    metric = MembershipInference.Config().make()
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = config.step.make()
        data = config.dataset.make()
        step.loss = _ContextSpy(step.loss, contexts)
        for epoch in range(LOCKSTEP_EPOCHS):
            step.timer_epoch.global_count = epoch
            for batch in data.train_dataloader():
                result = step.train_step(**batch)
                group = step.optimizer.param_groups[0]
                rates.append(float(cast(float, group["lr"])))
                trained["loss"].append(result["loss"])
                trained["logits"].append(result["model"])
                trained["label"].append(batch["label"])
                if states is not None:
                    states.append(
                        {k: v.clone() for k, v in step.model.state_dict().items()},
                    )
        for batch in data.eval_dataloader():
            logits = step.call_eval(media=batch["media"])
            assert isinstance(logits, Tensor)
            metric.update(logits.clone(), **batch)
        metrics = metric.compute()
        draws = rng_fingerprint()
    key = f"lockstep/{name}"
    return scored(key, metric) | {
        f"{key}/metrics": torch.tensor(
            [metrics[metric_name] for metric_name in METRICS],
            dtype=torch.float64,
        ),
        f"{key}/loss": torch.stack(trained["loss"]),
        f"{key}/logits": joined(trained["logits"]),
        f"{key}/label": joined(trained["label"]),
        f"{key}/epoch": torch.tensor([c.epoch for c in contexts]),
        f"{key}/lr": torch.tensor(rates, dtype=torch.float64),
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
        defense_record()
        | attack_record()
        | augment_record(pipeline_record()["pipeline/cifar10/augment"])
        | recipe_record()
        | split_record(reference.LOADER_IMAGES)
        | network_record()
    )
    for name in DEFENSES:
        record |= lockstep_record(name, directory)
    return record


def _lockstep_config(name: str, directory: Path) -> MembershipTrainLoop:
    """Shrink a variant's ten-class experiment to a lockstep run over the synthetic split."""
    media, label = reference.synthetic_split()
    for split in ("train", "test"):
        torch.save({"media": media, "label": label}, directory / f"{split}.pt")
    config = DEFENSES[name][0]()
    config.dataset.base_dir = None
    config.dataset.working_dir = directory
    config.dataset.batch_size = config.dataset.eval_batch_size = LOCKSTEP_BATCH
    schedule = config.step.schedule
    assert isinstance(schedule, PartialConfig)
    schedule.milestones = list(LOCKSTEP_MILESTONES)
    config.step.train_budget_epochs = LOCKSTEP_EPOCHS
    config.step.parallelism = NoParallel.Config(device="cpu")
    reference.tiny_model(config.step.model)
    return config


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
