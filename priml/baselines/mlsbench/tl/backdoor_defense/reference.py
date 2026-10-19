"""Inputs and port-side runs the ``security-backdoor-defense`` capture and its test share.

``scripts/capture_reference.py`` runs the driver and these runs on the same
inputs under the same pins, compares them bit for bit, and stores the driver's
values under ``testdata/``; ``reference_test.py`` replays these runs against
them. The step, schedule, optimizer and networks are the vision port's, proven
there; this task adds the poisoning, the triggers, the defenses, the filter,
the backdoor's measurement and its three settings.

Every input is built from integers and powers of two, so it is exact in float32
and the same on every host; every run that computes in torch does so under
:func:`~priml.testing.bfb.host_agnostic_numerics`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from torch import Tensor, nn
from treekle import from_plain

import numpy as np
import torch

from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.data import normalize
from priml.baselines.mlsbench.dl.vision.draws import recorded_draws
from priml.baselines.mlsbench.tl.backdoor_defense.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
    exp004,
    exp005,
    exp006,
)
from priml.baselines.mlsbench.tl.backdoor_defense.harness import (
    attack_success_rate,
    clean_accuracy,
    collect_features,
    filter_poison,
)
from priml.baselines.mlsbench.tl.backdoor_defense.poison import (
    PoisonedData,
    python_generator,
    select_poison_indices,
)
from priml.baselines.mlsbench.tl.backdoor_defense.train_loop import (
    BackdoorTrainLoop,
)
from priml.baselines.mlsbench.tl.backdoor_defense.triggers import (
    TriggerFn,
    badnets,
    blend,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from priml.baselines.mlsbench.dl.vision.upstream import Batches


@dataclass(frozen=True, slots=True, kw_only=True)
class Setting:
    """One of the task's settings: the driver's flags and the port's experiment."""

    vision: str
    """The vision port's key for the setting's network and dataset."""

    arch: str
    """The driver's ``--arch``."""

    dataset: str
    """The driver's ``--dataset``."""

    classes: int
    """Classes of the dataset."""

    trigger: str
    """The driver's ``--trigger``."""

    fraction: float
    """The driver's ``--poison-fraction``."""

    target: int = 0
    """The driver's ``--target-class``, at its default."""

    factory: Callable[[], BackdoorTrainLoop.Config]
    """The port's experiment at the setting."""


SETTINGS: Final = {
    "resnet20-cifar10-badnets": Setting(
        vision="resnet20-cifar10",
        arch="resnet20",
        dataset="cifar10",
        classes=10,
        trigger="badnets",
        fraction=0.05,
        factory=exp000,
    ),
    "vgg16bn-cifar100-blend": Setting(
        vision="vgg16bn-cifar100",
        arch="vgg16bn",
        dataset="cifar100",
        classes=100,
        trigger="blend",
        fraction=0.01,
        factory=exp005,
    ),
    "mobilenetv2-fmnist-badnets": Setting(
        vision="mobilenetv2-fmnist",
        arch="mobilenetv2",
        dataset="fmnist",
        classes=10,
        trigger="badnets",
        fraction=0.08,
        factory=exp006,
    ),
}
"""The task's settings, keyed by its own labels."""

FIRST: Final = "resnet20-cifar10-badnets"
"""The setting ``exp000`` and every baseline run at."""

VARIANTS: Final[dict[str, Callable[[], BackdoorTrainLoop.Config]]] = {
    "pristine": exp000,
    "confidence_filter": exp001,
    "spectral_signature": exp002,
    "activation_clustering": exp003,
    "zscore_outlier": exp004,
}
"""The experiment carrying each defense, keyed as the driver's variants."""

TRIGGERS: Final[dict[str, TriggerFn]] = {"badnets": badnets, "blend": blend}
"""The port's trigger for each ``--trigger``."""

STATS: Final = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
"""CIFAR-10's channel means and deviations, which the triggers are run with."""

POISON_IMAGES: Final = 43
"""Labels :func:`poison_labels` holds; each setting's fraction poisons another count."""

DEFENSE_FRACTION: Final = 0.25
"""Poisoned fraction each defense is told on :func:`defense_inputs`."""

DEFENSE_SUBSET: Final = 9
"""Images of the second scoring, shorter than the fit split, which takes predictions."""

LOADER_FRACTION: Final = 0.3
"""Poisoned fraction of the loader runs: two of the seven synthetic images."""

MAIN_IMAGES: Final = 40
"""Images of each split ``main`` is run on: two poisoned at 5%, three at 8%."""

EPOCHS_MAIN: Final = 20
"""Epochs ``main`` is run for; it evaluates after epochs 1, 10 and 20, as the port does."""


def python_fingerprint(state: tuple[object, ...]) -> Tensor:
    """Return the next eight 31-bit draws of a Python generator in ``state``, unadvanced."""
    rng = python_generator(0)
    rng.setstate(state)
    return torch.tensor([rng.getrandbits(31) for _ in range(8)])


def numpy_fingerprint(rng: np.random.RandomState) -> Tensor:
    """Return the next eight draws of a NumPy ``RandomState``, unadvanced."""
    copy = np.random.RandomState()
    copy.set_state(rng.get_state())
    return torch.from_numpy(copy.randint(2**31 - 1, size=8).astype(np.int64))


def poison_labels() -> list[int]:
    """Return :data:`POISON_IMAGES` labels cycling over ten classes with a stride of seven."""
    return [index * 7 % 10 for index in range(POISON_IMAGES)]


def trigger_media() -> Tensor:
    """Return ``(2, 3, 5, 6)`` normalized images whose pixels leave ``[0, 1]`` both ways.

    Returns:
      media: Float32 multiples of one eighth in ``[-3, 3]``.

    """
    return ((torch.arange(2 * 3 * 5 * 6) * 11 % 49 - 24) / 8).reshape(2, 3, 5, 6)


def defense_inputs() -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.int64],
]:
    """Return 13 images' features, logits and labels for the defenses.

    Class 0 has six images, class 1 four, class 2 two and class 3 one, so the
    small-class branches are reached; classes 0 and 1 each have one outlier.
    Each class's features vary along one axis, so its top singular vector is
    that axis and its norms are exact: an SVD or norm of features that span
    more directions rounds differently under each OpenBLAS kernel, as the
    driver's own does. Features are eighths, logits multiples of 1/256 whose
    rows' top probabilities differ.

    Returns:
      features: ``(13, 3)`` float32.
      logits: ``(13, 4)`` float32.
      labels: ``(13,)`` int64.

    """
    features = (
        np.array(
            [
                [6, 16, -8],
                [-16, 7, 24],
                [7, 16, -8],
                [24, -8, 15],
                [9, 16, -8],
                [-16, 9, 24],
                [17, 16, -8],
                [0, 24, 8],
                [-16, 15, 24],
                [10, 16, -8],
                [24, -8, 18],
                [-16, 6, 24],
                [5, 16, -8],
            ],
        )
        / 8
    )
    steps = np.arange(13 * 4)
    # A ramp linear in the step would shift rows by a constant, which softmax ignores.
    logits = ((steps * 23 % 53 - 26) / 8 + steps**2 / 256).reshape(13, 4)
    labels = np.array([0, 1, 0, 2, 0, 1, 0, 3, 1, 0, 2, 1, 0], dtype=np.int64)
    return features.astype(np.float32), logits.astype(np.float32), labels


def main_split(dataset: str) -> tuple[Tensor, Tensor]:
    """Return :data:`MAIN_IMAGES` distinct images, as the dataset ships them, and labels.

    Args:
      dataset: The driver's ``--dataset``; ``fmnist`` gives 28x28 grayscale
        images, the others 32x32 color ones.

    Returns:
      media: uint8 images.
      label: :func:`main_labels`.

    """
    channels, side = (1, 28) if dataset == "fmnist" else (3, 32)
    position = torch.arange(channels * side * side)
    media = torch.stack(
        [(position * 37 + 101 * index) % 251 for index in range(MAIN_IMAGES)],
    ).reshape(MAIN_IMAGES, channels, side, side)
    return media.to(torch.uint8), torch.tensor(main_labels())


def main_labels() -> list[int]:
    """Return the labels of :func:`main_split`: each image's index modulo ten."""
    return [index % 10 for index in range(MAIN_IMAGES)]


def synthetic(dataset: str) -> tuple[Tensor, Tensor]:
    """Return the vision port's synthetic split for ``dataset``, as ``prepare`` caches it."""
    if dataset == "fmnist":
        return reference.prepared_gray_split()
    return reference.synthetic_split()


def setting_data(
    setting: str,
    directory: Path,
    *,
    split: tuple[Tensor, Tensor],
    fraction: float,
) -> PoisonedData:
    """Build a setting's poisoned dataset over ``split``, written to ``directory``.

    Args:
      setting: A key of :data:`SETTINGS`.
      directory: Scratch directory ``split`` is written to, as both splits.
      split: Prepared uint8 images and their labels.
      fraction: Poisoned fraction.

    Returns:
      data: The dataset, its batches :data:`vision.LOADER_BATCH` images.

    """
    media, label = split
    for name in ("train", "test"):
        torch.save({"media": media, "label": label}, directory / f"{name}.pt")
    config = SETTINGS[setting].factory().dataset
    config.base_dir = None
    config.working_dir = directory
    config.batch_size = config.eval_batch_size = reference.LOADER_BATCH
    config.poison_fraction = fraction
    data = config.make()
    assert isinstance(data, PoisonedData)
    return data


def poison_record() -> dict[str, Tensor]:
    """Pick each setting's poisoned positions of :func:`poison_labels`.

    Returns:
      record: ``poison/<setting>/indices`` and where Python's generator stands
        afterwards, ``poison/<setting>/python``.

    """
    record: dict[str, Tensor] = {}
    for name, spec in SETTINGS.items():
        rng = python_generator(reference.SEED)
        indices = select_poison_indices(
            poison_labels(),
            poison_fraction=spec.fraction,
            target_class=spec.target,
            rng=rng,
        )
        record[f"poison/{name}/indices"] = torch.tensor(indices)
        record[f"poison/{name}/python"] = python_fingerprint(rng.getstate())
    return record


def trigger_record() -> dict[str, Tensor]:
    """Stamp each trigger into :func:`trigger_media`.

    Returns:
      record: ``trigger/<name>``, the triggered images.

    """
    mean, std = torch.tensor(STATS[0]), torch.tensor(STATS[1])
    return {
        f"trigger/{name}": trigger(trigger_media(), mean=mean, std=std)
        for name, trigger in TRIGGERS.items()
    }


def defense_record() -> dict[str, Tensor]:
    """Fit every variant's defense on :func:`defense_inputs`, score them, then a prefix.

    Returns:
      record: ``defense/<variant>/scores`` and, but for activation clustering,
        which can only score the split it was fit on,
        ``defense/<variant>/subset``.

    """
    features, logits, labels = defense_inputs()
    record: dict[str, Tensor] = {}
    for name, factory in VARIANTS.items():
        defense = factory().defense.make()
        with host_agnostic_numerics():
            defense.fit(features, labels, DEFENSE_FRACTION)
            record[f"defense/{name}/scores"] = _float64(
                defense.score_samples(features, logits),
            )
            if name != "activation_clustering":
                record[f"defense/{name}/subset"] = _float64(
                    defense.score_samples(
                        features[:DEFENSE_SUBSET],
                        logits[:DEFENSE_SUBSET],
                    ),
                )
    return record


def tiny_record(setting: str, directory: Path) -> dict[str, Tensor]:
    """Build a setting's shallowest network and run every measurement on it.

    Args:
      setting: A key of :data:`SETTINGS`.
      directory: Scratch directory the synthetic split is written to.

    Returns:
      record: Under ``tiny/<setting>/``: the generator after the build; the
        features, logits and labels of two fixed batches; the clean accuracy
        and the attack success rate on the synthetic test split.

    """
    spec = SETTINGS[setting]
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = reference.tiny_step(
            spec.factory,
            blocks=(1, 1, 1),
            classes=spec.classes,
            epochs=reference.EPOCHS,
        ).model
        draws = rng_fingerprint()
    data = setting_data(
        setting,
        directory,
        split=synthetic(spec.dataset),
        fraction=LOADER_FRACTION,
    )
    key = f"tiny/{setting}"
    return {f"{key}/draws": draws} | measurements(model, data, key=key)


def measurements(
    model: nn.Module,
    data: PoisonedData,
    *,
    key: str,
) -> dict[str, Tensor]:
    """Run the harness's feature extraction, clean accuracy and ASR on a network.

    Args:
      model: A network of the setting ``data`` belongs to.
      data: Its test split is measured; :func:`fixed_batches` are featurized.
      key: Prefix of the record's keys.

    Returns:
      record: ``features``, ``logits``, ``labels``, ``clean_acc`` and ``asr``.

    """
    config = data.config
    device = torch.device("cpu")
    with host_agnostic_numerics():
        features, logits, labels = collect_features(
            model,
            fixed_batches(),
            device=device,
        )
        accuracy = clean_accuracy(model, data.eval_dataloader(), device=device)
        asr = attack_success_rate(
            model,
            data.eval_media,
            data.eval_label,
            trigger=config.trigger,
            target_class=config.target_class,
            mean=torch.tensor(config.mean),
            std=torch.tensor(config.std),
            batch_size=config.batch_size,
            device=device,
        )
    return {
        f"{key}/features": torch.from_numpy(features),
        f"{key}/logits": torch.from_numpy(logits),
        f"{key}/labels": torch.from_numpy(labels),
        f"{key}/clean_acc": torch.tensor(accuracy, dtype=torch.float64),
        f"{key}/asr": torch.tensor(asr, dtype=torch.float64),
    }


def fixed_batches() -> list[dict[str, Tensor]]:
    """Return the vision port's first fixed batch, as a ``media``/``label`` dict.

    One batch: MobileNetV2's 1,280 features per image fill the golden.

    Returns:
      batches: The one batch, in a list as the harness iterates batches.

    """
    return [{"media": reference.fixed_media(0), "label": reference.fixed_label(0)}]


def lockstep_record(
    name: str = "pristine",
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train the shallowest ResNet for two updates, one per epoch.

    Args:
      name: A key of :data:`VARIANTS`; the defense does not touch training.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: Per update the loss, the logits it came from and, in
        evaluation mode afterwards, the logits of a fixed probe.

    """
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    probes: list[Tensor] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = reference.tiny_step(
            VARIANTS[name],
            blocks=(1, 1, 1),
            classes=reference.CLASSES,
            epochs=reference.EPOCHS,
        )
        for index in range(reference.STEPS):
            step.timer_epoch.global_count = index
            result = step.train_step(
                media=reference.fixed_media(index),
                label=reference.fixed_label(index),
            )
            losses.append(result["loss"])
            logits.append(result["model"])
            step.model.eval()
            with torch.no_grad():
                probes.append(
                    reference.logits_of(
                        step.model,
                        reference.fixed_media(reference.STEPS),
                    ),
                )
            if states is not None:
                states.append(
                    {k: v.clone() for k, v in step.model.state_dict().items()},
                )
    return {
        "lockstep/loss": torch.stack(losses),
        "lockstep/logits": torch.stack(logits),
        "lockstep/probe": torch.stack(probes),
    }


def main_record(setting: str, *, epochs: int = EPOCHS_MAIN) -> dict[str, Tensor]:
    """Return what a setting's experiment decides around training, for :func:`main_split`.

    The poisoned images are those ``PoisonedData`` picks from the experiment's
    dataset config, by the same call with the same seeded generator; the
    evaluation epochs, schedule and optimizer come from its loop and step.

    Args:
      setting: A key of :data:`SETTINGS`.
      epochs: The run's length; the driver's ``main`` is captured at
        :data:`EPOCHS_MAIN`.

    Returns:
      record: Under ``main/<setting>/``: which images are poisoned, the epochs
        after which the test split is evaluated, every epoch's rate, the
        optimizer's settings, and where Python's and NumPy's generators stand
        after the poisoning, which is all they serve.

    """
    spec = SETTINGS[setting]
    config = spec.factory()
    python = python_generator(config.dataset.poison_seed)
    is_poison = torch.zeros(MAIN_IMAGES, dtype=torch.uint8)
    is_poison[
        select_poison_indices(
            main_labels(),
            poison_fraction=config.dataset.poison_fraction,
            target_class=config.dataset.target_class,
            rng=python,
        )
    ] = 1
    evaluate = config.eval_epochs.make()
    schedule = config.step.schedule.make()
    settings = reference.optimizer_record(spec.factory)["optimizer/settings"]
    base_lr = float(settings[reference.HYPERPARAMETERS.index("lr")])
    key = f"main/{setting}"
    return {
        f"{key}/is_poison": is_poison,
        f"{key}/evaluated": torch.tensor(
            [
                epoch + 1
                for epoch in range(epochs)
                if evaluate(epoch + 1, total_epochs=epochs)
            ],
        ),
        f"{key}/rates": torch.tensor(
            [
                schedule(epoch, base_lr=base_lr, total_epochs=epochs)
                for epoch in range(epochs)
            ],
            dtype=torch.float64,
        ),
        f"{key}/settings": settings,
        f"{key}/python": python_fingerprint(python.getstate()),
        # The driver seeds NumPy's legacy generator and nothing draws from it.
        f"{key}/numpy": numpy_fingerprint(np.random.RandomState(reference.SEED)),
    }


def filter_record(
    inputs: Mapping[str, Tensor],
    *,
    setting: str = FIRST,
) -> dict[str, Tensor]:
    """Filter the features ``main`` gave its defense with every variant's defense.

    Args:
      inputs: ``main/<setting>/{features,logits,labels,is_poison}``, as the
        driver's ``main`` produced them.
      setting: A key of :data:`SETTINGS`.

    Returns:
      record: ``filter/<variant>/{scores,kept,recall}``.

    """
    key = f"main/{setting}"
    record: dict[str, Tensor] = {}
    for name, factory in VARIANTS.items():
        with host_agnostic_numerics():
            filtered = filter_poison(
                factory().defense.make(),
                features=inputs[f"{key}/features"].numpy(),
                logits=inputs[f"{key}/logits"].numpy(),
                labels=inputs[f"{key}/labels"].numpy(),
                is_poison=inputs[f"{key}/is_poison"].bool().numpy(),
                poison_fraction=SETTINGS[setting].fraction,
                multiplier=factory().filter_multiplier,
            )
        record[f"filter/{name}/scores"] = torch.from_numpy(filtered.scores)
        record[f"filter/{name}/kept"] = torch.from_numpy(filtered.kept)
        # At the four decimals the driver prints, the only form it gives.
        record[f"filter/{name}/recall"] = torch.tensor(
            float(f"{filtered.poison_recall:.4f}"),
            dtype=torch.float64,
        )
    return record


def loader_batches(setting: str, directory: Path) -> tuple[Batches, Batches]:
    """Run a setting's poisoned training loader twice, then its analysis loader once.

    Args:
      setting: A key of :data:`SETTINGS`.
      directory: Scratch directory the synthetic split is written to.

    Returns:
      passes: The two training passes, then the analysis pass.

    """
    data = setting_data(
        setting,
        directory,
        split=synthetic(SETTINGS[setting].dataset),
        fraction=LOADER_FRACTION,
    )
    with reference.seeded(reference.SEED):
        train = [list(data.train_dataloader()) for _ in range(2)]
        analysis = [list(data.analysis_dataloader())]
    return train, analysis


def decode_record(setting: str, passes: tuple[Batches, Batches]) -> dict[str, Tensor]:
    """Recover each loaded image's source, crop offsets and mirror, trigger or not.

    Args:
      setting: A key of :data:`SETTINGS`.
      passes: The training passes and the analysis pass, from either side.

    Returns:
      record: Under ``loader/<setting>/``: ``label``, ``source``, ``top``,
        ``left`` and ``flip``, each ``(passes, images)``.

    Raises:
      ValueError: An image is no crop of any source, triggered or not.

    """
    spec = SETTINGS[setting]
    config = spec.factory().dataset
    candidates, names = _candidates(
        synthetic(spec.dataset)[0],
        pad=config.translate_pad,
        mean=torch.tensor(config.mean),
        std=torch.tensor(config.std),
        trigger=config.trigger,
    )
    found: dict[str, list[list[int]]] = {
        key: [] for key in ("label", "source", "top", "left", "flip")
    }
    # The centre column tells most candidates apart; only its matches are compared whole.
    centre = candidates[..., candidates.shape[-1] // 2]
    for batches in (*passes[0], *passes[1]):
        for values in found.values():
            values.append([])
        for batch in batches:
            labels = from_plain(batch["label"].tolist(), list[int])
            for image, label in zip(batch["media"], labels, strict=True):
                near = (
                    (centre == image[..., image.shape[-1] // 2]).flatten(1).all(dim=1)
                )
                (rows,) = near.nonzero(as_tuple=True)
                whole = (candidates[rows] == image).flatten(1).all(dim=1)
                hits = rows[whole]
                if not len(hits):
                    raise ValueError(
                        "The image is no crop of any source image, triggered or not.",
                    )
                decoded = names[int(hits[0])]
                for key, value in zip(found, (label, *decoded), strict=True):
                    found[key][-1].append(value)
    return {
        f"loader/{setting}/{key}": torch.tensor(values) for key, values in found.items()
    }


def port_record(directory: Path) -> dict[str, Tensor]:
    """Run every port-side comparison whose inputs the port builds itself.

    Args:
      directory: Scratch directory the splits are written to.

    Returns:
      record: Keyed as the stored golden is; :func:`filter_record` is apart,
        since it reads the driver's own features.

    """
    record = poison_record() | trigger_record() | defense_record() | lockstep_record()
    for setting, spec in SETTINGS.items():
        record |= tiny_record(setting, directory)
        record |= main_record(setting)
        record |= decode_record(setting, loader_batches(setting, directory))
        record |= reference.setting_record(spec.vision, spec.factory, key=setting)
    return record


def draw_sites(directory: Path) -> dict[str, int]:
    """Count the torch draws of building the shallowest ResNet and of a loader run.

    Args:
      directory: Scratch directory the split is written to.

    Returns:
      counts: Maps ``"<api>@<file>:<function>"`` to the number of calls.

    """
    with recorded_draws() as counts:
        _ = tiny_record(FIRST, directory)
        _ = loader_batches(FIRST, directory)
    return counts


def optimizer_group(groups: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    """Return an optimizer's first parameter group, each value for the caller to narrow."""
    return groups[0]


def _candidates(
    source: Tensor,
    *,
    pad: int,
    mean: Tensor,
    std: Tensor,
    trigger: TriggerFn,
) -> tuple[Tensor, list[tuple[int, int, int, int]]]:
    """Return every crop and mirror of every source, normalized, then each triggered."""
    size = source.shape[-1]
    padded = torch.nn.functional.pad(source, (pad,) * 4)
    crops: list[Tensor] = []
    names: list[tuple[int, int, int, int]] = []
    for index in range(len(source)):
        for top in range(2 * pad + 1):
            for left in range(2 * pad + 1):
                crop = padded[index, :, top : top + size, left : left + size]
                for flip in (0, 1):
                    crops.append(crop.flip(-1) if flip else crop)
                    names.append((index, top, left, flip))
    clean = normalize(torch.stack(crops), mean=mean, std=std)
    return torch.cat([clean, trigger(clean, mean=mean, std=std)]), names + names


def _float64(scores: NDArray[np.floating]) -> Tensor:
    """Return scores as the driver takes them: a float64 tensor."""
    return torch.from_numpy(np.asarray(scores, dtype=np.float64))
