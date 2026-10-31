"""Port-side runs the ``cv-data-augmentation`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script, each baseline's edit
and these runs on the same inputs under the same pins, compares them bit for
bit, and stores the script's values under ``testdata/``; ``reference_test.py``
replays these runs against them. The network, schedule and loss are the
``cv-classification-loss`` port's, already proven there; this task adds the
training augmentation.

The augmentations are integer and correctly rounded float32 arithmetic, so
they run outside :func:`~priml.testing.bfb.host_agnostic_numerics`, as the
reference's PIL pipelines do; only the network runs under it.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Final, cast
from unittest import mock

from torch import Tensor

import torch

from priml.baselines.mlsbench.dl.augmentation import augment
from priml.baselines.mlsbench.dl.augmentation.experiments import (
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
from priml.baselines.mlsbench.dl.augmentation.upstream import OPS
from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.data import normalize, resized
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping
    from pathlib import Path

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop
    from priml.baselines.mlsbench.dl.vision.upstream import Batches


AUGMENTS: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "pad_crop_flip": exp000,
    "cutout": exp001,
    "randaugment": exp002,
    "trivialaugment": exp003,
}
"""The experiment that carries each augmentation, keyed as the script's variants."""

SETTINGS: Final[dict[str, dict[str, Callable[[], VisionTrainLoop]]]] = {
    "resnet20-cifar10": {
        "pad_crop_flip": exp004,
        "cutout": exp005,
        "randaugment": exp006,
        "trivialaugment": exp007,
    },
    "mobilenetv2-fmnist": {
        "pad_crop_flip": exp008,
        "cutout": exp009,
        "randaugment": exp010,
        "trivialaugment": exp011,
    },
}
"""Per other setting of the task, the experiment that carries each augmentation."""

PIPELINE_IMAGES: Final = 160
"""Images the capture runs each pipeline on, enough to reach every operation."""

GOLDEN_IMAGES: Final = 16
"""The first of those images whose classified pipeline output the golden pins."""


def pipeline_images(count: int) -> Tensor:
    """Return ``count`` uint8 images of random pixels, the same on every host.

    Args:
      count: Images to build.

    Returns:
      images: ``(count, 3, 32, 32)`` pixels.

    """
    generator = torch.Generator().manual_seed(reference.SEED)
    return torch.randint(0, 256, (count, 3, 32, 32), generator=generator).to(
        torch.uint8,
    )


def source_images(setting: str, count: int) -> Tensor:
    """Return ``count`` uint8 images of random pixels as a setting's dataset ships them.

    Args:
      setting: A key of :data:`SETTINGS`.
      count: Images to build.

    Returns:
      images: ``(count, 3, 32, 32)`` CIFAR images, or ``(count, 1, 28, 28)``
        grayscale ones for Fashion-MNIST, before its ``Resize(32)``.

    """
    if reference.SETTINGS[setting].dataset != "fmnist":
        return pipeline_images(count)
    generator = torch.Generator().manual_seed(reference.SEED)
    return torch.randint(256, (count, 1, 28, 28), generator=generator).to(torch.uint8)


def prepared(setting: str, images: Tensor) -> Tensor:
    """Return :func:`source_images` as the setting's prepared cache holds them."""
    if reference.SETTINGS[setting].dataset != "fmnist":
        return images
    return resized(images, size=32)


def augmented(
    name: str,
    images: Tensor,
    *,
    seed: int,
    factory: Callable[[], VisionTrainLoop] | None = None,
) -> Tensor:
    """Run the augmentation of ``name``'s experiment on ``images`` from one stream.

    Args:
      name: A key of :data:`AUGMENTS`.
      images: ``(B, C, H, W)`` uint8 pixels, three channels or one.
      seed: Seed of the stream, as a loader worker's is seeded.
      factory: The experiment whose dataset runs; ``AUGMENTS[name]`` by
        default, a setting's fork of it otherwise.

    Returns:
      media: ``(B, 3, H, W)`` float32 images, normalized.

    """
    config = (factory or AUGMENTS[name])().dataset
    function = config.augment.make()
    return function(
        images,
        torch.Generator().manual_seed(seed),
        pad=config.translate_pad,
        mean=torch.tensor(config.mean),
        std=torch.tensor(config.std),
    )


def classified(media: Tensor) -> Tensor:
    """Classify ``media`` with the shallowest ``exp000`` network in evaluation mode.

    The logits depend on every pixel of the batch at a fraction of its bytes,
    so a golden of them pins an augmentation without storing its images.

    Args:
      media: ``(B, 3, H, W)`` float32 images.

    Returns:
      logits: ``(B, CLASSES)`` float32.

    """
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = reference.tiny_step(
            exp000,
            blocks=(1, 1, 1),
            classes=reference.CLASSES,
            epochs=reference.EPOCHS,
        ).model
        model.eval()
        return reference.model_logits(model, media).detach()


def pipeline_record(name: str) -> dict[str, Tensor]:
    """Return the classified augmentation of the golden images.

    Args:
      name: A key of :data:`AUGMENTS`.

    Returns:
      record: ``augment/<name>/logits``.

    """
    media = augmented(name, pipeline_images(GOLDEN_IMAGES), seed=reference.SEED)
    return {f"augment/{name}/logits": classified(media)}


def test_record() -> dict[str, Tensor]:
    """Return the classified test pipeline of the golden images: normalization only.

    Returns:
      record: ``augment/test/logits``.

    """
    config = exp000().dataset
    media = normalize(
        pipeline_images(GOLDEN_IMAGES),
        mean=torch.tensor(config.mean),
        std=torch.tensor(config.std),
    )
    return {"augment/test/logits": classified(media)}


def lockstep_images(setting: str = "") -> Tensor:
    """Return the two fixed images a lockstep augments, as the setting's dataset ships them.

    Args:
      setting: A key of :data:`SETTINGS`, or empty for ``exp000``'s chain.

    Returns:
      images: ``(2, 3, 32, 32)``, or ``(2, 1, 28, 28)`` for Fashion-MNIST.

    """
    if setting and reference.SETTINGS[setting].dataset == "fmnist":
        return reference.fixed_images(2, channels=1, side=28)
    return reference.fixed_images(2)


def lockstep_key(name: str, *, setting: str = "") -> str:
    """Return the prefix of a lockstep's record: ``exp000``'s chain, or a setting's."""
    return f"settings/{setting}/lockstep/{name}" if setting else f"lockstep/{name}"


def lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
    setting: str = "",
) -> dict[str, Tensor]:
    """Train the shallowest network for two updates on augmented fixed images.

    Update ``i`` augments two fixed images from a stream seeded ``SEED + i``.

    Args:
      name: A key of :data:`AUGMENTS`.
      states: When given, receives a copy of the model's whole state after
        each update.
      setting: A key of :data:`SETTINGS` to train that setting's fork, its
        network shrunk by ``tiny_model`` and at its class count; empty for
        ``exp000``'s chain.

    Returns:
      record: Per update the augmented batch, the loss, the logits it came
        from and the logits of a fixed probe in evaluation mode afterwards,
        under :func:`lockstep_key`.

    """
    factory = SETTINGS[setting][name] if setting else AUGMENTS[name]
    images = lockstep_images(setting)
    if setting:
        images = prepared(setting, images)
    media = [
        augmented(name, images, seed=reference.SEED + i, factory=factory)
        for i in range(reference.STEPS)
    ]
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    probes: list[Tensor] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = reference.tiny_step(
            factory,
            blocks=(1, 1, 1),
            classes=reference.SETTINGS[setting].classes
            if setting
            else reference.CLASSES,
            epochs=reference.EPOCHS,
        )
        for index, batch in enumerate(media):
            step.timer_epoch.global_count = index
            result = step.train_step(media=batch, label=reference.fixed_label(index))
            losses.append(result["loss"])
            logits.append(result["model"])
            probe = step.call_eval(media=reference.fixed_media(reference.STEPS))
            assert isinstance(probe, Tensor)
            probes.append(probe)
            if states is not None:
                states.append(
                    {k: v.clone() for k, v in step.model.state_dict().items()},
                )
    key = lockstep_key(name, setting=setting)
    return {
        f"{key}/media": torch.stack(media),
        f"{key}/loss": torch.stack(losses),
        f"{key}/logits": torch.stack(logits),
        f"{key}/probe": torch.stack(probes),
    }


def port_record() -> dict[str, Tensor]:
    """Run every port-side comparison the golden stores and return their values.

    Returns:
      record: Keyed as the stored golden is.

    """
    record = (
        reference.schedule_record(exp000)
        | reference.optimizer_record(exp000)
        | reference.draws_record()
        | test_record()
    )
    for name in AUGMENTS:
        record |= pipeline_record(name) | stored(lockstep_record(name))
    return record


def stored(record: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Drop a lockstep's augmented batches, which the golden pins by their logits.

    Two updates of two float32 32x32 images are 48 KiB per variant, past the
    golden ceiling; the training logits are a function of every pixel of them,
    and the capture compares the pixels themselves.

    Args:
      record: A lockstep record.

    Returns:
      record: Every entry but ``lockstep/<name>/media``.

    """
    return {k: v for k, v in record.items() if not k.endswith("/media")}


def setting_pipeline_record(setting: str, name: str) -> dict[str, Tensor]:
    """Return the classified augmentation of the golden images at a setting.

    Args:
      setting: A key of :data:`SETTINGS`.
      name: A key of :data:`AUGMENTS`.

    Returns:
      record: ``settings/<setting>/<name>/augment``.

    """
    media = augmented(
        name,
        prepared(setting, source_images(setting, GOLDEN_IMAGES)),
        seed=reference.SEED,
        factory=SETTINGS[setting][name],
    )
    return {f"settings/{setting}/{name}/augment": classified(media)}


def setting_loader_batches(
    setting: str,
    name: str,
    directory: Path,
) -> tuple[Batches, Batches]:
    """Run one variant's loaders at a setting over the setting's synthetic split.

    Args:
      setting: A key of :data:`SETTINGS`.
      name: A key of :data:`AUGMENTS`.
      directory: Scratch directory the split is written to.

    Returns:
      passes: Per training pass the batches in order, then per evaluation pass.

    """
    return reference.dataset_batches(
        SETTINGS[setting][name]().dataset,
        directory,
        split=reference.SETTINGS[setting].split(),
    )


def settings_record(directory: Path) -> dict[str, Tensor]:
    """Run every other setting's comparisons the golden stores and return their values.

    Args:
      directory: Scratch directory the loader runs write their splits to.

    Returns:
      record: The shared per-variant networks and pristine loaders, and each
        variant's classified pipeline; keyed as the settings golden is.

    """
    record = reference.task_settings_record(SETTINGS, directory)
    for setting, variants in SETTINGS.items():
        for name in variants:
            record |= setting_pipeline_record(setting, name)
            record |= stored(lockstep_record(name, setting=setting))
    return record


def loader_batches(name: str, directory: Path) -> tuple[Batches, Batches]:
    """Run ``name``'s loaders over the synthetic split, alternating the splits.

    Args:
      name: A key of :data:`AUGMENTS`.
      directory: Scratch directory the split is written to.

    Returns:
      passes: Per training pass the batches in order, then per evaluation pass
        the batches in order.

    """
    return reference.dataset_batches(
        AUGMENTS[name]().dataset,
        directory,
        split=reference.synthetic_split(),
    )


@contextmanager
def op_census() -> Generator[dict[str, int]]:
    """Count the port's policy operations applied inside the block, by sign.

    The policies look their operations up in the spaces when called, so the
    spaces are rebuilt from the patched names.

    Yields:
      counts: Maps ``"<op><sign>"``, the sign that of the magnitude, to calls.

    """
    counts: dict[str, int] = {}
    with ExitStack() as stack:
        for op in OPS.values():
            original = cast("augment.Op", getattr(augment, op))
            stack.enter_context(
                mock.patch.object(augment, op, _Counted(op, original, counts)),
            )
        yield counts


class _Counted:
    """Calls an operation and tallies it by the sign of its magnitude."""

    def __init__(self, name: str, original: augment.Op, counts: dict[str, int]) -> None:
        self.name = name
        self.original = original
        self.counts = counts

    def __call__(self, image: Tensor, magnitude: float) -> Tensor:
        """Count the call, then make it."""
        key = f"{self.name}{'-' if magnitude < 0 else '+'}"
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.original(image, magnitude)
