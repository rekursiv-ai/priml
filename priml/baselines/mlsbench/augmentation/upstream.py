"""Runs of the pristine ``cv-data-augmentation`` script, under the capture's pins.

The script and each baseline's edit are imported from a clone of the benchmark,
not vendored. Each variant's own ``get_dataloaders`` builds its pipelines, so
the editable ``build_train_transform`` and the fixed test pipeline both run as
written, on PIL images, drawing from the global generator the way a loader
worker does. Nothing runs on a GPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast
from unittest import mock

import functools

from PIL import Image
from torch.utils import data
from torchvision.transforms import autoaugment

import torch

from priml.baselines.mlsbench.pytorch_vision import reference, upstream
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from torch import Tensor, nn

    from priml.baselines.mlsbench.pytorch_vision.upstream import (
        Batches,
        Lockstep,
        MainTrace,
        Upstream,
    )

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__cv-data-augmentation/tests/meta/pristine"
    "/pytorch-vision/custom_augment.py",
)
"""The unedited training script, relative to the clone."""

EDITS: Final = Path("tasks/cv-data-augmentation/edits")
"""Directory of the task's baseline edits, relative to the clone."""

EDITED: Final = ("cutout", "randaugment", "trivialaugment")
"""The published baselines, each named as its edit file and as its port's variant."""

OPS: Final = {
    "Identity": "identity",
    "ShearX": "shear_x",
    "ShearY": "shear_y",
    "TranslateX": "translate_x",
    "TranslateY": "translate_y",
    "Rotate": "rotate",
    "Brightness": "brightness",
    "Color": "color",
    "Contrast": "contrast",
    "Sharpness": "sharpness",
    "Posterize": "posterize",
    "Solarize": "solarize",
    "AutoContrast": "autocontrast",
    "Equalize": "equalize",
}
"""Maps torchvision's name of each policy operation to its port in ``augment``."""

SIGNED: Final = (
    "shear_x",
    "shear_y",
    "translate_x",
    "translate_y",
    "rotate",
    "brightness",
    "color",
    "contrast",
    "sharpness",
)
"""The operations whose magnitude a coin negates."""


class Script(Protocol):
    """The part of the ``cv-data-augmentation`` script the lockstep uses."""

    def train_epoch(
        self,
        model: nn.Module,
        loader: Iterable[tuple[Tensor, Tensor]],
        criterion: Callable[[Tensor, Tensor], Tensor],
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[float, float]:
        """Train for one pass over ``loader``."""
        raise NotImplementedError


def variants(root: Path, scratch: Path) -> dict[str, Upstream]:
    """Import the pristine script and each baseline's edit of it.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts are written to.

    Returns:
      scripts: ``pad_crop_flip`` (the pristine script) and each of :data:`EDITED`.

    """
    pristine = upstream.load_script(root / PRISTINE)
    return {"pad_crop_flip": pristine} | {
        name: upstream.edited_script(
            root,
            name,
            scratch,
            pristine=PRISTINE,
            edits=EDITS,
        )
        for name in EDITED
    }


def pipelines(script: Upstream) -> tuple[_Pipeline, _Pipeline]:
    """Return the training and test pipelines the script's loaders apply per image.

    Args:
      script: A variant of the script.

    Returns:
      train: The training pipeline ``build_train_transform`` returned.
      test: The fixed test pipeline.

    """
    with mock.patch.object(script.torchvision.datasets, "CIFAR100", _Recorded):
        train, test, _ = script.get_dataloaders("cifar100", "unused", num_workers=0)
    return (
        cast(_Recorded, train.dataset).transform,
        cast(_Recorded, test.dataset).transform,
    )


def upstream_pipeline(
    pipeline: _Pipeline,
    images: Tensor,
    *,
    seed: int,
) -> Tensor:
    """Run a pipeline on each image in turn, from one seed of the global generator.

    Args:
      pipeline: A pipeline of :func:`pipelines`.
      images: ``(B, 3, H, W)`` uint8 pixels.
      seed: Where the global generator starts, as a loader worker's is seeded.

    Returns:
      media: ``(B, 3, H, W)`` float32 images, normalized.

    """
    with reference.seeded(seed):
        return torch.stack([pipeline(_pil(image)) for image in images])


@contextmanager
def op_census() -> Generator[dict[str, int]]:
    """Count the policy operations torchvision applies inside the block, by sign.

    Yields:
      counts: Maps ``"<op><sign>"``, keyed as the port's census, to calls.

    """
    counts: dict[str, int] = {}
    spy = _OpSpy(cast("_ApplyOp", vars(autoaugment)["_apply_op"]), counts)
    with mock.patch.object(autoaugment, "_apply_op", spy):
        yield counts


def upstream_batches(script: Upstream) -> tuple[Batches, Batches]:
    """Run the script's loaders with four forked workers, alternating the splits.

    Args:
      script: A variant of the script.

    Returns:
      passes: The training passes, then the test passes.

    """
    # The Cutout edit defines its transform class inside a function, which a
    # forkserver worker cannot unpickle; a forked worker inherits it. The start
    # method changes no draw: each worker's seed comes from the parent.
    forked = functools.partial(data.DataLoader, multiprocessing_context="fork")
    with mock.patch.object(data, "DataLoader", forked):
        return upstream.upstream_batches(script)


def upstream_lockstep(script: Upstream, name: str, trace: MainTrace) -> Lockstep:
    """Train the shallowest network for two updates on the script's augmentation.

    Update ``i`` trains on the variant's pipeline applied to two fixed images
    from seed ``SEED + i``, through one pass of the script's ``train_epoch``.

    Args:
      script: The script carrying the augmentation under test.
      name: Which variant, for the record's keys.
      trace: Source of the optimizer settings ``main`` used.

    Returns:
      lockstep: Per update the augmented batch, the loss, its logits and a
        probe's logits afterwards; and the model's whole state after each update.

    """
    train, _ = pipelines(script)
    media = [
        upstream_pipeline(train, reference.fixed_images(2), seed=reference.SEED + i)
        for i in range(reference.STEPS)
    ]
    task = cast(Script, cast(object, script))
    criterion = upstream.CriterionSpy()
    probes: list[Tensor] = []
    states: list[dict[str, Tensor]] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = script.ResNet(script.BasicBlock, [1, 1, 1], reference.CLASSES)
        script.initialize_weights(model)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=trace.settings["lr"],
            momentum=trace.settings["momentum"],
            weight_decay=trace.settings["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=reference.EPOCHS,
        )
        for index, batch in enumerate(media):
            loader = [(batch, reference.fixed_label(index))]
            task.train_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
            model.eval()
            with torch.no_grad():
                probes.append(
                    reference.logits_of(model, reference.fixed_media(reference.STEPS)),
                )
            states.append({k: v.clone() for k, v in model.state_dict().items()})
            scheduler.step()
    record = {
        f"lockstep/{name}/media": torch.stack(media),
        f"lockstep/{name}/loss": torch.stack(criterion.losses),
        f"lockstep/{name}/logits": torch.stack(criterion.logits),
        f"lockstep/{name}/probe": torch.stack(probes),
    }
    return record, states


type _Pipeline = Callable[[Image.Image], Tensor]


class _ApplyOp(Protocol):
    """torchvision's ``_apply_op``, which every policy calls once per operation."""

    def __call__(
        self,
        img: Image.Image,
        op_name: str,
        magnitude: float,
        interpolation: object,
        fill: object,
    ) -> Image.Image:
        """Return the image with the named operation applied."""
        raise NotImplementedError


class _OpSpy:
    """Calls ``_apply_op`` and tallies the operation by the sign of its magnitude."""

    def __init__(self, original: _ApplyOp, counts: dict[str, int]) -> None:
        self.original = original
        self.counts = counts

    def __call__(
        self,
        img: Image.Image,
        op_name: str,
        magnitude: float,
        interpolation: object,
        fill: object,
    ) -> Image.Image:
        """Count the call, then make it."""
        key = f"{OPS[op_name]}{'-' if magnitude < 0 else '+'}"
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.original(img, op_name, magnitude, interpolation, fill)


class _Recorded:
    """A dataset that is never read and keeps the pipeline the script gave it."""

    def __init__(
        self,
        *,
        root: str,
        train: bool,
        download: bool,
        transform: _Pipeline,
    ) -> None:
        del root, train, download
        self.transform = transform

    def __len__(self) -> int:
        # A shuffling loader refuses an empty dataset when it is built.
        return 1


def census_gaps(counts: dict[str, int]) -> list[str]:
    """Return every operation, and negated signed one, a census never saw.

    Args:
      counts: A census, keyed ``"<op><sign>"`` with the port's names.

    Returns:
      gaps: One key per operation and sign never reached.

    """
    wanted = [f"{op}+" for op in OPS.values()] + [f"{op}-" for op in SIGNED]
    return [key for key in wanted if key not in counts]


def _pil(image: Tensor) -> Image.Image:
    """Return a ``(3, H, W)`` uint8 tensor as the RGB image a dataset yields."""
    return Image.fromarray(image.permute(1, 2, 0).contiguous().numpy())
