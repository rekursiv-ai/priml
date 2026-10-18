"""Port-side runs the ``dl-activation-function`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script and these runs on the
same inputs under the same pins, compares them bit for bit, and stores the
script's values under ``testdata/``; ``reference_test.py`` replays these runs
against them. The step and loader are the ``cv-classification-loss`` port's,
already proven there; this task adds the activation and the
``resnet20-cifar10`` setting's depth, classes and channel statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from configgle import Makeable
from torch import Tensor

import torch

from priml.baselines.mlsbench.dl.activation.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop
    from priml.baselines.mlsbench.dl.vision.upstream import Batches


ACTIVATIONS: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "relu": exp000,
    "gelu": exp001,
    "silu": exp002,
    "mish": exp003,
}
"""The experiment that carries each activation, keyed as the script's variants."""


def fixed_inputs() -> Tensor:
    """Return ``(3, 67)`` float32 quarters from -25 to 25.

    The range crosses softplus's threshold of 20 on both sides and holds an
    exact zero, where ReLU's gradient is decided by convention.

    Returns:
      x: Every multiple of one quarter in ``[-25, 25]``.

    """
    return (torch.arange(-100, 101) / 4).reshape(3, 67)


def activation_record() -> dict[str, Tensor]:
    """Apply every experiment's activation to :func:`fixed_inputs`, forward and back.

    Returns:
      record: Each activation's values and its gradient at every input.

    """
    record: dict[str, Tensor] = {}
    for name, factory in ACTIVATIONS.items():
        model = factory().step.model
        assert isinstance(model, ResNet.Config)
        activation = model.activation
        if isinstance(activation, Makeable):
            raise TypeError("Expected not isinstance(activation, Makeable).")
        leaf = fixed_inputs().requires_grad_()
        with host_agnostic_numerics():
            value = activation(leaf)
            value.sum().backward()
        assert isinstance(leaf.grad, Tensor)
        record[f"activation/{name}/value"] = value.detach()
        record[f"activation/{name}/grad"] = leaf.grad
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
    """Train the shallowest network for two updates, one per epoch.

    Args:
      name: A key of :data:`ACTIVATIONS`.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: Per update the loss, the logits it came from and the logits of
        a fixed probe in evaluation mode afterwards.

    """
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    probes: list[Tensor] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = reference.tiny_step(
            ACTIVATIONS[name],
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
            probe = step.call_eval(media=reference.fixed_media(reference.STEPS))
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
    """Run ``exp000``'s loaders over the synthetic split, alternating the splits.

    Args:
      directory: Scratch directory the split is written to.

    Returns:
      passes: Per training pass the batches in order, then per evaluation pass
        the batches in order, normalized with CIFAR-10's statistics.

    """
    media, label = reference.synthetic_split()
    for split in ("train", "test"):
        torch.save({"media": media, "label": label}, directory / f"{split}.pt")
    config = exp000().dataset
    config.working_dir = directory
    config.batch_size = config.eval_batch_size = reference.LOADER_BATCH
    train: Batches = []
    evaluation: Batches = []
    with reference.seeded(reference.SEED):
        data = config.make()
        for _ in range(reference.LOADER_PASSES):
            train.append(list(data.train_dataloader()))
            evaluation.append(list(data.eval_dataloader()))
    return train, evaluation


def port_record() -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Returns:
      record: The union of the records above, keyed as the stored golden is.

    """
    record = (
        activation_record()
        | reference.schedule_record(exp000)
        | reference.optimizer_record(exp000)
        | draws_record()
        | reference.best_record()
    )
    for name in ACTIVATIONS:
        record |= lockstep_record(name)
    return record
