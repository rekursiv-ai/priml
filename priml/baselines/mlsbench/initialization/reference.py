"""Port-side runs the ``dl-weight-initialization`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script, each baseline's edit
and these runs on the same inputs under the same pins, compares them bit for
bit, and stores the script's values under ``testdata/``; ``reference_test.py``
replays these runs against them. The data, loss and schedule are the
``cv-classification-loss`` port's, already proven there; this task changes
only how the network is initialized.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, cast

from torch import Tensor
from treekle import from_plain

import torch

from priml.baselines.mlsbench.initialization.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.mlsbench.pytorch_vision import reference
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from torch import nn

    from priml.baselines.mlsbench.pytorch_vision.experiments import VisionTrainLoop


INITIALIZATIONS: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "normal": exp000,
    "kaiming_normal": exp001,
    "fixup": exp002,
    "orthogonal": exp003,
}
"""The experiment that carries each initialization, keyed as the script's variants."""


def tiny_model(name: str) -> nn.Module:
    """Build the shallowest network at the reference's widths, initialized as ``name``.

    Args:
      name: A key of :data:`INITIALIZATIONS`.

    Returns:
      model: One block per stage, :data:`reference.CLASSES` logits.

    """
    return reference.tiny_step(
        INITIALIZATIONS[name],
        blocks=(1, 1, 1),
        classes=reference.CLASSES,
        epochs=reference.EPOCHS,
    ).model


def recipe_record() -> dict[str, Tensor]:
    """Return the 200 learning rates and the optimizer settings ``exp000`` builds.

    Returns:
      record: ``schedule/lr`` and ``optimizer/settings``, as float64.

    """
    step = reference.tiny_step(exp000, blocks=(1, 1, 1), classes=3, epochs=200)
    group = cast(Mapping[str, object], step.optimizer.param_groups[0])
    base_lr = from_plain(group["lr"], float)
    rates = [
        step.schedule(epoch, base_lr=base_lr, total_epochs=200) for epoch in range(200)
    ]
    values = [float(cast(float, group[name])) for name in reference.HYPERPARAMETERS]
    return {
        "schedule/lr": torch.tensor(rates, dtype=torch.float64),
        "optimizer/settings": torch.tensor(values, dtype=torch.float64),
    }


def draws_record() -> dict[str, Tensor]:
    """Return where the generator stands after building each full network.

    Returns:
      record: ``draws/<name>``, the next eight draws, per initialization.

    """
    record: dict[str, Tensor] = {}
    for name, factory in INITIALIZATIONS.items():
        with reference.seeded(reference.SEED), host_agnostic_numerics():
            _ = reference.full_step(factory)
            record[f"draws/{name}"] = rng_fingerprint()
    return record


def network_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Initialize the shallowest network as ``name`` and run it.

    Args:
      name: A key of :data:`INITIALIZATIONS`.
      states: When given, receives a copy of the network's state after
        initialization.

    Returns:
      record: Where the generator stands afterwards, and the logits of a
        training-mode and then an evaluation-mode forward.

    """
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = tiny_model(name)
        if states is not None:
            states.append({k: v.clone() for k, v in model.state_dict().items()})
        draws = rng_fingerprint()
        media = reference.fixed_media(reference.STEPS)
        model.train()
        train = reference.logits_of(model, media)
        model.eval()
        evaluated = reference.logits_of(model, media)
    return {
        f"network/{name}/draws": draws,
        f"network/{name}/train": train.detach(),
        f"network/{name}/eval": evaluated.detach(),
    }


def lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train the shallowest network for two updates, one per epoch.

    Args:
      name: A key of :data:`INITIALIZATIONS`.
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
            INITIALIZATIONS[name],
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


def port_record() -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Returns:
      record: The union of the records above, keyed as the stored golden is.

    """
    record = recipe_record() | draws_record()
    for name in INITIALIZATIONS:
        record |= network_record(name) | lockstep_record(name)
    return record
