"""Port-side runs the ``dl-lr-schedule`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script and these runs on the
same inputs under the same pins, compares them bit for bit, and stores the
script's values under ``testdata/``; ``reference_test.py`` replays these runs
against them. The network, data and loss are the ``cv-classification-loss``
port's, already proven there; this task adds only the schedule and the
optimizer the script builds without a scheduler.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, cast

from torch import Tensor
from treekle import from_plain

import torch

from priml.baselines.mlsbench.dl.lr_schedule.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.mlsbench.dl.lr_schedule.upstream import SETTINGS
from priml.baselines.mlsbench.dl.vision import reference
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


SCHEDULES: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "constant": exp000,
    "cosine": exp001,
    "warmup_cosine": exp002,
    "one_cycle": exp003,
}
"""The experiment that carries each schedule, keyed as the script's variants."""


def schedule_record() -> dict[str, Tensor]:
    """Return each schedule's rate for all 200 epochs, as its experiment builds it.

    Returns:
      record: ``schedule/<name>``, 200 float64 rates each.

    """
    record: dict[str, Tensor] = {}
    for name, factory in SCHEDULES.items():
        step = reference.tiny_step(
            factory,
            blocks=(1, 1, 1),
            classes=reference.CLASSES,
            epochs=200,
        )
        base_lr = from_plain(cast(object, step.optimizer.param_groups[0]["lr"]), float)
        record[f"schedule/{name}"] = torch.tensor(
            [
                step.schedule(epoch, base_lr=base_lr, total_epochs=200)
                for epoch in range(200)
            ],
            dtype=torch.float64,
        )
    return record


def optimizer_record() -> dict[str, Tensor]:
    """Return the settings of the optimizer ``exp000`` builds.

    Returns:
      record: The entries of :data:`SETTINGS`, as float64.

    """
    step = reference.tiny_step(
        exp000,
        blocks=(1, 1, 1),
        classes=reference.CLASSES,
        epochs=200,
    )
    group = cast(Mapping[str, object], step.optimizer.param_groups[0])
    values = [float(cast(float, group[name])) for name in SETTINGS]
    return {"optimizer/settings": torch.tensor(values, dtype=torch.float64)}


def lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train the shallowest network for two updates, one per epoch.

    Args:
      name: A key of :data:`SCHEDULES`.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: Per update the rate, the loss, the logits it came from and the
        logits of a fixed probe in evaluation mode afterwards.

    """
    rates: list[float] = []
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    probes: list[Tensor] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = reference.tiny_step(
            SCHEDULES[name],
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
            rates.append(
                from_plain(cast(object, step.optimizer.param_groups[0]["lr"]), float),
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
        f"lockstep/{name}/lr": torch.tensor(rates, dtype=torch.float64),
        f"lockstep/{name}/loss": torch.stack(losses),
        f"lockstep/{name}/logits": torch.stack(logits),
        f"lockstep/{name}/probe": torch.stack(probes),
    }


def port_record() -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Returns:
      record: The union of the records above, keyed as the stored golden is.

    """
    record = schedule_record() | optimizer_record() | reference.best_record()
    for name in SCHEDULES:
        record |= lockstep_record(name)
    return record
