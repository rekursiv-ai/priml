"""Port-side runs the ``dl-lr-schedule`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script and these runs on the
same inputs under the same pins, compares them bit for bit, and stores the
script's values under ``testdata/``; ``reference_test.py`` replays these runs
against them. The network, data and loss are the ``cv-classification-loss``
port's, already proven there; this task adds only the schedule and the
optimizer the script builds without a scheduler, and each schedule's fork at
every other setting.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import TYPE_CHECKING, Final, cast

from torch import Tensor
from treekle import from_plain

import torch

from priml.baselines.mlsbench.dl.lr_schedule.experiments import (
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

SETTING_SCHEDULES: Final[dict[str, dict[str, Callable[[], VisionTrainLoop]]]] = {
    "resnet20-cifar10": {
        "constant": exp004,
        "cosine": exp005,
        "warmup_cosine": exp006,
        "one_cycle": exp007,
    },
    "mobilenetv2-fmnist": {
        "constant": exp008,
        "cosine": exp009,
        "warmup_cosine": exp010,
        "one_cycle": exp011,
    },
}
"""Per other setting, the experiment that carries each schedule on that setting."""


def schedule_record() -> dict[str, Tensor]:
    """Return each schedule's rate for all 200 epochs, as its experiment builds it.

    Returns:
      record: ``schedule/<name>``, 200 float64 rates each.

    """
    return {f"schedule/{name}": _rates(factory) for name, factory in SCHEDULES.items()}


def setting_schedule_record() -> dict[str, Tensor]:
    """Return each setting fork's rate for all 200 epochs, from its own config.

    Returns:
      record: ``settings/<setting>/schedule/<name>``, 200 float64 rates each.

    """
    return {
        f"settings/{setting}/schedule/{name}": _rates(factory)
        for setting, forks in SETTING_SCHEDULES.items()
        for name, factory in forks.items()
    }


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
    return _lockstep(SCHEDULES[name], f"lockstep/{name}", states=states)


def setting_lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train a setting fork's shallowest network for two updates, one per epoch.

    Args:
      name: ``<setting>/<schedule>``, a key of :data:`SETTING_SCHEDULES` and of
        its entry.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: As :func:`lockstep_record`, under
        ``settings/<setting>/lockstep/<schedule>/``.

    """
    setting, schedule = name.split("/")
    return _lockstep(
        SETTING_SCHEDULES[setting][schedule],
        f"settings/{setting}/lockstep/{schedule}",
        states=states,
    )


def settings_port_record() -> dict[str, Tensor]:
    """Return :func:`setting_schedule_record` and every fork's :func:`setting_lockstep_record`."""
    record = setting_schedule_record()
    for setting, forks in SETTING_SCHEDULES.items():
        for schedule in forks:
            record |= setting_lockstep_record(f"{setting}/{schedule}")
    return record


def port_record() -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Returns:
      record: The union of the records above, keyed as the stored golden is.

    """
    record = (
        schedule_record()
        | optimizer_record()
        | reference.best_record()
        | reference.draws_record(exp000)
    )
    for name in SCHEDULES:
        record |= lockstep_record(name)
    return record


# The schedule, base rate and horizon all come from the step config, so an experiment
# that moved any of them shows here.
def _rates(factory: Callable[[], VisionTrainLoop]) -> Tensor:
    """Return ``factory``'s rate at every epoch of its budget, read off its config."""
    step = factory().step
    schedule = step.schedule.make()
    optimizer = step.optimizer.make()
    assert isinstance(optimizer, partial)
    base_lr = from_plain(cast(object, optimizer.keywords["lr"]), float)
    total = int(step.train_budget_epochs)
    return torch.tensor(
        [
            schedule(epoch, base_lr=base_lr, total_epochs=total)
            for epoch in range(total)
        ],
        dtype=torch.float64,
    )


def _lockstep(
    factory: Callable[[], VisionTrainLoop],
    prefix: str,
    *,
    states: list[dict[str, Tensor]] | None,
) -> dict[str, Tensor]:
    """Train ``factory``'s shallowest network for two updates; see :func:`lockstep_record`."""
    rates: list[float] = []
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
        f"{prefix}/lr": torch.tensor(rates, dtype=torch.float64),
        f"{prefix}/loss": torch.stack(losses),
        f"{prefix}/logits": torch.stack(logits),
        f"{prefix}/probe": torch.stack(probes),
    }
