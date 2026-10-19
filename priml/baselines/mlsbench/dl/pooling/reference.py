"""Port-side runs the ``cv-pooling-aggregation`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script and these runs on the
same inputs under the same pins, compares them bit for bit, and stores the
script's values under ``testdata/``; ``reference_test.py`` replays these runs
against them. The data, loss, schedule and optimizer are the
``cv-classification-loss`` port's, already proven there; this task adds only
the pool, and each pool's fork at every other setting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from torch import Tensor

import torch

from priml.baselines.mlsbench.dl.pooling.experiments import (
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
from priml.baselines.mlsbench.dl.pooling.upstream import pooled
from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import nn

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


POOLS: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "avg": exp000,
    "global_max": exp001,
    "gem": exp002,
    "avg_max": exp003,
}
"""The experiment that carries each pool, keyed as the script's variants."""

SETTING_POOLS: Final[dict[str, dict[str, Callable[[], VisionTrainLoop]]]] = {
    "vgg16bn-cifar100": {
        "avg": exp004,
        "global_max": exp005,
        "gem": exp006,
        "avg_max": exp007,
    },
    "mobilenetv2-fmnist": {
        "avg": exp008,
        "global_max": exp009,
        "gem": exp010,
        "avg_max": exp011,
    },
}
"""Per other setting, the experiment that carries each pool on that setting.

The full network of each runs its pool on that network's last features, which
:func:`~priml.baselines.mlsbench.dl.vision.reference.task_settings_record`
compares with the script's."""


def pool_of(name: str) -> nn.Module:
    """Build the pool of experiment :data:`POOLS` ``[name]`` alone."""
    model = POOLS[name]().step.model
    assert isinstance(model, ResNet.Config)
    return model.pool.make()


def pool_record() -> dict[str, Tensor]:
    """Run every pool on fixed features, forward and backward.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.dl.pooling.upstream.pooled`.

    """
    record: dict[str, Tensor] = {}
    for name in POOLS:
        record |= pooled(pool_of(name), name)
    return record


def draws_record(names: Iterable[str]) -> dict[str, Tensor]:
    """Return where the generator stands after building each full network.

    Args:
      names: Keys of :data:`POOLS` to build.

    Returns:
      record: ``draws/<name>``, the next eight draws after ResNet-56 with
        that pool is built from the task's seed.

    """
    record: dict[str, Tensor] = {}
    for name in names:
        factory = POOLS[name]
        with reference.seeded(reference.SEED), host_agnostic_numerics():
            _ = reference.full_step(factory)
            record[f"draws/{name}"] = rng_fingerprint()
    return record


def lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train the shallowest network for two updates, one per epoch.

    Args:
      name: A key of :data:`POOLS`.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: Per update the loss, the logits it came from and the logits of
        a fixed probe in evaluation mode afterwards.

    """
    return _lockstep(POOLS[name], f"lockstep/{name}", states=states)


def setting_lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train a setting fork's shallowest network for two updates, one per epoch.

    Args:
      name: ``<setting>/<pool>``, a key of :data:`SETTING_POOLS` and of its entry.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: As :func:`lockstep_record`, under ``settings/<setting>/lockstep/<pool>/``.

    """
    setting, pool = name.split("/")
    return _lockstep(
        SETTING_POOLS[setting][pool],
        f"settings/{setting}/lockstep/{pool}",
        states=states,
    )


def settings_lockstep_record() -> dict[str, Tensor]:
    """Return every setting fork's :func:`setting_lockstep_record`."""
    record: dict[str, Tensor] = {}
    for setting, forks in SETTING_POOLS.items():
        for pool in forks:
            record |= setting_lockstep_record(f"{setting}/{pool}")
    return record


def port_record() -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Returns:
      record: The union of the records above and the shared step's schedule,
        optimizer and best accuracy, keyed as the stored golden is.

    """
    record = (
        pool_record()
        | reference.schedule_record()
        | reference.optimizer_record()
        | reference.best_record()
        | draws_record(POOLS)
    )
    for name in POOLS:
        record |= lockstep_record(name)
    return record


def _lockstep(
    factory: Callable[[], VisionTrainLoop],
    prefix: str,
    *,
    states: list[dict[str, Tensor]] | None,
) -> dict[str, Tensor]:
    """Train ``factory``'s shallowest network for two updates; see :func:`lockstep_record`."""
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
        f"{prefix}/loss": torch.stack(losses),
        f"{prefix}/logits": torch.stack(logits),
        f"{prefix}/probe": torch.stack(probes),
    }
