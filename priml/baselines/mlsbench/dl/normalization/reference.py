"""Port-side runs the ``dl-normalization`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script and these runs on the
same inputs under the same pins, compares them bit for bit, and stores the
script's values under ``testdata/``; ``reference_test.py`` replays these runs
against them. The data, loss and schedule are the ``cv-classification-loss``
port's, already proven there; this task adds only the normalization layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from torch import Tensor

import torch

from priml.baselines.mlsbench.dl.normalization.experiments import (
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
from priml.baselines.mlsbench.dl.normalization.norms import GroupNorm
from priml.baselines.mlsbench.dl.normalization.upstream import (
    GROUP_CHANNELS,
    norm_run,
)
from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.model import ResNet
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import nn

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


NORMS: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "batch_norm": exp000,
    "group_norm": exp001,
    "batch_instance_norm": exp002,
    "switchable_norm": exp003,
}
"""The experiment that carries each normalization, keyed as the script's variants."""

SETTING_NORMS: Final[dict[str, dict[str, Callable[[], VisionTrainLoop]]]] = {
    "resnet110-cifar100": {
        "batch_norm": exp004,
        "group_norm": exp005,
        "batch_instance_norm": exp006,
        "switchable_norm": exp007,
    },
    "mobilenetv2-fmnist": {
        "batch_norm": exp008,
        "group_norm": exp009,
        "batch_instance_norm": exp010,
        "switchable_norm": exp011,
    },
}
"""Per other setting of the task, keyed as the vision ``SETTINGS``, each norm's experiment."""

FMNIST: Final = "mobilenetv2-fmnist"
"""The setting whose dataset differs from ``exp000``'s, so its loaders are compared."""


def norm_of(factory: Callable[[], VisionTrainLoop]) -> Callable[[int], nn.Module]:
    """Return the normalization builder an experiment's model holds.

    Args:
      factory: The experiment.

    Returns:
      norm: ``step.model.norm``.

    """
    model = factory().step.model
    assert isinstance(model, ResNet.Config)
    return model.norm


def norm_record() -> dict[str, Tensor]:
    """Run each experiment's normalization layer alone, and count group norm's groups.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.dl.normalization.upstream.norm_run`,
        plus ``norm/group_norm/groups``.

    """
    record: dict[str, Tensor] = {}
    for name, factory in NORMS.items():
        record |= norm_run(norm_of(factory), name)
    record["norm/group_norm/groups"] = torch.tensor(
        [GroupNorm(c).num_groups for c in GROUP_CHANNELS],
    )
    return record


def draws_record(names: Iterable[str]) -> dict[str, Tensor]:
    """Return where the generator stands after building each full network.

    Args:
      names: Keys of :data:`NORMS` to build.

    Returns:
      record: ``draws/<name>``, keyed as :data:`NORMS`, the next eight draws.

    """
    return {
        f"draws/{name}": next(iter(reference.draws_record(NORMS[name]).values()))
        for name in names
    }


def lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train the shallowest network for two updates, one per epoch.

    Args:
      name: A key of :data:`NORMS`.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: As :func:`lockstep_run`, under ``lockstep/<name>/``.

    """
    return lockstep_run(NORMS[name], prefix=f"lockstep/{name}/", states=states)


def setting_lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
) -> dict[str, Tensor]:
    """Train a setting's shallowest network for two updates, one per epoch.

    Args:
      name: ``<setting>/<norm>``, keys of :data:`SETTING_NORMS`.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: As :func:`lockstep_run`, under ``settings/<setting>/lockstep/<norm>/``.

    """
    setting, norm = name.split("/")
    return lockstep_run(
        SETTING_NORMS[setting][norm],
        prefix=f"settings/{setting}/lockstep/{norm}/",
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
      record: Per update the loss, the logits it came from and the logits of
        a fixed probe in evaluation mode afterwards.

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
        f"{prefix}loss": torch.stack(losses),
        f"{prefix}logits": torch.stack(logits),
        f"{prefix}probe": torch.stack(probes),
    }


def settings_record() -> dict[str, Tensor]:
    """Run every setting's lockstep, keyed as the settings golden is.

    Returns:
      record: :func:`setting_lockstep_record` of every entry of :data:`SETTING_NORMS`.

    """
    record: dict[str, Tensor] = {}
    for setting, norms in SETTING_NORMS.items():
        for name in norms:
            record |= setting_lockstep_record(f"{setting}/{name}")
    return record


def port_record() -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Returns:
      record: The union of the records above and the reported accuracy, keyed
        as the stored golden is.

    """
    record = (
        norm_record()
        | reference.schedule_record(exp000)
        | reference.optimizer_record(exp000)
        | draws_record(NORMS)
        | reference.best_record()
    )
    for name in NORMS:
        record |= lockstep_record(name)
    return record
