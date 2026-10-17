"""Port-side runs the ``dl-normalization`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script and these runs on the
same inputs under the same pins, compares them bit for bit, and stores the
script's values under ``testdata/``; ``reference_test.py`` replays these runs
against them. The data, loss and schedule are the ``cv-classification-loss``
port's, already proven there; this task adds only the normalization layer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, cast

from torch import Tensor

import torch

from priml.baselines.mlsbench.normalization.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.mlsbench.normalization.norms import GroupNorm
from priml.baselines.mlsbench.normalization.upstream import (
    GROUP_CHANNELS,
    SETTINGS,
    norm_run,
)
from priml.baselines.mlsbench.pytorch_vision import reference
from priml.baselines.mlsbench.pytorch_vision.model import ResNet
from priml.baselines.mlsbench.pytorch_vision.train_step import VisionTrainStep
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from torch import nn

    from priml.baselines.mlsbench.pytorch_vision.experiments import VisionTrainLoop


NORMS: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "batch_norm": exp000,
    "group_norm": exp001,
    "batch_instance_norm": exp002,
    "switchable_norm": exp003,
}
"""The experiment that carries each normalization, keyed as the script's variants."""


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
      record: Keyed as :func:`~priml.baselines.mlsbench.normalization.upstream.norm_run`,
        plus ``norm/group_norm/groups``.

    """
    record: dict[str, Tensor] = {}
    for name, factory in NORMS.items():
        record |= norm_run(norm_of(factory), name)
    record["norm/group_norm/groups"] = torch.tensor(
        [GroupNorm(c).num_groups for c in GROUP_CHANNELS],
    )
    return record


def schedule_record() -> dict[str, Tensor]:
    """Return the learning rate of each of the 200 epochs of ``exp000``.

    Returns:
      record: The 200 rates, as float64.

    """
    step = reference.tiny_step(exp000, blocks=(1, 1, 1), classes=3, epochs=200)
    base_lr = cast(float, step.optimizer.param_groups[0]["lr"])
    rates = [
        step.schedule(epoch, base_lr=base_lr, total_epochs=200) for epoch in range(200)
    ]
    return {"schedule/lr": torch.tensor(rates, dtype=torch.float64)}


def optimizer_record() -> dict[str, Tensor]:
    """Return the settings of the optimizer ``exp000`` builds.

    Returns:
      record: The entries of :data:`SETTINGS`, as float64.

    """
    step = reference.tiny_step(exp000, blocks=(1, 1, 1), classes=3, epochs=200)
    group = cast(Mapping[str, object], step.optimizer.param_groups[0])
    values = [float(cast(float, group[name])) for name in SETTINGS]
    return {"optimizer/settings": torch.tensor(values, dtype=torch.float64)}


def full_step(factory: Callable[[], VisionTrainLoop]) -> VisionTrainStep:
    """Build an experiment's step at full size on the CPU.

    Args:
      factory: The experiment.

    Returns:
      step: ResNet-56 with 100 classes and the experiment's normalization.

    """
    config = factory().step
    config.parallelism = NoParallel.Config(device="cpu")
    step = config.make()
    assert isinstance(step, VisionTrainStep)
    return step


def draws_record() -> dict[str, Tensor]:
    """Return where the generator stands after building each full network.

    Returns:
      record: ``draws/<name>``, the next eight draws.

    """
    record: dict[str, Tensor] = {}
    for name, factory in NORMS.items():
        with reference.seeded(reference.SEED), host_agnostic_numerics():
            _ = full_step(factory)
            record[f"draws/{name}"] = rng_fingerprint()
    return record


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
      record: Per update the loss, the logits it came from and the logits of
        a fixed probe in evaluation mode afterwards.

    """
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    probes: list[Tensor] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = reference.tiny_step(
            NORMS[name],
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
      record: The union of the records above and the reported accuracy, keyed
        as the stored golden is.

    """
    record = (
        norm_record()
        | schedule_record()
        | optimizer_record()
        | draws_record()
        | reference.best_record()
    )
    for name in NORMS:
        record |= lockstep_record(name)
    return record
