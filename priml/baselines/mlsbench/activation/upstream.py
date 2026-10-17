"""Runs of the pristine ``dl-activation-function`` script, under the capture's pins.

The script and each baseline's edit are imported from a clone of the benchmark,
not vendored. ``main`` is traced as the ``cv-classification-loss`` capture
traces that task's, at the ``resnet20-cifar10`` setting; the lockstep run
repeats ``main``'s epoch loop on the shallowest network. Nothing runs on a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from torch import Tensor

import torch

from priml.baselines.mlsbench.activation.reference import fixed_inputs
from priml.baselines.mlsbench.pytorch_vision import reference, upstream
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import nn

    from priml.baselines.mlsbench.pytorch_vision.upstream import (
        Lockstep,
        MainTrace,
        Upstream,
    )
    from priml.math.custom_types import TensorFn

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__dl-activation-function/tests/meta/pristine"
    "/pytorch-vision/custom_activation.py",
)
"""The unedited training script, relative to the clone."""

EDITS: Final = Path("tasks/dl-activation-function/edits")
"""Directory of the task's baseline edits, relative to the clone."""

EDITED: Final = ("gelu", "silu", "mish")
"""The baselines, each named as its edit file and as its port's variant."""

ARCH: Final = "resnet20"
"""The ``--arch`` of the ``resnet20-cifar10`` setting."""

DATASET: Final = "cifar10"
"""The ``--dataset`` of the ``resnet20-cifar10`` setting."""

CLASSES: Final = 10
"""Classes of the ``resnet20-cifar10`` setting."""


@runtime_checkable
class Script(Protocol):
    """The part of the ``dl-activation-function`` script the capture uses."""

    BasicBlock: type[nn.Module]

    ResNet: Callable[[type[nn.Module], list[int], int], nn.Module]

    CustomActivation: Callable[[], TensorFn]

    def initialize_weights(self, model: nn.Module) -> None:
        """Initialize in place."""

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
      scripts: ``relu`` (the pristine script) and each of :data:`EDITED`.

    """
    pristine = upstream.load_script(root / PRISTINE)
    return {"relu": pristine} | {
        name: upstream.edited_script(
            root,
            name,
            scratch,
            pristine=PRISTINE,
            edits=EDITS,
        )
        for name in EDITED
    }


def trace_setting(script: Upstream) -> MainTrace:
    """Run ``main`` for 200 epochs of the ``resnet20-cifar10`` setting.

    Args:
      script: The pristine script.

    Returns:
      trace: What ``main``'s replaced calls saw.

    """
    return upstream.trace_main(script, arch=ARCH, dataset=DATASET, classes=CLASSES)


def upstream_activation(script: Upstream, name: str) -> dict[str, Tensor]:
    """Apply one ``CustomActivation`` to fixed inputs, forward and backward.

    Args:
      script: The script carrying the activation under test.
      name: Which activation, for the record's keys.

    Returns:
      record: The activation's values and its gradient at every input.

    """
    task = _task(script)
    leaf = fixed_inputs().requires_grad_()
    with host_agnostic_numerics():
        value = task.CustomActivation()(leaf)
        value.sum().backward()
    assert isinstance(leaf.grad, Tensor)
    return {
        f"activation/{name}/value": value.detach(),
        f"activation/{name}/grad": leaf.grad,
    }


def upstream_lockstep(script: Upstream, name: str, trace: MainTrace) -> Lockstep:
    """Repeat ``main``'s epoch loop on the shallowest network for two epochs.

    Args:
      script: The script carrying the activation under test.
      name: Which activation, for the record's keys.
      trace: Source of the optimizer settings ``main`` used.

    Returns:
      lockstep: Per update the loss, its logits and a probe's logits
        afterwards; and the model's whole state after each update.

    """
    task = _task(script)
    criterion = upstream.CriterionSpy()
    probes: list[Tensor] = []
    states: list[dict[str, Tensor]] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = task.ResNet(task.BasicBlock, [1, 1, 1], reference.CLASSES)
        task.initialize_weights(model)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=trace.settings["lr"],
            momentum=trace.settings["momentum"],
            dampening=trace.settings["dampening"],
            weight_decay=trace.settings["weight_decay"],
            nesterov=bool(trace.settings["nesterov"]),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=reference.EPOCHS,
        )
        for index in range(reference.STEPS):
            batch = [(reference.fixed_media(index), reference.fixed_label(index))]
            task.train_epoch(model, batch, criterion, optimizer, torch.device("cpu"))
            model.eval()
            with torch.no_grad():
                probes.append(
                    reference.logits_of(model, reference.fixed_media(reference.STEPS)),
                )
            states.append({k: v.clone() for k, v in model.state_dict().items()})
            scheduler.step()
    record = {
        f"lockstep/{name}/loss": torch.stack(criterion.losses),
        f"lockstep/{name}/logits": torch.stack(criterion.logits),
        f"lockstep/{name}/probe": torch.stack(probes),
    }
    return record, states


def upstream_record(
    scripts: dict[str, Upstream],
    trace: MainTrace,
    lockstep: dict[str, Lockstep],
) -> dict[str, Tensor]:
    """Return the script's value of every quantity the default test replays.

    Args:
      scripts: Each variant, keyed as :func:`variants`.
      trace: :func:`trace_setting` of the pristine script.
      lockstep: Each variant's :func:`upstream_lockstep`.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.activation.reference.port_record`.

    """
    record: dict[str, Tensor] = {}
    for name, script in scripts.items():
        record |= upstream_activation(script, name)
    record["schedule/lr"] = torch.tensor(trace.rates, dtype=torch.float64)
    record["optimizer/settings"] = torch.tensor(
        [trace.settings[name] for name in reference.HYPERPARAMETERS],
        dtype=torch.float64,
    )
    record["draws/exp000"] = trace.draws
    record["best/accuracy"] = torch.tensor(trace.best, dtype=torch.float64)
    for values, _ in lockstep.values():
        record |= values
    return record


# The capture shares the vision port's ``Upstream`` type with its helpers, but this
# script has no ``compute_loss`` and its ``train_epoch`` takes a criterion; narrowing
# from ``object`` checks the members this module uses instead of intersecting both.
def _task(script: object) -> Script:
    """Narrow an imported script to the members this module calls."""
    assert isinstance(script, Script)
    return script
