"""Runs of the pristine ``dl-normalization`` script, under the capture's pins.

The script is imported from a clone of the benchmark, not vendored. Its
``main`` is traced as the ``cv-classification-loss`` capture traces that
task's, which records the full network's initial state, the rate of each of
the 200 epochs and the optimizer; the lockstep run repeats ``main``'s epoch
loop on the shallowest network; the norm run drives one ``CustomNorm`` alone.
Nothing runs on a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import torch

from priml.baselines.mlsbench.pytorch_vision import reference, upstream
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import joined


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import Tensor, nn

    from priml.baselines.mlsbench.pytorch_vision.upstream import (
        Lockstep,
        MainTrace,
        Upstream,
    )

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__dl-normalization/tests/meta/pristine"
    "/pytorch-vision/custom_norm.py",
)
"""The unedited training script, relative to the clone."""

EDITS: Final = Path("tasks/dl-normalization/edits")
"""Directory of the task's baseline edits, relative to the clone."""

EDITED: Final = ("group_norm", "batch_instance_norm", "switchable_norm")
"""The baselines, each named as its edit file and as its port's variant."""

FEATURES_SHAPE: Final = (2, 5, 3, 4)
"""Batch, channels, height and width of the norm run's input."""

GROUP_CHANNELS: Final = (5, 16, 32, 33, 64, 100)
"""Channel counts whose group count is compared; 33 and 100 are not divisible by 32."""


class Script(Protocol):
    """The part of the ``dl-normalization`` script the capture uses."""

    BasicBlock: type[nn.Module]

    ResNet: Callable[[type[nn.Module], list[int], int], nn.Module]

    CustomNorm: Callable[[int], nn.Module]

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
      scripts: ``batch_norm`` (the pristine script) and each of :data:`EDITED`.

    """
    pristine = upstream.load_script(root / PRISTINE)
    return {"batch_norm": pristine} | {
        name: upstream.edited_script(
            root,
            name,
            scratch,
            pristine=PRISTINE,
            edits=EDITS,
        )
        for name in EDITED
    }


def fixed_features() -> tuple[Tensor, Tensor]:
    """Return the norm run's input and the cotangent its output is weighted by.

    Returns:
      features: Multiples of one eighth, of shape :data:`FEATURES_SHAPE`.
      cotangent: Multiples of one quarter, of the same shape.

    """
    count = (
        FEATURES_SHAPE[0] * FEATURES_SHAPE[1] * FEATURES_SHAPE[2] * FEATURES_SHAPE[3]
    )
    index = torch.arange(count)
    features = ((index * 7 % 23 - 11) / 8).reshape(FEATURES_SHAPE)
    cotangent = ((index * 5 % 13 - 6) / 4).reshape(FEATURES_SHAPE)
    return features, cotangent


def norm_run(build: Callable[[int], nn.Module], name: str) -> dict[str, Tensor]:
    """Run one normalization layer forward and backward, then forward in evaluation.

    Args:
      build: Builds the layer from a channel count, initialized.
      name: Which layer, for the record's keys.

    Returns:
      record: The training-mode output, the gradients of its weighted sum with
        respect to the input and to every parameter in registration order, and
        the evaluation-mode output afterwards.

    """
    features, cotangent = fixed_features()
    with host_agnostic_numerics():
        layer = build(FEATURES_SHAPE[1])
        leaf = features.clone().requires_grad_()
        layer.train()
        output = cast("Tensor", layer(leaf))
        (output * cotangent).sum().backward()
        layer.eval()
        with torch.no_grad():
            evaluated = cast("Tensor", layer(features))
    return {
        f"norm/{name}/train": output.detach(),
        f"norm/{name}/grad_input": cast("Tensor", leaf.grad),
        f"norm/{name}/grad_params": joined(
            cast("Tensor", p.grad) for p in layer.parameters()
        ),
        f"norm/{name}/eval": evaluated,
    }


def upstream_norm(script: Upstream, name: str) -> dict[str, Tensor]:
    """Run the script's ``CustomNorm`` as :func:`norm_run` does, after its own init.

    Args:
      script: The script carrying the layer under test.
      name: Which layer, for the record's keys.

    Returns:
      record: Keyed as :func:`norm_run`.

    """
    task = cast(Script, cast(object, script))

    def build(channels: int) -> nn.Module:
        layer = task.CustomNorm(channels)
        task.initialize_weights(layer)
        return layer

    return norm_run(build, name)


def upstream_groups(script: Upstream) -> dict[str, Tensor]:
    """Return the group count the ``group_norm`` edit picks for each channel count.

    Args:
      script: The ``group_norm`` script.

    Returns:
      record: ``norm/group_norm/groups``, one entry per :data:`GROUP_CHANNELS`.

    """
    task = cast(Script, cast(object, script))
    return {
        "norm/group_norm/groups": torch.tensor(
            [
                cast(
                    "nn.GroupNorm",
                    task.CustomNorm(c).get_submodule("norm"),
                ).num_groups
                for c in GROUP_CHANNELS
            ],
        ),
    }


def upstream_lockstep(script: Upstream, name: str, trace: MainTrace) -> Lockstep:
    """Repeat ``main``'s epoch loop on the shallowest network for two epochs.

    Args:
      script: The script carrying the normalization under test.
      name: Which normalization, for the record's keys.
      trace: Source of the optimizer settings ``main`` used.

    Returns:
      lockstep: Per update the loss, its logits and a probe's logits
        afterwards; and the model's whole state after each update.

    """
    task = cast(Script, cast(object, script))
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
    traces: dict[str, MainTrace],
    lockstep: dict[str, Lockstep],
) -> dict[str, Tensor]:
    """Return the script's value of every quantity the default test replays.

    Args:
      scripts: Each variant's script, keyed as :func:`variants`.
      traces: A run of each variant's ``main``.
      lockstep: Each variant's :func:`upstream_lockstep`.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.normalization.reference.port_record`.

    """
    record: dict[str, Tensor] = {}
    for name, script in scripts.items():
        record |= upstream_norm(script, name)
    record |= upstream_groups(scripts["group_norm"])
    pristine = traces["batch_norm"]
    record["schedule/lr"] = torch.tensor(pristine.rates, dtype=torch.float64)
    record["optimizer/settings"] = torch.tensor(
        [pristine.settings[name] for name in reference.HYPERPARAMETERS],
        dtype=torch.float64,
    )
    record["best/accuracy"] = torch.tensor(pristine.best, dtype=torch.float64)
    for name, trace in traces.items():
        record[f"draws/{name}"] = trace.draws
    for values, _ in lockstep.values():
        record |= values
    return record
