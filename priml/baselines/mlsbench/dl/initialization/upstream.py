"""Runs of the pristine ``dl-weight-initialization`` script, under the capture's pins.

The script is imported from a clone of the benchmark, not vendored. Its
``main`` is traced as the ``cv-classification-loss`` capture traces that
task's, which records ResNet-56's state after the script's own seeding and
initialization; the network and lockstep runs repeat ``main``'s build,
initialization and epoch loop on the shallowest network. Nothing runs on a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import torch

from priml.baselines.mlsbench.dl.vision import reference, upstream
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import Tensor, nn

    from priml.baselines.mlsbench.dl.vision.upstream import (
        Lockstep,
        MainTrace,
        Upstream,
    )

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__dl-weight-initialization/tests/meta/pristine"
    "/pytorch-vision/custom_init.py",
)
"""The unedited training script, relative to the clone."""

EDITS: Final = Path("tasks/dl-weight-initialization/edits")
"""Directory of the task's baseline edits, relative to the clone."""

EDITED: Final = ("kaiming_normal", "fixup", "orthogonal")
"""The published baselines, each named as its edit file and as its port's variant."""

ARCH: Final = "resnet56"
"""What ``main`` tells ``initialize_weights`` the architecture is."""


class Script(Protocol):
    """The part of the ``dl-weight-initialization`` script the capture uses."""

    BasicBlock: type[nn.Module]

    ResNet: Callable[[type[nn.Module], list[int], int], nn.Module]

    def initialize_weights(self, model: nn.Module, config: dict[str, object]) -> None:
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
      scripts: ``normal`` (the pristine script) and each of :data:`EDITED`.

    """
    pristine = upstream.load_script(root / PRISTINE)
    return {"normal": pristine} | {
        name: upstream.edited_script(
            root,
            name,
            scratch,
            pristine=PRISTINE,
            edits=EDITS,
        )
        for name in EDITED
    }


def initialized(script: Upstream) -> nn.Module:
    """Build the shallowest network and initialize it as ``main`` does.

    Args:
      script: The script carrying the initialization under test.

    Returns:
      model: The network, with ``main``'s configuration given to the initializer.

    """
    task = cast(Script, cast(object, script))
    model = task.ResNet(task.BasicBlock, [1, 1, 1], reference.CLASSES)
    depth = sum(
        1 for m in model.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
    )
    config: dict[str, object] = {
        "arch": ARCH,
        "num_classes": reference.CLASSES,
        "depth": depth,
    }
    task.initialize_weights(model, config)
    return model


def upstream_network(
    script: Upstream,
    name: str,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Initialize the shallowest network at the script's widths and run it.

    Args:
      script: The script carrying the initialization under test.
      name: Which initialization, for the record's keys.

    Returns:
      record: Where the generator stands afterwards, and the logits of a
        training-mode and then an evaluation-mode forward.
      state: The network's whole state after initialization.

    """
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = initialized(script)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        draws = rng_fingerprint()
        media = reference.fixed_media(reference.STEPS)
        model.train()
        train = reference.logits_of(model, media)
        model.eval()
        evaluated = reference.logits_of(model, media)
    record = {
        f"network/{name}/draws": draws,
        f"network/{name}/train": train.detach(),
        f"network/{name}/eval": evaluated.detach(),
    }
    return record, state


def upstream_lockstep(script: Upstream, name: str, trace: MainTrace) -> Lockstep:
    """Repeat ``main``'s epoch loop on the shallowest network for two epochs.

    Args:
      script: The script carrying the initialization under test.
      name: Which initialization, for the record's keys.
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
        model = initialized(script)
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
    traces: dict[str, MainTrace],
    networks: dict[str, dict[str, Tensor]],
    lockstep: dict[str, Lockstep],
) -> dict[str, Tensor]:
    """Return the script's value of every quantity the default test replays.

    Args:
      traces: A run of each variant's ``main``, keyed as :func:`variants`.
      networks: Each variant's :func:`upstream_network` record.
      lockstep: Each variant's :func:`upstream_lockstep`.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.dl.initialization.reference.port_record`.

    """
    pristine = traces["normal"]
    record = {
        "schedule/lr": torch.tensor(pristine.rates, dtype=torch.float64),
        "optimizer/settings": torch.tensor(
            [pristine.settings[name] for name in reference.HYPERPARAMETERS],
            dtype=torch.float64,
        ),
    }
    record |= {f"draws/{name}": trace.draws for name, trace in traces.items()}
    for values in networks.values():
        record |= values
    for values, _ in lockstep.values():
        record |= values
    return record
