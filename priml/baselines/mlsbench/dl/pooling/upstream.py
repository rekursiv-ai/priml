"""Runs of the pristine ``cv-pooling-aggregation`` script, under the capture's pins.

The script is imported from a clone of the benchmark, not vendored. Its
``main`` is traced as the ``cv-classification-loss`` capture traces that
task's; each variant's ``CustomPool`` runs alone on fixed features; and the
lockstep run repeats ``main``'s epoch loop on the shallowest network. Nothing
runs on a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast

import torch

from priml.baselines.mlsbench.dl.vision import reference, upstream
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import Tensor, nn

    from priml.baselines.mlsbench.dl.vision.upstream import (
        Lockstep,
        MainTrace,
        Upstream,
    )

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__cv-pooling-aggregation/tests/meta/pristine"
    "/pytorch-vision/custom_pool.py",
)
"""The unedited training script, relative to the clone."""

EDITS: Final = Path("tasks/cv-pooling-aggregation/edits")
"""Directory of the task's baseline edits, relative to the clone."""

EDITED: Final = ("global_max", "gem", "avg_max")
"""The published baselines, each named as its edit file and as its port's variant."""

FEATURES: Final = (2, 3, 4, 5)
"""Batch, channels, height and width of the features each pool is run on."""


class Script(Protocol):
    """The part of the ``cv-pooling-aggregation`` script the capture uses."""

    BasicBlock: type[nn.Module]

    ResNet: Callable[[type[nn.Module], list[int], int], nn.Module]

    CustomPool: Callable[[], nn.Module]

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
      scripts: ``avg`` (the pristine script) and each of :data:`EDITED`.

    """
    pristine = upstream.load_script(root / PRISTINE)
    return {"avg": pristine} | {
        name: upstream.edited_script(
            root,
            name,
            scratch,
            pristine=PRISTINE,
            edits=EDITS,
        )
        for name in EDITED
    }


def fixed_features() -> Tensor:
    """Return features of shape :data:`FEATURES`: multiples of one eighth, some negative.

    Returns:
      features: Float32, distinct within every channel so no maximum is tied.

    """
    count = FEATURES[0] * FEATURES[1] * FEATURES[2] * FEATURES[3]
    return ((torch.arange(count) * 7 % 23 - 11) / 8).reshape(FEATURES)


def fixed_cotangent() -> Tensor:
    """Return the ``(B, C)`` weights the pooled output is summed with before backward.

    Returns:
      cotangent: Distinct multiples of one quarter, so every gradient differs.

    """
    return ((torch.arange(FEATURES[0] * FEATURES[1]) - 2) / 4).reshape(FEATURES[:2])


def pooled(pool: nn.Module, name: str) -> dict[str, Tensor]:
    """Run ``pool`` on the fixed features, forward and backward.

    Args:
      pool: A global pool, freshly built.
      name: Which pool, for the record's keys.

    Returns:
      record: The pooled value, the gradient of the features and, for a pool
        with parameters, their gradients joined in registration order.

    """
    with host_agnostic_numerics():
        leaf = fixed_features().requires_grad_()
        value = cast(torch.Tensor, pool(leaf))
        (value * fixed_cotangent()).sum().backward()
    record = {
        f"pool/{name}/value": value.detach(),
        f"pool/{name}/grad": cast(torch.Tensor, leaf.grad),
    }
    grads = [cast(torch.Tensor, p.grad).flatten() for p in pool.parameters()]
    if grads:
        record[f"pool/{name}/param_grad"] = torch.cat(grads)
    return record


def upstream_lockstep(
    script: Upstream,
    name: str,
    trace: MainTrace,
    *,
    arch: str = "resnet56",
    prefix: str = "",
) -> Lockstep:
    """Repeat ``main``'s epoch loop on the shallowest network for two epochs.

    Args:
      script: The script carrying the pool under test.
      name: Which pool, for the record's keys.
      trace: Source of the optimizer settings ``main`` used.
      arch: The setting's ``--arch``; its network is
        :func:`~priml.baselines.mlsbench.dl.vision.upstream.tiny_model`.
      prefix: Prepended to the record's keys, ``settings/<setting>/`` for a
        setting's run.

    Returns:
      lockstep: Per update the loss, its logits and a probe's logits
        afterwards; and the model's whole state after each update.

    """
    task = cast(Script, cast(object, script))
    criterion = upstream.CriterionSpy()
    probes: list[Tensor] = []
    states: list[dict[str, Tensor]] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = upstream.tiny_model(script, arch, reference.CLASSES)
        task.initialize_weights(model)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=trace.settings["lr"],
            momentum=trace.settings["momentum"],
            weight_decay=trace.settings["weight_decay"],
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
        f"{prefix}lockstep/{name}/loss": torch.stack(criterion.losses),
        f"{prefix}lockstep/{name}/logits": torch.stack(criterion.logits),
        f"{prefix}lockstep/{name}/probe": torch.stack(probes),
    }
    return record, states


def upstream_record(
    scripts: dict[str, Upstream],
    traces: dict[str, MainTrace],
    lockstep: dict[str, Lockstep],
) -> dict[str, Tensor]:
    """Return the reference's value of every quantity the default test replays.

    Args:
      scripts: Each variant's script, keyed as :func:`variants`.
      traces: A run of each variant's ``main``, keyed alike.
      lockstep: Each variant's :func:`upstream_lockstep`.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.dl.pooling.reference.port_record`.

    """
    record: dict[str, Tensor] = {}
    for name, script in scripts.items():
        record |= pooled(cast(Script, cast(object, script)).CustomPool(), name)
    pristine = traces["avg"]
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
