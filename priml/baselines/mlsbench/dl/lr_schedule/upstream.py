"""Runs of the pristine ``dl-lr-schedule`` script, under the capture's pins.

The script is imported from a clone of the benchmark, not vendored. Its
``main`` is traced as the ``cv-classification-loss`` capture traces that
task's, which records the rate of each of the 200 epochs; the lockstep run
repeats ``main``'s epoch loop on the shallowest network. Nothing runs on a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

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
    "harbor/tasks-docker/mls-bench__dl-lr-schedule/tests/meta/pristine"
    "/pytorch-vision/custom_schedule.py",
)
"""The unedited training script, relative to the clone."""

EDITS: Final = Path("tasks/dl-lr-schedule/edits")
"""Directory of the task's baseline edits, relative to the clone."""

EDITED: Final = ("cosine", "warmup_cosine", "one_cycle")
"""The baselines, each named as its edit file and as its port's variant."""

SETTINGS: Final = (
    "lr",
    "momentum",
    "dampening",
    "weight_decay",
    "nesterov",
    "maximize",
    "differentiable",
)
"""Parameter-group entries compared; the script builds no scheduler, so no ``initial_lr``."""

CONFIG: Final = {"arch": "resnet56", "dataset": "cifar100"}
"""What ``main`` tells ``get_lr`` in the ``resnet56-cifar100`` setting."""


@runtime_checkable
class Script(Protocol):
    """The part of the ``dl-lr-schedule`` script the capture uses."""

    BasicBlock: type[nn.Module]

    ResNet: Callable[[type[nn.Module], list[int], int], nn.Module]

    def initialize_weights(self, model: nn.Module) -> None:
        """Initialize in place."""

    def get_lr(
        self,
        epoch: int,
        total_epochs: int,
        base_lr: float,
        config: dict[str, str],
    ) -> float:
        """Return the rate of ``epoch``."""
        raise NotImplementedError

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
      scripts: ``constant`` (the pristine script) and each of :data:`EDITED`.

    """
    pristine = upstream.load_script(root / PRISTINE)
    return {"constant": pristine} | {
        name: upstream.edited_script(
            root,
            name,
            scratch,
            pristine=PRISTINE,
            edits=EDITS,
        )
        for name in EDITED
    }


def upstream_lockstep(
    script: Upstream,
    name: str,
    trace: MainTrace,
    *,
    setting: str = "",
) -> Lockstep:
    """Repeat ``main``'s epoch loop on the shallowest network for two epochs.

    Args:
      script: The script carrying the schedule under test.
      name: Which schedule, for the record's keys.
      trace: The pristine ``main``'s run, whose optimizer settings, ``--lr``
        included, the loop builds its own from.
      setting: A key of :data:`~priml.baselines.mlsbench.dl.vision.reference.SETTINGS`
        whose shallowest network
        (:func:`~priml.baselines.mlsbench.dl.vision.upstream.tiny_model`)
        trains and whose ``--arch``/``--dataset`` ``get_lr`` is told, under
        ``settings/<setting>/`` keys; ``resnet56-cifar100`` when empty.

    Returns:
      lockstep: Per update the rate, the loss, its logits and a probe's logits
        afterwards; and the model's whole state after each update.

    """
    task = upstream.narrowed(script, Script)
    criterion = _CriterionSpy()
    rates: list[float] = []
    probes: list[Tensor] = []
    states: list[dict[str, Tensor]] = []
    config = dict(CONFIG)
    prefix = ""
    if setting:
        spec = reference.SETTINGS[setting]
        config = {"arch": spec.arch, "dataset": spec.dataset}
        prefix = f"settings/{setting}/"
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = upstream.tiny_model(script, config["arch"], reference.CLASSES)
        task.initialize_weights(model)
        base_lr = trace.settings["lr"]
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=trace.settings["momentum"],
            weight_decay=trace.settings["weight_decay"],
        )
        for index in range(reference.STEPS):
            rate = task.get_lr(index, reference.EPOCHS, base_lr, dict(config))
            for group in optimizer.param_groups:
                group["lr"] = rate
            rates.append(rate)
            batch = [(reference.fixed_media(index), reference.fixed_label(index))]
            task.train_epoch(model, batch, criterion, optimizer, torch.device("cpu"))
            model.eval()
            with torch.no_grad():
                probes.append(
                    reference.model_logits(
                        model,
                        reference.fixed_media(reference.STEPS),
                    ),
                )
            states.append({k: v.clone() for k, v in model.state_dict().items()})
    record = {
        f"{prefix}lockstep/{name}/lr": torch.tensor(rates, dtype=torch.float64),
        f"{prefix}lockstep/{name}/loss": torch.stack(criterion.losses),
        f"{prefix}lockstep/{name}/logits": torch.stack(criterion.logits),
        f"{prefix}lockstep/{name}/probe": torch.stack(probes),
    }
    return record, states


def upstream_setting_schedules(
    scripts: dict[str, Upstream],
    settings: Iterable[str],
) -> dict[str, Tensor]:
    """Trace each variant's ``main`` at each setting and keep the rate of every epoch.

    ``main`` tells ``get_lr`` the setting's ``--arch`` and ``--dataset``, so a
    schedule may differ per setting; the trace records what it set.

    Args:
      scripts: The script carrying each schedule, keyed as :func:`variants`.
      settings: Keys of :data:`~priml.baselines.mlsbench.dl.vision.reference.SETTINGS`.

    Returns:
      record: Keyed as
        :func:`~priml.baselines.mlsbench.dl.lr_schedule.reference.setting_schedule_record`.

    """
    record: dict[str, Tensor] = {}
    for setting in settings:
        spec = reference.SETTINGS[setting]
        for name, script in scripts.items():
            trace = upstream.trace_main(
                script,
                arch=spec.arch,
                dataset=spec.dataset,
                classes=spec.classes,
            )
            record[f"settings/{setting}/schedule/{name}"] = torch.tensor(
                trace.rates,
                dtype=torch.float64,
            )
    return record


def upstream_record(
    traces: dict[str, MainTrace],
    lockstep: dict[str, Lockstep],
) -> dict[str, Tensor]:
    """Return the reference's value of every quantity the default test replays.

    Args:
      traces: A run of each variant's ``main``, keyed as :func:`variants`.
      lockstep: Each variant's :func:`upstream_lockstep`.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.dl.lr_schedule.reference.port_record`.

    """
    record = {
        f"schedule/{name}": torch.tensor(trace.rates, dtype=torch.float64)
        for name, trace in traces.items()
    }
    record["optimizer/settings"] = torch.tensor(
        [traces["constant"].settings[name] for name in SETTINGS],
        dtype=torch.float64,
    )
    record["best/accuracy"] = torch.tensor(traces["constant"].best, dtype=torch.float64)
    for values, _ in lockstep.values():
        record |= values
    return record


class _CriterionSpy:
    """Cross-entropy, as ``main`` builds it, keeping what it saw."""

    def __init__(self) -> None:
        self.criterion = torch.nn.CrossEntropyLoss()
        self.losses: list[Tensor] = []
        self.logits: list[Tensor] = []

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Return the batch's loss."""
        loss = self.criterion(logits, targets)
        self.losses.append(loss.detach())
        self.logits.append(logits.detach())
        return loss
