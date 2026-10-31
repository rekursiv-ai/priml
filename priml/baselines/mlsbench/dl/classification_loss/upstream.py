"""Runs of the pristine ``cv-classification-loss`` script and its baselines' edits.

The script is imported from a clone of the benchmark, not vendored. It is the
script the shared recipe ports, and that recipe's capture proves its network,
data, schedule and optimizer. Here each variant's ``compute_loss`` runs alone on
fixed logits and on each other setting's logit width, and the lockstep run
repeats ``main``'s epoch loop on the shallowest network with the variant's
loss. Nothing runs on a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from unittest import mock

from torch import Tensor

import torch

from priml.baselines.mlsbench.dl.classification_loss.reference import (
    setting_logits,
)
from priml.baselines.mlsbench.dl.vision import reference, upstream
from priml.baselines.mlsbench.dl.vision.upstream import PRISTINE
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch import nn

    from priml.baselines.mlsbench.dl.vision.upstream import (
        Lockstep,
        MainTrace,
        Upstream,
    )

EDITS: Final = Path("tasks/cv-classification-loss/edits")
"""Directory of the task's baseline edits, relative to the clone."""

VARIANTS: Final = {
    "label_smoothing": "label_smoothing",
    "focal": "focal_loss",
    "poly": "poly_loss",
}
"""Maps each published loss but ``cross_entropy`` to the stem of its edit file."""


def variants(root: Path, scratch: Path) -> dict[str, Upstream]:
    """Import the pristine script and each baseline's edit of it.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts are written to.

    Returns:
      scripts: ``cross_entropy`` (the pristine script) and each of
        :data:`VARIANTS`, keyed as the port's losses are.

    """
    pristine = upstream.load_script(root / PRISTINE)
    return {"cross_entropy": pristine} | {
        name: upstream.edited_script(
            root,
            file,
            scratch,
            pristine=PRISTINE,
            edits=EDITS,
        )
        for name, file in VARIANTS.items()
    }


def upstream_lockstep(
    script: Upstream,
    name: str,
    trace: MainTrace,
    *,
    arch: str = "resnet56",
    prefix: str = "",
) -> Lockstep:
    """Train the shallowest network of an ``--arch`` for two updates, one per epoch.

    Args:
      script: The reference script carrying the loss under test.
      name: Which loss, for the record's keys.
      trace: Source of the optimizer settings ``main`` used.
      arch: The setting's ``--arch``; its network is
        :func:`~priml.baselines.mlsbench.dl.vision.upstream.tiny_model`.
      prefix: Prepended to the record's keys, ``settings/<setting>/`` for a
        setting's run.

    Returns:
      lockstep: Per update the loss, its logits, a probe's logits afterwards
        and the configuration the loss was given; and the model's whole state
        after each update.

    """
    spy = _LossSpy(script)
    probes: list[Tensor] = []
    states: list[dict[str, Tensor]] = []
    with (
        reference.seeded(reference.SEED),
        host_agnostic_numerics(),
        mock.patch.object(script, "compute_loss", spy),
    ):
        model = upstream.tiny_model(script, arch, reference.CLASSES)
        script.initialize_weights(model)
        optimizer = _sgd(model, trace)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=reference.EPOCHS,
        )
        config = {
            "num_classes": reference.CLASSES,
            "epoch": 0,
            "total_epochs": reference.EPOCHS,
        }
        for index in range(reference.STEPS):
            config["epoch"] = index
            batch = [(reference.fixed_media(index), reference.fixed_label(index))]
            script.train_epoch(model, batch, optimizer, torch.device("cpu"), config)
            model.eval()
            with torch.no_grad():
                probes.append(
                    reference.model_logits(
                        model,
                        reference.fixed_media(reference.STEPS),
                    ),
                )
            states.append({k: v.clone() for k, v in model.state_dict().items()})
            scheduler.step()
    record = {
        f"{prefix}lockstep/{name}/loss": torch.stack(spy.losses),
        f"{prefix}lockstep/{name}/logits": torch.stack(spy.logits),
        f"{prefix}lockstep/{name}/probe": torch.stack(probes),
        f"{prefix}lockstep/{name}/context": torch.tensor(
            [[c["num_classes"], c["epoch"], c["total_epochs"]] for c in spy.configs],
        ),
    }
    return record, states


def upstream_setting_losses(
    scripts: dict[str, Upstream],
    settings: Iterable[str],
) -> dict[str, Tensor]:
    """Run each variant's ``compute_loss`` on every setting's logit width.

    Args:
      scripts: The script carrying each loss, keyed as :func:`variants`.
      settings: Keys of :data:`~priml.baselines.mlsbench.dl.vision.reference.SETTINGS`.

    Returns:
      record: Keyed as
        :func:`~priml.baselines.mlsbench.dl.classification_loss.reference.setting_loss_record`.

    """
    record: dict[str, Tensor] = {}
    with host_agnostic_numerics():
        for setting in settings:
            classes = reference.SETTINGS[setting].classes
            logits, label = setting_logits(classes)
            # What ``main`` tells the loss in the first epoch of a 200-epoch run.
            config = {"num_classes": classes, "epoch": 0, "total_epochs": 200}
            for name, script in scripts.items():
                leaf = logits.clone().requires_grad_()
                loss = script.compute_loss(leaf, label, config)
                loss.backward()
                prefix = f"settings/{setting}/loss/{name}"
                record[f"{prefix}/value"] = loss.detach()
                record[f"{prefix}/grad"] = cast(Tensor, leaf.grad)
    return record


def upstream_record(
    scripts: dict[str, Upstream],
    lockstep: dict[str, Lockstep],
) -> dict[str, Tensor]:
    """Return the script's value of every quantity the default test replays.

    Args:
      scripts: The script carrying each loss, keyed as :func:`variants`.
      lockstep: Each loss's :func:`upstream_lockstep`.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.dl.classification_loss.reference.port_record`.

    """
    record: dict[str, Tensor] = {}
    logits, label = reference.fixed_logits()
    with host_agnostic_numerics():
        for name, script in scripts.items():
            leaf = logits.clone().requires_grad_()
            config = {"num_classes": 4, "epoch": 0, "total_epochs": 1}
            loss = script.compute_loss(leaf, label, config)
            loss.backward()
            record[f"loss/{name}/value"] = loss.detach()
            record[f"loss/{name}/grad"] = cast(Tensor, leaf.grad)
    for values, _ in lockstep.values():
        record |= values
    return record


def _sgd(model: nn.Module, trace: MainTrace) -> torch.optim.SGD:
    """Build the optimizer ``main`` builds, from the settings it was traced with."""
    return torch.optim.SGD(
        model.parameters(),
        lr=trace.settings["lr"],
        momentum=trace.settings["momentum"],
        dampening=trace.settings["dampening"],
        weight_decay=trace.settings["weight_decay"],
        nesterov=bool(trace.settings["nesterov"]),
    )


class _LossSpy:
    """Forwards to the reference's loss and keeps what it saw."""

    def __init__(self, script: Upstream) -> None:
        self.compute_loss = script.compute_loss
        self.losses: list[Tensor] = []
        self.logits: list[Tensor] = []
        self.configs: list[dict[str, int]] = []

    def __call__(
        self,
        logits: Tensor,
        targets: Tensor,
        config: dict[str, int],
    ) -> Tensor:
        """Return the reference's loss for the batch."""
        loss = self.compute_loss(logits, targets, config)
        self.losses.append(loss.detach())
        self.logits.append(logits.detach())
        self.configs.append(dict(config))
        return loss
