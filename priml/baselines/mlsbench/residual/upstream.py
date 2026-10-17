"""Runs of the pristine ``dl-residual-connection`` script, under the capture's pins.

The script is imported from a clone of the benchmark, not vendored. Its
``main`` is traced as the ``cv-classification-loss`` capture traces that
task's; each variant's ``CustomBlock`` runs alone on fixed features; and the
lockstep run repeats ``main``'s epoch loop on the shallowest network. Every
``torch.rand`` draw of a block run is recorded, so the capture can show that
stochastic depth both kept and dropped its branch. Nothing runs on a GPU.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast
from unittest import mock

import functools

import torch

from priml.baselines.mlsbench.pytorch_vision import reference, upstream
from priml.testing.bfb import host_agnostic_numerics


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    from torch import Tensor, nn

    from priml.baselines.mlsbench.pytorch_vision.upstream import (
        Lockstep,
        MainTrace,
        Upstream,
    )

PRISTINE: Final = Path(
    "harbor/tasks-docker/mls-bench__dl-residual-connection/tests/meta/pristine"
    "/pytorch-vision/custom_residual.py",
)
"""The unedited training script, relative to the clone."""

EDITS: Final = Path("tasks/dl-residual-connection/edits")
"""Directory of the task's baseline edits, relative to the clone."""

EDITED: Final = ("pre_activation", "gated_residual", "stochastic_depth")
"""The published baselines, each named as its edit file and as its port's variant."""

FEATURES: Final = (2, 3, 4, 5)
"""Batch, channels, height and width of the features each block is run on."""

BLOCKS: Final = (("identity", 3, 1), ("projected", 4, 2))
"""Each standalone block run: its name, output channels and stride."""

FORWARDS: Final = 4
"""Training forwards of each standalone block, so a random block shows both outcomes."""


class Script(Protocol):
    """The part of the ``dl-residual-connection`` script the capture uses."""

    CustomBlock: Callable[[int, int, int], nn.Module]

    ResNet: Callable[[Callable[[int, int, int], nn.Module], list[int], int], nn.Module]

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

    Each imported module also gets ``BasicBlock`` naming its ``CustomBlock``,
    the name the ``cv-classification-loss`` capture builds its network with;
    only the in-memory module changes, never the clone.

    Args:
      root: Clone of the benchmark.
      scratch: Directory the edited scripts are written to.

    Returns:
      scripts: ``basic`` (the pristine script) and each of :data:`EDITED`.

    """
    scripts = {"basic": upstream.load_script(root / PRISTINE)} | {
        name: upstream.edited_script(
            root,
            name,
            scratch,
            pristine=PRISTINE,
            edits=EDITS,
        )
        for name in EDITED
    }
    for script in scripts.values():
        script.BasicBlock = cast(
            "type[nn.Module]",
            cast(Script, cast(object, script)).CustomBlock,
        )
    return scripts


def fixed_features() -> Tensor:
    """Return features of shape :data:`FEATURES`: multiples of one eighth, some negative.

    Returns:
      features: Float32 values.

    """
    count = FEATURES[0] * FEATURES[1] * FEATURES[2] * FEATURES[3]
    return ((torch.arange(count) * 7 % 23 - 11) / 8).reshape(FEATURES)


@contextmanager
def drawn() -> Generator[list[Tensor]]:
    """Record every ``torch.rand`` result drawn inside the block.

    Yields:
      draws: Filled in call order.

    """
    draws: list[Tensor] = []
    rand = torch.rand

    def recorded(*args: object, **kwargs: object) -> Tensor:
        value = cast("Callable[..., Tensor]", rand)(*args, **kwargs)
        draws.append(value.detach().clone().flatten())
        return value

    with mock.patch.object(torch, "rand", recorded):
        yield draws


def block_run(make: Callable[[], nn.Module], key: str) -> dict[str, Tensor]:
    """Build a block from the task's seed and run it on the fixed features.

    The block takes :data:`FORWARDS` training forwards, each summed against a
    fixed cotangent and backpropagated, then one evaluation forward.

    Args:
      make: Builds the block; called after seeding, so its initialization is
        drawn from the task's seed.
      key: Prefix of the record's keys.

    Returns:
      record: The training outputs stacked, the features' and the parameters'
        accumulated gradients (a parameter no forward reached counts zero),
        the evaluation output, and every ``torch.rand`` draw.

    """
    with (
        reference.seeded(reference.SEED),
        host_agnostic_numerics(),
        drawn() as draws,
    ):
        block = make()
        leaf = fixed_features().requires_grad_()
        block.train()
        values: list[Tensor] = []
        for index in range(FORWARDS):
            value = cast(torch.Tensor, block(leaf))
            ((value * (index + 1)) / 8).sum().backward()
            values.append(value.detach())
        block.eval()
        with torch.no_grad():
            evaluated = cast(torch.Tensor, block(leaf))
    grads = [
        torch.zeros_like(p).flatten() if p.grad is None else p.grad.flatten()
        for p in block.parameters()
    ]
    return {
        f"{key}/train": torch.stack(values),
        f"{key}/grad": cast(torch.Tensor, leaf.grad),
        f"{key}/param_grad": torch.cat(grads),
        f"{key}/eval": evaluated,
        f"{key}/draws": torch.cat(draws) if draws else torch.empty(0),
    }


def upstream_blocks(script: Upstream, name: str) -> dict[str, Tensor]:
    """Run the script's ``CustomBlock`` alone, identity and projected, in that order.

    Args:
      script: The script carrying the block under test.
      name: Which block, for the record's keys.

    Returns:
      record: :func:`block_run` of each of :data:`BLOCKS`.

    """
    task = cast(Script, cast(object, script))
    record: dict[str, Tensor] = {}
    for kind, channels_out, stride in BLOCKS:
        with _fresh_counter(task.CustomBlock):
            record |= block_run(
                functools.partial(task.CustomBlock, FEATURES[1], channels_out, stride),
                f"block/{name}/{kind}",
            )
    return record


# The stochastic-depth block numbers itself from a class-wide counter that only
# a 16-to-16-channel block resets, so a block built alone would otherwise take
# its survival from every block built before it in this process.
@contextmanager
def _fresh_counter(block: object) -> Generator[None]:
    """Start a block class's ``_block_counter`` at zero for the block, if it has one."""
    if hasattr(block, "_block_counter"):
        with mock.patch.object(block, "_block_counter", 0):
            yield
    else:
        yield


def upstream_lockstep(script: Upstream, name: str, trace: MainTrace) -> Lockstep:
    """Repeat ``main``'s epoch loop on the shallowest network for two epochs.

    Args:
      script: The script carrying the block under test.
      name: Which block, for the record's keys.
      trace: Source of the optimizer settings ``main`` used.

    Returns:
      lockstep: Per update the loss, its logits and a probe's logits
        afterwards; and the model's whole state after each update.

    """
    task = cast(Script, cast(object, script))
    criterion = _CriterionSpy()
    probes: list[Tensor] = []
    states: list[dict[str, Tensor]] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        model = task.ResNet(task.CustomBlock, [1, 1, 1], reference.CLASSES)
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
    """Return the reference's value of every quantity the default test replays.

    Args:
      scripts: Each variant's script, keyed as :func:`variants`.
      traces: A run of each variant's ``main``, keyed alike.
      lockstep: Each variant's :func:`upstream_lockstep`.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.residual.reference.port_record`.

    """
    record: dict[str, Tensor] = {}
    for name, script in scripts.items():
        record |= upstream_blocks(script, name)
    pristine = traces["basic"]
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
