"""Inputs and port-side runs the ``cv-multitask-loss`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script, each baseline's edit
and these runs on the same inputs under the same pins, compares them bit for
bit, and stores the script's values under ``testdata/``; ``reference_test.py``
replays these runs against them. The pixels, shuffles and augmentations are
the ``cv-classification-loss`` port's, already proven there; this task adds
the second head, the superclass labels, the combination of the two losses and
the optimizer over the network and the combination.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final, cast, override

from torch import Tensor, nn
from treekle import from_plain

import torch

from priml.baselines.mlsbench.dl.multitask.data import (
    COARSE_OF_FINE,
    NUM_COARSE,
    MultiTaskCifar100Data,
)
from priml.baselines.mlsbench.dl.multitask.experiments import (
    MultiTaskTrainLoop,
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.mlsbench.dl.multitask.model import TwoHeadResNet
from priml.baselines.mlsbench.dl.multitask.train_step import MultiTaskStep
from priml.baselines.mlsbench.dl.vision import reference as vision
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import joined, rng_fingerprint
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from pathlib import Path

COMBINATIONS: Final[dict[str, Callable[[], MultiTaskTrainLoop]]] = {
    "sum": exp000,
    "uncertainty": exp001,
    "dwa": exp002,
    "pcgrad": exp003,
}
"""The experiment that carries each combination, keyed as the script's variants."""

CALL_EPOCHS: Final = (0, 1, 1)
"""Epoch of each fixed call of a combination: one in the first epoch, two after."""

type BeforeFirst = Callable[[Tensor], object]
"""Runs on the fine loss just before a combination's first call."""

type Batches = list[list[dict[str, Tensor]]]
"""Per pass, the batches of one loader in order."""


class CombinationSpy(nn.Module):
    """Forwards to a combination and keeps what it was given and returned."""

    def __init__(self, inner: nn.Module, *, before_first: BeforeFirst | None) -> None:
        super().__init__()
        self.inner = inner
        self.before_first = before_first
        self.losses_fine: list[Tensor] = []
        self.losses_coarse: list[Tensor] = []
        self.losses: list[Tensor] = []
        self.epochs: list[list[int]] = []

    @override
    def forward(
        self,
        loss_fine: Tensor,
        loss_coarse: Tensor,
        epoch: int,
        total_epochs: int,
    ) -> Tensor:
        """Record the call and return the combination's loss."""
        if self.before_first is not None and not self.losses:
            self.before_first(loss_fine)
        loss = cast(Tensor, self.inner(loss_fine, loss_coarse, epoch, total_epochs))
        self.losses_fine.append(loss_fine.detach())
        self.losses_coarse.append(loss_coarse.detach())
        self.losses.append(loss.detach())
        self.epochs.append([epoch, total_epochs])
        return loss


def coarse_label(step: int) -> Tensor:
    """Return superclass targets for ``step``'s two images, distinct and below 20.

    Args:
      step: Index that shifts the classes.

    Returns:
      label_coarse: One superclass per image of :func:`vision.fixed_media`.

    """
    return torch.tensor([(3 + step) % NUM_COARSE, (11 + step) % NUM_COARSE])


def run_combination(
    module: nn.Module,
    name: str,
    *,
    before_first: BeforeFirst | None = None,
) -> dict[str, Tensor]:
    """Call a combination three times on losses of three fixed leaves.

    ``w`` feeds both losses, ``u`` only the fine one and ``v`` only the
    coarse one. The two losses' gradients with respect to ``w`` agree in the
    first call and conflict in the others, so a projection runs and is
    skipped.

    Args:
      module: A freshly built combination.
      name: Which combination, for the record's keys.
      before_first: Runs on the first call's fine loss before the call.

    Returns:
      record: Per call the returned value and the gradients of ``w``, ``u``,
        ``v`` and the combination's parameters (zeros where none reached).

    """
    w = (torch.arange(6.0).reshape(2, 3) / 4).requires_grad_()
    u = (torch.arange(3.0) / 2 - 1).requires_grad_()
    v = (torch.arange(4.0) / 8).requires_grad_()
    leaves = [w, u, v, *module.parameters()]
    spy = CombinationSpy(module, before_first=before_first)
    grads: list[Tensor] = []
    with host_agnostic_numerics():
        for call, epoch in enumerate(CALL_EPOCHS):
            sign = 1 if call == 0 else -1
            loss_fine = ((w - call / 2) ** 2).sum() + (u * w[0]).sum()
            loss_coarse = (sign * (w - call / 2) ** 2).sum() / 2 + (v**2).sum() / 2
            spy(loss_fine, loss_coarse, epoch, 5).backward()
            grads.append(
                joined(
                    p.grad if p.grad is not None else torch.zeros_like(p)
                    for p in leaves
                ),
            )
            for p in leaves:
                p.grad = None
    return {
        f"combination/{name}/value": torch.stack(spy.losses),
        f"combination/{name}/grad": torch.stack(grads),
    }


def combination_record() -> dict[str, Tensor]:
    """Run every experiment's combination on the fixed calls.

    Returns:
      record: The union of each combination's :func:`run_combination`.

    """
    record: dict[str, Tensor] = {}
    for name, factory in COMBINATIONS.items():
        module = factory().step.mtl_loss.make()
        assert isinstance(module, nn.Module)
        record |= run_combination(module, name)
    return record


def tiny_step(
    factory: Callable[[], MultiTaskTrainLoop],
    *,
    epochs: int,
) -> MultiTaskStep:
    """Build the step of ``factory`` with one block per stage on the CPU.

    The heads keep the reference's 100 and 20 classes, which its network
    hard-codes.

    Args:
      factory: The experiment whose recipe is kept.
      epochs: Schedule horizon.

    Returns:
      step: The experiment's step, shallow.

    """
    config = factory().step
    assert isinstance(config.model, TwoHeadResNet.Config)
    config.model.trunk.blocks_per_stage = (1, 1, 1)
    config.parallelism = NoParallel.Config(device="cpu")
    config.train_budget_epochs = epochs
    step = config.make()
    assert isinstance(step, MultiTaskStep)
    return step


def full_step(factory: Callable[[], MultiTaskTrainLoop]) -> MultiTaskStep:
    """Build the step of ``factory`` at full size on the CPU.

    Args:
      factory: The experiment to build.

    Returns:
      step: Two-head ResNet-56.

    """
    config = factory().step
    config.parallelism = NoParallel.Config(device="cpu")
    step = config.make()
    assert isinstance(step, MultiTaskStep)
    return step


def schedule_record() -> dict[str, Tensor]:
    """Return the learning rate of each of the 200 epochs of ``exp000``.

    Returns:
      record: The 200 rates, as float64.

    """
    step = tiny_step(exp000, epochs=200)
    base_lr = from_plain(cast(object, step.optimizer.param_groups[0]["lr"]), float)
    rates = [
        step.schedule(epoch, base_lr=base_lr, total_epochs=200) for epoch in range(200)
    ]
    return {"schedule/lr": torch.tensor(rates, dtype=torch.float64)}


def optimizer_record() -> dict[str, Tensor]:
    """Return the optimizer settings of ``exp000`` and every experiment's tensor sizes.

    Returns:
      record: The entries of :data:`vision.HYPERPARAMETERS` as float64, and per
        experiment the size of every optimized tensor in the optimizer's order.

    """
    record: dict[str, Tensor] = {}
    for name, factory in COMBINATIONS.items():
        step = full_step(factory)
        groups: list[Mapping[str, object]] = list(step.optimizer.param_groups)
        record[f"optimizer/{name}/numels"] = torch.tensor(
            [p.numel() for g in groups for p in cast(list[Tensor], g["params"])],
        )
        if name == "sum":
            record["optimizer/settings"] = torch.tensor(
                [float(cast(float, groups[0][k])) for k in vision.HYPERPARAMETERS],
                dtype=torch.float64,
            )
    return record


def draws_record() -> dict[str, Tensor]:
    """Return where the generator stands after building each full experiment.

    Returns:
      record: ``draws/<name>``, the next eight draws.

    """
    record: dict[str, Tensor] = {}
    for name, factory in COMBINATIONS.items():
        with vision.seeded(vision.SEED), host_agnostic_numerics():
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
      name: A key of :data:`COMBINATIONS`.
      states: When given, receives a copy of the network's whole state after
        each update.

    Returns:
      record: Keyed as :func:`spy_record`.

    """
    probes: list[Tensor] = []
    with vision.seeded(vision.SEED), host_agnostic_numerics():
        step = tiny_step(COMBINATIONS[name], epochs=vision.EPOCHS)
        inner = step.mtl_loss
        spy = step.mtl_loss = CombinationSpy(inner, before_first=None)
        for index in range(vision.STEPS):
            step.timer_epoch.global_count = index
            _ = step.train_step(
                media=vision.fixed_media(index),
                label=vision.fixed_label(index),
                label_coarse=coarse_label(index),
            )
            probe = step.call_eval(media=vision.fixed_media(vision.STEPS))
            assert isinstance(probe, tuple)
            probes.append(torch.cat(cast(tuple[Tensor, Tensor], probe), dim=-1))
            if states is not None:
                states.append(
                    {k: v.clone() for k, v in step.model.state_dict().items()},
                )
    return spy_record(name, spy, probes, inner)


def spy_record(
    name: str,
    spy: CombinationSpy,
    probes: list[Tensor],
    inner: nn.Module,
) -> dict[str, Tensor]:
    """Key a lockstep run's observations.

    Args:
      name: Which combination.
      spy: The wrapper the run's combination was called through.
      probes: Both heads' probe logits after each update, joined.
      inner: The combination, for its parameters after the run.

    Returns:
      record: Per update both losses, their combination, the epochs given and
        the probe; and the combination's parameters, when it has any.

    """
    record = {
        f"lockstep/{name}/loss_fine": torch.stack(spy.losses_fine),
        f"lockstep/{name}/loss_coarse": torch.stack(spy.losses_coarse),
        f"lockstep/{name}/loss": torch.stack(spy.losses),
        f"lockstep/{name}/epochs": torch.tensor(spy.epochs),
        f"lockstep/{name}/probe": torch.stack(probes),
    }
    params = list(inner.parameters())
    if params:
        record[f"lockstep/{name}/mtl_params"] = joined(params)
    return record


def loader_batches(directory: Path) -> tuple[Batches, Batches]:
    """Run the loaders over the synthetic split, alternating the two splits.

    Args:
      directory: Scratch directory the split is written to.

    Returns:
      passes: Per training pass the batches in order, then per evaluation pass.

    """
    media, label = vision.synthetic_split()
    for split in ("train", "test"):
        torch.save({"media": media, "label": label}, directory / f"{split}.pt")
    config = MultiTaskCifar100Data.Config()
    config.data.working_dir = directory
    config.data.batch_size = config.data.eval_batch_size = vision.LOADER_BATCH
    train: Batches = []
    evaluation: Batches = []
    with vision.seeded(vision.SEED):
        data = config.make()
        for _ in range(vision.LOADER_PASSES):
            train.append(list(data.train_dataloader()))
            evaluation.append(list(data.eval_dataloader()))
    return train, evaluation


def port_record() -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Returns:
      record: The union of the records above, the superclass map and the
        shared best-accuracy record, keyed as the stored golden is.

    """
    record = (
        combination_record()
        | schedule_record()
        | optimizer_record()
        | draws_record()
        | vision.best_record()
        | {"data/coarse_of_fine": torch.tensor(COARSE_OF_FINE)}
    )
    for name in COMBINATIONS:
        record |= lockstep_record(name)
    return record
