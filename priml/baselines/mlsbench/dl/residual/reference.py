"""Port-side runs the ``dl-residual-connection`` capture and its test share.

``scripts/capture_reference.py`` runs the pristine script and these runs on the
same inputs under the same pins, compares them bit for bit, and stores the
script's values under ``testdata/``; ``reference_test.py`` replays these runs
against them. The data, loss, schedule and optimizer are the
``cv-classification-loss`` port's, already proven there; this task adds only
the block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import functools

from configgle.walk import copy_tree
from torch import Tensor

import torch

from priml.baselines.mlsbench.dl.residual.blocks import (
    PreActivationBlock,
    StochasticDepthBlock,
)
from priml.baselines.mlsbench.dl.residual.experiments import (
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
from priml.baselines.mlsbench.dl.residual.upstream import (
    BLOCKS,
    FEATURES,
    block_run,
)
from priml.baselines.mlsbench.dl.vision import reference
from priml.baselines.mlsbench.dl.vision.model import (
    BasicBlock,
    ResNet,
    sized_blocks,
)
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from torch import nn

    from priml.baselines.mlsbench.dl.vision.experiments import VisionTrainLoop


VARIANTS: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "basic": exp000,
    "pre_activation": exp001,
    "gated_residual": exp002,
    "stochastic_depth": exp003,
}
"""The experiment that carries each block, keyed as the script's variants."""

SETTINGS: Final[dict[str, dict[str, Callable[[], VisionTrainLoop]]]] = {
    "resnet20-cifar10": {
        "basic": exp004,
        "pre_activation": exp005,
        "gated_residual": exp006,
        "stochastic_depth": exp007,
    },
    "resnet110-cifar100": {
        "basic": exp008,
        "pre_activation": exp009,
        "gated_residual": exp010,
        "stochastic_depth": exp011,
    },
}
"""Per other setting of the task, the experiment that carries each block there."""


def block_of(name: str, *, channels_out: int, stride: int) -> nn.Module:
    """Build the block of experiment :data:`VARIANTS` ``[name]`` alone.

    A stochastic-depth template built alone keeps its defaults, position 1 of
    1, as upstream's counter stands for a block built alone.

    Args:
      name: A key of :data:`VARIANTS`.
      channels_out: Output channels; the input has ``FEATURES[1]``.
      stride: Stride of the block.

    Returns:
      block: Built from the experiment's block config.

    """
    model = VARIANTS[name]().step.model
    assert isinstance(model, ResNet.Config)
    config = copy_tree(model.block)
    assert isinstance(config, BasicBlock.Config | PreActivationBlock.Config)
    config.channels_in = FEATURES[1]
    config.channels_out = channels_out
    config.stride = stride
    return config.make()


def block_record() -> dict[str, Tensor]:
    """Run every block alone, identity then projected, as the script's are run.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.dl.residual.upstream.upstream_blocks`.

    """
    record: dict[str, Tensor] = {}
    for name in VARIANTS:
        for kind, channels_out, stride in BLOCKS:
            record |= block_run(
                functools.partial(
                    block_of,
                    name,
                    channels_out=channels_out,
                    stride=stride,
                ),
                f"block/{name}/{kind}",
            )
    return record


def draws_record(names: Iterable[str]) -> dict[str, Tensor]:
    """Return where the generator stands after building each full network.

    Args:
      names: Keys of :data:`VARIANTS` to build.

    Returns:
      record: ``draws/<name>``, the next eight draws after ResNet-56 with
        that block is built from the task's seed.

    """
    record: dict[str, Tensor] = {}
    for name in names:
        config = VARIANTS[name]().step
        config.parallelism = NoParallel.Config(device="cpu")
        with reference.seeded(reference.SEED), host_agnostic_numerics():
            _ = config.make()
            record[f"draws/{name}"] = rng_fingerprint()
    return record


def lockstep_record(
    name: str,
    *,
    states: list[dict[str, Tensor]] | None = None,
    setting: str = "",
) -> dict[str, Tensor]:
    """Train the shallowest network for two updates, one per epoch.

    Args:
      name: A key of :data:`VARIANTS`.
      states: When given, receives a copy of the model's whole state after
        each update.
      setting: A key of :data:`SETTINGS` to train that setting's fork with
        the setting's class count; empty for ``exp000``'s chain at
        ``CLASSES`` logits.

    Returns:
      record: Per update the loss, the logits it came from and the logits of
        a fixed probe in evaluation mode afterwards, under
        :func:`lockstep_key`.

    """
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    probes: list[Tensor] = []
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = reference.tiny_step(
            SETTINGS[setting][name] if setting else VARIANTS[name],
            blocks=(1, 1, 1),
            classes=reference.SETTINGS[setting].classes
            if setting
            else reference.CLASSES,
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
    key = lockstep_key(name, setting=setting)
    return {
        f"{key}/loss": torch.stack(losses),
        f"{key}/logits": torch.stack(logits),
        f"{key}/probe": torch.stack(probes),
    }


def lockstep_key(name: str, *, setting: str = "") -> str:
    """Return the prefix of a lockstep's record: ``exp000``'s chain, or a setting's."""
    return f"settings/{setting}/lockstep/{name}" if setting else f"lockstep/{name}"


def port_record() -> dict[str, Tensor]:
    """Run every port-side comparison and return all their values.

    Returns:
      record: The union of the records above and the shared step's schedule,
        optimizer and best accuracy, keyed as the stored golden is.

    """
    record = (
        block_record()
        | reference.schedule_record()
        | reference.optimizer_record()
        | reference.best_record()
        | draws_record(VARIANTS)
    )
    for name in VARIANTS:
        record |= lockstep_record(name)
    return record


def positions_record(setting: str) -> dict[str, Tensor]:
    """Return each stochastic-depth block's position at a setting, then the count.

    Args:
      setting: A key of :data:`SETTINGS`.

    Returns:
      record: ``settings/<setting>/stochastic_depth/positions``, every block's
        ``depth`` in network order followed by the shared ``num_layers``.

    """
    model = SETTINGS[setting]["stochastic_depth"]().step.model
    assert isinstance(model, ResNet.Config)
    blocks = [
        block
        for stage in sized_blocks(model)
        for block, _ in stage
        if isinstance(block, StochasticDepthBlock.Config)
    ]
    return {
        f"settings/{setting}/stochastic_depth/positions": torch.tensor(
            [b.depth for b in blocks] + [blocks[-1].num_layers],
        ),
    }


def settings_record(directory: Path) -> dict[str, Tensor]:
    """Run every other setting's comparisons and return their values.

    Args:
      directory: Scratch directory the loader run writes its split to.

    Returns:
      record: The shared per-variant networks and the CIFAR-10 loader, and
        each setting's stochastic-depth positions; keyed as the golden is.

    """
    record = reference.task_settings_record(SETTINGS, directory)
    for setting, variants in SETTINGS.items():
        record |= positions_record(setting)
        for name in variants:
            record |= lockstep_record(name, setting=setting)
    return record
