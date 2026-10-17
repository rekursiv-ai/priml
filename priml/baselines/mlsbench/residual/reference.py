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

from priml.baselines.mlsbench.pytorch_vision import reference
from priml.baselines.mlsbench.pytorch_vision.model import BasicBlock, ResNet
from priml.baselines.mlsbench.pytorch_vision.train_step import VisionTrainStep
from priml.baselines.mlsbench.residual.blocks import (
    PreActivationBlock,
    StochasticDepthBlock,
    stochastic_depth,
)
from priml.baselines.mlsbench.residual.experiments import (
    exp000,
    exp001,
    exp002,
    exp003,
)
from priml.baselines.mlsbench.residual.upstream import BLOCKS, FEATURES, block_run
from priml.testing.bfb import host_agnostic_numerics
from priml.testing.golden import rng_fingerprint
from priml.train.parallelism import NoParallel


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import nn

    from priml.baselines.mlsbench.pytorch_vision.experiments import VisionTrainLoop


VARIANTS: Final[dict[str, Callable[[], VisionTrainLoop]]] = {
    "basic": exp000,
    "pre_activation": exp001,
    "gated_residual": exp002,
    "stochastic_depth": exp003,
}
"""The experiment that carries each block, keyed as the script's variants."""


def resized(
    factory: Callable[[], VisionTrainLoop],
    *,
    blocks: tuple[int, ...],
) -> VisionTrainStep.Config:
    """Return the step of ``factory`` with ``blocks`` blocks per stage on the CPU.

    A per-position block list is rebuilt for the new depth with the same
    survival schedule, since its length and depths follow the block count.

    Args:
      factory: The experiment whose recipe is kept.
      blocks: Blocks per stage; the widths stay the reference's.

    Returns:
      config: The step's config, unbuilt.

    """
    config = factory().step
    config.parallelism = NoParallel.Config(device="cpu")
    model = config.model
    assert isinstance(model, ResNet.Config)
    model.blocks_per_stage = blocks
    if isinstance(model.block, list):
        first = model.block[0]
        assert isinstance(first, StochasticDepthBlock.Config)
        model.block = stochastic_depth(
            model.channels_hidden,
            blocks,
            survival_last=first.survival_last,
        )
    return config


def block_of(name: str, *, channels_out: int, stride: int, depth: int) -> nn.Module:
    """Build the block of experiment :data:`VARIANTS` ``[name]`` alone.

    Args:
      name: A key of :data:`VARIANTS`.
      channels_out: Output channels; the input has ``FEATURES[1]``.
      stride: Stride of the block.
      depth: Position, and deepest position, on a survival schedule: a block
        built alone is the only one its schedule counts.

    Returns:
      block: Built from the experiment's block config.

    """
    model = VARIANTS[name]().step.model
    assert isinstance(model, ResNet.Config)
    template = model.block[0] if isinstance(model.block, list) else model.block
    config = copy_tree(template)
    assert isinstance(config, BasicBlock.Config | PreActivationBlock.Config)
    config.channels_in = FEATURES[1]
    config.channels_out = channels_out
    config.stride = stride
    if isinstance(config, StochasticDepthBlock.Config):
        config.depth = config.num_layers = depth
    return config.make()


def block_record() -> dict[str, Tensor]:
    """Run every block alone, identity then projected, as the script's are run.

    Returns:
      record: Keyed as :func:`~priml.baselines.mlsbench.residual.upstream.upstream_blocks`.

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
                    depth=1,
                ),
                f"block/{name}/{kind}",
            )
    return record


def draws_record() -> dict[str, Tensor]:
    """Return where the generator stands after building each full network.

    Returns:
      record: ``draws/<name>``, the next eight draws after ResNet-56 with
        that block is built from the task's seed.

    """
    record: dict[str, Tensor] = {}
    for name in VARIANTS:
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
) -> dict[str, Tensor]:
    """Train the shallowest network for two updates, one per epoch.

    Args:
      name: A key of :data:`VARIANTS`.
      states: When given, receives a copy of the model's whole state after
        each update.

    Returns:
      record: Per update the loss, the logits it came from and the logits of
        a fixed probe in evaluation mode afterwards.

    """
    losses: list[Tensor] = []
    logits: list[Tensor] = []
    probes: list[Tensor] = []
    config = resized(VARIANTS[name], blocks=(1, 1, 1))
    assert isinstance(config.model, ResNet.Config)
    config.model.channels_out = reference.CLASSES
    config.train_budget_epochs = reference.EPOCHS
    with reference.seeded(reference.SEED), host_agnostic_numerics():
        step = config.make()
        assert isinstance(step, VisionTrainStep)
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
      record: The union of the records above and the shared step's schedule,
        optimizer and best accuracy, keyed as the stored golden is.

    """
    record = (
        block_record()
        | reference.schedule_record()
        | reference.optimizer_record()
        | reference.best_record()
        | draws_record()
    )
    for name in VARIANTS:
        record |= lockstep_record(name)
    return record
