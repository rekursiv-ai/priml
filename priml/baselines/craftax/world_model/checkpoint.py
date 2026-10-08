"""Rebuild a trained world model from its experiment factory and a checkpoint."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from configgle import Makeable, traverse
from configgle.cli_override import apply_overrides
from configgle.launch import resolve_config

import torch

from priml.baselines.craftax.world_model.model import WorldModel
from priml.baselines.craftax.world_model.train_step import (
    WorldModelTrainStep,
)
from priml.lib.codec import from_plain
from priml.model.attention.flash4 import Flash4Varlen
from priml.model.attention.kernel import SdpaVarlen


def build_world_model(
    experiment: str,
    *,
    overrides: Sequence[str] = (),
) -> tuple[WorldModel, Makeable[object]]:
    """Build an experiment's world model, initialized from the global generator.

    Every FlashAttention 4 kernel is built as ``SdpaVarlen``: a kernel holds no
    weights, and FA4 runs only on Linux CUDA, so the model builds anywhere.

    Args:
      experiment: Dotted path of the experiment factory: a ``TrainLoop.Config``
        whose ``WorldModelTrainStep`` trains a ``WorldModel``.
      overrides: ``PATH=VALUE`` config overrides the run was launched with.

    Returns:
      model: The model as its config builds it, float32 on the CPU.
      config: The experiment's finalized config, overrides applied.

    Raises:
      TypeError: The experiment does not train a ``WorldModel``.

    """
    config = resolve_config(experiment)
    apply_overrides(config, list(overrides))
    config = config.copy_tree().finalize()
    step: object = getattr(config, "step", None)
    if not isinstance(step, WorldModelTrainStep.Config) or not isinstance(
        step.model,
        WorldModel.Config,
    ):
        raise TypeError(f"{experiment} does not train a WorldModel.")
    for match in traverse(step.model, Flash4Varlen.Config):
        match.replace(SdpaVarlen.Config())
    return step.model.make(), config


def load_world_model(
    experiment: str,
    checkpoint: Path,
    *,
    overrides: Sequence[str] = (),
) -> tuple[WorldModel, Makeable[object]]:
    """Build an experiment's world model and load a training checkpoint into it.

    Args:
      experiment: Dotted path of the experiment factory that trained it; see
        ``build_world_model``.
      checkpoint: A ``TrainLoop`` checkpoint, the model under ``step``/``model``.
      overrides: ``PATH=VALUE`` config overrides the run was launched with.

    Returns:
      model: The model in eval mode, float32 on the CPU.
      config: The experiment's finalized config, overrides applied.

    """
    model, config = build_world_model(experiment, overrides=overrides)
    model.load_state_dict(model_state(checkpoint))
    return model.eval(), config


def model_state(checkpoint: Path) -> dict[str, object]:
    """Return the model's state dict of a ``TrainLoop`` checkpoint, on the CPU.

    Args:
      checkpoint: A ``TrainLoop`` checkpoint, the model under ``step``/``model``.

    Returns:
      state: The model's tensors by name, memory-mapped.

    """
    # ``weights_only`` admits only tensors and plain containers, which is all
    # ``TrainLoop.state_dict`` holds; ``mmap`` leaves the optimizer state unread.
    state = from_plain(
        cast(
            "object",
            torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True),
        ),
        dict[str, object],
    )
    return from_plain(
        from_plain(state["step"], dict[str, object])["model"],
        dict[str, object],
    )
