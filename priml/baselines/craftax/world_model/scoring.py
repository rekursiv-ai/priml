"""Load and run a trained world model as its training loop scored it.

A score is comparable with the loop's own only under the loop's numerics: the
experiment's attention kernels and its autocast. ``checkpoint.load_world_model``
builds every FlashAttention 4 kernel as SDPA, so that a model loads on any host;
SDPA and FA4 round differently in bfloat16, so a score on the wrong kernel
differs from the logged one in its last digits. On CUDA, ``load_trained``
keeps the kernels the experiment declares and ``autocast`` enters its
``dtype_autocast``, so evaluating a checkpoint reproduces what its run logged;
elsewhere the model scores in float32 on SDPA, as no experiment trains there.
"""

from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

from configgle.cli_override import apply_overrides
from configgle.launch import resolve_config

import torch

from priml.baselines.craftax.world_model.checkpoint import (
    load_world_model,
    model_state,
)
from priml.baselines.craftax.world_model.experiments import (
    WorldModelLoop,
)
from priml.baselines.craftax.world_model.model import WorldModel


def load_trained(
    experiment: str,
    checkpoint: Path,
    *,
    overrides: Sequence[str] = (),
    device: torch.device,
) -> tuple[WorldModel, WorldModelLoop.Config]:
    """Rebuild a trained world model on ``device`` with the kernels it trained on.

    Args:
      experiment: Dotted path of the ``WorldModelLoop`` experiment factory that
        trained it.
      checkpoint: A ``TrainLoop`` checkpoint, the model under ``step``/``model``.
      overrides: ``PATH=VALUE`` config overrides the run was launched with.
      device: Where the model scores.

    Returns:
      model: The model in eval mode, float32 on ``device``; on CUDA with the
        experiment's own kernels, elsewhere with SDPA in place of FA4.
      config: The experiment's finalized config, overrides applied.

    """
    if device.type != "cuda":
        loaded, config = load_world_model(experiment, checkpoint, overrides=overrides)
        assert isinstance(config, WorldModelLoop.Config)
        return loaded.to(device), config
    declared = resolve_config(experiment)
    apply_overrides(declared, list(overrides))
    declared = declared.copy_tree().finalize()
    assert isinstance(declared, WorldModelLoop.Config)
    model = declared.step.model.make()
    assert isinstance(model, WorldModel)
    model.load_state_dict(model_state(checkpoint))
    return model.eval().to(device), declared


def autocast(
    config: WorldModelLoop.Config,
    device: torch.device,
) -> AbstractContextManager[object]:
    """Return the experiment's training autocast on CUDA, else no autocast.

    Args:
      config: The experiment's config; its step names the autocast dtype.
      device: Where the model scores.

    Returns:
      context: ``torch.autocast`` of ``config.step.dtype_autocast`` on CUDA,
        or a null context elsewhere or when the experiment trains without one.

    """
    dtype = config.step.dtype_autocast
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(
        device.type,
        dtype=dtype,
        cache_enabled=config.step.autocast_cache_enabled,
    )
