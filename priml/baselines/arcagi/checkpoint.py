"""Load evaluation weights into the ARC-AGI training step."""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

from configgle.custom_types import Makeable
from numpy._core.multiarray import _reconstruct
from torch import Tensor
from treekle import from_plain

import numpy as np
import torch

from priml.baselines.arcagi.train_step import PooledActTrainStep
from priml.model.ae.checkpoint import CheckpointFile, LocalFile
from priml.model.grid_solver import GridSolver
from priml.train.custom_types import EMAProtocol
from priml.train.ema import EMA


class CheckpointActTrainStep(PooledActTrainStep):
    """Initialize ARC evaluation from saved weights with a fresh optimizer."""

    class Config(PooledActTrainStep.Config):
        """ARC recipe plus the weights file to evaluate."""

        if TYPE_CHECKING:

            @override
            def make(self) -> "CheckpointActTrainStep":
                """Build this nested configuration's step."""
                ...

        checkpoint: Makeable[CheckpointFile] = field(default_factory=LocalFile.Config)
        """Native or released checkpoint; resolved before evaluation starts."""

    def __init__(self, config: Config) -> None:
        checkpoint_path = config.checkpoint.make().path()
        super().__init__(config)
        state = read_step_state(checkpoint_path, model=self.net, ema=self.ema)
        self.load_state_dict(state, load_optimizer=False)
        if "ema" in state:
            self.ema.load_state_dict(from_plain(state["ema"], dict[str, object]))


def read_step_state(
    path: Path,
    *,
    model: GridSolver,
    ema: EMAProtocol,
) -> dict[str, object]:
    """Map archived tensors and progress to the native step/EMA state schema.

    Args:
      path: A native ARC checkpoint or an original released checkpoint.
      model: Target architecture defining the complete state contract.
      ema: Native averaging implementation that will restore the shadow.

    Returns:
      state: Model, progress, and EMA state; optimizer state is excluded.

    """
    # Released training archives include a NumPy RNG array. Keep its allowlist
    # scoped to this read. Torch removes every entry the context registers, so
    # register only additions or a caller's existing entries would be removed.
    existing = set(torch.serialization.get_safe_globals())
    with torch.serialization.safe_globals(
        [
            allowed
            for allowed in (
                _reconstruct,
                np.ndarray,
                np.dtype,
                np.dtypes.UInt32DType,
                # The native mapping subclass has no custom behavior. Rebuild
                # it as its builtin base, which weights_only accepts SETITEMS on.
                (OrderedDict, f"{EMA.__module__}._StateDict"),
            )
            if allowed not in existing
        ],
    ):
        state = from_plain(
            cast(
                "object",
                torch.load(path, map_location="cpu", weights_only=True, mmap=True),
            ),
            dict[str, object],
        )
    payload = (
        state
        if "model" in state or "model_state_dict" in state
        else from_plain(state["step"], dict[str, object])
    )
    released = "model_state_dict" in payload
    live = from_plain(
        payload["model_state_dict" if released else "model"],
        dict[str, Tensor],
    )
    mapped = _map_state(live, expected=model.state_dict(), label="model")
    shadow = from_plain(
        payload.get("ema_state" if released else "ema", {}),
        dict[str, object],
    )
    progress = from_plain(
        payload["step"]
        if released
        else payload.get("global_step", shadow.get("global_step", 0)),
        int,
    )
    result: dict[str, object] = {
        "model": mapped,
        "timer_step": payload.get(
            "timer_step",
            {"global_count": progress, "global_sec": 0.0},
        ),
    }
    if "shadow_model" in shadow:
        shadow["shadow_model"] = _map_state(
            from_plain(shadow["shadow_model"], dict[str, Tensor]),
            expected=model.state_dict(),
            label="EMA",
        )
    elif "shadow_params" in shadow:
        shadow["shadow_params"] = _map_state(
            from_plain(shadow["shadow_params"], dict[str, Tensor]),
            expected=dict(model.named_parameters()),
            label="EMA",
            complete=False,
        )
    elif released or any(isinstance(value, Tensor) for value in shadow.values()):
        parameters = _map_state(
            from_plain(shadow, dict[str, Tensor]),
            expected=dict(model.named_parameters()),
            label="EMA",
        )
        shadow = dict[str, object](global_step=progress)
        if isinstance(ema, EMA) and ema.shadow_kind == "module":
            shadow["shadow_model"] = {**mapped, **parameters}
        else:
            shadow["shadow_params"] = parameters
    if shadow:
        shadow.setdefault("global_step", progress)
        result["ema"] = shadow
    return result


def _map_state(
    source: Mapping[str, Tensor],
    *,
    expected: Mapping[str, Tensor],
    label: str,
    complete: bool = True,
) -> dict[str, Tensor]:
    mapped: dict[str, Tensor] = {}
    for name, value in source.items():
        target = _target_key(name)
        if target in mapped:
            raise ValueError(f"{label} checkpoint has duplicate key {target!r}.")
        if target not in expected:
            raise ValueError(f"{label} checkpoint has unexpected key {name!r}.")
        if value.shape != expected[target].shape:
            raise ValueError(
                f"{label} checkpoint shape mismatch for {name!r}: "
                f"{tuple(value.shape)} != {tuple(expected[target].shape)}.",
            )
        mapped[target] = value
    missing = expected.keys() - mapped.keys()
    if complete and missing:
        raise ValueError(f"{label} checkpoint is missing keys: {sorted(missing)}.")
    return mapped


def _target_key(name: str) -> str:
    if name.startswith(("_orig_mod.model.inner.", "model.inner.")):
        return _released_key(
            name.removeprefix("_orig_mod.").removeprefix("model.inner."),
        )
    for old, new in (
        ("embed_tokens.", "embedding.embed_tokens."),
        ("embed_feedback", "embedding.channels.0.embed_feedback"),
        ("q_head.", "halt_head."),
        ("puzzle_emb.weights", "prefix.weights"),
        ("register_tokens", "prefix.register_tokens"),
    ):
        if name.startswith(old):
            return new + name.removeprefix(old)
    return name


def _released_key(name: str) -> str:
    direct = {
        "embed_tokens.embedding_weight": "embedding.embed_tokens.weight",
        "lm_head.weight": "head.weight",
        "q_head.weight": "halt_head.weight",
        "q_head.bias": "halt_head.bias",
        "init_hidden": "slow_init",
    }
    if name in direct:
        return direct[name]
    if name.startswith("layers."):
        key = "reasoning." + name.removeprefix("layers.")
        for old, new in (
            (".self_attn.qkv_proj.", ".attn.proj_qkv."),
            (".self_attn.o_proj.", ".attn.proj_out."),
            (".mlp.gate_up_proj.", ".ffn.up_proj."),
            (".mlp.dwconv.", ".ffn.middle.conv."),
            (".mlp.down_proj.", ".ffn.down_proj."),
        ):
            key = key.replace(old, new)
        return key
    composition = {
        "puzzle_embed.weight": "category_embed.weight",
        "category_embed.weight": "category_embed.weight",
        "dih_embed.weight": "augmentation_embed.weight",
        "augmentation_embed.weight": "augmentation_embed.weight",
        "colorperm_base": "slot_base",
        "colorperm_slots": "slot_embeddings",
        "slot_base": "slot_base",
        "slot_embeddings": "slot_embeddings",
        "film_gamma.weight": "film_gamma.weight",
        "film_gamma.bias": "film_gamma.bias",
        "film_beta.weight": "film_beta.weight",
        "film_beta.bias": "film_beta.bias",
    }
    suffix = name.removeprefix("puzzle_emb.")
    if name.startswith("puzzle_emb.") and suffix in composition:
        return "prefix.composition." + composition[suffix]
    for old, new in (
        ("puzzle_emb.instance_residual_embed.weights", "prefix.residual.weights"),
        ("puzzle_emb.instance_residual_up_proj.", "prefix.residual_projection."),
        ("puzzle_emb.interaction_gate.", "prefix.residual_gate.linear."),
    ):
        if name.startswith(old):
            return new + name.removeprefix(old)
    return name
