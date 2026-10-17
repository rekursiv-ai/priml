"""Strict source checkpoint key mapping for the native CHARM model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import re


if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from priml.model.grid_solver import SingleLatentGridSolver


def remap_charm_checkpoint(
    live_state: Mapping[str, Tensor],
    *,
    model: SingleLatentGridSolver,
    ema_state: Mapping[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Map a complete released source state to native keys and layouts.

    Only the released compiled ``_orig_mod.model.inner`` and source
    ``model.inner`` wrappers are stripped.
    Source EMA state is overlaid for dense parameters; live state supplies
    buffers such as the sparse instance table, which EMA does not contain.
    ``ema_state=None`` means no EMA overlay; an explicit empty mapping is an
    incomplete EMA checkpoint and fails the strict schema check.
    Every input key, target key, and tensor shape is checked explicitly.

    Args:
      live_state: Complete live source model state, including buffers.
      model: Native model that defines the exact expected key/shape contract.
      ema_state: Optional complete source EMA parameter mapping.

    Returns:
      state: Native state using live buffers and EMA dense parameters.

    """
    target_state = model.state_dict()
    live = _map_state(live_state, expected=target_state, label="live")
    if ema_state is None:
        return live

    parameters = dict(model.named_parameters())
    ema = _map_state(ema_state, expected=parameters, label="EMA")
    result = dict(live)
    result.update(ema)
    return result


def _map_state(
    source_state: Mapping[str, Tensor],
    *,
    expected: Mapping[str, Tensor],
    label: str,
) -> dict[str, Tensor]:
    mapped: dict[str, Tensor] = {}
    for source_key, value in source_state.items():
        target_key = _target_key(source_key)
        if target_key in mapped:
            raise ValueError(f"{label} checkpoint maps duplicate key {target_key!r}.")
        if target_key not in expected:
            raise ValueError(f"{label} checkpoint has unexpected key {source_key!r}.")
        if value.shape != expected[target_key].shape:
            raise ValueError(
                f"{label} checkpoint shape mismatch for {source_key!r}: "
                f"got {tuple(value.shape)}, expected {tuple(expected[target_key].shape)}.",
            )
        mapped[target_key] = value

    missing = expected.keys() - mapped.keys()
    if missing:
        raise ValueError(f"{label} checkpoint is missing keys: {sorted(missing)}.")
    return mapped


def _target_key(source_key: str) -> str:
    released_prefixes = ("_orig_mod.model.inner.", "model.inner.")
    direct_keys = {
        "embed_tokens.embedding_weight": "embedding.embed_tokens.weight",
        "lm_head.weight": "head.weight",
        "q_head.weight": "halt_head.weight",
        "q_head.bias": "halt_head.bias",
        "init_hidden": "slow_init",
    }
    composition_keys = {
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
    block_children = {"self_attn": "attn", "mlp": "ffn"}
    key = source_key
    prefix = next(
        (value for value in released_prefixes if key.startswith(value)),
        None,
    )
    if prefix is None:
        raise ValueError(f"unexpected source model key {source_key!r}.")
    key = key.removeprefix(prefix)

    direct = direct_keys.get(key)
    if direct is not None:
        return direct

    if key.startswith("layers."):
        match = re.fullmatch(r"layers\.(\d+)\.(self_attn|mlp)\.(.+)", key)
        if match is None:
            raise ValueError(f"unexpected source model key {source_key!r}.")
        layer, child, suffix = match.groups()
        target_child = block_children[child]
        if child == "self_attn":
            projections = {
                "qkv_proj.weight": "proj_qkv.weight",
                "o_proj.weight": "proj_out.weight",
            }
        else:
            projections = {
                "gate_up_proj.weight": "up_proj.weight",
                "dwconv.weight": "middle.conv.weight",
                "dwconv.bias": "middle.conv.bias",
                "down_proj.weight": "down_proj.weight",
            }
        try:
            suffix = projections[suffix]
        except KeyError:
            raise ValueError(
                f"unexpected source model key {source_key!r}.",
            ) from None
        return f"reasoning.{layer}.{target_child}.{suffix}"

    if key.startswith("puzzle_emb."):
        suffix = key.removeprefix("puzzle_emb.")
        composition = composition_keys.get(suffix)
        if composition is not None:
            return f"prefix.composition.{composition}"
        if suffix.startswith("interaction_gate."):
            return "prefix.residual_gate.linear." + suffix.removeprefix(
                "interaction_gate.",
            )
        if suffix == "instance_residual_embed.weights":
            return "prefix.residual.weights"
        if suffix == "instance_residual_up_proj.weight":
            return "prefix.residual_projection.weight"
        raise ValueError(f"unexpected source model key {source_key!r}.")

    raise ValueError(f"unexpected source model key {source_key!r}.")
