"""Strict Hugging Face tensor mapping for the Qwen3.5 dense text language model."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

import torch

from priml.model.attention.gated_self_attention import GatedSelfAttention
from priml.model.attention.qwen3_5_delta import Qwen35GatedDeltaNet
from priml.model.custom_types import ChannelsInOutConfig, DeepModelConfig
from priml.model.special import TiedLinear
from priml.model.transformer.block import TransformerBlock


def remap_hf_state_dict(
    state: dict[str, Tensor],
    config: DeepModelConfig,
    *,
    non_text: str = "reject",
) -> dict[str, Tensor]:
    """Map all text parameters, rejecting unexpected keys and malformed shapes.

    Args:
      state: HF text or conditional-generation checkpoint tensors.
      config: Native configuration parsed from the checkpoint's metadata.
      non_text: Reject extra weights by default. ``discard`` permits only
        ``model.visual.*``, ``visual.*``, ``mtp.*``, and ``model.mtp.*``.
        Language-model head weights always belong to the text model.

    Returns:
      mapped: Native parameters, preserving source dtype and device.

    Raises:
      ValueError: Missing, unexpected, or incorrectly shaped weights.
      TypeError: A custom block cannot be mapped from the HF architecture.

    """
    if non_text not in ("reject", "discard"):
        raise ValueError(f"Unsupported non_text policy: {non_text!r}.")
    prefixes = [
        prefix
        for prefix in ("model.language_model.", "model.", "")
        if f"{prefix}embed_tokens.weight" in state
    ]
    if len(prefixes) != 1:
        raise ValueError("Expected exactly one text embedding namespace.")
    prefix = prefixes[0]
    remaining = set(state)
    with torch.device("meta"):
        expected_module = config.make()
    if not isinstance(expected_module, nn.Module):
        raise TypeError("A deep model config must build an nn.Module.")
    expected = expected_module.state_dict()
    finalized = config.copy_tree().finalize()
    assert isinstance(finalized.block, list)
    blocks = finalized.block
    mapped: dict[str, Tensor] = {}
    for target, template in expected.items():
        sources = _sources(target, prefix=prefix, blocks=blocks)
        shape = tuple(template.shape)
        if len(sources) == 2:
            part_shape = (shape[0] // 2, *shape[1:])
            mapped[target] = torch.cat(
                [
                    _take(state, remaining=remaining, name=name, shape=part_shape)
                    for name in sources
                ],
                dim=0,
            )
        else:
            mapped[target] = _take(
                state, remaining=remaining, name=sources[0], shape=shape
            )
    if (
        isinstance(finalized.proj_out, TiedLinear.Config)
        and "lm_head.weight" in remaining
    ):
        head = _take(
            state,
            remaining=remaining,
            name="lm_head.weight",
            shape=tuple(mapped["proj_in.weight"].shape),
        )
        embedding = mapped["proj_in.weight"]
        if head.dtype != embedding.dtype:
            raise ValueError("Tied lm_head.weight must match the embedding dtype.")
        if head.device != embedding.device:
            raise ValueError("Tied lm_head.weight must match the embedding device.")
        if not torch.equal(head, embedding):
            raise ValueError("Tied lm_head.weight differs from the embedding weight.")
    if non_text == "discard":
        remaining = {
            name
            for name in remaining
            if not name.startswith(
                ("model.visual.", "visual.", "mtp.", "model.mtp."),
            )
        }
    if remaining:
        raise ValueError(f"Unexpected checkpoint weights: {sorted(remaining)}.")
    return mapped


def _take(
    state: dict[str, Tensor],
    *,
    remaining: set[str],
    name: str,
    shape: tuple[int, ...],
) -> Tensor:
    if name not in remaining:
        raise ValueError(f"Missing checkpoint weight: {name}.")
    value = state[name]
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}; expected {shape}.")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be an unquantized floating-point tensor.")
    remaining.remove(name)
    return value


def _sources(
    target: str,
    *,
    prefix: str,
    blocks: Sequence[ChannelsInOutConfig],
) -> list[str]:
    if target == "proj_in.weight":
        return [f"{prefix}embed_tokens.weight"]
    if target == "norm.weight":
        return [f"{prefix}norm.weight"]
    if target == "proj_out.weight":
        return ["lm_head.weight"]
    parts = target.split(".")
    if len(parts) < 4 or parts[0] != "blocks":
        raise ValueError(f"Unsupported native checkpoint parameter: {target}.")
    layer = int(parts[1])
    block = blocks[layer]
    if not isinstance(block, TransformerBlock.Config):
        raise TypeError("HF mapping requires native TransformerBlock configurations.")
    base = f"{prefix}layers.{layer}."
    tail = ".".join(parts[2:])
    norms = {
        "norm1.weight": "input_layernorm.weight",
        "norm2.weight": "post_attention_layernorm.weight",
    }
    if tail in norms:
        return [base + norms[tail]]
    if tail == "ffn.up_proj.weight":
        return [base + "mlp.gate_proj.weight", base + "mlp.up_proj.weight"]
    if tail == "ffn.down_proj.weight":
        return [base + "mlp.down_proj.weight"]
    if tail.startswith("attn."):
        name = tail.removeprefix("attn.")
        if isinstance(block.attn, Qwen35GatedDeltaNet.Config):
            projections = {
                "proj_qkv.": "in_proj_qkv.",
                "proj_z.": "in_proj_z.",
                "proj_b.": "in_proj_b.",
                "proj_a.": "in_proj_a.",
                "proj_out.": "out_proj.",
            }
            for native, hf in projections.items():
                if name.startswith(native):
                    return [base + "linear_attn." + hf + name.removeprefix(native)]
            return [base + "linear_attn." + name]
        if isinstance(block.attn, GatedSelfAttention.Config):
            projections = {
                "proj_q.": "q_proj.",
                "proj_k.": "k_proj.",
                "proj_v.": "v_proj.",
                "proj_out.": "o_proj.",
                "norm_q.": "q_norm.",
                "norm_k.": "k_norm.",
            }
            for native, hf in projections.items():
                if name.startswith(native):
                    return [base + "self_attn." + hf + name.removeprefix(native)]
            return [base + "self_attn." + name]
    raise ValueError(f"Unsupported native checkpoint parameter: {target}.")
