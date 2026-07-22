"""Qwen3 dense LM: configgle-native Config + HF weight loader.

Subclasses :class:`CausalLM` per the library idiom
(``Makes[X]`` re-parents ``.make()``). ``Qwen3.Config`` carries the
HF-shaped arch fields; ``finalize()`` wires them into the inherited
``block``/``final_norm``/``channels``/``num_layers``/``lm_head``
slots.

Qwen3 vs. LLaMA:
  - Explicit ``head_dim`` (not ``hidden_size / num_heads``).
  - Per-head QK-norm — independent ``q_norm`` and ``k_norm`` RMSNorms
    (via ``SelfAttention.Config.share_qk_norm=False``).
  - GQA via ``num_key_value_heads``.
  - No bias on attention or MLP projections.
  - RoPE base ``rope_theta=1_000_000``, HF half-split pairing.

Usage::

    from priml.model.qwen3 import Qwen3

    model = Qwen3.Config.from_hf(hf_config).make()   # architecture only
    model = Qwen3.load("/path/to/Qwen3-0.6B")        # + weights from disk
    model = Qwen3.load("Qwen/Qwen3-0.6B")            # HF repo id (downloads)

Only the dense Qwen3 family is handled here; Qwen3-MoE is a follow-up.
"""

from __future__ import annotations

from dataclasses import KW_ONLY
from pathlib import Path
from typing import Any, Self, cast, override

import json

from configgle import Makes
from torch import Tensor

import torch

from priml import hub
from priml.model.attention import SelfAttention
from priml.model.causal_lm import CausalLM
from priml.model.norm import RMSNorm
from priml.model.rope import RoPE
from priml.model.swiglu import SwiGLU
from priml.model.transformer import TransformerBlock


class Qwen3(CausalLM):
    """Qwen3 dense causal LM — ``CausalLM`` pre-wired for the Qwen3 arch."""

    class Config(Makes["Qwen3"], CausalLM.Config, kw_only=False):
        # HF-shaped positional required fields match config.json keys.
        vocab_size: int = -1

        _: KW_ONLY

        hidden_size: int = -1
        intermediate_size: int = -1
        num_hidden_layers: int = -1
        num_attention_heads: int = -1
        num_key_value_heads: int = -1
        head_dim: int = -1
        rms_norm_eps: float = 1e-6
        rope_theta: float = 1_000_000.0
        hf_inv_freq: bool = True
        hf_split_projections: bool = False
        """Use HF's split projection order for exact parity tests.

        Normal Qwen3 models keep the loop-native fused QKV and SwiGLU
        projections. Enable this only when comparing against HuggingFace
        outputs where matmul grouping affects low-order floating-point bits.
        """

        @classmethod
        def from_hf(cls, config: dict[str, Any]) -> Self:
            """Parse an HF ``config.json`` dict. Validates model_type."""
            model_type = config.get("model_type")
            if model_type != "qwen3":
                raise ValueError(
                    f"Expected model_type='qwen3', got {model_type!r}. "
                    "Qwen3-MoE and earlier Qwen versions need their own loader.",
                )
            # transformers 4.55+ nests rope params; earlier has rope_theta flat.
            rope_theta = cast("float | None", config.get("rope_theta"))
            if rope_theta is None:
                rope_params = cast(
                    "dict[str, Any]",
                    config.get("rope_parameters") or {},
                )
                rope_theta = cast("float | None", rope_params.get("rope_theta"))
            hidden_size = int(config["hidden_size"])
            num_heads = int(config["num_attention_heads"])
            return cls(
                vocab_size=int(config["vocab_size"]),
                hidden_size=hidden_size,
                intermediate_size=int(config["intermediate_size"]),
                num_hidden_layers=int(config["num_hidden_layers"]),
                num_attention_heads=num_heads,
                num_key_value_heads=int(
                    config.get("num_key_value_heads") or num_heads,
                ),
                head_dim=int(
                    config.get("head_dim") or (hidden_size // num_heads),
                ),
                rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
                rope_theta=float(rope_theta if rope_theta is not None else 1e6),
                tie_embeddings=bool(config.get("tie_word_embeddings", False)),
            )

        @override
        def finalize(self) -> Self:
            if self.hidden_size < 1:
                raise ValueError(f"hidden_size must be > 0, got {self.hidden_size}.")
            if self.num_attention_heads < 1:
                raise ValueError(
                    f"num_attention_heads must be > 0, got {self.num_attention_heads}.",
                )
            # Wire arch fields into the inherited CausalLM.Config slots
            # before ``super().finalize()`` (mutate-before-super is
            # library convention — matches TransformerBlock/SwiGLU/MoE).
            self.channels = self.hidden_size
            self.num_layers = self.num_hidden_layers
            self.block = TransformerBlock.Config(
                channels_in=self.hidden_size,
                attn=SelfAttention.Config(
                    channels_in=self.hidden_size,
                    heads=self.num_attention_heads,
                    channels_head=self.head_dim,
                    num_heads_kv=self.num_key_value_heads,
                    bias=False,
                    causal=True,
                    rope=RoPE.Config(
                        channels_head=self.head_dim,
                        base=self.rope_theta,
                        hf_inv_freq=self.hf_inv_freq,
                    ),
                    norm_qk=RMSNorm.Config(
                        channels_in=self.head_dim,
                        eps=self.rms_norm_eps,
                        elementwise_affine=True,
                    ),
                    share_qk_norm=False,
                    split_qkv_projection=self.hf_split_projections,
                ),
                ffn=SwiGLU.Config(
                    channels_in=self.hidden_size,
                    channels_hidden=self.intermediate_size,
                    gate=True,
                    bias=False,
                    split_gate_projection=self.hf_split_projections,
                ),
                norm1=RMSNorm.Config(
                    eps=self.rms_norm_eps,
                    elementwise_affine=True,
                ),
                norm2=RMSNorm.Config(
                    eps=self.rms_norm_eps,
                    elementwise_affine=True,
                ),
                prenorm=True,
            )
            self.final_norm = RMSNorm.Config(
                eps=self.rms_norm_eps,
                elementwise_affine=True,
            )
            return super().finalize()

    # Inherits CausalLM.__init__, embed, blocks, final_norm, project_to_logits,
    # forward, reset_parameters. No per-arch Module-level behavior needed.

    @classmethod
    def load(
        cls,
        path_or_repo: Path | str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Qwen3:
        """Build a Qwen3 with HF weights loaded.

        Args:
          path_or_repo: Local directory with ``config.json`` + weight
              shards, OR a HuggingFace repo id (downloaded via
              ``priml.hub``).
          device: Target device (default: CPU).
          dtype: Override the dtype recorded in ``config.json``.

        """
        path = Path(path_or_repo)
        if path.is_dir() and (path / "config.json").exists():
            hf_config = json.loads((path / "config.json").read_text())
            hf_sd = hub.load_local_state_dict(path)
        else:
            hf_model = hub.load_transformers_model(
                str(path_or_repo),
                "AutoModelForCausalLM",
                dtype=dtype,
            )
            hf_config = hf_model.config.to_dict()
            hf_sd = {k: v.detach().cpu() for k, v in hf_model.state_dict().items()}
            del hf_model

        config = cls.Config.from_hf(hf_config).finalize()
        model = config.make()
        model.load_state_dict(remap_hf_state_dict(hf_sd, config), strict=True)
        model = model.to(
            dtype=dtype
            or hub.resolve_hf_dtype(str(hf_config.get("torch_dtype", "bfloat16"))),
        )
        if device is not None:
            model = model.to(device=device)
        return model


# -- HF weight remap ---------------------------------------------------


def remap_hf_state_dict(
    hf_sd: dict[str, Tensor],
    config: Qwen3.Config,
) -> dict[str, Tensor]:
    """Convert an HF Qwen3 ``state_dict`` to loop-native parameter names.

    Pure transform — no device moves, no dtype changes.
    """
    h = config.hidden_size
    n_q = config.num_attention_heads
    n_kv = config.num_key_value_heads
    d = config.head_dim
    out: dict[str, Tensor] = {
        "embed.weight": hf_sd["model.embed_tokens.weight"],
        "final_norm.weight": hf_sd["model.norm.weight"],
    }
    if not config.tie_embeddings:
        out["lm_head.weight"] = hf_sd["lm_head.weight"]
    for i in range(config.num_hidden_layers):
        p, b = f"model.layers.{i}", f"blocks.{i}"
        out[f"{b}.norm1.weight"] = hf_sd[f"{p}.input_layernorm.weight"]
        out[f"{b}.norm2.weight"] = hf_sd[f"{p}.post_attention_layernorm.weight"]
        # QKV: HF [q_heads*d, h], [kv*d, h], [kv*d, h] → loop
        # EnsembleLinear [q+2kv, d, h]. View+cat preserves row order.
        q = hf_sd[f"{p}.self_attn.q_proj.weight"].view(n_q, d, h)
        k = hf_sd[f"{p}.self_attn.k_proj.weight"].view(n_kv, d, h)
        v = hf_sd[f"{p}.self_attn.v_proj.weight"].view(n_kv, d, h)
        out[f"{b}.attn.proj_qkv.weight"] = torch.cat([q, k, v], dim=0)
        out[f"{b}.attn.proj_out.weight"] = hf_sd[f"{p}.self_attn.o_proj.weight"]
        out[f"{b}.attn.norm_q.weight"] = hf_sd[f"{p}.self_attn.q_norm.weight"]
        out[f"{b}.attn.norm_k.weight"] = hf_sd[f"{p}.self_attn.k_norm.weight"]
        # SwiGLU: HF split (gate, up) → loop fused up_proj [2*inter, h].
        # ``x.chunk(2, dim=-1)`` inside loop.SwiGLU yields (gate, x).
        gate = hf_sd[f"{p}.mlp.gate_proj.weight"]
        up = hf_sd[f"{p}.mlp.up_proj.weight"]
        out[f"{b}.ffn.up_proj.weight"] = torch.cat([gate, up], dim=0)
        out[f"{b}.ffn.down_proj.weight"] = hf_sd[f"{p}.mlp.down_proj.weight"]
    return out
