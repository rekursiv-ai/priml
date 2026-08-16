"""Kimi-K2 (DeepSeek-V3 architecture) LM: configgle-native Config + loader.

Subclasses :class:`CausalLM` via the library idiom (``Makes[X]``
re-parents ``.make()``). ``KimiK2.Config`` carries HF-shaped arch
fields; ``finalize()`` wires them into the inherited slots, including
a per-layer ``block`` list (dense SwiGLU for the first
``first_k_dense_replace`` layers, :class:`MoE` (sigmoid-routed with
shared experts) thereafter).

Architecture (from Kimi-K2-Instruct ``config.json``)::

    model_type           : "kimi_k2" | "deepseek_v3"
    hidden_size          : 7168
    num_hidden_layers    : 61
    num_attention_heads  : 64
    qk_nope_head_dim     : 128    -> channels_qk_nope_head
    qk_rope_head_dim     : 64     -> channels_qk_rope_head
    v_head_dim           : 128    -> channels_v_head
    q_lora_rank          : null       (Kimi-K2; DSV3 uses 1536)
    kv_lora_rank         : 512
    n_routed_experts     : 384
    num_experts_per_tok  : 8
    n_shared_experts     : 1
    first_k_dense_replace: 1          (layer 0 dense, 1..60 MoE)
    moe_intermediate_size: 2048
    intermediate_size    : 18432
    scoring_func         : "sigmoid"
    norm_topk_prob       : True
    routed_scaling_factor: 2.827
    rope_theta           : 50000
    rope_scaling         : YaRN       (applied via YarnScaling)

Usage::

    from priml.model.kimi_k2 import KimiK2

    model = KimiK2.Config.from_hf(hf_config).make()   # architecture only
    model = KimiK2.load("/path/to/Kimi-K2-Instruct")  # + weights from disk
    model = KimiK2.load("moonshotai/Kimi-K2-Instruct")  # HF repo id
"""

from __future__ import annotations

from dataclasses import KW_ONLY
from pathlib import Path
from typing import Any, Literal, Self, cast, override

import json

from configgle import Makeable, Makes
from torch import Tensor

import torch

from priml import hub
from priml.model.causal_lm import CausalLM
from priml.model.custom_types import TensorModule
from priml.model.mla import MultiHeadLatentAttention
from priml.model.moe import MoE, Router
from priml.model.norm import RMSNorm
from priml.model.rope import RoPE, YarnScaling
from priml.model.swiglu import SwiGLU
from priml.model.transformer import TransformerBlock


_VALID_MODEL_TYPES = frozenset({"kimi_k2", "deepseek_v3"})


class KimiK2(CausalLM):
    """Kimi-K2 / DeepSeek-V3 causal LM — MLA + DS-V3 MoE."""

    class Config(Makes["KimiK2"], CausalLM.Config, kw_only=False):
        vocab_size: int = -1

        _: KW_ONLY

        hidden_size: int = -1
        num_hidden_layers: int = -1
        num_attention_heads: int = -1

        channels_qk_nope_head: int = 128
        channels_qk_rope_head: int = 64
        channels_v_head: int = 128
        q_lora_rank: int | None = None
        kv_lora_rank: int = 512

        intermediate_size: int = -1
        """Dense-layer MLP hidden."""

        moe_intermediate_size: int = -1
        """Per-expert hidden in MoE layers (and per shared expert)."""

        n_routed_experts: int = 0
        num_experts_per_tok: int = 1
        n_shared_experts: int = 0
        first_k_dense_replace: int = 0
        n_group: int = 1
        topk_group: int = 1
        norm_topk_prob: bool = True
        routed_scaling_factor: float = 1.0
        scoring_func: Literal["softmax", "sigmoid"] = "sigmoid"

        rms_norm_eps: float = 1e-6
        rope_theta: float = 10_000.0

        yarn: YarnScaling | None = None
        """YaRN RoPE scaling. ``None`` = vanilla RoPE."""

        @classmethod
        def from_hf(cls, config: dict[str, Any]) -> Self:
            """Parse an HF ``config.json`` dict."""
            model_type = config.get("model_type")
            if model_type not in _VALID_MODEL_TYPES:
                raise ValueError(
                    f"Expected model_type in {sorted(_VALID_MODEL_TYPES)}, "
                    f"got {model_type!r}.",
                )
            scoring_func = config.get("scoring_func", "sigmoid")
            if scoring_func not in ("softmax", "sigmoid"):
                raise ValueError(
                    f"Expected scoring_func in ('softmax', 'sigmoid'), "
                    f"got {scoring_func!r}.",
                )
            return cls(
                vocab_size=int(config["vocab_size"]),
                hidden_size=int(config["hidden_size"]),
                num_hidden_layers=int(config["num_hidden_layers"]),
                num_attention_heads=int(config["num_attention_heads"]),
                # Quoted keys are HF's JSON schema, not priml names.
                channels_qk_nope_head=int(config.get("qk_nope_head_dim", 128)),
                channels_qk_rope_head=int(config.get("qk_rope_head_dim", 64)),
                channels_v_head=int(config.get("v_head_dim", 128)),
                q_lora_rank=(
                    int(config["q_lora_rank"])
                    if config.get("q_lora_rank") is not None
                    else None
                ),
                kv_lora_rank=int(config.get("kv_lora_rank", 512)),
                intermediate_size=int(config["intermediate_size"]),
                moe_intermediate_size=int(
                    config.get("moe_intermediate_size") or config["intermediate_size"],
                ),
                n_routed_experts=int(config.get("n_routed_experts", 0)),
                num_experts_per_tok=int(config.get("num_experts_per_tok", 1)),
                n_shared_experts=int(config.get("n_shared_experts", 0)),
                first_k_dense_replace=int(config.get("first_k_dense_replace", 0)),
                n_group=int(config.get("n_group", 1)),
                topk_group=int(config.get("topk_group", 1)),
                norm_topk_prob=bool(config.get("norm_topk_prob", True)),
                routed_scaling_factor=float(
                    config.get("routed_scaling_factor", 1.0),
                ),
                scoring_func=scoring_func,
                rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
                rope_theta=float(config.get("rope_theta", 10_000.0)),
                yarn=_parse_yarn(config.get("rope_scaling")),
                tie_embeddings=bool(config.get("tie_word_embeddings", False)),
            )

        @override
        def finalize(self) -> Self:
            if self.hidden_size < 1:
                raise ValueError(f"hidden_size must be > 0, got {self.hidden_size}.")
            if self.num_hidden_layers < 1:
                raise ValueError(
                    f"num_hidden_layers must be > 0, got {self.num_hidden_layers}.",
                )
            self.channels = self.hidden_size
            self.num_layers = self.num_hidden_layers
            # Per-layer block list: dense for first_k_dense_replace layers,
            # MoE for the rest. CausalLM.Config.block accepts a list.
            self.block = [
                self._block_for_layer(i) for i in range(self.num_hidden_layers)
            ]
            self.final_norm = RMSNorm.Config(
                eps=self.rms_norm_eps,
                elementwise_affine=True,
            )
            return super().finalize()

        def _attn_config(self) -> MultiHeadLatentAttention.Config:
            return MultiHeadLatentAttention.Config(
                channels_in=self.hidden_size,
                heads=self.num_attention_heads,
                channels_qk_nope_head=self.channels_qk_nope_head,
                channels_qk_rope_head=self.channels_qk_rope_head,
                channels_v_head=self.channels_v_head,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                bias=False,
                causal=True,
                rope=RoPE.Config(
                    channels_head=self.channels_qk_rope_head,
                    base=self.rope_theta,
                    yarn=self.yarn,
                ),
                rms_norm_eps=self.rms_norm_eps,
            )

        def _ffn_for_layer(self, layer_idx: int) -> Makeable[TensorModule]:
            if layer_idx < self.first_k_dense_replace:
                return SwiGLU.Config(
                    channels_in=self.hidden_size,
                    channels_hidden=self.intermediate_size,
                    gate=True,
                    bias=False,
                )
            return MoE.Config(
                channels_in=self.hidden_size,
                channels_out=self.hidden_size,
                router=Router.Config(
                    channels_in=self.hidden_size,
                    num_experts=self.n_routed_experts,
                    top_k=self.num_experts_per_tok,
                    scoring_func=self.scoring_func,
                    norm_topk_prob=self.norm_topk_prob,
                    routed_scaling_factor=self.routed_scaling_factor,
                    n_group=self.n_group,
                    topk_group=self.topk_group,
                ),
                expert=SwiGLU.Config(
                    channels_in=self.hidden_size,
                    channels_hidden=self.moe_intermediate_size,
                    gate=True,
                    bias=False,
                ),
                num_shared_experts=self.n_shared_experts,
                shared_expert=SwiGLU.Config(
                    channels_in=self.hidden_size,
                    channels_hidden=self.moe_intermediate_size,
                    gate=True,
                    bias=False,
                ),
            )

        def _block_for_layer(self, layer_idx: int) -> TransformerBlock.Config:
            return TransformerBlock.Config(
                channels_in=self.hidden_size,
                attn=self._attn_config(),
                ffn=self._ffn_for_layer(layer_idx),
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

    @classmethod
    def load(
        cls,
        path_or_repo: Path | str,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KimiK2:
        """Build a KimiK2 with HF weights loaded.

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
                trust_remote_code=True,
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


def _parse_yarn(rope_scaling: Any) -> YarnScaling | None:
    """Parse HF ``rope_scaling`` → YarnScaling. None-pass-through; strict on type."""
    if not rope_scaling:
        return None
    scaling = cast("dict[str, Any]", rope_scaling)
    stype = scaling.get("type") or scaling.get("rope_type")
    if stype is None:
        return None
    if stype != "yarn":
        raise ValueError(
            f"Unsupported rope_scaling type={stype!r}; only yarn is implemented.",
        )
    return YarnScaling(
        factor=float(scaling["factor"]),
        original_max_position_embeddings=int(
            scaling["original_max_position_embeddings"],
        ),
        beta_fast=float(scaling.get("beta_fast", 32.0)),
        beta_slow=float(scaling.get("beta_slow", 1.0)),
        mscale=float(scaling.get("mscale", 1.0)),
        mscale_all_dim=float(scaling.get("mscale_all_dim", 0.0)),
    )


# -- HF weight remap ---------------------------------------------------


def remap_hf_state_dict(
    hf_sd: dict[str, Tensor],
    config: KimiK2.Config,
) -> dict[str, Tensor]:
    """Convert an HF Kimi-K2 / DSV3 state_dict to loop-native names."""
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

        # -- MLA --------------------------------------------------------
        attn, ba = f"{p}.self_attn", f"{b}.attn"
        if config.q_lora_rank is None:
            out[f"{ba}.q_proj.weight"] = hf_sd[f"{attn}.q_proj.weight"]
        else:
            out[f"{ba}.q_a_proj.weight"] = hf_sd[f"{attn}.q_a_proj.weight"]
            out[f"{ba}.q_a_layernorm.weight"] = hf_sd[f"{attn}.q_a_layernorm.weight"]
            out[f"{ba}.q_b_proj.weight"] = hf_sd[f"{attn}.q_b_proj.weight"]
        # HF's kv_a_proj_with_mqa → loop's kv_a_proj.
        out[f"{ba}.kv_a_proj.weight"] = hf_sd[f"{attn}.kv_a_proj_with_mqa.weight"]
        out[f"{ba}.kv_a_layernorm.weight"] = hf_sd[f"{attn}.kv_a_layernorm.weight"]
        out[f"{ba}.kv_b_proj.weight"] = hf_sd[f"{attn}.kv_b_proj.weight"]
        out[f"{ba}.o_proj.weight"] = hf_sd[f"{attn}.o_proj.weight"]

        # -- FFN --------------------------------------------------------
        bf = f"{b}.ffn"
        if i < config.first_k_dense_replace:
            gate = hf_sd[f"{p}.mlp.gate_proj.weight"]
            up = hf_sd[f"{p}.mlp.up_proj.weight"]
            out[f"{bf}.up_proj.weight"] = torch.cat([gate, up], dim=0)
            out[f"{bf}.down_proj.weight"] = hf_sd[f"{p}.mlp.down_proj.weight"]
        else:
            # MoE router lives at ``ffn.router`` (gate + optional
            # aux-loss-free correction bias).
            out[f"{bf}.router.gate.weight"] = hf_sd[f"{p}.mlp.gate.weight"]
            out[f"{bf}.router.e_score_correction_bias"] = hf_sd.get(
                f"{p}.mlp.gate.e_score_correction_bias",
                torch.zeros(config.n_routed_experts),
            )
            for e in range(config.n_routed_experts):
                ep, be = f"{p}.mlp.experts.{e}", f"{bf}.experts.{e}"
                gate = hf_sd[f"{ep}.gate_proj.weight"]
                up = hf_sd[f"{ep}.up_proj.weight"]
                out[f"{be}.up_proj.weight"] = torch.cat([gate, up], dim=0)
                out[f"{be}.down_proj.weight"] = hf_sd[f"{ep}.down_proj.weight"]
            # Shared experts: HF collapses ``n_shared_experts=1`` into
            # a single module; loop stores a ModuleList indexed from 0.
            if config.n_shared_experts == 1:
                _remap_shared(
                    hf_sd, f"{p}.mlp.shared_experts", f"{bf}.shared_experts.0", out
                )
            else:
                for s in range(config.n_shared_experts):
                    _remap_shared(
                        hf_sd,
                        f"{p}.mlp.shared_experts.{s}",
                        f"{bf}.shared_experts.{s}",
                        out,
                    )
    return out


def _remap_shared(
    hf_sd: dict[str, Tensor],
    sp: str,
    bs: str,
    out: dict[str, Tensor],
) -> None:
    gate = hf_sd[f"{sp}.gate_proj.weight"]
    up = hf_sd[f"{sp}.up_proj.weight"]
    out[f"{bs}.up_proj.weight"] = torch.cat([gate, up], dim=0)
    out[f"{bs}.down_proj.weight"] = hf_sd[f"{sp}.down_proj.weight"]
