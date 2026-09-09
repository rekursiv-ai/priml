"""Qwen3 dense LM: configgle-native Config + HF weight loader.

Subclasses :class:`Transformer` per the library idiom
(``Makes[X]`` re-parents ``.make()``). ``Qwen3.Config`` carries the
HF-shaped arch fields; ``finalize()`` wires them into the inherited
``block``/``final_norm``/``channels_in``/``num_layers``/``out_proj``
slots.

Qwen3 vs. LLaMA:
  - Explicit ``head_dim`` (not ``hidden_size / num_heads``).
  - Per-head QK-norm — independent ``q_norm`` and ``k_norm`` RMSNorms
    (via ``SelfAttention.Config.share_qk_norm=False``).
  - GQA via ``num_key_value_heads``.
  - No bias on attention or MLP projections.
  - RoPE base ``rope_theta=1_000_000``, HF half-split pairing.

Usage::

    from priml.model.transformer.qwen3 import Qwen3

    model = Qwen3.Config.from_hf(hf_config).make()   # architecture only
    model = Qwen3.load("/path/to/Qwen3-0.6B")        # + weights from disk
    model = Qwen3.load("Qwen/Qwen3-0.6B")            # HF repo id (downloads)

Only the dense Qwen3 family is handled here; Qwen3-MoE is a follow-up.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from functools import partial
from pathlib import Path
from typing import Any, Literal, Self, override

import json

from configgle import Makeable, Makes
from torch import Tensor, nn

import torch

from priml import hub
from priml.lib.custom_json import DictCodec, FloatCodec
from priml.model.attention.rope import HuggingFaceFrequencies, RoPE
from priml.model.attention.self_attention import SelfAttention
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsInOutConfig,
    TensorBlockConfig,
    TensorModule,
    propagate_attr,
)
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer


class Qwen3(Transformer):
    """Qwen3 dense causal LM -- ``Transformer`` pre-wired for the Qwen3 arch."""

    class Config(Makes["Qwen3"], Transformer.Config, kw_only=False):
        vocab_size: int = 151_936
        """Token vocabulary size; also the width of the output projection."""

        _: KW_ONLY

        channels_in: int = 1_024
        """Residual-stream width. Qwen3-0.6B's, so the defaults load it."""

        num_layers: int = 28
        """Blocks in the stack. Qwen3-0.6B's."""

        in_proj: Makeable[TensorModule] | None = field(
            default_factory=lambda: Embedding.Config(
                init_weight=partial(nn.init.normal_, std=0.02),
                shard="vocab",
            )
        )
        """Reference token embedding initialization."""

        out_proj: ChannelsInOutConfig | Literal["tied"] | None = field(
            default_factory=lambda: Linear.Config(
                init_weight=partial(nn.init.normal_, std=0.02),
                shard="vocab",
            )
        )
        """Reference output-head initialization when embeddings are untied."""

        final_norm: Makeable[TensorModule] = field(
            default_factory=lambda: RMSNorm.Config(elementwise_affine=True)
        )
        """Learned final RMS scale, initialized to ones."""

        block: TensorBlockConfig | list[TensorBlockConfig] = field(
            default_factory=lambda: TransformerBlock.Config(
                attn=SelfAttention.Config(
                    init_weight=partial(nn.init.normal_, std=0.02),
                    num_heads=16,
                    num_heads_kv=8,
                    channels_head=128,
                    bias=False,
                    causal=True,
                    share_qk_norm=False,
                    rope=RoPE.Config(
                        frequencies=HuggingFaceFrequencies.Config(base=1e6),
                    ),
                    norm_qk=RMSNorm.Config(elementwise_affine=True),
                ),
                ffn=SwiGLU.Config(
                    init_weight=partial(nn.init.normal_, std=0.02),
                    init_weight_out=partial(nn.init.normal_, std=0.02),
                    gate=True,
                    bias=False,
                    channels_hidden=3_072,
                ),
                norm1=RMSNorm.Config(elementwise_affine=True),
                norm2=RMSNorm.Config(elementwise_affine=True),
                prenorm=True,
            ),
        )
        """Block template (broadcast ``num_layers`` times), or a list."""

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
            rope_theta = config.get("rope_theta")
            if rope_theta is None:
                # Validated rather than cast: this is an HF ``config.json``, so
                # a malformed field is caller input. Casting produced an
                # ``AttributeError`` from inside ``.get`` instead.
                rope_params = DictCodec.coerce(config.get("rope_parameters") or {})
                rope_theta = FloatCodec.coerce(rope_params.get("rope_theta"), 1e6)
            channels_in = int(config["hidden_size"])
            num_heads = int(config["num_attention_heads"])
            # HF's schema is parsed into the CHILD configs; the parent does
            # not mirror foreign names onto itself. Everything below hangs off
            # the ONE block template, which is where each value lives.
            norm = RMSNorm.Config(elementwise_affine=True)
            norm.eps = float(config.get("rms_norm_eps", 1e-6))

            frequencies = HuggingFaceFrequencies.Config()
            frequencies.base = FloatCodec.coerce(rope_theta, 1e6)
            rope = RoPE.Config()
            rope.frequencies = frequencies

            attn = SelfAttention.Config(bias=False, causal=True, share_qk_norm=False)
            attn.num_heads = num_heads
            num_heads_kv = int(config.get("num_key_value_heads", num_heads))
            if num_heads_kv < 1:
                raise ValueError(
                    f"num_key_value_heads must be > 0, got {num_heads_kv}."
                )
            attn.num_heads_kv = num_heads_kv
            # Qwen3 states the head width, so it need not divide the model
            # width -- the attention's inner width is decoupled from the
            # residual. Falling back to the quotient matches HF's own default.
            channels_head = int(
                config["head_dim"] if "head_dim" in config else channels_in // num_heads
            )
            if channels_head < 1:
                raise ValueError(f"head_dim must be > 0, got {channels_head}.")
            attn.channels_head = channels_head
            attn.rope = rope
            attn.norm_qk = norm.copy_tree()

            init_weight = partial(
                nn.init.normal_,
                std=FloatCodec.coerce(
                    config.get("initializer_range", 0.02),
                    default=None,
                ),
            )
            attn.init_weight = init_weight
            block = TransformerBlock.Config(prenorm=True)
            block.attn = attn
            block.ffn = SwiGLU.Config(
                init_weight=init_weight,
                init_weight_out=init_weight,
                gate=True,
                bias=False,
                channels_hidden=int(config["intermediate_size"]),
            )
            block.norm1 = norm.copy_tree()
            block.norm2 = norm.copy_tree()

            return cls(
                vocab_size=int(config["vocab_size"]),
                channels_in=channels_in,
                num_layers=int(config["num_hidden_layers"]),
                in_proj=Embedding.Config(init_weight=init_weight, shard="vocab"),
                out_proj=(
                    "tied"
                    if bool(config.get("tie_word_embeddings", False))
                    else Linear.Config(init_weight=init_weight, shard="vocab")
                ),
                block=block,
                final_norm=norm.copy_tree(),
            )

        @override
        def finalize(self) -> Self:
            # Mutate-before-super is library convention -- matches
            # TransformerBlock/SwiGLU/MoE.
            if not isinstance(self.block, list):
                # One template, copied per layer: a shared node would have each
                # block's own finalize push its widths into the others.
                self.block = [self.block.copy_tree() for _ in range(self.num_layers)]
            for block in self.block:
                self._size_block(block)
            return super().finalize()

        def _size_block(self, block: TensorBlockConfig) -> None:
            """Push the widths the PARENT owns into one already-shaped block.

            Only the widths: everything else on the block is the caller's, so
            an edit to the template survives ``finalize`` rather than being
            rebuilt over.
            """
            propagate_attr(block, "channels_in", self.channels_in, protocol=ChannelsIn)
            if not isinstance(block, TransformerBlock.Config):
                return
            attn = block.attn
            if isinstance(attn, SelfAttention.Config):
                attn.channels_in = self.channels_in
                rope = attn.rope
                if isinstance(rope, RoPE.Config):
                    rope.channels_head = attn.channels_head
            ffn = block.ffn
            if isinstance(ffn, SwiGLU.Config):
                ffn.channels_in = self.channels_in

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
        hf_config, hf_sd = _load_hf_checkpoint(path_or_repo, dtype=dtype)
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


def _load_hf_checkpoint(
    path_or_repo: Path | str, *, dtype: torch.dtype | None
) -> tuple[dict[str, object], dict[str, Tensor]]:
    """Read Qwen checkpoint metadata and tensors once, locally or through the hub."""
    path = Path(path_or_repo)
    if path.is_dir() and (path / "config.json").exists():
        hf_config = DictCodec.coerce(json.loads((path / "config.json").read_text()))
        return hf_config, hub.load_local_state_dict(path)
    hf_model = hub.load_transformers_model(
        str(path_or_repo), "AutoModelForCausalLM", dtype=dtype
    )
    return DictCodec.coerce(hf_model.config.to_dict()), {
        key: value.detach().cpu() for key, value in hf_model.state_dict().items()
    }


def _attn_of(config: Qwen3.Config, layer: int = 0) -> SelfAttention.Config:
    """Return one layer's attention config.

    Read off the BLOCK rather than a parent mirror of it: the geometry lives
    where the layer is built, so a per-layer list and a broadcast template
    both answer here without this function knowing which it was given.
    """
    blocks = config.block if isinstance(config.block, list) else [config.block]
    # ``len == 1`` is the pre-finalize broadcast template, which answers for
    # every layer. Any other short list is a genuine index error, and falling
    # back to layer 0 there remapped excess layers against the wrong geometry.
    block = blocks[0] if len(blocks) == 1 else blocks[layer]
    if not isinstance(block, TransformerBlock.Config):
        raise TypeError(f"layer {layer} is {type(block).__name__}, not a transformer.")
    attn = block.attn
    if not isinstance(attn, SelfAttention.Config):
        raise TypeError(
            f"layer {layer} attention is {type(attn).__name__}, not self-attention.",
        )
    return attn


def remap_hf_state_dict(
    hf_sd: dict[str, Tensor],
    config: Qwen3.Config,
) -> dict[str, Tensor]:
    """Convert an HF Qwen3 ``state_dict`` to loop-native parameter names.

    Pure transform — no device moves, no dtype changes.
    """
    h = config.channels_in
    attn = _attn_of(config)
    n_q = attn.num_heads
    n_kv = attn.num_heads_kv
    d = attn.channels_head
    out: dict[str, Tensor] = {
        "in_proj.weight": hf_sd["model.embed_tokens.weight"],
        "final_norm.weight": hf_sd["model.norm.weight"],
    }
    if config.out_proj != "tied":
        out["out_proj.weight"] = hf_sd["lm_head.weight"]
    for i in range(config.num_layers):
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
