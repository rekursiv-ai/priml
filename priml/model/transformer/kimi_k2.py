"""Kimi-K2 (DeepSeek-V3 architecture) LM: configgle-native Config + loader.

Subclasses :class:`Transformer` via the library idiom (``Makes[X]``
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

    from priml.model.transformer.kimi_k2 import KimiK2

    model = KimiK2.Config.from_hf(hf_config).make()   # architecture only
    model = KimiK2.load("/path/to/Kimi-K2-Instruct")  # + weights from disk
    model = KimiK2.load("moonshotai/Kimi-K2-Instruct")  # HF repo id
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from functools import partial
from pathlib import Path
from typing import Any, Self, override

from configgle import Makeable, Makes
from torch import Tensor, nn

import torch

from priml import hub
from priml.lib.custom_json import DictCodec, FloatCodec, IntCodec, decode
from priml.model.attention.mla import MultiHeadLatentAttention
from priml.model.attention.rope import HuggingFaceFrequencies, RoPE, YarnScaling
from priml.model.custom_types import (
    ChannelsIn,
    TensorBlockConfig,
    TensorModule,
    propagate_attr,
)
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.moe import MoE, Router
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.special import TiedLinear
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.qwen3 import _tied
from priml.model.transformer.transformer import Transformer


_VALID_MODEL_TYPES = frozenset({"kimi_k2", "deepseek_v3"})


class KimiK2(Transformer):
    """Kimi-K2 / DeepSeek-V3 causal LM — MLA + DS-V3 MoE."""

    class Config(Makes["KimiK2"], Transformer.Config, kw_only=False):
        """Widths come from ``from_hf``: ``channels_in`` is ``hidden_size`` and
        ``channels_out`` the vocabulary, which sizes the embedding's rows too.
        """

        _: KW_ONLY

        num_layers: int = 61
        """Blocks in the stack. Kimi-K2's."""

        channels_hidden_dense: int = 18_432
        """FFN hidden width in the leading dense layers.

        Named apart from the MoE width because the two differ: the dense
        prefix is a full-width MLP, each expert is a narrow one.
        """

        channels_hidden_expert: int = 2_048
        """Per-expert hidden width (and per shared expert)."""

        first_k_dense_replace: int = 1
        """Leading layers whose ``ffn`` is replaced by a dense SwiGLU."""

        in_proj: Makeable[TensorModule] | None = field(
            default_factory=lambda: Embedding.Config(
                init_weight=partial(nn.init.normal_, std=0.02), shard="vocab"
            )
        )
        """Reference token embedding initialization."""

        out_proj: Makeable[TensorModule] | None = field(
            default_factory=lambda: Sequential.Config(
                elements=[
                    RMSNorm.Config(elementwise_affine=True),
                    Linear.Config(
                        init_weight=partial(nn.init.normal_, std=0.02),
                        shard="vocab",
                    ),
                ]
            )
        )
        """Learned final RMS scale, then the reference untied head.

        A tied checkpoint replaces the ``Linear`` with
        ``TiedLinear.Config(tied="in_proj")``."""

        block: TensorBlockConfig | list[TensorBlockConfig] = field(
            default_factory=lambda: TransformerBlock.Config(
                attn=MultiHeadLatentAttention.Config(
                    init_weight=partial(nn.init.normal_, std=0.02),
                    num_heads=64,
                    channels_qk_nope_head=128,
                    channels_qk_rope_head=64,
                    channels_v_head=128,
                    kv_lora_rank=512,
                    bias=False,
                    causal=True,
                ),
                ffn=MoE.Config(
                    router=Router.Config(num_experts=384),
                    expert=SwiGLU.Config(
                        init_weight=partial(nn.init.normal_, std=0.02),
                        init_weight_out=partial(nn.init.normal_, std=0.02),
                    ),
                    shared_expert=SwiGLU.Config(
                        init_weight=partial(nn.init.normal_, std=0.02),
                        init_weight_out=partial(nn.init.normal_, std=0.02),
                    ),
                    num_shared_experts=1,
                ),
                norm1=RMSNorm.Config(elementwise_affine=True),
                norm2=RMSNorm.Config(elementwise_affine=True),
                prenorm=True,
            ),
        )
        """Block template (broadcast ``num_hidden_layers`` times), or a list.

        ONE slot rather than a `router`, a `norm` and a `rope` beside it: each
        of those belongs to something the block already holds, and hoisting
        them flattened the tree the reader is supposed to descend one node at
        a time. The routing policy is ``block.ffn.router``, the epsilon is
        ``block.norm1.eps``, the rotary base is
        ``block.attn.rope.frequencies.base`` -- each named by its position, and
        each editable without this class knowing the field exists.

        ``finalize`` copies the template per layer and pushes only the widths
        the PARENT owns (``channels_in`` and the hidden widths), so an edit to
        the template survives it.
        """

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
            # HF's schema is parsed into the CHILD configs; the parent does
            # not mirror foreign names onto itself. Everything below hangs off
            # the ONE block template, which is where each value lives.
            eps = float(config.get("rms_norm_eps", 1e-6))
            norm = RMSNorm.Config(elementwise_affine=True)
            norm.eps = eps

            frequencies = HuggingFaceFrequencies.Config()
            frequencies.base = float(config.get("rope_theta", 10_000.0))
            rope = RoPE.Config()
            rope.frequencies = _parse_yarn(config.get("rope_scaling")) or frequencies

            init_weight = partial(
                nn.init.normal_,
                std=FloatCodec.coerce(
                    config.get("initializer_range", 0.02), default=None
                ),
            )
            attn = MultiHeadLatentAttention.Config(
                bias=False, causal=True, init_weight=init_weight
            )
            attn.num_heads = int(config["num_attention_heads"])
            attn.channels_qk_nope_head = int(config.get("qk_nope_head_dim", 128))
            attn.channels_qk_rope_head = int(config.get("qk_rope_head_dim", 64))
            attn.channels_v_head = int(config.get("v_head_dim", 128))
            attn.q_lora_rank = (
                int(config["q_lora_rank"])
                if config.get("q_lora_rank") is not None
                else None
            )
            attn.kv_lora_rank = int(config.get("kv_lora_rank", 512))
            attn.rope = rope
            attn.norm_q_lora = norm.copy_tree()
            attn.norm_kv_lora = norm.copy_tree()

            router = Router.Config()
            router.top_k = int(config.get("num_experts_per_tok", 1))
            router.scoring_func = scoring_func
            router.norm_topk_prob = bool(config.get("norm_topk_prob", True))
            router.routed_scaling_factor = float(
                config.get("routed_scaling_factor", 1.0),
            )
            router.n_group = int(config.get("n_group", 1))
            router.topk_group = int(config.get("topk_group", 1))

            moe = MoE.Config(
                expert=SwiGLU.Config(
                    init_weight=init_weight, init_weight_out=init_weight
                ),
                shared_expert=SwiGLU.Config(
                    init_weight=init_weight, init_weight_out=init_weight
                ),
            )
            moe.router = router
            moe.num_shared_experts = int(config.get("n_shared_experts", 0))
            router.num_experts = int(config.get("n_routed_experts", 0))

            block = TransformerBlock.Config(prenorm=True)
            block.attn = attn
            block.ffn = moe
            block.norm1 = norm.copy_tree()
            block.norm2 = norm.copy_tree()

            # ABSENCE, not falsiness, selects the dense width: ``... or
            # config["intermediate_size"]`` also swallowed an explicit
            # ``moe_intermediate_size: 0``, which then built every expert at the
            # 18432-wide dense size and only surfaced as a shape mismatch when
            # the checkpoint failed to load.
            channels_hidden_expert = int(
                config.get(
                    "moe_intermediate_size",
                    config["intermediate_size"],
                )
            )
            if channels_hidden_expert < 1:
                raise ValueError(
                    f"moe_intermediate_size must be > 0, got {channels_hidden_expert}.",
                )

            head: Makeable[TensorModule] = (
                TiedLinear.Config(tied="in_proj")
                if bool(config.get("tie_word_embeddings", False))
                else Linear.Config(init_weight=init_weight, shard="vocab")
            )
            return cls(
                channels_in=int(config["hidden_size"]),
                channels_out=int(config["vocab_size"]),
                num_layers=int(config["num_hidden_layers"]),
                in_proj=Embedding.Config(init_weight=init_weight, shard="vocab"),
                out_proj=Sequential.Config(elements=[norm.copy_tree(), head]),
                channels_hidden_dense=int(config["intermediate_size"]),
                channels_hidden_expert=channels_hidden_expert,
                first_k_dense_replace=int(config.get("first_k_dense_replace", 0)),
                block=block,
            )

        @override
        def finalize(self) -> Self:
            if not isinstance(self.block, list):
                # One template, copied per layer: a shared node would have each
                # block's own finalize push its widths into the others.
                self.block = [self.block.copy_tree() for _ in range(self.num_layers)]
            for layer, block in enumerate(self.block):
                self._size_block(block, layer)
            # The vocabulary is stated once, as the output width; the table's
            # row count follows from it.
            propagate_attr(self.in_proj, "num_embeddings", self.channels_out)
            return super().finalize()

        def _size_block(self, block: TensorBlockConfig, layer: int) -> None:
            """Push the widths the PARENT owns into one already-shaped block.

            Only the widths: everything else on the block is the caller's, so
            an edit to the template survives ``finalize`` rather than being
            rebuilt over.
            """
            propagate_attr(block, "channels_in", self.channels_in, protocol=ChannelsIn)
            if not isinstance(block, TransformerBlock.Config):
                return
            attn = block.attn
            if isinstance(attn, MultiHeadLatentAttention.Config):
                attn.channels_in = self.channels_in
                rope = attn.rope
                if isinstance(rope, RoPE.Config):
                    rope.channels_head = attn.channels_qk_rope_head
            if layer < self.first_k_dense_replace:
                if isinstance(block.ffn, MoE.Config) and isinstance(
                    block.ffn.expert, SwiGLU.Config
                ):
                    block.ffn = block.ffn.expert.copy_tree()
                if isinstance(block.ffn, SwiGLU.Config):
                    block.ffn.channels_in = self.channels_in
                    block.ffn.channels_hidden = self.channels_hidden_dense
                return
            ffn = block.ffn
            if not isinstance(ffn, MoE.Config):
                return
            ffn.channels_in = self.channels_in
            ffn.channels_out = self.channels_in
            if isinstance(ffn.router, Router.Config):
                ffn.router.channels_in = self.channels_in
            for expert in (ffn.expert, ffn.shared_expert):
                if isinstance(expert, SwiGLU.Config):
                    expert.channels_in = self.channels_in
                    expert.channels_hidden = self.channels_hidden_expert

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
            hf_config = DictCodec.coerce(
                decode("object", (path / "config.json").read_text())
            )
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


def _parse_yarn(rope_scaling: object) -> YarnScaling.Config | None:
    """Parse HF ``rope_scaling`` → config. None-pass-through; strict on type."""
    if not rope_scaling:
        return None
    # Validated rather than cast: this is an HF ``config.json``, so a
    # malformed field is caller input, and casting surfaced it as an
    # ``AttributeError`` from inside ``.get``.
    scaling = DictCodec.coerce(rope_scaling)
    stype = scaling.get("type") or scaling.get("rope_type")
    if stype is None:
        return None
    if stype != "yarn":
        raise ValueError(
            f"Unsupported rope_scaling type={stype!r}; only yarn is implemented.",
        )
    config = YarnScaling.Config()
    config.factor = FloatCodec.coerce(scaling["factor"])
    config.original_max_position_embeddings = IntCodec.coerce(
        scaling["original_max_position_embeddings"],
        default=4_096,
    )
    config.beta_fast = FloatCodec.coerce(scaling.get("beta_fast"), 32.0)
    config.beta_slow = FloatCodec.coerce(scaling.get("beta_slow"), 1.0)
    config.mscale = FloatCodec.coerce(scaling.get("mscale"), 1.0)
    config.mscale_all_dim = FloatCodec.coerce(scaling.get("mscale_all_dim"), 0.0)
    return config


# -- HF weight remap ---------------------------------------------------


def _attn_of(config: KimiK2.Config, layer: int) -> MultiHeadLatentAttention.Config:
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
    if not isinstance(attn, MultiHeadLatentAttention.Config):
        raise TypeError(f"layer {layer} attention is {type(attn).__name__}, not MLA.")
    return attn


def _moe_of(config: KimiK2.Config, layer: int) -> MoE.Config:
    """Return one layer's MoE config, where the expert counts live."""
    blocks = config.block if isinstance(config.block, list) else [config.block]
    block = blocks[0] if len(blocks) == 1 else blocks[layer]
    if not isinstance(block, TransformerBlock.Config):
        raise TypeError(f"layer {layer} is {type(block).__name__}, not a transformer.")
    ffn = block.ffn
    if not isinstance(ffn, MoE.Config):
        raise TypeError(f"layer {layer} FFN is {type(ffn).__name__}, not MoE.")
    return ffn


def remap_hf_state_dict(
    hf_sd: dict[str, Tensor],
    config: KimiK2.Config,
) -> dict[str, Tensor]:
    """Convert an HF Kimi-K2 / DSV3 state_dict to loop-native names."""
    out: dict[str, Tensor] = {
        "in_proj.weight": hf_sd["model.embed_tokens.weight"],
        "out_proj.0.weight": hf_sd["model.norm.weight"],
    }
    if not _tied(config):
        out["out_proj.1.weight"] = hf_sd["lm_head.weight"]

    for i in range(config.num_layers):
        p, b = f"model.layers.{i}", f"blocks.{i}"
        out[f"{b}.norm1.weight"] = hf_sd[f"{p}.input_layernorm.weight"]
        out[f"{b}.norm2.weight"] = hf_sd[f"{p}.post_attention_layernorm.weight"]

        # -- MLA --------------------------------------------------------
        attn, ba = f"{p}.self_attn", f"{b}.attn"
        if _attn_of(config, i).q_lora_rank is None:
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
            moe = _moe_of(config, i)
            router = moe.router
            assert isinstance(router, Router.Config)
            out[f"{bf}.router.e_score_correction_bias"] = hf_sd.get(
                f"{p}.mlp.gate.e_score_correction_bias",
                torch.zeros(router.num_experts),
            )
            for e in range(router.num_experts):
                ep, be = f"{p}.mlp.experts.{e}", f"{bf}.experts.{e}"
                gate = hf_sd[f"{ep}.gate_proj.weight"]
                up = hf_sd[f"{ep}.up_proj.weight"]
                out[f"{be}.up_proj.weight"] = torch.cat([gate, up], dim=0)
                out[f"{be}.down_proj.weight"] = hf_sd[f"{ep}.down_proj.weight"]
            # Shared experts: HF collapses ``n_shared_experts=1`` into
            # a single module; loop stores a ModuleList indexed from 0.
            if moe.num_shared_experts == 1:
                _remap_shared(
                    hf_sd, f"{p}.mlp.shared_experts", f"{bf}.shared_experts.0", out
                )
            else:
                for s in range(moe.num_shared_experts):
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
