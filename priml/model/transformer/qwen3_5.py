"""Native Qwen3.5 dense hybrid text language model."""

from __future__ import annotations

from dataclasses import field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, cast, override

import math

from configgle import Makeable, Makes
from torch import Tensor, nn

import torch

from priml import hub
from priml.lib.custom_json import (
    BoolCodec,
    DictCodec,
    FloatCodec,
    IntCodec,
    ListCodec,
    StrCodec,
    loads,
)
from priml.model.attention.gated_self_attention import GatedSelfAttention
from priml.model.attention.qwen3_5_delta import Qwen35GatedDeltaNet
from priml.model.attention.rope import HuggingFaceFrequencies, RoPE
from priml.model.custom_types import (
    ChannelsIn,
    TensorModule,
    has_forward_cached,
    is_cached_attention,
    propagate_attr,
)
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.norm import CenteredRMSNorm
from priml.model.special import TiedLinear
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.qwen3_5_weights import remap_hf_state_dict
from priml.model.transformer.transformer import Transformer


if TYPE_CHECKING:
    from collections.abc import Mapping


class Qwen35(Transformer):
    """Compose a text backbone and language-model head from native blocks."""

    class Config(Makes["Qwen35"], Transformer.Config):
        norm: Makeable[TensorModule] = field(default_factory=CenteredRMSNorm.Config)
        """Final hidden-state normalization, before the language-model head."""

        @classmethod
        def from_hf(cls, config: Mapping[str, object]) -> Self:
            """Parse the dense text architecture from a Hugging Face configuration.

            Args:
              config: Text config, or a Qwen3.5 config containing text_config.

            Returns:
              result: Native configuration with independently injectable blocks.

            Raises:
              ValueError: An architecture field is unsupported or invalid.
              TypeError: A configuration field has the wrong type.

            """
            config = _text_config(config)
            result = cls()
            result.channels_in = _positive(config, name="hidden_size")
            result.channels_out = _positive(config, name="vocab_size")
            count = _positive(config, name="num_hidden_layers")
            norm = CenteredRMSNorm.Config()
            norm.eps = FloatCodec.coerce(config.get("rms_norm_eps", 1e-6), default=None)
            if not math.isfinite(norm.eps) or norm.eps <= 0:
                raise ValueError("rms_norm_eps must be finite and positive.")
            result.norm = norm.copy_tree()
            initializer_range = FloatCodec.coerce(
                config.get("initializer_range", 0.02),
                default=None,
            )
            if not math.isfinite(initializer_range) or initializer_range <= 0:
                raise ValueError("initializer_range must be finite and positive.")
            init = partial(
                nn.init.normal_,
                std=initializer_range,
            )
            embedding = Embedding.Config()
            embedding.channels_in = result.channels_out
            embedding.init_weight = init
            result.proj_in = embedding
            if BoolCodec.coerce(config.get("tie_word_embeddings", False), default=None):
                head = TiedLinear.Config()
                head.tied = "proj_in"
                result.proj_out = head
            else:
                projection = Linear.Config()
                projection.init_weight = init
                result.proj_out = projection
            layers = _layer_types(config, count=count)
            result.block = []
            for layer_type in layers:
                block = TransformerBlock.Config()
                block.norm1 = norm.copy_tree()
                block.norm2 = norm.copy_tree()
                ffn = SwiGLU.Config()
                ffn.channels_hidden = _positive(config, name="intermediate_size")
                ffn.init_weight = init
                ffn.init_weight_out = init
                block.ffn = ffn
                if layer_type == "full_attention":
                    attention = _full_attention(config)
                    attention.init_weight = init
                    attention.norm_qk = norm.copy_tree()
                    block.attn = attention
                else:
                    delta = Qwen35GatedDeltaNet.Config()
                    delta.num_heads_k = _positive(config, name="linear_num_key_heads")
                    delta.num_heads_v = _positive(config, name="linear_num_value_heads")
                    if delta.num_heads_v % delta.num_heads_k:
                        raise ValueError(
                            "linear_num_value_heads must be a multiple of "
                            "linear_num_key_heads.",
                        )
                    delta.channels_k_head = _positive(
                        config,
                        name="linear_key_head_dim",
                    )
                    delta.channels_v_head = _positive(
                        config,
                        name="linear_value_head_dim",
                    )
                    delta.conv_kernel_size = _positive(
                        config,
                        name="linear_conv_kernel_dim",
                    )
                    delta.init_weight = init
                    propagate_attr(config=delta.norm, name="eps", value=norm.eps)
                    block.attn = delta
                result.block.append(block)
            return result

        @override
        def finalize(self) -> Self:
            if isinstance(self.norm, ChannelsIn):
                if self.norm.channels_in == -1:
                    propagate_attr(
                        config=self.norm,
                        name="channels_in",
                        value=self.channels_in,
                        protocol=ChannelsIn,
                    )
                elif self.norm.channels_in != self.channels_in:
                    raise ValueError(
                        f"norm.channels_in={self.norm.channels_in} must equal "
                        f"channels_in={self.channels_in}.",
                    )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.norm = config.norm.make()

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        non_text: Literal["reject", "discard"] = "reject",
    ) -> Qwen35:
        """Load a local Hugging Face text checkpoint, without a vision encoder.

        Args:
          path: Directory containing config.json and safetensors or PyTorch shards.
          device: Target device for the native model.
          dtype: Target dtype; defaults to the serialized embedding dtype.
          non_text: Reject extra weights unless explicitly discarding documented
            non-text subtrees. The pretrained language-model head is always loaded.

        Returns:
          model: Native text language model with strict text-parameter coverage.

        """
        directory = Path(path)
        metadata = DictCodec.coerce(
            loads((directory / "config.json").read_text()),
            default=None,
        )
        config = cls.Config.from_hf(metadata)
        weights = remap_hf_state_dict(
            state=hub.load_local_state_dict(directory),
            config=config,
            non_text=non_text,
        )
        target_dtype = weights["proj_in.weight"].dtype if dtype is None else dtype
        model = config.make().to(device=device, dtype=target_dtype)
        model.load_state_dict(weights, strict=True)
        return model

    @override
    def reset_parameters(self) -> None:
        """Initialize the base transformer and final normalization parameters."""
        super().reset_parameters()
        self.norm.reset_parameters()

    @override
    def project_to_logits(self, hidden: Tensor, **kwargs: object) -> Tensor:
        """Normalize residual stream and apply the pretrained language-model head."""
        return super().project_to_logits(self.norm(hidden), **kwargs)

    def hidden_states(
        self,
        x: Tensor,
        *,
        cache: list[object] | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Return normalized text hidden states from token IDs or input embeddings.

        Args:
          x: Integer token IDs [batch, sequence], or floating input embeddings.
          cache: One cache per block, optionally updated in place.
          **kwargs: Messages forwarded to native blocks, including positions and a
            2-D padding or prepared floating additive causal 4-D text attention
            mask.

        Returns:
          hidden: Normalized hidden states [batch, sequence, channels_in].

        """
        if not x.is_floating_point():
            if self.proj_in is None:
                raise ValueError("Token IDs require an input embedding.")
            x = self.proj_in(x)
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError("The cache must have one entry per transformer block.")
        attention_mask = _pop_tensor(kwargs, name="attention_mask")
        positions = _pop_tensor(kwargs, name="positions")
        position_ids = _pop_tensor(kwargs, name="position_ids")
        if positions is not None and position_ids is not None:
            raise ValueError("Pass either positions or position_ids, not both.")
        if position_ids is not None:
            positions = position_ids
        if positions is not None and positions.ndim == 2:
            positions = positions.unsqueeze(-1)
        full_mask = _full_attention_mask(attention_mask, x=x)
        padding_mask = (
            attention_mask
            if attention_mask is not None and attention_mask.ndim == 2
            else None
        )
        # Built once: nothing below depends on `index` or `block`, and `**block_kwargs`
        # hands each callee a fresh set of keyword arguments without mutating this dict.
        block_kwargs = dict(kwargs)
        if padding_mask is not None:
            block_kwargs["attention_mask"] = padding_mask
        if full_mask is not None:
            block_kwargs.setdefault("attn_mask", full_mask)
        if positions is not None:
            block_kwargs["positions"] = positions
        if position_ids is not None:
            # Forwarded raw, unlike `positions` above (never unsqueezed to
            # 3-D): no priml block declares a `position_ids` parameter today,
            # so it is absorbed by **kwargs, but the first one that does would
            # see a different shape than `positions`.
            block_kwargs["position_ids"] = position_ids
        for index, block in enumerate(self.blocks):
            if cache is None:
                x = cast(Tensor, block(x, **block_kwargs))
            else:
                if not has_forward_cached(block):
                    raise TypeError(
                        "Cached decoding requires blocks with a forward_cached method.",
                    )
                x, cache[index] = block.forward_cached(
                    x,
                    cache=cache[index],
                    **block_kwargs,
                )
        return self.norm(x)

    @override
    def forward(self, x: Tensor, /, **kwargs: object) -> Tensor:
        """Compute text logits from token IDs or input embeddings.

        Args:
          x: Token IDs or input embeddings.
          **kwargs: Block messages and an optional native cache list.

        Returns:
          output: Logits, or hidden states when no output projection is present.

        """
        cache = kwargs.pop("cache", None)
        if cache is not None and not isinstance(cache, list):
            raise TypeError("cache must be a list or None.")
        cache = cast(list[object] | None, cache)
        hidden = self.hidden_states(x, cache=cache, **kwargs)
        return hidden if self.proj_out is None else self.proj_out(hidden, **kwargs)

    def alloc_cache(
        self,
        *,
        batch: int,
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> list[object]:
        """Allocate one native attention state per block for cached decoding.

        Args:
          batch: Batch size.
          max_seq: Maximum cache length.
          device: Cache device; None uses each block's parameter device.
          dtype: Cache dtype; None uses each block's parameter dtype.

        Returns:
          cache: One attention cache per transformer block.

        """
        caches: list[object] = []
        for block in self.blocks:
            try:
                attention = block.get_submodule("attn")
            except AttributeError as error:
                raise TypeError(
                    "Cached decoding requires blocks with an attn submodule.",
                ) from error
            if not is_cached_attention(attention):
                raise TypeError(
                    "Cached decoding requires attention with an alloc_kv_cache method.",
                )
            caches.append(
                attention.alloc_kv_cache(
                    batch=batch,
                    max_seq=max_seq,
                    device=device,
                    dtype=dtype,
                ),
            )
        return caches

    def forward_cached(
        self,
        x: Tensor,
        *,
        cache: list[object],
        **kwargs: object,
    ) -> tuple[Tensor, list[object]]:
        """Compute logits while updating recurrent, convolution, and KV states.

        Args:
          x: Token IDs or input embeddings for the new positions.
          cache: One mutable cache per transformer block.
          **kwargs: Block messages, including text positions and attention masks.

        Returns:
          output: Logits or hidden states when no output projection is present.
          cache: The updated input cache list.

        """
        hidden = self.hidden_states(x, cache=cache, **kwargs)
        return (
            hidden if self.proj_out is None else self.proj_out(hidden, **kwargs),
            cache,
        )


def _text_config(config: Mapping[str, object]) -> dict[str, object]:
    model_type = config.get("model_type")
    if model_type == "qwen3_5":
        result = DictCodec.coerce(config.get("text_config"), default=None)
        if "tie_word_embeddings" in config:
            result["tie_word_embeddings"] = config["tie_word_embeddings"]
    elif model_type == "qwen3_5_text":
        result = dict(config)
    else:
        raise ValueError(f"Expected a dense Qwen3.5 text config, got {model_type!r}.")
    if result.get("model_type", "qwen3_5_text") != "qwen3_5_text":
        raise ValueError("The nested text config must have model_type='qwen3_5_text'.")
    if result.get("hidden_act", "silu") != "silu":
        raise ValueError("Only the SwiGLU/silu Qwen3.5 architecture is supported.")
    if (
        result.get("quantization_config") is not None
        or config.get("quantization_config") is not None
    ):
        raise ValueError("Quantized checkpoint configurations are unsupported.")
    if result.get("sliding_window") is not None:
        raise ValueError("Sliding-window attention is unsupported.")
    return result


def _positive(config: Mapping[str, object], *, name: str) -> int:
    value = IntCodec.coerce(config.get(name), default=None)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _layer_types(config: Mapping[str, object], *, count: int) -> list[str]:
    raw = config.get("layer_types")
    if raw is None:
        interval = IntCodec.coerce(
            config.get("full_attention_interval", 4),
            default=None,
        )
        if interval <= 0:
            raise ValueError("full_attention_interval must be positive.")
        return [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(count)
        ]
    layers = ListCodec.coerce(raw, default=None)
    if len(layers) != count or any(
        layer not in ("full_attention", "linear_attention") for layer in layers
    ):
        raise ValueError(
            "layer_types must name one supported attention type per layer.",
        )
    return [StrCodec.coerce(layer, default=None) for layer in layers]


def _full_attention(config: Mapping[str, object]) -> GatedSelfAttention.Config:
    attention = GatedSelfAttention.Config()
    attention.num_heads = _positive(config, name="num_attention_heads")
    attention.num_heads_kv = _positive(config, name="num_key_value_heads")
    if attention.num_heads % attention.num_heads_kv:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads.",
        )
    attention.channels_head = _positive(config, name="head_dim")
    attention.bias = BoolCodec.coerce(config.get("attention_bias", False), default=None)
    attention.dropout = FloatCodec.coerce(
        config.get("attention_dropout", 0.0),
        default=None,
    )
    if (
        not math.isfinite(attention.dropout)
        or attention.dropout < 0
        or attention.dropout >= 1
    ):
        raise ValueError("attention_dropout must be finite and in [0, 1).")
    params = DictCodec.coerce(config.get("rope_parameters", {}), default=None)
    if (
        params.get("rope_type", "default") != "default"
        or config.get("rope_scaling") is not None
    ):
        raise ValueError("Only default text rotary frequencies are supported.")
    fraction = FloatCodec.coerce(
        params.get("partial_rotary_factor", config.get("partial_rotary_factor", 0.25)),
        default=None,
    )
    if not math.isfinite(fraction) or fraction <= 0 or fraction > 1:
        raise ValueError("partial_rotary_factor must be finite and in (0, 1].")
    width = int(attention.channels_head * fraction)
    if width < 2 or width % 2:
        raise ValueError("The rotary prefix must have a positive even width.")
    rope_theta = FloatCodec.coerce(
        params.get("rope_theta", config.get("rope_theta", 10_000_000.0)),
        default=None,
    )
    if not math.isfinite(rope_theta) or rope_theta <= 0:
        raise ValueError("rope_theta must be finite and positive.")
    frequencies = HuggingFaceFrequencies.Config()
    frequencies.base = rope_theta
    rope = RoPE.Config()
    rope.channels_head = width
    rope.frequencies = frequencies
    attention.rope = rope
    return attention


def _pop_tensor(messages: dict[str, object], *, name: str) -> Tensor | None:
    value = messages.pop(name, None)
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor or None.")
    return value


def _full_attention_mask(attention_mask: Tensor | None, *, x: Tensor) -> Tensor | None:
    """Build a padding mask or return a floating additive causal 4-D mask."""
    if attention_mask is None:
        return None
    if attention_mask.ndim == 4:
        if not attention_mask.is_floating_point():
            raise TypeError("attention_mask must be floating additive when it is 4-D.")
        return attention_mask
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be a 2-D padding mask or 4-D mask.")
    keys = attention_mask.shape[-1]
    queries = x.shape[-2]
    causal = torch.arange(keys, device=x.device) <= (
        torch.arange(queries, device=x.device).unsqueeze(-1) + keys - queries
    )
    padding = ~attention_mask.to(device=x.device, dtype=torch.bool)
    fill = torch.full(
        (1,),
        torch.finfo(x.dtype).min,
        device=x.device,
        dtype=x.dtype,
    )
    return torch.where(
        padding[:, None, None, :] & causal[None, None, :, :],
        fill,
        0,
    )
