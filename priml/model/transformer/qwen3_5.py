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
from priml.lib.codec import from_plain, loads
from priml.model.attention.gated_attention import GatedAttention
from priml.model.attention.kvcache import alloc_layer_cache
from priml.model.attention.qwen3_5_delta import Qwen35GatedDeltaNet
from priml.model.attention.rope import HuggingFaceFrequencies, RoPE, YarnScaling
from priml.model.custom_types import (
    ChannelsIn,
    DepthIndex,
    LayerCache,
    TensorModule,
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
            norm.eps = from_plain(config.get("rms_norm_eps"), float, default=1e-6)
            if not math.isfinite(norm.eps) or norm.eps <= 0:
                raise ValueError("rms_norm_eps must be finite and positive.")
            result.norm = norm.copy_tree()
            initializer_range = from_plain(
                config.get("initializer_range"),
                float,
                default=0.02,
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
            if from_plain(config.get("tie_word_embeddings"), bool, default=False):
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
        metadata = from_plain(
            loads((directory / "config.json").read_text()),
            dict[str, object],
        )
        config = cls.Config.from_hf(metadata)
        weights = remap_hf_state_dict(
            state=hub.load_local_state_dict(directory),
            config=config,
            non_text=non_text,
        )
        target_dtype = weights["proj_in.weight"].dtype if dtype is None else dtype
        model = config.make().to(device=device, dtype=target_dtype)
        model.load_state_dict(weights)
        return model

    @override
    def reset_parameters(self) -> None:
        """Initialize the base transformer and final normalization parameters."""
        super().reset_parameters()
        self.norm.reset_parameters()

    @override
    def project_to_logits(self, hidden: Tensor, **kwargs: object) -> Tensor:
        """Normalize the residual stream, then apply the language-model head.

        The final norm lives HERE rather than in :meth:`hidden_states`, so the
        residual from ``hidden_states`` -- and from ``generate``'s own block
        loop -- is projected exactly once through it.

        Args:
          hidden: Pre-norm residual stream [..., sequence, channels_in].
          **kwargs: Messages forwarded to the output head.

        Returns:
          output: Logits, or normalized hidden states without an output head.

        """
        return super().project_to_logits(self.norm(hidden), **kwargs)

    def hidden_states(
        self,
        x: Tensor,
        *,
        cache: LayerCache | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Return the residual stream after the last block, before the final norm.

        Args:
          x: Integer token IDs [batch, sequence], or floating input embeddings.
          cache: Shared layer cache, each attention's slot updated in place.
          **kwargs: Messages forwarded to native blocks, including positions and a
            2-D padding or prepared floating additive causal 4-D text attention
            mask.

        Returns:
          hidden: Pre-norm residual [batch, sequence, channels_in]; pass it to
            :meth:`project_to_logits`, which applies the final norm.

        """
        if not x.is_floating_point():
            if self.proj_in is None:
                raise ValueError("Token IDs require an input embedding.")
            x = self.proj_in(x)
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
        for block in self.blocks:
            x = cast(Tensor, block(x, cache=cache, **block_kwargs))
        return x

    @override
    def forward(self, x: Tensor, /, **kwargs: object) -> Tensor:
        """Compute text logits from token IDs or input embeddings.

        Args:
          x: Token IDs or input embeddings.
          **kwargs: Block messages and an optional shared layer cache.

        Returns:
          output: Logits, or normalized hidden states without an output head.

        """
        cache = kwargs.pop("cache", None)
        if cache is not None and not isinstance(cache, LayerCache):
            raise TypeError("cache must satisfy LayerCache or be None.")
        return self.project_to_logits(
            self.hidden_states(x, cache=cache, **kwargs),
            **kwargs,
        )

    def alloc_cache(
        self,
        *,
        batch: int,
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[DepthIndex, object]:
        """Allocate shared decode state for every cached attention layer.

        Args:
          batch: Batch size or batch shape.
          max_seq: Maximum cached sequence length.
          device: Device for the cache tensors.
          dtype: Dtype for the cache tensors.

        Returns:
          cache: One slot per cached attention, keyed by its depth index.

        """
        return alloc_layer_cache(
            self,
            batch=batch,
            max_seq=max_seq,
            device=device,
            dtype=dtype,
        )


def _text_config(config: Mapping[str, object]) -> dict[str, object]:
    model_type = config.get("model_type")
    if model_type == "qwen3_5":
        result = from_plain(config.get("text_config"), dict[str, object])
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
    value = from_plain(config.get(name), int)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _layer_types(config: Mapping[str, object], *, count: int) -> list[str]:
    raw = config.get("layer_types")
    if raw is None:
        interval = (
            4
            if "full_attention_interval" not in config
            else from_plain(config["full_attention_interval"], int)
        )
        if interval <= 0:
            raise ValueError("full_attention_interval must be positive.")
        return [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(count)
        ]
    layers = from_plain(raw, list[str])
    if len(layers) != count or any(
        layer not in ("full_attention", "linear_attention") for layer in layers
    ):
        raise ValueError(
            "layer_types must name one supported attention type per layer.",
        )
    return layers


def _full_attention(config: Mapping[str, object]) -> GatedAttention.Config:
    attention = GatedAttention.Config()
    attention.num_heads = _positive(config, name="num_attention_heads")
    attention.num_heads_kv = _positive(config, name="num_key_value_heads")
    if attention.num_heads % attention.num_heads_kv:
        raise ValueError(
            "num_attention_heads must be divisible by num_key_value_heads.",
        )
    attention.channels_head = _positive(config, name="head_dim")
    attention.bias = from_plain(config.get("attention_bias"), bool, default=False)
    attention.dropout = from_plain(config.get("attention_dropout"), float, default=0.0)
    if (
        not math.isfinite(attention.dropout)
        or attention.dropout < 0
        or attention.dropout >= 1
    ):
        raise ValueError("attention_dropout must be finite and in [0, 1).")
    if YarnScaling.Config.from_hf(config) is not None:
        raise ValueError("Only default text rotary frequencies are supported.")
    params = from_plain(config.get("rope_parameters"), dict[str, object], default={})
    fraction = from_plain(
        params.get("partial_rotary_factor"),
        float,
        default=from_plain(config.get("partial_rotary_factor"), float, default=0.25),
    )
    if not math.isfinite(fraction) or fraction <= 0 or fraction > 1:
        raise ValueError("partial_rotary_factor must be finite and in (0, 1].")
    width = int(attention.channels_head * fraction)
    if width < 2 or width % 2:
        raise ValueError("The rotary prefix must have a positive even width.")
    rope_theta = from_plain(
        params.get("rope_theta"),
        float,
        default=from_plain(config.get("rope_theta"), float, default=10_000_000.0),
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
    keys = attention_mask.shape[1]
    queries = x.shape[-2]
    causal = torch.arange(keys, device=x.device) <= (
        torch.arange(queries, device=x.device)[:, None] + keys - queries
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
