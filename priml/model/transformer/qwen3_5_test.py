"""CPU-tiny tests for the native Qwen3.5 hybrid text language model."""

from __future__ import annotations

from typing import override

from configgle import Fig
from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.self_attention import SelfAttention
from priml.model.generate import generate
from priml.model.norm import CenteredRMSNorm
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.qwen3_5 import Qwen35
from priml.testing.qwen3_5 import hf_config


def _prepared_causal_mask(*, queries: int, keys: int) -> torch.Tensor:
    """Return a text-only additive mask that never exposes future keys."""
    causal = torch.arange(keys) <= (
        torch.arange(queries).unsqueeze(-1) + keys - queries
    )
    return torch.zeros(1, 1, queries, keys).masked_fill(
        ~causal,
        torch.finfo(torch.float32).min,
    )


def test_qwen35_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="qwen3_5",
        config=Qwen35.Config.from_hf(hf_config()),
    )


def test_hybrid_config_preserves_injected_blocks() -> None:
    config = Qwen35.Config.from_hf(hf_config())
    assert isinstance(config.block, list)
    assert len(config.block) == 2
    assert isinstance(config.block[0], TransformerBlock.Config)
    config.block[0].checkpoint = True
    finalized = config.copy_tree().finalize()
    assert isinstance(finalized.block, list)
    assert isinstance(finalized.block[0], TransformerBlock.Config)
    assert finalized.block[0].checkpoint
    model = config.make()
    assert model(torch.tensor([[1, 2, 3]])).shape == (1, 3, 32)
    assert model.hidden_states(torch.tensor([[1, 2, 3]])).shape == (1, 3, 16)


def test_hidden_states_rejects_token_ids_without_input_embedding() -> None:
    config = Qwen35.Config.from_hf(hf_config())
    config.in_proj = None
    model = config.make()

    with pytest.raises(ValueError, match="Token IDs require an input embedding"):
        model.hidden_states(torch.tensor([[1, 2, 3]]))


def test_hidden_states_rejects_a_cache_with_the_wrong_number_of_entries() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()

    with pytest.raises(ValueError, match="one entry per transformer block"):
        model.hidden_states(torch.tensor([[1, 2, 3]]), cache=[])


def test_hidden_states_rejects_positions_and_position_ids_together() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()

    with pytest.raises(ValueError, match="Pass either positions or position_ids"):
        model(
            torch.tensor([[1, 2, 3]]),
            positions=torch.arange(3),
            position_ids=torch.arange(3).reshape(1, 3),
        )


def test_forward_rejects_a_non_list_cache() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()

    with pytest.raises(TypeError, match="cache must be a list or None"):
        model(torch.tensor([[1, 2, 3]]), cache="not-a-list")


def test_hidden_states_rejects_an_attention_mask_that_is_neither_2d_nor_4d() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()

    with pytest.raises(ValueError, match="2-D padding mask or 4-D mask"):
        model(torch.tensor([[1, 2, 3]]), attention_mask=torch.zeros(1, 3, 3))


def test_final_norm_width_inference_respects_explicit_configuration() -> None:
    """Final norms infer the model width without rewriting explicit configurations."""
    inferred = Qwen35.Config.from_hf(hf_config()).copy_tree().finalize()
    assert isinstance(inferred.norm, CenteredRMSNorm.Config)
    assert inferred.norm.channels_in == 16

    matching = Qwen35.Config.from_hf(hf_config())
    matching_norm = CenteredRMSNorm.Config()
    matching_norm.channels_in = 16
    matching.norm = matching_norm
    finalized = matching.copy_tree().finalize()
    assert isinstance(finalized.norm, CenteredRMSNorm.Config)
    assert finalized.norm.channels_in == 16

    incompatible = Qwen35.Config.from_hf(hf_config())
    incompatible_norm = CenteredRMSNorm.Config()
    incompatible_norm.channels_in = 8
    incompatible.norm = incompatible_norm
    with pytest.raises(
        ValueError,
        match=r"norm\.channels_in=8 must equal channels_in=16",
    ):
        incompatible.make()


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model_type", "qwen3_5_moe"),
        ("hidden_act", "gelu"),
        ("layer_types", ["linear_attention"]),
        ("num_attention_heads", 0),
        ("initializer_range", -0.02),
        ("num_key_value_heads", 3),
    ],
)
def test_unsupported_config_is_rejected(key: str, value: object) -> None:
    config = hf_config()
    config[key] = value
    with pytest.raises((ValueError, TypeError)):
        Qwen35.Config.from_hf(config)


def test_unsupported_delta_head_ratio_is_rejected() -> None:
    config = hf_config()
    config["linear_num_key_heads"] = 2
    config["linear_num_value_heads"] = 3

    with pytest.raises(ValueError, match="linear_num_value_heads"):
        Qwen35.Config.from_hf(config)


def test_unsupported_rotary_base_is_rejected() -> None:
    config = hf_config()
    rope = config["rope_parameters"]
    assert isinstance(rope, dict)
    rope["rope_theta"] = -1.0

    with pytest.raises(ValueError, match="rope_theta"):
        Qwen35.Config.from_hf(config)


def test_hybrid_forwards_messages_to_injected_attention() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make()
    recorders: list[_MessageRecorder] = []
    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        recorder = _MessageRecorder()
        block.attn = recorder
        recorders.append(recorder)
    attention_mask = torch.tensor([[0, 1, 1]])
    position_ids = torch.tensor([[0, 0, 1]])

    model(
        torch.tensor([[1, 2, 3]]),
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    for recorder in recorders:
        assert recorder.messages == [(attention_mask, position_ids)]


def test_hybrid_forwards_messages_to_injected_output_head_and_cache() -> None:
    """Preserve output-head messages across direct and cached Qwen calls."""
    model = Qwen35.Config.from_hf(hf_config()).make()
    recorder = _HeadMessageRecorder()
    model.out_proj = recorder
    message = torch.tensor([7])

    model(torch.tensor([[1, 2]]), marker=message)
    cache = model.alloc_cache(batch=1, max_seq=2)
    model.forward_cached(torch.tensor([[1, 2]]), cache=cache, marker=message)
    model.project_to_logits(torch.ones(1, 1, 16), marker=message)

    assert len(recorder.messages) == 3
    assert recorder.messages[0] is message
    assert recorder.messages[1] is message
    assert recorder.messages[2] is message


def test_prepared_causal_mask_reaches_injected_self_attention_prefill_and_cache() -> (
    None
):
    """A prepared causal 4-D text mask is forwarded through an injected attention."""
    config = Qwen35.Config.from_hf(hf_config())
    assert isinstance(config.block, list)
    block = config.block[1]
    assert isinstance(block, TransformerBlock.Config)
    attention = SelfAttention.Config()
    attention.num_heads = 2
    attention.num_heads_kv = 1
    attention.channels_head = 8
    attention.attn_kernel = SdpaNaive.Config()
    block.attn = attention
    model = config.make().eval()

    prompt = torch.tensor([[1, 3, 5]])
    prompt_mask = _prepared_causal_mask(queries=3, keys=3)
    assert not torch.equal(
        model(prompt),
        model(prompt, attention_mask=prompt_mask),
    )

    plain_cache = model.alloc_cache(batch=1, max_seq=3)
    masked_cache = model.alloc_cache(batch=1, max_seq=3)
    prefix = torch.tensor([[1, 3]])
    prefix_mask = _prepared_causal_mask(queries=2, keys=2)
    plain_prefix, _ = model.forward_cached(prefix, cache=plain_cache)
    masked_prefix, _ = model.forward_cached(
        prefix,
        cache=masked_cache,
        attention_mask=prefix_mask,
    )
    assert not torch.equal(plain_prefix, masked_prefix)

    continuation_mask = _prepared_causal_mask(queries=1, keys=3)
    continuation_mask[..., 0] = torch.finfo(continuation_mask.dtype).min
    continuation = torch.tensor([[5]])
    plain_continuation, _ = model.forward_cached(
        continuation,
        cache=plain_cache,
    )
    masked_continuation, _ = model.forward_cached(
        continuation,
        cache=masked_cache,
        attention_mask=continuation_mask,
    )
    assert not torch.equal(plain_continuation, masked_continuation)


def test_hybrid_preserves_caller_attn_mask_precedence() -> None:
    """An explicit additive mask wins over a 2-D padding mask derived mask."""
    model = Qwen35.Config.from_hf(hf_config()).make()
    recorders: list[_MaskMessageRecorder] = []
    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        recorder = _MaskMessageRecorder()
        block.attn = recorder
        recorders.append(recorder)
    padding = torch.tensor([[0, 1, 1]])
    explicit = torch.full((1, 1, 3, 3), -7.0)

    model(torch.tensor([[1, 2, 3]]), attention_mask=padding, attn_mask=explicit)

    for recorder in recorders:
        assert recorder.messages == [(padding, explicit)]


def test_native_generation_uses_hybrid_cache() -> None:
    model = Qwen35.Config.from_hf(hf_config()).make().eval()
    prompt = torch.tensor([[1, 2, 3]])
    output = generate(model, prompt, max_new_tokens=2, temperature=0, max_seq_len=8)
    expected = prompt
    with torch.no_grad():
        for _ in range(2):
            token = model(expected)[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat([expected, token], dim=-1)
    assert torch.equal(output, expected)


def test_hybrid_cached_dispatch_accepts_injected_cached_block() -> None:
    """An injected block can opt into cache handling structurally."""
    config = Qwen35.Config.from_hf(hf_config())
    assert isinstance(config.block, list)
    config.block[0] = _CachedIdentityBlock.Config()
    model = config.make().eval()
    tokens = torch.tensor([[1, 2]])

    cache = model.alloc_cache(batch=1, max_seq=tokens.shape[-1])
    actual, returned = model.forward_cached(tokens, cache=cache)

    assert actual.shape == (1, 2, 32)
    assert returned is cache
    assert cache[0] == {}


def test_forward_cached_rejects_a_block_without_forward_cached() -> None:
    config = Qwen35.Config.from_hf(hf_config())
    assert isinstance(config.block, list)
    config.block[0] = _AttnOnlyBlock.Config()
    model = config.make().eval()
    cache = model.alloc_cache(batch=1, max_seq=2)

    with pytest.raises(TypeError, match="forward_cached method"):
        model.forward_cached(torch.tensor([[1, 2]]), cache=cache)


def test_alloc_cache_rejects_a_block_without_an_attn_submodule() -> None:
    config = Qwen35.Config.from_hf(hf_config())
    assert isinstance(config.block, list)
    config.block[0] = _UncachedIdentityBlock.Config()
    model = config.make()

    with pytest.raises(TypeError, match="attn submodule"):
        model.alloc_cache(batch=1, max_seq=2)


def test_alloc_cache_rejects_attn_without_alloc_kv_cache() -> None:
    config = Qwen35.Config.from_hf(hf_config())
    assert isinstance(config.block, list)
    config.block[0] = _NonCacheableAttentionBlock.Config()
    model = config.make()

    with pytest.raises(TypeError, match="alloc_kv_cache method"):
        model.alloc_cache(batch=1, max_seq=2)


class _CachedIdentityAttention(torch.nn.Module):
    """Allocate a cache for an injected identity block."""

    def alloc_kv_cache(
        self,
        *,
        batch: int | tuple[int, ...],
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, int]:
        del batch, max_seq, device, dtype
        return {}


class _CachedIdentityBlock(torch.nn.Module):
    """A Configgle-injectable cached block preserving hidden states."""

    class Config(Fig["_CachedIdentityBlock"]):
        channels_in: int = -1
        """Input feature width."""

        channels_out: int = -1
        """Output feature width."""

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        self.attn = _CachedIdentityAttention()

    def reset_parameters(self) -> None:
        """Leave the parameter-free test block unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return x

    def forward_cached(
        self,
        x: torch.Tensor,
        *,
        cache: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, object]:
        del kwargs
        return x, cache


class _UncachedIdentityBlock(torch.nn.Module):
    """A Configgle-injectable block missing both attn and forward_cached."""

    class Config(Fig["_UncachedIdentityBlock"]):
        channels_in: int = -1
        """Input feature width."""

        channels_out: int = -1
        """Output feature width."""

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    def reset_parameters(self) -> None:
        """Leave the parameter-free test block unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return x


class _AttnOnlyBlock(torch.nn.Module):
    """A Configgle-injectable block with attn but no forward_cached."""

    class Config(Fig["_AttnOnlyBlock"]):
        channels_in: int = -1
        """Input feature width."""

        channels_out: int = -1
        """Output feature width."""

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        self.attn = _CachedIdentityAttention()

    def reset_parameters(self) -> None:
        """Leave the parameter-free test block unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return x


class _NonCacheableAttention(torch.nn.Module):
    """An attn submodule missing alloc_kv_cache."""

    def reset_parameters(self) -> None:
        """Leave the parameter-free test attention unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return x


class _NonCacheableAttentionBlock(torch.nn.Module):
    """A Configgle-injectable block whose attn lacks alloc_kv_cache."""

    class Config(Fig["_NonCacheableAttentionBlock"]):
        channels_in: int = -1
        """Input feature width."""

        channels_out: int = -1
        """Output feature width."""

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        self.attn = _NonCacheableAttention()

    def reset_parameters(self) -> None:
        """Leave the parameter-free test block unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        del kwargs
        return x


class _MessageRecorder(torch.nn.Module):
    """Record public attention messages while preserving the residual stream."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[tuple[torch.Tensor, torch.Tensor]] = []

    def reset_parameters(self) -> None:
        """Leave the parameter-free recorder unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        attention_mask = kwargs["attention_mask"]
        position_ids = kwargs["position_ids"]
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(position_ids, torch.Tensor)
        self.messages.append((attention_mask, position_ids))
        return x


class _HeadMessageRecorder(torch.nn.Module):
    """Record output-head messages while preserving normalized hidden states."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[object | None] = []

    def reset_parameters(self) -> None:
        """Leave the parameter-free recorder unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.messages.append(kwargs.get("marker"))
        return x


class _MaskMessageRecorder(torch.nn.Module):
    """Record the two public mask messages while preserving the residual stream."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[tuple[torch.Tensor, torch.Tensor]] = []

    def reset_parameters(self) -> None:
        """Leave the parameter-free recorder unchanged."""

    @override
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        attention_mask = kwargs["attention_mask"]
        attn_mask = kwargs["attn_mask"]
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(attn_mask, torch.Tensor)
        self.messages.append((attention_mask, attn_mask))
        return x


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
