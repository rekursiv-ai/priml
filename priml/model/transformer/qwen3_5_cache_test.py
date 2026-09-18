"""Qwen3.5 mixed attention-cache persistence checks."""

from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard, cast

import pytest
import torch

from priml.model.attention.kvcache import KVCache
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.qwen3_5 import Qwen35
from priml.model.transformer.qwen3_5_cache import (
    Qwen35CacheState,
    cache_from_state_dict,
    cache_state_dict,
)
from priml.model.transformer.qwen3_5_weights import remap_hf_state_dict
from priml.testing.qwen3_5 import (
    HfReference,
    hf_config,
    hf_logits,
    hf_reference,
    native_qwen35_config,
)


if TYPE_CHECKING:
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
else:
    from wrapt import lazy_import

    DynamicCache = lazy_import("transformers.cache_utils", "DynamicCache")
    Qwen3_5TextConfig = lazy_import(
        "transformers.models.qwen3_5.configuration_qwen3_5",
        "Qwen3_5TextConfig",
    )
    Qwen3_5ForCausalLM = lazy_import(
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "Qwen3_5ForCausalLM",
    )


@pytest.mark.parametrize("split_gate_projection", [False, True], ids=["fused", "split"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mixed_cache_weights_only_roundtrip_resumes_exactly(
    tmp_path: Path,
    dtype: torch.dtype,
    split_gate_projection: bool,
) -> None:
    """Resume native mixed attention exactly after a safe cache roundtrip."""
    pytest.importorskip("transformers")
    reference, model = _models(dtype, split_gate_projection=split_gate_projection)
    reference_cache = (
        DynamicCache(config=reference.config) if split_gate_projection else None
    )
    cache = model.alloc_cache(batch=1, max_seq=6, dtype=dtype)
    continuous_cache = model.alloc_cache(batch=1, max_seq=6, dtype=dtype)
    for tokens in (torch.tensor([[1, 3]]), torch.tensor([[5]])):
        actual, _ = model.forward_cached(tokens, cache=cache)
        continuous, _ = model.forward_cached(tokens, cache=continuous_cache)
        assert torch.equal(actual, continuous)
        if split_gate_projection:
            assert reference_cache is not None
            expected = hf_logits(
                reference(tokens, past_key_values=reference_cache, use_cache=True),
            )
            assert torch.equal(actual, expected)
    state = cache_state_dict(cache)
    model.forward_cached(torch.tensor([[7]]), cache=cache)
    path = tmp_path / "cache.pt"
    torch.save(state, path)
    restored_state = cast(object, torch.load(path, weights_only=True))
    restored = cache_from_state_dict(restored_state)
    corrupted = None
    if not split_gate_projection:
        corrupted_state = cache_state_dict(restored)
        _corrupt_full_attention_value(corrupted_state)
        corrupted = cache_from_state_dict(corrupted_state)
    continuation = torch.tensor([[7, 9]])
    continuous, _ = model.forward_cached(continuation, cache=continuous_cache)
    actual, _ = model.forward_cached(continuation, cache=restored)
    assert torch.equal(actual, continuous)
    if split_gate_projection:
        assert reference_cache is not None
        expected = hf_logits(
            reference(
                continuation,
                past_key_values=reference_cache,
                use_cache=True,
            ),
        )
        assert torch.equal(actual, expected)
    else:
        assert corrupted is not None
        corrupted_actual, _ = model.forward_cached(continuation, cache=corrupted)
        assert not torch.equal(corrupted_actual, continuous)


def test_cache_state_rejects_invalid_native_metadata() -> None:
    """Reject incomplete delta state and impossible full-cache progress."""
    with pytest.raises(TypeError, match="KV or delta"):
        cache_state_dict([{"conv_state": torch.zeros(1)}])
    with pytest.raises(TypeError, match="KV or delta"):
        cache_state_dict([object()])
    with pytest.raises(TypeError, match="KV or delta"):
        cache_state_dict([{"conv_state": 1, "recurrent_state": torch.zeros(1)}])
    with pytest.raises(ValueError, match="progress metadata"):
        cache_from_state_dict(
            [
                {
                    "kind": "full_attention",
                    "k": torch.zeros(1, 1, 2, 4),
                    "v": torch.zeros(1, 1, 2, 4),
                    "length": 2,
                    "seen": 1,
                },
            ],
        )


def test_full_attention_restores_own_independent_snapshot_tensors() -> None:
    """Restoring twice must not share full-attention tensors with the snapshot."""
    k = torch.zeros(1, 1, 2, 4)
    v = torch.zeros_like(k)
    state = [{"kind": "full_attention", "k": k, "v": v, "length": 0, "seen": 0}]

    first = cache_from_state_dict(state)
    second = cache_from_state_dict(state)
    first_cache, second_cache = first[0], second[0]
    assert isinstance(first_cache, KVCache)
    assert isinstance(second_cache, KVCache)

    first_cache.k.fill_(1)
    first_cache.v.fill_(2)

    assert torch.equal(k, torch.zeros_like(k))
    assert torch.equal(v, torch.zeros_like(v))
    assert torch.equal(second_cache.k, torch.zeros_like(k))
    assert torch.equal(second_cache.v, torch.zeros_like(v))


def test_linear_attention_restores_own_independent_snapshot_tensors() -> None:
    """Restoring twice must not share delta-attention tensors with the snapshot."""
    conv_state = torch.zeros(1, 2, 3)
    recurrent_state = torch.zeros(1, 2, 3, 4)
    state = [
        {
            "kind": "linear_attention",
            "conv_state": conv_state,
            "recurrent_state": recurrent_state,
        },
    ]

    first = cache_from_state_dict(state)
    second = cache_from_state_dict(state)
    first_cache, second_cache = first[0], second[0]
    assert _is_object_dict(first_cache)
    assert _is_object_dict(second_cache)
    first_conv = first_cache["conv_state"]
    first_recurrent = first_cache["recurrent_state"]
    second_conv = second_cache["conv_state"]
    second_recurrent = second_cache["recurrent_state"]
    assert isinstance(first_conv, torch.Tensor)
    assert isinstance(first_recurrent, torch.Tensor)
    assert isinstance(second_conv, torch.Tensor)
    assert isinstance(second_recurrent, torch.Tensor)

    first_conv.fill_(1)
    first_recurrent.fill_(2)

    assert torch.equal(conv_state, torch.zeros_like(conv_state))
    assert torch.equal(recurrent_state, torch.zeros_like(recurrent_state))
    assert torch.equal(second_conv, torch.zeros_like(conv_state))
    assert torch.equal(second_recurrent, torch.zeros_like(recurrent_state))


def test_frozen_view_serializes_as_independent_mutable_cache() -> None:
    """A frozen view captures progress while persistence clones its tensors."""
    source = KVCache.alloc(batch=1, num_heads=1, max_seq=2, channels_head=4)
    source.update(torch.ones(1, 1, 1, 4), torch.ones(1, 1, 1, 4))
    state = cache_state_dict([source.freeze()])
    saved_k = state[0]["k"]
    assert isinstance(saved_k, torch.Tensor)
    expected_k = saved_k.clone()

    source.k.fill_(3)
    frozen = source.freeze()
    source.update(torch.full((1, 1, 1, 4), 4), torch.full((1, 1, 1, 4), 4))
    assert torch.equal(frozen.k, source.k)
    assert frozen.length == 1
    assert frozen.seen == 1

    first = cache_from_state_dict(state)
    second = cache_from_state_dict(state)
    first_cache, second_cache = first[0], second[0]
    assert type(first_cache) is KVCache
    assert isinstance(second_cache, KVCache)
    first_cache.update(torch.full((1, 1, 1, 4), 2), torch.full((1, 1, 1, 4), 2))

    assert torch.equal(saved_k, expected_k)
    assert torch.equal(second_cache.k, expected_k)


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (None, "Qwen3.5 cache state must be a list"),
        ([object()], "Qwen3.5 cache layer must be a dictionary"),
        ([{1: "full_attention"}], "Qwen3.5 cache layer keys must be strings"),
        ([{"kind": 1}], "kind must be a str"),
        (
            [{"kind": "full_attention", "k": 1, "v": 1, "length": 0, "seen": 0}],
            "k must be a Tensor",
        ),
        (
            [{"kind": "linear_attention", "conv_state": 1}],
            "delta-attention cache must map strings to tensors",
        ),
    ],
)
def test_cache_restore_rejects_invalid_state_containers_and_metadata_types(
    state: object,
    message: str,
) -> None:
    """Reject malformed persistence containers before interpreting their fields."""
    with pytest.raises(TypeError, match=message):
        cache_from_state_dict(state)


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ([{"kind": "full_attention", "k": torch.zeros(1)}], "invalid field set"),
        (
            [{"kind": "linear_attention", "conv_state": torch.zeros(1)}],
            "invalid field set",
        ),
        ([{"kind": "sliding_attention"}], "unknown kind"),
    ],
)
def test_cache_restore_rejects_invalid_layer_kinds_and_field_sets(
    state: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        cache_from_state_dict(state)


def test_an_empty_delta_cache_round_trips() -> None:
    """A freshly allocated delta layer holds no tensors yet and still persists."""
    state = cache_state_dict([{}])
    assert state == [{"kind": "linear_attention"}]
    assert cache_from_state_dict(state) == [{}]


@pytest.mark.parametrize("name", ["length", "seen"])
def test_cache_restore_rejects_bool_full_attention_metadata(name: str) -> None:
    """Bool is not valid full-attention progress metadata at restore time."""
    state: list[dict[str, object]] = [
        {
            "kind": "full_attention",
            "k": torch.zeros(1, 1, 2, 4),
            "v": torch.zeros(1, 1, 2, 4),
            "length": 0,
            "seen": 0,
        },
    ]
    state[0][name] = True

    with pytest.raises(TypeError, match=f"{name} must be an int"):
        cache_from_state_dict(state)


@pytest.mark.parametrize("name", ["length", "seen"])
def test_cache_state_rejects_bool_full_attention_metadata(name: str) -> None:
    """Bool is not valid full-attention progress metadata at serialization time."""
    cache = KVCache.alloc(batch=1, num_heads=1, max_seq=2, channels_head=4)
    if name == "length":
        cache.length = True
    else:
        cache.seen = True

    with pytest.raises(TypeError, match="progress metadata must be integers"):
        cache_state_dict([cache])


@pytest.mark.parametrize(
    ("k", "v"),
    [
        (torch.zeros(2, 4), torch.zeros(2, 4)),
        (torch.zeros(1, 1, 2, 4), torch.zeros(1, 1, 3, 4)),
        (
            torch.zeros(1, 1, 2, 4, dtype=torch.float32),
            torch.zeros(1, 1, 2, 4, dtype=torch.float64),
        ),
    ],
)
def test_cache_restore_rejects_invalid_kv_layout_or_dtype(
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Reject serialized full-attention tensors with invalid layout or dtype."""
    with pytest.raises(ValueError, match="incompatible shapes, dtypes, or devices"):
        cache_from_state_dict(
            [{"kind": "full_attention", "k": k, "v": v, "length": 0, "seen": 0}],
        )


def _models(
    dtype: torch.dtype,
    *,
    split_gate_projection: bool,
) -> tuple[HfReference, Qwen35]:
    """Build matching native and pinned eager/pure-Torch CPU reference models."""
    raw_reference: object = Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(**hf_config(), attn_implementation="eager"),
    )
    reference = hf_reference(raw_reference, dtype=dtype)
    config = (
        native_qwen35_config()
        if split_gate_projection
        else Qwen35.Config.from_hf(hf_config())
    )
    assert isinstance(config.block, list)
    for block in config.block:
        assert isinstance(block, TransformerBlock.Config)
        assert isinstance(block.ffn, SwiGLU.Config)
        assert block.ffn.split_gate_projection is split_gate_projection
    native = config.make().to(dtype=dtype)
    assert isinstance(native, Qwen35)
    native.load_state_dict(remap_hf_state_dict(reference.state_dict(), config))
    return reference, native


def _corrupt_full_attention_value(state: Qwen35CacheState) -> None:
    """Change one restored full-attention value for the negative control."""
    for layer in state:
        if layer["kind"] == "full_attention":
            value = layer["v"]
            assert isinstance(value, torch.Tensor)
            value[..., 0, 0].add_(1)
            return
    raise AssertionError("Expected a full-attention cache layer.")


def _is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    """Narrow a restored dynamic cache dictionary for this boundary test."""
    return isinstance(value, dict)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
