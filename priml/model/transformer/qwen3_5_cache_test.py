"""Qwen3.5 mixed attention-cache persistence checks."""

from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard, cast

import re

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


_KEY = ((0, 1),)


def _state(layer: dict[str, object]) -> dict[tuple[tuple[int, int], ...], object]:
    return {_KEY: layer}


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
        actual = model.forward(tokens, cache=cache)
        continuous = model.forward(tokens, cache=continuous_cache)
        assert torch.equal(actual, continuous)
        if split_gate_projection:
            assert reference_cache is not None
            expected = hf_logits(
                reference(tokens, past_key_values=reference_cache, use_cache=True),
            )
            assert torch.equal(actual, expected)
    state = cache_state_dict(cache)
    model.forward(torch.tensor([[7]]), cache=cache)
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
    continuous = model.forward(continuation, cache=continuous_cache)
    actual = model.forward(continuation, cache=restored)
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
        corrupted_actual = model.forward(continuation, cache=corrupted)
        assert not torch.equal(corrupted_actual, continuous)


def test_cache_restore_accepts_three_dimensional_full_attention_caches() -> None:
    """Three dimensions are valid when batch, head, and sequence axes exist."""
    k = torch.zeros(2, 3, 5)
    v = torch.ones_like(k)

    restored = cache_from_state_dict(
        _state({"kind": "full_attention", "k": k, "v": v, "length": 3, "seen": 3}),
    )
    restored_cache = restored[_KEY]
    assert isinstance(restored_cache, KVCache)
    assert torch.equal(restored_cache.k, k)
    assert torch.equal(restored_cache.v, v)
    assert restored_cache.length == 3
    assert restored_cache.seen == 3


@pytest.mark.parametrize(("length", "seen"), [(-1, 0), (4, 4)])
def test_cache_restore_rejects_invalid_full_attention_progress(
    length: int,
    seen: int,
) -> None:
    k = torch.zeros(2, 3, 5)
    v = torch.ones_like(k)

    with pytest.raises(ValueError, match="progress metadata") as error:
        cache_from_state_dict(
            _state(
                {
                    "kind": "full_attention",
                    "k": k,
                    "v": v,
                    "length": length,
                    "seen": seen,
                },
            ),
        )

    assert str(error.value) == "Full-attention cache progress metadata is invalid."


def test_cache_state_rejects_invalid_native_metadata() -> None:
    """Reject incomplete delta state and impossible full-cache progress."""
    with pytest.raises(TypeError) as error:
        cache_state_dict(_state({"conv_state": torch.zeros(1)}))
    assert (
        str(error.value) == "A Qwen3.5 cache layer must be KV or delta attention state."
    )
    with pytest.raises(TypeError) as error:
        cache_state_dict({_KEY: object()})
    assert (
        str(error.value) == "A Qwen3.5 cache layer must be KV or delta attention state."
    )
    with pytest.raises(TypeError) as error:
        cache_state_dict(_state({"conv_state": 1, "recurrent_state": torch.zeros(1)}))
    assert (
        str(error.value) == "A Qwen3.5 cache layer must be KV or delta attention state."
    )
    with pytest.raises(ValueError, match="progress metadata") as error:
        cache_from_state_dict(
            _state(
                {
                    "kind": "full_attention",
                    "k": torch.zeros(2, 3, 4, 5),
                    "v": torch.zeros(2, 3, 4, 5),
                    "length": 2,
                    "seen": 1,
                },
            ),
        )
    assert str(error.value) == "Full-attention cache progress metadata is invalid."


def test_full_attention_restores_own_independent_snapshot_tensors() -> None:
    """Restoring twice must not share full-attention tensors with the snapshot."""
    k = torch.zeros(2, 3, 4, 5)
    v = torch.zeros_like(k)
    state = _state({"kind": "full_attention", "k": k, "v": v, "length": 0, "seen": 0})

    first = cache_from_state_dict(state)
    second = cache_from_state_dict(state)
    first_cache, second_cache = first[_KEY], second[_KEY]
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
    conv_state = torch.zeros(2, 3, 4)
    recurrent_state = torch.zeros(2, 3, 4, 5)
    state = _state(
        {
            "kind": "linear_attention",
            "conv_state": conv_state,
            "recurrent_state": recurrent_state,
        },
    )

    first = cache_from_state_dict(state)
    second = cache_from_state_dict(state)
    first_cache, second_cache = first[_KEY], second[_KEY]
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
    source = KVCache.alloc(batch=2, num_heads=3, max_seq=8, channels_head=5)
    source.update(torch.ones(2, 3, 4, 5), torch.ones(2, 3, 4, 5))
    state = cache_state_dict({_KEY: source.freeze()})
    saved_k = state[_KEY]["k"]
    assert isinstance(saved_k, torch.Tensor)
    expected_k = saved_k.clone()

    source.k.fill_(3)
    frozen = source.freeze()
    source.update(torch.full((2, 3, 4, 5), 4), torch.full((2, 3, 4, 5), 4))
    assert torch.equal(frozen.k, source.k)
    assert frozen.length == 4
    assert frozen.seen == 4

    first = cache_from_state_dict(state)
    second = cache_from_state_dict(state)
    first_cache, second_cache = first[_KEY], second[_KEY]
    assert type(first_cache) is KVCache
    assert isinstance(second_cache, KVCache)
    first_cache.update(torch.full((2, 3, 4, 5), 2), torch.full((2, 3, 4, 5), 2))

    assert torch.equal(saved_k, expected_k)
    assert torch.equal(second_cache.k, expected_k)


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (None, "Qwen3.5 cache state must be a dictionary."),
        ([object()], "Qwen3.5 cache state must be a dictionary."),
        ({1: "full_attention"}, "Qwen3.5 cache keys must be nonempty depth indices."),
        (_state({"kind": 1}), "kind must be a str."),
        (
            _state({"kind": "full_attention", "k": 1, "v": 1, "length": 0, "seen": 0}),
            "k must be a Tensor.",
        ),
        (
            _state({"kind": "linear_attention", "conv_state": 1}),
            "A delta-attention cache must map strings to tensors.",
        ),
    ],
)
def test_cache_restore_rejects_invalid_state_containers_and_metadata_types(
    state: object,
    message: str,
) -> None:
    """Reject malformed persistence containers before interpreting their fields."""
    with pytest.raises(TypeError, match=message) as error:
        cache_from_state_dict(state)
    assert str(error.value) == message


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (
            _state({"kind": "full_attention", "k": torch.zeros(1)}),
            "A full-attention cache has an invalid field set.",
        ),
        (
            _state({"kind": "linear_attention", "conv_state": torch.zeros(1)}),
            "A delta-attention cache has an invalid field set.",
        ),
        (
            _state({"kind": "sliding_attention"}),
            "A Qwen3.5 cache layer has an unknown kind.",
        ),
    ],
)
def test_cache_restore_rejects_invalid_layer_kinds_and_field_sets(
    state: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message) as error:
        cache_from_state_dict(state)
    assert str(error.value) == message


def test_an_empty_delta_cache_round_trips() -> None:
    """A freshly allocated delta layer holds no tensors yet and still persists."""
    state = cache_state_dict({_KEY: {}})
    assert state == {_KEY: {"kind": "linear_attention"}}
    assert cache_from_state_dict(state) == {_KEY: {}}


@pytest.mark.parametrize("name", ["length", "seen"])
def test_cache_restore_rejects_bool_full_attention_metadata(name: str) -> None:
    """Bool is not valid full-attention progress metadata at restore time."""
    state = _state(
        {
            "kind": "full_attention",
            "k": torch.zeros(2, 3, 4, 5),
            "v": torch.zeros(2, 3, 4, 5),
            "length": 0,
            "seen": 0,
        },
    )
    layer = state[_KEY]
    assert isinstance(layer, dict)
    layer[name] = True

    with pytest.raises(TypeError, match=f"{name} must be an int") as error:
        cache_from_state_dict(state)
    assert str(error.value) == f"{name} must be an int."


@pytest.mark.parametrize("name", ["length", "seen"])
def test_cache_state_rejects_bool_full_attention_metadata(name: str) -> None:
    """Bool is not valid full-attention progress metadata at serialization time."""
    cache = KVCache.alloc(batch=2, num_heads=3, max_seq=4, channels_head=5)
    if name == "length":
        cache.length = True
    else:
        cache.seen = True

    with pytest.raises(
        TypeError,
        match="Full-attention cache progress metadata must be integers",
    ) as error:
        cache_state_dict({_KEY: cache})
    assert (
        str(error.value) == "Full-attention cache progress metadata must be integers."
    )


@pytest.mark.parametrize(
    ("k", "v"),
    [
        (torch.zeros(2, 4), torch.zeros(2, 4)),
        (torch.zeros(2, 3, 4, 5), torch.zeros(2, 3, 5, 6)),
        (
            torch.zeros(2, 3, 4, 5, dtype=torch.float32),
            torch.zeros(2, 3, 4, 5, dtype=torch.float64),
        ),
        (
            torch.zeros(2, 3, 4, 5),
            torch.zeros(2, 3, 4, 5, device="meta"),
        ),
    ],
)
def test_cache_restore_rejects_invalid_kv_layout_or_dtype(
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Reject serialized full-attention tensors with invalid layout or dtype."""
    with pytest.raises(
        ValueError,
        match="Full-attention key and value caches have incompatible shapes",
    ) as error:
        cache_from_state_dict(
            _state({"kind": "full_attention", "k": k, "v": v, "length": 0, "seen": 0}),
        )
    assert (
        str(error.value)
        == "Full-attention key and value caches have incompatible shapes, dtypes, or devices."
    )


def _delta_layer(
    *,
    conv_state: torch.Tensor | None = None,
    recurrent_state: torch.Tensor | None = None,
) -> dict[str, object]:
    """One serialized delta layer; batch 2, 3 conv channels, 4-wide kernel."""
    return {
        "kind": "linear_attention",
        "conv_state": torch.zeros(2, 3, 4) if conv_state is None else conv_state,
        "recurrent_state": (
            torch.zeros(2, 5, 6, 7) if recurrent_state is None else recurrent_state
        ),
    }


@pytest.mark.parametrize(
    "layer",
    [
        _delta_layer(recurrent_state=torch.zeros(2, 5, 6, 7, dtype=torch.float64)),
        _delta_layer(conv_state=torch.zeros(2, 3, 4, dtype=torch.int64)),
        _delta_layer(conv_state=torch.zeros(3, 3, 4)),
        _delta_layer(conv_state=torch.zeros(2, 3)),
        _delta_layer(recurrent_state=torch.zeros(2, 5, 6)),
        _delta_layer(conv_state=torch.zeros(2, 3, 4, device="meta")),
    ],
    ids=[
        "recurrent-dtype",
        "conv-dtype",
        "batch",
        "conv-rank",
        "recurrent-rank",
        "device",
    ],
)
def test_delta_restore_and_save_reject_incompatible_tensors(
    layer: dict[str, object],
) -> None:
    """Delta tensors get the same boundary checks as full-attention ones."""
    message = (
        "Delta-attention conv and recurrent states have incompatible shapes, "
        "dtypes, or devices."
    )
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        cache_from_state_dict(_state(layer))
    assert str(error.value) == message
    live = {name: value for name, value in layer.items() if name != "kind"}
    with pytest.raises(ValueError, match=re.escape(message)) as error:
        cache_state_dict(_state(live))
    assert str(error.value) == message


def test_bfloat16_conv_state_round_trips() -> None:
    """The conv state follows the model dtype; only the recurrence is float32."""
    layer = _delta_layer(conv_state=torch.zeros(2, 3, 4, dtype=torch.bfloat16))
    restored = cache_from_state_dict(_state(layer))
    conv_state = cache_state_dict(restored)[_KEY]["conv_state"]
    assert isinstance(conv_state, torch.Tensor)
    assert conv_state.dtype == torch.bfloat16


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
    for layer in state.values():
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
