"""Pinned Qwen3.5 text-model reference checks on matching CPU kernels."""

from typing import TYPE_CHECKING, cast

import inspect

from torch import Tensor

import pytest
import torch

from priml.model.transformer.qwen3_5_weights import remap_hf_state_dict
from priml.testing.bfb import portable_half_precision
from priml.testing.qwen3_5 import (
    HfReference,
    hf_config,
    hf_logits,
    hf_reference,
    native_qwen35_config,
)


pytest.importorskip("transformers")


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


def _prepared_causal_mask(*, queries: int, keys: int) -> Tensor:
    """Return a text-only additive mask that never exposes future keys."""
    causal = torch.arange(keys) <= (
        torch.arange(queries).unsqueeze(-1) + keys - queries
    )
    return torch.zeros(1, 1, queries, keys).masked_fill(
        ~causal,
        torch.finfo(torch.float32).min,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_matching_kernel_model_outputs_and_gradients(dtype: torch.dtype) -> None:
    """Eager attention, separate SwiGLU matmuls and torch delta must match exactly."""
    raw_reference: object = Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(**hf_config(), attn_implementation="eager"),
    )
    reference = hf_reference(raw_reference, dtype=dtype)
    config = native_qwen35_config()
    native = config.make().to(dtype=dtype)
    native.load_state_dict(remap_hf_state_dict(reference.state_dict(), config))
    tokens = torch.tensor([[1, 3, 5, 7]])
    with portable_half_precision():
        expected = hf_logits(reference(tokens, use_cache=False))
        actual = native(tokens)
        assert torch.equal(actual, expected)
        expected.float().square().sum().backward()
        actual.float().square().sum().backward()
    reference_gradients: dict[str, Tensor] = {}
    for name, parameter in reference.named_parameters():
        assert parameter.grad is not None, name
        reference_gradients[name] = parameter.grad
    mapped = remap_hf_state_dict(reference_gradients, config)
    for name, parameter in native.named_parameters():
        assert parameter.grad is not None, name
        assert torch.equal(parameter.grad, mapped[name]), name


def test_padding_mask_and_positions_match_reference_prefill() -> None:
    """Route a text padding mask and explicit positions through both layer types."""
    raw_reference: object = Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(**hf_config(), attn_implementation="eager"),
    )
    reference = hf_reference(raw_reference)
    config = native_qwen35_config()
    native = config.make()
    native.load_state_dict(remap_hf_state_dict(reference.state_dict(), config))
    tokens = torch.tensor([[0, 0, 1, 3], [0, 1, 3, 5]])
    padding = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    positions = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 2]])
    expected = hf_logits(
        reference(
            tokens,
            attention_mask=padding,
            position_ids=positions,
            use_cache=False,
        ),
    )
    actual = native(tokens, attention_mask=padding, position_ids=positions)
    assert torch.equal(actual, expected)


def test_padding_mask_and_positions_match_reference_cached_continuation() -> None:
    """Keep full and delta layers aligned through a masked cached continuation."""
    raw_reference: object = Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(**hf_config(), attn_implementation="eager"),
    )
    reference = hf_reference(raw_reference)
    config = native_qwen35_config()
    native = config.make()
    native.load_state_dict(remap_hf_state_dict(reference.state_dict(), config))
    reference_cache: object = DynamicCache(config=reference.config)
    native_cache = native.alloc_cache(batch=2, max_seq=4)
    steps = (
        (
            torch.tensor([[0, 1], [1, 3]]),
            torch.tensor([[0, 1], [1, 1]]),
            torch.tensor([[0, 0], [0, 1]]),
        ),
        (
            torch.tensor([[3], [5]]),
            torch.tensor([[0, 1, 1], [1, 1, 1]]),
            torch.tensor([[1], [2]]),
        ),
    )
    for tokens, padding, positions in steps:
        expected = hf_logits(
            reference(
                tokens,
                attention_mask=padding,
                position_ids=positions,
                past_key_values=reference_cache,
                use_cache=True,
            ),
        )
        actual, returned_cache = native.forward_cached(
            tokens,
            cache=native_cache,
            attention_mask=padding,
            position_ids=positions,
        )
        assert returned_cache is native_cache
        assert torch.equal(actual, expected)


def test_prepared_causal_mask_matches_reference_prefill_and_cached_continuation() -> (
    None
):
    """A prepared causal 4-D text mask reaches full attention, never delta padding."""
    raw_reference: object = Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(**hf_config(), attn_implementation="eager"),
    )
    reference = hf_reference(raw_reference)
    config = native_qwen35_config()
    native = config.make()
    native.load_state_dict(remap_hf_state_dict(reference.state_dict(), config))

    tokens = torch.tensor([[1, 3, 5]])
    mask = _prepared_causal_mask(queries=3, keys=3)
    expected = hf_logits(reference(tokens, attention_mask=mask, use_cache=False))
    actual = native(tokens, attention_mask=mask)
    assert torch.equal(actual, expected)

    reference_cache: object = DynamicCache(config=reference.config)
    native_cache = native.alloc_cache(batch=1, max_seq=3)
    for tokens, mask in (
        (torch.tensor([[1, 3]]), _prepared_causal_mask(queries=2, keys=2)),
        (torch.tensor([[5]]), _prepared_causal_mask(queries=1, keys=3)),
    ):
        expected = hf_logits(
            reference(
                tokens,
                attention_mask=mask,
                past_key_values=reference_cache,
                use_cache=True,
            ),
        )
        actual, returned_cache = native.forward_cached(
            tokens,
            cache=native_cache,
            attention_mask=mask,
        )
        assert returned_cache is native_cache
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("cached", [False, True])
def test_boolean_prepared_masks_are_rejected_before_prefill_and_cache(
    cached: bool,
) -> None:
    """Require Qwen's documented additive 4-D mask representation."""
    raw_reference: object = Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(**hf_config(), attn_implementation="eager"),
    )
    reference = hf_reference(raw_reference)
    config = native_qwen35_config()
    native = config.make()
    native.load_state_dict(remap_hf_state_dict(reference.state_dict(), config=config))

    boolean = _prepared_causal_boolean_mask(queries=2, keys=2)
    numeric = boolean.to(torch.float32)
    tokens = torch.tensor([[1, 3]])
    assert torch.equal(
        hf_logits(reference(tokens, attention_mask=boolean)),
        hf_logits(reference(tokens, attention_mask=numeric)),
    )

    if cached:
        cache = native.alloc_cache(batch=1, max_seq=2)
        with pytest.raises(TypeError, match="floating additive"):
            native.forward_cached(tokens, cache=cache, attention_mask=boolean)
    else:
        with pytest.raises(TypeError, match="floating additive"):
            native(tokens, attention_mask=boolean)


class _StubReference:
    """Borrows every ``HfReference`` stub body, which must be inert."""

    __call__ = HfReference.__call__
    named_parameters = HfReference.named_parameters
    state_dict = HfReference.state_dict
    get_parameter = HfReference.get_parameter


def test_hf_reference_stub_bodies_are_inert() -> None:
    """The protocol narrows a lazy HF model; its own bodies compute nothing."""
    stub = _StubReference()
    assert stub(torch.zeros(1)) is None
    assert stub.named_parameters() is None
    assert stub.state_dict() is None
    assert stub.get_parameter("lm_head.weight") is None
    descriptor = cast(object, inspect.getattr_static(HfReference, "config"))
    assert isinstance(descriptor, property)
    assert descriptor.fget is not None
    assert descriptor.fget(stub) is None


def _prepared_causal_boolean_mask(*, queries: int, keys: int) -> Tensor:
    """Return the boolean representation of one text-only causal mask."""
    return (
        (torch.arange(keys) <= (torch.arange(queries).unsqueeze(-1) + keys - queries))
        .unsqueeze(0)
        .unsqueeze(0)
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
