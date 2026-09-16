"""Tests for the sampling helpers in :mod:`priml.model.generate`."""

from __future__ import annotations

from pathlib import Path
from typing import Final, override

from torch import Tensor, nn

import pytest
import torch

from priml.lib.custom_json import ListCodec
from priml.model.attention.kvcache import KVCache
from priml.model.generate import _sample, _topp_filter, generate
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.golden import assert_text_golden


_CWD: Final = Path(__file__).resolve().parent


def test_generate_public_contract(request: pytest.FixtureRequest) -> None:
    prompt = torch.tensor([[0, 1]])
    generated = _canonical_generate(model=_Transformer(), prompt=prompt)
    tokens = [ListCodec.coerce(row, int) for row in generated.tolist()]
    assert_text_golden(
        request,
        test_file=__file__,
        name="generate",
        rendered="\n".join(
            [
                "generate:",
                "  sampling: greedy",
                "  eos_token_id: 3",
                "  max_new_tokens: 4",
                f"  prompt: {tokens[0][:2]}",
                f"  tokens: {tokens[0]}",
            ],
        ),
    )


def test_generate_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="generate",
        build_module=_GenerateHarness,
        build_input=lambda: torch.tensor([[0, 1]]),
        seed=0,
    )


def test_sample_top_p_keeps_boundary_token():
    """Top-p must keep the smallest set whose cumulative prob reaches ``top_p``.

    Regression for GEN-TOPP (Issue#333): the nucleus mask used ``>=`` on
    the exclusive cumulative probability, dropping the token that brings
    the running mass exactly to ``top_p``. The HuggingFace convention uses
    strict ``>``. With a uniform 4-token distribution and ``top_p=0.5`` the
    exclusive cumsum is ``[0, .25, .5, .75]``; ``>=`` keeps only 2 tokens
    while the correct ``>`` keeps 3 (mass through the boundary token).
    """
    logits = torch.zeros(1, 4)  # `softmax` -> uniform 0.25 each.
    probs = _topp_probs(logits, top_p=0.5)
    kept = (probs > 0).sum(dim=-1).item()
    assert kept == 3


def test_sample_top_p_restores_vocab_order():
    """Filtered logits must map back to original vocab positions.

    Regression for GEN-TOPP (Issue#333): the surviving token's probability
    mass must land on its original vocab index, not a sorted position.
    """
    logits = torch.tensor([[1.0, 0.0, 9.0, 0.5, 0.2]])  # `argmax` at index 2.
    probs = _topp_probs(logits, top_p=0.5)
    assert probs.argmax(dim=-1).item() == 2


def test_sample_greedy_is_argmax():
    """Temperature 0 returns the argmax token id."""
    logits = torch.tensor([[1.0, 9.0, 0.5]])
    token = _sample(logits, temperature=0.0, top_k=0, top_p=1.0)
    assert token.item() == 1


def test_sample_applies_temperature_top_k_and_top_p() -> None:
    logits = torch.tensor([[1.0, 2.0, 9.0]])
    token = _sample(logits, temperature=2.0, top_k=1, top_p=0.5)
    assert token.item() == 2


def test_sample_float16_filters_excluded_tokens() -> None:
    """Filtering preserves eligible logits and gives excluded tokens zero mass."""
    logits = torch.tensor([[0.0, -1.0, -2.0]], dtype=torch.float16)

    filtered = _topp_filter(logits, top_p=0.8)
    token = _sample(logits, temperature=1.0, top_k=2, top_p=1.0)
    probs = filtered.softmax(dim=-1)

    assert torch.equal(filtered[0, :2], logits[0, :2])
    assert probs[0, 2] == 0
    assert token.item() in (0, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_topp_filter_masks_finite_floor_and_preserves_infinite_tails(
    dtype: torch.dtype,
) -> None:
    """Top-p must mask finite floors while preserving pre-masked tails."""
    floor_logits = torch.full((1, 2), torch.finfo(dtype).min, dtype=dtype)
    floor_probs = _topp_probs(floor_logits, top_p=0.4)
    assert torch.equal(floor_probs, torch.tensor([[1.0, 0.0]], dtype=dtype))

    tail_logits = torch.tensor([[0.0, float("-inf"), float("-inf")]], dtype=dtype)
    tail_probs = _topp_probs(tail_logits, top_p=0.4)
    assert torch.equal(tail_probs, torch.tensor([[1.0, 0.0, 0.0]], dtype=dtype))


def test_generate_pads_finished_rows_with_eos() -> None:
    """Finished batch rows emit EOS while active rows continue sampling."""
    model = _BatchTransformer()
    prompt = torch.tensor([[0, 1], [0, 1]])

    result = generate(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=3,
        temperature=0.0,
        eos_token_id=3,
        max_seq_len=5,
    )

    assert torch.equal(result, torch.tensor([[0, 1, 3, 3, 3], [0, 1, 2, 2, 2]]))


@pytest.mark.parametrize(
    ("name", "value", "match"),
    [
        ("max_new_tokens", -1, "max_new_tokens"),
        ("temperature", -0.1, "temperature"),
        ("top_k", -1, "top_k"),
        ("top_p", 0.0, "top_p"),
        ("top_p", 1.1, "top_p"),
        ("max_seq_len", 0, "max_seq_len"),
    ],
)
def test_generate_rejects_invalid_boundaries(
    name: str,
    value: float,
    match: str,
) -> None:
    """Generation rejects invalid public parameter boundaries."""
    with pytest.raises(ValueError, match=match):
        _generate_with_invalid_parameter(_Transformer(), name=name, value=value)


def _generate_with_invalid_parameter(
    model: _Transformer,
    *,
    name: str,
    value: float,
) -> Tensor:
    """Call generation with one typed invalid parameter."""
    prompt = torch.tensor([[0, 1]])
    if name == "max_new_tokens":
        return generate(model=model, prompt_ids=prompt, max_new_tokens=int(value))
    if name == "temperature":
        return generate(model=model, prompt_ids=prompt, temperature=float(value))
    if name == "top_k":
        return generate(model=model, prompt_ids=prompt, top_k=int(value))
    if name == "top_p":
        return generate(model=model, prompt_ids=prompt, top_p=float(value))
    if name == "max_seq_len":
        return generate(model=model, prompt_ids=prompt, max_seq_len=int(value))
    raise AssertionError(f"Unknown parameter: {name}")


@pytest.mark.parametrize(
    "prompt",
    [
        torch.empty((1, 0), dtype=torch.long),
        torch.empty((0, 2), dtype=torch.long),
        torch.tensor([0, 1]),
    ],
)
def test_generate_rejects_empty_or_malformed_prompt(prompt: Tensor) -> None:
    """Generation requires a non-empty two-dimensional prompt."""
    with pytest.raises(ValueError, match="prompt_ids"):
        generate(model=_Transformer(), prompt_ids=prompt)


def test_generate_rejects_prompt_and_generation_over_cache_capacity() -> None:
    """Generation rejects a requested sequence longer than the KV cache."""
    with pytest.raises(ValueError, match="prompt length 2 plus max_new_tokens=3"):
        generate(
            model=_Transformer(),
            prompt_ids=torch.tensor([[0, 1]]),
            max_new_tokens=3,
            max_seq_len=4,
        )


def test_generate_rejects_prompt_longer_than_cache() -> None:
    model = _Transformer()

    with pytest.raises(
        ValueError,
        match=r"prompt length 3 exceeds max_seq_len=2\.",
    ):
        generate(model=model, prompt_ids=torch.tensor([[0, 1, 2]]), max_seq_len=2)

    assert model.block.attn.cache is None


def test_generate_rejects_a_block_without_an_attn_attribute() -> None:
    with pytest.raises(TypeError, match="attn attribute"):
        generate(
            model=_TransformerWithBlock(_NoAttn()),
            prompt_ids=torch.tensor([[0, 1]]),
            max_new_tokens=1,
            max_seq_len=4,
        )


def test_generate_rejects_a_block_without_forward_cached() -> None:
    with pytest.raises(TypeError, match="forward_cached method"):
        generate(
            model=_TransformerWithBlock(_NoForwardCached()),
            prompt_ids=torch.tensor([[0, 1]]),
            max_new_tokens=1,
            max_seq_len=4,
        )


def test_generate_returns_prompt_when_no_tokens_requested() -> None:
    model = _Transformer()
    prompt = torch.tensor([[0, 1]])

    result = generate(model=model, prompt_ids=prompt, max_new_tokens=0, max_seq_len=4)

    assert result is prompt
    assert model.project_calls == 1
    assert model.block.attn.max_seq == 4


def test_generate_forwards_cache_metadata_and_stops_at_eos() -> None:
    model = _Transformer()
    prompt = torch.tensor([[0, 1]])

    result = generate(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=4,
        temperature=0.0,
        eos_token_id=3,
        max_seq_len=6,
    )

    assert torch.equal(result, torch.tensor([[0, 1, 2, 3]]))
    assert model.block.attn.batch == 1
    assert model.block.attn.max_seq == 6
    assert model.block.attn.device == prompt.device
    assert model.block.attn.dtype == model.proj_in.weight.dtype
    assert model.block.seen_caches == [model.block.attn.cache] * 2
    assert len(model.proj_in.inputs) == 2
    assert torch.equal(model.proj_in.inputs[0], prompt)
    assert torch.equal(model.proj_in.inputs[1], torch.tensor([[2]]))


# The nucleus filter sets out-of-nucleus logits to ``-inf``, which softmaxes to exactly
# 0, so the filtered softmax IS the kept distribution -- the same distribution
# ``_sample`` draws from, but without the 20k-iteration Monte-Carlo loop (or its
# sampling flakiness).
def _topp_probs(logits: Tensor, *, top_p: float) -> Tensor:
    """Recover the kept-token distribution under top-p, deterministically."""
    return _topp_filter(logits, top_p=top_p).softmax(dim=-1)


class _Lookup:
    def __init__(self) -> None:
        self.weight = torch.empty(4, 1)
        self.inputs: list[Tensor] = []

    def __call__(self, tokens: Tensor, /, **kwargs: object) -> Tensor:
        del kwargs
        self.inputs.append(tokens.clone())
        return tokens.unsqueeze(-1).float()

    def reset_parameters(self) -> None:
        pass

    def to(self, *, dtype: torch.dtype) -> _Lookup:
        self.weight = self.weight.to(dtype=dtype)
        return self


class _Attention:
    def __init__(self) -> None:
        self.batch: int | tuple[int, ...] | None = None
        self.max_seq: int | None = None
        self.device: torch.device | str | None = None
        self.dtype: torch.dtype | None = None
        self.cache: KVCache | None = None

    def alloc_kv_cache(
        self,
        *,
        batch: int | tuple[int, ...],
        max_seq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        self.batch = batch
        self.max_seq = max_seq
        self.device = device
        self.dtype = dtype
        self.cache = KVCache.alloc(
            batch=batch,
            num_heads=1,
            max_seq=max_seq,
            channels_head=1,
            device=device,
            dtype=dtype,
        )
        return self.cache


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()
        self.seen_caches: list[KVCache] = []

    def forward_cached(
        self,
        x: Tensor,
        /,
        *,
        cache: KVCache,
    ) -> tuple[Tensor, KVCache]:
        self.seen_caches.append(cache)
        return x, cache


class _NoAttn(nn.Module):
    """A block missing the ``attn`` attribute ``generate`` requires."""

    def forward_cached(self, x: Tensor, /, *, cache: object) -> tuple[Tensor, object]:
        return x, cache


class _NoForwardCached(nn.Module):
    """A block missing the ``forward_cached`` method ``generate`` requires."""

    def __init__(self) -> None:
        super().__init__()
        self.attn = _Attention()


class _Transformer:
    def __init__(self) -> None:
        self.proj_in = _Lookup()
        self.block = _Block()
        self.blocks: list[nn.Module] = [self.block]
        self.project_calls = 0

    def project_to_logits(self, hidden: Tensor, /) -> Tensor:
        next_token = (2, 3)[self.project_calls]
        self.project_calls += 1
        logits = torch.zeros(*hidden.shape[:-1], 4)
        logits[..., next_token] = 1
        return logits


class _TransformerWithBlock:
    """A minimal model wrapping one caller-supplied block."""

    def __init__(self, block: nn.Module) -> None:
        self.proj_in = _Lookup()
        self.blocks: list[nn.Module] = [block]

    def project_to_logits(self, hidden: Tensor, /) -> Tensor:
        return torch.zeros(*hidden.shape[:-1], 4)


class _BatchTransformer(_Transformer):
    @override
    def project_to_logits(self, hidden: Tensor, /) -> Tensor:
        logits = torch.full((*hidden.shape[:-1], 4), float("-inf"))
        if self.project_calls == 0:
            logits[0, ..., 3] = 1
            logits[1, ..., 2] = 1
        else:
            logits[0, ..., 1] = 1
            logits[1, ..., 2] = 1
        self.project_calls += 1
        return logits


class _GenerateHarness(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Transformer()

    @override
    def forward(self, prompt: Tensor) -> Tensor:
        return _canonical_generate(model=self.model, prompt=prompt)


def _canonical_generate(model: _Transformer, prompt: Tensor) -> Tensor:
    return generate(
        model=model,
        prompt_ids=prompt,
        max_new_tokens=4,
        temperature=0.0,
        eos_token_id=3,
        max_seq_len=6,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
