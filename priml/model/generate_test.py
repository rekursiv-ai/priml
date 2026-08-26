"""Tests for the sampling helpers in :mod:`priml.model.generate`."""

from __future__ import annotations

from pathlib import Path
from typing import cast, override

from torch import Tensor, nn

import pytest
import torch

from priml.model.attention.kvcache import KVCache
from priml.model.generate import _sample, _topp_filter, generate
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.golden import assert_text_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def test_generate_public_contract(request: pytest.FixtureRequest) -> None:
    prompt = torch.tensor([[0, 1]])
    generated = _canonical_generate(_CausalLM(), prompt)
    tokens = cast(list[list[int]], generated.tolist())
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
            ]
        ),
    )


def test_generate_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
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
    logits = torch.zeros(1, 4)  # softmax -> uniform 0.25 each.
    probs = _topp_probs(logits, top_p=0.5)
    kept = (probs > 0).sum(dim=-1).item()
    assert kept == 3


def test_sample_top_p_restores_vocab_order():
    """Filtered logits must map back to original vocab positions.

    Regression for GEN-TOPP (Issue#333): the surviving token's probability
    mass must land on its original vocab index, not a sorted position.
    """
    logits = torch.tensor([[1.0, 0.0, 9.0, 0.5, 0.2]])  # argmax at index 2.
    probs = _topp_probs(logits, top_p=0.5)
    assert probs.argmax(dim=-1).item() == 2


def test_sample_greedy_is_argmax():
    """Temperature 0 returns the argmax token id."""
    logits = torch.tensor([[1.0, 9.0, 0.5]])
    token = _sample(logits, 0.0, 0, 1.0)
    assert token.item() == 1


def test_sample_applies_temperature_top_k_and_top_p() -> None:
    logits = torch.tensor([[1.0, 2.0, 9.0]])
    token = _sample(logits, 2.0, 1, 0.5)
    assert token.item() == 2


def test_generate_rejects_prompt_longer_than_cache() -> None:
    model = _CausalLM()

    with pytest.raises(
        ValueError,
        match=r"prompt length 3 exceeds max_seq_len=2\.",
    ):
        generate(model, torch.tensor([[0, 1, 2]]), max_seq_len=2)

    assert model.block.attn.cache is None


def test_generate_returns_prompt_when_no_tokens_requested() -> None:
    model = _CausalLM()
    prompt = torch.tensor([[0, 1]])

    result = generate(model, prompt, max_new_tokens=0, max_seq_len=4)

    assert result is prompt
    assert model.project_calls == 1
    assert model.block.attn.max_seq == 4


def test_generate_forwards_cache_metadata_and_stops_at_eos() -> None:
    model = _CausalLM()
    prompt = torch.tensor([[0, 1]])

    result = generate(
        model,
        prompt,
        max_new_tokens=4,
        temperature=0.0,
        eos_token_id=3,
        max_seq_len=6,
    )

    assert torch.equal(result, torch.tensor([[0, 1, 2, 3]]))
    assert model.block.attn.batch == 1
    assert model.block.attn.max_seq == 6
    assert model.block.attn.device == prompt.device
    assert model.block.attn.dtype == model.embed.weight.dtype
    assert model.block.seen_caches == [model.block.attn.cache] * 2
    assert len(model.embed.inputs) == 2
    assert torch.equal(model.embed.inputs[0], prompt)
    assert torch.equal(model.embed.inputs[1], torch.tensor([[2]]))


def _topp_probs(logits: Tensor, *, top_p: float) -> Tensor:
    """Recover the kept-token distribution under top-p, deterministically.

    The nucleus filter sets out-of-nucleus logits to ``-1e10``, which softmaxes
    to exactly 0, so the filtered softmax IS the kept distribution -- the same
    distribution ``_sample`` draws from, but without the 20k-iteration
    Monte-Carlo loop (or its sampling flakiness).
    """
    return _topp_filter(logits, top_p).softmax(dim=-1)


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


class _Norm:
    def __call__(self, x: Tensor, /, **kwargs: object) -> Tensor:
        del kwargs
        return x

    def reset_parameters(self) -> None:
        pass


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


class _CausalLM:
    def __init__(self) -> None:
        self.embed = _Lookup()
        self.block = _Block()
        self.blocks: list[nn.Module] = [self.block]
        self.final_norm = _Norm()
        self.project_calls = 0

    def project_to_logits(self, hidden: Tensor, /) -> Tensor:
        next_token = (2, 3)[self.project_calls]
        self.project_calls += 1
        logits = torch.zeros(*hidden.shape[:-1], 4)
        logits[..., next_token] = 1
        return logits


class _GenerateHarness(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _CausalLM()

    @override
    def forward(self, prompt: Tensor) -> Tensor:
        return _canonical_generate(self.model, prompt)


def _canonical_generate(model: _CausalLM, prompt: Tensor) -> Tensor:
    return generate(
        model,
        prompt,
        max_new_tokens=4,
        temperature=0.0,
        eos_token_id=3,
        max_seq_len=6,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
