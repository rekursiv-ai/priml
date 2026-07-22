"""Tests for priml.model.causal_lm."""

from __future__ import annotations

import pytest
import torch

from priml.model.attention import SelfAttention
from priml.model.causal_lm import CausalLM
from priml.model.generate import generate
from priml.model.norm import RMSNorm
from priml.model.rope import RoPE
from priml.model.transformer import TransformerBlock


def _tiny_config(tie: bool = False) -> CausalLM.Config:
    return CausalLM.Config(
        vocab_size=128,
        channels=32,
        num_layers=2,
        block=TransformerBlock.Config(
            attn=SelfAttention.Config(
                heads=4,
                channels_head=8,
                causal=True,
                rope=RoPE.Config(channels_head=8),
            ),
        ),
        final_norm=RMSNorm.Config(),
        tie_embeddings=tie,
    )


def test_forward_shape():
    m = _tiny_config().make()
    toks = torch.randint(0, 128, (2, 6))
    out = m(toks)
    assert out.shape == (2, 6, 128)


def test_tied_embeddings():
    m = _tiny_config(tie=True).make()
    assert m.lm_head is None
    toks = torch.randint(0, 128, (1, 4))
    out = m(toks)
    assert out.shape == (1, 4, 128)


def test_separate_lm_head():
    m = _tiny_config(tie=False).make()
    assert m.lm_head is not None
    # Distinct parameter, not the embed matrix.
    assert m.lm_head.weight.data_ptr() != m.embed.weight.data_ptr()


def test_num_layers_materialized():
    m = _tiny_config().make()
    assert len(m.blocks) == 2


def test_generate_interop():
    m = _tiny_config().make()
    m.eval()
    prompt = torch.randint(0, 128, (1, 4))
    gen = generate(m, prompt, max_new_tokens=3, temperature=0.0, max_seq_len=16)
    # Prompt + 3 generated.
    assert gen.shape == (1, 7)


def test_generate_rejects_prompt_longer_than_cache():
    """A prompt longer than ``max_seq_len`` is rejected up front.

    Regression for MODEL-003: an over-long prompt previously hit the
    KVCache overflow path with corrupt slices instead of a clear error.
    """
    m = _tiny_config().make()
    m.eval()
    prompt = torch.randint(0, 128, (1, 8))
    with pytest.raises(ValueError, match="prompt length"):
        generate(m, prompt, max_new_tokens=1, max_seq_len=4)


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="vocab_size"):
        CausalLM.Config(vocab_size=0, channels=8, num_layers=1).make()
    with pytest.raises(ValueError, match="num_layers"):
        CausalLM.Config(vocab_size=8, channels=8, num_layers=0).make()
    with pytest.raises(ValueError, match="channels"):
        CausalLM.Config(vocab_size=8, channels=0, num_layers=1).make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
