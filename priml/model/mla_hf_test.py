"""Numerical parity: priml.model.mla vs a direct HF-formula reference.

DeepSeek-V3 / Kimi-K2 aren't in stable transformers, so we build the
MLA reference forward from scratch here, matching the published math
(decoupled RoPE, per-head kv_b expansion, channels_v_head on the value
path). Our fused ``MultiHeadLatentAttention`` must produce the same
output on the same weights.

Covers:
  - Q-LoRA ON and OFF (DSV3 vs. Kimi-K2).
  - KV compression + decode via cache.
  - Asymmetric channels_qk_head vs. channels_v_head.

Integration-marked (heavier forward).
"""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional as f

import pytest
import torch

from priml.model.mla import MultiHeadLatentAttention
from priml.model.rope import RoPE


pytestmark = pytest.mark.network_huggingface


def _reference_mla_forward(
    m: MultiHeadLatentAttention,
    x: Tensor,
) -> Tensor:
    """Hand-written MLA forward matching the published formulas.

    Independent of our fused implementation so that any reordering,
    stacking, or batching trick in ``MultiHeadLatentAttention`` is
    checked against first-principles math.
    """
    S = x.shape[-2]
    heads = m.heads
    qk_nope = m.channels_qk_nope_head
    qk_rope = m.channels_qk_rope_head
    v_dim = m.channels_v_head
    qk_head = m.channels_qk_head

    # Q path.
    if m.q_proj is not None:
        q = m.q_proj(x)
    else:
        assert m.q_a_proj is not None
        assert m.q_a_layernorm is not None
        assert m.q_b_proj is not None
        q = m.q_b_proj(m.q_a_layernorm(m.q_a_proj(x)))
    q = q.view(*q.shape[:-1], heads, qk_head)
    q_nope, q_pe = q[..., :qk_nope], q[..., qk_nope:]

    # KV compressed latent.
    compressed = m.kv_a_proj(x)
    c_kv_raw = compressed[..., : m.kv_lora_rank]
    k_pe_raw = compressed[..., m.kv_lora_rank :]
    c_kv = m.kv_a_layernorm(c_kv_raw)
    kv = m.kv_b_proj(c_kv).view(*x.shape[:-1], heads, qk_nope + v_dim)
    k_nope, v = kv[..., :qk_nope], kv[..., qk_nope:]
    k_pe = k_pe_raw.unsqueeze(-2)  # [B, S, 1, R]

    # RoPE on q_pe and k_pe only.
    assert m.rope is not None
    positions = torch.arange(S, device=x.device)
    cos, sin = m.rope(positions)
    # DSV3/Kimi-K2 apply RoPE on an interleaved pairing (see HF's
    # ``apply_rotary_pos_emb`` pre-shuffle in modeling_deepseek.py).
    q_pe, k_pe = RoPE.rotate(q_pe, k_pe, cos, sin, interleave=True)
    k_pe = k_pe.expand(*k_nope.shape[:-1], qk_rope)

    q_full = torch.cat([q_nope, q_pe], dim=-1).movedim(-3, -2)
    k_full = torch.cat([k_nope, k_pe], dim=-1).movedim(-3, -2)
    v = v.movedim(-3, -2)
    out = f.scaled_dot_product_attention(
        q_full,
        k_full,
        v,
        is_causal=True,
        scale=m.softmax_scale,
    )
    return m.o_proj(out.movedim(-3, -2).flatten(-2))


@pytest.mark.parametrize("q_lora_rank", [None, 32])
def test_mla_matches_reference(q_lora_rank: int | None):
    torch.manual_seed(0)
    m = MultiHeadLatentAttention.Config(
        channels_in=128,
        heads=4,
        channels_qk_nope_head=16,
        channels_qk_rope_head=8,
        channels_v_head=16,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=24,
        rope=RoPE.Config(channels_head=8, base=50_000),
    ).make()
    m.eval()
    x = torch.randn(2, 6, 128)
    with torch.no_grad():
        fused, _ = m(x)
        ref = _reference_mla_forward(m, x)
    diff = (fused - ref).abs().max().item()
    assert torch.allclose(fused, ref, atol=5e-5, rtol=1e-4), f"max abs diff: {diff:.3e}"


def test_mla_decode_matches_reference():
    """Cached incremental decode matches the from-scratch reference
    computed on the full concatenated sequence.
    """
    torch.manual_seed(0)
    m = MultiHeadLatentAttention.Config(
        channels_in=64,
        heads=4,
        channels_qk_nope_head=8,
        channels_qk_rope_head=8,
        channels_v_head=8,
        kv_lora_rank=16,
        rope=RoPE.Config(channels_head=8, base=50_000),
    ).make()
    m.eval()
    prompt = torch.randn(1, 4, 64)
    steps = [torch.randn(1, 1, 64) for _ in range(3)]
    full_input = torch.cat([prompt, *steps], dim=1)

    with torch.no_grad():
        ref_full = _reference_mla_forward(m, full_input)

        cache = m.alloc_kv_cache(batch=1, max_seq=16)
        _, cache = m(prompt, cache=cache)
        decoded: list[Tensor] = []
        for step in steps:
            out, cache = m(step, cache=cache)
            decoded.append(out)
    cached_tail = torch.cat(decoded, dim=1)
    diff = (ref_full[:, 4:] - cached_tail).abs().max().item()
    assert torch.allclose(ref_full[:, 4:], cached_tail, atol=5e-5, rtol=1e-4), (
        f"max abs diff: {diff:.3e}"
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
