"""Check the world model's varlen self-attention and its max logit."""

import torch

from priml.baselines.craftax.world_model.attention import (
    VarlenAttention,
    row_cu_seqlens,
)
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE
from priml.model.norm import RMSNorm


def test_varlen_attention_keeps_its_max_logit_detached() -> None:
    config = VarlenAttention.Config(
        channels_in=24,
        num_heads=6,
        num_heads_kv=3,
        share_qk_norm=False,
    )
    config.norm_qk = RMSNorm.Config(elementwise_affine=True)
    attention = config.make()
    x = torch.randn(2, 7, 24)
    cu_seqlens = torch.tensor([0, 3, 7, 14], dtype=torch.int32)
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6]])
    attention(x, positions=positions, cu_seqlens=cu_seqlens)
    base = attention.max_logit
    assert base is not None
    assert base.shape == ()
    assert not base.requires_grad
    norm_q = attention.norm_q
    assert isinstance(norm_q, RMSNorm)
    assert norm_q.weight is not None
    with torch.no_grad():
        norm_q.weight.mul_(2)
    attention(x, positions=positions, cu_seqlens=cu_seqlens)
    assert attention.max_logit is not None
    torch.testing.assert_close(attention.max_logit, 2 * base)


def test_row_cu_seqlens_gives_one_segment_per_row() -> None:
    assert row_cu_seqlens(3, 5, device=torch.device("cpu")).tolist() == [0, 5, 10, 15]


def test_varlen_attention_isolates_segments_and_resets_positions() -> None:
    config = VarlenAttention.Config(channels_in=24, num_heads=6, num_heads_kv=3)
    config.norm_qk = RMSNorm.Config(elementwise_affine=True)
    config.rope = RoPE.Config(channels_head=4)
    attention = config.make()
    x = torch.randn(2, 7, 24)
    cu_seqlens = torch.tensor([0, 3, 7, 10, 14], dtype=torch.int32)
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3]]).expand(2, -1)
    packed = attention(x, positions=positions, cu_seqlens=cu_seqlens)
    alone = attention(
        x[:, 3:],
        positions=torch.arange(4).expand(2, -1),
        cu_seqlens=row_cu_seqlens(2, 4, device=x.device),
    )
    torch.testing.assert_close(packed[:, 3:], alone)


def test_varlen_attention_costs_what_causal_attention_costs() -> None:
    # One segment per row is priml's causal Attention: the same projections,
    # kernel products, and per-token cache.
    varlen = VarlenAttention.Config(channels_in=24, num_heads=6, num_heads_kv=3)
    reference = Attention.Config(channels_in=24, num_heads=6, num_heads_kv=3)
    reference.causal = True
    costs = [
        config.finalize().cost(seq_len=7, batch_size=2, dtype=torch.float32)
        for config in (varlen, reference)
    ]
    assert costs[0] == costs[1]


def test_varlen_attention_shares_attention_parameter_names() -> None:
    varlen = VarlenAttention.Config(channels_in=8, num_heads=2).make()
    reference = Attention.Config(channels_in=8, num_heads=2).make()
    assert varlen.state_dict().keys() == reference.state_dict().keys()


def test_varlen_attention_is_attention_with_its_options() -> None:
    # One segment per row, causal: priml's Attention computes the same thing.
    varlen_config = VarlenAttention.Config(channels_in=16, num_heads=4)
    reference_config = Attention.Config(channels_in=16, num_heads=4, causal=True)
    for config in (varlen_config, reference_config):
        config.num_heads_kv = 2
        config.norm_out = RMSNorm.Config(elementwise_affine=True)
        config.split_qkv_projection = True
    varlen = varlen_config.finalize().make()
    reference = reference_config.finalize().make()
    norm_out = varlen.norm_out
    assert isinstance(norm_out, RMSNorm)
    assert norm_out.weight is not None
    with torch.no_grad():
        norm_out.weight.uniform_(0.5, 1.5)
    reference.load_state_dict(varlen.state_dict())
    x = torch.randn(2, 5, 16)
    ours = varlen(
        x,
        positions=torch.arange(5)[None],
        cu_seqlens=row_cu_seqlens(2, 5, device=x.device),
    )
    torch.testing.assert_close(ours, reference(x))
    ours.sum().backward()
    assert norm_out.weight.grad is not None


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
