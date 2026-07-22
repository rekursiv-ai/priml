"""Single-source-of-truth contract: ``reset_parameters`` defines every init.

A param-bearing module must allocate storage in ``__init__`` and delegate the
actual initial values to ``reset_parameters`` (which ``__init__`` calls once).
Then ``reset_parameters`` is the *sole* source of init values, with no second
place where an init value can silently drift.

The check: after wiping every parameter to NaN, a single ``reset_parameters``
call must restore every one to a finite value. A parameter left NaN means some
module owns a parameter that ``reset_parameters`` does not initialize -- so
``reset_parameters`` is not the complete single source, and that parameter
would ship ``to_empty`` garbage on the meta-device materialization path.

Bit-equality between eager construction and a *second* ``reset_parameters`` is
deliberately NOT asserted: children that initialize at construction time
consume RNG before the first reset, so the two RNG sequences differ by design
(this is inherent to the construct-then-reset idiom, not a defect). The
total-finite re-init invariant is the achievable, meaningful contract.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

import pytest
import torch

from priml.model.attention import SelfAttention
from priml.model.gated_delta_net import GatedDeltaNet
from priml.model.linear import EnsembleLinear, Linear
from priml.model.mla import MultiHeadLatentAttention
from priml.model.moe import MoE, Router
from priml.model.norm import CenteredRMSNorm
from priml.model.rope import RoPE, RoPEMixed
from priml.model.transformer import TransformerBlock


_BUILDERS: dict[str, Callable[[], nn.Module]] = {
    "linear": lambda: Linear.Config(channels_in=16, channels_out=8, bias=True).make(),
    "ensemble_linear": lambda: EnsembleLinear.Config(
        channels_in=8,
        channels_out=8,
        num_ensemble=2,
        bias=True,
    ).make(),
    "centered_rmsnorm": lambda: CenteredRMSNorm(CenteredRMSNorm.Config(channels_in=8)),
    "self_attention": lambda: SelfAttention.Config(
        channels_in=16,
        heads=2,
        channels_head=8,
    ).make(),
    "transformer_block": lambda: TransformerBlock.Config(
        channels_in=16,
        attn=SelfAttention.Config(heads=2, channels_head=8),
    ).make(),
    "gated_delta_net": lambda: GatedDeltaNet.Config(
        channels_in=16,
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=8,
        head_v_dim=8,
    ).make(),
    "moe": lambda: MoE.Config(
        channels_in=16,
        router=Router.Config(num_experts=4, top_k=2),
    ).make(),
    "mla": lambda: MultiHeadLatentAttention.Config(
        channels_in=16,
        heads=2,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        kv_lora_rank=8,
        rope=RoPE.Config(channels_head=4, base=10_000),
    ).make(),
    "rope_mixed_learnable": lambda: RoPEMixed(
        RoPEMixed.Config(channels_head=8, base=10_000, heads=2, learnable=True),
    ),
    # heads=1 takes the directions=None branch (no per-head scaling);
    # reduction_mode="sum" shares channels across axes. Both are distinct
    # shape-allocation paths in reset_parameters that must round-trip.
    "rope_mixed_heads1": lambda: RoPEMixed(
        RoPEMixed.Config(channels_head=8, base=10_000, heads=1, learnable=True),
    ),
    "rope_mixed_sum": lambda: RoPEMixed(
        RoPEMixed.Config(
            channels_head=8,
            base=10_000,
            heads=2,
            reduction_mode="sum",
            learnable=True,
        ),
    ),
}


@pytest.mark.parametrize("name", list(_BUILDERS))
def test_reset_parameters_reinitializes_every_param(name: str) -> None:
    """After wiping every param and float buffer to NaN, one ``reset_parameters``
    restores all of them to finite values.

    A param or float buffer left NaN means a module owns state that
    ``reset_parameters`` does not initialize -- so it is not the complete single
    source of truth and would ship ``to_empty`` garbage on the meta path. This
    mirrors the production ``_materialize_meta`` audit, which poisons and checks
    params AND float buffers (integer buffers cannot hold NaN and are skipped).
    """
    model = _BUILDERS[name]()
    # name -> tensor over params + buffers, matching the materialize audit.
    state = [*model.named_parameters(), *model.named_buffers()]
    with torch.no_grad():
        for _, tensor in state:
            if tensor.is_floating_point():
                tensor.fill_(float("nan"))
    model.reset_parameters()
    for key, tensor in state:
        if not tensor.is_floating_point():
            continue
        assert torch.isfinite(tensor).all(), (
            f"{name}.{key}: reset_parameters left it NaN (incomplete init source)"
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
