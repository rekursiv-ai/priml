"""Require an attention specialization to own rotary arithmetic at both seams."""

from typing import override

from torch import Tensor

import pytest
import torch

from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.rope import RoPE


class UnrotatedAttention(Attention):
    """Leave queries and keys unchanged through the inherited forward path."""

    @classmethod
    @override
    def rotate(
        cls,
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> tuple[Tensor, Tensor]:
        del cos, sin
        return q, k


@pytest.mark.parametrize("internal_factors", [False, True])
def test_specialized_rotation_runs_for_internal_and_external_factors(
    internal_factors: bool,
) -> None:
    config = Attention.Config()
    config.channels_in = 7
    config.num_heads = 4
    config.num_heads_kv = 2
    config.channels_head = 6
    config.attn_kernel = SdpaNaive.Config()
    ordinary = config.make().eval()
    specialized = config.copy_tree()
    rope = RoPE.Config()
    rope.channels_head = 6
    if internal_factors:
        specialized.rope = rope
    model = UnrotatedAttention(specialized.finalized()).eval()
    model.load_state_dict(ordinary.state_dict())
    media = torch.arange(105).reshape(3, 5, 7).float() / 11
    factors = None if internal_factors else rope.make()(torch.arange(5))
    assert torch.equal(model(media, cos_sin=factors), ordinary(media))


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
