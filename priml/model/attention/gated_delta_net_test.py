"""Tests for gated_delta_net module.

Regenerate the numerical golden after an intentional change with::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest \
        priml/model/attention/gated_delta_net_test.py

Run regeneration through pytest so Priml's deterministic math setup applies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from configgle.testing import assert_pprint_golden

import pytest
import torch

from priml.model.attention.gated_delta_net import (
    GatedDeltaNet,
    _torch_chunk_gated_delta_rule,
)
from priml.model.cost import cost
from priml.testing.bfb import (
    assert_bfb_against_golden,
    bfb_devices,
    first_tensor,
    move_to_device,
)
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def test_gated_delta_net_rejects_non_multiple_v_heads():
    """``num_heads_v`` must be an integer multiple of ``num_heads_k``.

    Regression for MODEL-004: the GQA replication
    ``num_heads_v // num_heads_k`` silently truncated for non-multiples.
    """
    with pytest.raises(ValueError, match="multiple"):
        GatedDeltaNet.Config(
            channels_in=64,
            num_heads_k=4,
            num_heads_v=6,
            channels_k_head=16,
            channels_v_head=16,
        ).make()


def test_gated_delta_net_config_pprint() -> None:
    config = GatedDeltaNet.Config(
        channels_in=16,
        num_heads_k=2,
        num_heads_v=2,
        channels_k_head=8,
        channels_v_head=8,
    )
    assert_pprint_golden(
        test_file=__file__,
        name="gated_delta_net",
        config=config,
    )


def test_gated_delta_net_mismatched_widths_validate_on_construction() -> None:
    config = GatedDeltaNet.Config()
    config.channels_in = 8
    config.channels_out = 16

    finalized = config.copy_tree().finalize()
    assert (finalized.channels_in, finalized.channels_out) == (8, 16)
    with pytest.raises(ValueError, match="for GatedDeltaNet"):
        config.make()
    with pytest.raises(ValueError, match="for GatedDeltaNet"):
        GatedDeltaNet(config)

    class DerivedGatedDeltaNet(GatedDeltaNet):
        pass

    with pytest.raises(ValueError, match="for DerivedGatedDeltaNet"):
        DerivedGatedDeltaNet(config)


def test_gated_delta_net_forward():
    m = GatedDeltaNet.Config(
        channels_in=64,
        num_heads_k=2,
        num_heads_v=4,
        channels_k_head=16,
        channels_v_head=16,
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)


def test_gated_delta_net_single_token():
    m = GatedDeltaNet.Config(
        channels_in=32,
        num_heads_k=2,
        num_heads_v=2,
        channels_k_head=8,
        channels_v_head=8,
    ).make()
    x = torch.randn(1, 1, 32)
    out = m(x)
    assert out.shape == (1, 1, 32)


def test_torch_chunk_fallback_shapes():
    B, S, H, dk, dv = 1, 16, 2, 8, 8
    q = torch.randn(B, S, H, dk)
    k = torch.randn(B, S, H, dk)
    v = torch.randn(B, S, H, dv)
    g = torch.randn(B, S, H)
    beta = torch.rand(B, S, H)
    out, state = _torch_chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        chunk_size=8,
    )
    assert out.shape == (B, S, H, dv)
    assert state is None


def test_torch_chunk_fallback_with_final_state():
    B, S, H, dk, dv = 1, 8, 2, 8, 8
    q = torch.randn(B, S, H, dk)
    k = torch.randn(B, S, H, dk)
    v = torch.randn(B, S, H, dv)
    g = torch.randn(B, S, H)
    beta = torch.rand(B, S, H)
    out, state = _torch_chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        output_final_state=True,
    )
    assert out.shape == (B, S, H, dv)
    assert state is not None
    assert state.shape == (B, H, dk, dv)


def test_gated_delta_net_arbitrary_leading_dims():
    m = GatedDeltaNet.Config(
        channels_in=64,
        num_heads_k=2,
        num_heads_v=4,
        channels_k_head=16,
        channels_v_head=16,
    ).make()
    x = torch.randn(2, 3, 8, 64)
    out = m(x)
    assert out.shape == (2, 3, 8, 64)


def test_gated_delta_net_rejects_every_degenerate_dimension():
    """Each count is checked, not just ``num_heads_k``.

    A zero elsewhere reached torch and failed as a tensor-shape error naming
    no config field: ``num_heads_v=0`` built a zero-width projection,
    ``conv_kernel_size=0`` produced negative padding.
    """
    with pytest.raises(ValueError, match="num_heads_k"):
        _ = GatedDeltaNet.Config(channels_in=64, num_heads_k=0, num_heads_v=2).make()
    with pytest.raises(ValueError, match="num_heads_v"):
        _ = GatedDeltaNet.Config(channels_in=64, num_heads_k=2, num_heads_v=0).make()
    with pytest.raises(ValueError, match="channels_in"):
        _ = GatedDeltaNet.Config(channels_in=0, num_heads_k=2, num_heads_v=2).make()
    with pytest.raises(ValueError, match="channels_k_head"):
        _ = GatedDeltaNet.Config(
            channels_in=64,
            num_heads_k=2,
            num_heads_v=2,
            channels_k_head=0,
        ).make()
    with pytest.raises(ValueError, match="channels_v_head"):
        _ = GatedDeltaNet.Config(
            channels_in=64,
            num_heads_k=2,
            num_heads_v=2,
            channels_v_head=0,
        ).make()
    with pytest.raises(ValueError, match="conv_kernel_size"):
        _ = GatedDeltaNet.Config(
            channels_in=64,
            num_heads_k=2,
            num_heads_v=2,
            conv_kernel_size=0,
        ).make()


def test_gated_delta_net_never_initializes_a_closed_gate():
    """``A_log`` must stay finite: ``log(0)`` would close a head forever.

    ``uniform_(0, 16)`` is half-open and really does return exactly 0.0
    (measured once in 10M draws); the forward exponentiates ``A_log``, so a
    single ``-inf`` is a head that can never open again.
    """
    model = GatedDeltaNet.Config(channels_in=64, num_heads_k=2, num_heads_v=2).make()
    for _ in range(20):
        model.reset_parameters()
        assert bool(torch.isfinite(model.A_log).all())


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_gated_delta_net_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="gated_delta_net",
        build_module=lambda: (
            GatedDeltaNet.Config(
                channels_in=16,
                num_heads_k=2,
                num_heads_v=2,
                channels_k_head=8,
                channels_v_head=8,
            )
            .make()
            .to(device)
        ),
        build_input=lambda: move_to_device(torch.randn(2, 4, 16), device),
        seed=0,
        run=lambda m, x: first_tensor(m(x)),  # pyright: ignore[reportAny] -- the test helper accepts the model's untyped tuple output.
    )


def test_gated_delta_net_cost_is_projections_conv_and_state_update() -> None:
    """The scan prices as a per-token state read and write, not a growing cache."""
    config = GatedDeltaNet.Config(
        channels_in=16,
        num_heads_k=2,
        num_heads_v=4,
        channels_k_head=8,
        channels_v_head=4,
        conv_kernel_size=3,
    )
    finalized = config.copy_tree().finalize()
    model_cost = finalized.cost()
    k_dim, v_dim = 2 * 8, 4 * 4
    conv_dim = 2 * k_dim + v_dim
    projections = 16 * conv_dim + 16 * v_dim + 2 * 16 * 4 + v_dim * 16
    conv = conv_dim * 3
    gates = 2 * 4  # dt_bias and A_log, one per value head.
    norm = cost(finalized.norm).params
    assert norm == 4
    assert model_cost.params == projections + conv + gates + norm
    assert model_cost.params == sum(p.numel() for p in config.make().parameters())
    state = 4 * 8 * 4  # num_heads_v x channels_k_head x channels_v_head.
    assert model_cost.primal.flops.matmul == 2 * (projections + conv) + 4 * state
    assert model_cost.adjoint.flops.matmul == 4 * (projections + conv) + 8 * state
    assert model_cost.bytes_state == 0
    # The q/k L2 norms sum each key row once each way, per value head.
    assert model_cost.primal.flops.reduction == 4 * (8 - 1) + norm_reduction(finalized)


def norm_reduction(finalized: GatedDeltaNet.Config) -> float:
    """Reduction FLOPs the injected norm contributes, tiled over the value heads."""
    return cost(finalized.norm).tile(finalized.num_heads_v).primal.flops.reduction


def test_gated_delta_net_projections_match_torch() -> None:
    """The projections and the depthwise conv are torch's whole matmul count.

    The CPU scan is chunked: it pads the sequence to 64-token chunks and runs
    in-chunk triangular solves, so at four tokens it executes ~50x the products
    the recurrent-model proxy prices. The proxy is the analytical policy; the
    ratio pins torch's measured count at this geometry (541,664 FLOPs/token to
    the analytical 10,464) so a change to either side is visible.
    """
    assert_cost_matches_torch(
        GatedDeltaNet.Config(
            channels_in=16,
            num_heads_k=2,
            num_heads_v=2,
            channels_k_head=8,
            channels_v_head=8,
            conv_kernel_size=3,
        ),
        build_input=lambda: torch.randn(1, 4, 16, requires_grad=True),
        num_tokens=4,
        expected_ratio=10_464 / 541_664,
    )


def test_gated_delta_net_cost_ignores_seq_len() -> None:
    config = GatedDeltaNet.Config(channels_in=8, num_heads_k=1, num_heads_v=1)
    finalized = config.copy_tree().finalize()
    assert finalized.cost(seq_len=8) == finalized.cost(seq_len=1024)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
