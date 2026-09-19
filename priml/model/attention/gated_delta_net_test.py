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

from priml.cost import cost
from priml.model.attention.gated_delta_net import GatedDeltaNet
from priml.model.special import Identity
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


def test_gated_delta_net_mismatched_widths_reject_at_make() -> None:
    config = GatedDeltaNet.Config()
    config.channels_in = 8
    config.channels_out = 16

    with pytest.raises(ValueError, match="channels_in=8 must equal channels_out=16"):
        config.make()


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
    """The padded chunk executes its full products even for a one-token input."""
    config = GatedDeltaNet.Config(
        channels_in=16,
        num_heads_k=2,
        num_heads_v=4,
        channels_k_head=8,
        channels_v_head=4,
        conv_kernel_size=3,
    )
    finalized = config.copy_tree().finalize()
    model_cost = finalized.cost(seq_len=1, batch_size=1, dtype=None)
    k_dim, v_dim = 2 * 8, 4 * 4
    conv_dim = 2 * k_dim + v_dim
    projections = 16 * conv_dim + 16 * v_dim + 2 * 16 * 4 + v_dim * 16
    conv = conv_dim * 3
    gates = 2 * 4  # dt_bias and A_log, one per value head.
    norm = cost(finalized.norm, seq_len=1, batch_size=1, dtype=None).params
    assert norm == 4
    assert model_cost.params == projections + conv + gates + norm
    assert model_cost.params == sum(p.numel() for p in config.make().parameters())
    chunk_products = 64 * 64 * (3 * 8 + 2 * 4)
    chunk_adjoint_products = 64 * 64 * (2 * 8 + 4)
    state_products = 64 * 8 * 4
    assert model_cost["flops", "primal", "matmul"].sum() == 2 * (
        projections + 3 * conv
    ) + 2 * 4 * (chunk_products + 3 * state_products)
    assert model_cost["flops", "adjoint", "matmul"].sum() == 4 * (
        projections + 3 * conv
    ) + 2 * 4 * (2 * chunk_adjoint_products + 2 * state_products)
    assert model_cost.bytes_state == 0
    # The q/k L2 norms sum each key row once each way, per value head.
    assert model_cost["flops", "primal", "reduction"].sum() == 2 * 4 * (
        8 - 1
    ) + norm_reduction(
        finalized,
    )
    norm_cost = cost(
        finalized.norm,
        seq_len=1,
        batch_size=finalized.num_heads_v,
        dtype=None,
    )
    assert (
        model_cost["flops", "adjoint", "reduction"].sum()
        == 2 * 4 * (8 - 1) + norm_cost["flops", "adjoint", "reduction"].sum()
    )
    assert (
        model_cost["bytes", "primal", "reduction"].sum()
        == 2 * 4 * 4 * (8 + 1) + norm_cost["bytes", "primal", "reduction"].sum()
    )


def norm_reduction(finalized: GatedDeltaNet.Config) -> int:
    """Reduction FLOPs for one norm invocation over all value-head rows."""
    return cost(
        finalized.norm,
        seq_len=1,
        batch_size=finalized.num_heads_v,
        dtype=None,
    )["flops", "primal", "reduction"].sum()


@pytest.mark.parametrize("seq_len", [1, 4, 64, 65, 129])
@pytest.mark.parametrize("geometry", [(1, 2, 8, 8), (2, 4, 4, 3)])
def test_gated_delta_net_chunk_cost_matches_torch(
    seq_len: int,
    geometry: tuple[int, int, int, int],
) -> None:
    """Count padded chunks, initial constant state, and the unused final update."""
    batch_size, value_heads, key_width, value_width = geometry
    assert_cost_matches_torch(
        GatedDeltaNet.Config(
            channels_in=16,
            num_heads_k=2,
            num_heads_v=value_heads,
            channels_k_head=key_width,
            channels_v_head=value_width,
            conv_kernel_size=3,
        ),
        build_input=lambda: torch.randn(batch_size, seq_len, 16, requires_grad=True),
        seq_len=seq_len,
        batch_size=batch_size,
        dtype=None,
    )


def test_gated_delta_net_flops_scale_with_batch_size() -> None:
    config = GatedDeltaNet.Config(channels_in=8, num_heads_k=1, num_heads_v=1)
    finalized = config.copy_tree().finalize()
    one = finalized.cost(seq_len=8, batch_size=1, dtype=None)
    batch = finalized.cost(seq_len=8, batch_size=4, dtype=None)
    assert batch["flops", "matmul"].sum() == 4 * one["flops", "matmul"].sum()
    assert batch["bytes", "matmul"].sum() > one["bytes", "matmul"].sum()
    assert one.params == batch.params


def test_delta_traffic_counts_shared_weights_once_per_invocation() -> None:
    config = GatedDeltaNet.Config()
    config.channels_in = 8
    config.num_heads_k = 1
    config.num_heads_v = 2
    config.channels_k_head = 4
    config.channels_v_head = 3
    config = config.copy_tree().finalize()
    one = config.cost(seq_len=4, batch_size=1, dtype=torch.bfloat16)
    batch = config.cost(seq_len=4, batch_size=4, dtype=torch.bfloat16)
    weights = 8 * 14 + 8 * 6 + 2 * 8 * 2 + 6 * 8 + 14 * 4
    assert (
        4 * one["bytes", "primal", "matmul"].sum()
        - batch[
            "bytes",
            "primal",
            "matmul",
        ].sum()
        == 3 * torch.bfloat16.itemsize * weights
    )
    assert batch["matmul", torch.float32] == one["matmul", torch.float32].tile(4)
    wide = config.cost(seq_len=4, batch_size=4, dtype=torch.float32)
    assert (
        wide["bytes", "matmul", torch.float32].sum()
        == batch["bytes", "matmul", torch.bfloat16].sum() * 2
        + batch["bytes", "matmul", torch.float32].sum()
    )
    assert batch["bytes", "primal", "reduction"].sum() > 0
    full = config.cost(seq_len=64, batch_size=1, dtype=torch.bfloat16)
    padded = config.cost(seq_len=65, batch_size=1, dtype=torch.bfloat16)
    assert full["flops", "matmul"].sum() > one["flops", "matmul"].sum()
    assert padded["flops", "matmul"].sum() > full["flops", "matmul"].sum()
    assert (
        padded["bytes", "primal", "matmul"].sum()
        > full[
            "bytes",
            "primal",
            "matmul",
        ].sum()
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_delta_normalizes_query_and_key_with_separate_reductions(
    dtype: torch.dtype,
) -> None:
    config = GatedDeltaNet.Config()
    config.channels_in = 8
    config.num_heads_k = 1
    config.num_heads_v = 2
    config.channels_k_head = 4
    config.channels_v_head = 3
    config.norm = Identity.Config()
    actual = config.copy_tree().finalize().cost(seq_len=1, batch_size=1, dtype=dtype)
    itemsize = dtype.itemsize
    assert actual["flops", "primal", "reduction"].sum() == 2 * 2 * (4 - 1)
    assert actual["flops", "adjoint", "reduction"].sum() == 2 * 2 * (4 - 1)
    assert actual["bytes", "primal", "reduction"].sum() == itemsize * 2 * 2 * (4 + 1)
    # The two learned decay vectors additionally reduce their row gradients.
    assert actual["bytes", "adjoint", "reduction"].sum() == itemsize * (
        2 * 2 * (4 + 1) + 2 * 2 * 2
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
