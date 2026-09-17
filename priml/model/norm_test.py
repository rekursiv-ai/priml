"""Tests for norm module."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final, cast

from configgle import Fig, Makeable
from configgle.testing import assert_pprint_golden
from torch import nn

import pytest
import torch

from priml.model.cost import Bytes, Compute, Cost, Flops, cost
from priml.model.custom_types import ChannelsInOut
from priml.model.norm import (
    BatchNorm,
    BatchNorm2d,
    BatchRenorm,
    CenteredRMSNorm,
    GroupNorm,
    GroupNorm2d,
    LayerNorm,
    RMSNorm,
)
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


def _config_id(value: object) -> str | None:
    return type(value).__qualname__ if isinstance(value, Fig) else None


_CWD: Final = Path(__file__).resolve().parent


def test_rmsnorm():
    cfg = RMSNorm.Config(64)
    m = cfg.make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)
    assert cfg.channels_in == 64


def test_layernorm():
    cfg = LayerNorm.Config(64)
    m = cfg.make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)
    assert cfg.channels_in == 64


def test_batchnorm():
    cfg = BatchNorm.Config(64)
    m = cfg.make()
    m.train()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)
    assert cfg.channels_in == 64


def test_groupnorm():
    cfg = GroupNorm.Config(64, num_groups=4)
    m = cfg.make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)
    assert cfg.channels_in == 64


def test_batchnorm2d():
    cfg = BatchNorm2d.Config(64)
    m = cfg.make()
    m.train()
    x = torch.randn(2, 64, 8, 8)
    assert m(x).shape == (2, 64, 8, 8)
    assert cfg.channels_in == 64


def test_groupnorm2d():
    cfg = GroupNorm2d.Config(64, num_groups=4)
    m = cfg.make()
    x = torch.randn(2, 64, 8, 8)
    assert m(x).shape == (2, 64, 8, 8)
    assert cfg.channels_in == 64


def test_groupnorm2d_eval_matches_train():
    """Batch-independent: no running stats, train/eval outputs identical."""
    m = GroupNorm2d.Config(8, num_groups=2).make()
    x = torch.randn(3, 8, 4, 4)
    m.train()
    out_train = m(x)
    m.eval()
    out_eval = m(x)
    assert torch.allclose(out_train, out_eval)


def test_norm_forward_accepts_messages_and_rejects_positional_extras():
    m = RMSNorm.Config(64).make()
    x = torch.randn(2, 8, 64)
    assert m(x, key="val").shape == (2, 8, 64)
    with pytest.raises(TypeError):
        cast(Callable[..., object], m)(x, "extra")


def test_centered_rmsnorm():
    m = CenteredRMSNorm.Config(64).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert out.shape == (2, 8, 64)
    # Weight starts at zeros → effective scale is (1 + 0) = 1.
    assert torch.allclose(m.weight, torch.zeros(64))


def test_centered_rmsnorm_identity_at_init():
    """At init (weight=0), CenteredRMSNorm ≈ plain RMSNorm."""
    m = CenteredRMSNorm.Config(32).make()
    x = torch.randn(4, 16, 32)
    out = m(x)
    x_f32 = x.float()
    expected = x_f32 * torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(out.float(), expected, atol=1e-5)


def test_centered_rmsnorm_accepts_messages_and_rejects_positional_extras():
    m = CenteredRMSNorm.Config(32).make()
    x = torch.randn(2, 8, 32)
    assert m(x, key="val").shape == (2, 8, 32)
    with pytest.raises(TypeError):
        cast(Callable[..., object], m)(x, "extra")


def test_groupnorm_arbitrary_leading_dims():
    m = GroupNorm.Config(64, num_groups=4).make()
    x = torch.randn(2, 3, 8, 64)
    out = m(x)
    ref = m(x.reshape(-1, 8, 64)).reshape_as(x)
    assert out.shape == x.shape
    assert torch.allclose(out, ref)


def _norm(**overrides: float) -> BatchRenorm:
    config = BatchRenorm.Config()
    config.channels_in = 4
    config.warmup_steps = 2
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_batch_renorm_reset_parameters_restores_the_identity_transform() -> None:
    """Reset restores the affine AND the running estimates it normalizes by.

    The buffers are part of the layer's learned state, so a reset that left them
    warm would reinitialize into another run's statistics.
    """
    layer = _norm()
    layer.train()
    layer(torch.randn(32, 4) * 3.0 + 5.0)
    nn.init.constant_(layer.weight, 7.0)
    nn.init.constant_(layer.bias, 7.0)

    layer.reset_parameters()

    assert torch.equal(layer.weight, torch.ones(4))
    assert torch.equal(layer.bias, torch.zeros(4))
    assert torch.equal(layer.running_mean, torch.zeros(4))
    assert torch.equal(layer.running_var, torch.ones(4))
    assert int(layer.steps) == 0


@torch.no_grad()
def test_it_standardizes_the_last_axis() -> None:
    layer = _norm(warmup_steps=1_000)
    layer.train()
    torch.manual_seed(0)
    output = layer(torch.randn(256, 4) * 3.0 + 5.0)
    assert float(output.mean().abs()) < 1e-4
    assert float(output.std()) == pytest.approx(1.0, abs=1e-2)


def test_before_warmup_it_is_plain_batch_normalization() -> None:
    """The correction is measured against the running statistics.

    Applying it before those mean anything would correct toward noise, so the
    warmup exists and this pins that it is honored.
    """
    layer = _norm(warmup_steps=1_000)
    layer.train()
    torch.manual_seed(1)
    batch = torch.randn(64, 4)
    output = layer(batch)
    expected = (batch - batch.mean(0)) / (batch.var(0, unbiased=False) + 1e-3).sqrt()
    assert torch.allclose(output, expected, atol=1e-5)


def test_after_warmup_the_correction_engages() -> None:
    warm = _norm(warmup_steps=0)
    cold = _norm(warmup_steps=1_000)
    warm.train()
    cold.train()
    torch.manual_seed(2)
    # A batch far from the running estimate, so the correction has work to do.
    batch = torch.randn(64, 4) * 4.0 + 2.0
    assert not torch.allclose(warm(batch), cold(batch), atol=1e-3)


def test_the_correction_is_bounded() -> None:
    # Unbounded, a single outlying batch would move the normalization
    # arbitrarily far, which is what the clipping exists to prevent.
    layer = _norm(warmup_steps=0, max_ratio=1.0, max_drift=0.0)
    layer.train()
    torch.manual_seed(3)
    batch = torch.randn(64, 4) * 50.0
    # With both bounds collapsed the correction is the identity, so this is
    # plain batch normalization again.
    expected = (batch - batch.mean(0)) / (batch.var(0, unbiased=False) + 1e-3).sqrt()
    assert torch.allclose(layer(batch), expected, atol=1e-5)


def test_evaluation_uses_the_running_statistics() -> None:
    # The mode difference is the whole problem batch renormalization solves:
    # eval must not depend on whatever else is in the batch.
    layer = _norm()
    layer.train()
    torch.manual_seed(4)
    for _ in range(5):
        layer(torch.randn(64, 4))

    layer.eval()
    probe = torch.randn(8, 4)
    alone = layer(probe)
    crowded = layer(torch.cat([probe, torch.randn(64, 4) * 10.0]))[:8]
    assert torch.allclose(alone, crowded)


def test_the_running_statistics_track_the_data() -> None:
    layer = _norm()
    layer.train()
    torch.manual_seed(5)
    for _ in range(200):
        layer(torch.randn(64, 4) + 3.0)
    assert float(layer.running_mean.mean()) > 0.1


def test_the_statistics_are_buffers_not_parameters() -> None:
    # An optimizer stepping a running estimate would be a silent disaster.
    layer = _norm()
    names = {name for name, _ in layer.named_parameters()}
    assert names == {"weight", "bias"}
    assert "running_mean" in dict(layer.named_buffers())


def test_the_correction_carries_no_gradient() -> None:
    """Detached by design: it is a correction, not a path to learn along.

    Left attached, a stale running estimate would inject gradients into every
    upstream layer.
    """
    layer = _norm(warmup_steps=0)
    layer.train()
    torch.manual_seed(6)
    for _ in range(3):
        layer(torch.randn(32, 4))

    batch = torch.randn(32, 4, requires_grad=True)
    layer(batch).sum().backward()
    assert batch.grad is not None
    assert bool(torch.isfinite(batch.grad).all())


def test_a_checkpoint_round_trips_the_statistics() -> None:
    layer = _norm()
    layer.train()
    torch.manual_seed(7)
    for _ in range(4):
        layer(torch.randn(32, 4))

    restored = _norm()
    restored.load_state_dict(layer.state_dict())
    assert torch.equal(restored.running_mean, layer.running_mean)
    assert int(restored.steps) == int(layer.steps)


def test_a_constant_feature_stays_finite() -> None:
    layer = _norm()
    layer.train()
    assert bool(torch.isfinite(layer(torch.ones(16, 4))).all())


@pytest.mark.parametrize(
    ("field", "value"),
    [("channels_in", 0), ("max_ratio", 0.5), ("max_drift", -1.0), ("momentum", 1.0)],
)
def test_an_invalid_setting_is_refused(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="must"):
        _norm(**{field: value})


@pytest.mark.parametrize(
    "config",
    [
        RMSNorm.Config(),
        CenteredRMSNorm.Config(),
        LayerNorm.Config(),
        BatchNorm.Config(),
        BatchRenorm.Config(),
        BatchNorm2d.Config(),
        GroupNorm2d.Config(),
        GroupNorm.Config(),
    ],
    ids=_config_id,
)
def test_norm_infers_the_missing_width_and_rejects_unequal_widths(
    config: Fig[nn.Module],
) -> None:
    """One finalize serves every norm: either width fills the other, both must agree."""
    assert isinstance(config, ChannelsInOut)
    config.channels_out = 8
    finalized = config.copy_tree().finalize()
    assert isinstance(finalized, ChannelsInOut)
    assert finalized.channels_in == 8
    config.channels_in = 4
    with pytest.raises(ValueError, match="channels_in=4 must equal channels_out=8"):
        config.make()


def test_rms_norm_config_pprint() -> None:
    config = RMSNorm.Config(4)
    assert_pprint_golden(
        test_file=__file__,
        name="rms_norm",
        config=config,
    )


def test_centered_rms_norm_config_pprint() -> None:
    config = CenteredRMSNorm.Config(4)
    assert_pprint_golden(
        test_file=__file__,
        name="centered_rms_norm",
        config=config,
    )


def test_layer_norm_config_pprint() -> None:
    config = LayerNorm.Config(4)
    assert_pprint_golden(
        test_file=__file__,
        name="layer_norm",
        config=config,
    )


def test_batch_norm_config_pprint() -> None:
    config = BatchNorm.Config(4)
    assert_pprint_golden(
        test_file=__file__,
        name="batch_norm",
        config=config,
    )


def test_batch_renorm_config_pprint() -> None:
    config = BatchRenorm.Config(channels_in=4)
    assert_pprint_golden(
        test_file=__file__,
        name="batch_renorm",
        config=config,
    )


def test_batch_norm2d_config_pprint() -> None:
    config = BatchNorm2d.Config(4)
    assert_pprint_golden(
        test_file=__file__,
        name="batch_norm2d",
        config=config,
    )


def test_group_norm2d_config_pprint() -> None:
    config = GroupNorm2d.Config(4, num_groups=2)
    assert_pprint_golden(
        test_file=__file__,
        name="group_norm2d",
        config=config,
    )


def test_group_norm_config_pprint() -> None:
    config = GroupNorm.Config(4, num_groups=2)
    assert_pprint_golden(
        test_file=__file__,
        name="group_norm",
        config=config,
    )


def test_rms_norm_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="rms_norm",
        build_module=lambda: RMSNorm.Config(4).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_centered_rms_norm_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="centered_rms_norm",
        build_module=lambda: CenteredRMSNorm.Config(4).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_layer_norm_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="layer_norm",
        build_module=lambda: LayerNorm.Config(4).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_batch_norm_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="batch_norm",
        build_module=lambda: BatchNorm.Config(4).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_batch_renorm_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="batch_renorm",
        build_module=lambda: BatchRenorm.Config(channels_in=4).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_batch_norm2d_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="batch_norm2d",
        build_module=lambda: BatchNorm2d.Config(4).make(),
        build_input=lambda: torch.randn(2, 4, 2, 2),
        seed=0,
    )


def test_group_norm2d_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="group_norm2d",
        build_module=lambda: GroupNorm2d.Config(4, num_groups=2).make(),
        build_input=lambda: torch.randn(2, 4, 2, 2),
        seed=0,
    )


def test_group_norm_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="group_norm",
        build_module=lambda: GroupNorm.Config(4, num_groups=2).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def _rows() -> torch.Tensor:
    return torch.randn(2, 3, 8, requires_grad=True)


def _image() -> torch.Tensor:
    return torch.randn(2, 8, 3, 1, requires_grad=True)


@pytest.mark.parametrize(
    ("config", "params", "primal", "adjoint"),
    [
        (RMSNorm.Config(8), 0, (2 * 8 + 3, 7), (4 * 8 + 4, 7)),
        (RMSNorm.Config(8, elementwise_affine=True), 8, (3 * 8 + 3, 7), (6 * 8 + 4, 7)),
        (CenteredRMSNorm.Config(8), 8, (4 * 8 + 3, 7), (6 * 8 + 4, 7)),
        (
            LayerNorm.Config(8, elementwise_affine=True),
            16,
            (3 * 8 + 4 + 16, 14),
            (5 * 8 + 2 + 16, 14),
        ),
        (
            BatchNorm.Config(8, elementwise_affine=True),
            16,
            (7 * 8 + 16 + 8 * 8, 0),
            (7 * 8 + 16, 0),
        ),
        (
            BatchRenorm.Config(channels_in=8),
            16,
            (7 * 8 + 16 + 18 * 8, 0),
            (7 * 8 + 16 + 5 * 8, 0),
        ),
        (
            GroupNorm.Config(8, elementwise_affine=True),
            16,
            (7 * 8 + 16, 0),
            (7 * 8 + 16, 0),
        ),
        (
            BatchNorm2d.Config(8, elementwise_affine=True),
            16,
            (7 * 8 + 16 + 8 * 8, 0),
            (7 * 8 + 16, 0),
        ),
        (
            GroupNorm2d.Config(8, elementwise_affine=True),
            16,
            (7 * 8 + 16, 0),
            (7 * 8 + 16, 0),
        ),
    ],
    ids=_config_id,
)
def test_norm_cost_splits_elementwise_from_row_sums(
    config: Makeable[nn.Module],
    params: int,
    primal: tuple[float, float],
    adjoint: tuple[float, float],
) -> None:
    """A norm is elementwise work plus sums over its group; never a matmul.

    Priced at one token: a group spanning a whole row is ``groups_per_token =
    1`` and holds two ``R - 1`` sums each way, while a group of one element
    (the batch norms at one token, eight groups of one) sums nothing. The
    expected pairs are ``(elementwise, reduction)``.
    """
    model_cost = cost(config.copy_tree().finalize())
    traffic: dict[type[Makeable[nn.Module]], tuple[float, float, float]] = {
        RMSNorm.Config: (4 * 8 + 7 + 3 * params, 10 * 8 + 12 + 6 * params, 9),
        CenteredRMSNorm.Config: (9 * 8 + 7, 16 * 8 + 12, 9),
        LayerNorm.Config: (6 * 8 + 10 + 3 * params, 12 * 8 + 7 + 3 * params, 18),
        BatchNorm.Config: (
            6 * 8 + 10 * 8 + 3 * params + 18 * 8,
            12 * 8 + 7 * 8 + 3 * params,
            32,
        ),
        BatchNorm2d.Config: (
            6 * 8 + 10 * 8 + 3 * params + 18 * 8,
            12 * 8 + 7 * 8 + 3 * params,
            32,
        ),
        BatchRenorm.Config: (
            6 * 8 + 10 * 8 + 3 * params + 46 * 8,
            12 * 8 + 7 * 8 + 3 * params + 13 * 8,
            32,
        ),
        GroupNorm.Config: (
            6 * 8 + 10 * 8 + 3 * params,
            12 * 8 + 7 * 8 + 3 * params,
            32,
        ),
        GroupNorm2d.Config: (
            6 * 8 + 10 * 8 + 3 * params,
            12 * 8 + 7 * 8 + 3 * params,
            32,
        ),
    }
    primal_io, adjoint_io, reduction_io = traffic[type(config)]
    assert model_cost == Cost(
        primal=Compute(
            flops=Flops(elementwise=primal[0], reduction=primal[1]),
            bytes=Bytes(elementwise=4 * primal_io, reduction=4 * reduction_io),
        ),
        adjoint=Compute(
            flops=Flops(elementwise=adjoint[0], reduction=adjoint[1]),
            bytes=Bytes(
                elementwise=4 * adjoint_io,
                reduction=4 * (reduction_io + 2 * params),
            ),
        ),
        params=params,
        params_active=params,
    )


@pytest.mark.parametrize(
    ("config", "build_input"),
    [
        (RMSNorm.Config(8, elementwise_affine=True), _rows),
        (CenteredRMSNorm.Config(8), _rows),
        (LayerNorm.Config(8, elementwise_affine=True), _rows),
        (BatchNorm.Config(8, elementwise_affine=True), _rows),
        (BatchRenorm.Config(channels_in=8), _rows),
        (GroupNorm.Config(8, elementwise_affine=True), _rows),
        (BatchNorm2d.Config(8, elementwise_affine=True), _image),
        (GroupNorm2d.Config(8, elementwise_affine=True), _image),
    ],
    ids=_config_id,
)
def test_norm_cost_is_matmul_free(
    config: Makeable[nn.Module],
    build_input: Callable[[], torch.Tensor],
) -> None:
    """Torch counts no matmul FLOPs in any norm, and the parameters agree."""
    model_cost = assert_cost_matches_torch(
        config,
        build_input=build_input,
        num_tokens=6,
    )
    assert model_cost.training.flops.matmul == 0
    assert model_cost.training.flops.elementwise > 0


@pytest.mark.parametrize(
    "config",
    [
        RMSNorm.Config(8, elementwise_affine=True),
        CenteredRMSNorm.Config(8),
        LayerNorm.Config(8, elementwise_affine=True),
        GroupNorm.Config(8, elementwise_affine=True),
    ],
    ids=_config_id,
)
def test_affine_gradients_reduce_over_the_rows(config: Makeable[nn.Module]) -> None:
    """Every owned parameter's gradient is summed over ``rows`` rows.

    Forward work per row is unchanged (the ``1 + weight`` fold aside); backward
    gains exactly ``(N - 1) / N`` additions per parameter, the primitive's rule.
    """
    finalized = config.copy_tree().finalize()
    one = cost(finalized, rows=1)
    four = cost(finalized, rows=4)
    assert four.params == one.params > 0
    fold = one.primal.flops.elementwise - four.primal.flops.elementwise
    assert fold in (0, 8 * (1 - 1 / 4))
    assert four.adjoint.flops.elementwise == one.adjoint.flops.elementwise
    assert four.adjoint.flops.reduction - one.adjoint.flops.reduction == (
        one.params * 3 / 4
    )


def test_affine_norm_pullback_reads_only_scale_not_shift() -> None:
    config = LayerNorm.Config(8)
    plain = config.cost(rows=4, itemsize=2)
    config.elementwise_affine = True
    affine = config.cost(rows=4, itemsize=2)
    assert affine.adjoint.bytes.elementwise - plain.adjoint.bytes.elementwise == 2 * (
        5 * 8 + 8 / 4
    )


def test_rms_norm_cost_counts_unfused_tensor_operands() -> None:
    result = RMSNorm.Config(8).cost(itemsize=2)
    # Square, three scalar transforms, then vector-by-scalar scaling.
    assert result.primal.bytes.elementwise == 2 * (2 * 8 + 6 + 2 * 8 + 1)
    assert result.primal.bytes.reduction == 2 * (8 + 1)
    assert result.adjoint.bytes.reduction == 2 * (8 + 1)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
