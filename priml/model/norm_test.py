"""Tests for norm module."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

from configgle.testing import assert_pprint_golden
from torch import nn

import pytest
import torch

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
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


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
        golden_dir=_TESTDATA,
        golden_name="rms_norm",
        build_module=lambda: RMSNorm.Config(4).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_centered_rms_norm_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="centered_rms_norm",
        build_module=lambda: CenteredRMSNorm.Config(4).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_layer_norm_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="layer_norm",
        build_module=lambda: LayerNorm.Config(4).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_batch_norm_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="batch_norm",
        build_module=lambda: BatchNorm.Config(4).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_batch_renorm_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="batch_renorm",
        build_module=lambda: BatchRenorm.Config(channels_in=4).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


def test_batch_norm2d_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="batch_norm2d",
        build_module=lambda: BatchNorm2d.Config(4).make(),
        build_input=lambda: torch.randn(2, 4, 2, 2),
        seed=0,
    )


def test_group_norm2d_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="group_norm2d",
        build_module=lambda: GroupNorm2d.Config(4, num_groups=2).make(),
        build_input=lambda: torch.randn(2, 4, 2, 2),
        seed=0,
    )


def test_group_norm_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="group_norm",
        build_module=lambda: GroupNorm.Config(4, num_groups=2).make(),
        build_input=lambda: torch.randn(2, 3, 4),
        seed=0,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
