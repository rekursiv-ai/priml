"""Tests for norm module."""

from __future__ import annotations

import torch

from priml.model.norm import (
    BatchNorm,
    BatchNorm2d,
    CenteredRMSNorm,
    GroupNorm,
    GroupNorm2d,
    LayerNorm,
    RMSNorm,
)
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


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


def test_norm_forward_drops_extra_args():
    m = RMSNorm.Config(64).make()
    x = torch.randn(2, 8, 64)
    assert m(x, "extra", key="val").shape == (2, 8, 64)


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


def test_centered_rmsnorm_drops_extra_args():
    m = CenteredRMSNorm.Config(32).make()
    x = torch.randn(2, 8, 32)
    assert m(x, "extra", key="val").shape == (2, 8, 32)


def test_groupnorm_arbitrary_leading_dims():
    m = GroupNorm.Config(64, num_groups=4).make()
    x = torch.randn(2, 3, 8, 64)
    out = m(x)
    ref = m(x.reshape(-1, 8, 64)).reshape_as(x)
    assert out.shape == x.shape
    assert torch.allclose(out, ref)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
