"""Tests for init module."""

from __future__ import annotations

from torch import nn

import torch

from priml.model.init import (
    call_init,
    dirac,
    kaiming_normal,
    kaiming_uniform,
    mup_output,
    normal,
    truncated_normal,
    xavier_normal,
    xavier_uniform,
)
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


def test_call_init_with_depth():
    w = torch.empty(16, 16)
    kaiming_uniform(w, depth=3)
    assert w.std() > 0


def test_call_init_without_depth():
    """call_init skips depth kwarg for fns that don't accept it."""
    w = torch.empty(16, 16)
    call_init(nn.init.xavier_uniform_, w, depth=3)
    assert w.std() > 0


def test_call_init_signature_inspection_failure():
    """call_init handles uninspectable callables gracefully."""

    class Uninspectable:  # pyright: ignore[reportUnusedClass]
        __call__ = None  # breaks inspect.signature

    # Should not raise
    w = torch.empty(4, 4)
    nn.init.zeros_(w)
    call_init(nn.init.zeros_, w, depth=2)
    assert w.abs().sum() == 0


def test_all_init_fns():
    for fn in (
        kaiming_uniform,
        kaiming_normal,
        xavier_uniform,
        xavier_normal,
        normal,
        truncated_normal,
        mup_output,
    ):
        w = torch.empty(32, 32)
        fn(w, depth=2)
        assert w.std() > 0, f"{fn.__name__} produced zero std"


def test_depth_scaling():
    torch.manual_seed(0)
    w0 = torch.empty(64, 64)
    kaiming_uniform(w0, depth=0)

    torch.manual_seed(0)
    w3 = torch.empty(64, 64)
    kaiming_uniform(w3, depth=3)

    assert w0.std() > w3.std()


def test_depth_zero_no_scaling():
    """depth=0 and depth=-1 should produce no scaling."""
    torch.manual_seed(0)
    w_neg = torch.empty(64, 64)
    kaiming_uniform(w_neg, depth=-1)

    torch.manual_seed(0)
    w_zero = torch.empty(64, 64)
    kaiming_uniform(w_zero, depth=0)

    assert torch.allclose(w_neg, w_zero)


def test_truncated_normal_variance_correction_realizes_requested_std():
    """With correction on, realized std equals the request; off, it undershoots."""
    torch.manual_seed(0)
    corrected = torch.empty(400_000)
    truncated_normal(corrected, std=1.0, depth=-1, variance_correction=True)
    torch.manual_seed(0)
    uncorrected = torch.empty(400_000)
    truncated_normal(uncorrected, std=1.0, depth=-1)

    assert abs(corrected.std().item() - 1.0) < 0.01
    # ~0.88: the truncated tail mass the correction restores.
    assert uncorrected.std().item() < 0.92


def test_truncated_normal_default_is_uncorrected():
    """The flag defaults off, so existing callers keep their init unchanged."""
    torch.manual_seed(0)
    default = torch.empty(4096)
    truncated_normal(default, std=0.02, depth=-1)
    torch.manual_seed(0)
    explicit = torch.empty(4096)
    truncated_normal(explicit, std=0.02, depth=-1, variance_correction=False)

    assert torch.equal(default, explicit)


def test_truncated_normal_respects_scaled_bounds():
    """Correction scales the truncation bounds along with the std."""
    torch.manual_seed(0)
    w = torch.empty(100_000)
    truncated_normal(w, std=1.0, depth=-1, variance_correction=True)

    assert w.abs().max().item() <= 2.0 * 1.1372


def test_truncated_normal_corrected_zero_std_zeros_tensor():
    w = torch.ones(16)
    truncated_normal(w, std=0.0, depth=-1, variance_correction=True)
    assert w.abs().sum() == 0


def test_truncated_normal_corrected_depth_scaling():
    torch.manual_seed(0)
    w0 = torch.empty(4096)
    truncated_normal(w0, std=1.0, depth=0, variance_correction=True)
    torch.manual_seed(0)
    w3 = torch.empty(4096)
    truncated_normal(w3, std=1.0, depth=3, variance_correction=True)
    assert torch.allclose(w3, w0 / 2.0)


def test_dirac_conv2d():
    w = torch.empty(8, 8, 3, 3)
    dirac(w)
    # Center pixel of each filter for matching in/out channel should be ~1.
    center = w[:, :, 1, 1]
    assert torch.allclose(center, torch.eye(8), atol=1e-6)
    # Non-center pixels should be zero.
    mask = torch.ones(3, 3, dtype=torch.bool)
    mask[1, 1] = False
    assert torch.allclose(w[:, :, mask], torch.zeros(8, 8, 8))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
