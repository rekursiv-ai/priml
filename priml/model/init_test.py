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
