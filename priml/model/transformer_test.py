"""Tests for transformer module."""

from __future__ import annotations

from typing import Any

import torch

from priml.model.attention import SelfAttention
from priml.model.swiglu import SwiGLU
from priml.model.transformer import TransformerBlock
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)


def test_transformer_block_prenorm():
    m = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(heads=4, channels_head=16),
        prenorm=True,
    ).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)


def test_transformer_block_postnorm():
    m = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(heads=4, channels_head=16),
        prenorm=False,
    ).make()
    x = torch.randn(2, 8, 64)
    assert m(x).shape == (2, 8, 64)


def test_transformer_block_reset():
    m = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(heads=4, channels_head=16),
    ).make()
    m.reset_parameters()


def test_transformer_block_depth_propagation():
    cfg = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(heads=4, channels_head=16),
        depth=5,
    ).finalize()
    assert isinstance(cfg.ffn, SwiGLU.Config)
    assert cfg.ffn.depth == 5


def test_block_checkpoint_skipped_under_eval():
    """``checkpoint=True`` must NOT wrap the block in ``torch.utils.checkpoint``
    when grad is off (eval / ``no_grad`` / ``inference_mode``).

    Wrapping a block in ``torch.utils.checkpoint`` under ``inference_mode`` can
    deadlock a multi-rank eval, and checkpointing saves no memory without a
    backward, so the grad-mode gate must skip it. Patches ``torch_checkpoint`` to
    fail if called, then forwards under both eval contexts.
    """
    from unittest.mock import patch  # noqa: PLC0415

    m = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(heads=4, channels_head=16),
        checkpoint=True,
    ).make()
    x = torch.randn(2, 8, 64)

    def _boom(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("torch_checkpoint must not run under eval (grad off)")

    for ctx in (torch.no_grad, torch.inference_mode):
        with patch("priml.model.transformer.torch_checkpoint", _boom), ctx():
            assert m(x).shape == (2, 8, 64)


def test_block_checkpoint_wraps_under_grad():
    """With grad enabled, ``checkpoint=True`` does enter ``torch_checkpoint``."""
    from unittest.mock import patch  # noqa: PLC0415

    from torch.utils.checkpoint import checkpoint as real_checkpoint  # noqa: PLC0415

    m = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(heads=4, channels_head=16),
        checkpoint=True,
    ).make()
    x = torch.randn(2, 8, 64, requires_grad=True)
    seen = [0]

    def _spy(*args: Any, **kwargs: Any) -> Any:
        seen[0] += 1
        return real_checkpoint(*args, **kwargs)

    with patch("priml.model.transformer.torch_checkpoint", _spy):
        m(x).sum().backward()
    assert seen[0] > 0, "checkpoint=True did not checkpoint during training"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
