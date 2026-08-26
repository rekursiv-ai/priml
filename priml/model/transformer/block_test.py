"""Tests for ``TransformerBlock``, including bit-for-bit golden coverage.

Regenerate after an intentional numeric change::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest \
        priml/model/transformer/block_test.py

Run through ``pytest``: the priml ``conftest.py`` sets ``MKL_CBWR`` and caps
math threads before torch imports. Minting from bare Python skips that setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from priml.model.attention.self_attention import SelfAttention
from priml.model.linear import Linear
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.testing.bfb import (
    assert_bfb_against_golden,
    bfb_devices,
    move_to_device,
)
from priml.testing.fixtures import (
    cleanup_cuda,  # noqa: F401 -- pytest fixture, injected by name not called
)
from priml.testing.golden import assert_text_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def _canonical_config() -> TransformerBlock.Config:
    return TransformerBlock.Config(
        channels_in=16,
        attn=SelfAttention.Config(num_heads=2, channels_head=8),
    )


def test_transformer_block_config_pprint(request: pytest.FixtureRequest) -> None:
    assert_text_golden(
        request,
        test_file=__file__,
        name="transformer_block",
        rendered=_canonical_config().pformat(hide_default_values=False),
    )


def test_transformer_block_prenorm():
    m = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(num_heads=4, channels_head=16),
        prenorm=True,
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 8, 64)


def test_transformer_block_postnorm():
    m = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(num_heads=4, channels_head=16),
        prenorm=False,
    ).make()
    x = torch.randn(2, 8, 64)
    out = m(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 8, 64)


@pytest.mark.parametrize("prenorm", [True, False])
def test_transformer_block_cached(prenorm: bool) -> None:
    m = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(
            num_heads=4,
            channels_head=16,
            causal=True,
        ),
        prenorm=prenorm,
    ).make()
    assert isinstance(m.attn, SelfAttention)
    cache = m.attn.alloc_kv_cache(batch=2, max_seq=8)

    out, cache = m.forward_cached(torch.randn(2, 8, 64), cache=cache)

    assert out.shape == (2, 8, 64)
    assert cache.length == 8


def test_transformer_block_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    m = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(num_heads=4, channels_head=16),
    ).make()
    resetters: list[Mock] = []
    for module in (m.attn, m.ffn, m.norm1, m.norm2):
        reset = Mock()
        monkeypatch.setattr(module, "reset_parameters", reset)
        resetters.append(reset)

    m.reset_parameters()

    for reset in resetters:
        reset.assert_called_once_with()


def test_transformer_block_config_reports_attention_dimensions() -> None:
    config = TransformerBlock.Config(
        channels_in=16,
        attn=SelfAttention.Config(num_heads=2, channels_head=8),
    )
    assert config.num_heads == 2
    assert config.channels_head == 8

    fallback = TransformerBlock.Config(
        channels_in=16,
        attn=Linear.Config(16, 16),
    )
    assert fallback.num_heads == 1
    assert fallback.channels_head == 16


def test_transformer_block_infers_input_width_from_output() -> None:
    model = TransformerBlock.Config(
        channels_out=16,
        attn=SelfAttention.Config(num_heads=2, channels_head=8),
    ).make()

    assert model(torch.randn(2, 4, 16)).shape == (2, 4, 16)


def test_transformer_block_rejects_width_changing_config() -> None:
    with pytest.raises(ValueError, match="channels_in=16 must equal channels_out=8"):
        TransformerBlock.Config(
            channels_in=16,
            channels_out=8,
            attn=SelfAttention.Config(num_heads=2, channels_head=8),
        ).make()


def test_transformer_block_depth_propagation():
    cfg = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(num_heads=4, channels_head=16),
        depth_index=((5, 6),),
    ).finalize()
    assert isinstance(cfg.ffn, SwiGLU.Config)
    assert cfg.ffn.depth_index == ((5, 6),)


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
        attn=SelfAttention.Config(num_heads=4, channels_head=16),
        checkpoint=True,
    ).make()
    x = torch.randn(2, 8, 64)

    def _boom(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("torch_checkpoint must not run under eval (grad off)")

    for ctx in (torch.no_grad, torch.inference_mode):
        with patch("priml.model.transformer.block.torch_checkpoint", _boom), ctx():
            out = m(x)
            assert isinstance(out, torch.Tensor)
            assert out.shape == (2, 8, 64)


def test_block_checkpoint_wraps_under_grad():
    """With grad enabled, ``checkpoint=True`` does enter ``torch_checkpoint``."""
    from unittest.mock import patch  # noqa: PLC0415

    from torch.utils.checkpoint import checkpoint as real_checkpoint  # noqa: PLC0415

    m = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(num_heads=4, channels_head=16),
        checkpoint=True,
    ).make()
    x = torch.randn(2, 8, 64, requires_grad=True)
    seen = [0]

    def _spy(*args: Any, **kwargs: Any) -> Any:
        seen[0] += 1
        return real_checkpoint(*args, **kwargs)

    with patch("priml.model.transformer.block.torch_checkpoint", _spy):
        out = m(x)
        assert isinstance(out, torch.Tensor)
        out.sum().backward()
    assert seen[0] > 0, "checkpoint=True did not checkpoint during training"


@pytest.mark.parametrize("device", bfb_devices(), ids=str)
def test_transformer_block_bfb(device: str) -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="transformer_block",
        build_module=lambda: _canonical_config().make().to(device),
        build_input=lambda: move_to_device(torch.randn(2, 4, 16), device),
        seed=0,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
