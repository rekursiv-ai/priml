"""Tests for ``TransformerBlock``, including bit-for-bit golden coverage.

Regenerate after an intentional numeric change::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest \
        priml/model/transformer/block_test.py

Run through ``pytest``: the priml ``conftest.py`` sets ``MKL_CBWR`` and caps
math threads before torch imports. Minting from bare Python skips that setup.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import warnings

from configgle.testing import assert_pprint_golden
from torch.utils.checkpoint import checkpoint as real_checkpoint

import pytest
import torch

from priml.model.attention.kvcache import KVCache
from priml.model.attention.self_attention import SelfAttention
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
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


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def _canonical_config() -> TransformerBlock.Config:
    return TransformerBlock.Config(
        channels_in=16,
        attn=SelfAttention.Config(num_heads=2, channels_head=8),
    )


def test_transformer_block_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="transformer_block",
        config=_canonical_config(),
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


def test_transformer_block_cached_rejects_attention_without_cached_path() -> None:
    model = TransformerBlock.Config(
        channels_in=16,
        attn=Linear.Config(16, 16),
    ).make()
    cache = KVCache.alloc(batch=1, num_heads=1, max_seq=1, channels_head=1)

    with pytest.raises(TypeError, match="cached attention"):
        model.forward_cached(torch.randn(1, 1, 16), cache=cache)


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
    config = TransformerBlock.Config(
        channels_in=16,
        channels_out=8,
        attn=SelfAttention.Config(num_heads=2, channels_head=8),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "TransformerBlock.Config" in config.pformat(hide_default_values=False)
    with pytest.raises(ValueError, match="channels_in=16 must equal channels_out=8"):
        config.make()


def test_transformer_block_depth_propagation():
    cfg = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(num_heads=4, channels_head=16),
        depth_index=((5, 6),),
    ).finalize()
    assert isinstance(cfg.ffn, SwiGLU.Config)
    assert cfg.ffn.depth_index == ((5, 6),)


def test_transformer_block_preserves_explicit_child_configuration() -> None:
    config = TransformerBlock.Config(
        channels_in=16,
        norm1=RMSNorm.Config(channels_in=7, channels_out=7),
        ffn=SwiGLU.Config(shard="rowwise", depth_index=((1, 2),)),
    )

    finalized = config.copy_tree().finalize()

    assert isinstance(finalized.norm1, RMSNorm.Config)
    assert finalized.norm1.channels_in == 7
    assert isinstance(finalized.ffn, SwiGLU.Config)
    assert finalized.ffn.shard == "rowwise"
    assert finalized.ffn.depth_index == ((1, 2),)
    # The norm owns its width: it raises from torch at the shape it was built
    # for, which the block could not have named.
    with pytest.raises(RuntimeError, match="normalized_shape"):
        config.make()(torch.randn(2, 3, 16))


def test_block_checkpoint_skipped_under_eval():
    """``checkpoint=True`` must NOT wrap the block in ``torch.utils.checkpoint``
    when grad is off (eval / ``no_grad`` / ``inference_mode``).

    Wrapping a block in ``torch.utils.checkpoint`` under ``inference_mode`` can
    deadlock a multi-rank eval, and checkpointing saves no memory without a
    backward, so the grad-mode gate must skip it. Patches ``torch_checkpoint`` to
    fail if called, then forwards under both eval contexts.
    """
    m = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(num_heads=4, channels_head=16),
        checkpoint=True,
    ).make()
    x = torch.randn(2, 8, 64)

    def _boom(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("torch_checkpoint must not run under eval (grad off)")

    for ctx in (torch.no_grad, torch.inference_mode):
        with patch("priml.model.transformer.block.torch_checkpoint", _boom), ctx():
            out = m(x)
            assert isinstance(out, torch.Tensor)
            assert out.shape == (2, 8, 64)


def test_block_checkpoint_wraps_under_grad():
    """With grad enabled, ``checkpoint=True`` does enter ``torch_checkpoint``."""
    m = TransformerBlock.Config(
        channels_in=64,
        attn=SelfAttention.Config(num_heads=4, channels_head=16),
        checkpoint=True,
    ).make()
    x = torch.randn(2, 8, 64, requires_grad=True)
    spy = Mock(wraps=real_checkpoint)

    with patch("priml.model.transformer.block.torch_checkpoint", spy):
        out = m(x)
        assert isinstance(out, torch.Tensor)
        out.sum().backward()
    assert spy.call_count > 0, "checkpoint=True did not checkpoint during training"


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
