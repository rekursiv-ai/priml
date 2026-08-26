"""Tests for priml.model.transformer.causal_lm."""

from __future__ import annotations

from pathlib import Path
from typing import Self, override
from unittest.mock import Mock

from configgle import PartialConfig
from torch import Tensor

import pytest
import torch

from priml.model.attention.rope import RoPE
from priml.model.attention.self_attention import SelfAttention
from priml.model.custom_types import DepthIndex
from priml.model.generate import generate
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.causal_lm import CausalLM
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.golden import assert_text_golden


_TESTDATA = Path(__file__).parent.resolve() / "testdata"


def _tiny_config(tie: bool = False) -> CausalLM.Config:
    return CausalLM.Config(
        vocab_size=128,
        channels_in=32,
        num_layers=2,
        block=TransformerBlock.Config(
            attn=SelfAttention.Config(
                num_heads=4,
                channels_head=8,
                causal=True,
                rope=RoPE.Config(channels_head=8),
            ),
        ),
        final_norm=RMSNorm.Config(),
        tie_embeddings=tie,
    )


def _canonical_config() -> CausalLM.Config:
    return CausalLM.Config(
        vocab_size=32,
        channels_in=16,
        num_layers=1,
        block=TransformerBlock.Config(
            attn=SelfAttention.Config(num_heads=2, channels_head=8, causal=True),
        ),
        final_norm=RMSNorm.Config(),
    )


def test_causal_lm_config_pprint(request: pytest.FixtureRequest) -> None:
    assert_text_golden(
        request,
        test_file=__file__,
        name="causal_lm",
        rendered=_canonical_config().pformat(hide_default_values=False),
    )


def test_causal_lm_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_TESTDATA,
        golden_name="causal_lm",
        build_module=lambda: _canonical_config().make(),
        build_input=lambda: torch.tensor([[0, 1, 2, 3]]),
        seed=0,
    )


def test_model_forwards_the_open_message_bus_to_every_block() -> None:
    messages: list[object] = []

    def kernel(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        message: object,
        **kwargs: object,
    ) -> Tensor:
        del k, v, kwargs
        messages.append(message)
        return q

    config = _tiny_config()
    assert isinstance(config.block, TransformerBlock.Config)
    assert isinstance(config.block.attn, SelfAttention.Config)
    config.block.attn.attn_kernel = PartialConfig(kernel)
    message = object()

    config.make()(torch.randint(0, 128, (1, 4)), message=message)

    assert messages == [message, message]


def test_forward_shape():
    m = _tiny_config().make()
    toks = torch.randint(0, 128, (2, 6))
    out = m(toks)
    assert out.shape == (2, 6, 128)


def test_tied_embeddings():
    m = _tiny_config(tie=True).make()
    assert m.lm_head is None
    toks = torch.randint(0, 128, (1, 4))
    out = m(toks)
    assert out.shape == (1, 4, 128)


def test_separate_lm_head():
    m = _tiny_config(tie=False).make()
    assert isinstance(m.lm_head, Linear)
    # Distinct parameter, not the embed matrix.
    assert m.lm_head.weight.data_ptr() != m.embed.weight.data_ptr()


def test_explicit_lm_head_receives_model_dimensions() -> None:
    config = _tiny_config()
    config.lm_head = Linear.Config()

    model = config.make()

    assert isinstance(model.lm_head, Linear)
    assert model.lm_head.in_features == config.channels_in
    assert model.lm_head.out_features == config.vocab_size


def test_num_layers_materialized():
    m = _tiny_config().make()
    assert len(m.blocks) == 2
    assert [block.depth_index for block in m.blocks] == [((0, 2),), ((1, 2),)]


def test_explicit_block_list_gets_global_depth_indices() -> None:
    config = _tiny_config()
    assert isinstance(config.block, TransformerBlock.Config)
    config.block = [config.block.copy_tree(), config.block.copy_tree()]

    model = config.make()

    assert [block.depth_index for block in model.blocks] == [((0, 2),), ((1, 2),)]


def test_generate_interop():
    m = _tiny_config().make()
    m.eval()
    prompt = torch.randint(0, 128, (1, 4))
    gen = generate(m, prompt, max_new_tokens=3, temperature=0.0, max_seq_len=16)
    # Prompt + 3 generated.
    assert gen.shape == (1, 7)


def test_generate_rejects_prompt_longer_than_cache():
    """A prompt longer than ``max_seq_len`` is rejected up front.

    Regression for MODEL-003: an over-long prompt previously hit the
    KVCache overflow path with corrupt slices instead of a clear error.
    """
    m = _tiny_config().make()
    m.eval()
    prompt = torch.randint(0, 128, (1, 8))
    with pytest.raises(ValueError, match="prompt length"):
        generate(m, prompt, max_new_tokens=1, max_seq_len=4)


def test_tied_embeddings_ignore_lm_head_config() -> None:
    class RaisingHead(Linear.Config):
        @override
        def finalize(self) -> Self:
            raise ValueError("lm_head finalized")

    config = _tiny_config(tie=True)
    config.lm_head = RaisingHead()

    config.make()


def test_causal_lm_rejects_width_changing_blocks() -> None:
    config = CausalLM.Config(
        vocab_size=128,
        channels_in=32,
        num_layers=2,
        block=Linear.Config(32, 16),
        final_norm=RMSNorm.Config(),
    )

    with pytest.raises(ValueError, match=r"block.*channels_out"):
        config.make()


def test_block_expansion_preserves_identity_sensitive_leaves() -> None:
    class Initializer:
        def __call__(
            self,
            tensor: Tensor,
            *,
            depth_index: DepthIndex = (),
        ) -> None:
            del depth_index
            torch.nn.init.zeros_(tensor)

        def __deepcopy__(self, memo: dict[int, object]) -> object:
            del memo
            raise AssertionError("leaf must remain aliased")

    config = CausalLM.Config(
        vocab_size=128,
        channels_in=32,
        num_layers=2,
        block=Linear.Config(32, 32, init_weight=Initializer()),
        final_norm=RMSNorm.Config(),
    )

    config.make()


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="vocab_size"):
        CausalLM.Config(vocab_size=0, channels_in=8, num_layers=1).make()
    with pytest.raises(ValueError, match="num_layers"):
        CausalLM.Config(vocab_size=8, channels_in=8, num_layers=0).make()
    with pytest.raises(ValueError, match="channels"):
        CausalLM.Config(vocab_size=8, channels_in=0, num_layers=1).make()


def test_causal_lm_config_reports_residual_width() -> None:
    assert _tiny_config().finalize().channels_out == 32


def test_causal_lm_rejects_wrong_block_count() -> None:
    config = _tiny_config()
    assert isinstance(config.block, TransformerBlock.Config)
    config.block = [config.block]

    with pytest.raises(ValueError, match="block list length 1 != num_layers=2"):
        config.make()


def test_causal_lm_rejects_wrong_block_input_width() -> None:
    config = _tiny_config()
    assert isinstance(config.block, TransformerBlock.Config)
    config.block.channels_in = 16

    with pytest.raises(ValueError, match=r"block\[0\].channels_in=16"):
        config.make()


def test_causal_lm_reset_visits_every_parameterized_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_config().make()
    modules = [model.embed, model.final_norm, *model.blocks, model.lm_head]
    resetters: list[Mock] = []
    for module in modules:
        assert module is not None
        reset = Mock()
        monkeypatch.setattr(module, "reset_parameters", reset)
        resetters.append(reset)

    model.reset_parameters()

    for reset in resetters:
        reset.assert_called_once_with()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
