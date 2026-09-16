"""Tests for priml.model.transformer.transformer."""

from __future__ import annotations

from pathlib import Path
from typing import Final, override
from unittest.mock import Mock

import warnings

from configgle import Fig, PartialConfig
from configgle.testing import assert_pprint_golden
from torch import Tensor, nn

import pytest
import torch

from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.rope import RoPE
from priml.model.attention.self_attention import SelfAttention
from priml.model.cost import Cost, cost
from priml.model.custom_types import DepthIndex, TensorModule
from priml.model.embedding import Embedding
from priml.model.generate import generate
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.sequential import Sequential
from priml.model.special import TiedLinear
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


_CWD: Final = Path(__file__).resolve().parent


def _head(tie: bool = False) -> Sequential.Config:
    """Return the language-model head: a final norm, then the vocabulary projection."""
    return Sequential.Config(
        elements=[
            RMSNorm.Config(),
            TiedLinear.Config(tied="proj_in") if tie else Linear.Config(shard="vocab"),
        ],
    )


def _tiny_config(tie: bool = False) -> Transformer.Config:
    return Transformer.Config(
        proj_in=Embedding.Config(channels_in=128, shard="vocab"),
        channels_in=32,
        channels_out=128,
        num_layers=2,
        block=TransformerBlock.Config(
            attn=SelfAttention.Config(
                num_heads=4,
                channels_head=8,
                causal=True,
                rope=RoPE.Config(channels_head=8),
            ),
        ),
        proj_out=_head(tie),
    )


def _canonical_config() -> Transformer.Config:
    return Transformer.Config(
        proj_in=Embedding.Config(channels_in=32, shard="vocab"),
        proj_out=_head(),
        channels_in=16,
        channels_out=32,
        num_layers=1,
        block=TransformerBlock.Config(
            attn=SelfAttention.Config(num_heads=2, channels_head=8, causal=True),
        ),
    )


def test_transformer_config_pprint() -> None:
    assert_pprint_golden(
        test_file=__file__,
        name="transformer",
        config=_canonical_config(),
    )


def test_load_state_dict_absorbs_legacy_projection_keys() -> None:
    model = _tiny_config().make()
    legacy = {
        "in_proj." + k.removeprefix("proj_in.")
        if k.startswith("proj_in.")
        else "out_proj." + k.removeprefix("proj_out.")
        if k.startswith("proj_out.")
        else k: v
        for k, v in model.state_dict().items()
    }
    assert any(k.startswith(("in_proj.", "out_proj.")) for k in legacy)
    fresh = _tiny_config().make()
    fresh.load_state_dict(legacy, strict=True)
    for k, v in model.state_dict().items():
        assert torch.equal(fresh.state_dict()[k], v)


def test_transformer_bfb() -> None:
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="transformer",
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


def test_model_forwards_the_open_message_bus_through_output_layers() -> None:
    messages: list[object] = []
    model = _tiny_config().make()

    class RecordingLayer(torch.nn.Module):
        def __init__(self, wrapped: TensorModule) -> None:
            super().__init__()
            self.wrapped = wrapped

        @override
        def forward(self, input: Tensor, **kwargs: object) -> Tensor:
            messages.append(kwargs["message"])
            return self.wrapped(input)

        def reset_parameters(self) -> None:
            self.wrapped.reset_parameters()

    head = model.proj_out
    assert isinstance(head, Sequential)
    norm, proj = head[0], head[1]
    assert isinstance(norm, RMSNorm)
    assert isinstance(proj, Linear)
    head[0] = RecordingLayer(norm)
    head[1] = RecordingLayer(proj)
    message = object()

    model(torch.randint(0, 128, (1, 4)), message=message)

    assert messages == [message, message]


def test_forward_shape():
    m = _tiny_config().make()
    toks = torch.randint(0, 128, (2, 6))
    out = m(toks)
    assert out.shape == (2, 6, 128)


def test_tied_embeddings():
    m = _tiny_config(tie=True).make()
    assert isinstance(m.proj_out, Sequential)
    assert isinstance(m.proj_out[1], TiedLinear)
    assert [n for n, _ in m.named_parameters() if n.startswith("proj_out.")] == []
    toks = torch.randint(0, 128, (1, 4))
    out = m(toks)
    assert out.shape == (1, 4, 128)


def test_separate_proj_out():
    m = _tiny_config(tie=False).make()
    assert isinstance(m.proj_out, Sequential)
    head = m.proj_out[1]
    assert isinstance(head, Linear)
    assert isinstance(m.proj_in, Embedding)
    # Distinct parameter, not the embed matrix.
    assert head.weight.data_ptr() != m.proj_in.weight.data_ptr()


def test_explicit_proj_out_receives_model_dimensions() -> None:
    config = _tiny_config()
    config.proj_out = Linear.Config()

    model = config.make()

    assert isinstance(model.proj_out, Linear)
    assert model.proj_out.in_features == config.channels_in
    assert model.proj_out.out_features == config.channels_out


def test_transformer_preserves_an_explicit_proj_out_width() -> None:
    """An explicit head width survives finalize; torch rejects a wrong one.

    A wrong input width fails the matmul, naming both operands. The output
    width is the head's to state: the stack reports whatever it says.
    """
    config = _tiny_config()
    config.proj_out = Linear.Config(channels_in=7, channels_out=128)

    finalized = config.copy_tree().finalize()

    assert isinstance(finalized.proj_out, Linear.Config)
    assert finalized.proj_out.channels_in == 7
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        config.make()(torch.zeros(2, 3, dtype=torch.long))


def test_transformer_reports_the_head_width_when_composed() -> None:
    """A composed head derives its width in its own finalize; the stack reads it."""
    config = _tiny_config()
    config.channels_out = -1
    config.proj_out = Sequential.Config(
        elements=[RMSNorm.Config(), Linear.Config(32, 99)],
    )

    finalized = config.copy_tree().finalize()

    assert finalized.channels_out == 99
    assert config.make()(torch.zeros(2, 3, dtype=torch.long)).shape == (2, 3, 99)


def test_num_layers_materialized():
    m = _tiny_config().make()
    assert len(m.blocks) == 2
    depth_indices: list[DepthIndex] = []
    for block in m.blocks:
        assert isinstance(block, TransformerBlock)
        depth_indices.append(block.depth_index)
    assert depth_indices == [((0, 2),), ((1, 2),)]


def test_explicit_block_list_gets_global_depth_indices() -> None:
    config = _tiny_config()
    assert isinstance(config.block, TransformerBlock.Config)
    config.block = [config.block.copy_tree(), config.block.copy_tree()]

    model = config.make()

    depth_indices: list[DepthIndex] = []
    for block in model.blocks:
        assert isinstance(block, TransformerBlock)
        depth_indices.append(block.depth_index)
    assert depth_indices == [((0, 2),), ((1, 2),)]


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


def test_transformer_rejects_width_changing_blocks() -> None:
    config = Transformer.Config(
        proj_in=Embedding.Config(channels_in=128, shard="vocab"),
        proj_out=_head(),
        channels_in=32,
        channels_out=128,
        num_layers=2,
        block=Linear.Config(32, 16),
    )

    # A width-CHANGING block in a width-preserving slot: torch names both
    # operands of the matmul that cannot compose.
    with pytest.raises(RuntimeError, match="shapes cannot be multiplied"):
        config.make()(torch.zeros(2, 3, dtype=torch.long))


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

    config = Transformer.Config(
        proj_in=Embedding.Config(channels_in=128, shard="vocab"),
        proj_out=_head(),
        channels_in=32,
        channels_out=128,
        num_layers=2,
        block=Linear.Config(32, 32, init_weight=Initializer()),
    )

    config.make()


@pytest.mark.parametrize(
    "config",
    [
        Transformer.Config(channels_out=0, channels_in=8, num_layers=1),
        Transformer.Config(channels_out=8, channels_in=8, num_layers=0),
        Transformer.Config(channels_out=8, channels_in=0, num_layers=1),
    ],
)
def test_a_nonsense_width_still_prints(config: Transformer.Config) -> None:
    """A config too degenerate to build is exactly the one worth printing.

    ``pformat`` finalizes a copy, so a ``finalize`` that raised would downgrade
    to a warning and print the tree as TYPED -- without a single propagated
    value. Rejecting the width is torch's job at build time, not finalize's.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "Transformer.Config" in config.pformat(hide_default_values=False)


def test_transformer_config_reports_output_width() -> None:
    assert _tiny_config().finalize().channels_out == 128


def test_transformer_preserves_an_explicit_head_norm_width() -> None:
    """An explicit child width survives finalize; the child owns rejecting it.

    The parent does not re-derive a child's invariant: a norm built at the wrong
    width raises from torch, naming both the shape it expected and the one it
    got, which the parent could not have reported.
    """
    config = _tiny_config()
    config.channels_out = -1
    config.proj_out = RMSNorm.Config(channels_in=7)

    finalized = config.copy_tree().finalize()

    assert isinstance(finalized.proj_out, RMSNorm.Config)
    assert finalized.proj_out.channels_in == 7
    assert finalized.channels_out == 7
    with pytest.raises(RuntimeError, match="normalized_shape"):
        config.make()(torch.zeros(2, 3, dtype=torch.long))


def test_transformer_rejects_wrong_block_count() -> None:
    config = _tiny_config()
    assert isinstance(config.block, TransformerBlock.Config)
    config.block = [config.block]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "Transformer.Config" in config.pformat(hide_default_values=False)
    with pytest.raises(ValueError, match="block list length 1 != num_layers=2"):
        config.make()


def test_transformer_rejects_wrong_block_input_width() -> None:
    config = _tiny_config()
    assert isinstance(config.block, TransformerBlock.Config)
    config.block.channels_in = 16

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert "Transformer.Config" in config.pformat(hide_default_values=False)
    # The BLOCK owns its own width invariant and names itself in the failure.
    with pytest.raises(ValueError, match="for TransformerBlock"):
        config.make()


def test_transformer_reset_visits_every_parameterized_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_config().make()
    modules = [model.proj_in, *model.blocks, model.proj_out]
    resetters: list[Mock] = []
    for module in modules:
        assert module is not None
        reset = Mock()
        monkeypatch.setattr(module, "reset_parameters", reset)
        resetters.append(reset)

    model.reset_parameters()

    for reset in resetters:
        reset.assert_called_once_with()


def test_transformer_cost_is_the_palm_formula() -> None:
    """``6N`` over matrix params plus ``12 * L * inner * seq_len`` attention."""
    config = _tiny_config()
    finalized = config.copy_tree().finalize()
    cost = finalized.cost(seq_len=16)

    model = config.make()
    params = sum(p.numel() for p in model.parameters())
    assert cost.params == params
    embedding = 128 * 32
    matrix = params - embedding
    attention = 12 * 2 * (4 * 8) * 16
    assert cost.training.flops.matmul == 6 * matrix + attention
    assert cost.bytes_state == 2 * 2 * 4 * 8


def test_transformer_cost_matches_torch_through_a_naive_kernel() -> None:
    """The dispatcher's count is the oracle for every ``mm`` and ``bmm``.

    With the naive kernel every product is a registered op, so the whole
    model -- embedding excluded, attention included -- is torch's count.
    """
    config = _tiny_config()
    assert isinstance(config.block, TransformerBlock.Config)
    assert isinstance(config.block.attn, SelfAttention.Config)
    config.block.attn.attn_kernel = SdpaNaive.Config()
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 128, (1, 8)),
        num_tokens=8,
        bus={"seq_len": 8},
        run=_logits_float,
    )


def test_tied_head_cost_owns_no_parameters_but_pays_the_matmul() -> None:
    tied = _tiny_config(tie=True).copy_tree().finalize().cost(seq_len=8)
    untied = _tiny_config(tie=False).copy_tree().finalize().cost(seq_len=8)
    assert tied.training.flops.matmul == untied.training.flops.matmul
    assert tied.params == untied.params - 128 * 32


def test_transformer_cost_reads_every_block_of_a_list() -> None:
    config = _tiny_config()
    wide = TransformerBlock.Config(
        attn=SelfAttention.Config(num_heads=8, channels_head=8, causal=True),
    )
    assert isinstance(config.block, TransformerBlock.Config)
    config.block = [config.block, wide]
    finalized = config.copy_tree().finalize()
    assert isinstance(finalized.block, list)
    blocks = sum((cost(block, seq_len=8) for block in finalized.block), Cost())
    assert finalized.cost(seq_len=8).params == blocks.params + 2 * 128 * 32


class _Unpriced(torch.nn.Module):
    class Config(Fig["_Unpriced"]):
        channels_in: int = -1
        channels_out: int = -1

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    def reset_parameters(self) -> None:
        pass

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return x


def test_transformer_cost_rejects_an_unpriced_slot() -> None:
    config = _tiny_config()
    config.proj_out = _Unpriced.Config()
    with pytest.raises(TypeError, match=r"_Unpriced\.Config has no cost"):
        config.copy_tree().finalize().cost(seq_len=8)


def _logits_float(module: nn.Module, tokens: Tensor) -> Tensor:
    """Run the stack and widen its logits for the harness's reduction."""
    assert isinstance(module, Transformer)
    return module(tokens).float()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
