"""Tests for the nanochat language model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

import pytest
import torch

from priml.baselines.nanochat.model import (
    NanoChatLM,
    ValueGatedAttention,
    head_width,
    window_sizes,
)
from priml.model.norm import RMSNorm
from priml.model.rope import RoPE
from priml.testing.bfb import assert_bfb_against_golden, randomize_parameters


_GOLDEN_DIR = Path(__file__).parent / "goldens"

VOCAB = 32
SEQ = 16


def _config(**overrides: Any) -> NanoChatLM.Config:
    config = NanoChatLM.Config()
    config.vocab_size = VOCAB
    config.max_seq_len = SEQ
    config.channels = 16
    config.num_layers = 2
    config.window_pattern = "L"
    assert isinstance(config.block.attn, ValueGatedAttention.Config)
    config.block.attn.channels_head = 8
    config.block.attn.gate_channels = 4
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def _model(**overrides: Any) -> NanoChatLM:
    torch.manual_seed(0)
    return _config(**overrides).make()


def _mixing_model(**overrides: Any) -> NanoChatLM:
    """A model whose blocks actually mix positions.

    Every output projection is zero-initialized -- that is the recipe, so a
    fresh block is the identity on its residual stream. A test of MIXING
    (causality, windowing) would therefore pass on a model that never attends
    at all, so the weights are randomized before asking.
    """
    model = _model(**overrides)
    randomize_parameters(model, seed=1, std=0.5)
    return model


def _tokens() -> torch.Tensor:
    return torch.randint(0, VOCAB, (2, SEQ))


def test_forward_returns_logits_per_position() -> None:
    assert _model()(_tokens()).shape == (2, SEQ, VOCAB)


def test_a_longer_input_than_the_context_is_rejected() -> None:
    """Silently truncating would score a prefix and report it as the whole."""
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        _model()(torch.randint(0, VOCAB, (2, SEQ + 1)))


def test_logits_are_bounded_by_the_softcap() -> None:
    """The cap is what keeps a large learning rate stable, so it must bind."""
    model = _model(logit_softcap=2.0)
    head = model.lm_head
    assert isinstance(head, nn.Linear)
    with torch.no_grad():
        # Drive the head hard enough that an uncapped model would exceed it.
        head.weight.mul_(1e3)
        logits = model(_tokens())
    assert float(logits.abs().max()) <= 2.0


def test_attention_is_causal() -> None:
    """A position must not see its own future, or the loss is not a prediction.

    Changing a suffix token and watching an EARLIER position's logits is the
    direct check; a shape or mask error that leaks the future fails here.
    """
    model = _mixing_model()
    tokens = _tokens()
    perturbed = tokens.clone()
    perturbed[:, -1] = (perturbed[:, -1] + 1) % VOCAB
    with torch.no_grad():
        before, after = model(tokens), model(perturbed)
    # The last position legitimately changes; every earlier one must not.
    assert not torch.equal(before[:, -1], after[:, -1])
    assert torch.equal(before[:, :-1], after[:, :-1])


def test_a_window_hides_distant_positions() -> None:
    """Attention past the window must not reach the output.

    Tested on the attention module rather than the model: every stack ends in
    a full-context layer by construction, so a model-level probe would see the
    perturbation regardless and could not fail.
    """
    config = ValueGatedAttention.Config()
    config.channels_in = 16
    config.channels_head = 8
    config.gate_channels = 4
    torch.manual_seed(0)
    attention = config.copy_tree().finalize().make()
    randomize_parameters(attention, seed=1, std=0.5)

    x = torch.randn(2, SEQ, 16)
    perturbed = x.clone()
    perturbed[:, 0] = torch.randn(2, 16)
    cos_sin = RoPE.Config(channels_head=8).make()(torch.arange(SEQ))

    def moved(*, window: int) -> bool:
        with torch.no_grad():
            before = attention(x, cos_sin=cos_sin, window=window)[:, -1]
            after = attention(perturbed, cos_sin=cos_sin, window=window)[:, -1]
        return not torch.equal(before, after)

    # Position 0 lies SEQ-1 back from the last query: inside a full window,
    # outside a half one.
    assert moved(window=-1)
    assert not moved(window=SEQ // 2)


def test_value_embeddings_add_parameters_only_where_named() -> None:
    """A layer not listed must not carry a table, or the ladder confounds
    the mechanism with capacity.
    """
    plain = sum(p.numel() for p in _model().parameters())
    gated = sum(p.numel() for p in _model(value_embedding_layers=[1]).parameters())
    assert gated > plain
    assert len(_model().value_embeds) == 0
    assert set(_model(value_embedding_layers=[0, 1]).value_embeds) == {"0", "1"}


def test_the_value_gate_starts_transparent() -> None:
    """Zero-initialized, ``2 * sigmoid(0)`` is exactly 1.

    The gate must pass the value embedding through unchanged at init, so an
    A/B against the no-embedding baseline starts from the same behavior and
    the model has to LEARN to attenuate.
    """
    model = _model(value_embedding_layers=[0, 1])
    for block in model.blocks:
        assert torch.equal(
            block.attn.value_gate.weight,
            torch.zeros_like(
                block.attn.value_gate.weight,
            ),
        )


def test_a_layer_outside_the_stack_is_rejected() -> None:
    """Naming layer 5 of a 2-layer model builds a table nothing ever reads."""
    with pytest.raises(ValueError, match="outside the"):
        _config(value_embedding_layers=[5]).copy_tree().finalize()


def test_window_sizes_always_end_long() -> None:
    """The last layer predicts the next token, so it must see everything."""
    windows = window_sizes(num_layers=5, max_seq_len=64, pattern="SSSL")
    assert windows == [32, 32, 32, 64, 64]


def test_head_width_falls_back_to_the_model_width() -> None:
    """A block whose attention declares no head width is single-headed."""
    assert head_width(RMSNorm.Config(), 128) == 128


def test_flops_read_the_blocks_real_head_count() -> None:
    """Heads are their own field, not ``channels // channels_head``.

    A model whose attention is wider than its residual stream is legal -- the
    two widths are decoupled -- so deriving the head count from the model width
    silently reports the wrong FLOPs for exactly the configs that need the
    estimate most.
    """
    config = _config()
    attention = config.block.attn
    assert isinstance(attention, ValueGatedAttention.Config)
    attention.heads = 4  # 4 * 8 = 32 wide, against 16 channels
    torch.manual_seed(0)
    wide = config.copy_tree().finalize().make()

    narrow = _model()  # heads inferred: 16 // 8 = 2
    attention_flops = wide.flops_per_token() - _matrix_flops(wide)
    assert attention_flops == 2 * (narrow.flops_per_token() - _matrix_flops(narrow))


def _matrix_flops(model: NanoChatLM) -> int:
    """The parameter-driven half of the estimate, for isolating attention."""
    gathered = {
        id(parameter)
        for module in (model.embed, *model.value_embeds.values())
        for parameter in module.parameters()
    }
    gathered |= {id(model.residual_scale), id(model.skip_scale)}
    return 6 * sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in gathered
    )


def test_flops_exclude_lookup_tables() -> None:
    """A gather does no arithmetic, so a bigger vocabulary is not more FLOPs."""
    small = _model().flops_per_token()
    large = _model(vocab_size=VOCAB * 4).flops_per_token()
    # The head is a matmul and does grow; the embedding tables must not.
    assert large > small
    assert large - small == 6 * (VOCAB * 4 - VOCAB) * 16


def test_forward_bfb() -> None:
    """Freeze exp000's architecture: same weights in, same logits out.

    Guards the whole forward path -- embedding, rotary factors, the windowed
    attention, the residual mixing, the soft cap -- against a refactor that
    changes arithmetic while keeping every shape and name intact.
    """
    assert_bfb_against_golden(
        golden_dir=_GOLDEN_DIR,
        golden_name="plain_min_cpu",
        build_module=lambda: _config().make(),
        build_input=_tokens,
        seed=0,
    )


def test_value_embedding_forward_bfb() -> None:
    """Freeze the gated path separately: it is a different slot value.

    A change to the gate would leave the plain golden green, since that model
    builds no value embedding at all.
    """
    assert_bfb_against_golden(
        golden_dir=_GOLDEN_DIR,
        golden_name="value_embedding_min_cpu",
        build_module=lambda: _config(value_embedding_layers=[0, 1]).make(),
        build_input=_tokens,
        seed=0,
    )


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
