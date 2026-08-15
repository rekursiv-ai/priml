"""Tests for the nanochat language model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

import pytest
import torch

from priml.baselines.nanochat import experiments
from priml.baselines.nanochat.model import NanoChatLM
from priml.model.attention import OutputGate, SelfAttention
from priml.model.rope import RoPE
from priml.model.softcap import SoftCap
from priml.model.transformer import TransformerBlock
from priml.model.value_gated_attention import ValueGatedAttention
from priml.testing.bfb import assert_bfb_against_golden, randomize_parameters


_GOLDEN_DIR = Path(__file__).parent / "goldens"

VOCAB = 32
SEQ = 16


def _config(**overrides: Any) -> NanoChatLM.Config:
    config = NanoChatLM.Config()
    config.vocab_size = VOCAB
    config.max_seq_len = SEQ
    config.channels_in = 16
    config.num_layers = 2
    long_attn = config.template.attn
    assert isinstance(long_attn, ValueGatedAttention.Config)
    long_attn.window_pattern = "L"
    assert isinstance(config.template.attn, ValueGatedAttention.Config)
    config.template.attn.channels_head = 8
    config.template.attn.gate_channels = 4
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
    model = _model()
    head = model.lm_head
    assert isinstance(head, SoftCap)
    inner = head.inner
    assert isinstance(inner, nn.Linear)
    with torch.no_grad():
        # Drive the head hard enough that an uncapped model would exceed it.
        inner.weight.mul_(1e3)
        logits = model(_tokens())
    assert float(logits.abs().max()) <= head.cap


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

    The window is a CONFIG field, so each arm is its own module -- which is the
    point of moving it there: a layer's reach is fixed when it is built, not
    chosen per call by whoever holds it.
    """

    def moved(*, window: int) -> bool:
        config = ValueGatedAttention.Config()
        config.channels_in = 16
        config.channels_head = 8
        config.gate_channels = 4
        config.window = window
        torch.manual_seed(0)
        attention = config.make()
        randomize_parameters(attention, seed=1, std=0.5)

        torch.manual_seed(2)
        x = torch.randn(2, SEQ, 16)
        perturbed = x.clone()
        perturbed[:, 0] = torch.randn(2, 16)
        cos_sin = RoPE.Config(channels_head=8).make()(torch.arange(SEQ))
        with torch.no_grad():
            before = attention(x, cos_sin=cos_sin)[:, -1]
            after = attention(perturbed, cos_sin=cos_sin)[:, -1]
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
    gated = sum(p.numel() for p in _model(value_embedding_stride=2).parameters())
    assert gated > plain
    assert len(_model().value_embeds) == 0
    assert set(_model(value_embedding_stride=1).value_embeds) == {"0", "1"}


def test_the_value_gate_starts_transparent() -> None:
    """Zero-initialized, ``2 * sigmoid(0)`` is exactly 1.

    The gate must pass the value embedding through unchanged at init, so an
    A/B against the no-embedding baseline starts from the same behavior and
    the model has to LEARN to attenuate.
    """
    model = _model(value_embedding_stride=1)
    for block in model.blocks:
        assert torch.equal(
            block.attn.value_gate.weight,
            torch.zeros_like(
                block.attn.value_gate.weight,
            ),
        )


def test_a_negative_stride_is_rejected() -> None:
    """The stride is the ONLY way to name the gated layers, so a bad one has
    no list to fall back to and would silently gate nothing.
    """
    with pytest.raises(ValueError, match="value_embedding_stride"):
        _config(value_embedding_stride=-1).copy_tree().finalize()


def test_the_gated_layers_count_back_from_the_last() -> None:
    """The deepest layer always gets a table: the embedding is a path from the
    raw tokens to the output, worth the most where the stream is most
    processed. Counting FORWARD would gate layer 0 and skip the last.
    """
    config = _config(num_layers=4, value_embedding_stride=2)
    assert config.value_embedding_layers == [1, 3]


def test_layers_disagreeing_on_head_shape_are_rejected() -> None:
    """The value embeddings and rotary factors are shared across layers.

    Both are built once, to layer 0's geometry, so a per-layer list declaring a
    different head shape further down is a contradiction -- one that otherwise
    survives construction and dies in the forward as a bare reshape failure
    naming a tensor size rather than the layer.
    """
    config = _config()
    template = config.template.attn
    assert isinstance(template, ValueGatedAttention.Config)
    template.heads = 2

    blocks: list[Any] = []
    for heads in (2, 4):  # 2 * 8 = 16 inner, against 4 * 8 = 32
        block = config.template.copy_tree()
        attention = block.attn
        assert isinstance(attention, ValueGatedAttention.Config)
        attention.heads = heads
        blocks.append(block)
    config.block = blocks
    config.num_layers = len(blocks)

    with pytest.raises(ValueError, match="same attention head geometry"):
        config.copy_tree().finalize()


def test_a_uniform_stack_of_explicit_blocks_still_builds() -> None:
    """The check must not reject the ordinary per-layer list."""
    config = _config()
    config.block = [config.template.copy_tree() for _ in range(config.num_layers)]
    torch.manual_seed(0)
    assert config.make()(_tokens()).shape[-1] == VOCAB


def test_a_wrapped_attention_still_reports_its_own_head_geometry() -> None:
    """The block answers, so a wrapper cannot hide the shape it composes.

    ``OutputGate`` scales an output without reshaping one, so its geometry is
    the wrapped module's. A reader reaching for ``block.attn.channels_head``
    instead reads the gate's own absent field and silently reports one head of
    the full model width -- which builds rotary factors and a value-embedding
    table of the wrong size.
    """
    gated = TransformerBlock.Config(
        channels_in=512,
        attn=OutputGate.Config(
            channels_in=512,
            inner=SelfAttention.Config(heads=4, channels_head=128),
        ),
    )
    assert (gated.channels_head, gated.heads) == (128, 4)


def test_an_injected_layer_keeps_the_values_it_was_given() -> None:
    """A per-layer list is the caller's, so finalize must not copy it away.

    The stack settles ``block`` to a list in place for exactly this reason: a
    copy would take every per-layer value the caller set -- here one layer's
    hand-fixed reach -- and leave the originals unreachable, with no error to
    say so.
    """
    blocks = [
        TransformerBlock.Config(
            attn=ValueGatedAttention.Config(channels_head=8, gate_channels=4),
        )
        for _ in range(2)
    ]
    first = blocks[0].attn
    assert isinstance(first, ValueGatedAttention.Config)
    first.window = 3

    config = _config(num_layers=2)
    config.block = blocks
    final = config.copy_tree().finalize()
    assert isinstance(final.block, list)
    windows: list[int] = []
    for built in final.block:
        attn = built.attn
        assert isinstance(attn, ValueGatedAttention.Config)
        windows.append(attn.window)
    # Layer 0 keeps what it was handed; the last is always the full context.
    assert windows == [3, SEQ]


def test_flops_read_the_blocks_real_head_count() -> None:
    """Heads are their own field, not ``channels_in // channels_head``.

    A model whose attention is wider than its residual stream is legal -- the
    two widths are decoupled -- so deriving the head count from the model width
    silently reports the wrong FLOPs for exactly the configs that need the
    estimate most.
    """

    # Widening the heads moves BOTH terms -- bigger projections and a bigger
    # attention span. They are separated by changing ONLY the span: halving
    # every window leaves every parameter untouched, so the drop is purely
    # attention, and its ABSOLUTE size is pinned against the closed form
    # ``12 * inner * span`` rather than against a mirror of the source.
    def span_drop(*, heads: int) -> int:
        """FLOPs lost when layer 0's window halves; layer 1 is always full."""
        built: list[int] = []
        for pattern in ("L", "SL"):
            variant = _config()
            variant_attention = variant.template.attn
            assert isinstance(variant_attention, ValueGatedAttention.Config)
            variant_attention.window_pattern = pattern
            variant_attention.heads = heads
            torch.manual_seed(0)
            built.append(variant.make().flops_per_token())
        return built[0] - built[1]

    # One layer drops from SEQ to SEQ // 2 positions, at 12 * inner each.
    assert span_drop(heads=2) == 12 * (2 * 8) * (SEQ - SEQ // 2)
    assert span_drop(heads=4) == 12 * (4 * 8) * (SEQ - SEQ // 2)


def test_flops_exclude_lookup_tables() -> None:
    """A gather does no arithmetic, so a bigger vocabulary is not more FLOPs."""
    small = _model().flops_per_token()
    large = _model(vocab_size=VOCAB * 4).flops_per_token()
    # The head is a matmul and does grow; the embedding tables must not.
    assert large > small
    assert large - small == 6 * (VOCAB * 4 - VOCAB) * 16


def test_the_token_table_is_drawn_at_unit_variance() -> None:
    """The recipe's spread, and the one thing no shape check can see.

    Every priml initializer divides by ``sqrt(depth + 1)`` and ``normal``
    defaults that depth to 1, so a table asking for ``std=1.0`` and omitting
    the depth is drawn at 0.707 -- a real difference in the model, invisible to
    every name, shape, and dtype assertion in this file, and erased by
    ``karpathy_parity.py`` before it compares anything (that script copies the
    reference's weights in). The table feeds an RMS norm, which divides its
    scale out, so what this pins is the RELATIVE spread the recipe specifies.

    References:
      https://github.com/karpathy/autoresearch
        ``train.py:150``: ``normal_(wte.weight, mean=0.0, std=1.0)``.

    """
    # Built through the MODEL: the vocabulary and the width are pushed down by
    # its finalize, so a table built from its own config alone has neither.
    torch.manual_seed(0)
    model = experiments.exp001().step.model.copy_tree().finalize().make()
    weight = model.embed.inner.weight.detach().float()
    # Loose enough for the draw, far tighter than the 0.707 the bug produced.
    assert abs(float(weight.std()) - 1.0) < 0.02


def test_the_output_projection_is_drawn_near_zero() -> None:
    """Near-uniform first logits, so early gradients teach the body.

    The same depth trap reaches this one: it asks for ``std=0.001`` and would
    otherwise realize 0.0007.
    """
    torch.manual_seed(0)
    model = experiments.exp001().step.model.copy_tree().finalize().make()
    head = model.lm_head
    assert isinstance(head, SoftCap)
    inner = head.inner
    assert isinstance(inner, nn.Linear)
    weight = inner.weight.detach().float()
    assert abs(float(weight.std()) / 0.001 - 1.0) < 0.05


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
        build_module=lambda: _config(value_embedding_stride=1).make(),
        build_input=_tokens,
        seed=0,
    )


def test_the_shipped_experiments_forward_bfb() -> None:
    """Freeze the model the LADDER builds, not one this file assembles.

    The two goldens above are minted over a config written here, so a change to
    ``exp001`` -- a different norm epsilon, a dropped bf16 knob, another window
    pattern -- leaves them green while every shipped rung moves. This one is
    built from ``exp_smoke``, which differs from ``exp001`` only in size, so a
    change to any shared field lands here.

    Run under autocast because the recipe declares narrow tables: outside it a
    bfloat16 stream reaches a float32 projection and the matmul refuses. That
    pairing IS the recipe, so the golden freezes it rather than widening it
    away.
    """

    def _model() -> Any:
        return experiments.exp_smoke().step.model

    def build() -> nn.Module:
        built = _model().make()
        assert isinstance(built, NanoChatLM)
        return built

    def run(module: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            out = module(tokens)
        assert isinstance(out, torch.Tensor)
        return out

    def build_input() -> torch.Tensor:
        return torch.randint(0, _model().vocab_size, (2, _model().max_seq_len))

    assert_bfb_against_golden(
        golden_dir=_GOLDEN_DIR,
        golden_name="exp_smoke_forward_min_cpu",
        build_module=build,
        build_input=build_input,
        seed=0,
        run=run,
    )


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
