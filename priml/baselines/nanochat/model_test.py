"""Tests for the nanochat language model."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Final, override

from configgle import Fig, Makeable
from torch import Tensor, nn

import pytest
import torch

from priml.baselines.nanochat import experiments
from priml.baselines.nanochat.attention import CausalAttention
from priml.baselines.nanochat.model import (
    GatedResidualMix,
    MemoryNanoChatLM,
    NanoChatLM,
    OutputNormFeedForward,
    SourceReuseTransformerBlock,
)
from priml.baselines.nanochat.ngram import HashedNgramTables
from priml.cost import cost
from priml.custom_types import HasCost
from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.output_gate import OutputGate
from priml.model.attention.rope import RoPE
from priml.model.attention.self_attention import SelfAttention
from priml.model.attention.value_gated_attention import ValueGatedAttention
from priml.model.custom_types import TensorModule
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.narrow_embedding import NarrowEmbedding
from priml.model.norm import RMSNorm
from priml.model.residual_mix import ResidualMix
from priml.model.softcap import SoftCap
from priml.model.special import Identity
from priml.model.transformer.block import TransformerBlock
from priml.testing.bfb import assert_bfb_against_golden, randomize_parameters
from priml.testing.cost import assert_cost_matches_torch
from priml.train.parallelism import materialize_meta


if TYPE_CHECKING:
    from priml.model.custom_types import HasAttention


_CWD: Final = Path(__file__).resolve().parent

VOCAB = 32
SEQ = 16


def _config(**overrides: object) -> NanoChatLM.Config:
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


def _model(**overrides: object) -> NanoChatLM:
    torch.manual_seed(0)
    return _config(**overrides).make()


def _memory_config(*, buckets: int = 16) -> MemoryNanoChatLM.Config:
    config = MemoryNanoChatLM.Config()
    config.vocab_size = VOCAB
    config.max_seq_len = 4
    config.channels_in = 16
    config.num_layers = 1
    attention = CausalAttention.Config()
    attention.channels_head = 8
    attention.num_heads = 2
    attention.gate_channels = 8
    attention.window_pattern = "L"
    attention.bigram = True
    config.block = TransformerBlock.Config(attn=attention)
    config.bigrams["0"] = HashedNgramTables.Config(
        num_embeddings=buckets,
        hash_multipliers=((1, 3), (5, 7)),
    )
    return config


# Every output projection is zero-initialized -- that is the recipe, so a fresh block is
# the identity on its residual stream. A test of MIXING (causality, windowing) would
# therefore pass on a model that never attends at all, so the weights are randomized
# before asking.
def _mixing_model(**overrides: object) -> NanoChatLM:
    """Return a model whose blocks actually mix positions."""
    model = _model(**overrides)
    randomize_parameters(model, seed=1, std=0.5)
    return model


def _tokens() -> Tensor:
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
    """A layer not listed must not carry a table.

    Or the ladder confounds the mechanism with capacity.
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
        assert isinstance(block, TransformerBlock)
        assert isinstance(block.attn, ValueGatedAttention)
        gate = block.attn.value_gate
        assert isinstance(gate, Linear)
        assert torch.equal(gate.weight, torch.zeros_like(gate.weight))


def test_a_negative_stride_is_rejected() -> None:
    """The stride is the ONLY way to name the gated layers.

    A bad one has no list to fall back to and would silently gate nothing.
    """
    with pytest.raises(ValueError, match="value_embedding_stride"):
        _config(value_embedding_stride=-1).copy_tree().finalize()


def test_the_gated_layers_count_back_from_the_last() -> None:
    """The deepest layer always gets a table.

    The embedding is a path from the raw tokens to the output, worth the most where the
    stream is most processed. Counting FORWARD would gate layer 0 and skip the last.
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
    template.num_heads = 2

    blocks: list[HasAttention] = []
    for num_heads in (2, 4):  # 2 * 8 = 16 inner, against 4 * 8 = 32.
        block = config.template.copy_tree()
        attention = block.attn
        assert isinstance(attention, ValueGatedAttention.Config)
        attention.num_heads = num_heads
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


def test_a_wrapped_attention_exposes_its_own_head_attributes() -> None:
    """Shape-preserving wrappers expose the wrapped attention's attributes."""
    gated = TransformerBlock.Config(
        channels_in=512,
        attn=OutputGate.Config(
            channels_in=512,
            inner=SelfAttention.Config(num_heads=4, channels_head=128),
        ),
    )
    assert gated.channels_head == 128
    assert gated.num_heads == 4


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

    # Widening the num_heads moves BOTH terms -- bigger projections and a bigger
    # attention span. They are separated by changing ONLY the span: halving
    # every window leaves every parameter untouched, so the drop is purely
    # attention, and its ABSOLUTE size is pinned against the closed form
    # ``12 * inner * span`` rather than against a mirror of the source.
    def span_drop(*, num_heads: int) -> int:
        """FLOPs lost when layer 0's window halves; layer 1 is always full."""
        built: list[int] = []
        for pattern in ("L", "SL"):
            variant = _config()
            variant_attention = variant.template.attn
            assert isinstance(variant_attention, ValueGatedAttention.Config)
            variant_attention.window_pattern = pattern
            variant_attention.num_heads = num_heads
            torch.manual_seed(0)
            built.append(variant.make().flops_per_token())
        return built[0] - built[1]

    # One layer drops from SEQ to SEQ // 2 positions, at 12 * inner each.
    assert span_drop(num_heads=2) == 12 * (2 * 8) * (SEQ - SEQ // 2)
    assert span_drop(num_heads=4) == 12 * (4 * 8) * (SEQ - SEQ // 2)


def test_flops_exclude_lookup_tables() -> None:
    """A gather does no arithmetic, so a bigger vocabulary is not more FLOPs."""
    small = _model().flops_per_token()
    large = _model(vocab_size=VOCAB * 4).flops_per_token()
    # The head is a matmul and does grow; the embedding tables must not.
    assert large > small
    assert large - small == 6 * (VOCAB * 4 - VOCAB) * 16


def test_the_config_prices_itself() -> None:
    """``Utilization`` binds only to a model config implementing ``HasCost``."""
    assert isinstance(_config(), HasCost)


def test_cost_matches_torch_through_a_naive_kernel() -> None:
    """Every matmul the forward issues is in the cost, and every parameter.

    ``SdpaCausal`` dispatches to the CPU SDPA op, which ``FlopCounterMode``
    does not register; the naive kernel makes the same products countable.
    """
    config = _config(value_embedding_stride=1)
    attention = config.template.attn
    assert isinstance(attention, ValueGatedAttention.Config)
    attention.kernel = SdpaNaive.Config()
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, VOCAB, (2, SEQ)),
        seq_len=SEQ,
        batch_size=2,
        num_tokens=SEQ * 2,
        dtype=None,
    )


def test_cost_matmul_flops_agree_with_the_palm_estimate() -> None:
    """Both count six FLOPs per matrix parameter plus the attention products."""
    finalized = _config(value_embedding_stride=1).copy_tree().finalize()
    priced = cost(finalized, seq_len=SEQ, batch_size=1, dtype=None)
    torch.manual_seed(0)
    assert priced["flops", "matmul"].sum() == finalized.make().flops_per_token()


def test_cost_counts_every_lookup_table_but_no_lookup_flops() -> None:
    """The token and value tables are parameters that cost one gather each."""
    plain = cost(
        _config().copy_tree().finalize(),
        seq_len=SEQ,
        batch_size=1,
        dtype=None,
    )
    gated = cost(
        _config(value_embedding_stride=1).copy_tree().finalize(),
        seq_len=SEQ,
        batch_size=1,
        dtype=None,
    )
    # Two value tables of ``VOCAB x (num_heads * channels_head)`` and one gate
    # of ``gate_channels x num_heads`` per layer.
    assert gated.params - plain.params == 2 * (VOCAB * 16 + 4 * 2)
    assert (
        gated["flops", "primal", "selection"].sum()
        == plain["flops", "primal", "selection"].sum()
        == 0
    )
    # Two gathers: an int64 index each, a 16-wide row read and written at fp32.
    assert (
        gated["bytes", "primal", "selection", torch.int64]
        - plain["bytes", "primal", "selection", torch.int64]
        == 8 * 2
    )
    assert (
        gated["bytes", "primal", "selection", torch.float32]
        - plain["bytes", "primal", "selection", torch.float32]
        == 4 * 2 * 2 * 16
    )


def test_cost_distinguishes_batch_reuse_from_attention_window() -> None:
    config = _config().finalize()
    small = config.cost(seq_len=SEQ, batch_size=1, dtype=None)
    large = config.cost(seq_len=SEQ, batch_size=4, dtype=None)
    narrow = config.cost(seq_len=SEQ, batch_size=4, dtype=torch.bfloat16)
    assert large["flops", "primal"] == small["flops", "primal"]
    assert (
        large["bytes", "primal", "matmul"].sum()
        < small["bytes", "primal", "matmul"].sum()
    )
    assert large["flops", "matmul"].sum() / large["bytes", "matmul"].sum() > (
        small["flops", "matmul"].sum() / small["bytes", "matmul"].sum()
    )
    # Activations halve; the tables are stored at torch's default and the
    # lookups' indices are int64, so the selection silo is unchanged.
    for kernel in ("matmul", "elementwise", "reduction"):
        assert (
            narrow["bytes", kernel, torch.bfloat16].sum()
            == large["bytes", kernel, torch.float32].sum() / 2
        )
    assert narrow["bytes", "selection"] == large["bytes", "selection"]
    # The method and the dispatcher agree: seq_len alone means seq_len rows.
    assert small == cost(config, seq_len=SEQ, batch_size=1, dtype=None)
    assert large == cost(config, seq_len=SEQ, batch_size=4, dtype=None)
    # Past the window a longer sequence changes only weight amortization:
    # attention work is identical and matmul traffic only shrinks.
    long = config.cost(seq_len=8192, batch_size=1, dtype=None)
    longer = config.cost(seq_len=32_768, batch_size=1, dtype=None)
    assert longer["flops", "primal"] == long["flops", "primal"]
    assert (
        longer["bytes", "primal", "matmul"].sum()
        < long["bytes", "primal", "matmul"].sum()
    )
    assert (
        longer["bytes", "primal", "reduction"].sum()
        == long["bytes", "primal", "reduction"].sum()
    )


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
    config = experiments.exp001().step.model.copy_tree()
    config.channels_in = 16
    config.num_layers = 1
    config.max_seq_len = 4
    config.vocab_size = 4096
    attention = config.template.attn
    assert isinstance(attention, ValueGatedAttention.Config)
    attention.channels_head = 8
    attention.gate_channels = 4
    model = config.finalize().make()
    assert isinstance(model.embed, NarrowEmbedding)
    assert isinstance(model.embed.inner, Embedding)
    weight = model.embed.inner.weight.detach().float()
    # Loose enough for the draw, far tighter than the 0.707 the bug produced.
    assert abs(float(weight.std()) - 1.0) < 0.02


def test_the_output_projection_is_drawn_near_zero() -> None:
    """Near-uniform first logits, so early gradients teach the body.

    The same depth trap reaches this one: it asks for ``std=0.001`` and would
    otherwise realize 0.0007.
    """
    torch.manual_seed(0)
    config = experiments.exp001().step.model.copy_tree()
    config.channels_in = 16
    config.num_layers = 1
    config.max_seq_len = 4
    config.vocab_size = 4096
    attention = config.template.attn
    assert isinstance(attention, ValueGatedAttention.Config)
    attention.channels_head = 8
    attention.gate_channels = 4
    model = config.finalize().make()
    head = model.lm_head
    assert isinstance(head, SoftCap)
    inner = head.inner
    assert isinstance(inner, nn.Linear)
    weight = inner.weight.detach().float()
    assert abs(float(weight.std()) / 0.001 - 1.0) < 0.05


def test_the_cached_rotation_table_is_the_one_a_fresh_build_produces() -> None:
    """The table is built once for ``max_seq_len`` and sliced per batch.

    Rebuilding it per forward cost 12.5 of the 13.1 ms/step this recipe stood
    above its reference. Caching is sound only if the slice is bit-identical to
    what the rope returns for those positions -- the factors are a
    transcendental, so this is a measurement rather than an algebraic identity.
    """
    model = _model()
    length = SEQ // 2
    cached_cos, cached_sin = model._rotation_table(length, device=torch.device("cpu"))
    fresh_cos, fresh_sin = model.rope(torch.arange(SEQ))
    assert torch.equal(cached_cos, fresh_cos[:length])
    assert torch.equal(cached_sin, fresh_sin[:length])


def test_the_rotation_table_is_rebuilt_when_the_frequencies_are() -> None:
    """``reset_parameters`` re-derives the rope, so a stale table cannot survive.

    Meta-device materialization drives init through it, and a move rebuilds the
    frequencies because the transcendental differs by a bit across devices --
    so a table cached before that would be one no device would have produced.
    """
    model = _model()
    model(torch.randint(0, VOCAB, (2, SEQ)))
    assert model._rotation is not None
    model.reset_parameters()
    assert model._rotation is None


class ResetlessBlock(nn.Module):
    """A parameterless injected block with no reset capability."""

    class Config(Fig["ResetlessBlock"]):
        attn: Makeable[TensorModule] = field(default_factory=SelfAttention.Config)
        """Attention metadata consumed by the model config."""

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return x


def test_reset_accepts_a_resetless_injected_block() -> None:
    config = _config()
    config.block = ResetlessBlock.Config()
    model = config.make()
    assert all(isinstance(block, ResetlessBlock) for block in model.blocks)
    model.reset_parameters()
    assert model(_tokens()).shape == (2, SEQ, VOCAB)


def test_forward_bfb() -> None:
    """Freeze exp000's architecture: same weights in, same logits out.

    Guards the whole forward path -- embedding, rotary factors, the windowed
    attention, the residual mixing, the soft cap -- against a refactor that
    changes arithmetic while keeping every shape and name intact.
    """
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="plain",
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
        golden_dir=_CWD / "testdata",
        golden_name="value_embedding",
        build_module=lambda: _config(value_embedding_stride=1).make(),
        build_input=_tokens,
        seed=0,
    )


def test_the_shipped_experiments_forward_bfb() -> None:
    """Freeze the model the LADDER builds, not one this file assembles.

    The two test artifacts above use a config written here, so a change to
    ``exp001`` -- a different norm epsilon, a dropped bf16 knob, another window
    pattern -- leaves them green while every shipped rung moves. This one is
    built from ``exp_smoke``, which differs from ``exp001`` only in size, so a
    change to any shared field lands here.

    Run under autocast because the recipe declares narrow tables: outside it a
    bfloat16 stream reaches a float32 projection and the matmul refuses. That
    pairing IS the recipe, so the golden freezes it rather than widening it
    away.
    """

    def _model() -> NanoChatLM.Config:
        model = experiments.exp_smoke().step.model
        assert isinstance(model, NanoChatLM.Config)
        return model

    def build() -> nn.Module:
        return _model().make()

    def run(module: nn.Module, tokens: Tensor) -> Tensor:
        assert isinstance(module, NanoChatLM)
        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            return module(tokens)

    def build_input() -> Tensor:
        return torch.randint(0, _model().vocab_size, (2, _model().max_seq_len))

    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="exp_smoke_forward",
        build_module=build,
        build_input=build_input,
        seed=0,
        run=run,
    )


def test_memory_model_defaults_preserve_the_base_decoder() -> None:
    memory = MemoryNanoChatLM.Config()
    memory.channels_in = 16
    memory.vocab_size = 16
    memory.num_layers = 2
    memory.max_seq_len = 4
    memory.value_embedding_stride = 2
    attention = memory.template.attn
    assert isinstance(attention, ValueGatedAttention.Config)
    attention.channels_head = 8
    attention.gate_channels = 4
    base = NanoChatLM.Config().update(memory, skip_missing=True)

    models: list[NanoChatLM] = []
    states: list[Tensor] = []
    for config in (base, memory):
        torch.manual_seed(42)
        models.append(config.make())
        states.append(torch.get_rng_state())
    reference, candidate = models
    assert isinstance(candidate, MemoryNanoChatLM)
    assert candidate.pool_weights is None
    assert not candidate.bigrams
    assert not candidate.trigrams
    assert torch.equal(states[0], states[1])
    assert (
        dict(candidate.named_parameters()).keys()
        == dict(reference.named_parameters()).keys()
    )
    for name, value in candidate.state_dict().items():
        assert torch.equal(value, reference.state_dict()[name]), name
    tokens = torch.tensor([[1, 2, 3, 4], [4, 1, 0, 2]])
    assert torch.equal(candidate(tokens), reference(tokens))


def test_memory_model_direct_make_applies_storage_dtype_without_extra_draws() -> None:
    states: list[Tensor] = []
    models: list[MemoryNanoChatLM] = []
    for dtype in (None, torch.bfloat16):
        config = _memory_config()
        config.dtype = dtype
        config.fused_ngram = True
        torch.manual_seed(43)
        models.append(config.make())
        states.append(torch.get_rng_state())

    assert torch.equal(states[0], states[1])
    narrowed = models[1]
    assert all(parameter.dtype == torch.bfloat16 for parameter in narrowed.parameters())
    table = narrowed.bigrams["0"]
    assert isinstance(table, HashedNgramTables)
    assert len(table.gradient_sinks) == len(table.tables) == 2
    assert all(sink.dtype == torch.float32 for sink in table.gradient_sinks)


def test_memory_model_direct_fused_make_supports_forward_and_backward() -> None:
    config = _memory_config()
    config.fused_ngram = True
    model = config.make()
    block = model.blocks[0]
    assert isinstance(block, TransformerBlock)
    attention = block.attn
    assert isinstance(attention, CausalAttention)
    with torch.no_grad():
        attention.proj_out.weight.normal_()
    table = model.bigrams["0"]
    assert isinstance(table, HashedNgramTables)

    model(torch.tensor([[1, 2, 3, 4]])).sum().backward()

    assert all(torch.count_nonzero(sink) > 0 for sink in table.gradient_sinks)
    assert all(part.weight.grad is None for part in table.tables)


def test_memory_model_meta_materialization_preserves_dtype_and_fused_sinks() -> None:
    config = _memory_config()
    config.dtype = torch.bfloat16
    config.fused_ngram = True
    with torch.device("meta"):
        model = config.make()
    table = model.bigrams["0"]
    assert isinstance(table, HashedNgramTables)
    assert all(parameter.is_meta for parameter in model.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert len(table.gradient_sinks) == len(table.tables) == 2
    assert all(sink.is_meta for sink in table.gradient_sinks)

    materialize_meta(model, torch.device("cpu"))

    assert all(not parameter.is_meta for parameter in model.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert all(
        not sink.is_meta and sink.dtype == torch.float32
        for sink in table.gradient_sinks
    )


def test_memory_table_width_matches_attention_values_not_residual_stream() -> None:
    config = _memory_config()
    attention = config.template.attn
    assert isinstance(attention, CausalAttention.Config)
    attention.num_heads = 4

    finalized = config.copy_tree().finalize()
    table = finalized.bigrams["0"]

    assert table.channels_out == 4 * 8
    assert finalized.make()(torch.tensor([[1, 2, 3, 4]])).shape == (1, 4, VOCAB)


def test_memory_table_capacity_does_not_change_flops() -> None:
    small = _memory_config(buckets=16).make().flops_per_token()
    large = _memory_config(buckets=64).make().flops_per_token()

    assert large == small


def test_source_reuse_transformer_reads_attention_from_the_saved_source() -> None:
    config = SourceReuseTransformerBlock.Config()
    config.channels_in = 4
    config.attn = Linear.Config(4, 4)
    config.ffn = Identity.Config()
    config.norm1 = Identity.Config()
    config.norm2 = Identity.Config()
    block = config.make()
    assert isinstance(block.attn, Linear)
    with torch.no_grad():
        block.attn.weight.copy_(2 * torch.eye(4))
    current = torch.ones(1, 2, 4, requires_grad=True)
    source = torch.full((1, 2, 4), 3.0, requires_grad=True)
    output = block(current, attention_source=source)
    torch.testing.assert_close(output, torch.full_like(current, 14.0))
    output.sum().backward()
    torch.testing.assert_close(current.grad, torch.full_like(current, 2.0))
    torch.testing.assert_close(source.grad, torch.full_like(source, 4.0))


def test_gated_residual_starts_with_a_neutral_gate() -> None:
    config = GatedResidualMix.Config()
    config.num_layers = 1
    mix = config.make()
    current = torch.tensor([[[1.0, 3.0]]])
    original = torch.tensor([[[2.0, 4.0]]])
    torch.testing.assert_close(
        mix(current, original=original, layer=0),
        current + 0.1 * original,
    )


def test_output_norm_feed_forward_config_builds_the_specialized_class() -> None:
    config = OutputNormFeedForward.Config()
    config.channels_in = 4
    config.channels_out = 4
    config.round_to = 1
    assert isinstance(config.make(), OutputNormFeedForward)


def test_output_norm_feed_forward_cost_includes_output_norm() -> None:
    config = OutputNormFeedForward.Config()
    config.channels_in = 4
    config.channels_out = 4
    config.round_to = 1
    plain = config.copy_tree().finalize().cost(seq_len=8, batch_size=1, dtype=None)
    config.norm_out = RMSNorm.Config(elementwise_affine=True)
    config = config.finalize()
    assert config.cost(seq_len=8, batch_size=1, dtype=None) == plain + cost(
        config.norm_out,
        seq_len=8,
        batch_size=1,
        dtype=None,
    )


def test_gated_residual_cost_counts_mean_and_gate_parameters() -> None:
    config = GatedResidualMix.Config()
    config.num_layers = 2
    config.channels_in = 4
    priced = config.cost(seq_len=8, batch_size=1, dtype=torch.bfloat16)
    assert priced.params == 3 * 2
    assert priced["flops", "primal", "reduction"].sum() == 2 * (4 - 1)
    assert priced["bytes", "primal", "reduction"].sum() == 2 * 2 * (4 + 1)


def test_gated_residual_prices_sigmoid_as_one_tensor_operator() -> None:
    config = GatedResidualMix.Config()
    config.num_layers = 2
    config.channels_in = 4
    base = ResidualMix.Config()
    base.num_layers = 2
    base.channels_in = 4
    gated = config.cost(seq_len=8, batch_size=1, dtype=torch.bfloat16)
    plain = base.cost(seq_len=8, batch_size=1, dtype=torch.bfloat16)
    assert gated["bytes", "primal", "elementwise"].sum() - plain[
        "bytes",
        "primal",
        "elementwise",
    ].sum() == 2 * 2 * (11 + 1 / 8)
    assert gated["bytes", "adjoint", "elementwise"].sum() - plain[
        "bytes",
        "adjoint",
        "elementwise",
    ].sum() == 2 * 2 * (4 * 4 + 17 + 1 / 8)
    assert gated["flops", "adjoint", "elementwise"].sum() - plain[
        "flops",
        "adjoint",
        "elementwise",
    ].sum() == 2 * (8 + 2 * 4)


def test_memory_cost_counts_pool_parameters_and_extra_tables() -> None:
    config = MemoryNanoChatLM.Config()
    config.update(_config(), skip_missing=True)
    config.num_pool_layers = 2
    config.bigrams = {"0": HashedNgramTables.Config(num_embeddings=7)}
    config = config.finalize()
    base = NanoChatLM.Config().update(config, skip_missing=True).finalize()
    priced = config.cost(seq_len=SEQ, batch_size=4, dtype=None)
    plain = base.cost(seq_len=SEQ, batch_size=4, dtype=None)
    assert priced.params - plain.params == 7 * 16 + 1
    assert (
        priced["flops", "primal", "elementwise"].sum()
        - plain["flops", "primal", "elementwise"].sum()
        == 2 + 2 * 16
    )
    assert (
        priced["bytes", "primal", "selection"].sum()
        - plain["bytes", "primal", "selection"].sum()
        > 0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_pooling_gradient_temporary_is_written_only_by_width_reduction(
    dtype: torch.dtype,
    batch_size: int,
) -> None:
    config = MemoryNanoChatLM.Config()
    config.update(_config(), skip_missing=True)
    config.num_pool_layers = 2
    config = config.finalize()
    base = NanoChatLM.Config().update(config, skip_missing=True).finalize()
    pooled = config.cost(seq_len=SEQ, batch_size=batch_size, dtype=dtype)
    plain = base.cost(seq_len=SEQ, batch_size=batch_size, dtype=dtype)
    itemsize = dtype.itemsize
    width = config.channels_in
    rows = SEQ * batch_size
    assert pooled["bytes", "primal", "elementwise"].sum() - plain[
        "bytes",
        "primal",
        "elementwise",
    ].sum() == itemsize * (5 * width + 1 / rows)
    assert pooled["bytes", "adjoint", "elementwise"].sum() - plain[
        "bytes",
        "adjoint",
        "elementwise",
    ].sum() == itemsize * (6 * width + 1 / rows)
    assert pooled["bytes", "adjoint", "reduction"].sum() - plain[
        "bytes",
        "adjoint",
        "reduction",
    ].sum() == itemsize * (width + 1 + 1 + 1 / rows)
    assert (
        pooled["flops", "adjoint", "reduction"].sum()
        - plain["flops", "adjoint", "reduction"].sum()
        == width - 1 + (rows - 1) / rows
    )


def test_output_norm_feed_forward_reset_initializes_affine_output_norm() -> None:
    config = OutputNormFeedForward.Config()
    config.channels_in = 4
    config.channels_out = 4
    config.round_to = 1
    config.norm_out = RMSNorm.Config(elementwise_affine=True)
    ffn = config.make()
    norm_out = ffn.norm_out
    assert isinstance(norm_out, RMSNorm)
    assert norm_out.weight is not None
    with torch.no_grad():
        norm_out.weight.fill_(float("nan"))

    ffn.reset_parameters()

    assert torch.equal(norm_out.weight, torch.ones(4))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
