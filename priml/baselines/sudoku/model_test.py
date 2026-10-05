"""Tests for the sudoku model and its recurrence."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, cast, runtime_checkable

import math
import re

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

import pytest
import torch

from priml.baselines.sudoku import model
from priml.baselines.sudoku.embedding import GridEmbedding, PredictionFeedback
from priml.baselines.sudoku.model import (
    CoreCompile,
    CoreOutput,
    DeepRecurrence,
    ForwardOutput,
    SudokuNet,
    _latent_init,
    lattice_positions,
)
from priml.baselines.sudoku.prefix import RegisterTokens
from priml.cost import Cost, cost
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaNaive
from priml.model.attention.rope import RoPE
from priml.model.init import kaiming_uniform, truncated_normal, unit_normal
from priml.model.mlpmixer import MLPMixerBlock
from priml.model.norm import RMSNorm
from priml.model.special import Identity
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.testing.bfb import assert_bfb_against_golden
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.sudoku.model import GridConfig
    from priml.baselines.sudoku.prefix import PrefixConfig


_CWD: Final = Path(__file__).resolve().parent


def test_unfilled_vocabulary_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=r"^SudokuNet requires vocab_size from the dataset\.$",
    ):
        SudokuNet.Config(channels_in=4, num_layers=1).make()


def test_vocab_size_one_is_valid_and_zero_is_rejected() -> None:
    assert isinstance(_config(vocab_size=1).make(), SudokuNet)
    config = _config(vocab_size=0)
    with pytest.raises(
        ValueError,
        match=r"^SudokuNet requires vocab_size from the dataset\.$",
    ):
        config.make()


def test_rope_requires_grid_shape_with_its_contract_message() -> None:
    config = _config()
    config.rope = RoPE.Config(channels_head=2)
    with pytest.raises(
        ValueError,
        match=r"^SudokuNet requires rope_grid_shape from the dataset\.$",
    ):
        config.make()


def test_empty_grid_is_rejected_with_its_contract_message() -> None:
    class EmptyGrid:
        channels_in = -1
        channels_out = -1
        grid_len = 0

        def make(self) -> GridEmbedding:
            return cast(GridEmbedding, nn.Identity())

    config = SudokuNet.Config(channels_in=4, vocab_size=2, num_layers=1)
    config.embedding = cast("GridConfig", EmptyGrid())
    with pytest.raises(
        ValueError,
        match=r"^SudokuNet requires embedding grid_shape from the dataset\.$",
    ):
        SudokuNet(config)


def test_one_cell_grid_is_valid() -> None:
    _config(grid_shape=(1,)).make()


def _config(
    *,
    recurrent: bool = False,
    mixer: bool = False,
    channels_in: int = 4,
    vocab_size: int = 2,
    grid_shape: tuple[int, ...] = (2,),
) -> SudokuNet.Config:
    config = SudokuNet.Config(
        channels_in=channels_in,
        vocab_size=vocab_size,
        num_layers=1,
    )
    config.embedding = GridEmbedding.Config(grid_shape=grid_shape)
    config.block = TransformerBlock.Config(
        prenorm=False,
        attn=Attention.Config(num_heads=2, channels_head=2),
        ffn=SwiGLU.Config(
            channels_hidden=4,
            round_to=1,
            init_weight=kaiming_uniform,
            init_weight_out=kaiming_uniform,
        ),
    )
    if mixer:
        config.block = MLPMixerBlock.Config(
            seq_len=math.prod(grid_shape),
            prenorm=False,
            token_mixer=SwiGLU.Config(
                norm=RMSNorm.Config(),
                channels_hidden=4,
                round_to=1,
                init_weight=kaiming_uniform,
                init_weight_out=kaiming_uniform,
            ),
            channel_mixer=SwiGLU.Config(
                norm=RMSNorm.Config(),
                channels_hidden=4,
                round_to=1,
                init_weight=kaiming_uniform,
                init_weight_out=kaiming_uniform,
            ),
        )
    if recurrent:
        config.recurrence = DeepRecurrence.Config(slow_cycles=2, fast_cycles=2)
    return config


def _model(
    *,
    recurrent: bool = False,
    mixer: bool = False,
    channels_in: int = 4,
    vocab_size: int = 2,
    grid_shape: tuple[int, ...] = (2,),
) -> SudokuNet:
    torch.manual_seed(0)
    return _config(
        recurrent=recurrent,
        mixer=mixer,
        channels_in=channels_in,
        vocab_size=vocab_size,
        grid_shape=grid_shape,
    ).make()


@pytest.mark.parametrize("mixer", [False, True])
@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.compute_large_fixture
def test_every_corner_of_the_lattice_runs(mixer: bool, recurrent: bool) -> None:
    """Architecture and recurrence vary independently, as config values."""
    model = _model(
        mixer=mixer,
        recurrent=recurrent,
        channels_in=16,
        vocab_size=11,
        grid_shape=(81,),
    )
    out = model(torch.randint(0, 11, (2, 81)))
    assert out.logits.shape == (2, 81, 11)
    assert out.halt.shape == (2,)


def test_recurrence_adds_no_parameters() -> None:
    """The recurrence schedules the core; it does not own weights.

    If it did, the plain-vs-recurrent comparison would confound depth with
    capacity and neither result would mean what it claims.
    """
    plain = sum(p.numel() for p in _model().parameters())
    recurrent = sum(p.numel() for p in _model(recurrent=True).parameters())
    assert plain == recurrent


@pytest.mark.compute_training
def test_backward_graph_is_flat_in_recurrence_depth() -> None:
    """Gradient cost must not grow with cycle count.

    That is the whole reason deep recurrence is affordable: all but the last
    cycle run under ``no_grad``, so a 32-cycle forward backpropagates through
    one cycle. A graph that grew with depth would mean the truncation broke.
    """
    assert _graph_size(1) == _graph_size(8) == _graph_size(32)


def test_prenorm_diverges_under_recurrence() -> None:
    """Post-norm is a requirement of feeding a block its own output.

    Pre-norm leaves the residual stream unnormalized, which is harmless in one
    pass and compounds when the output is fed back. This pins the reason the
    default is post-norm, so a future change to priml's default cannot silently
    reintroduce the divergence.
    """
    postnorm = _carried_magnitude(prenorm=False)
    prenorm = _carried_magnitude(prenorm=True)
    assert postnorm < prenorm / 10, f"post-norm {postnorm}; pre-norm {prenorm}"


def test_latents_carry_between_calls() -> None:
    """A second call from the first call's latents differs from a fresh one."""
    model = _model(recurrent=True, channels_in=16, vocab_size=11, grid_shape=(81,))
    tokens = torch.randint(0, 11, (2, 81))
    first = model(tokens)
    carried = model(tokens, first.z_slow, first.z_fast)
    fresh = model(tokens)
    assert not torch.equal(carried.logits, fresh.logits)


def test_intermediates_are_one_per_cycle() -> None:
    model = _model(recurrent=True, channels_in=16, vocab_size=11, grid_shape=(81,))
    out = model(torch.randint(0, 11, (2, 81)), collect_intermediates=True)
    assert len(out.all_logits) == 2  # slow_cycles.


def test_sequence_length_counts_the_prefix_before_finalize() -> None:
    """A parent reading ``total_seq_len`` must not see the sentinel.

    A parent's ``finalize`` runs BEFORE this config's, so a naive
    ``num_prefix_tokens + grid_len`` returns one short by the whole prefix
    while the sentinel is still set. Anything sized from it -- an ACT pool's
    latent buffers -- is then built to the wrong shape and fails only later,
    deep in a matmul. Measured: 80 against the true 81.
    """
    config = _config(channels_in=16, vocab_size=11, grid_shape=(81,))
    registers = RegisterTokens.Config(num_tokens=4)
    config.prefix = registers
    assert config.num_prefix_tokens == -1  # Not yet finalized.
    assert config.total_seq_len == 81 + 4
    # After finalize the count is materialized and agrees.
    final = config.copy_tree().finalize()
    assert final.num_prefix_tokens == 4
    assert final.total_seq_len == 81 + 4


def test_embed_casts_prefix_to_embedding_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    config.prefix = RegisterTokens.Config(num_tokens=2)
    network = config.make()

    def double_prefix(batch_size: int) -> Tensor:
        return torch.ones(batch_size, 2, 4, dtype=torch.float64)

    assert network.prefix is not None
    monkeypatch.setattr(network.prefix, "forward", double_prefix)
    embedded = network._embed(torch.zeros(3, 6, dtype=torch.long), {})
    assert (
        embedded.dtype is network.embedding(torch.zeros(3, 6, dtype=torch.long)).dtype
    )
    assert torch.equal(embedded[:, :2], torch.ones(3, 2, 4))


def test_prefix_tokens_reach_the_sequence() -> None:
    """Prefix logits are stripped, so the output is still one row per cell."""
    config = _config(channels_in=16, vocab_size=11, grid_shape=(81,))
    config.prefix = RegisterTokens.Config(num_tokens=3)
    torch.manual_seed(0)
    out = config.make()(torch.randint(0, 11, (2, 81)))
    assert out.logits.shape == (2, 81, 11)


def test_prefix_parameters_lead_the_parameter_order() -> None:
    """Prefix parameters come first, where the reference TRM registered its prefix.

    A body norm sums parameters in registration order, and the compiled float32
    reduction behind it lands a different last bit when that order changes:
    measured 1 ULP on ``param_norm`` against the reference step.
    """
    config = _config()
    config.prefix = RegisterTokens.Config(num_tokens=2)
    names = [name for name, _ in config.make().named_parameters()]
    assert names[0].startswith("prefix."), names


def test_a_prefix_without_a_token_count_is_rejected() -> None:
    """Guessing 0 would silently shift every grid position."""
    config = _config()
    config.prefix = cast("PrefixConfig", RMSNorm.Config(channels_in=16))
    with pytest.raises(AttributeError, match="num_tokens"):
        config.copy_tree().finalize()


def test_an_embedding_without_a_grid_is_rejected() -> None:
    """The model reads its grid from the embedding; one without a grid has none."""
    config = _config()
    config.embedding = cast("GridConfig", RMSNorm.Config(channels_in=16))
    with pytest.raises(AttributeError, match="grid_len"):
        _ = config.grid_len


def test_a_rope_lattice_must_cover_the_grid() -> None:
    """A lattice smaller than the grid would mis-place every grid position."""
    config = _config(grid_shape=(2, 3))
    config.rope = RoPE.Config(channels_head=[2])
    config.rope_grid_shape = (2, 2)
    with pytest.raises(ValueError, match="rope_grid_shape"):
        config.make()
    config.rope_grid_shape = (3, 2)
    assert config.make().rope is not None


@pytest.mark.parametrize(
    ("slow_cycles", "fast_cycles"),
    [(0, 1), (1, 0)],
)
def test_cycle_counts_must_be_positive(slow_cycles: int, fast_cycles: int) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        DeepRecurrence.Config(
            slow_cycles=slow_cycles,
            fast_cycles=fast_cycles,
        ).make()


def test_one_fast_cycle_is_valid() -> None:
    assert isinstance(DeepRecurrence.Config(fast_cycles=1).make(), DeepRecurrence)


def test_one_slow_cycle_is_valid() -> None:
    assert isinstance(DeepRecurrence.Config(slow_cycles=1).make(), DeepRecurrence)


def _tiny_tokens() -> Tensor:
    return torch.randint(0, 2, (3, 2))


def test_plain_forward_bfb() -> None:
    """Freeze exp000's architecture: same weights in, same logits out.

    Guards the whole forward path -- embedding composition, block stack,
    output head -- against a refactor that changes arithmetic while keeping
    every shape and name intact.
    """
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="plain",
        build_module=_config().make,
        build_input=_tiny_tokens,
        seed=0,
        run=_logits,
    )


def test_recurrent_forward_bfb() -> None:
    """Freeze the recurrence: the cycle schedule is part of the arithmetic.

    Separate from the plain golden because the two differ only in a slot's
    value, so a change to the recurrence would leave the plain golden green.
    """
    assert_bfb_against_golden(
        golden_dir=_CWD / "testdata",
        golden_name="recurrent",
        build_module=lambda: _config(recurrent=True).make(),
        build_input=_tiny_tokens,
        seed=0,
        run=_logits,
    )


def test_deep_recurrence_cost_is_zero() -> None:
    """The recurrence schedules the core and owns nothing; the model costs its cycles."""
    analytical = assert_cost_matches_torch(
        DeepRecurrence.Config(slow_cycles=2, fast_cycles=2),
        build_input=lambda: torch.randn(2, 3, 4, requires_grad=True),
        seq_len=2,
        batch_size=1,
        dtype=None,
        run=_run_identity_core,
    )
    assert analytical == Cost()


@pytest.mark.parametrize("prefix", [False, True])
def test_plain_cost_matches_torch(prefix: bool) -> None:
    """One core application per forward; a prefix widens the latent sequence.

    A two-cell grid with a two-token prefix exercises complete-batch totals,
    so the analytical count matches torch's exactly.
    """
    config = _cost_config(prefix=prefix)
    assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 11, (3, 2)),
        batch_size=3,
        dtype=None,
        run=_logits_and_halt,
    )


def test_recurrent_cost_matches_torch() -> None:
    """Every slow cycle runs forward; only the last runs backward."""
    config = _cost_config(prefix=True)
    config.recurrence = DeepRecurrence.Config(slow_cycles=2, fast_cycles=2)
    analytical = assert_cost_matches_torch(
        config,
        build_input=lambda: torch.randint(0, 11, (3, 2)),
        batch_size=3,
        dtype=None,
        run=_logits_and_halt,
    )
    one_cycle = _cost_config(prefix=True)
    one_cycle.recurrence = DeepRecurrence.Config(slow_cycles=1, fast_cycles=2)
    single = cost(one_cycle.copy_tree().finalize(), batch_size=3, dtype=None)
    assert analytical["flops", "adjoint"] == single["flops", "adjoint"]
    assert analytical["bytes", "adjoint"] == single["bytes", "adjoint"]
    assert analytical.params == single.params
    assert (
        analytical["flops", "primal", "matmul"].sum()
        > single["flops", "primal", "matmul"].sum()
    )


def test_cost_uses_puzzles_for_halt_and_register_gradient_reductions() -> None:
    config = _cost_config(prefix=True)
    config.block = Identity.Config()
    config = config.finalize()
    two = cost(config, batch_size=2, dtype=None)
    four = cost(config, batch_size=4, dtype=None)
    # Each puzzle owns one halt row and two register rows, not four latent rows.
    assert two["flops", "adjoint", "reduction"].sum() == (2 + 2 * 16) * (2 - 1)
    assert four["flops", "adjoint", "reduction"].sum() == (2 + 2 * 16) * (4 - 1)
    assert (
        four["bytes", "primal", "matmul"].sum() > two["bytes", "primal", "matmul"].sum()
    )
    assert four["flops", "primal"].sum() == 2 * two["flops", "primal"].sum()


def test_cost_counts_grid_prefix_concatenation() -> None:
    plain = _cost_config(prefix=False).finalize().cost(batch_size=2, dtype=None)
    prefix = _cost_config(prefix=True).finalize().cost(batch_size=2, dtype=None)
    assert (
        prefix["bytes", "primal", "selection"].sum()
        - plain["bytes", "primal", "selection"].sum()
        == 4 * 2 * 2 * (2 + 2) * 16
    )


def test_cost_halt_bias_reduction_uses_puzzle_batch_geometry() -> None:
    config = _cost_config(prefix=False)
    config.block = Identity.Config()
    config = config.finalize()
    costed = cost(config, batch_size=4, dtype=None)
    assert costed["flops", "adjoint", "reduction"].sum() == 2 * (4 - 1)


def _cost_config(*, prefix: bool) -> SudokuNet.Config:
    """Build a two-cell solver whose attention torch can count."""
    config = SudokuNet.Config(channels_in=16, num_layers=1, vocab_size=11)
    config.embedding = GridEmbedding.Config(grid_shape=(2,))
    config.block = TransformerBlock.Config(
        prenorm=False,
        attn=Attention.Config(
            num_heads=2,
            channels_head=8,
            attn_kernel=SdpaNaive.Config(),
        ),
        ffn=SwiGLU.Config(channels_hidden=32, round_to=1),
    )
    if prefix:
        config.prefix = RegisterTokens.Config(num_tokens=2)
    return config


def _logits_and_halt(module: nn.Module, tokens: Tensor) -> Tensor:
    """Reduce both heads, so the halt head's backward is counted too."""
    out = cast(object, module(tokens))
    assert isinstance(out, ForwardOutput)
    return out.logits.sum() + out.halt.sum()


def _run_identity_core(module: nn.Module, x: Tensor) -> Tensor:
    """Drive the recurrence with a core that does no arithmetic."""
    out = cast(object, module(_identity_core, x, x, x, None))
    assert isinstance(out, ForwardOutput)
    return out.logits


def _identity_core(
    input_emb: Tensor,
    z_slow: Tensor,
    z_fast: Tensor,
    cos_sin: tuple[Tensor, Tensor] | None = None,
) -> CoreOutput:
    del cos_sin
    return CoreOutput(input_emb, input_emb[:, 0, 0], z_slow, z_fast)


def _logits(module: nn.Module, tokens: object) -> Tensor:
    """Run the model and return the logits the golden compares."""
    runner = cast(_CallableModule, module)
    raw = runner(tokens)
    assert isinstance(raw, ForwardOutput)
    return raw.logits


@runtime_checkable
class _CallableModule(Protocol):
    def __call__(self, tokens: object) -> object: ...


@runtime_checkable
class _AutogradNode(Protocol):
    next_functions: tuple[tuple[_AutogradNode | None, int], ...]


def _graph_size(slow_cycles: int) -> int:
    """Count the autograd nodes reachable from one forward's loss."""
    config = _config(
        recurrent=True,
        channels_in=16,
        vocab_size=11,
        grid_shape=(81,),
    )
    assert isinstance(config.recurrence, DeepRecurrence.Config)
    config.recurrence.slow_cycles = slow_cycles
    torch.manual_seed(0)
    out = config.make()(torch.randint(0, 11, (2, 81)))
    loss = out.logits.square().mean()
    # Hold a reference to every node: ``next_functions`` yields fresh wrappers,
    # so a set of ids alone undercounts once one is collected.
    keep: list[_AutogradNode] = []
    seen: set[_AutogradNode] = set()
    initial = loss.grad_fn
    assert isinstance(initial, _AutogradNode)
    stack: list[_AutogradNode] = [initial]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        keep.append(node)
        # Autograd nodes are untyped at the Python boundary.
        stack.extend(nxt for nxt, _ in node.next_functions if nxt is not None)
    return len(keep)


def _carried_magnitude(*, prenorm: bool) -> float:
    """Largest carried-latent value after three recurrent steps."""
    config = _config(
        recurrent=True,
        channels_in=16,
        vocab_size=11,
        grid_shape=(81,),
    )
    assert isinstance(config.recurrence, DeepRecurrence.Config)
    config.recurrence.slow_cycles = 4
    config.block = TransformerBlock.Config(prenorm=prenorm)
    torch.manual_seed(0)
    model = config.make()
    tokens = torch.randint(2, 11, (2, 81))
    z_slow, z_fast = model.init_latents(2)
    for _ in range(3):
        out = model(tokens, z_slow, z_fast)
        z_slow, z_fast = out.z_slow, out.z_fast
    return float(z_slow.abs().max())


@pytest.mark.parametrize(
    ("checkpoint_core", "fraction"),
    [(True, 0.0), (False, 0.5), (True, 1.0)],
)
def test_activation_checkpointing_is_bit_identical_and_recomputes(
    *,
    checkpoint_core: bool,
    fraction: float,
) -> None:
    """Checkpointing changes only what backward keeps, never the numbers."""
    plain_out, plain_grads, plain_saved = _checkpoint_run(
        checkpoint_core=False,
        fraction=0.0,
    )
    out, grads, saved = _checkpoint_run(
        checkpoint_core=checkpoint_core,
        fraction=fraction,
    )
    assert torch.equal(out, plain_out)
    assert grads.keys() == plain_grads.keys()
    for name, grad in grads.items():
        assert torch.equal(grad, plain_grads[name]), name
    # Recomputation is the mechanism: fewer activations are held for backward.
    assert saved < plain_saved


def test_zero_block_checkpoint_fraction_skips_ceil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(value: float) -> int:
        del value
        pytest.fail("zero checkpoint fraction must not calculate a count")

    monkeypatch.setattr(math, "ceil", fail)
    assert len(model._checkpointed_blocks(_config())) == 1


def test_block_checkpoint_fraction_marks_an_even_subset() -> None:
    config = _config(recurrent=True, channels_in=4)
    config.num_layers = 4
    config.block_checkpoint_fraction = 0.5
    torch.manual_seed(0)
    model = config.make()
    flags = [
        block.checkpoint
        for block in model.reasoning
        if isinstance(block, TransformerBlock)
    ]
    assert flags == [False, True, False, True]
    config.block_checkpoint_fraction = 1.5
    message = "block_checkpoint_fraction must be in [0, 1], got 1.5."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        config.make()
    config.block_checkpoint_fraction = 0.5
    config.block = MLPMixerBlock.Config(seq_len=2)
    with pytest.raises(
        ValueError,
        match=(
            r"^block_checkpoint_fraction requires a block config with a "
            r"checkpoint field\.$"
        ),
    ):
        config.make()


def test_checkpointed_block_copies_are_deep_and_independent() -> None:
    config = _config()
    config.num_layers = 2
    blocks = model._checkpointed_blocks(config)

    assert all(isinstance(block, TransformerBlock.Config) for block in blocks)
    first, second = cast(tuple[TransformerBlock.Config, ...], tuple(blocks))
    assert first is not second
    assert first.attn is not second.attn
    assert first.ffn is not second.ffn
    assert isinstance(config.block, TransformerBlock.Config)
    assert first.attn is not config.block.attn
    assert first.ffn is not config.block.ffn


def test_checkpoint_core_forwards_rotary_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    config.checkpoint_core = True
    config.rope = RoPE.Config(channels_head=2)
    config.rope_grid_shape = (6,)
    network = config.make()
    original_checkpoint = checkpoint
    received: list[tuple[object, ...]] = []

    def tracking_checkpoint(
        fn: Callable[..., CoreOutput],
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
        *,
        use_reentrant: bool,
    ) -> CoreOutput:
        received.append((input_emb, z_slow, z_fast, cos_sin))
        return original_checkpoint(
            fn,
            input_emb,
            z_slow,
            z_fast,
            cos_sin,
            use_reentrant=use_reentrant,
        )

    monkeypatch.setattr(model, "torch_checkpoint", tracking_checkpoint)
    embedded = network.embedding(torch.tensor([[0, 1, 2, 3, 4, 5]]))
    z_slow, z_fast = network.init_latents(1)
    cos_sin = network._cos_sin(embedded)
    assert cos_sin is not None
    output = network.core(embedded, z_slow, z_fast, cos_sin)
    output.logits.sum().backward()
    assert len(received) == 1
    assert len(received[0]) == 4
    assert received[0][3] is cos_sin
    assert any(parameter.grad is not None for parameter in network.parameters())


def test_checkpoint_core_recomputes_only_the_grad_bearing_cycle() -> None:
    """Backward re-runs the wrapped cycle once; ``no_grad`` cycles never rerun."""
    config = _config(recurrent=True, channels_in=4)
    assert isinstance(config.recurrence, DeepRecurrence.Config)
    config.recurrence.slow_cycles = 3
    config.checkpoint_core = True
    torch.manual_seed(0)
    model = config.make()
    calls: list[int] = []

    def count(module: nn.Module, args: object, output: object) -> None:
        del module, args, output
        calls.append(1)

    model.head.register_forward_hook(count)
    tokens = torch.randint(0, 2, (3, 2))
    model(tokens).logits.sum().backward()
    # Three forward core applications, plus one recompute of the last.
    assert len(calls) == 3 + 1
    calls.clear()
    with torch.inference_mode():
        model(tokens)
    assert len(calls) == 3


def test_checkpoint_core_never_enters_checkpoint_without_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrapping a compiled core under inference_mode deadlocks multi-rank eval."""
    entered: list[bool] = []

    def refuse(*args: object, **kwargs: object) -> object:
        del args, kwargs
        entered.append(torch.is_grad_enabled())
        raise AssertionError("checkpoint entered")

    monkeypatch.setattr(model, "torch_checkpoint", refuse)
    config = _config(recurrent=True, channels_in=4)
    config.checkpoint_core = True
    torch.manual_seed(0)
    network = config.make()
    with torch.inference_mode():
        network(torch.randint(0, 2, (3, 2)))
    assert entered == []
    with pytest.raises(AssertionError, match="checkpoint entered"):
        network(torch.randint(0, 2, (3, 2)))
    assert entered == [True]


def _checkpoint_run(
    *,
    checkpoint_core: bool,
    fraction: float,
) -> tuple[Tensor, dict[str, Tensor], int]:
    """Forward + backward one recurrent model; count saved activation bytes."""
    config = _config(recurrent=True, channels_in=8, vocab_size=5, grid_shape=(6,))
    config.num_layers = 2
    config.checkpoint_core = checkpoint_core
    config.block_checkpoint_fraction = fraction
    torch.manual_seed(0)
    model = config.make()
    tokens = torch.randint(0, 5, (3, 6), generator=torch.Generator().manual_seed(1))
    saved: list[int] = []

    def pack(tensor: Tensor) -> Tensor:
        saved.append(tensor.numel() * tensor.element_size())
        return tensor

    def unpack(tensor: Tensor) -> Tensor:
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        out = model(tokens)
        loss = out.logits.square().mean() + out.halt.square().mean()
    loss.backward()
    grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return out.logits.detach(), grads, sum(saved)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_latent_init_preserves_dtype_and_variance(dtype: torch.dtype) -> None:
    # Latent initialization has a fixed singleton sample axis.
    expected = torch.empty(1, 257, dtype=dtype)
    torch.manual_seed(0)
    truncated_normal(
        expected,
        std=1.0,
        depth_index=(),
        variance_correction=True,
    )
    config = SudokuNet.Config()
    config.channels_in = 257
    config.dtype_latent_init = dtype
    torch.manual_seed(0)
    actual = _latent_init(config)
    assert torch.equal(actual, expected)


def test_lattice_positions_include_prefix_offsets_and_device() -> None:
    positions = lattice_positions(8, grid_shape=(2, 3), device=torch.device("cpu"))
    assert torch.equal(
        positions,
        torch.tensor(
            [
                [0, 0],
                [0, 1],
                [2, 0],
                [2, 1],
                [2, 2],
                [3, 0],
                [3, 1],
                [3, 2],
            ],
        ),
    )
    three_axes = lattice_positions(
        26,
        grid_shape=(2, 3, 4),
        device=torch.device("cpu"),
    )
    assert torch.equal(three_axes[:2], torch.tensor([[0, 0, 0], [0, 0, 1]]))
    assert (
        lattice_positions(
            8,
            grid_shape=(2, 3),
            device=torch.device("meta"),
        ).device.type
        == "meta"
    )


def test_initialization_and_lattice_degenerate_branch() -> None:
    positions = lattice_positions(6, grid_shape=(6,), device=torch.device("cpu"))
    assert torch.equal(positions, torch.arange(6))


def test_two_latent_refine_forwards_rotary_factors_to_every_mix() -> None:
    factors = (torch.ones(2), torch.zeros(2))
    received: list[tuple[Tensor, Tensor] | None] = []

    def mix(
        z: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
    ) -> Tensor:
        received.append(cos_sin)
        return z + 1

    x = torch.zeros(2, 3, 4)
    model.two_latent_refine(mix, x, x, x, factors, fast_cycles=2)
    assert len(received) == 3
    assert all(value is factors for value in received)


def test_unit_normal_matches_torch_truncated_normal() -> None:
    expected = torch.empty(2, 3, 4)
    torch.manual_seed(0)
    torch.nn.init.trunc_normal_(expected, std=1.0)
    actual = torch.empty_like(expected)
    torch.manual_seed(0)
    unit_normal(actual)
    assert torch.equal(actual, expected)


def test_lattice_position_factories_receive_exact_dtype_and_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arange = cast("Callable[..., Tensor]", torch.arange)
    zeros = cast("Callable[..., Tensor]", torch.zeros)
    arange_kwargs: list[dict[str, object]] = []
    zeros_kwargs: list[dict[str, object]] = []

    def recording_arange(*args: object, **kwargs: object) -> Tensor:
        arange_kwargs.append(kwargs)
        return arange(*args, **kwargs)

    def recording_zeros(*args: object, **kwargs: object) -> Tensor:
        zeros_kwargs.append(kwargs)
        return zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "arange", recording_arange)
    monkeypatch.setattr(torch, "zeros", recording_zeros)
    device = torch.device("meta")
    positions = lattice_positions(8, grid_shape=(2, 3), device=device)
    assert positions.device == device
    assert positions.dtype is torch.long
    assert arange_kwargs == [
        {"device": device},
        {"device": device},
        {"device": device},
    ]
    assert zeros_kwargs == [{"dtype": torch.long, "device": device}]
    one_axis = lattice_positions(6, grid_shape=(6,), device=device)
    assert one_axis.device == device
    assert arange_kwargs[-1] == {"device": device}


def test_model_initial_latents_have_batch_sequence_and_channel_axes() -> None:
    model = _model(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    z_slow, z_fast = model.init_latents(3)
    assert z_slow.shape == (3, 6, 4)
    assert z_fast.shape == (3, 6, 4)
    assert torch.equal(z_slow[0], z_slow[2])
    assert torch.equal(z_fast[0], z_fast[2])


def test_cos_sin_uses_the_first_rotary_factors_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = _model(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    first = network.reasoning[0]
    factors = (torch.ones(3, 2), torch.zeros(3, 2))
    calls: list[tuple[int, torch.device]] = []

    def build_factors(
        seq_len: int,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        calls.append((seq_len, device))
        return factors

    monkeypatch.setattr(first, "factors", build_factors, raising=False)
    input_emb = torch.empty(2, 3, 4)
    assert network._cos_sin(input_emb) is factors
    assert calls == [(3, input_emb.device)]


def test_model_buffers_are_persistent_and_halt_is_float32() -> None:
    config = _config(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    config.dtype = torch.bfloat16
    network = config.make()
    assert {"slow_init", "fast_init", "_dummy"} <= network.state_dict().keys()
    assert isinstance(network._dummy, Tensor)
    assert network._dummy.shape == (0,)
    assert network.device == torch.device("cpu")
    assert all(parameter.dtype is torch.bfloat16 for parameter in network.parameters())
    output = network(torch.randint(0, 11, (2, 6)))
    assert output.halt.dtype is torch.float32


def test_dummy_buffer_is_a_persistent_empty_tensor() -> None:
    network = _config().make()
    assert isinstance(network._dummy, Tensor)
    assert network._dummy.shape == (0,)
    assert network.device == torch.device("cpu")
    assert "_dummy" in network.state_dict()


def test_model_initial_buffers_are_explicitly_persistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_buffer = nn.Buffer
    persistent_flags: list[bool] = []

    def track_buffer(tensor: Tensor, *, persistent: bool) -> Tensor:
        persistent_flags.append(persistent)
        return original_buffer(tensor, persistent=persistent)

    monkeypatch.setattr(nn, "Buffer", track_buffer)
    config = _config()
    config.block = Identity.Config()
    config.make()
    assert persistent_flags == [True, True, True]


@pytest.mark.parametrize("halt_outputs", [1, 2])
def test_core_returns_the_first_halt_output_with_batch_axis(
    halt_outputs: int,
) -> None:
    config = _config(channels_in=4, vocab_size=11)
    config.block = Identity.Config()
    config.halt_outputs = halt_outputs
    network = config.make()
    weight = network.halt_head.weight
    bias = network.halt_head.bias
    assert weight is not None
    assert bias is not None
    with torch.no_grad():
        weight.copy_(torch.arange(1, 4 * halt_outputs + 1).reshape(halt_outputs, 4))
        bias.copy_(torch.arange(halt_outputs) + 5)

    input_emb = torch.zeros(2, 3, 4)
    z_slow = torch.stack(
        [torch.ones(2, 4), torch.full((2, 4), 3.0), torch.full((2, 4), 7.0)],
        dim=1,
    )
    z_fast = torch.zeros_like(z_slow)
    output = network._core(input_emb, z_slow, z_fast)
    expected = network.halt_head(output.z_slow[:, 0]).to(torch.float32)[..., 0]
    assert output.halt.shape == (2,)
    assert torch.equal(output.halt, expected)


def test_core_forwards_rotary_factors_to_each_mix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = _model(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    factors = (torch.ones(6, 2), torch.zeros(6, 2))
    received: list[tuple[Tensor, Tensor] | None] = []

    def mix(
        z: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
    ) -> Tensor:
        received.append(cos_sin)
        return z

    monkeypatch.setattr(network, "_mix", mix)
    state = torch.zeros(2, 6, 4)
    network._core(state, state, state, factors)
    assert len(received) == 2
    assert all(value is factors for value in received)


def test_compile_unit_selects_the_requested_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled: list[object] = []

    def compile_eager(
        fn: Callable[..., object],
        **kwargs: object,
    ) -> Callable[..., object]:
        del kwargs
        compiled.append(fn)
        return fn

    monkeypatch.setattr(torch, "compile", compile_eager)
    core_config = _config(channels_in=4, vocab_size=11)
    core_config.compile_core = CoreCompile.Config(unit="core")
    core_network = core_config.make()
    assert core_network._compiled_core is not None
    assert core_network._compiled_mix is None

    reasoning_config = _config(channels_in=4, vocab_size=11)
    reasoning_config.compile_core = CoreCompile.Config(unit="reasoning")
    reasoning_network = reasoning_config.make()
    assert reasoning_network._compiled_core is None
    assert reasoning_network._compiled_mix is not None
    assert len(compiled) == 2


def test_core_uses_the_compiled_callable(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []

    def compile_eager(
        fn: Callable[..., object],
        **kwargs: object,
    ) -> Callable[..., object]:
        del kwargs

        def compiled(*args: object, **call_kwargs: object) -> object:
            calls.append(args)
            return fn(*args, **call_kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", compile_eager)
    config = _config(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    config.compile_core = CoreCompile.Config(unit="core")
    network = config.make()
    assert network._compiled_core is not None
    embedded = network.embedding(torch.zeros(2, 6, dtype=torch.long))
    z_slow, z_fast = network.init_latents(2)
    network.core(embedded, z_slow, z_fast)
    assert len(calls) == 1


def test_cos_sin_builds_rope_positions_on_the_input_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    config.rope = RoPE.Config(channels_head=2)
    config.rope_grid_shape = (2, 3)
    network = config.make()
    factors = (torch.ones(2), torch.zeros(2))
    received: list[tuple[int, tuple[int, ...], torch.device | None]] = []

    def positions(
        seq_len: int,
        *,
        grid_shape: tuple[int, ...],
        device: torch.device | None,
    ) -> Tensor:
        received.append((seq_len, grid_shape, device))
        return torch.zeros(seq_len, len(grid_shape), dtype=torch.long, device=device)

    def rotary(positions: Tensor) -> tuple[Tensor, Tensor]:
        del positions
        return factors

    monkeypatch.setattr(model, "lattice_positions", positions)
    assert network.rope is not None
    monkeypatch.setattr(network.rope, "forward", rotary)
    input_emb = torch.zeros(2, 8, 4)
    actual = network._cos_sin(input_emb)
    assert actual is not None
    assert actual[0] is factors[0]
    assert actual[1] is factors[1]
    assert received == [(8, (2, 3), input_emb.device)]


def test_embed_forwards_prefix_keyword_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    config.prefix = RegisterTokens.Config(num_tokens=3)
    network = config.make()
    received: list[int] = []

    def prefix(batch_size: int, *, marker: int) -> Tensor:
        received.append(marker)
        return torch.full((batch_size, 3, 4), marker, dtype=torch.float32)

    assert network.prefix is not None
    monkeypatch.setattr(network.prefix, "forward", prefix)
    embedded = network._embed(torch.zeros(2, 6, dtype=torch.long), {"marker": 7})
    assert received == [7]
    assert torch.equal(embedded[:, :3], torch.full((2, 3, 4), 7.0))


def test_compiled_mix_forwards_rotary_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[Tensor, tuple[Tensor, Tensor] | None]] = []

    def compile_eager(
        fn: Callable[..., object],
        **kwargs: object,
    ) -> Callable[..., object]:
        del kwargs

        def compiled(z: Tensor, cos_sin: tuple[Tensor, Tensor] | None) -> Tensor:
            received.append((z, cos_sin))
            return cast(Tensor, fn(z, cos_sin=cos_sin))

        return compiled

    monkeypatch.setattr(torch, "compile", compile_eager)
    config = _config(channels_in=4, vocab_size=11)
    config.compile_core = CoreCompile.Config(unit="reasoning")
    network = config.make()
    # Attention's RoPE factors are [sequence=2, heads=2, half-width=1].
    factors = (torch.ones(2, 2, 1), torch.zeros(2, 2, 1))
    z = torch.zeros(3, 2, 4)
    network._mix(z, factors)
    assert len(received) == 1
    assert received[0][0] is z
    assert received[0][1] is factors


def test_strip_prefix_strips_cycle_logits_and_preserves_unprefixed_tensors() -> None:
    config = _config(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    config.prefix = RegisterTokens.Config(num_tokens=2)
    network = config.make()
    logits = torch.randn(2, 8, 11)
    cycle_logits = torch.randn(2, 8, 11)
    latents = torch.randn(2, 8, 4, requires_grad=True)
    prefixed = network._strip_prefix(
        ForwardOutput(logits, torch.zeros(2), latents, latents, (cycle_logits,)),
    )
    assert torch.equal(prefixed.logits, logits[:, 2:])
    assert torch.equal(prefixed.all_logits[0], cycle_logits[:, 2:])
    assert not prefixed.z_slow.requires_grad
    assert not prefixed.z_fast.requires_grad

    config = _config(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    network = config.make()
    plain_logits = logits[:, :6]
    plain_cycle_logits = cycle_logits[:, :6]
    plain = network._strip_prefix(
        ForwardOutput(
            plain_logits,
            torch.zeros(2),
            latents,
            latents,
            (plain_cycle_logits,),
        ),
    )
    assert plain.logits is plain_logits
    assert plain.all_logits[0] is plain_cycle_logits


def test_halt_head_uses_configured_bias() -> None:
    config = _config()
    config.halt_init_bias = -3.25
    network = config.make()
    assert network.halt_head.bias is not None
    assert torch.equal(
        network.halt_head.bias,
        torch.full_like(network.halt_head.bias, -3.25),
    )


def test_model_parameter_count_log_uses_exact_millions_divisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: list[tuple[object, tuple[object, ...]]] = []

    def record(message: object, *args: object) -> None:
        records.append((message, args))

    monkeypatch.setattr(model.logger, "info", record)
    network = _config().make()
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    assert records == [
        ("model parameters: %.2fM", (parameter_count / 1e6,)),
    ]


def test_rotary_changes_recurrent_model_output() -> None:
    plain_config = _config(
        recurrent=True,
        channels_in=4,
        vocab_size=11,
        grid_shape=(2, 3),
    )
    torch.manual_seed(0)
    plain = plain_config.make()
    rotary_config = _config(
        recurrent=True,
        channels_in=4,
        vocab_size=11,
        grid_shape=(2, 3),
    )
    rotary_config.rope = RoPE.Config(channels_head=2)
    rotary_config.rope_grid_shape = (6,)
    torch.manual_seed(0)
    rotary = rotary_config.make()
    rotary.load_state_dict(plain.state_dict())
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]])
    assert not torch.equal(plain(tokens).logits, rotary(tokens).logits)


def test_step_matches_one_plain_forward_with_prefix() -> None:
    config = _config(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    config.prefix = RegisterTokens.Config(num_tokens=2)
    model = config.make()
    tokens = torch.randint(0, 11, (2, 6))
    z_slow, z_fast = model.init_latents(2)
    forward = model(tokens, z_slow, z_fast)
    stepped = model.step(tokens, z_slow, z_fast)
    assert torch.equal(stepped.logits, forward.logits)
    assert torch.equal(stepped.halt, forward.halt)
    assert torch.equal(stepped.z_slow, forward.z_slow)
    assert torch.equal(stepped.z_fast, forward.z_fast)


def test_step_uses_rotary_factors() -> None:
    plain_config = _config(
        recurrent=True,
        channels_in=4,
        vocab_size=11,
        grid_shape=(2, 3),
    )
    torch.manual_seed(0)
    plain = plain_config.make()
    rotary_config = _config(
        recurrent=True,
        channels_in=4,
        vocab_size=11,
        grid_shape=(2, 3),
    )
    rotary_config.rope = RoPE.Config(channels_head=2)
    rotary_config.rope_grid_shape = (6,)
    torch.manual_seed(0)
    rotary = rotary_config.make()
    rotary.load_state_dict(plain.state_dict())
    tokens = torch.tensor([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]])
    z_slow, z_fast = plain.init_latents(2)
    plain_step = plain.step(tokens, z_slow, z_fast)
    rotary_step = rotary.step(tokens, z_slow, z_fast)
    assert not torch.equal(plain_step.logits, rotary_step.logits)


def test_model_set_feedback_reaches_the_prediction_channel() -> None:
    config = _config(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    assert isinstance(config.embedding, GridEmbedding.Config)
    config.embedding.channels = [PredictionFeedback.Config(init_std=1.0)]
    model = config.make()
    channel = model.embedding.channels[0]
    assert isinstance(channel, PredictionFeedback)
    tokens = torch.randint(0, 11, (2, 6))
    feedback = torch.randint(0, 11, (2, 6))
    z_slow, z_fast = model.init_latents(2)
    baseline = model(tokens, z_slow, z_fast)
    model.set_feedback(feedback)
    with_feedback = model(tokens, z_slow, z_fast)
    consumed = model(tokens, z_slow, z_fast)
    assert not torch.equal(with_feedback.logits, baseline.logits)
    assert torch.equal(consumed.logits, baseline.logits)


def test_core_compile_and_rotary_optional_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compile_calls: list[dict[str, object]] = []

    def compile_eager(
        fn: Callable[..., object],
        **kwargs: object,
    ) -> Callable[..., object]:
        assert callable(fn)
        compile_calls.append(kwargs)
        return fn

    monkeypatch.setattr(torch, "compile", compile_eager)
    config = _config(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    model = config.make()
    tokens = torch.randint(0, 11, (2, 6))
    output = model(tokens)
    assert output.logits.shape == (2, 6, 11)
    assert output.all_logits == ()
    with model.eager(enabled=False):
        assert model.compiled is False
    compiled = _config(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    compiled.compile_core = CoreCompile.Config(mode="reduce-overhead", fullgraph=False)
    compiled_model = compiled.make()
    assert compiled_model.compiled
    assert compiled_model.device.type == "cpu"
    assert compile_calls == [{"mode": "reduce-overhead", "fullgraph": False}]
    compiled_model(torch.randint(0, 11, (2, 6)))
    reasoning = _config(channels_in=4, vocab_size=11, grid_shape=(2, 3))
    reasoning.compile_core = CoreCompile.Config(
        unit="reasoning",
        mode="max-autotune",
        fullgraph=True,
    )
    reasoning_model = reasoning.make()
    assert reasoning_model.compiled
    assert compile_calls == [
        {"mode": "reduce-overhead", "fullgraph": False},
        {"mode": "max-autotune", "fullgraph": True},
    ]
    assert reasoning_model(torch.randint(0, 11, (2, 6))).logits.shape == (2, 6, 11)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
