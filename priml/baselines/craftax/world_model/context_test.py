"""Check the context replay against the training forward of each step's own context.

A seeded two-block world model a few channels wide, over frames of 3 cells
and 4 scalars (``testing.tiny_model``), in float32 on the CPU with masked
attention. The replay's features are the training forward's final-normed
``obs`` hidden states of every step's exact context, alone in its window,
within ``feature_test``'s 2e-6, for both histories' shapes of contexts:
refill (contexts restart from their last decisions, few long shared passes)
and exact sliding windows (once full, every step its own pass), and for a tap
below the last block (the largest difference measured is 7.2e-7 on an arm64
Mac). Its backward is the vector-Jacobian product of that forward: every
trained weight's gradient of ``<G, F>`` matches autograd through the
per-step forward within 1e-5 of its largest entry (2.0e-6 measured). On
CUDA, FlashAttention 4 replays and backpropagates as masked attention in
bfloat16.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import (
    TYPE_CHECKING,
    Final,
    cast,
)

import copy
import dataclasses

from configgle import Fig, PartialConfig
from torch import nn
from torch._dynamo.config import patch

import pytest
import torch

from priml.baselines.craftax.world_model.batch import Kind
from priml.baselines.craftax.world_model.context import (
    ContextReplay,
    Contexts,
    JointWorldModel,
    feature_parameter_names,
    plan_replay,
)
from priml.baselines.craftax.world_model.feature import (
    InitialWeights,
    encode_frames,
    post_attention,
    pre_attention,
)
from priml.baselines.craftax.world_model.model import FrameEncoder
from priml.baselines.craftax.world_model.schema import craftax_schema
from priml.baselines.craftax.world_model.testing import (
    context_reference,
    random_contexts,
    small_schema,
    tiny_model,
)
from priml.model.attention.flash4 import Flash4UnavailableError, Flash4Varlen
from priml.model.attention.kernel import SdpaNaive, SdpaVarlen
from priml.model.attention.rope import RoPE
from priml.model.transformer.block import TransformerBlock


if TYPE_CHECKING:
    from collections.abc import Callable

    from configgle import Makeable
    from torch import Tensor

    from priml.baselines.craftax.world_model.attention import (
        VarlenAttention,
        VarlenKernel,
    )
    from priml.baselines.craftax.world_model.model import WorldModel


SLOTS: Final = 6
"""Prefix slots ``P``: decisions a step's context may reach before the window."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Table:
    """Each step's context length and anchor, and each row's filled prefix slots."""

    lengths: tuple[tuple[int, ...], ...]
    anchored: tuple[tuple[int, ...], ...]
    counts: tuple[int, ...]


REFILL: Final = Table(
    # Row 0 reaches 3 prefix slots, restarts from its last 3 decisions at step
    # 3 and begins an episode at step 6; row 1's episode began at prefix slot 4
    # and restarts at step 4; row 2 resumes at step 0 and begins an episode at 3.
    lengths=(
        (4, 5, 6, 4, 5, 6, 1, 2),
        (3, 4, 5, 6, 4, 5, 6, 7),
        (1, 2, 3, 1, 2, 3, 4, 5),
    ),
    anchored=(
        (0, 0, 0, 0, 0, 0, 1, 1),
        (1, 1, 1, 1, 0, 0, 0, 0),
        (0, 0, 0, 1, 1, 1, 1, 1),
    ),
    counts=(5, 2, 0),
)
"""Lengths, anchors and filled prefix slots of refill contexts, 7 passes."""

SLIDING: Final = Table(
    # Windows of 4 decisions: row 0's episode began at prefix slot 3 and
    # restarts at step 5, row 1's began before the prefix, row 2's at step 0.
    lengths=((4, 4, 4, 4, 4, 1, 2, 3), (4,) * 8, (1, 2, 3, 4, 4, 4, 4, 4)),
    anchored=((1, 0, 0, 0, 0, 1, 1, 1), (0,) * 8, (1, 1, 1, 1, 0, 0, 0, 0)),
    counts=(3, 6, 0),
)
"""Lengths, anchors and filled prefix slots of exact sliding contexts, 19 passes."""


def test_steps_whose_contexts_share_a_start_share_one_pass() -> None:
    """One pass per (row, first decision, anchor), as long as its longest member.

    Refill's 24 steps run 66 tokens in 7 passes, against 175 alone; once a
    sliding window is full every step is its own pass: 134 tokens in 19.
    """
    for table, passes, tokens in ((REFILL, 7, 66), (SLIDING, 19, 134)):
        plan = plan_replay(_contexts(table), bin_tokens=1, pass_tokens=16)
        assert plan.passes == passes
        computed = torch.cat([batch.steps for batch in plan.batches])
        assert torch.equal(computed.sort().values, torch.arange(3 * 8))
        padding = int(Kind.PAD)
        assert (
            sum(int((batch.kind != padding).sum()) for batch in plan.batches) == tokens
        )
        assert len(plan.batches) > 1


@pytest.mark.parametrize("bin_tokens", [1, 30], ids=["longest-pass", "wide"])
@pytest.mark.parametrize("table", [REFILL, SLIDING], ids=["refill", "sliding"])
def test_the_replay_is_the_training_forward_of_each_steps_context(
    table: Table,
    bin_tokens: int,
    model: WorldModel,
) -> None:
    """Bins as wide as the longest pass, or wider, holding several and padding."""
    contexts = _contexts(table)
    replay = _replay(bin_tokens=bin_tokens).forward(model, contexts, layers=2)
    assert len(replay.plan.batches) > 1
    with torch.no_grad():
        want = context_reference(model, contexts, layers=2)
    torch.testing.assert_close(replay.features, want, rtol=0, atol=2e-6)


def test_a_tap_below_the_last_block_replays_the_forward_cut_there(
    model: WorldModel,
) -> None:
    contexts = _contexts(REFILL)
    got = _replay().forward(model, contexts, layers=1).features
    with torch.no_grad():
        want = context_reference(model, contexts, layers=1)
        full = context_reference(model, contexts, layers=2)
    torch.testing.assert_close(got, want, rtol=0, atol=2e-6)
    assert not torch.allclose(got, full, atol=1e-2)


@pytest.mark.parametrize("table", [REFILL, SLIDING], ids=["refill", "sliding"])
def test_the_backward_is_the_vector_jacobian_product_of_the_forward(
    table: Table,
    model: WorldModel,
) -> None:
    contexts = _contexts(table)
    joint = JointWorldModel(model, layers=2, replay=_replay())
    replay = joint.forward(contexts)
    grad = torch.randn(
        replay.features.shape,
        generator=torch.Generator().manual_seed(2),
    )
    joint.backward(replay, grad)
    reference = copy.deepcopy(model).requires_grad_()
    (context_reference(reference, contexts, layers=2) * grad).sum().backward()
    names = joint.weights()
    for (name, weight), leaf in zip(names.items(), joint.parameters(), strict=True):
        assert leaf.grad is not None, name
        # The fused optimizer falls back to its reference path for a strided one.
        assert leaf.grad.is_contiguous(), name
        want = reference.get_parameter(name).grad
        assert want is not None, name
        torch.testing.assert_close(
            leaf.grad.view(weight.shape),
            want,
            rtol=0,
            atol=1e-5 * float(want.abs().max()),
            msg=name,
        )
    untrained = {name for name, _ in reference.named_parameters()} - set(names)
    assert all(reference.get_parameter(name).grad is None for name in untrained)


class _Held:
    """A guard that counts the calls it was held over."""

    def __init__(self) -> None:
        self.entered = 0
        self.held = False

    def __enter__(self) -> None:
        self.entered += 1
        self.held = True

    def __exit__(self, *exc: object) -> None:
        self.held = False


def test_every_replay_runs_under_the_guard(model: WorldModel) -> None:
    """The step's capture lock: no compiled kernel runs during another thread's compile."""
    guard = _Held()
    replay = _replay()
    kernel = replay.kernels.pre
    seen: list[bool] = []

    def watched(
        block: TransformerBlock,
        attn: VarlenAttention,
        *args: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        seen.append(guard.held)
        return kernel(block, attn, *args)

    replay.kernels = dataclasses.replace(replay.kernels, pre=watched)
    joint = JointWorldModel(model, layers=2, replay=replay, guard=guard)
    replayed = joint.forward(_contexts(REFILL))
    joint.backward(replayed, torch.ones_like(replayed.features))
    assert guard.entered == 2
    assert seen
    assert all(seen)


@pytest.mark.parametrize("encoder", [True, False], ids=["encoder", "global"])
def test_an_attention_kernel_the_engine_refuses_the_replay_refuses(
    *,
    encoder: bool,
    model: WorldModel,
) -> None:
    """The replay computes what the engine does, and refuses the same layouts."""
    frames = model.encoder
    assert isinstance(frames, FrameEncoder)
    stack = frames.stack if encoder else model.transformer
    block = stack.blocks[0]
    assert isinstance(block, TransformerBlock)
    assert isinstance(block.attn, nn.Module)
    block.attn.register_module("attn_kernel", SdpaNaive.Config().make())
    with pytest.raises(ValueError, match="not as SdpaNaive"):
        _replay().forward(model, _contexts(REFILL), layers=2)


def test_the_trained_copy_publishes_into_its_source(model: WorldModel) -> None:
    """A fused QKV ensemble trains as its matrix of rows; the source keeps its shapes."""
    source = copy.deepcopy(model).requires_grad_(requires_grad=False)
    joint = JointWorldModel(source, layers=1, replay=_replay())
    shapes = [tuple(leaf.shape) for leaf in joint.parameters()]
    # 9 + 3 + 3 heads of 4 channels over the width-36 stream.
    assert (15 * 4, 36) in shapes
    assert all(len(shape) <= 2 for shape in shapes)
    assert all(leaf.requires_grad for leaf in joint.parameters())
    assert not any(p.requires_grad for p in source.parameters())
    trained = {id(leaf) for leaf in joint.parameters()}
    assert not any(
        p.requires_grad for p in joint.model.parameters() if id(p) not in trained
    )
    with torch.no_grad():
        for leaf in joint.parameters():
            leaf.add_(1)
    for name, weight in joint.weights().items():
        assert torch.equal(weight, model.get_parameter(name) + 1), name
        assert torch.equal(source.get_parameter(name), model.get_parameter(name)), name
    joint.publish()
    for name, weight in joint.weights().items():
        assert torch.equal(source.get_parameter(name), weight), name
    fresh = JointWorldModel(copy.deepcopy(model), layers=1, replay=_replay())
    fresh.load_weights(joint.weights())
    for name, weight in fresh.weights().items():
        assert torch.equal(weight, source.get_parameter(name)), name
    with pytest.raises(ValueError, match="differ by"):
        fresh.load_weights({"start": source.start})


def test_the_feature_reads_its_encoder_tables_and_blocks_up_to_the_tap(
    model: WorldModel,
) -> None:
    names = feature_parameter_names(model, layers=1)
    assert "start" in names
    assert "transformer.proj_out.weight" in names
    assert "transformer.blocks.0.attn.proj_qkv.weight" in names
    assert not [name for name in names if name.startswith("transformer.blocks.1.")]
    for unread in ("decoder.", "action_head.", "cond_proj."):
        assert not [name for name in names if name.startswith(unread)], unread


@pytest.mark.parametrize(
    ("step", "length", "count"),
    [(2, 0, 5), (0, 3, 1)],
    ids=["empty", "past-the-prefix"],
)
def test_a_context_the_actor_could_not_have_read_is_refused(
    step: int,
    length: int,
    count: int,
) -> None:
    contexts = _contexts(REFILL)
    contexts.lengths[0, step] = length
    contexts.prefix_counts[0] = count
    with pytest.raises(ValueError, match="filled prefix slots"):
        plan_replay(contexts, bin_tokens=1, pass_tokens=16)


def test_a_context_before_the_rows_first_decision_is_refused() -> None:
    """More filled prefix slots than slots: step 0's 8 decisions reach before slot 0."""
    contexts = random_contexts(
        torch.tensor(((8, 2), (1, 2), (2, 3))),
        torch.zeros(3, 2, dtype=torch.bool),
        schema=small_schema(),
        slots=SLOTS,
        counts=torch.tensor((SLOTS + 1, 0, 1)),
        seed=7,
    )
    with pytest.raises(ValueError, match="filled prefix slots"):
        plan_replay(contexts, bin_tokens=1, pass_tokens=16)


@pytest.mark.parametrize("name", ["bin_tokens", "pass_tokens", "frames_per_batch"])
def test_a_bin_and_a_micro_batch_must_hold_something(name: str) -> None:
    config = ContextReplay.Config()
    setattr(config, name, 0)
    with pytest.raises(ValueError, match=f"{name}=0"):
        config.make()


@pytest.mark.compute_torch_compile
@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_a_compiled_replay_compiles_each_kernel_once_per_grad_mode(
    model: WorldModel,
) -> None:
    """Under dynamo's eager backend the replay is the eager one, bit for bit.

    Two graphs per kernel, with grad and without, whatever the blocks and
    micro-batches: dynamo guards on a parametrized module's per-instance class,
    which compiled once per block and ran a 20-block model past its limit.
    """
    contexts = _cuda(_contexts(SLIDING))
    model = model.cuda()
    runs: list[tuple[Tensor, list[Tensor]]] = []
    for compiled in (False, True):
        replay = _replay()
        if compiled:
            config = ContextReplay.Config()
            config.pass_tokens, config.frames_per_batch = 26, 5
            config.bin_tokens = 1
            config.compile = PartialConfig(
                torch.compile,
                backend="eager",
                fullgraph=True,
                dynamic=False,
            )
            replay = config.make()
        joint = JointWorldModel(copy.deepcopy(model), layers=2, replay=replay)
        torch.compiler.reset()
        with patch(recompile_limit=2):
            replayed = joint.forward(contexts)
            joint.backward(replayed, torch.ones_like(replayed.features))
        gradients = [leaf.grad for leaf in joint.parameters()]
        runs.append(
            (replayed.features, [grad for grad in gradients if grad is not None]),
        )
    (eager, eager_grads), (compiled_features, compiled_grads) = runs
    assert torch.equal(compiled_features, eager)
    assert len(compiled_grads) == len(eager_grads) == len(list(joint.parameters()))
    for got, want in zip(compiled_grads, eager_grads, strict=True):
        assert torch.equal(got, want)


def test_the_compile_slot_wraps_every_kernel_the_replay_runs(model: WorldModel) -> None:
    """The slot wraps the block and encoder functions, and the replay runs only those.

    ``torch.compile`` in the slot is the GPU test's above; here a wrapper that
    counts its calls, so a function the replay ran around the slot would be
    missed, in the forward or in the backward.
    """
    config = ContextReplay.Config()
    config.pass_tokens, config.frames_per_batch = 26, 5
    config.bin_tokens = 1
    config.compile = _CountingCompile.Config()
    replay = config.make()
    kernels = (replay.kernels.pre, replay.kernels.post, replay.kernels.encode)
    counted = [kernel for kernel in kernels if isinstance(kernel, _CountedCalls)]
    assert [kernel.function for kernel in counted] == [
        pre_attention,
        post_attention,
        encode_frames,
    ]
    joint = JointWorldModel(model, layers=2, replay=replay)
    replayed = joint.forward(_contexts(REFILL))
    forward = [kernel.calls for kernel in counted]
    joint.backward(replayed, torch.ones_like(replayed.features))
    assert all(calls > 0 for calls in forward)
    assert all(
        kernel.calls > calls for kernel, calls in zip(counted, forward, strict=True)
    )
    eager = _replay().forward(model, _contexts(REFILL), layers=2)
    assert torch.equal(replayed.features, eager.features)


@pytest.mark.filterwarnings(
    "ignore::DeprecationWarning:(flash_attn|cutlass|quack)",
    "ignore::UserWarning:cutlass",
)
@pytest.mark.gpu_flash_attention
@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "compiled",
    [False, pytest.param(True, marks=pytest.mark.compute_torch_compile)],
    ids=["eager", "compiled"],
)
def test_flash4_replays_and_backpropagates_as_masked_attention_in_bfloat16(
    *,
    compiled: bool,
) -> None:
    """FA4's varlen forward and backward, held to float32 as masked SDPA in bfloat16 is.

    The smoke model with heads of 64 channels, the least FA4 takes, over the
    full schema: in float32 with masked attention, the reference, then placed
    as the actor places it -- bfloat16 with float32 rotary tables -- with
    masked attention and with FA4. Each bfloat16 feature is within 0.05 of the
    reference, and each weight's gradient no further from the reference's, in
    relative norm, than twice masked attention's in bfloat16 plus 1%: a
    gradient that sums many terms of either sign is far from float32 under any
    bfloat16 kernel, and an FA4 fault is further. Compiled, FA4's run wraps the
    per-block and encoder functions in ``torch.compile``, is held to the same,
    and compiles each kernel once per grad mode, not once per block.
    """
    try:
        Flash4Varlen.Config().make()
    except Flash4UnavailableError as error:
        pytest.skip(str(error))
    config = InitialWeights.Config()
    config.experiment = "priml.baselines.craftax.world_model.experiments.exp_smoke"
    config.overrides = [
        "step.model.transformer.num_layers=2",
        "step.model.transformer.block.attn.channels_head=64",
    ]
    reference = config.make()().cuda()
    half = copy.deepcopy(reference).to(torch.bfloat16)
    for module in half.modules():
        if isinstance(module, RoPE):
            module.to(torch.float32)
    table = REFILL
    contexts = _cuda(
        random_contexts(
            torch.tensor(table.lengths),
            torch.tensor(table.anchored).bool(),
            schema=craftax_schema(),
            slots=SLOTS,
            counts=torch.tensor(table.counts),
            seed=7,
        ),
    )
    runs = [
        _cuda_replay(model, contexts, attention=attention, compiled=wrap)
        for model, attention, wrap in (
            (reference, SdpaVarlen.Config(), False),
            (half, SdpaVarlen.Config(), False),
            (half, Flash4Varlen.Config(), compiled),
        )
    ]
    (exact, exact_grads), (masked, masked_grads), (flash, flash_grads) = runs
    torch.testing.assert_close(masked, exact, rtol=0, atol=0.05)
    torch.testing.assert_close(flash, exact, rtol=0, atol=0.05)
    far = {
        name: (_relative(flash_grads[name], want), _relative(masked_grads[name], want))
        for name, want in exact_grads.items()
        if _relative(flash_grads[name], want)
        > 2 * _relative(masked_grads[name], want) + 0.01
    }
    assert not far, far


def _cuda_replay(
    model: WorldModel,
    contexts: Contexts,
    *,
    attention: Makeable[VarlenKernel],
    compiled: bool,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Replay and backpropagate a fixed gradient; return float32 features and gradients."""
    replay = ContextReplay.Config()
    replay.attention = attention
    if compiled:
        replay.compile = PartialConfig(torch.compile, fullgraph=True, dynamic=False)
        torch.compiler.reset()
    replay.bin_tokens = 30
    joint = JointWorldModel(model, layers=2, replay=replay.make())
    # Each compiled kernel compiles once without grad and once with: a third
    # graph is a compile per block or per micro-batch, which at 20 blocks runs
    # past dynamo's limit of 8, so a fullgraph kernel raises here instead.
    with patch(recompile_limit=2):
        replayed = joint.forward(contexts)
        grad = torch.randn(
            replayed.features.shape,
            generator=torch.Generator().manual_seed(2),
        ).to("cuda", replayed.features.dtype)
        joint.backward(replayed, grad)
    gradients: dict[str, Tensor] = {}
    for (name, weight), leaf in zip(
        joint.weights().items(),
        joint.parameters(),
        strict=True,
    ):
        assert leaf.grad is not None, name
        # The fused optimizer falls back to its reference path for a strided one.
        assert leaf.grad.is_contiguous(), name
        gradients[name] = leaf.grad.view(weight.shape).float()
    return replayed.features.float(), gradients


class _CountedCalls:
    """A function, and how often it was called."""

    def __init__(self, function: Callable[..., object]) -> None:
        self.function = function
        self.calls = 0

    def __call__(self, *args: object) -> object:
        """Count the call, then make it."""
        self.calls += 1
        return self.function(*args)


class _CountingCompile:
    """A ``compile`` slot that counts each kernel's calls rather than compiling it."""

    class Config(Fig["_CountingCompile"]):
        """No options."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, function: Callable[..., object]) -> Callable[..., object]:
        """Return ``function``, counted."""
        return _CountedCalls(function)


def _cuda(contexts: Contexts) -> Contexts:
    """Return a copy of ``contexts`` on the CUDA device."""
    return Contexts(
        **{
            entry.name: cast("torch.Tensor", getattr(contexts, entry.name)).cuda()
            for entry in fields(contexts)
        },
    )


def _relative(got: Tensor, want: Tensor) -> float:
    """Return ``|got - want| / |want|`` in the Frobenius norm."""
    return float((got - want).norm() / want.norm())


def _contexts(table: Table) -> Contexts:
    """Return random decisions read under a table's lengths, anchors and prefix counts."""
    return random_contexts(
        torch.tensor(table.lengths),
        torch.tensor(table.anchored).bool(),
        schema=small_schema(),
        slots=SLOTS,
        counts=torch.tensor(table.counts),
        seed=7,
    )


def _replay(*, bin_tokens: int = 1) -> ContextReplay:
    """Return masked attention in micro-batches of 26 tokens and five frames."""
    config = ContextReplay.Config()
    config.bin_tokens = bin_tokens
    config.pass_tokens = 26
    config.frames_per_batch = 5
    return config.make()


@pytest.fixture
def model() -> WorldModel:
    """Return the two-block tiny world model over the small schema, float32 on the CPU."""
    return tiny_model(small_schema(), global_layers=2)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
