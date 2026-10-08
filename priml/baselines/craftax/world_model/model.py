"""Hierarchical world model: frame encoder, global transformer, local decoder.

A packed window (``batch.PackedBatch``) has a ``start``, ``obs``, or ``act``
kind at each global position. The frame encoder compresses each frame's 150
slots into one pooled vector (the ``obs`` input) plus 150 per-slot vectors (the
local decoder's memory). The global transformer, a Qwen3 block stack, runs
causally within each episode segment. The action head reads it at ``obs``
positions. At each ``start`` and ``act`` position a local job begins: the local
decoder generates reward, done, and the next frame slot by slot, cross-attending
to the current frame's memory, or to a learned null vector for a ``start`` job.

Every size default is the design's: a ~283M global stack of width 1152, and a
width-512 encoder (~12.4M) and decoder (~16.6M) sharing a tied 461-row table.
"""

from collections.abc import Callable
from dataclasses import KW_ONLY, field
from functools import partial
from typing import Protocol, Self, override

import dataclasses

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn
from torch._ops import OpOverload
from torch.nn import functional
from torch.utils.checkpoint import (
    CheckpointPolicy,
    SelectiveCheckpointContext,
    checkpoint,
    create_selective_checkpoint_contexts,
)

import torch

from priml.baselines.craftax.lib.costs import (
    broadcast_add_cost,
    cast_cost,
    concat_cost,
    gather_cost,
    logits_cost,
    residual_cost,
)
from priml.baselines.craftax.world_model.attention import VarlenAttention
from priml.baselines.craftax.world_model.batch import Kind, PackedBatch
from priml.baselines.craftax.world_model.loss import (
    ModalityLoss,
    Nll,
    cell_nll,
    cross_entropy,
    modality_loss,
    scalar_nll,
)
from priml.baselines.craftax.world_model.schema import (
    FrameSchema,
    craftax_schema,
)
from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
    set_cost,
    traffic,
)
from priml.loss import simple_loss
from priml.math.custom_types import TensorFn
from priml.model.attention.attention import Attention
from priml.model.attention.rope import HuggingFaceFrequencies, RoPE
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsInOutConfig,
    ChannelsOut,
    HasResetParameters,
    TensorModule,
    propagate_attr,
)
from priml.model.embedding import Embedding
from priml.model.init import InitFn, call_init
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock
from priml.model.transformer.transformer import Transformer


class MemoryAttention(Protocol):
    """Attention from ``x`` to a separate ``memory`` sequence."""

    def __call__(self, x: Tensor, /, *, memory: Tensor) -> Tensor:
        """Attend ``[..., S, C]`` queries to ``[..., M, C]`` memory."""
        ...


class FrameEncoderLike(Protocol):
    """Encoder of embedded frame slots: one pooled vector plus per-slot memory.

    ``WorldModel`` embeds every frame's slots outside the encoder, feeds the
    pooled vector to the global stack and the memory to the local decoder's
    cross-attention; any encoder making this contract can take its place.
    """

    def __call__(self, slots: Tensor, /) -> tuple[Tensor, Tensor]:
        """Map ``[..., num_slots, C]`` slots to pooled ``[..., C]`` and memory."""
        ...

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        ...

    def frame_macs(self) -> int:
        """Return one frame's multiply-adds in its matrices (attention excluded)."""
        ...


class FrameEncoderConfig(Makeable[FrameEncoderLike], Protocol):
    """A frame encoder's config, as ``WorldModel.Config`` reads and edits it."""

    channels_in: int
    """Width of the slots, the memory and the pooled vector."""

    num_slots: int
    """Slots per frame; -1 until the world model pushes its schema's."""


class BoardEmbedding(Protocol):
    """Embeds board cells from their field IDs' rows of a table."""

    def __call__(self, table: nn.Embedding, ids: Tensor, /) -> Tensor:
        """Map ``[..., cells, fields]`` IDs to ``[..., cells, C]`` cell embeddings."""
        ...


class RecomputePolicy(Protocol):
    """Which ops' outputs a selective recompute stores (``torch.utils.checkpoint``)."""

    def __call__(
        self,
        ctx: SelectiveCheckpointContext,
        op: OpOverload,
        /,
        *args: object,
        **kwargs: object,
    ) -> CheckpointPolicy:
        """Return whether backward reads ``op``'s stored outputs or recomputes them."""
        ...


class DecoderBlock(nn.Module):
    """Pre-norm block: causal self-attention, cross-attention, then SwiGLU."""

    class Config(Fig["DecoderBlock"], kw_only=False):
        """Set ``channels_in``; every sublayer inherits it."""

        channels_in: int = -1
        """Residual width."""

        channels_out: int = -1
        """Output width; must equal ``channels_in`` (-1 to infer)."""

        _: KW_ONLY

        attn: Makeable[TensorModule] = field(
            default_factory=lambda: _local_attention(causal=True),
        )
        """Causal self-attention within the job."""

        cross_attn: Makeable[MemoryAttention] = field(
            default_factory=lambda: Attention.Config(
                norm_qk=RMSNorm.Config(elementwise_affine=True),
                share_qk_norm=False,
                init_weight=partial(nn.init.normal_, std=0.02),
            ),
        )
        """Attention to the job's memory: the queries' and the memory's own norms."""

        ffn: Makeable[TensorModule] = field(default_factory=lambda: _swiglu(1_344))
        """Feed-forward sublayer."""

        norm1: Makeable[TensorModule] = field(
            default_factory=lambda: RMSNorm.Config(elementwise_affine=True),
        )
        """Norm before self-attention."""

        norm2: Makeable[TensorModule] = field(
            default_factory=lambda: RMSNorm.Config(elementwise_affine=True),
        )
        """Norm before cross-attention."""

        norm3: Makeable[TensorModule] = field(
            default_factory=lambda: RMSNorm.Config(elementwise_affine=True),
        )
        """Norm before the feed-forward."""

        checkpoint: bool = False
        """Recompute the block's activations in backward instead of storing them."""

        recompute_policy: RecomputePolicy | None = None
        """With ``checkpoint``, which ops' outputs the recompute stores, e.g.
        ``keep_attention``; None stores only the block's inputs."""

        @override
        def finalize(self) -> Self:
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            for child in (
                self.attn,
                self.cross_attn,
                self.ffn,
                self.norm1,
                self.norm2,
                self.norm3,
            ):
                if isinstance(child, ChannelsIn) and child.channels_in == -1:
                    propagate_attr(child, "channels_in", self.channels_in)
                if isinstance(child, ChannelsOut) and child.channels_out == -1:
                    propagate_attr(child, "channels_out", self.channels_in)
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            memory_len: int,
            **kwargs: object,
        ) -> Cost:
            """Sum the sublayers, their norms, and the three residual adds.

            ``checkpoint`` is recompute, not model work, as priml's block counts it.

            Args:
              seq_len: Positions per job.
              batch_size: Jobs in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              memory_len: Memory positions each job cross-attends to.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            sublayers = sum(
                (
                    cost(
                        child,
                        seq_len=seq_len,
                        batch_size=batch_size,
                        dtype=dtype,
                        **kwargs,
                    )
                    for child in (
                        self.attn,
                        self.ffn,
                        self.norm1,
                        self.norm2,
                        self.norm3,
                    )
                ),
                Cost(),
            )
            cross = cost(
                self.cross_attn,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                memory_len=memory_len,
                **kwargs,
            )
            adds = residual_cost(
                channels=3 * self.channels_in,
                rows=seq_len * batch_size,
                dtype=dtype,
            )
            return sublayers + cross + adds

    def __init__(self, config: Config) -> None:
        if config.channels_in != config.channels_out:
            raise ValueError(
                f"channels_in={config.channels_in} must equal "
                f"channels_out={config.channels_out} for DecoderBlock.",
            )
        super().__init__()
        self.checkpoint = config.checkpoint
        self.recompute_policy = config.recompute_policy
        self.attn = config.attn.make()
        self.cross_attn = config.cross_attn.make()
        self.ffn = config.ffn.make()
        self.norm1 = config.norm1.make()
        self.norm2 = config.norm2.make()
        self.norm3 = config.norm3.make()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        for module in self.children():
            if isinstance(module, HasResetParameters):
                module.reset_parameters()

    @override
    def forward(
        self,
        x: Tensor,
        /,
        *,
        memory: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Apply the block to ``[..., S, C]`` inputs with ``[..., M, C]`` memory.

        ``memory`` is optional only so the block fits priml's ``Transformer``,
        which forwards keyword messages to every block; it must be given.
        """
        del kwargs
        if memory is None:
            raise ValueError("DecoderBlock needs memory.")
        # Gated like priml's ``TransformerBlock``: recomputing saves memory only
        # when a backward will run, and a checkpoint under ``inference_mode`` can
        # deadlock a multi-rank evaluation.
        if self.checkpoint and torch.is_grad_enabled():
            return _recompute(self._forward, x, memory, policy=self.recompute_policy)
        return self._forward(x, memory)

    def _forward(self, x: Tensor, memory: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), memory=memory)
        return x + self.ffn(self.norm3(x))


class EncoderBlock(TransformerBlock):
    """priml's pre-norm block, whose recompute can store its attention output."""

    class Config(Makes["EncoderBlock"], TransformerBlock.Config):
        """priml's block, plus what its recompute stores."""

        recompute_policy: RecomputePolicy | None = None
        """With ``checkpoint``, which ops' outputs the recompute stores, e.g.
        ``keep_attention``; None stores only the block's inputs."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.recompute_policy = config.recompute_policy

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        """Apply the block, recomputing it in backward when ``checkpoint`` is set."""
        # priml's gate, for the reason ``DecoderBlock.forward`` gives.
        if self.checkpoint and torch.is_grad_enabled():
            return _recompute(
                partial(self._forward, **kwargs),
                x,
                policy=self.recompute_policy,
            )
        return self._forward(x, **kwargs)


class GlobalTransformer(Transformer):
    """Qwen3-style causal stack over a packed window's global positions.

    Pre-norm affine RMSNorm, QK-norm, GQA (9 query heads of 128, 3 KV heads),
    SwiGLU, and RoPE with base 500,000 over each position's segment offset.
    ``forward(x, positions=pos, cu_seqlens=cu_seqlens)`` returns the final-normed
    hidden states.
    """

    class Config(Makes["GlobalTransformer"], Transformer.Config, kw_only=False):
        """The design's 20 x 1152 global model; ``block`` holds everything else."""

        channels_in: int = 1_152
        """Residual width."""

        _: KW_ONLY

        num_layers: int = 20
        """Blocks in the stack."""

        block: ChannelsInOutConfig | list[ChannelsInOutConfig] = field(
            default_factory=lambda: _global_block(channels_head=128),
        )
        """Block template (broadcast ``num_layers`` times), or a list."""

        proj_out: Makeable[TensorModule] | None = field(
            default_factory=lambda: RMSNorm.Config(elementwise_affine=True),
        )
        """Final norm; the heads live on the model that reads the stack."""

        @override
        def finalize(self) -> Self:
            blocks = self.block if isinstance(self.block, list) else [self.block]
            for block in blocks:
                if isinstance(block, TransformerBlock.Config):
                    attn = block.attn
                    if (
                        isinstance(attn, VarlenAttention.Config)
                        and isinstance(attn.rope, RoPE.Config)
                        and attn.rope.channels_head == -1
                    ):
                        attn.rope.channels_head = attn.channels_head
            return super().finalize()

    def max_attention_logit(self) -> Tensor:
        """Return the last forward's largest attention logit over every block.

        The kernel sets its meaning: exact under ``SdpaVarlen``, an upper bound
        within ``log(t_g)`` under ``Flash4Varlen`` (see ``attention``).

        Returns:
          max_logit: A detached float32 device scalar; logging it adds no host
            sync until the logger reads it.

        """
        values: list[Tensor] = []
        for module in self.modules():
            if isinstance(module, VarlenAttention):
                if module.max_logit is None:
                    raise ValueError("Read it after a forward.")
                values.append(module.max_logit)
        return torch.stack(values).amax()


class FrameEncoder(nn.Module):
    """Bidirectional encoder of one frame's slots plus a learned pooling token."""

    class Config(Fig["FrameEncoder"]):
        """The design's 4 x 512 encoder."""

        channels_in: int = 512
        """Width of the slot embeddings and the stack."""

        num_slots: int = -1
        """Slots per frame (-1 until the world model pushes its schema's)."""

        stack: Transformer.Config = field(
            default_factory=lambda: _local_stack(
                EncoderBlock.Config(
                    attn=_local_attention(causal=False),
                    ffn=_swiglu(1_344),
                    norm1=_norm(),
                    norm2=_norm(),
                ),
            ),
        )
        """Bidirectional block stack ending in a norm."""

        init_weight: InitFn = partial(nn.init.normal_, std=0.02)
        """Init of the slot embeddings and the pooling token."""

        cast_stream: TensorFn | None = None
        """Applied to the stack's input, its residual stream, e.g.
        ``to_autocast_dtype``: autocast leaves the slot embeddings, and so the
        stream, in float32. None leaves the stream as embedded."""

        @override
        def finalize(self) -> Self:
            if self.stack.channels_in == -1:
                self.stack.channels_in = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the slot embeddings' add, the pooling token, and the stack.

            The stack runs over every slot and the pooling token; its outputs
            split into views.

            Args:
              batch_size: Frames in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to the stack.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            width, positions = self.channels_in, self.num_slots + 1
            values = batch_size * positions * width
            # The pooling token is copied in by the concatenation; its gradient
            # sums over the frames.
            pool = elementwise_cost(
                primal=0,
                adjoint=0,
                channels=width,
                params=width,
                rows=batch_size,
                dtype=dtype,
            )
            inputs = (
                broadcast_add_cost(
                    params=self.num_slots * width,
                    rows=batch_size,
                    dtype=dtype,
                )
                + pool
                + concat_cost(elements=values, dtype=dtype)
            )
            if self.cast_stream is not None:
                inputs += cast_cost(elements=values, dtype=dtype)
            return inputs + self.stack.cost(
                seq_len=positions,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.slot_embedding = nn.Parameter(
            torch.empty(config.num_slots, config.channels_in),
        )
        self.pool = nn.Parameter(torch.empty(config.channels_in))
        self.stack = config.stack.make()
        self.cast_stream = config.cast_stream
        self._init_weight = config.init_weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        call_init(self._init_weight, self.slot_embedding)
        call_init(self._init_weight, self.pool)
        self.stack.reset_parameters()

    def frame_macs(self) -> int:
        """Return the stack's matrices times the positions they run: slots and pool."""
        matrices = sum(p.numel() for p in self.stack.parameters() if p.ndim >= 2)
        return matrices * (len(self.slot_embedding) + 1)

    @override
    def forward(self, slots: Tensor) -> tuple[Tensor, Tensor]:
        """Encode frames.

        Args:
          slots: Embedded frame slots ``[..., num_slots, C]``.

        Returns:
          pooled: The pooling token's output, ``[..., C]``.
          memory: The per-slot outputs, ``[..., num_slots, C]``.

        """
        pool = self.pool.expand(*slots.shape[:-2], 1, -1)
        x = torch.cat([pool, slots + self.slot_embedding], dim=-2)
        out = self.stack(x if self.cast_stream is None else self.cast_stream(x))
        return out[..., 0, :], out[..., 1:, :]


class LocalDecoder(nn.Module):
    """Causal decoder of one local job, cross-attending to its memory."""

    class Config(Fig["LocalDecoder"]):
        """The design's 4 x 512 decoder."""

        channels_in: int = 512
        """Width of the inputs, slot embeddings, and stack."""

        num_slots: int = -1
        """Positions per job (-1 until the world model pushes its schema's)."""

        stack: Transformer.Config = field(
            default_factory=lambda: _local_stack(DecoderBlock.Config()),
        )
        """Decoder block stack ending in a norm."""

        init_weight: InitFn = partial(nn.init.normal_, std=0.02)
        """Init of the slot embeddings and the null memory."""

        cast_stream: TensorFn | None = None
        """Applied to the stack's residual stream and memory, e.g.
        ``to_autocast_dtype``: autocast leaves the slot embeddings, and so both,
        in float32. None leaves both as embedded."""

        @override
        def finalize(self) -> Self:
            if self.stack.channels_in == -1:
                self.stack.channels_in = self.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            memory_len: int,
            **kwargs: object,
        ) -> Cost:
            """Cost the null memory's select, the slot embeddings' add, and the stack.

            The select does no arithmetic: it reads the memory and the null
            vector and writes one, and back routes the gradient, the null
            vector's share summed over every memory position.

            Args:
              batch_size: Jobs in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              memory_len: Memory positions per job.
              **kwargs: The open bus, forwarded to the stack.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            width = self.channels_in
            memory = batch_size * memory_len
            select = elementwise_cost(
                primal=0,
                adjoint=0,
                channels=width,
                params=width,
                rows=memory,
                adjoint_inputs=1,
                dtype=dtype,
            )
            inputs = select + broadcast_add_cost(
                params=self.num_slots * width,
                rows=batch_size,
                dtype=dtype,
            )
            if self.cast_stream is not None:
                inputs += cast_cost(
                    elements=(batch_size * self.num_slots + memory) * width,
                    dtype=dtype,
                )
            return inputs + self.stack.cost(
                seq_len=self.num_slots,
                batch_size=batch_size,
                dtype=dtype,
                memory_len=memory_len,
                **kwargs,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.slot_embedding = nn.Parameter(
            torch.empty(config.num_slots, config.channels_in),
        )
        self.memory_null = nn.Parameter(torch.empty(config.channels_in))
        self.stack = config.stack.make()
        self.cast_stream = config.cast_stream
        self._init_weight = config.init_weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        call_init(self._init_weight, self.slot_embedding)
        call_init(self._init_weight, self.memory_null)
        self.stack.reset_parameters()

    @override
    def forward(self, inputs: Tensor, *, memory: Tensor, has_memory: Tensor) -> Tensor:
        """Decode jobs.

        Args:
          inputs: ``[c, x_0, ...]`` per job, ``[..., num_slots, C]``.
          memory: Each job's current-frame memory, ``[..., M, C]``.
          has_memory: False where the job reads the null memory instead, ``[...]``.

        Returns:
          hidden: Final-normed hidden state per position, ``[..., num_slots, C]``.

        """
        # Every key of a null job is the same learned vector, so attention is
        # uniform over identical values: exactly attending to one null key,
        # while keeping one memory shape for the whole batch.
        memory = torch.where(has_memory[..., None, None], memory, self.memory_null)
        x = inputs + self.slot_embedding
        if self.cast_stream is not None:
            x, memory = self.cast_stream(x), self.cast_stream(memory)
        return self.stack(x, memory=memory)


class GlobalLanguageModel(nn.Module):
    """The global stack as a language model: token embedding, stack, tied head.

    Its tests check it against transformers' ``Qwen3ForCausalLM``; ``FlatModel``
    runs it over every Craftax slot.
    """

    class Config(Fig["GlobalLanguageModel"]):
        """A design-size stack over the Craftax vocabulary."""

        vocab_size: int = field(default_factory=lambda: craftax_schema().vocab_size)
        """Token vocabulary: the tied table's rows."""

        transformer: GlobalTransformer.Config = field(
            default_factory=GlobalTransformer.Config,
        )
        """The global block stack."""

        init_weight: InitFn = partial(nn.init.normal_, std=0.02)
        """Init of the tied token table."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the token lookup, the stack, and the tied head, which owns nothing.

            Args:
              seq_len: Tokens per row.
              batch_size: Rows in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to the stack.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            width, rows = self.transformer.channels_in, seq_len * batch_size
            table = Embedding.Config(channels_out=width, channels_in=self.vocab_size)
            return (
                table.cost(seq_len=seq_len, batch_size=batch_size, dtype=dtype)
                + cost(
                    self.transformer,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + logits_cost(
                    channels_in=width,
                    channels_out=self.vocab_size,
                    rows=rows,
                    weight=False,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.embedding = Embedding.Config(
            channels_out=config.transformer.channels_in,
            channels_in=config.vocab_size,
            init_weight=config.init_weight,
        ).make()
        self.transformer = config.transformer.make()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        self.embedding.reset_parameters()
        self.transformer.reset_parameters()

    @override
    def forward(
        self,
        tokens: Tensor,
        *,
        positions: Tensor,
        cu_seqlens: Tensor,
    ) -> Tensor:
        """Return float32 next-token logits ``[B, T, vocab_size]``.

        Args:
          tokens: Token IDs ``[B, T]``.
          positions: RoPE position within each segment, ``[B, T]``.
          cu_seqlens: Segment boundaries of the flattened batch, int32.

        Returns:
          logits: Tied-head logits.

        """
        hidden = self.transformer(
            self.embedding(tokens),
            positions=positions,
            cu_seqlens=cu_seqlens,
        )
        return functional.linear(hidden, self.embedding.weight).float()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class WorldModelLogits:
    """Float32 logits of one packed batch.

    Attributes:
      action: Action-head logits at every global position, ``[B, t_g, actions]``.
      local: Local-decoder logits per job and local slot, ``[J, local_slots, V]``.

    """

    action: Tensor
    local: Tensor


# A board embedding answers ``cost`` for ``rows`` cells of ``fields`` IDs each into a
# ``vocab`` x ``width`` table it never owns: the world model's table owns it.
def _gathered_board_cost(
    *,
    rows: int,
    fields: int,
    vocab: int,
    width: int,
    dtype: torch.dtype | None,
) -> Cost:
    return gather_cost(
        rows=rows * fields,
        width=width,
        source=vocab,
        dtype=dtype,
    ) + reduction_cost(
        input_elements=rows * fields * width,
        output_groups=rows * width,
        dtype=dtype,
    )


# The multi-hot rows are built without a gradient, so back the matmul computes
# only the table's gradient: a second product the primal's size.
def _multi_hot_board_cost(
    *,
    rows: int,
    fields: int,
    vocab: int,
    width: int,
    dtype: torch.dtype | None,
) -> Cost:
    product = matmul_cost(
        channels_in=vocab,
        channels_out=width,
        rows=rows,
        weight=False,
        dtype=dtype,
    ).only("primal")
    hot = traffic(
        "primal",
        "selection",
        elements=rows * vocab + 2 * rows * fields,
        flops=rows * fields,
        dtype=dtype,
    )
    return hot + product + product.relabel("adjoint")


@set_cost(_gathered_board_cost)
def gathered_board(table: nn.Embedding, ids: Tensor) -> Tensor:
    """Return each cell's embedding: its field IDs' rows of ``table``, summed."""
    return table(ids).sum(-2)


@set_cost(_multi_hot_board_cost)
def multi_hot_board(table: nn.Embedding, ids: Tensor) -> Tensor:
    """Return each cell's embedding as one multi-hot row times ``table``'s weight.

    ``gathered_board``'s sum as a matmul, so the table's gradient is a matmul
    too. Both are exact in float32; under autocast the matmul runs in the
    autocast dtype, so the two differ in its rounding.

    Args:
      table: The token table.
      ids: Each cell's field IDs ``[..., cells, fields]``.

    Returns:
      board: ``[..., cells, C]``.

    """
    weight = table.weight
    board_hot = torch.zeros(
        *ids.shape[:-1],
        len(weight),
        dtype=weight.dtype,
        device=ids.device,
    )
    board_hot.scatter_add_(-1, ids, torch.ones_like(ids, dtype=weight.dtype))
    return board_hot @ weight


class WorldModel(nn.Module):
    """Encoder, global transformer, and local decoder, trained end to end."""

    class Config(Fig["WorldModel"]):
        """The design's ~313M hierarchical model over the Craftax schema."""

        schema: FrameSchema = field(default_factory=craftax_schema)
        """Slot layout and allowed IDs of frames and local jobs."""

        scalar_offset: int = 155
        """ID of scalar value 0: Craftax's ``number_id(0)``; 0 if values are IDs."""

        num_actions: int = 43
        """The game's actions; the action head's width and the action table's rows."""

        encoder: FrameEncoderConfig = field(default_factory=FrameEncoder.Config)
        """Frame encoder of any kind; its width is the shared table's."""

        transformer: GlobalTransformer.Config = field(
            default_factory=GlobalTransformer.Config,
        )
        """Global transformer over start, obs, and act positions."""

        decoder: LocalDecoder.Config = field(default_factory=LocalDecoder.Config)
        """Local decoder; must match the encoder's width."""

        init_weight: InitFn = partial(nn.init.normal_, std=0.02)
        """Init of the tables, the start vector, and the heads."""

        z_loss: float = 1e-5
        """Weight of each modality's mean squared log-normalizer."""

        output_table: Embedding.Config | None = None
        """The table local slots are scored against, with its own init (plan fork
        B4); None scores them against the table that embeds frames."""

        embed_board: BoardEmbedding = gathered_board
        """Embeds each board cell from its field IDs' rows of the table; e.g.
        ``multi_hot_board``, a matmul forward and backward."""

        @override
        def finalize(self) -> Self:
            if self.encoder.num_slots == -1:
                self.encoder.num_slots = self.schema.frame_slots
            if self.decoder.num_slots == -1:
                self.decoder.num_slots = self.schema.local_slots
            if self.output_table is not None:
                if self.output_table.channels_in == -1:
                    self.output_table.channels_in = self.schema.vocab_size
                if self.output_table.channels_out == -1:
                    self.output_table.channels_out = self.encoder.channels_in
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            frames: int,
            jobs: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one packed batch's forward: the three modules, the heads, the loss.

            A packed batch's shape does not fix how many frames and jobs it
            holds, so both are named: ``len(batch.aux)`` and
            ``len(batch.job_at)``. The encoder runs once per frame, the global
            stack over every position, and the decoder once per job against
            its frame's memory. The logits and the loss are float32.

            Args:
              seq_len: Global positions per window, ``t_g``.
              batch_size: Windows in this invocation.
              frames: Frames the encoder embeds and encodes.
              jobs: Local jobs the decoder generates.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every module.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            schema = self.schema
            frame_slots = schema.frame_slots
            return (
                _frame_cost(self, frames=frames, dtype=dtype)
                + cost(self.encoder, batch_size=frames, dtype=dtype, **kwargs)
                + _global_cost(
                    self,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    frames=frames,
                    jobs=jobs,
                    dtype=dtype,
                    **kwargs,
                )
                + self.decoder.cost(
                    batch_size=jobs,
                    dtype=dtype,
                    memory_len=frame_slots,
                    **kwargs,
                )
                + _local_cost(self, frames=frames, jobs=jobs, dtype=dtype)
                + slot_loss_cost(
                    schema,
                    positions=seq_len * batch_size,
                    jobs=jobs,
                    num_actions=self.num_actions,
                )
            )

    def __init__(self, config: Config) -> None:
        schema = config.schema
        if config.encoder.channels_in != config.decoder.channels_in:
            raise ValueError("The encoder and decoder share one table, so one width.")
        if schema.prefix_names not in {(), ("reward", "done")}:
            raise ValueError(f"Unsupported local prefix {schema.prefix_names}.")
        super().__init__()
        self.schema = schema
        self.scalar_offset = config.scalar_offset
        self.z_loss = config.z_loss
        self.embed_board = config.embed_board
        width, width_global = config.encoder.channels_in, config.transformer.channels_in
        init = config.init_weight
        self.table = _embedding(schema.vocab_size, width, init)
        self.output_table = (
            None if config.output_table is None else config.output_table.make()
        )
        self.encoder = config.encoder.make()
        self.transformer = config.transformer.make()
        self.decoder = config.decoder.make()
        self.start = nn.Parameter(torch.empty(width_global))
        self.action_embedding = _embedding(config.num_actions, width_global, init)
        self.obs_proj = _linear(width, width_global, init)
        self.action_head = _linear(width_global, config.num_actions, init)
        self.cond_proj = _linear(width_global, width, init)
        self._init_weight = init
        self.cell_offsets: Tensor
        self.cell_index: Tensor
        self.scalar_rows: Tensor
        self.scalar_allowed: Tensor
        for name, table in schema_tables(self.schema).items():
            self.register_buffer(name, table, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every parameter, and refill the schema tables, in place."""
        for module in (
            self.table,
            self.encoder,
            self.transformer,
            self.decoder,
            self.action_embedding,
            self.obs_proj,
            self.action_head,
            self.cond_proj,
        ):
            module.reset_parameters()
        if self.output_table is not None:
            self.output_table.reset_parameters()
        call_init(self._init_weight, self.start)
        # Meta materialization leaves these as uninitialized integer storage,
        # which its NaN audit cannot see; they are derived from the schema.
        with torch.no_grad():
            for name, table in schema_tables(self.schema).items():
                self.get_buffer(name).copy_(table)

    @override
    def forward(self, batch: PackedBatch) -> ModalityLoss:
        """Return per-modality NLL sums and counts, the objective, and z-loss."""
        return self.loss(batch, self.logits(batch))

    def logits(self, batch: PackedBatch) -> WorldModelLogits:
        """Run the three modules on one packed batch.

        Args:
          batch: The packed micro-batch.

        Returns:
          logits: Action logits per global position and local logits per job.

        """
        slots = self.frame_slots(batch.cells, batch.aux)
        pooled, memory = self.encoder(slots)
        hidden = self.transformer(
            self._global_inputs(batch, pooled),
            positions=batch.pos,
            cu_seqlens=batch.cu_seqlens,
        )
        cond = self.cond_proj(hidden.flatten(0, 1)[batch.job_at.long()])
        local = self.decoder(
            self._local_inputs(batch, slots, cond=cond),
            memory=memory[batch.job_memory.clamp(min=0).long()],
            has_memory=batch.job_memory >= 0,
        )
        return WorldModelLogits(
            action=self.action_head(hidden).float(),
            local=functional.linear(local, self.output_weight).float(),
        )

    @property
    def output_weight(self) -> Tensor:
        """Return the table local logits score against: its own, else the tied one."""
        table = self.table if self.output_table is None else self.output_table
        return table.weight

    def loss(self, batch: PackedBatch, logits: WorldModelLogits) -> ModalityLoss:
        """Score logits against the batch's targets.

        Args:
          batch: The packed micro-batch.
          logits: ``self.logits(batch)``.

        Returns:
          loss: Per-modality sums and counts, the objective, and z-loss.

        """
        return modality_loss(self.target_terms(batch, logits), z_loss=self.z_loss)

    def target_terms(
        self,
        batch: PackedBatch,
        logits: WorldModelLogits,
    ) -> dict[str, tuple[Nll, Tensor]]:
        """Return each modality's per-target NLL and scored mask; ``loss`` sums these.

        ``action`` is aligned to global positions, ``[B, t_g]``: column ``t``
        scores the action at ``t + 1`` from position ``t``; the last column is
        never scored. Every other modality is aligned to jobs, which sit at
        ``batch.job_at``: ``reward`` and ``done`` ``[J]``, ``board``
        ``[J, cell_slots]``, and ``hud`` ``[J, scalars]``. An unscored target's
        NLL is arbitrary, possibly infinite; select with the mask.

        Args:
          batch: The packed micro-batch.
          logits: ``self.logits(batch)``.

        Returns:
          terms: Per modality, the NLL and a bool mask of the same shape.

        """
        return slot_terms(
            batch,
            logits,
            schema=self.schema,
            scalar_offset=self.scalar_offset,
            cell_index=self.cell_index,
            scalar_rows=self.scalar_rows,
            scalar_allowed=self.scalar_allowed,
        )

    def frame_slots(self, cells: Tensor, aux: Tensor) -> Tensor:
        """Embed frames as ``embed_frames`` does, in this model's table.

        Args:
          cells: The game's cell field values ``[..., cell_slots, K]``.
          aux: The game's scalar values ``[..., scalars]``.

        Returns:
          slots: The encoder's input, ``[..., frame_slots, C]``.

        """
        return embed_frames(
            self.table,
            cells,
            aux,
            cell_offsets=self.cell_offsets,
            scalar_offset=self.scalar_offset,
            embed_board=self.embed_board,
        )

    def _global_inputs(self, batch: PackedBatch, pooled: Tensor) -> Tensor:
        """Place the start vector, projected frames, and action embeddings."""
        kind = batch.kind[..., None]
        obs = self.obs_proj(pooled)[batch.frame_of.clamp(min=0).long()]
        act = self.action_embedding(batch.action.long()) * (kind == Kind.ACT)
        inputs = torch.where(kind == Kind.OBS, obs, act)
        return torch.where(kind == Kind.START, self.start, inputs)

    def _local_inputs(
        self,
        batch: PackedBatch,
        slots: Tensor,
        *,
        cond: Tensor,
    ) -> Tensor:
        """Return ``[c, x_0, ..., x_{L-2}]``: position ``j`` predicts local slot ``j``."""
        next_slots = _next(slots, batch)[..., :-1, :]
        prefix = self.table(
            prefix_ids(batch, schema=self.schema, scalar_offset=self.scalar_offset),
        )
        return torch.cat([cond[:, None], prefix, next_slots], dim=-2)


def slot_terms(
    batch: PackedBatch,
    logits: WorldModelLogits,
    *,
    schema: FrameSchema,
    scalar_offset: int,
    cell_index: Tensor,
    scalar_rows: Tensor,
    scalar_allowed: Tensor,
) -> dict[str, tuple[Nll, Tensor]]:
    """Return each modality's per-target NLL and scored mask, as ``WorldModel`` scores.

    The one definition of what a model of the local slots scores, which
    ``WorldModel.target_terms`` and ``flat.FlatModel.target_terms`` share.

    Args:
      batch: The packed micro-batch.
      logits: Action logits per global position and local logits per job.
      schema: The local slot layout.
      scalar_offset: ID of scalar value 0.
      cell_index: ``schema_tables(schema)["cell_index"]``.
      scalar_rows: ``schema_tables(schema)["scalar_rows"]``.
      scalar_allowed: ``schema_tables(schema)["scalar_allowed"]``.

    Returns:
      terms: Per modality, the NLL and a bool mask of the same shape.

    """
    kind = batch.kind
    action = cross_entropy(logits.action, batch.action.roll(-1, dims=-1).long())
    scored_action = functional.pad(
        (kind[:, :-1] == Kind.OBS) & (kind[:, 1:] == Kind.ACT),
        (0, 1),
    )
    prefix = len(schema.prefix_ranges)
    has_next = (batch.job_next >= 0)[:, None]
    is_act = ~batch.job_is_start
    target_scalars = torch.cat(
        [
            prefix_ids(batch, schema=schema, scalar_offset=scalar_offset),
            _next(batch.aux, batch).long() + scalar_offset,
        ],
        dim=-1,
    )
    scalars = scalar_nll(
        logits.local[:, scalar_rows],
        allowed=scalar_allowed,
        target=target_scalars,
    )
    terms: dict[str, tuple[Nll, Tensor]] = {"action": (action, scored_action)}
    if prefix:
        terms["reward"] = (_column(scalars, index=0), is_act)
        terms["done"] = (_column(scalars, index=1), is_act)
    if schema.cell_fields:
        cells = cell_nll(
            logits.local[:, prefix : prefix + schema.cell_slots],
            index_table=cell_index,
            target=_next(batch.cells, batch).long(),
        )
        terms["board"] = (cells, has_next)
    hud = Nll(nll=scalars.nll[:, prefix:], logz_sq=scalars.logz_sq[:, prefix:])
    terms["hud"] = (hud, has_next)
    return {
        name: (value, scored.expand(value.nll.shape))
        for name, (value, scored) in terms.items()
    }


def slot_loss_cost(
    schema: FrameSchema,
    *,
    positions: int,
    jobs: int,
    num_actions: int,
) -> Cost:
    """Cost what :func:`slot_terms` and its loss compute, on float32 logits.

    Each target is one row of priml's ``cross_entropy`` over the IDs its slot
    allows: an action's row is the action head's logits, a single-ID slot's
    is all ``vocab_size`` logits plus a ``-inf`` bias, and a cell field's is
    its padded column of the index table. The per-modality sums and means, a
    few values per target, are left out.

    Args:
      schema: The local slot layout.
      positions: Global positions, each scoring one action.
      jobs: Local jobs, each scoring its prefix and next frame.
      num_actions: The action head's width.

    Returns:
      cost: Whole-invocation FLOPs and logical bytes; it owns nothing.

    """
    fp32, vocab = torch.float32, schema.vocab_size
    single = jobs * (len(schema.prefix_ranges) + len(schema.scalar_ranges))
    terms = _cross_entropy_rows(rows=positions, classes=num_actions)
    terms += (
        gather_cost(
            rows=single,
            width=vocab,
            source=jobs * schema.local_slots,
            dtype=fp32,
        )
        + elementwise_cost(
            primal=single * vocab,
            adjoint=0,
            channels=vocab,
            rows=single,
            inputs=2,
            dtype=fp32,
        )
        + _cross_entropy_rows(rows=single, classes=vocab)
    )
    if not schema.cell_fields:
        return terms
    fields = len(schema.cell_fields)
    columns = max(field.size for field in schema.cell_fields)
    cells = jobs * schema.cell_slots
    padded = concat_cost(elements=cells * (vocab + 1), dtype=fp32)
    picked = gather_cost(
        rows=cells * fields,
        width=columns,
        source=cells * (vocab + 1),
        dtype=fp32,
    )
    summed = reduction_cost(
        input_elements=cells * fields,
        output_groups=cells,
        dtype=fp32,
    )
    scored = _cross_entropy_rows(rows=cells * fields, classes=columns)
    return terms + padded + picked + scored + summed


def embed_frames(
    table: nn.Embedding,
    cells: Tensor,
    aux: Tensor,
    *,
    cell_offsets: Tensor,
    scalar_offset: int,
    embed_board: BoardEmbedding,
) -> Tensor:
    """Embed frames: each cell from its field IDs' rows, then one ID per scalar.

    Args:
      table: The token table.
      cells: The game's cell field values ``[..., cell_slots, K]``.
      aux: The game's scalar values ``[..., scalars]``.
      cell_offsets: ``schema_tables(schema)["cell_offsets"]``.
      scalar_offset: ID of scalar value 0.
      embed_board: Embeds the cells from their IDs' rows of ``table``.

    Returns:
      slots: ``[..., frame_slots, C]``.

    """
    board = embed_board(table, cells.long() + cell_offsets)
    hud = table(aux.long() + scalar_offset)
    return torch.cat([board, hud], dim=-2)


def keep_attention(
    ctx: SelectiveCheckpointContext,
    op: OpOverload,
    *args: object,
    **kwargs: object,
) -> CheckpointPolicy:
    """Store what a fused attention kernel returns; recompute everything else.

    A ``RecomputePolicy``: a block recomputed under it reruns its matmuls and
    pointwise ops in backward, but not its attention.

    Args:
      ctx: The selective checkpoint's context; unread.
      op: The op whose outputs are to be stored or recomputed.
      *args: The op's arguments; unread.
      **kwargs: The op's keyword arguments; unread.

    Returns:
      policy: ``MUST_SAVE`` for a fused attention kernel, else
        ``PREFER_RECOMPUTE``.

    """
    del ctx, args, kwargs
    fused = {
        "aten._scaled_dot_product_cudnn_attention.default",
        "aten._scaled_dot_product_flash_attention.default",
        "aten._scaled_dot_product_efficient_attention.default",
        "aten._scaled_dot_product_flash_attention_for_cpu.default",
    }
    if str(op) in fused:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


# Only under autocast: inference loads the float32 weights and runs without it,
# and a bfloat16 stream would not match them.
def to_autocast_dtype(x: Tensor) -> Tensor:
    """Return ``x`` in its device's autocast dtype when autocast is on."""
    device = x.device.type
    if torch.is_autocast_enabled(device):
        return x.to(torch.get_autocast_dtype(device))
    return x


def schema_tables(schema: FrameSchema) -> dict[str, Tensor]:
    """Return the schema-derived index tables a model's loss and embedding read.

    Args:
      schema: The local slot layout.

    Returns:
      tables: ``cell_offsets``, ``cell_index``, ``scalar_rows``, and
        ``scalar_allowed``.

    """
    prefix = len(schema.prefix_ranges)
    scalar_rows = torch.tensor(
        [*range(prefix), *range(prefix + schema.cell_slots, schema.local_slots)],
        dtype=torch.long,
    )
    index = schema.cell_index_table() if schema.cell_fields else torch.empty(0, 0)
    return {
        "cell_offsets": torch.tensor(
            [f.offset for f in schema.cell_fields],
            dtype=torch.long,
        ),
        "cell_index": index.long(),
        "scalar_rows": scalar_rows,
        "scalar_allowed": schema.local_allowed()[scalar_rows],
    }


def prefix_ids(
    batch: PackedBatch,
    *,
    schema: FrameSchema,
    scalar_offset: int,
) -> Tensor:
    """Return reward and done IDs per job; a ``start`` job holds 0 and False.

    Args:
      batch: The packed micro-batch.
      schema: The local slot layout; without a prefix there are no IDs.
      scalar_offset: ID of scalar value 0.

    Returns:
      ids: ``[J, 2]``, or ``[J, 0]`` without a prefix.

    """
    if not schema.prefix_ranges:
        return batch.job_at.new_zeros(len(batch.job_at), 0).long()
    is_start = batch.job_is_start
    reward = torch.where(is_start, 0, batch.job_reward.long()) + scalar_offset
    done = (batch.job_done & ~is_start).long() + schema.prefix_ranges[1][0]
    return torch.stack([reward, done], dim=-1)


def _next(frames: Tensor, batch: PackedBatch) -> Tensor:
    """Gather each job's next frame; a job without one reads frame 0, unscored."""
    return frames[batch.job_next.clamp(min=0).long()]


def _recompute(
    function: Callable[..., Tensor],
    *args: Tensor,
    policy: RecomputePolicy | None,
) -> Tensor:
    """Run ``function`` storing its inputs, and what ``policy`` keeps; recompute it."""
    if policy is None:
        return checkpoint(function, *args, use_reentrant=False)
    return checkpoint(
        function,
        *args,
        use_reentrant=False,
        context_fn=partial(create_selective_checkpoint_contexts, policy),
    )


def _frame_cost(
    config: WorldModel.Config,
    *,
    frames: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost embedding frames: cells through ``embed_board``, scalars one ID each."""
    schema = config.schema
    width = config.encoder.channels_in
    board = cost(
        config.embed_board,
        rows=frames * schema.cell_slots,
        fields=len(schema.cell_fields),
        vocab=schema.vocab_size,
        width=width,
        dtype=dtype,
    )
    # The scalar lookup owns the table every other lookup of it reads.
    table = Embedding.Config(channels_in=schema.vocab_size, channels_out=width)
    scalars = table.cost(
        seq_len=schema.frame_slots - schema.cell_slots,
        batch_size=frames,
        dtype=dtype,
    )
    slots = concat_cost(elements=frames * schema.frame_slots * width, dtype=dtype)
    return board + scalars + slots


# The action rows are masked by one multiply; two selects place the projected frames and
# the start vector, whose gradient sums over every position.
def _global_cost(
    config: WorldModel.Config,
    *,
    seq_len: int,
    batch_size: int,
    frames: int,
    jobs: int,
    dtype: torch.dtype | None,
    **kwargs: object,
) -> Cost:
    """Cost the global inputs, the stack, the action head, and the jobs' conditions."""
    width, wide = config.encoder.channels_in, config.transformer.channels_in
    positions = seq_len * batch_size
    actions = Embedding.Config(channels_in=config.num_actions, channels_out=wide)
    placed = elementwise_cost(
        primal=positions * wide,
        adjoint=positions * wide,
        channels=wide,
        rows=positions,
        inputs=2,
        dtype=dtype,
    ) + elementwise_cost(
        primal=0,
        adjoint=0,
        channels=wide,
        params=wide,
        rows=positions,
        inputs=2,
        adjoint_inputs=1,
        adjoint_outputs=2,
        dtype=dtype,
    )
    inputs = (
        matmul_cost(channels_in=width, channels_out=wide, rows=frames, dtype=dtype)
        + gather_cost(rows=positions, width=wide, source=frames, dtype=dtype)
        + actions.cost(seq_len=seq_len, batch_size=batch_size, dtype=dtype)
        + placed
    )
    conditions = gather_cost(
        rows=jobs,
        width=wide,
        source=positions,
        dtype=dtype,
    ) + matmul_cost(channels_in=wide, channels_out=width, rows=jobs, dtype=dtype)
    return (
        inputs
        + cost(
            config.transformer,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
            **kwargs,
        )
        + logits_cost(
            channels_in=wide,
            channels_out=config.num_actions,
            rows=positions,
            weight=True,
            dtype=dtype,
        )
        + conditions
    )


# A job's inputs are its condition, its prefix IDs' rows of the table, and its next
# frame's slots but the last; its memory is its frame's encoder output. The logits score
# against the tied table, or ``output_table``.
def _local_cost(
    config: WorldModel.Config,
    *,
    frames: int,
    jobs: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost the jobs' inputs and memory, gathered per job, and the local logits."""
    schema = config.schema
    width = config.encoder.channels_in
    frame = schema.frame_slots * width
    table = Embedding.Config(channels_in=schema.vocab_size, channels_out=width)
    prefix = table.cost(
        seq_len=len(schema.prefix_ranges),
        batch_size=jobs,
        dtype=dtype,
    ).tile(1, copies=0)
    gathered = gather_cost(rows=jobs, width=frame, source=frames, dtype=dtype)
    return (
        gathered.tile(2)
        + prefix
        + concat_cost(elements=jobs * schema.local_slots * width, dtype=dtype)
        + logits_cost(
            channels_in=width,
            channels_out=schema.vocab_size,
            rows=jobs * schema.local_slots,
            weight=config.output_table is not None,
            dtype=dtype,
        )
    )


def _cross_entropy_rows(*, rows: int, classes: int) -> Cost:
    """Cost ``rows`` float32 rows of priml's ``cross_entropy`` over ``classes``."""
    return cost(
        simple_loss.cross_entropy,
        dtype=torch.float32,
        channels_out=classes,
        weighted=False,
        rescale=0,
    ).tile(rows)


def _column(value: Nll, *, index: int) -> Nll:
    """Return one slot's NLL from a per-slot ``Nll``."""
    return Nll(nll=value.nll[:, index], logz_sq=value.logz_sq[:, index])


def _embedding(rows: int, width: int, init: InitFn) -> Embedding:
    """Return an embedding table of ``rows`` x ``width``."""
    return Embedding.Config(
        channels_out=width,
        channels_in=rows,
        init_weight=init,
    ).make()


def _linear(channels_in: int, channels_out: int, init: InitFn) -> Linear:
    """Return a bias-free linear map."""
    return Linear.Config(
        channels_in=channels_in,
        channels_out=channels_out,
        init_weight=init,
    ).make()


def _norm() -> RMSNorm.Config:
    """Return the shared norm: RMSNorm with a learned scale and eps 1e-6."""
    return RMSNorm.Config(elementwise_affine=True)


def _swiglu(channels_hidden: int) -> SwiGLU.Config:
    """Return a bias-free SwiGLU whose matrices draw from N(0, 0.02)."""
    init = partial(nn.init.normal_, std=0.02)
    return SwiGLU.Config(
        channels_hidden=channels_hidden,
        init_weight=init,
        init_weight_out=init,
    )


def _local_attention(*, causal: bool) -> Attention.Config:
    """Return 8-head SDPA self-attention with QK-norm and no position encoding."""
    return Attention.Config(
        num_heads=8,
        causal=causal,
        share_qk_norm=False,
        norm_qk=_norm(),
        init_weight=partial(nn.init.normal_, std=0.02),
    )


def _local_stack(block: ChannelsInOutConfig) -> Transformer.Config:
    """Return a 4-layer stack of ``block`` ending in a norm; width is pushed later."""
    stack = Transformer.Config(num_layers=4, proj_out=_norm())
    stack.block = block
    return stack


def _global_block(*, channels_head: int) -> TransformerBlock.Config:
    """Return the Qwen3 block: GQA 9 query over 3 KV heads, RoPE base 500,000."""
    # -1: ``GlobalTransformer.Config`` sets the head width the block ends up with,
    # which an experiment or override may change after this; a rotary given its
    # own width keeps it.
    rope = RoPE.Config(channels_head=-1)
    rope.frequencies = HuggingFaceFrequencies.Config(base=500_000.0)
    attn = VarlenAttention.Config(
        num_heads=9,
        num_heads_kv=3,
        channels_head=channels_head,
        share_qk_norm=False,
        norm_qk=_norm(),
        rope=rope,
        init_weight=partial(nn.init.normal_, std=0.02),
    )
    return TransformerBlock.Config(
        attn=attn,
        ffn=_swiglu(3_072),
        norm1=_norm(),
        norm2=_norm(),
    )
