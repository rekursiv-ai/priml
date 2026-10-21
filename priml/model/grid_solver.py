"""Grid-puzzle solver: an embedding, a block stack, and an optional recurrence.

The network reads a token grid and predicts a token grid. Three things vary
independently, so each is a slot rather than a flag:

* ``embedding`` -- what signals are added to the input (see
  :mod:`priml.model.grid_embedding`).
* ``block`` -- how tokens mix. Any module accepting ``(x, *args, **kwargs)``
  works; :class:`~priml.model.transformer.block.TransformerBlock` and
  :class:`~priml.model.mlpmixer.MLPMixerBlock` both do.
* ``recurrence`` -- ``None`` uses one fast and one slow latent update; a
  :class:`priml.model.recurrence.Recurrence` runs it many times over a carried latent state, which is
  what makes the model a Tiny Recursive Model.

Without a recurrence the core makes two stack passes, refining the fast latent
once and then the slow latent once. With one, a forward becomes ``slow_cycles``
applications of a core that refines two latent
states, gradient flowing only through the last -- so a fixed parameter budget
buys more computation per puzzle. The two share every other component, which is
why the comparison is a config delta rather than a second model.

The output head predicts a token per grid cell; a second, small head emits a
scalar per puzzle used by adaptive-computation-time schemes to decide whether
to keep thinking. That head exists whether or not a recurrence is attached; it
is simply unused without one.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import field
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    Self,
    cast,
    override,
    runtime_checkable,
)

import copy
import functools
import logging
import math

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    nbytes,
    traffic,
)
from priml.math.linear import cast_parameters_to_input
from priml.model.attention.attention import Attention
from priml.model.attention.rope import RoPE, rotation_cost
from priml.model.conv import Conv1d, SequenceConv1d
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    CoreFn,
    CoreOutput,
    ForwardOutput,
    HasResetParameters,
    MixFn,
    TensorModule,
)
from priml.model.embedding import Embedding
from priml.model.grid_embedding import GridEmbedding, PredictionFeedback
from priml.model.init import (
    InitFn,
    corrected_fan_in_normal,
    corrected_unit_normal,
)
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.prefix import ComposedSparsePrefix, PrefixConfig
from priml.model.recurrence import (
    Recurrence,
    RecurrenceConfig,
    SingleLatentRecurrence,
    two_latent_refine,
)
from priml.model.sequential import Sequential
from priml.model.swiglu import SwiGLU
from priml.model.transformer.block import TransformerBlock


if TYPE_CHECKING:
    from collections.abc import Generator


logger = logging.getLogger(__name__)


type CompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
"""The ``torch.compile`` modes a core compile accepts."""


class CoreCompile:
    """Compile the recurrence's hot loop with ``torch.compile``.

    Bound once at construction, before any data-parallel wrap, so every rank
    traces the same graph at the same point. ``unit`` picks the granularity:
    the whole core application, or one pass of the block stack -- far faster
    to trace for a deep inner loop, at a small runtime cost.
    """

    class Config(Fig["CoreCompile"]):
        """What to trace, and how."""

        unit: Literal["core", "reasoning"] = "core"
        """``core`` traces one core application; ``reasoning`` one block pass."""

        mode: CompileMode = "default"
        """``torch.compile`` mode."""

        fullgraph: bool = True
        """Refuse graph breaks. ``False`` tolerates them -- the lever when a
        strict single-graph trace of a deep recurrence stalls."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __call__[FnT: Callable[..., object]](self, fn: FnT) -> FnT:
        """Return ``fn`` compiled under this config."""
        return cast(
            FnT,
            torch.compile(fn, mode=self.config.mode, fullgraph=self.config.fullgraph),
        )


class GridConfig(Makeable[GridEmbedding], Protocol):
    """A config that builds a grid embedding and declares the grid it embeds.

    The model reads the puzzle's cell count from here: the token head emits
    one row per cell and the loss strips the prefix down to this many. A
    replacement embedding must say its grid; the widths are the model's to
    push down.
    """

    channels_in: int
    channels_out: int

    @property
    def grid_len(self) -> int:
        """Grid tokens per puzzle."""
        ...


@runtime_checkable
class RotaryFactors(Protocol):
    """A block whose rotary factors the model may precompute once per forward.

    Every block in the stack sees the same positions, so the first block's
    factors serve them all.
    """

    def factors(
        self,
        seq_len: int,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor] | None:
        """Return ``(cos, sin)`` for ``seq_len`` positions, or ``None``."""
        ...


def lattice_positions(
    seq_len: int,
    *,
    grid_shape: tuple[int, ...],
    device: torch.device,
) -> Tensor:
    """Place grid tokens on an N-D lattice after a run of prefix tokens.

    Grid tokens get their lattice coordinates, shifted along the first axis
    by the prefix length; prefix tokens count up along the last axis with
    every other coordinate zero. So no prefix token shares a position with a
    grid token. A one-axis grid degenerates to ``arange(seq_len)``.

    Args:
      seq_len: Prefix plus grid tokens; the leading tokens are the prefix.
      grid_shape: Grid extent per axis.
      device: Device to build the positions on.

    Returns:
      positions: ``[seq_len]`` below two axes, else ``[seq_len, len(grid_shape)]``.

    """
    if len(grid_shape) < 2:
        return torch.arange(seq_len, device=device)
    mesh = torch.meshgrid(
        *[torch.arange(extent, device=device) for extent in grid_shape],
        indexing="ij",
    )
    grid = torch.stack(mesh, dim=-1).reshape(-1, len(grid_shape))
    prefix_len = seq_len - grid.shape[0]
    grid[:, 0] += prefix_len
    prefix = torch.zeros(prefix_len, len(grid_shape), dtype=torch.long, device=device)
    prefix[:, -1] = torch.arange(prefix_len, device=device)
    return torch.cat([prefix, grid])


class GridSolver(nn.Module):
    """Grid-puzzle solver over an injected embedding, block stack, and recurrence.

    See the module docstring for what each slot varies. Without a recurrence the
    core refines the fast latent and then the slow latent; with one it is however
    many passes that recurrence prescribes, and the caller carries ``z_slow`` / ``z_fast``
    between calls to build an adaptive-computation-time rollout.
    """

    class Config(Fig["GridSolver"]):
        """Width, depth, and the three injected slots."""

        channels_in: int = 512
        """Token embedding and latent-state width."""

        num_layers: int = 2
        """Blocks in the reasoning stack, applied per core application."""

        embedding: GridConfig = field(default_factory=GridEmbedding.Config)
        """Input embedding: tokens plus whatever additive channels apply."""

        block: Makeable[TensorModule] = field(
            default_factory=lambda: TransformerBlock.Config(prenorm=False),
        )
        """Token-mixing block, repeated ``num_layers`` times.

        Any module taking ``(x, *args, **kwargs)`` works; the transformer and
        MLP-mixer blocks in priml both do, which is what makes the architecture
        comparison a value rather than a branch.

        POST-norm, against priml's pre-norm default, because a recurrence feeds
        the stack its own output: pre-norm leaves the residual stream
        unnormalized, which is harmless in one pass and compounds when the
        output is fed back. Measured at hidden 32 over 5 ACT steps, pre-norm
        drove the carried latent to 413.6 and the loss to 4473, while post-norm
        held the latent at 2.9 and the loss fell monotonically."""

        recurrence: RecurrenceConfig | None = None
        """Latent-refinement schedule. ``None`` refines fast, then slow, with two
        sequential block-stack passes."""

        prefix: PrefixConfig | None = None
        """Optional module producing ``[B, P, C]`` tokens prepended to the grid.

        A per-puzzle embedding lives here. The halt readout reads position 0, so
        a prefix also gives that readout a dedicated token."""

        num_prefix_tokens: int = -1
        """Tokens the prefix contributes; sizes the latent state.

        ``-1`` reads them from the prefix module itself, so the two cannot
        disagree -- a hand-set count that undercounts silently strips real grid
        logits, and one that overcounts strips nothing and shifts every
        position. No prefix means 0."""

        vocab_size: int = -1
        """Output vocabulary; must match the embedding's."""

        head: Linear.Config = field(
            default_factory=lambda: Linear.Config(init_weight=corrected_fan_in_normal),
        )
        """Token head; its widths are the model's to push down."""

        init_latent: InitFn = corrected_unit_normal
        """Initializer of the ``[1, C]`` learned starting latents."""

        halt_outputs: int = 2
        """Halt-head width. Only index 0 is read; a second column exists in
        reference checkpoints and is kept for weight-shape compatibility."""

        halt_init_bias: float = -5.0
        """Initial halt-head bias. Strongly negative so a fresh model does not
        halt on its first step before learning anything."""

        dtype: torch.dtype | None = None
        """Storage dtype every parameter and buffer is cast to once built.

        ``None`` keeps float32 masters, which an autocast forward reads at
        compute precision; ``torch.bfloat16`` trains the weights themselves in
        bfloat16."""

        rope: RoPE.Config | None = None
        """Rotary positions shared by every block; ``None`` leaves them to
        the blocks.

        Owned here rather than per block because a rotary with LEARNED or
        randomly initialized frequencies (``RoPEMixed``) is one table the
        whole stack shares, drawn once, after the prefix and before the
        initial latents."""

        rope_grid_shape: tuple[int, ...] = ()
        """Lattice the grid tokens sit on for ``rope``; fewer than two axes
        number the whole sequence in order. Its product is the grid length."""

        compile_core: CoreCompile.Config | None = None
        """Compile the recurrence's hot loop; ``None`` runs eager.

        Only the TRAINING forward uses it: evaluation stays eager unless
        :meth:`GridSolver.eager` is told otherwise, because a compiled bfloat16
        autocast graph elides the casts autocast inserts and a long carried
        rollout compounds the difference."""

        dtype_latent_init: torch.dtype | None = None
        """Dtype the learned initial latents are DRAWN in; ``None`` is float32.

        Separate from ``dtype`` because the draw itself runs at this width: a
        bfloat16 truncated normal is not a float32 one rounded, so a recipe
        that drew its latents in half precision must say so to reproduce them."""

        checkpoint_core: bool = False
        """Activation-checkpoint every grad-bearing core application.

        Wraps the core in ``torch.utils.checkpoint`` (non-reentrant) from
        OUTSIDE the compiled unit, so its inner fast-cycle activations are
        recomputed in backward while the compiled core is unchanged. Only
        applications under gradient are wrapped; a recurrence's ``no_grad``
        cycles and evaluation run the core directly."""

        block_checkpoint_fraction: float = 0.0
        """Fraction of the stack's blocks built with ``checkpoint=True``.

        An evenly spaced ``ceil(num_layers * fraction)`` of the copies are
        checkpointed inside the core; needs a block config with a
        ``checkpoint`` field. Prefer :attr:`checkpoint_core` for a recurrent
        model: in-graph recompute regions can blow up compile time."""

        @property
        def grid_len(self) -> int:
            """Grid tokens per puzzle, read from the embedding."""
            return self.embedding.grid_len

        @property
        def total_seq_len(self) -> int:
            """Prefix plus grid tokens: the latent state's sequence length.

            Counts the prefix directly while ``num_prefix_tokens`` still holds
            its sentinel, so a parent reading this during ITS finalize -- which
            runs before this config's -- gets the real length instead of one
            short by the whole prefix.
            """
            prefix = (
                _count_prefix_tokens(self.prefix)
                if self.num_prefix_tokens == -1
                else self.num_prefix_tokens
            )
            return prefix + self.grid_len

        @override
        def finalize(self) -> Self:
            if self.embedding.channels_out == -1:
                self.embedding.channels_out = self.channels_in
            if self.embedding.channels_in == -1:
                self.embedding.channels_in = self.vocab_size
            if self.head.channels_in == -1:
                self.head.channels_in = self.channels_in
            if self.head.channels_out == -1:
                self.head.channels_out = self.vocab_size
            propagate = self.block
            if isinstance(propagate, ChannelsIn) and propagate.channels_in == -1:
                propagate.channels_in = self.channels_in
            if isinstance(self.prefix, ChannelsOut) and self.prefix.channels_out == -1:
                self.prefix.channels_out = self.channels_in  # pyrefly: ignore[attribute-assign-on-never] -- pyrefly narrows isinstance(Optional, Protocol) to Never.
            if self.num_prefix_tokens == -1:
                self.num_prefix_tokens = _count_prefix_tokens(self.prefix)
            return super().finalize()

        def cost(self, *, batch_size: int, dtype: torch.dtype | None) -> Cost:
            """Cost one complete invocation over the supplied puzzles.

            Args:
              batch_size: Puzzles in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            config = self.copy_tree().finalize()
            recurrence = config.recurrence
            return _grid_solver_cost(
                config,
                batch_size=batch_size,
                dtype=dtype,
                outer_cycles=1 if recurrence is None else recurrence.slow_cycles,
                forward_passes=2 if recurrence is None else recurrence.forward_passes,
                grad_passes=2 if recurrence is None else recurrence.grad_passes,
                forward_additions=(
                    3 if recurrence is None else recurrence.forward_additions
                ),
                grad_additions=(3 if recurrence is None else recurrence.grad_additions),
                halt_head_config=_halt_head(config),
                latent_dtype=(
                    config.dtype or config.dtype_latent_init or torch.float32
                ),
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        _validate_grid_solver_config(config)
        self.config = config
        # Registered first, filled after the blocks: parameter ORDER follows
        # registration, and the reference TRM held its prefix parameters directly,
        # so they led. A norm over the body sums in that order, and a compiled float32
        # reduction lands a different last bit when the order changes.
        self.register_module("prefix", None)
        # Construction order fixes the global-RNG draw order, so a seeded init
        # is reproducible: embedding -> head -> blocks -> latent inits. The halt
        # head draws nothing (zeros and a constant bias).
        embedding = config.embedding.make()
        self.embedding = embedding

        self.head = config.head.make()
        self.halt_head = _halt_head(config).make()

        # ``repeat`` builds independent copies, each finalized separately, so
        # every block draws its own weights in stack order.
        self.reasoning = Sequential.Config(
            elements=_checkpointed_blocks(config),
        ).make()

        self.prefix = config.prefix.make() if config.prefix is not None else None
        # After the prefix, before the latent inits: RoPEMixed draws its
        # per-head directions from the global RNG at construction.
        self.rope = None if config.rope is None else config.rope.make()

        self.slow_init = nn.Buffer(_latent_init(config), persistent=True)
        self.fast_init = nn.Buffer(_latent_init(config), persistent=True)

        self.recurrence: Recurrence | None = (
            config.recurrence.make() if config.recurrence is not None else None
        )
        self._dummy = nn.Buffer(persistent=True)
        # A compiled callable, not a module: registering it would add
        # ``_orig_mod`` keys to the state dict.
        self._compiled_core: CoreFn | None = None
        self._compiled_mix: MixFn | None = None
        self.compiled = config.compile_core is not None
        if config.compile_core is not None:
            compile_fn = config.compile_core.make()
            if config.compile_core.unit == "core":
                self._compiled_core = compile_fn(self._core)
            else:
                self._compiled_mix = compile_fn(self._mix_eager)
        if config.dtype is not None:
            self.to(dtype=config.dtype)
        logger.info(
            "model parameters: %.2fM",
            sum(parameter.numel() for parameter in self.parameters()) / 1e6,
        )

    def reset_parameters(self) -> None:
        """Reset children in source construction order, then restore source halt state."""
        self.embedding.reset_parameters()
        self.head.reset_parameters()
        self.halt_head.reset_parameters()
        with torch.no_grad():
            self.halt_head.weight.zero_()
            if self.halt_head.bias is None:
                raise ValueError("Expected halt_head.bias to be present.")
            self.halt_head.bias.fill_(self.config.halt_init_bias)
        for block in self.reasoning:
            if isinstance(block, HasResetParameters):
                block.reset_parameters()
        if isinstance(self.prefix, HasResetParameters):
            self.prefix.reset_parameters()
        if isinstance(self.rope, HasResetParameters):
            self.rope.reset_parameters()
        slow = _latent_init(self.config)
        fast = _latent_init(self.config)
        if self.config.dtype is not None:
            slow = slow.to(dtype=self.config.dtype)
            fast = fast.to(dtype=self.config.dtype)
        with torch.no_grad():
            self.slow_init.copy_(slow)
            self.fast_init.copy_(fast)

    @property
    def device(self) -> torch.device:
        """Device the model's buffers live on."""
        return self._dummy.device

    def init_latents(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Return the initial ``(z_slow, z_fast)`` for a batch.

        Args:
          batch_size: Puzzles in the batch.

        Returns:
          z_slow: ``[B, S, C]`` slow latent, the init vector at every position.
          z_fast: ``[B, S, C]`` fast latent, likewise.

        """
        s = self.config.total_seq_len
        z_slow = self.slow_init[0].expand(batch_size, s, -1).contiguous()
        z_fast = self.fast_init[0].expand(batch_size, s, -1).contiguous()
        return z_slow, z_fast

    def set_feedback(self, grid: Tensor | None) -> None:
        """Hand ``grid`` to every feedback channel for the next forward."""
        for channel in self.embedding.channels:
            if isinstance(channel, PredictionFeedback):
                channel.set_feedback(grid)

    @contextmanager
    def eager(self, *, enabled: bool = True) -> Generator[None]:
        """Run the compiled paths eagerly for the duration.

        Args:
          enabled: Whether to force eager; ``False`` is a no-op, which lets a
            caller keep one ``with`` for both policies.

        Yields:
          context: Block in which no compiled graph runs.

        """
        previous = self.compiled
        self.compiled = previous and not enabled
        try:
            yield
        finally:
            self.compiled = previous

    def core(
        self,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> CoreOutput:
        """Apply one core call, refining both latent states.

        Args:
          input_emb: ``[B, S, C]`` prefix-prepended input embedding.
          z_slow: ``[B, S, C]`` slow latent.
          z_fast: ``[B, S, C]`` fast latent.
          cos_sin: Optional rotary ``(cos, sin)`` pair for the blocks.

        Returns:
          out: Logits, halt logit, and both updated latents.

        """
        run = (
            self._compiled_core
            if self.compiled and self._compiled_core is not None
            else self._core
        )
        # Checkpointing saves memory only by recomputing in backward, and
        # wrapping a compiled core under ``inference_mode`` deadlocks multi-rank
        # evaluation, so only grad-bearing applications are wrapped.
        if self.config.checkpoint_core and torch.is_grad_enabled():
            return torch_checkpoint(
                run,
                input_emb,
                z_slow,
                z_fast,
                cos_sin,
                use_reentrant=False,
            )
        return run(input_emb, z_slow, z_fast, cos_sin)

    def _core(
        self,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> CoreOutput:
        """Refine both latent states, then read the grid and halt heads."""
        if self.recurrence is None:
            z_slow, z_fast = two_latent_refine(
                self._mix,
                input_emb,
                z_slow,
                z_fast,
                cos_sin,
                fast_cycles=1,
            )
        else:
            z_slow, z_fast = self.recurrence.refine(
                self._mix,
                input_emb,
                z_slow,
                z_fast,
                cos_sin,
            )
        logits = self.head(z_slow)
        halt_logits = self.halt_head(z_slow[:, 0]).to(torch.float32)
        halt = halt_logits[..., 0]
        return CoreOutput(logits, halt, z_slow, z_fast)

    @override
    def forward(
        self,
        tokens: Tensor,
        z_slow: Tensor | None = None,
        z_fast: Tensor | None = None,
        *,
        collect_intermediates: bool = False,
        **prefix_kwargs: object,
    ) -> ForwardOutput:
        """Embed, run the core (once or recurrently), and strip the prefix.

        Args:
          tokens: ``[B, grid_len]`` input token ids.
          z_slow: Carried slow latent; ``None`` starts from the init vector.
          z_fast: Carried fast latent; ``None`` starts from the init vector.
          collect_intermediates: Also return each cycle's logits.
          **prefix_kwargs: Forwarded to the prefix module, if any.

        Returns:
          out: Grid logits with prefix tokens stripped, the halt logit, and
            both latents detached for carrying into the next step.

        """
        input_emb = self._embed(tokens, prefix_kwargs)
        if z_slow is None or z_fast is None:
            z_slow, z_fast = self.init_latents(tokens.shape[0])
        cos_sin = self._cos_sin(input_emb)
        if self.recurrence is None:
            out = self.core(input_emb, z_slow, z_fast, cos_sin)
            result = ForwardOutput(
                out.logits,
                out.halt,
                out.z_slow,
                out.z_fast,
                (out.logits.detach(),) if collect_intermediates else (),
            )
        else:
            result = self.recurrence.forward(
                self.core,
                input_emb,
                z_slow,
                z_fast,
                cos_sin,
                collect_intermediates=collect_intermediates,
            )
        return self._strip_prefix(result)

    def step(
        self,
        tokens: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        **prefix_kwargs: object,
    ) -> ForwardOutput:
        """Apply the core exactly once, whatever the recurrence prescribes.

        One slow cycle, not a forward: a rollout of these runs the latents in
        a regime a model trained on :meth:`forward` never saw, so this is a
        probe of the core, not an evaluation step.

        Args:
          tokens: ``[B, grid_len]`` input token ids.
          z_slow: Carried slow latent.
          z_fast: Carried fast latent.
          **prefix_kwargs: Forwarded to the prefix module, if any.

        Returns:
          out: Grid logits with prefix tokens stripped, the halt logit, and
            both latents detached.

        """
        input_emb = self._embed(tokens, prefix_kwargs)
        out = self.core(input_emb, z_slow, z_fast, self._cos_sin(input_emb))
        return self._strip_prefix(
            ForwardOutput(out.logits, out.halt, out.z_slow, out.z_fast),
        )

    def _strip_prefix(self, result: ForwardOutput) -> ForwardOutput:
        """Drop prefix positions from the logits and detach the latents."""
        n_prefix = self.config.num_prefix_tokens
        return ForwardOutput(
            result.logits[:, n_prefix:] if n_prefix else result.logits,
            result.halt,
            result.z_slow.detach(),
            result.z_fast.detach(),
            tuple(lg[:, n_prefix:] for lg in result.all_logits)
            if n_prefix
            else result.all_logits,
        )

    def _embed(self, tokens: Tensor, prefix_kwargs: dict[str, object]) -> Tensor:
        """Embed the grid and prepend the prefix module's tokens, if any."""
        embeddings = self.embedding(tokens)
        if self.prefix is None:
            return embeddings
        # The prefix is per-batch, not per-token, so it takes the row count and
        # whatever the batch carries (puzzle ids, say) rather than the grid.
        prefix = cast(Tensor, self.prefix(tokens.shape[0], **prefix_kwargs))
        return torch.cat([prefix.to(dtype=embeddings.dtype), embeddings], dim=1)

    def _cos_sin(self, input_emb: Tensor) -> tuple[Tensor, Tensor] | None:
        """Rotary factors shared by every block, or ``None`` if none need them."""
        seq_len, device = input_emb.shape[-2], input_emb.device
        if self.rope is not None:
            positions = lattice_positions(
                seq_len,
                grid_shape=self.config.rope_grid_shape,
                device=device,
            )
            cos, sin = self.rope(positions)
            return cos, sin
        first = self.reasoning[0]
        if not isinstance(first, RotaryFactors):
            return None
        return first.factors(seq_len, device=device)

    def _mix(self, z: Tensor, cos_sin: tuple[Tensor, Tensor] | None) -> Tensor:
        """Run the block stack once over a latent state."""
        if self.compiled and self._compiled_mix is not None:
            return self._compiled_mix(z, cos_sin)
        return self._mix_eager(z, cos_sin)

    def _mix_eager(self, z: Tensor, cos_sin: tuple[Tensor, Tensor] | None) -> Tensor:
        """Run the block stack once, never compiled."""
        return self.reasoning(z, cos_sin=cos_sin)


def _grid_solver_cost(
    config: GridSolver.Config,
    *,
    batch_size: int,
    dtype: torch.dtype | None,
    outer_cycles: int,
    forward_passes: int,
    grad_passes: int,
    forward_additions: int,
    grad_additions: int,
    halt_head_config: Linear.Config,
    latent_dtype: torch.dtype,
) -> Cost:
    """Cost shared solver geometry with its executed phase counts."""
    sequence = config.total_seq_len
    rows = sequence * batch_size
    stack = cost(
        config.block,
        seq_len=sequence,
        batch_size=batch_size,
        dtype=dtype,
    ).tile(config.num_layers, copies=config.num_layers)
    blocks = (
        stack.only("primal").tile(outer_cycles * forward_passes)
        + stack.only("adjoint").tile(grad_passes)
        + Cost(
            params=stack.params,
            params_active=stack.params_active,
        )
    )
    adds = elementwise_cost(
        primal=outer_cycles * forward_additions * rows * config.channels_in,
        adjoint=grad_additions * rows * config.channels_in,
        channels=max(forward_additions, grad_additions) * config.channels_in,
        inputs=2,
        rows=rows,
        dtype=dtype,
    )
    head = cost(
        config.head,
        seq_len=sequence,
        batch_size=batch_size,
        dtype=dtype,
    )
    halt = cost(
        halt_head_config,
        seq_len=1,
        batch_size=batch_size,
        dtype=dtype,
    )
    total = (
        cost(
            config.embedding,
            seq_len=config.grid_len,
            batch_size=batch_size,
            dtype=dtype,
        )
        + blocks
        + adds
        + head.only("primal").tile(outer_cycles)
        + head.only("adjoint")
        + halt.only("primal").tile(outer_cycles)
        + halt.only("adjoint")
        + Cost(
            params=head.params + halt.params,
            params_active=head.params_active + halt.params_active,
        )
    )
    if config.prefix is not None:
        total += cost(
            config.prefix,
            seq_len=1,
            batch_size=batch_size,
            dtype=dtype,
        )
        total += traffic(
            "primal",
            "selection",
            elements=2 * sequence * config.channels_in * batch_size,
            dtype=dtype,
        )
    if config.recurrence is not None:
        total += cost(
            config.recurrence,
            seq_len=sequence,
            batch_size=batch_size,
            dtype=dtype,
        )
    total += Cost(bytes_state=nbytes(2 * config.channels_in, latent_dtype))
    if config.rope is not None:
        total += cost(
            config.rope,
            seq_len=sequence,
            batch_size=1,
            dtype=dtype,
        )
        if isinstance(config.block, TransformerBlock.Config) and isinstance(
            config.block.attn,
            Attention.Config,
        ):
            rotated = rotation_cost(
                config.rope,
                rows=sequence * batch_size * config.num_layers,
                dtype=dtype,
                channels_head=config.block.attn.channels_head,
                heads=(config.block.attn.num_heads + config.block.attn.num_heads_kv),
            )
            total += rotated.only("primal").tile(outer_cycles * forward_passes)
            total += rotated.only("adjoint").tile(grad_passes)
    return total


def _checkpointed_blocks(config: GridSolver.Config) -> list[Makeable[TensorModule]]:
    """Copy the block config per layer, checkpointing an evenly spaced subset."""
    fraction = config.block_checkpoint_fraction
    count = math.ceil(config.num_layers * fraction) if fraction > 0 else 0
    chosen = {math.floor((i + 1) * config.num_layers / count) - 1 for i in range(count)}
    blocks: list[Makeable[TensorModule]] = []
    for layer in range(config.num_layers):
        block = copy.deepcopy(config.block)
        if layer in chosen:
            if not isinstance(block, _CheckpointableConfig):
                raise ValueError(
                    "block_checkpoint_fraction requires a block config with a "
                    "checkpoint field.",
                )
            block.checkpoint = True
        blocks.append(block)
    return blocks


@runtime_checkable
class _CheckpointableConfig(Protocol):
    """A block config whose built block can activation-checkpoint itself."""

    checkpoint: bool


def _halt_head(config: GridSolver.Config) -> Linear.Config:
    """Configure the halt head: zero weights and a constant, strongly negative bias."""
    return Linear.Config(
        channels_in=config.channels_in,
        channels_out=config.halt_outputs,
        bias=True,
        init_weight=nn.init.zeros_,
        init_bias=functools.partial(nn.init.constant_, val=config.halt_init_bias),
    )


# Without a recurrence the core runs once and refines fast, then slow, with two
# sequential stack passes. A recurrence declares its own refinement schedule -- read
# from the config so the core stays a plain method the recurrence can call, and so
# the same numbers cost a forward before anything is built.
# Read from the config rather than by constructing the module: ``finalize`` runs during
# ``pprint`` too, where building a large table would be both slow and surprising.
def _count_prefix_tokens(prefix: PrefixConfig | None) -> int:
    """How many tokens a prefix config contributes, before it is built."""
    return 0 if prefix is None else prefix.num_tokens


def _latent_init(config: GridSolver.Config) -> Tensor:
    """Return a ``[1, C]`` starting latent drawn in ``dtype_latent_init``."""
    w = torch.empty(1, config.channels_in, dtype=config.dtype_latent_init)
    # Called directly: ``call_init`` would draw a bfloat16 latent in float32.
    config.init_latent(w)
    return w


def _single_latent_embedding() -> GridEmbedding.Config:
    config = GridEmbedding.Config()
    config.embed_scale = 1.0
    table = Embedding.Config()
    table.init_weight = corrected_fan_in_normal
    table.lookup_dtype = torch.bfloat16
    config.token_table = table
    return config


def _single_latent_head() -> Linear.Config:
    return Linear.Config(
        init_weight=corrected_fan_in_normal,
        linear_fn=cast_parameters_to_input,
    )


def _single_latent_halt_head() -> Linear.Config:
    return Linear.Config(
        channels_out=1,
        bias=True,
        init_weight=corrected_fan_in_normal,
        init_bias=nn.init.zeros_,
        linear_fn=cast_parameters_to_input,
    )


def _single_latent_block() -> TransformerBlock.Config:
    block = TransformerBlock.Config(prenorm=False)
    attention = Attention.Config(
        channels_head=64,
        init_weight=corrected_fan_in_normal,
    )
    attention.proj_qkv = Linear.Config(
        bias=False,
        init_weight=corrected_fan_in_normal,
        linear_fn=cast_parameters_to_input,
    )
    attention.proj_out = Linear.Config(
        bias=False,
        init_weight=corrected_fan_in_normal,
        linear_fn=cast_parameters_to_input,
    )
    block.attn = attention
    ffn = SwiGLU.Config(
        expansion=8 / 3,
        round_to=256,
        init_weight=corrected_fan_in_normal,
        init_weight_out=corrected_fan_in_normal,
    )
    ffn.up_proj = Linear.Config(
        bias=False,
        init_weight=corrected_fan_in_normal,
        linear_fn=cast_parameters_to_input,
    )
    ffn.down_proj = Linear.Config(
        bias=False,
        init_weight=corrected_fan_in_normal,
        linear_fn=cast_parameters_to_input,
    )
    ffn.middle = SequenceConv1d.Config(
        conv=Conv1d.Config(
            kernel_size=2,
            padding=1,
            groups=-1,
            bias=True,
            dtype=torch.bfloat16,
            init_bias=None,
        ),
    )
    block.ffn = ffn
    block.norm1 = RMSNorm.Config(eps=1e-5)
    block.norm2 = RMSNorm.Config(eps=1e-5)
    return block


def _single_latent_recurrence() -> RecurrenceConfig:

    return SingleLatentRecurrence.Config(
        slow_cycles=1,
        fast_cycles=12,
        inner_grad_loops=6,
    )


class SingleLatentGridSolver(GridSolver):
    """Grid solver specialization that carries one recurrent latent state."""

    config: Config  # pyrefly: ignore[bad-override-mutable-attribute] -- Narrowed to this class's own config; nothing assigns the base's.

    class Config(Makes["SingleLatentGridSolver"], GridSolver.Config):
        """Single-state architecture with injectable grid and memory children."""

        embedding: GridConfig = field(default_factory=_single_latent_embedding)
        """Grid embedding with source lookup dtype and embedding scale."""

        head: Linear.Config = field(default_factory=_single_latent_head)
        """Token readout with input-dtype parameter casts."""

        halt_head: Linear.Config = field(default_factory=_single_latent_halt_head)
        """Halt readout with the source one-logit initialization recipe."""

        block: Makeable[TensorModule] = field(default_factory=_single_latent_block)
        """Repeated post-norm token-mixing block and depthwise convolution."""

        prefix: PrefixConfig | None = field(
            default_factory=ComposedSparsePrefix.Config,
        )
        """Composed puzzle prefix; ``None`` disables task memory."""

        recurrence: RecurrenceConfig | None = field(
            default_factory=_single_latent_recurrence,
        )
        """Single-state recurrent schedule; may be replaced or disabled."""

        rope: RoPE.Config | None = field(
            default_factory=lambda: RoPE.Config(channels_head=-1),
        )
        """Shared rotary table for the injected block attention geometry."""

        forward_dtype: torch.dtype = torch.bfloat16
        """Dtype for initial latents; it does not set child compute dtypes.

        The fast latent uses this dtype. The slow latent uses
        ``dtype_latent_init`` when set, otherwise this dtype; ``dtype`` then casts
        all stored parameters and buffers. For FP32 latents and storage, set
        ``forward_dtype=torch.float32``, ``dtype_latent_init=torch.float32``, and
        ``dtype=torch.float32``. Child dtypes are independent: FP32 source-style
        settings include ``embedding.token_table.lookup_dtype=torch.float32``,
        ``prefix.forward_dtype=torch.float32``, ``prefix.residual.dtype`` and
        ``prefix.residual.dtype_scale`` set to ``torch.float32``, and
        ``block.ffn.middle.conv.dtype=torch.float32``. Each applies only when that
        child exists and has the named slot."""

        @override
        def finalize(self) -> Self:
            if self.rope is not None and not self.rope_grid_shape:
                self.rope_grid_shape = (self.embedding.grid_len,)
            if (
                self.rope is not None
                and self.rope.channels_head == -1
                and isinstance(self.block, TransformerBlock.Config)
                and isinstance(self.block.attn, Attention.Config)
            ):
                self.rope.channels_head = self.block.attn.channels_head
            if self.halt_head.channels_in == -1:
                self.halt_head.channels_in = self.channels_in
            if self.halt_head.channels_out == -1:
                self.halt_head.channels_out = self.halt_outputs
            return super().finalize()

        @override
        def cost(self, *, batch_size: int, dtype: torch.dtype | None) -> Cost:
            """Cost one complete invocation over the supplied puzzles.

            Args:
              batch_size: Number of grid puzzles in the batch.
              dtype: Activation dtype; ``None`` uses torch's default.

            Returns:
              cost: Whole-model cost for one invocation.

            """
            config = self.copy_tree().finalize()
            recurrence = config.recurrence
            return _grid_solver_cost(
                config,
                batch_size=batch_size,
                dtype=dtype,
                outer_cycles=1 if recurrence is None else recurrence.slow_cycles,
                forward_passes=2 if recurrence is None else recurrence.forward_passes,
                grad_passes=2 if recurrence is None else recurrence.grad_passes,
                forward_additions=(
                    3 if recurrence is None else recurrence.forward_additions
                ),
                grad_additions=(3 if recurrence is None else recurrence.grad_additions),
                halt_head_config=config.halt_head,
                latent_dtype=(
                    config.dtype or config.dtype_latent_init or config.forward_dtype
                ),
            )

    def __init__(self, config: Config) -> None:
        nn.Module.__init__(self)
        _validate_grid_solver_config(config)
        self.config = config
        self.embedding = config.embedding.make()
        self.head = config.head.make()
        self.halt_head = config.halt_head.make()
        with torch.no_grad():
            self.halt_head.weight.zero_()
            if self.halt_head.bias is None:
                raise ValueError("Expected halt_head.bias to be present.")
            self.halt_head.bias.fill_(config.halt_init_bias)
        self.prefix = config.prefix.make() if config.prefix is not None else None
        self.rope = config.rope.make() if config.rope is not None else None
        self.reasoning = Sequential.Config(
            elements=_checkpointed_blocks(config),
        ).make()
        latent_dtype = (
            config.forward_dtype
            if config.dtype_latent_init is None
            else config.dtype_latent_init
        )
        self.slow_init = nn.Buffer(torch.empty(config.channels_in, dtype=latent_dtype))
        config.init_latent(self.slow_init)
        self.fast_init = nn.Buffer(
            torch.zeros(config.channels_in, dtype=config.forward_dtype),
            persistent=False,
        )
        self.recurrence = (
            config.recurrence.make() if config.recurrence is not None else None
        )
        self._dummy = nn.Buffer(persistent=False)
        self._compiled_core: CoreFn | None = None
        self._compiled_mix: MixFn | None = None
        self.compiled = config.compile_core is not None
        if config.compile_core is not None:
            compile_fn = config.compile_core.make()
            if config.compile_core.unit == "core":
                self._compiled_core = compile_fn(self._core)
            else:
                self._compiled_mix = compile_fn(self._mix_eager)
        if config.dtype is not None:
            self.to(dtype=config.dtype)

    @override
    def reset_parameters(self) -> None:
        """Reset source-owned children in their construction order."""
        self.embedding.reset_parameters()
        self.head.reset_parameters()
        self.halt_head.reset_parameters()
        with torch.no_grad():
            self.halt_head.weight.zero_()
            if self.halt_head.bias is None:
                raise ValueError("Expected halt_head.bias to be present.")
            self.halt_head.bias.fill_(self.config.halt_init_bias)
        if isinstance(self.prefix, HasResetParameters):
            self.prefix.reset_parameters()
        if isinstance(self.rope, HasResetParameters):
            self.rope.reset_parameters()
        for block in self.reasoning:
            if isinstance(block, HasResetParameters):
                block.reset_parameters()
        latent_dtype = (
            self.config.forward_dtype
            if self.config.dtype_latent_init is None
            else self.config.dtype_latent_init
        )
        latent = torch.empty(self.config.channels_in, dtype=latent_dtype)
        self.config.init_latent(latent)
        if self.config.dtype is not None:
            latent = latent.to(dtype=self.config.dtype)
        with torch.no_grad():
            self.slow_init.copy_(latent)
            self.fast_init.zero_()

    @override
    def init_latents(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Expand the one learned state and a zero fast-state placeholder."""
        sequence = self.config.total_seq_len
        slow = self.slow_init.expand(batch_size, sequence, -1).contiguous()
        fast = self.fast_init.expand(batch_size, sequence, -1).contiguous()
        return slow, fast

    @override
    def _embed(self, tokens: Tensor, prefix_kwargs: dict[str, object]) -> Tensor:
        """Scale the complete prefix-plus-grid sequence at the model boundary."""
        embedded = self.embedding(tokens)
        if self.prefix is not None:
            prefix = cast(Tensor, self.prefix(tokens.shape[0], **prefix_kwargs))
            embedded = torch.cat([prefix.to(embedded.dtype), embedded], dim=1)
        return float(self.config.channels_in**0.5) * embedded


def _validate_grid_solver_config(config: GridSolver.Config) -> None:
    """Validate geometry and checkpointing shared by both solver variants."""
    if config.vocab_size < 1:
        raise ValueError("GridSolver requires vocab_size from the dataset.")
    if config.embedding.grid_len < 1:
        raise ValueError(
            "GridSolver requires embedding grid_shape from the dataset.",
        )
    if config.rope is not None and not config.rope_grid_shape:
        raise ValueError("GridSolver requires rope_grid_shape from the dataset.")
    if config.rope is not None and math.prod(config.rope_grid_shape) != (
        config.embedding.grid_len
    ):
        raise ValueError(
            f"rope_grid_shape {config.rope_grid_shape} must cover the "
            f"{config.embedding.grid_len}-token grid.",
        )
    if config.block_checkpoint_fraction < 0 or config.block_checkpoint_fraction > 1:
        raise ValueError(
            "block_checkpoint_fraction must be in [0, 1], got "
            f"{config.block_checkpoint_fraction}.",
        )
