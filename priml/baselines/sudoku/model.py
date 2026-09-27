"""Grid-puzzle solver: an embedding, a block stack, and an optional recurrence.

The network reads a token grid and predicts a token grid. Three things vary
independently, so each is a slot rather than a flag:

* ``embedding`` -- what signals are added to the input (see
  :mod:`priml.baselines.sudoku.embedding`).
* ``block`` -- how tokens mix. Any module accepting ``(x, *args, **kwargs)``
  works; :class:`~priml.model.transformer.block.TransformerBlock` and
  :class:`~priml.model.mlpmixer.MLPMixerBlock` both do.
* ``recurrence`` -- ``None`` runs the stack once; a
  :class:`Recurrence` runs it many times over a carried latent state, which is
  what makes the model a Tiny Recursive Model.

Without a recurrence this is a plain encoder: embed, mix, project. With one, a
forward becomes ``slow_cycles`` applications of a core that refines two latent
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
    NamedTuple,
    Protocol,
    Self,
    cast,
    override,
    runtime_checkable,
)

import copy
import functools
import logging

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.baselines.sudoku.embedding import GridEmbedding
from priml.baselines.sudoku.prefix import PrefixConfig
from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    traffic,
)
from priml.model.attention.rope import RoPE
from priml.model.custom_types import ChannelsIn, ChannelsOut, TensorModule
from priml.model.init import truncated_normal
from priml.model.linear import Linear
from priml.model.sequential import Sequential
from priml.model.transformer.block import TransformerBlock


if TYPE_CHECKING:
    from collections.abc import Generator


logger = logging.getLogger(__name__)


def corrected_fan_in_normal(w: Tensor, *, depth: int = -1) -> None:
    """Initialize truncated normal at ``std = 1/sqrt(fan_in)``, variance-corrected.

    The initialization every projection in this baseline uses. ``depth`` is
    accepted and discarded: priml's layers pass it to every ``init_weight`` so
    depth-scaled schemes can use it, and this one does not scale with depth.

    Args:
      w: Tensor to initialize in place.
      depth: Ignored; present for the ``InitFn`` protocol.

    """
    del depth
    truncated_normal(
        w,
        std=w.shape[-1] ** -0.5,
        depth_index=(),
        variance_correction=True,
    )


def fan_in_normal(w: Tensor, *, depth: int = -1) -> None:
    """Initialize truncated normal at ``std = 1/sqrt(fan_in)``, clipped at 2 std.

    The uncorrected sibling of :func:`corrected_fan_in_normal`: the realized
    standard deviation is about 0.88x the request, as torch's own
    ``trunc_normal_`` gives. ``depth`` is accepted and discarded for the same
    reason as there.

    Args:
      w: Tensor to initialize in place.
      depth: Ignored; present for the ``InitFn`` protocol.

    """
    del depth
    truncated_normal(w, std=w.shape[-1] ** -0.5, depth_index=())


class CoreOutput(NamedTuple):
    """One application of the reasoning core."""

    logits: Tensor
    """``[B, S, V]`` token logits over the whole sequence."""

    halt: Tensor
    """``[B]`` halt logit read at the readout position."""

    z_slow: Tensor
    """``[B, S, C]`` updated slow latent."""

    z_fast: Tensor
    """``[B, S, C]`` updated fast latent."""


class ForwardOutput(NamedTuple):
    """One full forward: the final core output plus per-cycle intermediates."""

    logits: Tensor
    """``[B, grid_len, V]`` final logits, prefix tokens stripped."""

    halt: Tensor
    """``[B]`` final halt logit."""

    z_slow: Tensor
    """``[B, S, C]`` final slow latent, detached for carrying."""

    z_fast: Tensor
    """``[B, S, C]`` final fast latent, detached for carrying."""

    all_logits: tuple[Tensor, ...] = ()
    """Per-cycle logits when the caller asked for intermediates."""


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

        mode: Literal[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ] = "default"
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


class CoreFn(Protocol):
    """One application of the reasoning core."""

    def __call__(
        self,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> CoreOutput:
        """Apply to the input."""
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


class MixFn(Protocol):
    """One pass of the block stack over a latent state."""

    def __call__(self, z: Tensor, cos_sin: tuple[Tensor, Tensor] | None) -> Tensor:
        """Apply to the input."""
        ...


@runtime_checkable
class Recurrence(Protocol):
    """Drives repeated applications of a core over carried latent state.

    A recurrence decides HOW MANY times the reasoning core runs per forward,
    which of those applications carry gradient, and how one application
    updates the latents. It owns no parameters -- the core it drives does.
    """

    def refine(
        self,
        mix: MixFn,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:
        """Update ``(z_slow, z_fast)`` once; the readout reads ``z_slow``.

        Args:
          mix: One pass of the model's block stack.
          input_emb: ``[B, S, C]`` prefix-prepended input embedding.
          z_slow: ``[B, S, C]`` slow latent.
          z_fast: ``[B, S, C]`` fast latent.
          cos_sin: Optional rotary pair forwarded to ``mix``.

        Returns:
          z_slow: Updated slow latent, which the heads read.
          z_fast: Updated fast latent.

        """
        ...

    def forward(
        self,
        core: CoreFn,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
        *,
        collect_intermediates: bool,
    ) -> ForwardOutput:
        """Run the core to completion for one forward pass.

        Args:
          core: Core.
          input_emb: Input emb.
          z_slow: Z slow.
          z_fast: Z fast.
          cos_sin: Cos sin.
          collect_intermediates: Collect intermediates.

        Returns:
          result: The ForwardOutput.

        """
        ...


class RecurrenceConfig(Makeable[Recurrence], Protocol):
    """A config that builds a :class:`Recurrence` and declares its cycle counts.

    The model costs and runs the core by these numbers, so a recurrence that
    cannot say how often it runs the core cannot fill the slot.
    """

    slow_cycles: int
    """Core applications per forward; the model costs every one."""

    fast_cycles: int
    """Inner block-stack passes per core application."""


class DeepRecurrence(nn.Module):
    """Refine two latent states over ``slow_cycles`` x ``fast_cycles`` passes.

    Each slow cycle runs the block stack ``fast_cycles`` times over the fast
    latent, then once more to refresh the slow latent. Only the LAST slow cycle
    carries gradient; the rest run under ``no_grad`` on detached inputs. That is
    what makes deep recurrence affordable -- the backward graph is one cycle
    deep regardless of how many cycles ran forward.

    References:
      https://arxiv.org/abs/2510.04871
        Jolicoeur-Martineau. Less is More: Recursive Reasoning with Tiny
        Networks.

    """

    class Config(Fig["DeepRecurrence"]):
        """Cycle counts for the two nested loops."""

        slow_cycles: int = 6
        """Outer iterations per forward; all but the last run without grad."""

        fast_cycles: int = 9
        """Inner iterations refining the fast latent per slow cycle."""

        def cost(self, **kwargs: object) -> Cost:
            """Cost nothing: the schedule owns no weights and does no arithmetic.

            The core it repeats is the model's, which costs every cycle.

            Args:
              **kwargs: The open bus, unread.

            Returns:
              cost: Zero.

            """
            del kwargs
            return Cost()

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.slow_cycles < 1 or config.fast_cycles < 1:
            raise ValueError(
                f"slow_cycles and fast_cycles must be >= 1; got "
                f"{config.slow_cycles} and {config.fast_cycles}.",
            )
        self.config = config

    def refine(
        self,
        mix: MixFn,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:
        """Refine the fast latent ``fast_cycles`` times, then the slow one once.

        Args:
          mix: One pass of the model's block stack.
          input_emb: ``[B, S, C]`` input embedding.
          z_slow: ``[B, S, C]`` slow latent.
          z_fast: ``[B, S, C]`` fast latent.
          cos_sin: Optional rotary pair forwarded to ``mix``.

        Returns:
          z_slow: Updated slow latent.
          z_fast: Updated fast latent.

        """
        return two_latent_refine(
            mix,
            input_emb,
            z_slow,
            z_fast,
            cos_sin,
            fast_cycles=self.config.fast_cycles,
        )

    @override
    def forward(
        self,
        core: CoreFn,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
        *,
        collect_intermediates: bool = False,
    ) -> ForwardOutput:
        """Run every cycle, carrying gradient only through the last."""
        all_logits: list[Tensor] = []
        with torch.no_grad():
            for _ in range(self.config.slow_cycles - 1):
                out = core(
                    input_emb.detach(),
                    z_slow.detach(),
                    z_fast.detach(),
                    None
                    if cos_sin is None
                    else (cos_sin[0].detach(), cos_sin[1].detach()),
                )
                z_slow, z_fast = out.z_slow, out.z_fast
                if collect_intermediates:
                    all_logits.append(out.logits)
        final = core(input_emb, z_slow, z_fast, cos_sin)
        if collect_intermediates:
            all_logits.append(final.logits.detach())
        return ForwardOutput(
            final.logits,
            final.halt,
            final.z_slow,
            final.z_fast,
            tuple(all_logits),
        )


class SudokuNet(nn.Module):
    """Grid-puzzle solver over an injected embedding, block stack, and recurrence.

    See the module docstring for what each slot varies. Without a recurrence the
    forward is one pass of the block stack; with one it is however many passes
    that recurrence prescribes, and the caller carries ``z_slow`` / ``z_fast``
    between calls to build an adaptive-computation-time rollout.
    """

    class Config(Fig["SudokuNet"]):
        """Width, depth, and the three injected slots."""

        channels_in: int = 512
        """Token embedding and latent-state width."""

        num_layers: int = 2
        """Blocks in the reasoning stack, applied per core application."""

        embedding: GridConfig = field(default_factory=GridEmbedding.Config)
        """Input embedding: tokens plus whatever additive channels_in apply."""

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
        """Latent-refinement schedule. ``None`` runs the stack exactly once."""

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

        vocab_size: int = 11
        """Output vocabulary; must match the embedding's."""

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
        :meth:`SudokuNet.eager` is told otherwise, because a compiled bfloat16
        autocast graph elides the casts autocast inserts and a long carried
        rollout compounds the difference."""

        dtype_latent_init: torch.dtype | None = None
        """Dtype the learned initial latents are DRAWN in; ``None`` is float32.

        Separate from ``dtype`` because the draw itself runs at this width: a
        bfloat16 truncated normal is not a float32 one rounded, so a recipe
        that drew its latents in half precision must say so to reproduce them."""

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
            propagate = self.block
            if isinstance(propagate, ChannelsIn) and propagate.channels_in == -1:
                propagate.channels_in = self.channels_in
            if isinstance(self.prefix, ChannelsOut) and self.prefix.channels_out == -1:
                self.prefix.channels_out = self.channels_in
            if self.num_prefix_tokens == -1:
                self.num_prefix_tokens = _count_prefix_tokens(self.prefix)
            return super().finalize()

        def cost(self, *, batch_size: int, dtype: torch.dtype | None) -> Cost:
            """Cost one complete invocation over the supplied puzzles.

            The concrete ledger includes the latent sequence, every core pass,
            both heads, and the embedding for each puzzle. One core pass runs
            the stack ``fast_cycles + 1`` times over parameters owned once.
            Every slow cycle runs the core forward and only the last runs
            backward, so the primal is repeated ``slow_cycles`` times and the
            adjoint once.

            Args:
              batch_size: Puzzles in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            puzzles = batch_size
            dt = dtype
            grid_len = self.grid_len
            latent = self.total_seq_len
            slow_cycles, fast_cycles = _cycles(self.recurrence)
            width = self.channels_in

            over_rows = functools.partial(
                cost,
                seq_len=latent,
                batch_size=puzzles,
                dtype=dt,
            )
            over_puzzles = functools.partial(
                cost,
                seq_len=1,
                batch_size=puzzles,
                dtype=dt,
            )
            stack = over_rows(self.block).tile(
                self.num_layers,
                copies=self.num_layers,
            )
            rows = latent * puzzles
            # ``z_slow + input_emb``, one add per fast cycle, and the slow
            # update: each an add forward and an accumulation back.
            adds = elementwise_cost(
                primal=(fast_cycles + 2) * width * rows,
                adjoint=(fast_cycles + 2) * width * rows,
                channels=(fast_cycles + 2) * width,
                inputs=2,
                rows=rows,
                dtype=dt,
            )
            per_row = stack.tile(fast_cycles + 1) + adds + over_rows(_head(self))
            core = per_row + over_puzzles(_halt_head(self))
            # Every slow cycle runs the core forward; only the last runs backward.
            primal_only = core.only("primal")
            total = (
                cost(
                    self.embedding,
                    seq_len=grid_len,
                    batch_size=puzzles,
                    dtype=dt,
                )
                + core
                + primal_only.tile(slow_cycles - 1)
            )
            if self.prefix is not None:
                total += over_puzzles(self.prefix)
                total += traffic(
                    "primal",
                    "selection",
                    elements=2 * latent * width * puzzles,
                    dtype=dt,
                )
            if self.recurrence is not None:
                total += over_rows(self.recurrence)
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        c = config.channels_in
        # Construction order fixes the global-RNG draw order, so a seeded init
        # is reproducible: embedding -> head -> blocks -> latent inits. The halt
        # head draws nothing (zeros and a constant bias).
        embedding = config.embedding.make()
        self.embedding = embedding

        self.head = _head(config).make()
        self.halt_head = _halt_head(config).make()

        # ``repeat`` builds independent copies, each finalized separately, so
        # every block draws its own weights in stack order.
        self.reasoning = Sequential.Config(
            elements=copy.deepcopy(config.block),
            repeat=config.num_layers,
        ).make()

        self.prefix = config.prefix.make() if config.prefix is not None else None
        # After the prefix, before the latent inits: RoPEMixed draws its
        # per-head directions from the global RNG at construction.
        self.rope = None if config.rope is None else config.rope.make()

        self.slow_init = nn.Buffer(
            _latent_init(c, dtype=config.dtype_latent_init),
            persistent=True,
        )
        self.fast_init = nn.Buffer(
            _latent_init(c, dtype=config.dtype_latent_init),
            persistent=True,
        )

        self.recurrence: Recurrence | None = (
            config.recurrence.make() if config.recurrence is not None else None
        )
        self._dummy = nn.Buffer(torch.empty(0), persistent=True)
        if config.dtype is not None:
            self.to(dtype=config.dtype)
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
        logger.info(
            "model parameters: %.2fM",
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    @property
    def device(self) -> torch.device:
        """Device the model's buffers live on."""
        return self._dummy.device

    def init_latents(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Return the initial ``(z_slow, z_fast)`` for a batch.

        Args:
          batch_size: Batch size.

        Returns:
          result: The tuple[Tensor, Tensor].

        """
        s = self.config.total_seq_len
        z_slow = self.slow_init[0].expand(batch_size, s, -1).contiguous()
        z_fast = self.fast_init[0].expand(batch_size, s, -1).contiguous()
        return z_slow, z_fast

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
        """Apply the block stack once, refining both latent states.

        Args:
          input_emb: ``[B, S, C]`` prefix-prepended input embedding.
          z_slow: ``[B, S, C]`` slow latent.
          z_fast: ``[B, S, C]`` fast latent.
          cos_sin: Optional rotary ``(cos, sin)`` pair for the blocks.

        Returns:
          out: Logits, halt logit, and both updated latents.

        """
        if self.compiled and self._compiled_core is not None:
            return self._compiled_core(input_emb, z_slow, z_fast, cos_sin)
        return self._core(input_emb, z_slow, z_fast, cos_sin)

    def _core(
        self,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> CoreOutput:
        """Apply the block stack once, never compiled."""
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
        halt = (
            halt_logits.squeeze(-1)
            if self.config.halt_outputs == 1
            else halt_logits[..., 0]
        )
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


def two_latent_refine(
    mix: MixFn,
    input_emb: Tensor,
    z_slow: Tensor,
    z_fast: Tensor,
    cos_sin: tuple[Tensor, Tensor] | None,
    *,
    fast_cycles: int,
) -> tuple[Tensor, Tensor]:
    """Run the TRM update: ``fast_cycles`` fast refinements, then one slow.

    Args:
      mix: One pass of the block stack.
      input_emb: ``[B, S, C]`` input embedding.
      z_slow: ``[B, S, C]`` slow latent.
      z_fast: ``[B, S, C]`` fast latent.
      cos_sin: Optional rotary pair forwarded to ``mix``.
      fast_cycles: Fast refinements before the slow update.

    Returns:
      z_slow: Updated slow latent.
      z_fast: Updated fast latent.

    """
    combined = z_slow + input_emb
    for _ in range(fast_cycles):
        z_fast = mix(z_fast + combined, cos_sin)
    return mix(z_slow + z_fast, cos_sin), z_fast


def _head(config: SudokuNet.Config) -> Linear.Config:
    """Configure the token head: one logit row per latent position, no bias."""
    return Linear.Config(
        channels_in=config.channels_in,
        channels_out=config.vocab_size,
        bias=False,
        init_weight=corrected_fan_in_normal,
    )


def _halt_head(config: SudokuNet.Config) -> Linear.Config:
    """Configure the halt head: zero weights and a constant, strongly negative bias."""
    return Linear.Config(
        channels_in=config.channels_in,
        channels_out=config.halt_outputs,
        bias=True,
        init_weight=nn.init.zeros_,
        init_bias=functools.partial(nn.init.constant_, val=config.halt_init_bias),
    )


# Without a recurrence the core runs once, its inner loop once: the plain model is a
# single pass over the block stack. A recurrence declares both -- read from the
# config so the core stays a plain method the recurrence can call, and so the same
# numbers cost a forward before anything is built.
def _cycles(recurrence: RecurrenceConfig | None) -> tuple[int, int]:
    """Return ``(slow_cycles, fast_cycles)`` one forward runs."""
    if recurrence is None:
        return 1, 1
    return recurrence.slow_cycles, recurrence.fast_cycles


# Read from the config rather than by constructing the module: ``finalize`` runs during
# ``pprint`` too, where building a large table would be both slow and surprising.
def _count_prefix_tokens(prefix: PrefixConfig | None) -> int:
    """How many tokens a prefix config contributes, before it is built."""
    return 0 if prefix is None else prefix.num_tokens


def _latent_init(channels_in: int, *, dtype: torch.dtype | None) -> Tensor:
    """Return a ``[1, C]`` learned-ish starting latent, unit-scaled, drawn in ``dtype``."""
    w = torch.empty(1, channels_in, dtype=dtype)
    truncated_normal(w, std=1.0, depth_index=(), variance_correction=True)
    return w
