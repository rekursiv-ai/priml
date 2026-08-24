"""Grid-puzzle solver: an embedding, a block stack, and an optional recurrence.

The network reads a token grid and predicts a token grid. Three things vary
independently, so each is a slot rather than a flag:

* ``embedding`` -- what signals are added to the input (see
  :mod:`priml.baselines.sudoku.embedding`).
* ``block`` -- how tokens mix. Any module accepting ``(x, *args, **kwargs)``
  works; :class:`~priml.model.transformer.TransformerBlock` and
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

from dataclasses import field
from typing import Any, NamedTuple, Protocol, Self, override, runtime_checkable

import copy
import functools
import logging

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.baselines.sudoku.embedding import GridEmbedding, HasChannels
from priml.model.custom_types import ChannelsIn
from priml.model.init import truncated_normal
from priml.model.linear import Linear
from priml.model.sequential import Sequential
from priml.model.transformer import TransformerBlock


logger = logging.getLogger(__name__)


def corrected_fan_in_normal(w: Tensor, *, depth: int = -1) -> None:
    """Truncated normal at ``std = 1/sqrt(fan_in)``, variance-corrected.

    The initialization every projection in this baseline uses. ``depth`` is
    accepted and discarded: priml's layers pass it to every ``init_weight`` so
    depth-scaled schemes can use it, and this one does not scale with depth.

    Args:
      w: Tensor to initialize in place.
      depth: Ignored; present for the ``InitFn`` protocol.

    """
    del depth
    truncated_normal(w, std=w.shape[-1] ** -0.5, depth=-1, variance_correction=True)


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


@runtime_checkable
class HasNumTokens(Protocol):
    """A prefix config declaring how many tokens it contributes."""

    num_tokens: int


@runtime_checkable
class HasParts(Protocol):
    """A prefix config composed of other prefix configs."""

    parts: list[Makeable[nn.Module]]


@runtime_checkable
class Recurrence(Protocol):
    """Drives repeated applications of a core over carried latent state.

    A recurrence decides HOW MANY times the reasoning core runs per forward and
    which of those applications carry gradient. It owns no parameters -- the
    core it drives does.
    """

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
        """Run the core to completion for one forward pass."""
        ...


class CoreFn(Protocol):
    """One application of the reasoning core."""

    def __call__(
        self,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None = None,
    ) -> CoreOutput: ...


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

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.slow_cycles < 1 or config.fast_cycles < 1:
            raise ValueError(
                f"slow_cycles and fast_cycles must be >= 1; got "
                f"{config.slow_cycles} and {config.fast_cycles}.",
            )
        self.config = config

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
                    _detach_pair(cos_sin),
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

        embedding: Makeable[GridEmbedding] = field(
            default_factory=GridEmbedding.Config,
        )
        """Input embedding: tokens plus whatever additive channels_in apply."""

        block: Makeable[nn.Module] = field(
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

        recurrence: Makeable[Recurrence] | None = None
        """Latent-refinement schedule. ``None`` runs the stack exactly once."""

        prefix: Makeable[nn.Module] | None = None
        """Optional module producing ``[B, P, C]`` tokens prepended to the grid.

        A per-puzzle embedding lives here. The halt readout reads position 0, so
        a prefix also gives that readout a dedicated token."""

        num_prefix_tokens: int = -1
        """Tokens the prefix contributes; sizes the latent state.

        ``-1`` counts them from the prefix module itself, so the two cannot
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

        @property
        def grid_len(self) -> int:
            """Grid tokens per puzzle, read from the embedding."""
            embedding = self.embedding
            assert isinstance(embedding, GridEmbedding.Config)
            return embedding.grid_len

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
            embedding = self.embedding
            assert isinstance(embedding, GridEmbedding.Config)
            if embedding.channels_in == -1:
                embedding.channels_in = self.channels_in
            embedding.vocab_size = self.vocab_size
            propagate = self.block
            if isinstance(propagate, ChannelsIn) and propagate.channels_in == -1:
                propagate.channels_in = self.channels_in
            if isinstance(self.prefix, HasChannels) and self.prefix.channels == -1:
                self.prefix.channels = self.channels_in
            if self.num_prefix_tokens == -1:
                self.num_prefix_tokens = _count_prefix_tokens(self.prefix)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        c = config.channels_in
        # Construction order fixes the global-RNG draw order, so a seeded init
        # is reproducible: embedding -> head -> blocks -> latent inits. The halt
        # head draws nothing (zeros and a constant bias).
        embedding = config.embedding.make()
        assert isinstance(embedding, GridEmbedding)
        self.embedding = embedding

        self.head = Linear.Config(
            channels_in=c,
            channels_out=config.vocab_size,
            bias=False,
            init_weight=corrected_fan_in_normal,
        ).make()

        self.halt_head = Linear.Config(
            channels_in=c,
            channels_out=config.halt_outputs,
            bias=True,
            init_weight=nn.init.zeros_,
            init_bias=functools.partial(
                nn.init.constant_,
                val=config.halt_init_bias,
            ),
        ).make()

        # ``repeat`` builds independent copies, each finalized separately, so
        # every block draws its own weights in stack order.
        self.reasoning = Sequential.Config(
            copy.deepcopy(config.block),
            repeat=config.num_layers,
        ).make()

        self.prefix = config.prefix.make() if config.prefix is not None else None

        self.slow_init = nn.Buffer(_latent_init(c), persistent=True)
        self.fast_init = nn.Buffer(_latent_init(c), persistent=True)

        self.recurrence: Recurrence | None = (
            config.recurrence.make() if config.recurrence is not None else None
        )
        self._dummy = nn.Buffer(torch.empty(0), persistent=True)
        logger.info(
            "model parameters: %.2fM",
            sum(p.numel() for p in self.parameters()) / 1e6,
        )

    @property
    def device(self) -> torch.device:
        """Device the model's buffers live on."""
        return self._dummy.device

    def init_latents(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Return the initial ``(z_slow, z_fast)`` for a batch."""
        s = self.config.total_seq_len
        z_slow = self.slow_init[0].expand(batch_size, s, -1).contiguous()
        z_fast = self.fast_init[0].expand(batch_size, s, -1).contiguous()
        return z_slow, z_fast

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
        fast_cycles = _fast_cycles(self.recurrence)
        combined = z_slow + input_emb
        for _ in range(fast_cycles):
            z_fast = self._mix(z_fast + combined, cos_sin)
        z_slow = self._mix(z_slow + z_fast, cos_sin)
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
        **prefix_kwargs: Any,
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
        cos_sin = None
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
        n_prefix = self.config.num_prefix_tokens
        if n_prefix:
            result = ForwardOutput(
                result.logits[:, n_prefix:],
                result.halt,
                result.z_slow,
                result.z_fast,
                tuple(lg[:, n_prefix:] for lg in result.all_logits),
            )
        return ForwardOutput(
            result.logits,
            result.halt,
            result.z_slow.detach(),
            result.z_fast.detach(),
            result.all_logits,
        )

    def _embed(self, tokens: Tensor, prefix_kwargs: dict[str, Any]) -> Tensor:
        """Embed the grid and prepend the prefix module's tokens, if any."""
        embeddings = self.embedding(tokens)
        if self.prefix is None:
            return embeddings
        # The prefix is per-batch, not per-token, so it takes the row count and
        # whatever the batch carries (puzzle ids, say) rather than the grid.
        prefix = self.prefix(tokens.shape[0], **prefix_kwargs)
        assert isinstance(prefix, Tensor)
        return torch.cat([prefix.to(dtype=embeddings.dtype), embeddings], dim=1)

    def _mix(self, z: Tensor, cos_sin: tuple[Tensor, Tensor] | None) -> Tensor:
        """Run the block stack once over a latent state."""
        out = self.reasoning(z, cos_sin=cos_sin)
        assert isinstance(out, Tensor)
        return out


def _fast_cycles(recurrence: Recurrence | None) -> int:
    """Inner-loop count for one core application.

    Without a recurrence the core runs its inner loop exactly once, which makes
    the plain model a single pass over the block stack. A recurrence that
    declares ``fast_cycles`` overrides it -- read from the config so the core
    stays a plain method the recurrence can call.
    """
    if isinstance(recurrence, DeepRecurrence):
        return recurrence.config.fast_cycles
    return 1


def _count_prefix_tokens(prefix: Makeable[nn.Module] | None) -> int:
    """How many tokens a prefix config contributes, before it is built.

    Read from the config rather than by constructing the module: ``finalize``
    runs during ``pprint`` too, where building a large table would be both slow
    and surprising.

    Args:
      prefix: The prefix slot's config, or None.

    Returns:
      tokens: Prefix width, 0 when there is no prefix.

    Raises:
      ValueError: A prefix config that declares no token count. Falling back to
        0 would strip nothing and silently shift every grid position.

    """
    if prefix is None:
        return 0
    if isinstance(prefix, HasParts):
        return sum(_count_prefix_tokens(part) for part in prefix.parts)
    if isinstance(prefix, HasNumTokens):
        return prefix.num_tokens
    raise ValueError(
        f"{type(prefix).__name__} fills the prefix slot but declares no "
        "num_tokens, so the model cannot size its sequence; add the field or "
        "set num_prefix_tokens explicitly.",
    )


def _detach_pair(
    pair: tuple[Tensor, Tensor] | None,
) -> tuple[Tensor, Tensor] | None:
    """Detach both halves of an optional tensor pair."""
    return None if pair is None else (pair[0].detach(), pair[1].detach())


def _latent_init(channels_in: int) -> Tensor:
    """A ``[1, C]`` learned-ish starting latent, unit-scaled."""
    w = torch.empty(1, channels_in)
    truncated_normal(w, std=1.0, depth=-1, variance_correction=True)
    return w
