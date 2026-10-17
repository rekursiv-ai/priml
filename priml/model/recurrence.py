"""Reusable latent recurrence schedules for sequence models."""

from __future__ import annotations

from typing import Protocol, override, runtime_checkable

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn

import torch

from priml.cost import Cost
from priml.model.custom_types import CoreFn, ForwardOutput, MixFn


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
          core: One core application: a block-stack pass over the latents.
          input_emb: ``[B, S, C]`` prefix-prepended input embedding.
          z_slow: ``[B, S, C]`` slow latent.
          z_fast: ``[B, S, C]`` fast latent.
          cos_sin: Rotary tables, or ``None`` without rotary positions.
          collect_intermediates: Whether to record per-cycle latents.

        Returns:
          out: Logits, halt logit, and both updated latents.

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

    @property
    def forward_passes(self) -> int:
        """Actual block-stack applications per core invocation."""
        ...

    @property
    def grad_passes(self) -> int:
        """Block-stack applications retaining gradients in the final cycle."""
        ...

    @property
    def forward_additions(self) -> int:
        """Input additions performed by one complete core invocation."""
        ...

    @property
    def grad_additions(self) -> int:
        """Input additions retained in the final core invocation's graph."""
        ...


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

        @property
        def forward_passes(self) -> int:
            """Block-stack applications per core invocation."""
            return self.fast_cycles + 1

        @property
        def grad_passes(self) -> int:
            """Differentiable block-stack applications in the final cycle."""
            return self.fast_cycles + 1

        @property
        def forward_additions(self) -> int:
            """One slow-input add, each fast update add, and one slow update add."""
            return self.fast_cycles + 2

        @property
        def grad_additions(self) -> int:
            """Additions in the final two-latent core invocation."""
            return self.fast_cycles + 2

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


class SingleLatentRecurrence(DeepRecurrence):
    """Refine one carried latent through repeated applications of a model core."""

    class Config(Makes["SingleLatentRecurrence"], DeepRecurrence.Config):
        """Cycle counts and the number of differentiable inner passes."""

        inner_grad_loops: int = 0
        """Differentiable passes within a core application; zero keeps all."""

        @property
        @override
        def forward_passes(self) -> int:
            """Core applications per cycle, including no-gradient applications."""
            if self.inner_grad_loops > 0:
                return max(self.fast_cycles, self.inner_grad_loops)
            return self.fast_cycles

        @property
        @override
        def grad_passes(self) -> int:
            """Applications in the final core cycle that retain gradients."""
            return self.inner_grad_loops or self.fast_cycles

        @property
        @override
        def forward_additions(self) -> int:
            """One input addition before every forward stack application."""
            return self.forward_passes

        @property
        @override
        def grad_additions(self) -> int:
            """One input addition before every gradient-bearing stack application."""
            return self.grad_passes

        @property
        def outer_grad_cycles(self) -> int:
            """Outer cycles retaining gradients; the final cycle only."""
            return 1

    @override
    def __init__(self, config: DeepRecurrence.Config) -> None:
        assert isinstance(config, SingleLatentRecurrence.Config)
        if config.inner_grad_loops < 0:
            raise ValueError(
                f"inner_grad_loops must be >= 0; got {config.inner_grad_loops}.",
            )
        super().__init__(config)

    @override
    def refine(
        self,
        mix: MixFn,
        input_emb: Tensor,
        z_slow: Tensor,
        z_fast: Tensor,
        cos_sin: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:
        """Re-inject the input before each pass over the single state."""
        config = self.config
        assert isinstance(config, SingleLatentRecurrence.Config)
        state = z_slow
        grad_loops = config.inner_grad_loops
        if grad_loops > 0:
            with torch.no_grad():
                for _ in range(max(config.fast_cycles - grad_loops, 0)):
                    state = mix(state.detach() + input_emb.detach(), cos_sin)
            state = state.detach()
            for _ in range(grad_loops):
                state = mix(state + input_emb, cos_sin)
        else:
            for _ in range(config.fast_cycles):
                state = mix(state + input_emb, cos_sin)
        return state, z_fast
