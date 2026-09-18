"""Utility modules: Identity, Skip, TiedLinear."""

from __future__ import annotations

from dataclasses import KW_ONLY, replace
from operator import attrgetter
from typing import Self, override

from configgle import Fig, LateBound, Makeable
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
)
from priml.model.custom_types import (
    TensorModule,
    WeightedTensorModule,
    has_weight,
    infer_same_width,
)
from priml.model.passthrough import (
    PassthroughAttribute,
    ReadPassthroughMixin,
    ReadWritePassthroughMixin,
)


class Identity(nn.Identity):
    """Identity that returns only the first positional argument."""

    class Config(Fig["Identity"], kw_only=False):
        channels_in: int = -1
        """Input channel width."""

        channels_out: int = -1
        """Output channel width."""

        _: KW_ONLY

        @override
        def finalize(self) -> Self:
            """Fill and validate the preserved channel width."""
            infer_same_width(self)
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost nothing: no parameters, no arithmetic, no copy.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del seq_len, batch_size, dtype, kwargs
            return Cost()

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return input


class Skip(ReadPassthroughMixin, nn.Module, passthrough="inner"):
    """Residual connection: output = x + inner(x, ...)."""

    class Config(
        ReadWritePassthroughMixin,
        Fig["Skip"],
        passthrough="inner",
    ):
        inner: Makeable[TensorModule] | None = None
        """Submodule to wrap with a residual connection."""

        channels_in = PassthroughAttribute[int]()
        channels_out = PassthroughAttribute[int]()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the branch, residual add, and input-gradient accumulation.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            Raises:
              ValueError: No ``inner`` to wrap.

            """
            if self.inner is None:
                raise ValueError("Must specify `inner`.")
            return cost(
                self.inner,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + elementwise_cost(
                primal=self.channels_out,
                adjoint=self.channels_in,
                channels=self.channels_out,
                inputs=2,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        if config.inner is None:
            raise ValueError("Must specify `inner`.")
        # The inner module's own widths are its invariant, not this one's: a
        # residual add over mismatched widths raises from torch, naming both
        # sizes and the dimension they disagree on.
        super().__init__()
        self.inner = config.inner.make()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        self.inner.reset_parameters()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        inner = self.inner(x, **kwargs)
        return x + inner


class TiedLinear(nn.Module, LateBound):
    """Linear map through another module's weight; owns no parameter.

    The weight is borrowed, so it appears once in the state dict and once to
    the optimizer. ``tied`` is a dotted path from the ROOT of the built tree
    (``LateBound``: configgle's outermost ``make`` hands the root over after
    everything is built), so the tie prints, may point at a module built
    later, and the enclosing module writes nothing.

    Integer input selects rows -- the one-hot matmul, done as a lookup -- so
    the same module serves as a head tied to an embedding (``transpose=True``,
    the table read as ``[hidden, vocab]``) or as an embedding tied to a head
    (``transpose=False``, the head's ``[vocab, hidden]`` weight read as a
    table).
    """

    class Config(Fig["TiedLinear"], kw_only=False):
        channels_in: int = -1
        """Input width; the borrowed weight's column count after transpose."""

        channels_out: int = -1
        """Output width; the borrowed weight's row count after transpose."""

        _: KW_ONLY

        tied: str = ""
        """Dotted path from the built tree's root to the weight's owner."""

        transpose: bool = True
        """Read the weight as ``weight.T``; the usual head-over-embedding tie."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the matmul; the weight is counted where it is owned.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            return replace(
                matmul_cost(
                    channels_in=self.channels_in,
                    channels_out=self.channels_out,
                    bias=False,
                    rows=seq_len * batch_size,
                    dtype=dtype,
                ),
                params=0,
            )

    def __init__(self, config: Config) -> None:
        if not config.tied:
            raise ValueError("TiedLinear requires `tied`, a path to a sibling.")
        super().__init__()
        self.tied = config.tied
        self.transpose = config.transpose
        self._source: WeightedTensorModule | None = None

    @override
    def bind(self, root: object) -> None:
        """Resolve ``tied`` against the built tree's root.

        Raises:
          ValueError: The path names nothing with a tensor ``weight``.

        """
        source: object = attrgetter(self.tied)(root)  # pyright: ignore[reportAny] -- runtime path resolution returns the configured module.
        if not has_weight(source):
            raise ValueError(f"tied={self.tied!r} must name a module with a weight.")
        # Not ``self._source = source``: nn.Module registers a Module value as
        # a child, which would duplicate the source's parameters under this
        # module's prefix.
        object.__setattr__(self, "_source", source)

    def reset_parameters(self) -> None:
        """Nothing owned; the source resets its own weight."""

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        if self._source is None:
            raise RuntimeError(
                f"TiedLinear is unbound; a make() must build the root holding "
                f"{self.tied!r}.",
            )
        weight = self._source.weight.T if self.transpose else self._source.weight
        if x.dtype.is_floating_point:
            return x @ weight
        return functional.embedding(x, weight)
