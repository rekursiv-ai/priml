"""Utility modules: Identity, Skip, TiedLinear."""

from __future__ import annotations

from dataclasses import KW_ONLY
from operator import attrgetter
from typing import Self, override

from configgle import Fig, LateBound, Makeable
from torch import Tensor, nn
from torch.nn import functional

from priml.model.custom_types import (
    TensorModule,
    WeightedTensorModule,
    has_weight,
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
        """Number of input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if (
            -1 not in (config.channels_in, config.channels_out)
            and config.channels_in != config.channels_out
        ):
            raise ValueError(
                f"channels_in={config.channels_in} must equal "
                f"channels_out={config.channels_out} for Identity."
            )
        super().__init__()

    def reset_parameters(self) -> None:
        pass

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

    def __init__(self, config: Config) -> None:
        if config.inner is None:
            raise ValueError("Must specify `inner`.")
        # The inner module's own widths are its invariant, not this one's: a
        # residual add over mismatched widths raises from torch, naming both
        # sizes and the dimension they disagree on.
        super().__init__()
        self.inner = config.inner.make()

    def reset_parameters(self) -> None:
        if hasattr(self.inner, "reset_parameters"):
            self.inner.reset_parameters()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        inner = self.inner(x, **kwargs)
        assert isinstance(inner, Tensor)
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
        source = attrgetter(self.tied)(root)
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
                f"{self.tied!r}."
            )
        weight = self._source.weight.T if self.transpose else self._source.weight
        if x.dtype.is_floating_point:
            return x @ weight
        return functional.embedding(x, weight)
