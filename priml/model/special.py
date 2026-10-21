"""Utility modules: Identity, Skip, TiedLinear."""

from __future__ import annotations

from dataclasses import KW_ONLY, replace
from operator import attrgetter
from typing import Self, override

from configgle import Fig, LateBound, Makeable, Makes
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    traffic,
)
from priml.math.custom_types import TensorFn
from priml.model.custom_types import (
    PreservesGrid,
    TensorModule,
    WeightedTensorModule,
    has_weight,
    infer_same_width,
    validate_same_width,
)
from priml.model.passthrough import (
    PassthroughAttribute,
    ReadPassthroughMixin,
    ReadWritePassthroughMixin,
)
from priml.model.swiglu import silu


class Dropout(nn.Dropout):
    """Apply native elementwise dropout with explicit training cost."""

    class Config(PreservesGrid, Fig["Dropout"], kw_only=False):
        channels_in: int = -1
        """Width entering dropout, or supplied on the cost bus."""

        channels_out: int = -1
        """Width preserved by dropout."""

        _: KW_ONLY

        p: float = 0.5
        """Probability of zeroing each element during training."""

        inplace: bool = False
        """Update the input in place using the native Torch primitive."""

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            channels: int = -1,
            training: bool = True,
            **kwargs: object,
        ) -> Cost:
            """Count native mask scaling and forward/backward products.

            Args:
              seq_len: Positions per example.
              batch_size: Examples per invocation.
              dtype: Activation dtype.
              channels: Width when not configured on this leaf.
              training: Cost training; evaluation is an identity.
              **kwargs: Unused outer metadata.

            Returns:
              execution: Whole-invocation arithmetic and logical traffic.

            """
            del kwargs
            width = self.channels_in if self.channels_in != -1 else channels
            elements = width * seq_len * batch_size
            return dropout_cost(
                self.p,
                elements=elements,
                masks=elements,
                dtype=dtype,
                training=training,
            )

    def __init__(self, config: Config) -> None:
        validate_same_width(config)
        super().__init__(p=config.p, inplace=config.inplace)

    def reset_parameters(self) -> None:
        """Retain the stateless dropout operation."""

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return super().forward(input)


class Dropout2d(Dropout):
    """Apply native channel dropout with explicit training cost."""

    class Config(Makes["Dropout2d"], Dropout.Config):
        @override
        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            channels: int = -1,
            training: bool = True,
            **kwargs: object,
        ) -> Cost:
            """Count one mask per example and channel.

            Args:
              seq_len: Positions per example.
              batch_size: Examples per invocation.
              dtype: Activation dtype.
              channels: Width when not configured on this leaf.
              training: Cost training; evaluation is an identity.
              **kwargs: Unused outer metadata.

            Returns:
              execution: Whole-invocation arithmetic and logical traffic.

            """
            del kwargs
            width = self.channels_in if self.channels_in != -1 else channels
            return dropout_cost(
                self.p,
                elements=width * seq_len * batch_size,
                masks=width * batch_size,
                dtype=dtype,
                training=training,
            )

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return functional.dropout2d(input, self.p, self.training, self.inplace)


def dropout_cost(
    probability: float,
    *,
    elements: int,
    masks: int,
    dtype: torch.dtype | None,
    training: bool = True,
) -> Cost:
    """Cost native dropout with one independent sample per mask element.

    Bernoulli sampling writes each mask; scaling reads/writes it, then products
    read it. Sampling itself has no assigned FLOPs. A channel mask is scaled
    once per channel, while its forward and backward products visit every value.

    Args:
      probability: Probability of zeroing an activation.
      elements: Complete input element count.
      masks: Independent mask elements; may broadcast across input positions.
      dtype: Activation and scaled-mask dtype.
      training: Cost training; evaluation is an identity.

    Returns:
      execution: Mask scaling, products, and logical operand traffic.

    """
    if probability == 0 or not training:
        return Cost()
    if probability == 1:
        masks = 0
    return traffic(
        "primal",
        "elementwise",
        elements=2 * elements + 4 * masks,
        flops=elements + masks,
        dtype=dtype,
    ) + traffic(
        "adjoint",
        "elementwise",
        elements=2 * elements + masks,
        flops=elements,
        dtype=dtype,
    )


class Activation(nn.Module):
    """Apply an injected cost-bearing tensor function without changing its primitive."""

    class Config(PreservesGrid, Fig["Activation"], kw_only=False):
        channels_in: int = -1
        """Width entering the activation, or supplied on the cost bus."""

        channels_out: int = -1
        """Width preserved by the activation."""

        _: KW_ONLY

        act: TensorFn = silu
        """Injected activation carrying its own cost through set_cost."""

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            channels: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Delegate the whole activation geometry to its own cost.

            Args:
              seq_len: Positions per example.
              batch_size: Examples per invocation.
              dtype: Activation dtype.
              channels: Width when not configured on this leaf.
              **kwargs: Unused outer metadata.

            Returns:
              execution: Complete activation cost, owning no parameters.

            """
            del kwargs
            width = self.channels_in if self.channels_in != -1 else channels
            return cost(self.act, channels=width * seq_len * batch_size, dtype=dtype)

    def __init__(self, config: Config) -> None:
        validate_same_width(config)
        super().__init__()
        self.act = config.act

    def reset_parameters(self) -> None:
        """Retain the stateless activation."""

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return self.act(x)


class Identity(nn.Identity):
    """Identity that returns only the first positional argument."""

    class Config(PreservesGrid, Fig["Identity"], kw_only=False):
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
              cost: Integer FLOPs and logical bytes for the complete invocation.

            """
            del seq_len, batch_size, dtype, kwargs
            return Cost()

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""

    # Narrower than ``nn.Identity``'s on purpose: an extra positional is a bug.
    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:  # pyrefly: ignore[bad-override] -- Rejecting extra positionals is the contract ``test_identity`` pins.
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
              cost: Integer FLOPs and logical bytes for the complete invocation.

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
                primal=self.channels_out * seq_len * batch_size,
                adjoint=self.channels_in * seq_len * batch_size,
                channels=self.channels_out,
                rows=seq_len * batch_size,
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
              cost: Integer FLOPs and logical bytes for the complete invocation.

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

    @property
    def weight(self) -> Tensor:
        """The borrowed weight in ``nn.Linear``'s ``[out, in]`` layout.

        A head tied to an embedding returns the table itself, so readers that
        take ``weight.T`` as the ``[in, out]`` projection treat both alike.

        Raises:
          AttributeError: No ``make()`` has bound the source yet, so there is
            no weight; ``has_weight`` reads this as absent.

        """
        if self._source is None:
            raise AttributeError(f"TiedLinear for {self.tied!r} is unbound.")
        return self._source.weight if self.transpose else self._source.weight.T

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        if self._source is None:
            raise RuntimeError(
                f"TiedLinear is unbound; a make() must build the root holding "
                f"{self.tied!r}.",
            )
        if x.dtype.is_floating_point:
            return x @ self.weight.T
        return functional.embedding(x, self.weight.T)
