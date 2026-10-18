"""Sequential composition with optional repetition."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, cast, override

import math

from configgle import Fig, Makeable, Maker
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    cost,
)
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    DepthIndex,
    HasDepthIndex,
    HasOutputGrid,
    HasResetParameters,
    TensorModule,
)


class Sequential(nn.Sequential):
    """Sequential container that builds from a config.

    When ``repeat`` is set, the element is repeated that many times,
    with ``depth_index`` set to the loop index on each copy. Each element's
    ``finalize()`` is responsible for propagating ``depth_index`` to its
    own children.

    Widths chain through the elements: ``channels_in`` enters the first,
    each element's finalized ``channels_out`` becomes the next one's
    ``channels_in``, and ``channels_out`` is pushed into the last. A parent
    therefore sizes a composed head (``[RMSNorm, Linear]``) exactly as it
    would size a bare projection.

    Examples::

        # Single layer:
        Sequential.Config(elements=Linear.Config(128, 256))

        # MLP with depth-aware init:
        Sequential.Config(
            elements=Sequential.Config(elements=Linear.Config(128, 128)),
            repeat=4,
        )
    """

    class Config(Fig["Sequential"], kw_only=False):
        channels_in: int = -1
        """Width entering the first element (-1 to read it from the element)."""

        channels_out: int = -1
        """Width leaving the last element (-1 to read it from the element)."""

        _: KW_ONLY

        elements: Makeable[TensorModule] | list[Makeable[TensorModule]] = field(
            default_factory=list[Makeable[TensorModule]],
        )
        """Module config(s) to compose sequentially."""

        repeat: int = 1
        """Number of times to repeat the element(s), with depth set per copy."""

        depth_index: DepthIndex = ()
        """Global-to-local stack position inherited by repeated children."""

        @override
        def finalize(self) -> Self:
            elements = self.elements
            base: list[Makeable[TensorModule]] = (
                list(elements) if isinstance(elements, list) else [elements]
            )
            expanded: list[Makeable[TensorModule]] = []
            for index in range(self.repeat):
                for element in base:
                    copied = element.copy_tree()
                    if isinstance(copied, HasDepthIndex):
                        copied.depth_index = self.depth_index
                        if self.repeat > 1:
                            copied.depth_index += ((index, self.repeat),)
                    expanded.append(copied)
            # Each element is finalized as it is reached, so the next reads a
            # DERIVED width: a norm infers ``channels_out`` from ``channels_in``
            # inside its own finalize, and reading it earlier yields -1.
            width = self.channels_in
            for index, element in enumerate(expanded):
                if (
                    width != -1
                    and isinstance(element, ChannelsIn)
                    and element.channels_in == -1
                ):
                    element.channels_in = width
                if (
                    index == len(expanded) - 1
                    and self.channels_out != -1
                    and isinstance(element, ChannelsOut)
                    and element.channels_out == -1
                ):
                    element.channels_out = self.channels_out
                if not getattr(element, "_finalized", False):
                    element.finalize()
                width = element.channels_out if isinstance(element, ChannelsOut) else -1
            self.elements = expanded
            self.repeat = 1
            if expanded:
                first, last = expanded[0], expanded[-1]
                if self.channels_in == -1 and isinstance(first, ChannelsIn):
                    self.channels_in = first.channels_in
                if self.channels_out == -1 and isinstance(last, ChannelsOut):
                    self.channels_out = last.channels_out
            return super().finalize()

        def output_grid(self, input_grid: tuple[int, ...]) -> tuple[int, ...]:
            """Chain declared spatial transformations through the elements.

            Args:
                input_grid: Spatial extents entering the composition.

            Returns:
                output: Spatial extents leaving the final element.

            """
            assert isinstance(self.elements, list)
            for element in self.elements:
                if not isinstance(element, HasOutputGrid):
                    raise TypeError("Sequential requires declared spatial geometry.")
                input_grid = element.output_grid(input_grid)
            return input_grid

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            channels: int = -1,
            input_grid: tuple[int, ...] | None = None,
            **kwargs: object,
        ) -> Cost:
            """Sum the finalized element list.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              channels: Fallback width for a composition without a sized input.
              input_grid: Spatial extents, propagated through declared transforms.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            assert isinstance(self.elements, list)
            channels = self.channels_in if self.channels_in != -1 else channels
            total = Cost()
            for element in self.elements:
                next_grid = input_grid
                if input_grid is not None:
                    if not isinstance(element, HasOutputGrid):
                        raise TypeError(
                            "Sequential requires declared spatial geometry.",
                        )
                    next_grid = element.output_grid(input_grid)
                spatial = {} if input_grid is None else {"input_grid": input_grid}
                total += cost(
                    element,
                    seq_len=seq_len if input_grid is None else math.prod(input_grid),
                    batch_size=batch_size,
                    dtype=dtype,
                    channels=channels,
                    **spatial,
                    **kwargs,
                )
                input_grid = next_grid
                channels = (
                    element.channels_out if isinstance(element, ChannelsOut) else -1
                )
            return total

    def __init__(self, config: Config) -> None:
        # ``finalize`` has already flattened ``elements`` (repeat expanded, depth
        # assigned) and finalized each; just build them.
        elements = config.elements
        assert isinstance(elements, list)
        modules: list[nn.Module] = []
        for element in elements:
            assert isinstance(element, Maker)
            built = element.make()
            assert isinstance(built, nn.Module)
            modules.append(built)

        super().__init__(*modules)

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        for module in self:
            if isinstance(module, HasResetParameters):
                module.reset_parameters()

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        for module in self:
            output = cast(Tensor, module(input, **kwargs))
            input = output
        return input
