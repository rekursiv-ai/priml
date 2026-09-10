"""Sequential composition with optional repetition."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, override

from configgle import Fig, Makeable, Maker
from torch import Tensor, nn

from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    DepthIndex,
    HasDepthIndex,
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

        elements: Makeable[nn.Module] | list[Makeable[nn.Module]] = field(
            default_factory=list[Makeable[nn.Module]],
        )
        """Module config(s) to compose sequentially."""

        repeat: int = 1
        """Number of times to repeat the element(s), with depth set per copy."""

        depth_index: DepthIndex = ()
        """Global-to-local stack position inherited by repeated children."""

        @override
        def finalize(self) -> Self:
            elements = self.elements
            base: list[Makeable[nn.Module]] = (
                list(elements) if isinstance(elements, list) else [elements]
            )
            expanded: list[Makeable[nn.Module]] = []
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
        for module in self:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        for module in self:
            output = module(input, **kwargs)
            assert isinstance(output, Tensor)
            input = output
        return input
