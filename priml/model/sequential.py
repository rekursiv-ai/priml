"""Sequential composition with optional repetition."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, cast, override

from configgle import Fig, Makeable, Maker
from torch import Tensor, nn

from priml.model.custom_types import DepthIndex, HasDepthIndex


class Sequential(nn.Sequential):
    """Sequential container that builds from a config.

    When ``repeat`` is set, the element is repeated that many times,
    with ``depth_index`` set to the loop index on each copy. Each element's
    ``finalize()`` is responsible for propagating ``depth_index`` to its
    own children.

    Examples::

        # Single layer:
        Sequential.Config(Linear.Config(128, 256))

        # MLP with depth-aware init:
        Sequential.Config(
            Sequential.Config(Linear.Config(128, 128)),
            repeat=4,
        )
    """

    class Config(Fig["Sequential"], kw_only=False):
        elements: Makeable[nn.Module] | list[Makeable[nn.Module]] = field(  # pyright: ignore[reportUnknownVariableType]
            default_factory=list,
        )
        """Module config(s) to compose sequentially."""

        _: KW_ONLY

        repeat: int = 1
        """Number of times to repeat the element(s), with depth set per copy."""

        depth_index: DepthIndex = ()
        """Global-to-local stack position inherited by repeated children."""

        @override
        def finalize(self) -> Self:
            elements = self.elements
            base = cast(  # pyright: ignore[reportUnnecessaryCast] -- ty widens the loose field's list contents to object; pyright narrows it
                "list[Makeable[nn.Module]]",
                list(elements) if isinstance(elements, list) else [elements],
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
            self.elements = expanded
            self.repeat = 1
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
