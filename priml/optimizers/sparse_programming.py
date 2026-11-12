"""Ordered writes of fixed values into exact parameter addresses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast, override

from configgle import Makes

import torch

from priml.optimizers.weight_programming import UpdateOutput, WeightProgramming


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from torch import Tensor, nn


@dataclass(frozen=True, slots=True, kw_only=True)
class TensorWrite:
    """Set a whole parameter, or the region its leading indices select, to a value."""

    parameter: str
    """Name as ``named_parameters`` reports it."""

    indices: tuple[int, ...] = ()
    """Leading indices of the region; empty writes the whole parameter."""

    value: float = 0.0
    """Written to every scalar in the region."""


class SparseProgramming(WeightProgramming):
    """Apply an ordered list of fixed writes, or none of them.

    Every address resolves before any scalar is written, so a misspelled name,
    an index out of range, or a parameter another member owns leaves the model
    untouched. Later writes win where regions overlap. The update reports how
    many scalars now differ from their value before the call.
    """

    class Config(Makes["Callable[..., SparseProgramming]"], WeightProgramming.Config):
        """The writes, in the order applied."""

        writes: list[TensorWrite] = field(default_factory=list[TensorWrite])
        """Applied in order on every step."""

    def __init__(self, params: Iterable[Tensor], *, config: Config) -> None:
        super().__init__(params, config=config)
        self.writes = list(config.writes)

    @override
    def program_weights(
        self,
        model: nn.Module,
        *,
        next_batch: Callable[[], dict[str, object]],
    ) -> UpdateOutput:
        """Resolve every write, then apply them in order.

        Args:
          model: The full model; every written parameter must be owned here.
          next_batch: Unused; fixed writes read no calibration input.

        Returns:
          output: ``changed_scalars``, and whether it is nonzero.

        Raises:
          KeyError: A write names no parameter of the model.
          ValueError: A write names a parameter another member owns.
          IndexError: A write's indices fall outside its parameter.

        """
        del next_batch
        owned = {
            id(parameter)
            for group in self.param_groups
            for parameter in cast("list[Tensor]", group["params"])
        }
        named = dict(model.named_parameters())
        regions: list[Tensor] = []
        for write in self.writes:
            parameter = named[write.parameter]
            if id(parameter) not in owned:
                raise ValueError(
                    f"Parameter {write.parameter!r} belongs to another optimizer.",
                )
            regions.append(parameter[write.indices])
        before = {
            name: named[name].detach().clone()
            for name in {write.parameter for write in self.writes}
        }
        with torch.no_grad():
            for region, write in zip(regions, self.writes, strict=True):
                region.fill_(write.value)
        changed = sum(
            int((named[name] != value).sum()) for name, value in before.items()
        )
        return UpdateOutput(updated=changed > 0, metrics={"changed_scalars": changed})
