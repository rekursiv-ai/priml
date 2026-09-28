"""Parameter routing and optimizer stacks for the reference TRM recipes."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from configgle import PartialConfig

from priml.optimizers.composite import CompositeOptimizer
from priml.optimizers.muon import Muon
from priml.optimizers.parameter_filter import complement, excluding, filter_name


if TYPE_CHECKING:
    from torch.nn import Parameter

    import torch

    from priml.optimizers.parameter_filter import ParameterFilter
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")


class with_ndim:  # noqa: N801 -- The lowercase name matches the public combinator syntax.
    """Narrow a filter to parameters of exactly one rank.

    Muon folds a rank-3 weight's leading axis into an ensemble of matrices, so
    the reference routes rank-2 and rank-3 weights to separately configured
    Muon members; this is the predicate that splits them.

    Args:
      select: Filter to narrow.
      ndim: Rank to accept.

    """

    __slots__ = ("ndim", "select")

    def __init__(self, select: ParameterFilter, ndim: int) -> None:
        self.select = select
        self.ndim = ndim

    def __call__(self, name: str, parameter: Parameter) -> bool:
        """Whether ``select`` takes ``parameter`` and it has rank ``ndim``."""
        return parameter.ndim == self.ndim and self.select(name, parameter)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, with_ndim)
            and self.select == other.select
            and self.ndim == other.ndim
        )

    @override
    def __hash__(self) -> int:
        return hash((type(self), self.select, self.ndim))

    @override
    def __repr__(self) -> str:
        return f"with_ndim({filter_name(self.select)}, {self.ndim})"


def adamw_muon(*, adamw_lr: float, muon_lr: float) -> CompositeOptimizer.Config:
    """Return AdamW on tables, heads, and vectors; Muon on the body's matrices.

    Weight decay is ``1e-4 / lr`` for both, the reference's coupling: the
    decoupled decay per step, ``lr * weight_decay``, is then 1e-4 whatever the
    rate.

    Args:
      adamw_lr: AdamW rate.
      muon_lr: Muon rate, shared by the rank-2 and rank-3 members.

    Returns:
      config: A three-member composite; an empty Muon member is dropped.

    """
    on_muon = excluding(Muon.eligible_tensor, "embed", "head", "register_tokens")
    config = CompositeOptimizer.Config()
    config.optimizers = [
        PartialConfig(
            torch.optim.AdamW,
            lr=adamw_lr,
            betas=(0.9, 0.95),
            weight_decay=1e-4 / adamw_lr,
        ),
        Muon.Config(
            lr=muon_lr,
            momentum=0.6,
            ns_steps=3,
            weight_decay=1e-4 / muon_lr,
        ),
        Muon.Config(
            lr=muon_lr,
            momentum=0.6,
            ns_steps=3,
            weight_decay=1e-4 / muon_lr,
            ensemble_dims=1,
        ),
    ]
    config.select = [
        complement(on_muon),
        with_ndim(on_muon, 2),
        with_ndim(on_muon, 3),
    ]
    config.drop_empty = True
    return config
