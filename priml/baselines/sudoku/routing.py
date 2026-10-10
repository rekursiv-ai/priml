"""The routing slot: which latent chain each train step trains and carries.

:class:`SingleChain` is the default, the pool's one chain; the latent
hypothesis routings in :mod:`~priml.baselines.sudoku.hypotheses` pick
one of several.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol, override

from configgle import Fig


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from priml.baselines.sudoku.act import AtomicPool
    from priml.baselines.sudoku.latent_recursion import LatentRecursion


class Routed(NamedTuple):
    """One train forward: the chain the loss trains and the pool carries."""

    logits: Tensor
    """``[B, grid_len, V]`` logits of the trained chain."""

    q_halt: Tensor
    """``[B]`` halt logit of the trained chain."""

    z_slow: Tensor
    """``[B, S, C]`` slow latent the pool carries."""

    z_fast: Tensor
    """``[B, S, C]`` fast latent the pool carries."""

    metrics: dict[str, float | Tensor]
    """Routing diagnostics, merged into the train metrics."""


class Routing(Protocol):
    """Which latent chain each train step trains and the pool carries."""

    def seat(
        self,
        model: LatentRecursion,
        pool: AtomicPool,
        *,
        global_step: int,
    ) -> None:
        """Initialize routing state for the slots the pool just seated."""
        ...

    def __call__(
        self,
        model: LatentRecursion,
        pool: AtomicPool,
        *,
        token_loss: Callable[[Tensor, Tensor], Tensor],
        ignore_index: int,
    ) -> Routed:
        """Forward the pool under the caller's autocast and gradient mode."""
        ...


class SingleChain(Routing):
    """Train and carry the pool's one latent chain."""

    class Config(Fig["SingleChain"]):
        """No parameters."""

    def __init__(self, config: Config) -> None:
        del config

    @override
    def seat(
        self,
        model: LatentRecursion,
        pool: AtomicPool,
        *,
        global_step: int,
    ) -> None:
        del model, pool, global_step

    @override
    def __call__(
        self,
        model: LatentRecursion,
        pool: AtomicPool,
        *,
        token_loss: Callable[[Tensor, Tensor], Tensor],
        ignore_index: int,
    ) -> Routed:
        del token_loss, ignore_index
        out = model(
            pool.inputs,
            pool.z_slow,
            pool.z_fast,
            pool.puzzle_ids if model.puzzle_emb is not None else None,
        )
        return Routed(out["logits"], out["q_halt"], out["z_slow"], out["z_fast"], {})
