"""The reference TRM step, plus an optional loss focused on the path's boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from priml.baselines.arcagi.train_step import PooledActTrainStep


if TYPE_CHECKING:
    from torch import Tensor

    from priml.baselines.arcagi.train_step import _Correctness
    from priml.baselines.maze.loss import CanonicalBoundary


class BoundaryLossTrainStep(PooledActTrainStep):
    """Train as :class:`PooledActTrainStep`, adding a weighted boundary loss when set."""

    class Config(PooledActTrainStep.Config):
        """The reference step plus the boundary loss slot."""

        if TYPE_CHECKING:

            @override
            def make(self) -> BoundaryLossTrainStep:
                """Build this nested configuration's step."""
                ...

        boundary: CanonicalBoundary.Config | None = None
        """Extra token loss over the path and its open neighbours; ``None`` omits it."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.boundary = None if config.boundary is None else config.boundary.make()

    # The boundary term recomputes the token loss rather than reusing the one the base
    # loss summed: the two then backpropagate as separate branches, in the order the
    # recorded runs accumulated them.
    @override
    def _train_loss(
        self,
        logits: Tensor,
        *,
        halt: Tensor,
        labels: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, dict[str, float | Tensor], _Correctness | None]:
        loss, metrics, correctness = super()._train_loss(
            logits,
            halt=halt,
            labels=labels,
            active=active,
        )
        if self.boundary is None:
            return loss, metrics, correctness
        per_token = self.token_loss(
            logits,
            labels,
            ignore_index=self.config.ignore_label_id,
        )
        boundary_loss = self.reduction(self.boundary(per_token, labels), active=active)
        metrics["canonical_boundary_loss"] = boundary_loss.detach()
        return loss + self.boundary.weight * boundary_loss, metrics, correctness
