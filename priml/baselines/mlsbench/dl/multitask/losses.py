"""Combinations of two task losses: the task's editable ``MultiTaskLoss``.

Each is a module called as ``(loss_fine, loss_coarse, epoch, total_epochs)``
that returns the scalar the step backpropagates. A module rather than a
function because a combination may hold parameters, which the optimizer
updates beside the network's, or state carried between steps. They fill
:attr:`~priml.baselines.mlsbench.dl.multitask.train_step.MultiTaskStep.Config.mtl_loss`.

Every expression keeps the reference's operation order: a combination that
must reproduce the reference bit for bit follows its arithmetic, so an
algebraically equal rewrite is not a refactor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, elementwise_cost


if TYPE_CHECKING:
    from torch.autograd.graph import Node


class SumLoss(nn.Module):
    """Add the two losses: the task's pristine combination."""

    class Config(Fig["SumLoss"]):
        """No parameters; a config so the combination is a slot a fork can fill."""

        def cost(
            self,
            *,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one scalar addition.

            Args:
              dtype: Loss dtype; ``None`` is torch's default.
              **kwargs: The open bus; unused.

            Returns:
              cost: A few scalar operations on the two losses.

            """
            del kwargs
            return _combination_cost(params=0, dtype=dtype)

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()

    @override
    def forward(
        self,
        loss_fine: Tensor,
        loss_coarse: Tensor,
        epoch: int,
        total_epochs: int,
    ) -> Tensor:
        """Return ``loss_fine + loss_coarse``."""
        del epoch, total_epochs
        return loss_fine + loss_coarse


class UncertaintyLoss(nn.Module):
    """Weight each loss by a learned precision and penalize the learned variance.

    References:
      https://arxiv.org/abs/1705.07115
        Kendall, Gal and Cipolla 2018, "Multi-Task Learning Using Uncertainty
        to Weigh Losses for Scene Geometry and Semantics."

    """

    class Config(Fig["UncertaintyLoss"]):
        """No parameters besides the learned log-variances, which start at zero."""

        def cost(
            self,
            *,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the weighted sum and the two learned log-variances.

            Args:
              dtype: Loss dtype; ``None`` is torch's default.
              **kwargs: The open bus; unused.

            Returns:
              cost: A few scalar operations on the two losses, owning the log-variances.

            """
            del kwargs
            return _combination_cost(params=2, dtype=dtype)

    def __init__(self, config: Config) -> None:
        del config
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(2))
        """Log-variance of each task, fine then coarse; trained by the optimizer."""

    @override
    def forward(
        self,
        loss_fine: Tensor,
        loss_coarse: Tensor,
        epoch: int,
        total_epochs: int,
    ) -> Tensor:
        """Return ``sum_i exp(-s_i) * L_i + s_i`` for log-variances ``s``."""
        del epoch, total_epochs
        losses = [loss_fine, loss_coarse]
        # Python's ``sum`` starts from the integer 0, as the reference's does.
        return cast(
            Tensor,
            sum(
                torch.exp(-self.log_vars[i]) * losses[i] + self.log_vars[i]
                for i in range(2)
            ),
        )


class DynamicWeightAverage(nn.Module):
    """Weight each loss by how slowly it fell since the previous step.

    References:
      https://arxiv.org/abs/1803.10704
        Liu, Johns and Davison 2019, "End-to-End Multi-Task Learning with
        Attention."

    """

    class Config(Fig["DynamicWeightAverage"]):
        """Temperature and guard of the weighting."""

        temperature: float = 2.0
        """Softens the softmax over the loss ratios; larger is more uniform."""

        eps: float = 1e-8
        """Added to the previous losses before dividing by them."""

        def cost(
            self,
            *,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the softmax-weighted sum of two scalars.

            Args:
              dtype: Loss dtype; ``None`` is torch's default.
              **kwargs: The open bus; unused.

            Returns:
              cost: A few scalar operations on the two losses.

            """
            del kwargs
            return _combination_cost(params=0, dtype=dtype)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.temperature = config.temperature
        self.eps = config.eps
        self.prev_losses: Tensor | None = None
        """Both losses of the previous step, detached; ``None`` before the first."""

    @override
    def forward(
        self,
        loss_fine: Tensor,
        loss_coarse: Tensor,
        epoch: int,
        total_epochs: int,
    ) -> Tensor:
        """Return ``sum_i w_i * L_i`` with ``w = 2 softmax(L / L_prev / T)``.

        Args:
          loss_fine: Scalar loss of the fine head.
          loss_coarse: Scalar loss of the coarse head.
          epoch: Completed epochs; every step of epoch 0 weights both by one.
          total_epochs: Unused.

        Returns:
          loss: The weighted sum; the weights carry no gradient.

        """
        del total_epochs
        losses = torch.stack([loss_fine, loss_coarse])
        if self.prev_losses is None or epoch == 0:
            weights = torch.ones(2, device=losses.device)
        else:
            # A checkpoint restores the history on the CPU whatever device it
            # was saved from.
            previous = self.prev_losses.to(losses.device)
            ratios = losses.detach() / (previous + self.eps)
            weights = 2 * functional.softmax(ratios / self.temperature, dim=0)
        self.prev_losses = losses.detach().clone()
        return (weights * losses).sum()

    # Extra state rather than a buffer: the history starts absent, and torch loads
    # a buffer only into one that already holds a tensor of the saved shape.
    @override
    def get_extra_state(self) -> Tensor | None:
        """Return the previous step's losses, so a checkpoint carries them."""
        return self.prev_losses

    @override
    def set_extra_state(self, state: object) -> None:
        """Restore the previous step's losses from :meth:`get_extra_state`."""
        assert state is None or isinstance(state, Tensor)
        self.prev_losses = state


class PCGrad(nn.Module):
    """Project each task's gradient off the other's when the two conflict.

    The gradients are taken with respect to the leaves of the fine loss's
    graph, found once on the first call by :func:`graph_leaves` and kept. The
    returned value is a surrogate, ``sum_p <g_p, p>`` with ``g`` detached, so
    backpropagating it leaves the projected gradient ``g`` on each of those
    leaves and nothing on any other parameter, the coarse head's included.

    The reference's walk is NONDETERMINISTIC: it dedupes graph nodes by the
    ``id`` of short-lived Python wrappers, whose addresses are reused, so it
    skips a heap-dependent subset of leaves (measured on one graph: 22, 23,
    then 22 of ResNet-8's 29; 168 or 173 of ResNet-56's 173, by process).
    This port takes the full set, all 173 for ResNet-56. The capture pins the
    reference's instance to that same set and checks that the reference's
    own walk is always a subsequence of it.

    References:
      https://arxiv.org/abs/2001.06782
        Yu et al. 2020, "Gradient Surgery for Multi-Task Learning."

    """

    class Config(Fig["PCGrad"]):
        """Guard of the projection."""

        eps: float = 1e-12
        """Added to each squared norm before dividing by it."""

        def cost(
            self,
            *,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the projection's scalar bookkeeping; its gradients are the step's backward.

            Args:
              dtype: Loss dtype; ``None`` is torch's default.
              **kwargs: The open bus; unused.

            Returns:
              cost: A few scalar operations on the two losses.

            """
            del kwargs
            return _combination_cost(params=0, dtype=dtype)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.eps = config.eps
        self.shared: list[Tensor] | None = None
        """Leaves the gradients are taken against; found on the first call."""

    @override
    def forward(
        self,
        loss_fine: Tensor,
        loss_coarse: Tensor,
        epoch: int,
        total_epochs: int,
    ) -> Tensor:
        """Return a surrogate whose gradient is the projected sum of the task gradients.

        Args:
          loss_fine: Scalar loss of the fine head, with its graph.
          loss_coarse: Scalar loss of the coarse head, with its graph.
          epoch: Unused.
          total_epochs: Unused.

        Returns:
          loss: ``loss_fine + loss_coarse`` when the fine loss has no leaves;
            otherwise the surrogate.

        """
        del epoch, total_epochs
        if self.shared is None:
            self.shared = graph_leaves(loss_fine)
        params = self.shared
        if not params:
            return loss_fine + loss_coarse
        g0 = _flat_grad(loss_fine, params)
        g1 = _flat_grad(loss_coarse, params)
        dot = torch.dot(g0, g1)
        if dot < 0:
            # Both projections use the original gradients.
            g0_norm_sq = torch.dot(g0, g0) + self.eps
            g1_norm_sq = torch.dot(g1, g1) + self.eps
            g0, g1 = g0 - (dot / g1_norm_sq) * g1, g1 - (dot / g0_norm_sq) * g0
        g_pcgrad = g0 + g1
        offset = 0
        surrogate = torch.tensor(0.0, device=loss_fine.device)
        for p in params:
            numel = p.numel()
            chunk = g_pcgrad[offset : offset + numel].reshape(p.shape).detach()
            surrogate = surrogate + (chunk * p).sum()
            offset += numel
        return surrogate


def _combination_cost(*, params: int, dtype: torch.dtype | None) -> Cost:
    """Cost combining two scalar losses, owning ``params`` trained parameters."""
    return elementwise_cost(
        primal=2,
        adjoint=2,
        channels=2,
        params=params,
        dtype=dtype,
        inputs=2,
    )


def graph_leaves(loss: Tensor) -> list[Tensor]:
    """Return the leaves requiring grad reached by a depth-first walk of ``loss``'s graph.

    Children are visited in the order of ``next_functions``, a node once, as
    the reference's walk does. The reference marks a node seen by the ``id``
    of its Python wrapper and lets the wrapper go; a later wrapper can reuse
    the freed address, so its walk skips some leaves, and which ones varies
    from call to call with the heap. Every wrapper is kept alive here until
    the walk ends, so each ``id`` names one node and every leaf is found.

    Args:
      loss: A scalar with a graph.

    Returns:
      leaves: In the order the walk first reached them.

    """
    params: list[Tensor] = []
    alive: list[Node] = []
    _walk(loss.grad_fn, params, set(), alive)
    return params


def _walk(
    grad_fn: Node | None,
    params: list[Tensor],
    seen: set[int],
    alive: list[Node],
) -> None:
    """Visit every child of ``grad_fn`` not yet seen, collecting leaves."""
    if grad_fn is None:
        return
    for child, _ in grad_fn.next_functions:
        if child is None:
            continue
        alive.append(child)
        cid = id(child)
        if cid in seen:
            continue
        seen.add(cid)
        # Only an AccumulateGrad node carries ``variable``: the leaf it feeds.
        p = cast(object, getattr(child, "variable", None))
        if isinstance(p, Tensor) and p.requires_grad:
            params.append(p)
        _walk(child, params, seen, alive)


def _flat_grad(loss: Tensor, params: list[Tensor]) -> Tensor:
    """Return ``d loss / d params`` joined into one vector, zeros where unused."""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            g.flatten() if g is not None else torch.zeros_like(p).flatten()
            for g, p in zip(grads, params, strict=True)
        ],
    )
