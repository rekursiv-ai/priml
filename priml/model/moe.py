"""Mixture-of-experts layer.

Covers softmax (Switch Transformer) and sigmoid + bias-corrected
routing (DeepSeek-V3 / Kimi-K2). Shared experts, grouped top-k, and
``routed_scaling_factor`` are optional extensions used by the
sigmoid-routed variants. Dispatch is sort-and-dispatch: one expert
forward per *active* expert, not per registered expert.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, field, replace
from functools import partial
from typing import Literal, Protocol, Self, cast, override, runtime_checkable

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn

import torch

from priml.lib.custom_json import ListCodec
from priml.model.cost import (
    Bytes,
    Compute,
    Cost,
    Flops,
    cost,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
)
from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    DepthIndex,
    HasDepthIndex,
    HasResetParameters,
    Shardable,
    TensorModule,
    propagate_attr,
)
from priml.model.swiglu import SwiGLU


@runtime_checkable
class TokenRouter(Protocol):
    """Assigns each token to ``top_k`` experts and weights them."""

    @property
    def scoring_func(self) -> str:
        """Gate activation.

        ``MoE`` reads it to decide whether the load-balancing auxiliary loss applies:
        softmax routing needs it, sigmoid routing is aux-loss-free and carries its
        balance in the router's own bias.

        Read-only, so an implementation may hold it at a narrower type -- a
        mutable ``str`` member is invariant and would reject the ``Literal``
        the shipped ``Router`` stores.
        """
        ...

    def __call__(
        self,
        x: Tensor,
        /,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(weights, indices, logits)`` for one flat token batch."""
        ...

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        ...


@runtime_checkable
class TokenRouterConfig(Makeable[TokenRouter], Protocol):
    """What ``MoE`` needs of whatever fills its ``router`` slot.

    The width it must be told, and the two counts ``MoE`` reads back: how many
    experts to BUILD and how many gates each token carries. Stated as a shape
    rather than as ``Router.Config`` so a different routing rule -- a learned
    hash, a fixed assignment -- drops into the slot without editing ``MoE``.
    """

    channels_in: int
    """Model width, pushed down by ``MoE.finalize``."""

    num_experts: int
    """Experts to build from the ``expert`` template."""

    top_k: int
    """Experts each token is routed to."""


class Router(nn.Module):
    """Top-k token router over a linear gate; a subclass picks the activation.

    Holds what every gate shares -- the projection, the top-k pick, and the
    optional renormalization of the picked weights. :class:`SoftmaxRouter` and
    :class:`SigmoidRouter` each fix ``scoring_func`` in their ``__init__``, so
    this base is not buildable on its own: ``Router.Config().make()`` raises.
    """

    class Config(Fig["Router"], kw_only=False):
        channels_in: int = -1
        """Number of input channels."""

        _: KW_ONLY

        num_experts: int = 8
        """Total number of experts to route across."""

        top_k: int = 2
        """Number of experts each token is routed to."""

        norm_topk_prob: bool = False
        """Renormalize the top-k weights to sum to 1 after selection."""

        @override
        def finalize(self) -> Self:
            if self.top_k < 1 or self.top_k > self.num_experts:
                raise ValueError(
                    f"top_k={self.top_k} must satisfy 1 <= top_k <= "
                    f"num_experts={self.num_experts}.",
                )
            return super().finalize()

        def cost(
            self,
            *,
            rows: float = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Price the gate matmul, the top-k pick, and the picked weights' gather.

            The correction bias is a buffer, not a weight. Top-k traffic is
            its minimum operand I/O (input plus selected values and indices),
            not an implementation-dependent sorting-pass or scratch estimate.
            Indices use the uniform analytical itemsize too.

            Args:
              rows: Rows sharing each parameter and its gradient reduction.
              itemsize: Uniform bytes per operand element, including indices.
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del kwargs
            total = matmul_cost(
                channels_in=self.channels_in,
                channels_out=self.num_experts,
                bias=False,
                rows=rows,
                itemsize=itemsize,
            ) + Cost(
                primal=Compute(
                    bytes=Bytes(
                        sort=itemsize * (self.num_experts + 2 * self.top_k),
                        selection=itemsize * 3 * self.top_k,
                    ),
                ),
                adjoint=Compute(
                    flops=Flops(selection=self.top_k),
                    bytes=Bytes(
                        selection=itemsize * (4 * self.top_k + self.num_experts),
                    ),
                ),
            )
            if self.norm_topk_prob:
                # Sum the picked weights, then divide each by the sum.
                total += elementwise_cost(
                    primal=self.top_k,
                    adjoint=4 * self.top_k,
                    channels=self.top_k,
                    inputs=1,
                    outputs=1,
                    adjoint_inputs=5,
                    adjoint_outputs=4,
                    itemsize=itemsize,
                ) + Cost(
                    primal=reduction_cost(input_elements=self.top_k, itemsize=itemsize)
                    + Compute(bytes=Bytes(elementwise=itemsize)),
                    adjoint=Compute(
                        bytes=Bytes(
                            elementwise=3 * itemsize,
                            reduction=itemsize * (self.top_k + 1),
                        ),
                    ),
                )
            return total

    def __init__(
        self,
        config: Config,
        *,
        scoring_func: Literal["softmax", "sigmoid"],
    ) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.norm_topk_prob = config.norm_topk_prob
        self.scoring_func: Literal["softmax", "sigmoid"] = scoring_func
        self.gate = nn.Linear(config.channels_in, config.num_experts, bias=False)

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        nn.init.kaiming_uniform_(self.gate.weight, a=5**0.5)

    @override
    def forward(
        self,
        x: Tensor,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (weights, indices, logits).

        ``weights`` and ``indices`` are shape ``[T, top_k]``; ``logits``
        is ``[T, num_experts]`` (pre-activation, used by the load-
        balance loss for softmax routing).
        """
        del kwargs
        logits = self.gate(x)
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = logits.softmax(dim=-1)
        _, indices = self._selection(scores).topk(self.top_k, dim=-1)
        weights = scores.gather(-1, indices)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        return weights, indices, logits

    def _selection(self, scores: Tensor) -> Tensor:
        """Rank experts for the top-k pick; the gate weights stay ``scores``."""
        return scores


class SoftmaxRouter(Router):
    """Switch Transformer routing: softmax gate, balanced by an auxiliary loss.

    ``MoE`` computes the load-balancing loss from the logits this returns.
    """

    class Config(Makes["SoftmaxRouter"], Router.Config, kw_only=False):
        _: KW_ONLY

        jitter_noise: float = 0.0
        """Multiplicative uniform noise scale for training regularization."""

        @override
        def cost(
            self,
            *,
            rows: float = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Count stable softmax and optional training jitter.

            Softmax over ``E`` experts: max and sum are two reductions of
            ``E - 1``; subtract, exp, and divide are ``3E`` elementwise. The
            adjoint's dot product is the same split.

            Args:
              rows: Rows sharing each parameter and its gradient reduction.
              itemsize: Uniform bytes per operand element, including indices.
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            jitter = self.channels_in if self.jitter_noise > 0 else 0
            width = self.num_experts
            return (
                super().cost(rows=rows, itemsize=itemsize, **kwargs)
                + elementwise_cost(
                    primal=3 * width + jitter,
                    adjoint=3 * width + jitter,
                    channels=width,
                    inputs=3,
                    outputs=3,
                    adjoint_inputs=5,
                    adjoint_outputs=3,
                    itemsize=itemsize,
                )
                + Cost(
                    primal=2 * reduction_cost(input_elements=width, itemsize=itemsize)
                    + Compute(bytes=Bytes(elementwise=itemsize * (2 + 4 * jitter))),
                    adjoint=reduction_cost(input_elements=width, itemsize=itemsize)
                    + Compute(bytes=Bytes(elementwise=itemsize * (1 + 3 * jitter))),
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__(config, scoring_func="softmax")
        self.jitter_noise = config.jitter_noise

    @override
    def forward(
        self,
        x: Tensor,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.training and self.jitter_noise > 0:
            x = x * torch.empty_like(x).uniform_(
                1 - self.jitter_noise,
                1 + self.jitter_noise,
            )
        return super().forward(x, **kwargs)


class SigmoidRouter(Router):
    """DeepSeek-V3 / Kimi-K2 routing: sigmoid gate, aux-loss-free.

    A per-expert ``e_score_correction_bias`` shifts *selection* but not the
    returned gate weights, so the load balancer steers assignment without
    an auxiliary loss. Optional grouped top-k partitions the experts into
    ``n_group`` groups and keeps only ``topk_group`` of them per token.
    """

    class Config(Makes["SigmoidRouter"], Router.Config, kw_only=False):
        _: KW_ONLY

        norm_topk_prob: bool = True
        """Renormalize the top-k weights to sum to 1 after selection."""

        use_correction_bias: bool = True
        """Maintain ``e_score_correction_bias`` for aux-loss-free balancing."""

        routed_scaling_factor: float = 1.0
        """Multiplier on top-k weights. DSV3 uses 2.5; Kimi-K2 uses 2.827."""

        n_group: int = 1
        """Number of expert groups for grouped top-k (1 = no grouping)."""

        topk_group: int = 1
        """Number of groups to keep when ``n_group > 1``."""

        @override
        def finalize(self) -> Self:
            if self.num_experts % self.n_group != 0:
                raise ValueError(
                    f"num_experts={self.num_experts} must be divisible by "
                    f"n_group={self.n_group}.",
                )
            if self.topk_group > self.n_group:
                raise ValueError(
                    f"topk_group={self.topk_group} > n_group={self.n_group}.",
                )
            if self.n_group > 1:
                eligible = self.topk_group * (self.num_experts // self.n_group)
                if self.top_k > eligible:
                    raise ValueError(
                        f"top_k={self.top_k} exceeds the {eligible} experts "
                        f"eligible after grouped routing (topk_group="
                        f"{self.topk_group} groups of "
                        f"{self.num_experts // self.n_group}).",
                    )
            return super().finalize()

        @override
        def cost(
            self,
            *,
            rows: float = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Count sigmoid, selection-only bias/group sums, and weight scaling.

            Grouped routing sorts each group for its top-2 and the groups for
            ``topk_group``; the group sums are reductions.

            Args:
              rows: Rows sharing each parameter and its gradient reduction.
              itemsize: Uniform bytes per operand element, including indices.
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            width = self.num_experts
            bias = width if self.use_correction_bias else 0
            groups = Cost()
            if self.n_group > 1:
                group_size = width // self.n_group
                groups = Cost(
                    primal=Compute(
                        flops=Flops(reduction=self.n_group * (min(2, group_size) - 1)),
                        bytes=Bytes(
                            sort=itemsize
                            * (
                                width
                                + 2 * self.n_group * min(2, group_size)
                                + self.n_group
                                + 2 * self.topk_group
                            ),
                            reduction=itemsize
                            * self.n_group
                            * (min(2, group_size) + 1),
                            selection=itemsize
                            * (self.n_group + 2 * self.topk_group + 5 * width),
                        ),
                    ),
                )
            return (
                super().cost(rows=rows, itemsize=itemsize, **kwargs)
                + elementwise_cost(
                    primal=4 * width + bias + self.top_k,
                    adjoint=3 * width + self.top_k,
                    channels=width,
                    itemsize=itemsize,
                )
                + Cost(
                    primal=Compute(
                        bytes=Bytes(
                            elementwise=itemsize
                            * (2 * bias + bias / rows + 2 * self.top_k),
                        ),
                    ),
                    adjoint=Compute(bytes=Bytes(elementwise=itemsize * 2 * self.top_k)),
                )
                + groups
            )

    def __init__(self, config: Config) -> None:
        super().__init__(config, scoring_func="sigmoid")
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.e_score_correction_bias: Tensor | None
        if config.use_correction_bias:
            # Gradient-free: adjusted during training by the load
            # balancer, not by autograd. Buffer so load_state_dict
            # handles it without requires_grad surprises.
            self.register_buffer(
                "e_score_correction_bias",
                torch.zeros(config.num_experts),
            )
        else:
            self.e_score_correction_bias = None

    @override
    def reset_parameters(self) -> None:
        super().reset_parameters()
        if self.e_score_correction_bias is not None:
            self.e_score_correction_bias.zero_()

    @override
    def forward(
        self,
        x: Tensor,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor, Tensor]:
        weights, indices, logits = super().forward(x, **kwargs)
        return weights * self.routed_scaling_factor, indices, logits

    @override
    def _selection(self, scores: Tensor) -> Tensor:
        # Bias shifts which experts are picked, not the returned gate
        # weight (DSV3 decouples the two).
        selection = (
            scores + self.e_score_correction_bias
            if self.e_score_correction_bias is not None
            else scores
        )
        if self.n_group > 1:
            selection = self._mask_inactive_groups(selection)
        return selection

    def _mask_inactive_groups(self, selection: Tensor) -> Tensor:
        """Keep only ``topk_group`` groups live (DSV3 grouped routing)."""
        t = selection.shape[0]
        group_size = self.num_experts // self.n_group
        grouped = selection.view(t, self.n_group, group_size)
        # DSV3 convention: group score = sum of top-2 within the group.
        top2 = grouped.topk(min(2, group_size), dim=-1).values
        group_scores = top2.sum(dim=-1)
        top_groups = group_scores.topk(self.topk_group, dim=-1).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, top_groups, True)
        expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, group_size).reshape(t, -1)
        return selection.masked_fill(~expert_mask, float("-inf"))


class MoE(nn.Module):
    """Mixture-of-experts layer.

    Drop-in replacement for FFN. Routes each token to top-k experts
    via a :class:`Router`, plus optional always-active shared experts
    summed onto every token. Dispatch is sort-and-dispatch: tokens
    are grouped by expert so each *active* expert runs exactly one
    contiguous forward (vs. one forward per registered expert in the
    mask-per-expert form).

    Softmax routing stores the Switch Transformer load-balancing
    auxiliary loss in ``_aux_loss`` during training. Sigmoid routing
    is aux-loss-free (the bias in :class:`SigmoidRouter` handles
    balance); ``aux_loss_weight`` is ignored.
    """

    class Config(Fig["MoE"], kw_only=False):
        channels_in: int = -1
        """Number of input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        router: TokenRouterConfig = field(
            default_factory=partial[SoftmaxRouter.Config](SoftmaxRouter.Config),
        )
        """Token-to-expert assignment. Its ``num_experts`` is how many experts
        ``MoE`` builds from the ``expert`` template."""

        expert: Makeable[TensorModule] = field(default_factory=SwiGLU.Config)
        """Routed expert module config."""

        num_shared_experts: int = 0
        """Always-active experts summed onto every token's output.
        DSV3/Kimi-K2 use 1. 0 = no shared experts."""

        shared_expert: Makeable[TensorModule] = field(default_factory=SwiGLU.Config)
        """Shared expert config (instantiated ``num_shared_experts`` times).
        Ignored when ``num_shared_experts=0``."""

        aux_loss_weight: float = 0.01
        """Weight for the load-balancing auxiliary loss.
        Applied only with softmax routing (sigmoid is aux-loss-free)."""

        depth_index: DepthIndex = ()
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            self.router.channels_in = self.channels_in
            for cfg in (self.expert, self.shared_expert):
                propagate_attr(
                    cfg,
                    "channels_in",
                    self.channels_in,
                    protocol=ChannelsIn,
                )
                propagate_attr(
                    cfg,
                    "channels_out",
                    self.channels_out,
                    protocol=ChannelsOut,
                )
                propagate_attr(
                    cfg,
                    "depth_index",
                    self.depth_index,
                    protocol=HasDepthIndex,
                )
                # Each expert shards intra-expert over the tp dim (its own
                # block style handles the split alignment).
                if isinstance(cfg, Shardable):
                    cfg.shard = "colwise"
            return super().finalize()

        def cost(
            self,
            *,
            rows: float = 1,
            itemsize: int = 4,
            **kwargs: object,
        ) -> Cost:
            """Price the router, ``top_k`` routed experts, and every shared expert.

            ``params`` counts every routed expert; every other field counts what
            one token touches -- the ``top_k`` experts it is dispatched to and
            the shared experts, which are always active. Dispatch sorts the
            ``top_k`` assignments, gathers the token's row into each expert,
            and scatter-adds the gated outputs back; the adjoint's gate gradient
            is a dot product per routed expert. Expert batch reductions use the
            balanced occupancy ``max(1, rows * top_k / num_experts)``.
            This fractional average is an estimate; actual occupancy depends on
            routing. Indices, gate weights, and payloads use uniform itemsize.
            Dispatch counts assignment permutation and payload gather/scatter,
            excluding sorting workspace and the external auxiliary loss.

            Args:
              rows: Rows sharing each parameter and its gradient reduction.
              itemsize: Uniform bytes per operand element, including indices.
              **kwargs: The open message bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            top_k = self.router.top_k
            # Every expert exists, but one token runs ``top_k`` of them, so only
            # ``params`` follows the module count; ``tile`` would scale
            # ``params_active`` and the weight bytes by it too.
            expert = cost(
                self.expert,
                rows=max(1, rows * top_k / self.router.num_experts),
                itemsize=itemsize,
                **kwargs,
            )
            routed = replace(
                top_k * expert,
                params=self.router.num_experts * expert.params,
            )
            shared = Cost()
            if self.num_shared_experts:
                shared = self.num_shared_experts * cost(
                    self.shared_expert,
                    rows=rows,
                    itemsize=itemsize,
                    **kwargs,
                )
            width = self.channels_out
            width_in = self.channels_in
            dispatch = Cost(
                primal=Compute(
                    flops=Flops(
                        elementwise=top_k * width,
                        selection=(top_k + self.num_shared_experts) * width,
                    ),
                    bytes=Bytes(
                        sort=itemsize * 2 * top_k,
                        elementwise=itemsize * top_k * (2 * width + 1),
                        selection=itemsize
                        * (
                            top_k * (2 * width_in + 3 * width + 11)
                            + width
                            + 3 * self.num_shared_experts * width
                        ),
                    ),
                ),
                adjoint=Compute(
                    flops=Flops(
                        elementwise=2 * top_k * width,
                        reduction=top_k * (width - 1),
                        selection=(top_k + self.num_shared_experts) * width_in,
                    ),
                    bytes=Bytes(
                        elementwise=itemsize * top_k * (5 * width + 1),
                        reduction=itemsize * top_k * (width + 1),
                        selection=itemsize
                        * (
                            top_k * (3 * width_in + 2 * width + 11)
                            + width_in
                            + 3 * self.num_shared_experts * width_in
                        ),
                    ),
                ),
            )
            return (
                cost(self.router, rows=rows, itemsize=itemsize, **kwargs)
                + routed
                + shared
                + dispatch
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_experts = config.router.num_experts
        self.top_k = config.router.top_k
        self.channels_out = config.channels_out
        self.aux_loss_weight = config.aux_loss_weight
        self.depth_index = config.depth_index
        self.router = config.router.make()
        experts: list[nn.Module] = []
        for _ in range(self.num_experts):
            expert = config.expert.make()
            assert isinstance(expert, nn.Module)
            experts.append(expert)
        self.experts = nn.ModuleList(experts)
        shared_experts: list[nn.Module] = []
        for _ in range(config.num_shared_experts):
            expert = config.shared_expert.make()
            assert isinstance(expert, nn.Module)
            shared_experts.append(expert)
        self.shared_experts = nn.ModuleList(shared_experts)
        # Buffer (not a plain attribute) so ``.to(device)`` tracks it;
        # non-persistent since it is recomputed every training forward and
        # carries no learned state. The trailing assignment is routed into
        # the buffer dict by ``nn.Module.__setattr__`` and satisfies the
        # type checker's initialized-instance-variable check.
        self.register_buffer("_aux_loss", torch.tensor(0.0), persistent=False)
        self._aux_loss = torch.tensor(0.0)

    def reset_parameters(self) -> None:
        """Initialize every parameter in place."""
        # Sole init source for every owned tensor (meta-init audit
        # contract): ``_aux_loss`` is runtime scratch overwritten each
        # forward, but as a registered buffer it must still be reset here.
        self._aux_loss.zero_()
        self.router.reset_parameters()
        for group in (self.experts, self.shared_experts):
            for expert in group:
                if isinstance(expert, HasResetParameters):
                    expert.reset_parameters()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        rows = x_flat.shape[0]

        weights, indices, logits = self.router(x_flat, **kwargs)
        if (
            self.training
            and self.aux_loss_weight > 0
            and self.router.scoring_func == "softmax"
        ):
            self._aux_loss = self._load_balance_loss(logits, indices)

        y = self._dispatch_routed(
            x_flat,
            weights,
            indices,
            rows,
            **kwargs,
        )
        for shared in self.shared_experts:
            shared_out = cast(Tensor, shared(x_flat, **kwargs))
            y = y + shared_out
        return y.reshape(*shape[:-1], self.channels_out)

    # Produces one expert forward per *active* expert (at most ``top_k * rows``,
    # typically ≪ ``num_experts`` for large MoEs). Numerically equivalent to the mask-
    # per-expert form.
    def _dispatch_routed(
        self,
        x_flat: Tensor,
        weights: Tensor,
        indices: Tensor,
        rows: int,
        **kwargs: object,
    ) -> Tensor:
        """Sort (token, expert) pairs by expert; dispatch contiguously."""
        k = indices.shape[-1]
        flat_idx = indices.reshape(-1)
        flat_w = weights.reshape(-1)
        token_ix = (
            torch.arange(rows, device=x_flat.device).unsqueeze(-1).expand(-1, k)
        ).reshape(-1)

        order = flat_idx.argsort()
        sorted_expert = flat_idx[order]
        sorted_tok = token_ix[order]
        sorted_w = flat_w[order]

        unique_consecutive = cast(
            Callable[..., tuple[Tensor, Tensor]],
            torch.unique_consecutive,
        )
        result = unique_consecutive(sorted_expert, return_counts=True)
        active, counts = result
        offsets = torch.cumsum(counts, dim=0)
        starts = torch.cat([offsets.new_zeros(1), offsets[:-1]], dim=0)

        y = x_flat.new_zeros(x_flat.shape[0], self.channels_out)
        for expert_id, start, count in zip(
            ListCodec.coerce(active.tolist(), int),
            ListCodec.coerce(starts.tolist(), int),
            ListCodec.coerce(counts.tolist(), int),
            strict=True,
        ):
            end = start + count
            tok_slice = sorted_tok[start:end]
            w_slice = sorted_w[start:end].unsqueeze(-1)
            x_e = x_flat.index_select(0, tok_slice)
            expert_out = cast(Tensor, self.experts[expert_id](x_e, **kwargs))
            y.index_add_(0, tok_slice, expert_out * w_slice)
        return y

    def _load_balance_loss(self, logits: Tensor, indices: Tensor) -> Tensor:
        """Switch Transformer load-balancing loss (softmax routing only)."""
        t = logits.shape[0]
        probs = logits.softmax(dim=-1)
        counts = torch.zeros(
            self.num_experts,
            device=logits.device,
            dtype=logits.dtype,
        )
        for k in range(self.top_k):
            counts.scatter_add_(
                0,
                indices[:, k],
                torch.ones(t, device=logits.device, dtype=logits.dtype),
            )
        freq = counts / (t * self.top_k)
        mean_probs = probs.mean(dim=0)
        return self.num_experts * (freq * mean_probs).sum() * self.aux_loss_weight
