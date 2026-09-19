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

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
    resolve_dtype,
    traffic,
)
from priml.lib.custom_json import ListCodec
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
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the gate matmul, the top-k pick, and the picked weights' gather.

            The correction bias is a buffer, not a weight. Top-k traffic is
            its minimum operand I/O (input plus selected values and indices),
            not an implementation-dependent sorting-pass or scratch estimate.
            The ``top_k`` picked indices are ``int64``; the scores and the
            picked weights are at the batch's dtype.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            del kwargs
            rows = seq_len * batch_size
            dt = resolve_dtype(dtype)
            s = dt.itemsize
            index = torch.int64
            k, e = self.top_k, self.num_experts
            total = matmul_cost(
                channels_in=self.channels_in,
                channels_out=e,
                bias=False,
                rows=rows,
                dtype=dt,
            ) + Cost(
                cells={
                    ("flops", "adjoint", "selection", dt): rows * k,
                    ("bytes", "primal", "sort", dt): rows * s * (e + k),
                    ("bytes", "primal", "sort", index): rows * index.itemsize * k,
                    ("bytes", "primal", "selection", dt): rows * s * 2 * k,
                    ("bytes", "primal", "selection", index): rows * index.itemsize * k,
                    ("bytes", "adjoint", "selection", dt): rows * s * (3 * k + e),
                    ("bytes", "adjoint", "selection", index): rows * index.itemsize * k,
                },
            )
            if self.norm_topk_prob:
                # Sum the picked weights, then divide each by the sum.
                total += (
                    elementwise_cost(
                        primal=rows * k,
                        adjoint=rows * 4 * k,
                        channels=k,
                        rows=rows,
                        inputs=1,
                        outputs=1,
                        adjoint_inputs=5,
                        adjoint_outputs=4,
                        dtype=dt,
                    )
                    + reduction_cost(
                        input_elements=rows * k,
                        output_groups=rows,
                        dtype=dt,
                    )
                    + traffic(
                        "primal",
                        "elementwise",
                        elements=rows,
                        dtype=dt,
                    )
                    + traffic(
                        "adjoint",
                        "elementwise",
                        elements=rows * 3,
                        dtype=dt,
                    )
                    + traffic(
                        "adjoint",
                        "reduction",
                        elements=rows * (k + 1),
                        dtype=dt,
                    )
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
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count stable softmax and optional training jitter.

            Softmax over ``E`` experts: max and sum are two reductions of
            ``E - 1``; subtract, exp, and divide are ``3E`` elementwise. The
            adjoint's dot product is the same split.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            rows = seq_len * batch_size
            dt = resolve_dtype(dtype)
            jitter = self.channels_in if self.jitter_noise > 0 else 0
            width = self.num_experts
            return (
                super().cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + elementwise_cost(
                    primal=rows * (3 * width + jitter),
                    adjoint=rows * (3 * width + jitter),
                    channels=width,
                    rows=rows,
                    inputs=3,
                    outputs=3,
                    adjoint_inputs=5,
                    adjoint_outputs=3,
                    dtype=dt,
                )
                + reduction_cost(
                    input_elements=rows * width,
                    output_groups=rows,
                    dtype=dt,
                ).tile(2)
                + traffic(
                    "primal",
                    "elementwise",
                    elements=rows * (2 + 4 * jitter),
                    dtype=dt,
                )
                + reduction_cost(
                    input_elements=rows * width,
                    output_groups=rows,
                    dtype=dt,
                    phase="adjoint",
                )
                + traffic(
                    "adjoint",
                    "elementwise",
                    elements=rows * (1 + 3 * jitter),
                    dtype=dt,
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
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count sigmoid, selection-only bias/group sums, and weight scaling.

            Grouped routing sorts each group for its top-2 and the groups for
            ``topk_group``; the group sums are reductions. Picked group and
            expert indices are ``int64``.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            rows = seq_len * batch_size
            dt = resolve_dtype(dtype)
            s = dt.itemsize
            index = torch.int64
            width = self.num_experts
            bias = width if self.use_correction_bias else 0
            groups = Cost()
            if self.n_group > 1:
                picks = min(2, width // self.n_group)
                groups = Cost(
                    cells={
                        ("flops", "primal", "reduction", dt): rows
                        * self.n_group
                        * (picks - 1),
                        ("bytes", "primal", "sort", dt): rows
                        * s
                        * (
                            width
                            + self.n_group * picks
                            + self.n_group
                            + self.topk_group
                        ),
                        ("bytes", "primal", "sort", index): rows
                        * index.itemsize
                        * (self.n_group * picks + self.topk_group),
                        ("bytes", "primal", "reduction", dt): rows
                        * s
                        * self.n_group
                        * (picks + 1),
                        ("bytes", "primal", "selection", dt): rows
                        * s
                        * (self.topk_group + 5 * width),
                        ("bytes", "primal", "selection", index): rows
                        * index.itemsize
                        * (self.n_group + self.topk_group),
                    },
                )
            return (
                super().cost(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + elementwise_cost(
                    primal=rows * (4 * width + bias + self.top_k),
                    adjoint=rows * (3 * width + self.top_k),
                    channels=width,
                    rows=rows,
                    dtype=dt,
                )
                + traffic(
                    "primal",
                    "elementwise",
                    elements=rows * (2 * bias + 2 * self.top_k) + bias,
                    dtype=dt,
                )
                + traffic(
                    "adjoint",
                    "elementwise",
                    elements=rows * 2 * self.top_k,
                    dtype=dt,
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
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            expert_rows: tuple[int, ...] | None = None,
            **kwargs: object,
        ) -> Cost:
            """Cost one routed invocation at its realized expert occupancy.

            ``expert_rows[i]`` is the number of dispatched rows executed by
            expert ``i``. The tuple is required because expert activation bytes
            depend on the realized routing distribution; balanced occupancy is
            only an estimate and is never represented as exact ``Cost``.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              expert_rows: Realized dispatched rows for every routed expert.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost of this module.

            Raises:
              ValueError: Realized routing occupancy is absent or inconsistent.

            """
            rows = seq_len * batch_size
            if expert_rows is None:
                raise ValueError(
                    "MoE cost requires expert_rows for exact realized routing occupancy.",
                )
            if len(expert_rows) != self.router.num_experts:
                raise ValueError(
                    f"expert_rows has length {len(expert_rows)}; expected "
                    f"{self.router.num_experts}.",
                )
            if any(type(count) is not int or count < 0 for count in expert_rows):
                raise ValueError("expert_rows must contain nonnegative integers.")
            top_k = self.router.top_k
            expected_rows = rows * top_k
            if sum(expert_rows) != expected_rows:
                raise ValueError(
                    f"expert_rows sums to {sum(expert_rows)}; expected "
                    f"{expected_rows} for {rows} rows and top_k={top_k}.",
                )
            dt = resolve_dtype(dtype)
            s = dt.itemsize
            index = torch.int64
            one_expert = cost(
                self.expert,
                seq_len=1,
                batch_size=1,
                dtype=dtype,
                **kwargs,
            )
            routed = sum(
                (
                    cost(
                        self.expert,
                        seq_len=count,
                        batch_size=1,
                        dtype=dtype,
                        **kwargs,
                    )
                    for count in expert_rows
                    if count
                ),
                Cost(),
            )
            routed = replace(
                routed,
                params=self.router.num_experts * one_expert.params,
                params_active=sum(
                    one_expert.params_active for count in expert_rows if count
                ),
            )
            shared = Cost()
            if self.num_shared_experts:
                shared = cost(
                    self.shared_expert,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                ).tile(self.num_shared_experts, copies=self.num_shared_experts)
            width = self.channels_out
            width_in = self.channels_in
            n_shared = self.num_shared_experts
            # Per routed assignment, eleven scalar bookkeeping elements: the
            # expert id, the position, and the permutation entries that route
            # a row out and back. Those are the indices; the rest is payload.
            dispatch = Cost(
                cells={
                    ("flops", "primal", "elementwise", dt): rows * top_k * width,
                    ("flops", "primal", "selection", dt): rows
                    * (top_k + n_shared)
                    * width,
                    ("flops", "adjoint", "elementwise", dt): rows * 2 * top_k * width,
                    ("flops", "adjoint", "reduction", dt): rows * top_k * (width - 1),
                    ("flops", "adjoint", "selection", dt): rows
                    * (top_k + n_shared)
                    * width_in,
                    ("bytes", "primal", "sort", index): rows
                    * index.itemsize
                    * 2
                    * top_k,
                    ("bytes", "primal", "elementwise", dt): rows
                    * s
                    * top_k
                    * (2 * width + 1),
                    ("bytes", "primal", "selection", dt): rows
                    * s
                    * (
                        top_k * (2 * width_in + 3 * width)
                        + width
                        + 3 * n_shared * width
                    ),
                    ("bytes", "primal", "selection", index): rows
                    * index.itemsize
                    * top_k
                    * 11,
                    ("bytes", "adjoint", "elementwise", dt): rows
                    * s
                    * top_k
                    * (5 * width + 1),
                    ("bytes", "adjoint", "reduction", dt): rows
                    * s
                    * top_k
                    * (width + 1),
                    ("bytes", "adjoint", "selection", dt): rows
                    * s
                    * (
                        top_k * (3 * width_in + 2 * width)
                        + width_in
                        + 3 * n_shared * width_in
                    ),
                    ("bytes", "adjoint", "selection", index): rows
                    * index.itemsize
                    * top_k
                    * 11,
                },
            )
            return (
                cost(
                    self.router,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
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
