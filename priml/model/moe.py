"""Mixture-of-experts layer.

Covers softmax (Switch Transformer) and sigmoid + bias-corrected
routing (DeepSeek-V3 / Kimi-K2). Shared experts, grouped top-k, and
``routed_scaling_factor`` are optional extensions used by the
sigmoid-routed variants. Dispatch is sort-and-dispatch: one expert
forward per *active* expert, not per registered expert.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Any, Literal, Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    ShardableConfig,
    propagate_attr,
)
from priml.model.swiglu import SwiGLU


class Router(nn.Module):
    """Top-k token router.

    Softmax routing (Switch Transformer) returns load-balancing
    counts via an auxiliary loss that ``MoE`` computes. Sigmoid
    routing adds a per-expert ``e_score_correction_bias`` (aux-loss-
    free, DSV3 convention) that shifts *selection* but not the
    returned gate weights, plus optional group top-k where experts
    are partitioned into ``n_group`` groups and only ``topk_group``
    groups are eligible per token.
    """

    class Config(Fig["Router"], kw_only=False):
        channels_in: int = -1
        """Number of input channels."""

        _: KW_ONLY

        num_experts: int = 8
        """Total number of experts to route across."""

        top_k: int = 2
        """Number of experts each token is routed to."""

        jitter_noise: float = 0.0
        """Multiplicative uniform noise scale for training regularization.
        Softmax routing only."""

        scoring_func: Literal["softmax", "sigmoid"] = "softmax"
        """Gate activation. DSV3/Kimi-K2 use ``sigmoid``."""

        norm_topk_prob: bool | None = None
        """Renormalize top-k weights to sum to 1 after selection.
        ``None`` infers: ``True`` for sigmoid routing, ``False`` for
        softmax (softmax already sums to 1). Set explicitly to override."""

        routed_scaling_factor: float = 1.0
        """Multiplier on top-k weights. DSV3 uses 2.5; Kimi-K2 uses 2.827."""

        n_group: int = 1
        """Number of expert groups for grouped top-k (1 = no grouping)."""

        topk_group: int = 1
        """Number of groups to keep when ``n_group > 1``."""

        use_correction_bias: bool | None = None
        """Maintain ``e_score_correction_bias`` for aux-loss-free routing.
        ``None`` infers: ``True`` for sigmoid routing, ``False`` otherwise.
        Set explicitly to override."""

        @override
        def finalize(self) -> Self:
            sigmoid = self.scoring_func == "sigmoid"
            if self.norm_topk_prob is None:
                self.norm_topk_prob = sigmoid
            if self.use_correction_bias is None:
                self.use_correction_bias = sigmoid
            if self.num_experts % self.n_group != 0:
                raise ValueError(
                    f"num_experts={self.num_experts} must be divisible by "
                    f"n_group={self.n_group}.",
                )
            if self.topk_group > self.n_group:
                raise ValueError(
                    f"topk_group={self.topk_group} > n_group={self.n_group}.",
                )
            if not 1 <= self.top_k <= self.num_experts:
                raise ValueError(
                    f"top_k={self.top_k} must satisfy 1 <= top_k <= "
                    f"num_experts={self.num_experts}.",
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

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.scoring_func not in ("softmax", "sigmoid"):
            raise ValueError(
                f"scoring_func={config.scoring_func!r} must be one of "
                f"'softmax' or 'sigmoid'.",
            )
        # ``finalize`` resolves the ``None`` sentinels to concrete bools.
        assert config.norm_topk_prob is not None
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.jitter_noise = config.jitter_noise
        self.scoring_func = config.scoring_func
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.gate = nn.Linear(config.channels_in, config.num_experts, bias=False)
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

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.gate.weight, a=5**0.5)
        if self.e_score_correction_bias is not None:
            self.e_score_correction_bias.zero_()

    @override
    def forward(
        self,
        x: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return (weights, indices, logits).

        ``weights`` and ``indices`` are shape ``[T, top_k]``; ``logits``
        is ``[T, num_experts]`` (pre-activation, used by the load-
        balance loss for softmax routing).
        """
        del args, kwargs
        if self.training and self.jitter_noise > 0:
            x = x * torch.empty_like(x).uniform_(
                1 - self.jitter_noise,
                1 + self.jitter_noise,
            )
        logits = self.gate(x)
        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = logits.softmax(dim=-1)
        # Selection ranking (bias-corrected for sigmoid; identity for
        # softmax). Bias shifts which experts are picked, not the
        # returned gate weight (DSV3 decouples the two).
        selection = (
            scores + self.e_score_correction_bias
            if self.e_score_correction_bias is not None
            else scores
        )
        if self.n_group > 1:
            selection = self._mask_inactive_groups(selection)
        _, indices = selection.topk(self.top_k, dim=-1)
        weights = scores.gather(-1, indices)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        weights = weights * self.routed_scaling_factor
        return weights, indices, logits

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
    via :class:`Router`, plus optional always-active shared experts
    summed onto every token. Dispatch is sort-and-dispatch: tokens
    are grouped by expert so each *active* expert runs exactly one
    contiguous forward (vs. one forward per registered expert in the
    mask-per-expert form).

    Softmax routing stores the Switch Transformer load-balancing
    auxiliary loss in ``_aux_loss`` during training. Sigmoid routing
    is aux-loss-free (the bias in :class:`Router` handles balance);
    ``aux_loss_weight`` is ignored.
    """

    class Config(Fig["MoE"], kw_only=False):
        channels_in: int = -1
        """Number of input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        router: Router.Config = field(default_factory=Router.Config)
        """Router config for token-to-expert assignment."""

        expert: Makeable[nn.Module] = field(default_factory=SwiGLU.Config)
        """Routed expert module config."""

        num_shared_experts: int = 0
        """Always-active experts summed onto every token's output.
        DSV3/Kimi-K2 use 1. 0 = no shared experts."""

        shared_expert: Makeable[nn.Module] = field(default_factory=SwiGLU.Config)
        """Shared expert config (instantiated ``num_shared_experts`` times).
        Ignored when ``num_shared_experts=0``."""

        aux_loss_weight: float = 0.01
        """Weight for the load-balancing auxiliary loss.
        Applied only with softmax routing (sigmoid is aux-loss-free)."""

        depth: int = -1
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
                    cfg, "channels_in", self.channels_in, protocol=ChannelsIn
                )
                propagate_attr(
                    cfg, "channels_out", self.channels_out, protocol=ChannelsOut
                )
                propagate_attr(cfg, "depth", self.depth)
                # Each expert shards intra-expert over the tp dim (its own
                # block style handles the split alignment).
                if isinstance(cfg, ShardableConfig):
                    cfg.shard = "colwise"
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_experts = config.router.num_experts
        self.top_k = config.router.top_k
        self.channels_out = config.channels_out
        self.aux_loss_weight = config.aux_loss_weight
        self.depth = config.depth
        self.router = config.router.make()
        self.experts = nn.ModuleList(
            [config.expert.make() for _ in range(self.num_experts)],
        )
        self.shared_experts = nn.ModuleList(
            [config.shared_expert.make() for _ in range(config.num_shared_experts)],
        )
        # Buffer (not a plain attribute) so ``.to(device)`` tracks it;
        # non-persistent since it is recomputed every training forward and
        # carries no learned state. The trailing assignment is routed into
        # the buffer dict by ``nn.Module.__setattr__`` and satisfies the
        # type checker's initialized-instance-variable check.
        self.register_buffer("_aux_loss", torch.tensor(0.0), persistent=False)
        self._aux_loss = torch.tensor(0.0)

    def reset_parameters(self) -> None:
        # Sole init source for every owned tensor (meta-init audit
        # contract): ``_aux_loss`` is runtime scratch overwritten each
        # forward, but as a registered buffer it must still be reset here.
        self._aux_loss.zero_()
        self.router.reset_parameters()
        for group in (self.experts, self.shared_experts):
            for expert in group:
                if hasattr(expert, "reset_parameters"):
                    expert.reset_parameters()

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        num_tokens = x_flat.shape[0]

        weights, indices, logits = self.router(x_flat)
        if (
            self.training
            and self.aux_loss_weight > 0
            and self.router.scoring_func == "softmax"
        ):
            self._aux_loss = self._load_balance_loss(logits, indices)

        y = self._dispatch_routed(x_flat, weights, indices, num_tokens)
        for shared in self.shared_experts:
            y = y + shared(x_flat)
        assert isinstance(y, Tensor)
        return y.reshape(*shape[:-1], self.channels_out)

    def _dispatch_routed(
        self,
        x_flat: Tensor,
        weights: Tensor,
        indices: Tensor,
        num_tokens: int,
    ) -> Tensor:
        """Sort (token, expert) pairs by expert; dispatch contiguously.

        Produces one expert forward per *active* expert (at most
        ``top_k * num_tokens``, typically ≪ ``num_experts`` for large
        MoEs). Numerically equivalent to the mask-per-expert form.
        """
        k = indices.shape[-1]
        flat_idx = indices.reshape(-1)
        flat_w = weights.reshape(-1)
        token_ix = (
            torch.arange(num_tokens, device=x_flat.device).unsqueeze(-1).expand(-1, k)
        ).reshape(-1)

        order = flat_idx.argsort()
        sorted_expert = flat_idx[order]
        sorted_tok = token_ix[order]
        sorted_w = flat_w[order]

        active, counts = torch.unique_consecutive(sorted_expert, return_counts=True)
        offsets = torch.cumsum(counts, dim=0)
        starts = torch.cat([offsets.new_zeros(1), offsets[:-1]], dim=0)

        y = x_flat.new_zeros(x_flat.shape[0], self.channels_out)
        for expert_id, start, count in zip(
            _as_int_list(active),
            _as_int_list(starts),
            _as_int_list(counts),
            strict=True,
        ):
            end = start + count
            tok_slice = sorted_tok[start:end]
            w_slice = sorted_w[start:end].unsqueeze(-1)
            x_e = x_flat.index_select(0, tok_slice)
            y.index_add_(0, tok_slice, self.experts[expert_id](x_e) * w_slice)
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


def _as_int_list(t: Tensor) -> list[int]:
    """Materialize a 1-D tensor as ``list[int]``.

    Central chokepoint for the upstream torch stubs, whose
    ``Tensor.tolist()`` return type is ``list[Unknown]``.
    """
    return cast("list[int]", t.tolist())
