"""Graphormer: a transformer over graph nodes with a learned graph attention bias.

The encoder prepends one graph token to the node states and biases every
attention head by the shortest-path distance between nodes, by the multi-hop
products of the edge features along those paths, and by a learned bias between
the graph token and the nodes.

References:
    https://arxiv.org/abs/2106.05234
        Ying et al. Do Transformers Really Perform Bad for Graph Representation?

"""

from __future__ import annotations

from dataclasses import KW_ONLY, field, replace
from typing import TYPE_CHECKING, Self, override

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    matmul_cost,
    reduction_cost,
    traffic,
)
from priml.model.attention.attention import Attention
from priml.model.attention.kernel import SdpaFused
from priml.model.custom_types import TensorModule, infer_same_width
from priml.model.embedding import Embedding
from priml.model.linear import Linear
from priml.model.norm import LayerNorm
from priml.model.sequential import Sequential
from priml.model.special import Dropout
from priml.model.swiglu import SwiGLU, gelu
from priml.model.transformer.block import TransformerBlock


if TYPE_CHECKING:
    from priml.model.custom_types import LayerCache


def feed_forward() -> Sequential.Config:
    """Return the standard affine, GELU, affine feed-forward with two dropouts.

    Returns:
      config: Ungated SwiGLU with a dropout between its affine maps, then one
        dropout on the residual update.

    """
    swiglu = SwiGLU.Config(gate=False, act=gelu, round_to=1)
    swiglu.up_proj = Linear.Config(bias=True, init_bias=None)
    swiglu.middle = Dropout.Config(p=0.1)
    swiglu.down_proj = Linear.Config(bias=True, init_bias=None)
    ffn = Sequential.Config()
    ffn.elements = [swiglu, Dropout.Config(p=0.1)]
    return ffn


class GraphKernel(SdpaFused):
    """Transpose projected sequence-major heads around the shared SDPA kernel."""

    class Config(Makes["GraphKernel"], SdpaFused.Config):
        """Retain the standard kernel and its sequence-major layout boundary."""

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        attn_mask: Tensor | None = None,
        window: int = -1,
        **kwargs: object,
    ) -> Tensor:
        del kwargs
        return (
            super()
            .forward(
                q.transpose(0, 1),
                k.transpose(0, 1),
                v.transpose(0, 1),
                dropout_p=dropout_p,
                is_causal=is_causal,
                attn_mask=attn_mask,
                window=window,
            )
            .transpose(0, 1)
        )


class GraphAttention(nn.Module):
    """Attend over sequence-major states, then drop the projected residual update."""

    class Config(Fig["GraphAttention"]):
        channels_in: int = -1
        """Residual-stream width entering attention; -1 reads it from the kernel."""

        channels_out: int = -1
        """Residual-stream width leaving attention; -1 reads it from the kernel."""

        _: KW_ONLY

        attention: Attention.Config = field(
            default_factory=lambda: Attention.Config(
                dropout=0.1,
                proj_qkv=Linear.Config(bias=True, init_bias=None),
                proj_out=Linear.Config(bias=True, init_bias=None),
                attn_kernel=GraphKernel.Config(),
            ),
        )
        """Shared projections and the sequence-major SDPA kernel."""

        dropout: Dropout.Config = field(default_factory=lambda: Dropout.Config(p=0.1))
        """Drop the projected residual update."""

        @override
        def finalize(self) -> Self:
            infer_same_width(self)
            if self.attention.channels_in == -1:
                self.attention.channels_in = self.channels_in
            if self.attention.channels_out == -1:
                self.attention.channels_out = self.channels_out
            self.dropout.channels_in = (
                self.attention.channels_in
                if self.attention.channels_in != -1
                else self.attention.channels_out
            )
            self = super().finalize()
            self.channels_in = self.channels_out = self.attention.channels_in
            return self

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Count shared attention and residual dropout without a decoding cache.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per invocation.
              dtype: Activation dtype; None uses the torch default.
              **kwargs: Shape and policy messages forwarded to both children.

            Returns:
              execution: Complete attention and residual-dropout cost.

            """
            return replace(
                cost(
                    self.attention,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                ),
                bytes_state=0,
            ) + cost(
                self.dropout,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attention = config.attention.make()
        self.dropout = config.dropout.make()
        self._draw_projections()

    def reset_parameters(self) -> None:
        """Reset the projections, then draw Graphormer's attention initialization."""
        self.attention.reset_parameters()
        self.dropout.reset_parameters()
        self._draw_projections()

    def _draw_projections(self) -> None:
        _ = nn.init.xavier_uniform_(self.attention.proj_qkv.weight, gain=2**-0.5)
        _ = nn.init.xavier_uniform_(self.attention.proj_out.weight)
        if self.attention.proj_out.bias is not None:
            _ = nn.init.zeros_(self.attention.proj_out.bias)

    @override
    def forward(
        self,
        input: Tensor,
        *,
        memory: Tensor | None = None,
        cache: LayerCache | None = None,
        positions: Tensor | None = None,
        cos_sin: tuple[Tensor, Tensor] | None = None,
        dropout_p: float | None = None,
        is_causal: bool | None = None,
        attn_mask: Tensor | None = None,
        window: int = -1,
        **kwargs: object,
    ) -> Tensor:
        return self.dropout(
            self.attention.forward(
                input,
                memory=memory,
                cache=cache,
                positions=positions,
                cos_sin=cos_sin,
                dropout_p=dropout_p,
                is_causal=is_causal,
                attn_mask=attn_mask,
                window=window,
                **kwargs,
            ),
        )


class GraphBlock(TransformerBlock):
    """Use Priml's post-norm residual block with GELU graph updates."""

    class Config(Makes["GraphBlock"], TransformerBlock.Config, kw_only=False):
        prenorm: bool = False
        """Normalize after each residual addition."""

        attn: Makeable[TensorModule] = field(default_factory=GraphAttention.Config)
        """Graph-bias attention with its residual dropout."""

        ffn: Makeable[TensorModule] = field(default_factory=feed_forward)
        """Fourfold hidden expansion with GELU and two dropout sites."""

        norm1: Makeable[TensorModule] = field(
            default_factory=lambda: LayerNorm.Config(elementwise_affine=True),
        )
        """Affine attention-residual normalization."""

        norm2: Makeable[TensorModule] = field(
            default_factory=lambda: LayerNorm.Config(elementwise_affine=True),
        )
        """Affine feed-forward residual normalization."""


class Graphormer(nn.Module):
    """Encode nodes, degrees, shortest paths and learned multi-hop edge products."""

    class Config(Fig["Graphormer"]):
        channels_in: int = -1
        """Node feature width."""

        channels_hidden: int = 768
        """Residual stream width."""

        heads: int = 32
        """Attention heads of every block."""

        num_layers: int = 12
        """Independently initialized blocks."""

        edge_distances: int = 128
        """Learned multi-hop distance matrices, at least ``max_hops``."""

        max_hops: int = 5
        """Largest retained multi-hop path length."""

        init_std: float = 0.02
        """Standard deviation of the normal draws applied after construction."""

        node: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Node-feature projection."""

        degree: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(
                512,
                padding_idx=0,
                init_weight=nn.init.normal_,
            ),
        )
        """Degree features with a zero padding row."""

        token: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(1, init_weight=nn.init.normal_),
        )
        """Learned graph token."""

        edge: Linear.Config = field(
            default_factory=lambda: Linear.Config(bias=True, init_bias=None),
        )
        """Edge-feature projection into attention heads."""

        distance: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(
                channels_out=1,
                init_weight=nn.init.normal_,
            ),
        )
        """Head-to-head matrix for each path hop."""

        spatial: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(
                512,
                padding_idx=0,
                init_weight=nn.init.normal_,
            ),
        )
        """Shortest-path attention bias with a zero padding row."""

        virtual: Embedding.Config = field(
            default_factory=lambda: Embedding.Config(1, init_weight=nn.init.normal_),
        )
        """Bias between the graph token and the nodes."""

        block: GraphBlock.Config = field(default_factory=GraphBlock.Config)
        """Post-norm graph attention and feed-forward block."""

        dropout: Dropout.Config = field(default_factory=lambda: Dropout.Config(p=0.1))
        """Drop the initialized graph-token and node activations."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            hops: int,
            **kwargs: object,
        ) -> Cost:
            """Compose node/path projections, shared graph bias and all blocks.

            Args:
              seq_len: Padded nodes, excluding the graph token.
              batch_size: Graphs in the invocation.
              dtype: Floating operand dtype.
              hops: Available path positions before max_hops truncation.
              **kwargs: Additional block cost arguments.

            Returns:
              execution: Complete graph encoder, including learned hop products.

            """
            length = seq_len + 1
            retained = min(hops, self.max_hops)
            pairs = batch_size * seq_len * seq_len
            bias_elements = batch_size * self.heads * length * length
            path_elements = pairs * retained * self.heads
            direct = (
                self.token.channels_in * self.token.channels_out
                + self.virtual.channels_in * self.virtual.channels_out
                + self.distance.channels_in * self.distance.channels_out
            )
            total = (
                cost(self.node, seq_len=seq_len, batch_size=batch_size, dtype=dtype)
                + cost(self.degree, seq_len=seq_len, batch_size=batch_size, dtype=dtype)
                + cost(
                    self.edge,
                    seq_len=seq_len * seq_len * retained,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    self.spatial,
                    seq_len=seq_len * seq_len,
                    batch_size=batch_size,
                    dtype=dtype,
                )
                + cost(
                    self.dropout,
                    seq_len=length,
                    batch_size=batch_size,
                    channels=self.channels_hidden,
                    dtype=dtype,
                )
                + cost(
                    self.block,
                    seq_len=length,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                ).tile(self.num_layers, copies=self.num_layers)
                + matmul_cost(
                    channels_in=self.heads,
                    channels_out=self.heads,
                    rows=pairs,
                    weight=False,
                    dtype=dtype,
                ).tile(retained)
                + Cost(
                    params=direct,
                    params_active=self.channels_hidden
                    + self.heads
                    + retained * self.heads * self.heads,
                )
            )
            return (
                total
                + elementwise_cost(
                    primal=batch_size * seq_len * self.channels_hidden,
                    adjoint=batch_size * seq_len * self.channels_hidden,
                    channels=self.channels_hidden,
                    rows=batch_size * seq_len,
                    inputs=2,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=2 * bias_elements
                    + pairs * self.heads
                    + batch_size * self.heads * (2 * seq_len + 1),
                    adjoint=3 * bias_elements + 2 * pairs * self.heads,
                    channels=self.heads * length,
                    rows=batch_size * length,
                    inputs=3,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=3 * pairs,
                    adjoint=0,
                    channels=seq_len,
                    rows=batch_size * seq_len,
                    dtype=torch.int64,
                )
                + reduction_cost(
                    input_elements=path_elements,
                    output_groups=pairs * self.heads,
                    dtype=dtype,
                )
                + traffic(
                    "adjoint",
                    kernel="reduction",
                    elements=path_elements + pairs * self.heads,
                    dtype=dtype,
                )
                + reduction_cost(
                    input_elements=batch_size * self.channels_hidden,
                    output_groups=self.channels_hidden,
                    phase="adjoint",
                    dtype=dtype,
                )
                + reduction_cost(
                    input_elements=batch_size * self.heads * (2 * seq_len + 1),
                    output_groups=self.heads,
                    phase="adjoint",
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    kernel="selection",
                    elements=2
                    * (
                        bias_elements
                        + path_elements
                        + batch_size * length * self.channels_hidden
                    ),
                    dtype=dtype,
                )
                + traffic(
                    "adjoint",
                    kernel="selection",
                    elements=2
                    * (
                        bias_elements
                        + path_elements
                        + batch_size * length * self.channels_hidden
                    ),
                    dtype=dtype,
                )
            )

        @override
        def finalize(self) -> Self:
            self.node.channels_in = self.channels_in
            self.node.channels_out = self.degree.channels_out = (
                self.token.channels_out
            ) = self.channels_hidden
            self.edge.channels_out = self.spatial.channels_out = (
                self.virtual.channels_out
            ) = self.heads
            self.distance.channels_in = self.edge_distances * self.heads * self.heads
            self.block.channels_in = self.channels_hidden
            attn = self.block.attn
            if isinstance(attn, GraphAttention.Config):
                attn.attention.num_heads = self.heads
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        std = config.init_std
        self.node = config.node.make()
        self.degree = config.degree.make()
        self.token = config.token.make()
        # Initial component resets occur before attention-bias construction.
        _ = nn.init.normal_(self.node.weight, std=std / config.num_layers**0.5)
        if self.node.bias is not None:
            _ = nn.init.zeros_(self.node.bias)
        _ = nn.init.normal_(self.degree.weight, std=std)
        _ = nn.init.normal_(self.token.weight, std=std)
        self.edge = config.edge.make()
        self.distance = config.distance.make()
        self.spatial = config.spatial.make()
        self.virtual = config.virtual.make()
        _ = nn.init.normal_(self.edge.weight, std=std / config.num_layers**0.5)
        if self.edge.bias is not None:
            _ = nn.init.zeros_(self.edge.bias)
        for embedding in (self.distance, self.spatial, self.virtual):
            _ = nn.init.normal_(embedding.weight, std=std)
        self.dropout = config.dropout.make()
        self.blocks: nn.ModuleList[GraphBlock] = nn.ModuleList(
            [config.block.make() for _ in range(config.num_layers)],
        )
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            _ = nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                _ = nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            _ = nn.init.normal_(module.weight, std=std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        if isinstance(module, GraphAttention):
            _ = nn.init.normal_(module.attention.proj_qkv.weight, std=std)

    def attention_bias(self, bias: Tensor, spatial: Tensor, paths: Tensor) -> Tensor:
        """Combine the doubled base bias, spatial bias and distance-weighted paths.

        Args:
          bias: Padded graph-token/node attention bias.
          spatial: Shortest-path distances between nodes.
          paths: Ordered edge features along each retained path.

        Returns:
          combined: Complete per-head graph-token/node attention bias.

        """
        heads = self.config.heads
        batch, nodes = spatial.shape[:2]
        output = bias.clone().unsqueeze(1).repeat(1, heads, 1, 1)
        output[:, :, 1:, 1:] += self.spatial(spatial).permute(0, 3, 1, 2)
        virtual = self.virtual.weight.view(1, heads, 1)
        output[:, :, 1:, 0] += virtual
        output[:, :, 0, :] += virtual
        divisor = spatial.clone()
        divisor[divisor == 0] = 1
        divisor = torch.where(divisor > 1, divisor - 1, divisor)
        divisor = divisor.clamp(0, self.config.max_hops)
        encoded = self.edge(paths[:, :, :, : self.config.max_hops, :])
        hops = encoded.shape[-2]
        encoded = encoded.permute(3, 0, 1, 2, 4).reshape(hops, -1, heads)
        products = torch.bmm(
            encoded,
            self.distance.weight.reshape(-1, heads, heads)[:hops],
        )
        products = products.reshape(hops, batch, nodes, nodes, heads).permute(
            1,
            2,
            3,
            0,
            4,
        )
        edge_bias = (products.sum(-2) / divisor.float().unsqueeze(-1)).permute(
            0,
            3,
            1,
            2,
        )
        output[:, :, 1:, 1:] += edge_bias
        return output + bias.unsqueeze(1)

    @override
    def forward(
        self,
        features: Tensor,
        *,
        bias: Tensor,
        spatial: Tensor,
        degrees: Tensor,
        paths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        nodes = self.node(features) + self.degree(degrees)
        token = self.token.weight.unsqueeze(0).repeat(len(features), 1, 1)
        states = self.dropout(torch.cat([token, nodes], dim=1)).transpose(0, 1)
        attention_bias = self.attention_bias(bias, spatial, paths)
        for block in self.blocks:
            states = block.forward(states, attn_mask=attention_bias)
        return states[0], states[1:].transpose(0, 1)
