"""Gated graph neural network: typed edge messages, ordered scatter and shared GRU.

References:
    https://arxiv.org/abs/1511.05493
        Li et al. Gated Graph Sequence Neural Networks.

"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field, replace
from typing import Self, override

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.cost import Cost, cost, elementwise_cost, matmul_cost, traffic
from priml.model.init import InitFn, call_init
from priml.model.linear import Linear


@dataclass(frozen=True, slots=True, kw_only=True)
class GraphBatch:
    """Carry ordered directed edges and the node counts of disjoint graphs."""

    source: Tensor
    destination: Tensor
    edge_type: Tensor
    node_counts: Tensor

    def average(self, features: Tensor) -> Tensor:
        """Average each graph's node states, returning zero for an empty graph.

        Args:
          features: Concatenated node states in graph order.

        Returns:
          pooled: One feature row per graph, including empty graph segments.

        """
        segment = torch.repeat_interleave(self.node_counts)
        summed = features.new_zeros((self.node_counts.numel(), *features.shape[1:]))
        _ = summed.index_add_(0, segment, features)
        counts = self.node_counts.clamp(min=1)
        return summed / counts.view(-1, *([1] * (features.ndim - 1)))


class RecurrentCell(nn.GRUCell):
    """Expose torch's exact GRUCell through a Priml configuration leaf."""

    class Config(Fig["RecurrentCell"]):
        channels_in: int = -1
        """Aggregated message width."""

        channels_out: int = -1
        """Recurrent node-state width."""

        _: KW_ONLY

        bias: bool = True
        """Include both input and recurrent gate offsets."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            input_requires_grad: bool = True,
            state_requires_grad: bool = True,
            **kwargs: object,
        ) -> Cost:
            """Count both three-gate affines and Torch's reset-after-affine GRU.

            Args:
              seq_len: Packed node positions per example.
              batch_size: Examples represented by the packed rows.
              dtype: Floating operand dtype; None uses Torch's default.
              input_requires_grad: Differentiate the aggregated messages.
              state_requires_grad: Differentiate the supplied previous state.
              **kwargs: Unread outer cost metadata.

            Returns:
              execution: Complete affine and gate primal/adjoint cost.

            """
            del kwargs
            rows = seq_len * batch_size
            width = self.channels_out
            total = Cost()
            for channels, differentiable in (
                (self.channels_in, input_requires_grad),
                (width, state_requires_grad),
            ):
                affine = matmul_cost(
                    channels_in=channels,
                    channels_out=3 * width,
                    bias=self.bias,
                    rows=rows,
                    dtype=dtype,
                )
                if not differentiable:
                    # Constant messages still train the input gate weights. Only
                    # their input-gradient product disappears, not the weight VJP.
                    affine = replace(
                        affine,
                        cells={
                            key: value // 2
                            if key[1:3] == ("adjoint", "matmul")
                            else value
                            for key, value in affine.cells.items()
                        },
                    )
                total += affine
            # Two sigmoid gates cost ten scalar ops, the candidate costs three,
            # and interpolation costs four. Its VJP uses saved primal values:
            # sigmoid 6, tanh 3, reset products 2, interpolation 4, state sum 1.
            return total + elementwise_cost(
                primal=17 * rows * width,
                adjoint=(14 + 2 * int(state_requires_grad)) * rows * width,
                channels=width,
                rows=rows,
                inputs=18,
                outputs=11,
                adjoint_inputs=15 + 4 * int(state_requires_grad),
                adjoint_outputs=8 + 2 * int(state_requires_grad),
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__(config.channels_in, config.channels_out, bias=config.bias)


class GGNN(nn.Module):
    """Apply edge-specific affine maps, ordered scatter sums and shared GRU gates."""

    class Config(Fig["GGNN"]):
        channels_hidden: int = 128
        """Node-state and message width."""

        num_layers: int = 4
        """Recurrent message-passing steps sharing all parameters."""

        num_edge_types: int = 1
        """Typed edge maps, each with its own affine weights."""

        edge: Linear.Config = field(default_factory=lambda: Linear.Config(bias=True))
        """Affine geometry and first initialization of each edge map."""

        cell: RecurrentCell.Config = field(default_factory=RecurrentCell.Config)
        """Torch GRU gates and their initialization."""

        init_edge: InitFn | None = None
        """Second edge-weight initialization, drawn after the cell is redrawn.

        None resets each edge map by its own initializers. Otherwise each weight
        is drawn by this function and each bias is zeroed.
        """

        @override
        def finalize(self) -> Self:
            self.edge.channels_in = self.edge.channels_out = self.channels_hidden
            self.cell.channels_in = self.cell.channels_out = self.channels_hidden
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            edge_counts: tuple[int, ...],
            channels: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Compose the actual typed-edge rows and parameter-shared GRU steps.

            Args:
              seq_len: Packed real nodes per example, not padded node slots.
              batch_size: Examples represented by the packed node rows.
              dtype: Floating message/state dtype; None uses Torch's default.
              edge_counts: Actual directed edges of each declared type across
                this whole invocation, including zero entries for unused maps.
              channels: Input width before zero padding; -1 uses the state width.
              **kwargs: Cost metadata forwarded to affine and recurrent children.

            Returns:
              execution: Every owned map, shared gate, gather and ordered scatter.

            Raises:
              ValueError: Edge counts omit a type or include a negative count.

            """
            if len(edge_counts) != self.num_edge_types or any(
                count < 0 for count in edge_counts
            ):
                raise ValueError(
                    "edge_counts must contain one nonnegative count per edge type.",
                )
            rows = seq_len * batch_size
            width = self.channels_hidden
            input_width = width if channels == -1 else channels
            edges = sum(edge_counts)
            steps = max(0, self.num_layers)
            edge_params = self.edge.channels_in * self.edge.channels_out + (
                self.edge.channels_out if self.edge.bias else 0
            )
            step = cost(
                self.cell,
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                **(kwargs | {"input_requires_grad": edges > 0}),
            )
            active_maps = 0
            for count in edge_counts:
                if count > 0:
                    active_maps += 1
                    step += replace(
                        cost(
                            self.edge,
                            seq_len=count,
                            batch_size=1,
                            dtype=dtype,
                            **kwargs,
                        ),
                        params=0,
                    )
            step += (
                traffic(
                    "primal",
                    "elementwise",
                    elements=(edges + rows) * width,
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=7 * edges * width,
                    flops=edges * width,
                    dtype=dtype,
                )
                + traffic(
                    "adjoint",
                    "selection",
                    elements=7 * edges * width + (rows * width if edges else 0),
                    flops=edges * width,
                    dtype=dtype,
                )
                + traffic(
                    "adjoint",
                    "elementwise",
                    elements=3 * rows * width if edges else 0,
                    flops=rows * width if edges else 0,
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=6 * edges,
                    dtype=torch.int64,
                )
                + traffic(
                    "adjoint",
                    "selection",
                    elements=3 * edges,
                    dtype=torch.int64,
                )
            )
            repeated = step.tile(steps)
            if steps == 0:
                repeated = replace(repeated, params_active=0)
            shared_gradients = max(0, steps - 1)
            cell_params = step.params
            return (
                Cost(params=self.num_edge_types * edge_params)
                + repeated
                + traffic(
                    "primal",
                    "selection",
                    elements=2 * rows * width + rows * (width - input_width),
                    dtype=dtype,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=(self.num_edge_types + 1) * edges,
                    dtype=torch.int64,
                )
                + traffic(
                    "primal",
                    "selection",
                    elements=2 * self.num_edge_types * edges,
                    dtype=torch.bool,
                )
                + traffic(
                    "adjoint",
                    "elementwise",
                    elements=3 * shared_gradients * cell_params,
                    flops=shared_gradients * cell_params,
                    dtype=dtype,
                )
                + traffic(
                    "adjoint",
                    "elementwise",
                    elements=3 * shared_gradients * active_maps * edge_params,
                    flops=shared_gradients * active_maps * edge_params,
                    dtype=self.edge.dtype if self.edge.dtype is not None else dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.linears = nn.ModuleList(
            [config.edge.make() for _ in range(config.num_edge_types)],
        )
        self.gru = config.cell.make()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Redraw the cell, then the edge maps, in registration order."""
        self.gru.reset_parameters()
        init_edge = self.config.init_edge
        for linear in self.linears:
            if init_edge is None:
                linear.reset_parameters()
                continue
            call_init(init_edge, linear.weight)
            if linear.bias is not None:
                _ = nn.init.zeros_(linear.bias)

    @override
    def forward(self, topology: GraphBatch, features: Tensor) -> Tensor:
        """Update every node from its ordered incoming typed-edge messages.

        Args:
          topology: Directed edges in their original encounter order.
          features: Node features, zero-padded to the recurrent width when needed.

        Returns:
          states: Final recurrent node states for every graph in the batch.

        """
        state = torch.cat(
            [
                features,
                features.new_zeros(
                    (len(features), self.config.channels_hidden - features.shape[-1]),
                ),
            ],
            dim=-1,
        )
        groups = [
            (linear, torch.nonzero(topology.edge_type == index, as_tuple=True)[0])
            for index, linear in enumerate(self.linears)
        ]
        groups = [(linear, edges) for linear, edges in groups if edges.numel() > 0]
        for _ in range(self.config.num_layers):
            messages = state.new_zeros(
                (topology.source.numel(), self.config.channels_hidden),
            )
            for linear, edges in groups:
                messages[edges] = linear(state[topology.source[edges]])
            aggregate = state.new_zeros(state.shape).index_add_(
                0,
                topology.destination,
                messages,
            )
            state = self.gru(aggregate, state)
        return state
