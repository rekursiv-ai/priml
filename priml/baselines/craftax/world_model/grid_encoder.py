"""Grid convolution frame encoder: a cheap drop-in for the slot transformer.

The board slots are the observation's 9 x 11 grid, row-major, as the packed
observation holds its cells (``replay.token_frame_numba``). Each of a few ConvNeXt-style
blocks runs a depthwise spatial convolution, a per-cell RMSNorm whose scale and
shift come from the aux slots (FiLM), and a per-cell SwiGLU, as a residual. The
aux slots pass through one shared per-slot SwiGLU, and their mean conditions
the board's blocks. ``memory`` is the final-normed cell and aux outputs, laid
out as the slots, the local decoder's cross-attention input as the slot
encoder's; ``pooled`` is multi-head attention pooling with a learned query over
that memory, normed.
"""

from collections.abc import Mapping
from functools import partial
from typing import override

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.craftax.lib.costs import (
    activation_cost,
    broadcast_add_cost,
    concat_cost,
    residual_cost,
)
from priml.cost import Cost, elementwise_cost, matmul_cost, reduction_cost
from priml.model.conv import Conv2d
from priml.model.init import InitFn, call_init
from priml.model.norm import RMSNorm


class GridConvEncoder(nn.Module):
    """Grid convolution over the board, per-slot MLP over the aux, attention pooling."""

    class Config(Fig["GridConvEncoder"]):
        """Two 7 x 7 blocks at width 512, about 0.26 GFLOP per frame (inferred)."""

        channels_in: int = 512
        """Width of the slots, the memory and the pooled vector."""

        num_slots: int = -1
        """Slots per frame (-1 until the world model pushes its schema's)."""

        rows: int = 9
        """Board rows; the first ``rows x columns`` slots are cells, row-major."""

        columns: int = 11
        """Board columns."""

        num_blocks: int = 2
        """Convolution blocks over the board."""

        kernel_size: int = 7
        """Depthwise kernel size, odd; zero padding keeps the grid's shape."""

        channels_hidden: int = 320
        """Hidden width of every SwiGLU."""

        num_heads: int = 8
        """Heads of the attention pooling."""

        eps: float = 1e-6
        """RMSNorm epsilon, the world model's."""

        init_weight: InitFn = partial(nn.init.normal_, std=0.02)
        """Init of the slot embeddings, kernels, matrices and pooling query."""

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the aux MLP, FiLM, every board block, the final norm, and the pooling.

            Args:
              batch_size: Frames in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; unread.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes, and ownership.

            """
            del kwargs
            width, cells = self.channels_in, self.rows * self.columns
            frames, aux = batch_size, self.num_slots - cells
            film = self.num_blocks * 2 * width
            averaged = reduction_cost(
                input_elements=frames * aux * width,
                output_groups=frames * width,
                dtype=dtype,
            ) + elementwise_cost(
                primal=frames * width,
                adjoint=frames * aux * width,
                channels=width,
                rows=frames,
                dtype=dtype,
            )
            aux_stage = (
                broadcast_add_cost(
                    params=self.num_slots * width,
                    rows=frames,
                    dtype=dtype,
                )
                + _residual_mlp_cost(self, rows=frames * aux, dtype=dtype)
                + averaged
                + matmul_cost(
                    channels_in=width,
                    channels_out=film,
                    rows=frames,
                    dtype=dtype,
                )
            )
            board = _block_cost(self, frames=frames, dtype=dtype).tile(
                self.num_blocks,
                copies=self.num_blocks,
            )
            memory = concat_cost(
                elements=frames * self.num_slots * width,
                dtype=dtype,
            ) + _norm_cost(self, rows=frames * self.num_slots, dtype=dtype)
            return aux_stage + board + memory + _pooling_cost(self, frames, dtype)

    def __init__(self, config: Config) -> None:
        cells = config.rows * config.columns
        if config.num_slots <= cells:
            msg = (
                f"num_slots={config.num_slots} must exceed rows x columns={cells}: "
                "the aux slots follow the cells."
            )
            raise ValueError(msg)
        if config.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size={config.kernel_size} must be odd.")
        if config.channels_in % config.num_heads:
            raise ValueError("channels_in must be a multiple of num_heads.")
        super().__init__()
        width, hidden = config.channels_in, config.channels_hidden
        self.rows, self.columns = config.rows, config.columns
        self.num_heads, self.eps = config.num_heads, config.eps
        self.slot_embedding = nn.Parameter(torch.empty(config.num_slots, width))
        self.aux = _mlp(width, hidden)
        self.film = nn.Parameter(torch.empty(config.num_blocks * 2 * width, width))
        self.blocks = nn.ModuleList(
            _mlp(width, hidden, kernel_size=config.kernel_size)
            for _ in range(config.num_blocks)
        )
        self.final = nn.Parameter(torch.empty(width))
        self.query = nn.Parameter(torch.empty(width))
        self.pooled_norm = nn.Parameter(torch.empty(width))
        self._init_weight = config.init_weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place; FiLM starts as the identity."""
        for name, parameter in self.named_parameters():
            if name.endswith("norm") or name == "final":
                nn.init.ones_(parameter)
            elif name == "film":
                nn.init.zeros_(parameter)
            else:
                call_init(self._init_weight, parameter)

    def frame_macs(self) -> int:
        """Return one frame's multiply-adds: kernels and MLPs per cell, aux MLP, FiLM.

        Returns:
          macs: Multiply-adds in the encoder's matrices for one frame.

        """
        cells = self.rows * self.columns
        aux = len(self.slot_embedding) - cells
        per_cell = sum(
            p.numel()
            for block in self.blocks
            for n, p in block.named_parameters()
            if n != "norm"
        )
        per_aux = sum(p.numel() for n, p in self.aux.named_parameters() if n != "norm")
        return cells * per_cell + aux * per_aux + self.film.numel()

    @override
    def forward(self, slots: Tensor) -> tuple[Tensor, Tensor]:
        """Encode frames.

        Args:
          slots: Embedded frame slots ``[..., num_slots, C]``.

        Returns:
          pooled: The attention-pooled, normed memory, ``[..., C]``.
          memory: The per-slot outputs, ``[..., num_slots, C]``.

        """
        return grid_encode(
            dict(self.named_parameters()),
            slots,
            rows=self.rows,
            columns=self.columns,
            heads=self.num_heads,
            eps=self.eps,
        )


def grid_encode(
    params: Mapping[str, Tensor],
    slots: Tensor,
    *,
    rows: int,
    columns: int,
    heads: int,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Return ``GridConvEncoder``'s pooled vector and memory of embedded slots.

    Block count and kernel size are read from the parameters' shapes.

    Args:
      params: ``GridConvEncoder.named_parameters()`` by name.
      slots: Embedded frame slots ``[..., S, C]``; the first ``rows x columns``
        are the board's cells, row-major.
      rows: Board rows.
      columns: Board columns.
      heads: Attention-pooling heads.
      eps: RMSNorm epsilon.

    Returns:
      pooled: ``[..., C]``.
      memory: ``[..., S, C]``.

    """
    width, cells = slots.shape[-1], rows * columns
    lead = slots.shape[:-2]
    x = (slots + params["slot_embedding"]).reshape(-1, *slots.shape[-2:])
    aux = x[:, cells:]
    aux = aux + _swiglu(_norm(aux, params["aux.norm"], eps), params, "aux")
    blocks = len(params["film"]) // (2 * width)
    film = functional.linear(aux.mean(-2), params["film"])
    scale, shift = film.unflatten(-1, (blocks, 2, 1, 1, width)).unbind(2)
    board = x[:, :cells].unflatten(1, (rows, columns))
    for index in range(blocks):
        kernel = params[f"blocks.{index}.depthwise"]
        h = functional.conv2d(
            board.permute(0, 3, 1, 2),
            kernel,
            padding=kernel.shape[-1] // 2,
            groups=width,
        ).permute(0, 2, 3, 1)
        h = _norm(h, params[f"blocks.{index}.norm"], eps)
        h = h * (1 + scale[:, index]) + shift[:, index]
        board = board + _swiglu(h, params, f"blocks.{index}")
    memory = _norm(torch.cat([board.flatten(1, 2), aux], dim=1), params["final"], eps)
    keys = memory.unflatten(-1, (heads, -1))
    query = params["query"].unflatten(-1, (heads, -1))
    weights = torch.einsum("nshd,hd->nhs", keys, query) * keys.shape[-1] ** -0.5
    pooled = torch.einsum("nhs,nshd->nhd", weights.softmax(-1), keys).flatten(-2)
    pooled = _norm(pooled, params["pooled_norm"], eps)
    return pooled.reshape(*lead, width), memory.reshape(*lead, -1, width)


def _residual_mlp_cost(
    config: GridConvEncoder.Config,
    *,
    rows: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost ``x + _swiglu(_norm(x))`` over ``rows`` slots, the aux slots' stage."""
    width = config.channels_in
    return (
        _norm_cost(config, rows=rows, dtype=dtype)
        + _swiglu_cost(config, rows=rows, dtype=dtype)
        + residual_cost(channels=width, rows=rows, dtype=dtype)
    )


# FiLM adds 1 to each frame's scale, then multiplies and shifts every value; back, it
# multiplies the gradient by the scale and by the value, and sums both products' and the
# plain gradient's share over each frame's cells.
def _block_cost(
    config: GridConvEncoder.Config,
    *,
    frames: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost one board block: the depthwise window, the norm, FiLM, the residual SwiGLU."""
    width, rows = config.channels_in, frames * config.rows * config.columns
    values = rows * width
    window = Conv2d.Config(
        channels_in=width,
        channels_out=width,
        kernel_size=config.kernel_size,
        padding=config.kernel_size // 2,
        groups=width,
    )
    film = elementwise_cost(
        primal=2 * values + frames * width,
        adjoint=2 * values,
        channels=width,
        rows=rows,
        inputs=3,
        dtype=dtype,
    ) + reduction_cost(
        input_elements=2 * values,
        output_groups=2 * frames * width,
        dtype=dtype,
        phase="adjoint",
    )
    return (
        window.cost(
            input_grid=(config.rows, config.columns),
            batch_size=frames,
            dtype=dtype,
        )
        + _norm_cost(config, rows=rows, dtype=dtype)
        + film
        + _swiglu_cost(config, rows=rows, dtype=dtype)
        + residual_cost(channels=width, rows=rows, dtype=dtype)
    )


# The scores are one ``[slots, channels_head] @ [channels_head, 1]`` product per head
# against the query, which owns its weights; the pooled vector one ``[1, slots] @
# [slots, channels_head]`` per frame and head. Between them a scale and a softmax over
# each frame's slots per head.
def _pooling_cost(
    config: GridConvEncoder.Config,
    frames: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost the attention pooling: both einsums as torch's batched products, then the norm."""
    heads, slots = config.num_heads, config.num_slots
    head = config.channels_in // heads
    weights = frames * heads * slots
    scores = matmul_cost(
        channels_in=head,
        channels_out=1,
        rows=frames * slots,
        dtype=dtype,
    ).tile(heads, copies=heads)
    pooled = matmul_cost(
        channels_in=slots,
        channels_out=head,
        weight=False,
        dtype=dtype,
    ).tile(frames * heads)
    softmax = (
        elementwise_cost(
            primal=4 * weights,
            adjoint=3 * weights,
            channels=slots,
            rows=frames * heads,
            dtype=dtype,
        )
        + reduction_cost(
            input_elements=2 * weights,
            output_groups=2 * frames * heads,
            dtype=dtype,
        )
        + reduction_cost(
            input_elements=weights,
            output_groups=frames * heads,
            dtype=dtype,
            phase="adjoint",
        )
    )
    return scores + softmax + pooled + _norm_cost(config, rows=frames, dtype=dtype)


def _norm_cost(
    config: GridConvEncoder.Config,
    *,
    rows: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost ``_norm`` over ``rows`` rows: priml's affine RMSNorm, which it computes."""
    norm = RMSNorm.Config(
        channels_in=config.channels_in,
        eps=config.eps,
        elementwise_affine=True,
    )
    return norm.cost(seq_len=rows, batch_size=1, dtype=dtype)


def _swiglu_cost(
    config: GridConvEncoder.Config,
    *,
    rows: int,
    dtype: torch.dtype | None,
) -> Cost:
    """Cost ``_swiglu`` over ``rows`` rows: one fused gate-and-up product, the gate, down."""
    width, hidden = config.channels_in, config.channels_hidden
    elements = rows * hidden
    gate = activation_cost(
        functional.silu,
        elements=elements,
        dtype=dtype,
    ) + elementwise_cost(
        primal=elements,
        adjoint=2 * elements,
        channels=hidden,
        rows=rows,
        inputs=2,
        adjoint_outputs=2,
        dtype=dtype,
    )
    return (
        matmul_cost(channels_in=width, channels_out=2 * hidden, rows=rows, dtype=dtype)
        + gate
        + matmul_cost(channels_in=hidden, channels_out=width, rows=rows, dtype=dtype)
    )


def _mlp(width: int, hidden: int, *, kernel_size: int = 0) -> nn.ParameterDict:
    """Return a norm and SwiGLU's weights, with a depthwise kernel when sized."""
    weights = {
        "norm": nn.Parameter(torch.empty(width)),
        "up": nn.Parameter(torch.empty(2 * hidden, width)),
        "down": nn.Parameter(torch.empty(width, hidden)),
    }
    if kernel_size:
        shape = (width, 1, kernel_size, kernel_size)
        weights["depthwise"] = nn.Parameter(torch.empty(shape))
    return nn.ParameterDict(weights)


def _norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Return an affine RMSNorm over the last dimension."""
    return functional.rms_norm(x, (x.shape[-1],), weight, eps)


def _swiglu(x: Tensor, params: Mapping[str, Tensor], prefix: str) -> Tensor:
    """Return the bias-free SwiGLU stored under ``prefix``."""
    gate, up = functional.linear(x, params[f"{prefix}.up"]).chunk(2, dim=-1)
    return functional.linear(functional.silu(gate) * up, params[f"{prefix}.down"])
