"""Fixed position tables for spatial transformer tokens."""

from __future__ import annotations

from torch import Tensor

import torch

from priml.math.numeric import mesh_arange


def image_token_positions(grid_size: int, device: torch.device) -> Tensor:
    """Return axial positions for CLS followed by row-major image tokens."""
    spatial = mesh_arange((grid_size, grid_size), device=device)
    return torch.cat((spatial.new_zeros(1, 2), spatial))[None]


def sincos_position_table(
    channels: int,
    grid: int,
    *,
    lead: int = 1,
    compute_dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Build a 2D sine/cosine table with zeroed leading tokens.

    The original SpeedrunDiT builds the table in NumPy float64 and rounds
    once to float32. The later REG branch computes it directly in float32.
    Keeping the intermediate dtype explicit preserves both recipes.
    """
    if channels % 4:
        raise ValueError("position channels must be divisible by four")
    half = channels // 2
    omega = torch.arange(half // 2, dtype=compute_dtype) / (half / 2.0)
    omega = 1.0 / 10_000**omega
    steps = torch.arange(grid, dtype=compute_dtype)
    # Width varies fastest; channels encode columns before rows.
    cols, rows = torch.meshgrid(steps, steps, indexing="xy")
    parts = [
        torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        for out in (
            torch.outer(cols.reshape(-1), omega),
            torch.outer(rows.reshape(-1), omega),
        )
    ]
    table = torch.cat(parts, dim=1)
    return torch.cat([table.new_zeros(lead, channels), table]).float()
