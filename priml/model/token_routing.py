"""SPRINT token selection and sparse-to-dense residual fusion."""

from __future__ import annotations

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.cost import Cost, matmul_cost


def select_tokens(x: Tensor, drop_ratio: float) -> tuple[Tensor, Tensor | None]:
    """Keep a random subset of tokens and return their original indices."""
    if not 0 <= drop_ratio < 1:
        raise ValueError("drop_ratio must be in [0, 1)")
    keep = max(1, int(x.shape[1] * (1 - drop_ratio)))
    if keep >= x.shape[1]:
        return x, None
    ids = torch.rand(x.shape[:2], device=x.device).argsort(dim=1)[:, :keep]
    return x.gather(1, ids[..., None].expand(-1, -1, x.shape[-1])), ids


class SparseDenseFusion(nn.Module):
    """Pad routed tokens with a trainable mask, then fuse with the dense path."""

    class Config(Fig["SparseDenseFusion"]):
        channels: int = -1
        """Token width shared by dense and sparse streams."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the fusion projection and its learned mask token."""
            del kwargs
            return matmul_cost(
                channels_in=2 * self.channels,
                channels_out=self.channels,
                bias=True,
                rows=seq_len * batch_size,
                dtype=dtype,
            ) + Cost(params=self.channels, params_active=self.channels)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.channels))
        self.proj = nn.Linear(2 * config.channels, config.channels)

    def forward(
        self,
        dense: Tensor,
        sparse: Tensor,
        ids_keep: Tensor | None,
        *,
        drop_path: bool = False,
    ) -> Tensor:
        """Scatter sparse tokens and fuse them with the dense encoder path."""
        if ids_keep is None:
            padded = sparse
        else:
            padded = self.mask_token.expand_as(dense).clone()
            padded.scatter_(1, ids_keep[..., None].expand_as(sparse), sparse)
        if drop_path:
            padded = padded * 0 + self.mask_token.expand_as(dense)
        return self.proj(torch.cat((dense, padded), dim=-1))
