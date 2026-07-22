"""Patchify and unpatchify operations."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Any, override

import math

from configgle import Fig
from torch import Tensor, nn


class Patchify(nn.Module):
    """Reshape spatial dims into patches, increasing channels.

    Input: [..., C, *spatial]
    Output: [..., C * prod(patch_size), *spatial_reduced]
    """

    class Config(Fig["Patchify"], kw_only=False):
        channels_in: int = -1
        """Number of input channels."""

        _: KW_ONLY

        patch_size: list[int] = field(default_factory=lambda: [2, 2])
        """Patch dimensions per spatial axis."""

        @property
        def channels_out(self) -> int:
            return self.channels_in * math.prod(self.patch_size)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.channels_in = config.channels_in
        self.channels_out = config.channels_out
        self.patch_size = config.patch_size

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        rank = len(self.patch_size)
        spatial_dims = x.shape[-rank:]
        batch_shape = x.shape[: -rank - 1]
        c = x.shape[-rank - 1]

        # Interleave spatial dims with patch dims
        new_shape = [*batch_shape, c]
        for s, p in zip(spatial_dims, self.patch_size, strict=True):
            new_shape.extend([s // p, p])
        x = x.reshape(new_shape)

        # Permute: move patch dims next to channel dim
        # [..., C, s0, p0, s1, p1, ...] -> [..., C, p0, p1, ..., s0, s1, ...]
        n_batch = len(batch_shape)
        c_idx = [n_batch]
        p_indices = [n_batch + 2 + 2 * i for i in range(rank)]
        s_indices = [n_batch + 1 + 2 * i for i in range(rank)]
        perm = list(range(n_batch)) + c_idx + p_indices + s_indices
        x = x.permute(perm)

        reduced_spatial = [
            s // p for s, p in zip(spatial_dims, self.patch_size, strict=True)
        ]
        return x.reshape(*batch_shape, self.channels_out, *reduced_spatial)


class Unpatchify(nn.Module):
    """Reverse of Patchify: unflatten channels back into spatial dims."""

    class Config(Fig["Unpatchify"], kw_only=False):
        channels_out: int = -1
        """Number of output channels after unpatching."""

        _: KW_ONLY

        patch_size: list[int] = field(default_factory=lambda: [2, 2])
        """Patch dimensions per spatial axis."""

        @property
        def channels_in(self) -> int:
            return self.channels_out * math.prod(self.patch_size)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.channels_in = config.channels_in
        self.channels_out = config.channels_out
        self.patch_size = config.patch_size

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        rank = len(self.patch_size)
        spatial_dims = x.shape[-rank:]
        batch_shape = x.shape[: -rank - 1]

        # Unflatten channels: [..., C*prod(p), *s] -> [..., C, p0, p1, ..., s0, s1, ...]
        new_shape = [*batch_shape, self.channels_out, *self.patch_size, *spatial_dims]
        x = x.reshape(new_shape)

        # Permute: interleave spatial and patch dims
        # [..., C, p0, p1, ..., s0, s1, ...] -> [..., C, s0, p0, s1, p1, ...]
        n_batch = len(batch_shape)
        c_idx = [n_batch]
        interleaved: list[int] = []
        for i in range(rank):
            interleaved.append(n_batch + 1 + rank + i)  # s_i
            interleaved.append(n_batch + 1 + i)  # p_i
        perm = list(range(n_batch)) + c_idx + interleaved
        x = x.permute(perm)

        full_spatial = [
            s * p for s, p in zip(spatial_dims, self.patch_size, strict=True)
        ]
        return x.reshape(*batch_shape, self.channels_out, *full_spatial)
