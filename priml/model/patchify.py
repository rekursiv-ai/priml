"""Patchify and unpatchify operations."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Any, override

import math

from configgle import Fig
from torch import Tensor, nn

from priml.math.pixel import patchify, unpatchify


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
        _validate_patch_size(config.patch_size)
        self.channels_in = config.channels_in
        self.channels_out = config.channels_out
        self.patch_size = config.patch_size

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        return patchify(x, self.patch_size)


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
        _validate_patch_size(config.patch_size)
        self.channels_in = config.channels_in
        self.channels_out = config.channels_out
        self.patch_size = config.patch_size

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        return unpatchify(x, self.patch_size)


def _validate_patch_size(patch_size: list[int]) -> None:
    """Reject a patch that cannot tile anything, naming the config field.

    ``patchify`` checks the same thing, but only once a tensor arrives; a
    config error belongs at construction, where the field that carries it is
    still in scope.
    """
    if not patch_size:
        raise ValueError("patch_size must name at least one axis.")
    if any(p < 1 for p in patch_size):
        raise ValueError(f"patch_size entries must be positive; got {patch_size}.")
