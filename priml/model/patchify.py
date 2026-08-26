"""Patchify and unpatchify operations."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, override

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

        channels_out: int = -1
        """Number of output channels after patching."""

        _: KW_ONLY

        patch_size: list[int] = field(default_factory=lambda: [2, 2])
        """Patch dimensions per spatial axis."""

        @override
        def finalize(self) -> Self:
            _validate_patch_size(self.patch_size)
            factor = math.prod(self.patch_size)
            if self.channels_in == -1 and self.channels_out != -1:
                if self.channels_out % factor:
                    raise ValueError(
                        f"channels_out={self.channels_out} must be divisible by "
                        f"prod(patch_size)={factor}."
                    )
                self.channels_in = self.channels_out // factor
            if self.channels_out == -1 and self.channels_in != -1:
                self.channels_out = self.channels_in * factor
            if (
                -1 not in (self.channels_in, self.channels_out)
                and self.channels_out != self.channels_in * factor
            ):
                raise ValueError(
                    f"channels_out={self.channels_out} must equal "
                    f"channels_in={self.channels_in} * prod(patch_size)={factor}."
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        _validate_patch_size(config.patch_size)
        self.channels_in = config.channels_in
        self.channels_out = config.channels_out
        self.patch_size = config.patch_size

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return patchify(x, self.patch_size)


class Unpatchify(nn.Module):
    """Reverse of Patchify: unflatten channels back into spatial dims."""

    class Config(Fig["Unpatchify"], kw_only=False):
        channels_in: int = -1
        """Number of input channels before unpatching."""

        channels_out: int = -1
        """Number of output channels after unpatching."""

        _: KW_ONLY

        patch_size: list[int] = field(default_factory=lambda: [2, 2])
        """Patch dimensions per spatial axis."""

        @override
        def finalize(self) -> Self:
            _validate_patch_size(self.patch_size)
            factor = math.prod(self.patch_size)
            if self.channels_in == -1 and self.channels_out != -1:
                self.channels_in = self.channels_out * factor
            if self.channels_out == -1 and self.channels_in != -1:
                if self.channels_in % factor:
                    raise ValueError(
                        f"channels_in={self.channels_in} must be divisible by "
                        f"prod(patch_size)={factor}."
                    )
                self.channels_out = self.channels_in // factor
            if (
                -1 not in (self.channels_in, self.channels_out)
                and self.channels_in != self.channels_out * factor
            ):
                raise ValueError(
                    f"channels_in={self.channels_in} must equal "
                    f"channels_out={self.channels_out} * prod(patch_size)={factor}."
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        _validate_patch_size(config.patch_size)
        self.channels_in = config.channels_in
        self.channels_out = config.channels_out
        self.patch_size = config.patch_size

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        del kwargs
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
