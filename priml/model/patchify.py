"""Patchify and unpatchify operations."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, override

import math

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.math.pixel import patchify, unpatchify
from priml.model.cost import (
    Cost,
    traffic,
)


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
                        f"prod(patch_size)={factor}.",
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
                    f"channels_in={self.channels_in} * prod(patch_size)={factor}.",
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
            """Price one materialized payload permutation per patch, without FLOPs.

            The geometry is patch positions in both directions. Nontrivial
            patches assume a copy; degenerate spatial shapes may permit views.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del seq_len, batch_size, kwargs
            if math.prod(self.patch_size) == 1:
                return Cost()
            moved = 2 * max(self.channels_in, self.channels_out)
            dt = dtype
            return traffic("primal", "selection", elements=moved, dtype=dt) + traffic(
                "adjoint",
                "selection",
                elements=moved,
                dtype=dt,
            )

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
                        f"prod(patch_size)={factor}.",
                    )
                self.channels_out = self.channels_in // factor
            if (
                -1 not in (self.channels_in, self.channels_out)
                and self.channels_in != self.channels_out * factor
            ):
                raise ValueError(
                    f"channels_in={self.channels_in} must equal "
                    f"channels_out={self.channels_out} * prod(patch_size)={factor}.",
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
            """Price one materialized payload permutation per patch, without FLOPs.

            The geometry is patch positions in both directions. Nontrivial
            patches assume a copy; degenerate spatial shapes may permit views.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Per-token cost of this module.

            """
            del seq_len, batch_size, kwargs
            if math.prod(self.patch_size) == 1:
                return Cost()
            moved = 2 * max(self.channels_in, self.channels_out)
            dt = dtype
            return traffic("primal", "selection", elements=moved, dtype=dt) + traffic(
                "adjoint",
                "selection",
                elements=moved,
                dtype=dt,
            )

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


# ``patchify`` checks the same thing, but only once a tensor arrives; a config error
# belongs at construction, where the field that carries it is still in scope.
def _validate_patch_size(patch_size: list[int]) -> None:
    """Reject a patch that cannot tile anything, naming the config field."""
    if not patch_size:
        raise ValueError("patch_size must name at least one axis.")
    if any(p < 1 for p in patch_size):
        raise ValueError(f"patch_size entries must be positive; got {patch_size}.")
