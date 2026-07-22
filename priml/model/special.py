"""Utility modules: Identity, Skip."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Any, Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn


class Identity(nn.Identity):
    """Identity that returns only the first positional argument."""

    class Config(Fig["Identity"], kw_only=False):
        channels_in: int = -1
        """Number of input channels (-1 to infer from channels_out)."""

        channels_out: int = -1
        """Number of output channels (-1 to infer from channels_in)."""

        _: KW_ONLY

        @override
        def finalize(self) -> Self:
            if self.channels_in == -1:
                self.channels_in = self.channels_out
            if self.channels_out == -1:
                self.channels_out = self.channels_in
            return super().finalize()

    def reset_parameters(self) -> None:
        pass

    @override
    def forward(self, input: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        return input


class Skip(nn.Module):
    """Residual connection: output = x + inner(x, ...)."""

    class Config(Fig["Skip"], kw_only=False):
        inner: Makeable[nn.Module] | None = None
        """Submodule to wrap with a residual connection."""

    def __init__(self, config: Config) -> None:
        if config.inner is None:
            raise ValueError("Must specify `inner`.")
        super().__init__()
        self.inner = config.inner.make()

    def reset_parameters(self) -> None:
        if hasattr(self.inner, "reset_parameters"):
            self.inner.reset_parameters()

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return x + self.inner(x, *args, **kwargs)
