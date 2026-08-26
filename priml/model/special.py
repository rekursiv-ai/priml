"""Utility modules: Identity, Skip."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn

from priml.model.custom_types import (
    ChannelsIn,
    ChannelsOut,
    TensorModule,
)
from priml.model.passthrough import (
    PassthroughAttribute,
    ReadPassthroughMixin,
    ReadWritePassthroughMixin,
)


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
            if self.channels_in != self.channels_out:
                raise ValueError(
                    f"channels_in={self.channels_in} must equal "
                    f"channels_out={self.channels_out} for Identity."
                )
            return super().finalize()

    def __init__(self, config: Config) -> None:
        if (
            -1 not in (config.channels_in, config.channels_out)
            and config.channels_in != config.channels_out
        ):
            raise ValueError(
                f"channels_in={config.channels_in} must equal "
                f"channels_out={config.channels_out} for Identity."
            )
        super().__init__()

    def reset_parameters(self) -> None:
        pass

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return input


class Skip(ReadPassthroughMixin, nn.Module, passthrough="inner"):
    """Residual connection: output = x + inner(x, ...)."""

    class Config(
        ReadWritePassthroughMixin,
        Fig["Skip"],
        kw_only=False,
        passthrough="inner",
    ):
        inner: Makeable[TensorModule] | None = None
        """Submodule to wrap with a residual connection."""

        channels_in = PassthroughAttribute[int]()
        channels_out = PassthroughAttribute[int]()

    def __init__(self, config: Config) -> None:
        if config.inner is None:
            raise ValueError("Must specify `inner`.")
        inner = config.inner
        if (
            isinstance(inner, ChannelsIn)
            and isinstance(inner, ChannelsOut)
            and inner.channels_in != inner.channels_out
        ):
            raise ValueError(
                f"channels_in={inner.channels_in} must equal "
                f"channels_out={inner.channels_out} for Skip."
            )
        super().__init__()
        self.inner = config.inner.make()

    def reset_parameters(self) -> None:
        if hasattr(self.inner, "reset_parameters"):
            self.inner.reset_parameters()

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        inner = self.inner(x, **kwargs)
        assert isinstance(inner, Tensor)
        return x + inner
