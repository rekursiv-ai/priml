"""MLP-Mixer block."""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Any, Self, override

from configgle import Fig, Makeable
from torch import Tensor, nn

from priml.model.custom_types import ChannelsIn, ChannelsOut, propagate_attr
from priml.model.norm import RMSNorm
from priml.model.swiglu import SwiGLU


class MLPMixerBlock(nn.Module):
    """MLP-Mixer block: token-mixing (over seq dim) + channel-mixing."""

    class Config(Fig["MLPMixerBlock"], kw_only=False):
        channels_in: int = -1
        """Number of input channels."""

        _: KW_ONLY

        seq_len: int = -1
        """Sequence length (used as input dim for the token mixer)."""

        token_mixer: Makeable[nn.Module] = field(default_factory=SwiGLU.Config)
        """Module that mixes across the sequence (token) dimension."""

        channel_mixer: Makeable[nn.Module] = field(default_factory=SwiGLU.Config)
        """Module that mixes across the channel dimension."""

        norm_token: Makeable[nn.Module] = field(default_factory=RMSNorm.Config)
        """Normalization applied before/after token mixing."""

        norm_channel: Makeable[nn.Module] = field(default_factory=RMSNorm.Config)
        """Normalization applied before/after channel mixing."""

        prenorm: bool = True
        """Apply norm before (True) or after (False) each mixer."""

        depth: int = -1
        """Block depth index for depth-scaled init (-1 = no scaling)."""

        @override
        def finalize(self) -> Self:
            # Token mixer operates on the seq_len dimension.
            propagate_attr(
                self.token_mixer, "channels_in", self.seq_len, protocol=ChannelsIn
            )
            propagate_attr(
                self.token_mixer, "channels_out", self.seq_len, protocol=ChannelsOut
            )
            propagate_attr(
                self.norm_token, "channels_in", self.seq_len, protocol=ChannelsIn
            )
            # Channel mixer operates on the channels_in dimension.
            for cfg in (self.channel_mixer, self.norm_channel):
                propagate_attr(
                    cfg, "channels_in", self.channels_in, protocol=ChannelsIn
                )
                propagate_attr(
                    cfg, "channels_out", self.channels_in, protocol=ChannelsOut
                )
            for cfg in (self.token_mixer, self.channel_mixer):
                propagate_attr(cfg, "depth", self.depth)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.prenorm = config.prenorm
        self.depth = config.depth
        self.token_mixer = config.token_mixer.make()
        self.channel_mixer = config.channel_mixer.make()
        self.norm_token = config.norm_token.make()
        self.norm_channel = config.norm_channel.make()

    def reset_parameters(self) -> None:
        for m in (
            self.token_mixer,
            self.channel_mixer,
            self.norm_token,
            self.norm_channel,
        ):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    @override
    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        del args, kwargs
        if self.prenorm:
            xt = x.movedim(-2, -1)
            xt = xt + self.token_mixer(self.norm_token(xt))
            x = xt.movedim(-2, -1)
            x = x + self.channel_mixer(self.norm_channel(x))
        else:
            xt = x.movedim(-2, -1)
            xt = self.norm_token(xt + self.token_mixer(xt))
            x = xt.movedim(-2, -1)
            x = self.norm_channel(x + self.channel_mixer(x))
        assert isinstance(x, Tensor)
        return x
