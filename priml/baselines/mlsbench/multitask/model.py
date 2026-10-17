"""The vision ResNet with a second linear head beside its own.

The trunk is :class:`~priml.baselines.mlsbench.pytorch_vision.model.ResNet`,
unchanged; its head classifies the fine labels. A second head reads the same
pooled features. The reference registers both heads before it initializes
anything, so the second head's construction draws come before the trunk's
re-initialization draws. This module therefore builds the trunk with its
initializers held back, adds the head, and then applies the trunk's
initializers to the whole network in the reference's order.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import cast, override

from configgle import Fig
from configgle.walk import copy_tree
from torch import Tensor, nn

from priml.baselines.mlsbench.pytorch_vision.model import ResNet, keep_initialization
from priml.model.init import call_init


def keep_default(w: Tensor) -> None:
    """Leave a tensor as torch's constructor initialized it."""
    del w


class TwoHeadResNet(nn.Module):
    """A ResNet whose pooled features feed two linear heads."""

    class Config(Fig["TwoHeadResNet"]):
        """The trunk and the width of the added head."""

        channels_out: int = -1
        """Logits of the added head; the experiment sets it."""

        _: KW_ONLY

        trunk: ResNet.Config = field(default_factory=ResNet.Config)
        """The network up to and including its own head.

        Its ``init_conv``, ``init_head``, ``init_bias`` and ``init`` are applied
        here, after the added head exists: ``init_head`` and ``init_bias`` to
        both heads, ``init`` to the trunk."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        held = copy_tree(config.trunk)
        held.init_conv = held.init_head = held.init_bias = keep_default
        held.init = keep_initialization
        trunk = held.make()
        assert isinstance(trunk, ResNet)
        self.trunk: ResNet = trunk
        self.head = nn.Linear(config.trunk.channels_hidden[-1], config.channels_out)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                call_init(config.trunk.init_conv, module.weight)
        for head in (self.trunk.head, self.head):
            call_init(config.trunk.init_head, head.weight)
            call_init(config.trunk.init_bias, cast(Tensor, head.bias))
        config.trunk.init(self.trunk)

    @override
    def forward(self, media: Tensor) -> tuple[Tensor, Tensor]:
        """Score a batch of images with both heads.

        Args:
          media: ``(B, channels_in, H, W)`` normalized images.

        Returns:
          logits: ``(B, trunk.channels_out)`` scores of the trunk's head.
          logits_added: ``(B, channels_out)`` scores of the added head.

        """
        features = self.features(media)
        return self.trunk.head(features), self.head(features)

    def features(self, media: Tensor) -> Tensor:
        """Return the pooled ``(B, C)`` features both heads read."""
        trunk = self.trunk
        x = trunk.stages(trunk.act(cast(Tensor, trunk.norm(trunk.stem(media)))))
        return cast(Tensor, trunk.pool(x))
