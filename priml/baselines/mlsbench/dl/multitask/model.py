"""The vision ResNet with a second linear head beside its own.

The trunk is :class:`~priml.baselines.mlsbench.dl.vision.model.ResNet`,
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

import torch

from priml.baselines.mlsbench.dl.vision.model import ResNet, keep_initialization
from priml.cost import Cost, matmul_cost
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

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the trunk and its head, then the added head on the same features.

            Args:
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to the trunk.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes and ownership.

            """
            return self.trunk.cost(
                batch_size=batch_size,
                dtype=dtype,
                **kwargs,
            ) + matmul_cost(
                channels_in=self.trunk.channels_hidden[-1],
                channels_out=self.channels_out,
                bias=True,
                rows=batch_size,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        held = copy_tree(config.trunk)
        held.init_conv = held.init_head = held.init_bias = keep_default
        held.init = keep_initialization
        trunk = held.make()
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
        features = self.trunk.features(media)
        return self.trunk.head(features), self.head(features)
