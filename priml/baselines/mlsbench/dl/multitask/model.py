"""A vision network with a second head beside its own.

The trunk is any of the vision networks (``ResNet``, ``VGG16BN``,
``MobileNetV2``), unchanged; its head classifies the fine labels. A second
head, built by the trunk's config as its own head is but at another width,
reads the same pooled features. The reference registers both heads before it
initializes anything, so the second head's construction draws come before the
trunk's re-initialization draws. This module therefore builds the trunk with
its initializers held back, adds the head, and then applies the trunk's
initializers to the whole network in the reference's order.

The class keeps its first trunk's name: its config's printed form is a golden.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import TYPE_CHECKING, Protocol, cast, override, runtime_checkable

from configgle import Fig, Makeable
from configgle.walk import copy_tree
from torch import Tensor, nn

import torch

from priml.baselines.mlsbench.dl.vision.model import (
    ModelInitFn,
    ResNet,
    init_layers,
    keep_initialization,
)


if TYPE_CHECKING:
    from priml.cost import Cost
    from priml.model.init import InitFn


def keep_default(w: Tensor) -> None:
    """Leave a tensor as torch's constructor initialized it."""
    del w


# Only methods: an ``nn.Module`` keeps its children in ``_modules``, where the
# static lookup a runtime-checkable protocol uses never finds an attribute.
@runtime_checkable
class Trunk(Protocol):
    """A built vision network: pooled features, and its head among its children."""

    def features(self, media: Tensor) -> Tensor:
        """Return the pooled ``(B, C)`` features the head reads."""
        raise NotImplementedError

    def get_submodule(self, target: str) -> nn.Module:
        """Return the child named ``target``; the head is ``"head"``."""
        raise NotImplementedError


@runtime_checkable
class TrunkConfig(Makeable[nn.Module], Protocol):
    """A vision network's config: its initializers, and how it builds and costs a head."""

    channels_out: int
    init_conv: InitFn
    init_head: InitFn
    init_bias: InitFn
    init: ModelInitFn

    def head(self, channels_out: int) -> nn.Module:
        """Build a head over the pooled features, ``channels_out`` wide."""
        raise NotImplementedError

    def cost(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype | None,
        **kwargs: object,
    ) -> Cost:
        """Cost the network."""
        raise NotImplementedError

    def head_cost(
        self,
        *,
        channels_out: int,
        batch_size: int,
        dtype: torch.dtype | None,
    ) -> Cost:
        """Cost one head of width ``channels_out``."""
        raise NotImplementedError


class TwoHeadResNet(nn.Module):
    """A vision network whose pooled features feed its own head and a second one."""

    class Config(Fig["TwoHeadResNet"]):
        """The trunk and the width of the added head."""

        channels_out: int = -1
        """Logits of the added head; the experiment sets it."""

        _: KW_ONLY

        trunk: TrunkConfig = field(default_factory=ResNet.Config)
        """The network up to and including its own head.

        Its ``init_conv``, ``init_head``, ``init_bias`` and ``init`` are applied
        here, after the added head exists: ``init_head`` and ``init_bias`` to
        every linear layer of both heads, ``init`` to the trunk."""

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
            ) + self.trunk.head_cost(
                channels_out=self.channels_out,
                batch_size=batch_size,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        held = copy_tree(config.trunk)
        held.init_conv = held.init_head = held.init_bias = keep_default
        held.init = keep_initialization
        trunk = held.make()
        assert isinstance(trunk, Trunk)
        self.trunk: Trunk = trunk
        self.head = config.trunk.head(config.channels_out)
        init_layers(
            self,
            init_conv=config.trunk.init_conv,
            init_head=config.trunk.init_head,
            init_bias=config.trunk.init_bias,
        )
        config.trunk.init(trunk)

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
        fine = cast(Tensor, self.trunk.get_submodule("head")(features))
        return fine, cast(Tensor, self.head(features))
