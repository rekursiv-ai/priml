"""VGG-11 with batch normalization, sized for 32x32 images.

The network of the task's ``vgg11-cifar100`` setting: five stages of 3x3
convolutions, each followed by batch norm and ReLU and each stage by a 2x2 max
pool, then a global average pool and a three-layer classifier with dropout.
Every layer keeps torch's default initialization.

Layers are registered in the reference's order and under its names, so
initialization consumes the generator in the same order and the state dict
lines up key for key.
"""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import override

from configgle import Fig
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.baselines.mlsbench.dl.vision.model import BatchNorm, norm_cost
from priml.cost import Cost, cost, elementwise_cost, matmul_cost
from priml.model.conv import conv_cost
from priml.model.pool import avg_pool_cost, max_pool_cost
from priml.model.swiglu import relu


class VGG(nn.Module):
    """Convolution stages, a global average pool and a dropout classifier."""

    class Config(Fig["VGG"]):
        """Widths of the stages and of the classifier."""

        channels_in: int = -1
        """Input image channels; the experiment sets it."""

        channels_out: int = -1
        """Output logits, one per class; the experiment sets it."""

        _: KW_ONLY

        channels_hidden: tuple[tuple[int, ...], ...] = (
            (64,),
            (128,),
            (256, 256),
            (512, 512),
            (512, 512),
        )
        """Per stage, the width of each convolution; a max pool ends every stage."""

        channels_head: tuple[int, ...] = (512, 512)
        """Width of each hidden classifier layer, each followed by ReLU and dropout."""

        dropout: float = 0.5
        """Probability each hidden classifier unit is zeroed in training."""

        image_size: int = 32
        """Side of the square input image, which the cost is spread over."""

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost every stage, the global pool and the classifier.

            Args:
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to the norms.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes and ownership.

            """
            side, channels, total = self.image_size, self.channels_in, Cost()
            for stage in self.channels_hidden:
                for width in stage:
                    total += (
                        conv_cost(
                            channels_in=channels,
                            channels_out=width,
                            kernel_size=3,
                            ndim=2,
                            groups=1,
                            bias=True,
                            input_grid=(side, side),
                            batch_size=batch_size,
                            padding=1,
                            dtype=dtype,
                        )
                        + norm_cost(
                            BatchNorm,
                            channels=width,
                            side=side,
                            batch_size=batch_size,
                            dtype=dtype,
                            **kwargs,
                        )
                        + cost(relu, channels=width * batch_size * side**2, dtype=dtype)
                    )
                    channels = width
                side //= 2
                total += max_pool_cost(
                    channels=channels,
                    kernel_size=2,
                    rows=batch_size * side**2,
                    dtype=dtype,
                )
            total += avg_pool_cost(
                channels=channels,
                positions=side**2,
                batch_size=batch_size,
                dtype=dtype,
            )
            for width in self.channels_head:
                total += (
                    matmul_cost(
                        channels_in=channels,
                        channels_out=width,
                        bias=True,
                        rows=batch_size,
                        dtype=dtype,
                    )
                    + cost(relu, channels=width * batch_size, dtype=dtype)
                    # A Bernoulli draw and a scaled mask multiply forward; the
                    # mask multiply back.
                    + elementwise_cost(
                        primal=2 * width * batch_size,
                        adjoint=width * batch_size,
                        channels=width,
                        rows=batch_size,
                        dtype=dtype,
                        inputs=2,
                    )
                )
                channels = width
            return total + matmul_cost(
                channels_in=channels,
                channels_out=self.channels_out,
                bias=True,
                rows=batch_size,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = config.channels_in
        for stage in config.channels_hidden:
            for width in stage:
                layers += [
                    nn.Conv2d(channels, width, 3, padding=1),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                ]
                channels = width
            layers.append(nn.MaxPool2d(2, 2))
        self.features = nn.Sequential(*layers)
        head: list[nn.Module] = []
        for width in config.channels_head:
            head += [
                nn.Linear(channels, width),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
            ]
            channels = width
        head.append(nn.Linear(channels, config.channels_out))
        self.classifier = nn.Sequential(*head)

    @override
    def forward(self, media: Tensor) -> Tensor:
        """Classify a batch of images.

        Args:
          media: ``(B, channels_in, H, W)`` normalized images, ``H`` and ``W``
            at least ``2 ** len(channels_hidden)``.

        Returns:
          logits: ``(B, channels_out)`` class scores.

        """
        x = self.features(media)
        return self.classifier(functional.adaptive_avg_pool2d(x, 1).flatten(1))
