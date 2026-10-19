"""VGG-16 with batch normalization, in the CIFAR variant the MLS-Bench vision tasks use.

Thirteen 3x3 convolutions in five stages, each stage closed by a 2x2 max pool,
then a global pool and a two-layer classifier with dropout. The reference's
``VGG`` registers its convolutions, normalizations and activations in one
``features`` sequence, its pool after them and its classifier last; this port
registers them in that order, since registration order fixes the order
initialization consumes the generator.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import Self, cast, override

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.baselines.mlsbench.dl.vision.model import (
    BatchNorm,
    GlobalAvgPool,
    ModelInitFn,
    NormFn,
    activation_layer,
    init_layers,
    kaiming_normal_fan_in,
    kaiming_normal_fan_out,
    keep_initialization,
    norm_cost,
)
from priml.cost import Cost, cost, elementwise_cost, matmul_cost
from priml.model.conv import conv_cost
from priml.model.custom_types import ActivationFn
from priml.model.init import InitFn
from priml.model.pool import max_pool_cost
from priml.model.swiglu import relu


class VGG16BN(nn.Module):
    """VGG-16-BN for small images: convolution stages, a global pool, a classifier."""

    class Config(Fig["VGG16BN"]):
        """Widths, classifier and initialization of the network."""

        channels_in: int = -1
        """Input image channels; the experiment sets it."""

        channels_out: int = -1
        """Output logits, one per class; the experiment sets it."""

        _: KW_ONLY

        image_size: int = 32
        """Side of the square input image, which the cost is spread over."""

        channels_hidden: tuple[tuple[int, ...], ...] = (
            (64, 64),
            (128, 128),
            (256, 256, 256),
            (512, 512, 512),
            (512, 512, 512),
        )
        """Width of each convolution, per stage; every stage ends in a 2x2 max pool."""

        channels_hidden_head: int = 512
        """Width of the classifier's hidden layer."""

        dropout: float = 0.5
        """Drop probability before the classifier's last layer."""

        activation: ActivationFn = relu
        """Activation after every normalization and in the classifier."""

        norm: NormFn = BatchNorm
        """Builds the normalization after every convolution."""

        pool: Makeable[nn.Module] = field(default_factory=GlobalAvgPool.Config)
        """Reduces the last stage's ``(B, C, H, W)`` to ``(B, C)`` for the classifier."""

        init_conv: InitFn = kaiming_normal_fan_out
        """Initializer for every convolution weight; their biases keep torch's draw."""

        init_head: InitFn = kaiming_normal_fan_in
        """Initializer for every linear weight."""

        init_bias: InitFn = nn.init.zeros_
        """Initializer for every linear bias."""

        init: ModelInitFn = keep_initialization
        """Runs on the built network after the per-tensor initializers."""

        @override
        def finalize(self) -> Self:
            if not self.channels_hidden or min(map(len, self.channels_hidden)) < 1:
                raise ValueError(
                    f"Every stage needs a convolution; got {self.channels_hidden}.",
                )
            return super().finalize()

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost every stage, the pool and the classifier.

            Args:
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to the norms and the pool.

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
                            self.norm,
                            channels=width,
                            side=side,
                            batch_size=batch_size,
                            dtype=dtype,
                            **kwargs,
                        )
                        + cost(
                            self.activation,
                            channels=width * batch_size * side**2,
                            dtype=dtype,
                        )
                    )
                    channels = width
                side //= 2
                total += max_pool_cost(
                    channels=channels,
                    kernel_size=2,
                    rows=batch_size * side**2,
                    dtype=dtype,
                )
            return (
                total
                + cost(
                    self.pool,
                    channels=channels,
                    seq_len=side**2,
                    batch_size=batch_size,
                    dtype=dtype,
                    **kwargs,
                )
                + self.head_cost(
                    channels_out=self.channels_out,
                    batch_size=batch_size,
                    dtype=dtype,
                )
            )

        def head(self, channels_out: int) -> nn.Sequential:
            """Build the classifier over the pooled features, ``channels_out`` wide.

            Two linear layers with the activation and dropout between them, as
            the reference's ``classifier`` is.

            Args:
              channels_out: Logits of the head.

            Returns:
              head: The classifier, as torch's constructors initialized it.

            """
            return nn.Sequential(
                nn.Linear(self.channels_hidden[-1][-1], self.channels_hidden_head),
                activation_layer(self.activation),
                nn.Dropout(self.dropout),
                nn.Linear(self.channels_hidden_head, channels_out),
            )

        def head_cost(
            self,
            *,
            channels_out: int,
            batch_size: int,
            dtype: torch.dtype | None,
        ) -> Cost:
            """Cost one :meth:`head` of width ``channels_out``.

            Args:
              channels_out: Logits of the head.
              batch_size: Images in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.

            Returns:
              cost: Whole-invocation FLOPs, logical bytes and ownership.

            """
            hidden = self.channels_hidden_head
            return (
                matmul_cost(
                    channels_in=self.channels_hidden[-1][-1],
                    channels_out=hidden,
                    bias=True,
                    rows=batch_size,
                    dtype=dtype,
                )
                + cost(self.activation, channels=hidden * batch_size, dtype=dtype)
                # A Bernoulli draw and a scaled mask multiply forward; the mask
                # multiply back.
                + elementwise_cost(
                    primal=2 * hidden * batch_size,
                    adjoint=hidden * batch_size,
                    channels=hidden,
                    rows=batch_size,
                    dtype=dtype,
                    inputs=2,
                )
                + matmul_cost(
                    channels_in=hidden,
                    channels_out=channels_out,
                    bias=True,
                    rows=batch_size,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = config.channels_in
        for stage in config.channels_hidden:
            for width in stage:
                layers += [
                    nn.Conv2d(channels, width, 3, padding=1),
                    config.norm(width),
                    activation_layer(config.activation),
                ]
                channels = width
            layers.append(nn.MaxPool2d(2, 2))
        self.layers = nn.Sequential(*layers)
        # Between the stages and the classifier, as upstream's ``CustomPool`` is:
        # a pool with parameters draws its initialization in this position.
        self.pool = config.pool.make()
        self.head = config.head(config.channels_out)
        # Construction already drew torch's default initialization; those draws
        # stay in the stream, so the re-initialization lands where upstream's does.
        init_layers(
            self,
            init_conv=config.init_conv,
            init_head=config.init_head,
            init_bias=config.init_bias,
        )
        config.init(self)

    @override
    def forward(self, media: Tensor) -> Tensor:
        """Classify a batch of images.

        Args:
          media: ``(B, channels_in, H, W)`` normalized images.

        Returns:
          logits: ``(B, channels_out)`` class scores.

        """
        return self.head(self.features(media))

    def features(self, media: Tensor) -> Tensor:
        """Return the pooled ``(B, channels_hidden[-1][-1])`` features the head reads."""
        return cast(Tensor, self.pool(self.layers(media)))
