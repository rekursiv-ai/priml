"""Shared timestep and class embedders for latent diffusion transformers."""

from __future__ import annotations

from dataclasses import KW_ONLY
from typing import Self, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import Cost, elementwise_cost, matmul_cost, traffic
from priml.math.diffusion.conditioning import timestep_embedding


class TimestepEmbedder(nn.Module):
    """Embed a scalar flow time into the conditioning width."""

    class Config(Fig["TimestepEmbedder"], kw_only=False):
        """Configuration for TimestepEmbedder."""

        channels_out: int = -1
        """Conditioning width produced by the embedder."""

        _: KW_ONLY

        channels_frequency: int = 256
        """Width of the raw sinusoidal features feeding the projection."""

        max_period: float = 10_000.0
        """Longest sinusoid period; sets the lowest represented frequency."""

        activation: Makeable[nn.Module] | None = None
        """Activation between the two projections; ``None`` uses ``SiLU``."""

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one timestep embedding.

            Args:
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            del kwargs
            return (
                matmul_cost(
                    channels_in=self.channels_frequency,
                    channels_out=self.channels_out,
                    bias=True,
                    rows=batch_size,
                    dtype=dtype,
                )
                + matmul_cost(
                    channels_in=self.channels_out,
                    channels_out=self.channels_out,
                    bias=True,
                    rows=batch_size,
                    dtype=dtype,
                )
                + elementwise_cost(
                    primal=5,
                    adjoint=5,
                    channels=self.channels_out,
                    rows=batch_size,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.channels_frequency = config.channels_frequency
        self.max_period = config.max_period
        activation = (
            nn.SiLU() if config.activation is None else config.activation.make()
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.channels_frequency, config.channels_out, bias=True),
            activation,
            nn.Linear(config.channels_out, config.channels_out, bias=True),
        )

    def frequencies(self, t: Tensor) -> Tensor:
        """Build raw sinusoidal features for a batch of times.

        Args:
          t: Flow times, ``[batch]``.

        Returns:
          features: ``[batch, channels_frequency]``, ``cos`` then ``sin``.

        """
        return timestep_embedding(t, self.channels_frequency, self.max_period)

    @override
    def forward(self, t: Tensor, **kwargs: object) -> Tensor:
        """Embed flow times.

        Args:
          t: Flow times, ``[batch]``.
          **kwargs: The open bus, unread here.

        Returns:
          conditioning: ``[batch, channels_out]``.

        """
        del kwargs
        return self.mlp(self.frequencies(t).to(t.dtype))


class LabelEmbedder(nn.Module):
    """Embed class labels, dropping a fraction to a learned null class."""

    class Config(Fig["LabelEmbedder"], kw_only=False):
        """Configuration for LabelEmbedder."""

        channels_in: int = -1
        """Number of real classes; the null class is appended beyond them."""

        channels_out: int = -1
        """Conditioning width produced by the table."""

        _: KW_ONLY

        dropout: float = 0.1
        """Probability of replacing a label with the null class while training.

        Zero removes the null row entirely rather than leaving it untrained,
        so a run without classifier-free guidance carries no dead parameters."""

        @override
        def finalize(self) -> Self:
            if math.isnan(self.dropout) or self.dropout < 0.0 or self.dropout >= 1.0:
                raise ValueError(f"dropout must be in [0, 1); got {self.dropout}.")
            return super().finalize()

        @property
        def num_rows(self) -> int:
            """Rows in the embedding table, including any null class.

            Returns:
              rows: ``channels_in`` plus one when dropout is enabled.

            """
            return self.channels_in + int(self.dropout > 0)

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one label lookup.

            Args:
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            del kwargs
            # One row read per lookup, as ``priml.model.embedding`` counts it.
            table = self.num_rows * self.channels_out
            return traffic(
                "primal",
                "selection",
                elements=batch_size * self.channels_out,
                dtype=dtype,
            ) + Cost(params=table, params_active=self.channels_out)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_classes = config.channels_in
        self.dropout = config.dropout
        self.embedding_table = nn.Embedding(config.num_rows, config.channels_out)

    def token_drop(self, labels: Tensor) -> Tensor:
        """Replace a random fraction of labels with the null class.

        Args:
          labels: Class indices, ``[batch]``.

        Returns:
          labels: Indices with dropped entries set to the null class.

        """
        drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout
        return torch.where(drop, self.num_classes, labels)

    @override
    def forward(
        self,
        labels: Tensor,
        *,
        force_drop: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Embed labels, dropping some while training.

        Args:
          labels: Class indices, ``[batch]``.
          force_drop: Explicit mask selecting the null class when supplied.
          **kwargs: The open bus, unread here.

        Returns:
          conditioning: ``[batch, channels_out]``.

        """
        del kwargs
        if force_drop is not None:
            labels = torch.where(force_drop.bool(), self.num_classes, labels)
        elif self.training and self.dropout > 0:
            labels = self.token_drop(labels)
        return self.embedding_table(labels)
