"""Shared timestep and class embedders for latent diffusion transformers."""

from __future__ import annotations

from dataclasses import KW_ONLY
from functools import partial
from typing import Protocol, Self, override, runtime_checkable

import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import Cost, elementwise_cost, matmul_cost, traffic
from priml.math.diffusion.conditioning import timestep_embedding
from priml.model.custom_types import (
    ChannelsInOut,
    ChannelsOut,
    WeightedTensorModule,
)
from priml.model.embedding import Embedding
from priml.model.init import kaiming_uniform, normal
from priml.model.linear import Linear


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
            nn.Linear(config.channels_frequency, config.channels_out),
            activation,
            nn.Linear(config.channels_out, config.channels_out),
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


@runtime_checkable
class TaskComposer(Protocol):
    """A compositional task-memory module that returns one vector per row."""

    channels_out: int

    def __call__(
        self,
        category_indices: Tensor,
        augmentation_indices: Tensor,
        slot_permutation: Tensor,
    ) -> Tensor:
        """Compose semantic puzzle and augmentation identities."""
        ...

    def reset_parameters(self) -> None:
        """Reset parameters in the compositional module."""
        ...


class TaskComposerConfig(Makeable[TaskComposer], ChannelsOut, Protocol):
    """Injectable config for task composition, width, and its analytical cost."""

    def cost(
        self,
        *,
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype | None,
        **kwargs: object,
    ) -> Cost:
        """Price task composition for a complete batch."""
        ...


@runtime_checkable
class ResidualGate(Protocol):
    """A task-conditioned multiplicative gate for an instance residual."""

    channels_out: int

    def __call__(self, task: Tensor, /) -> Tensor:
        """Return one gate vector per task vector."""
        ...

    def reset_parameters(self) -> None:
        """Reset parameters in the residual gate."""
        ...


class ResidualGateConfig(Makeable[ResidualGate], ChannelsOut, Protocol):
    """Injectable config for an instance-residual gate and its cost."""

    def cost(
        self,
        *,
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype | None,
        **kwargs: object,
    ) -> Cost:
        """Price gate projection and elementwise sigmoid."""
        ...


class ResettableProjection(WeightedTensorModule, Protocol):
    """Weighted projection that participates in module reinitialization."""

    @override
    def reset_parameters(self) -> None:
        """Reset the projection's parameters."""
        ...


class WeightedProjectionConfig(
    Makeable[ResettableProjection],
    ChannelsInOut,
    Protocol,
):
    """Injectable weighted projection with explicit width and cost contracts."""

    def cost(
        self,
        *,
        seq_len: int,
        batch_size: int,
        dtype: torch.dtype | None,
        **kwargs: object,
    ) -> Cost:
        """Price the projection for a complete batch."""
        ...


class CategorySlotFiLM(nn.Module):
    """Compose category, augmentation, and permuted slot embeddings with FiLM."""

    class Config(Fig["CategorySlotFiLM"]):
        """Category, augmentation, and slot table geometry."""

        num_categories: int = 1
        """Number of semantic category identities."""

        num_augmentations: int = 1
        """Number of augmentation identities."""

        num_slots: int = 1
        """Number of independently permuted learned slots."""

        channels_out: int = -1
        """Composed task-vector width; inherited from the prefix."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost embeddings, slot gathering, FiLM projections, and composition.

            Args:
              seq_len: Ignored; one task vector is composed per puzzle.
              batch_size: Puzzle rows in the complete batch.
              dtype: Activation dtype used by the FiLM projections.
              **kwargs: Unread cost bus entries.

            Returns:
              cost: FLOPs, logical bytes, and owned parameters.

            """
            del seq_len, kwargs
            width = self.channels_out
            slot_dim = width // (self.num_slots + 1)
            base_dim = width - self.num_slots * slot_dim
            rows = batch_size
            projection = matmul_cost(
                channels_in=2 * width,
                channels_out=width,
                bias=True,
                rows=rows,
                dtype=dtype,
            )
            projection = projection + projection
            table_params = (
                self.num_categories + self.num_augmentations
            ) * width + width
            active = (
                min(rows, self.num_categories) * width
                + min(rows, self.num_augmentations) * width
                + width
            )
            embeddings = traffic(
                "primal",
                "selection",
                elements=(2 + self.num_slots) * rows,
                dtype=torch.int64,
            ) + traffic(
                "primal",
                "selection",
                elements=rows * (2 * width + self.num_slots * slot_dim + base_dim),
                dtype=dtype,
            )
            formula = elementwise_cost(
                primal=3 * rows * width,
                adjoint=4 * rows * width,
                channels=3 * width,
                inputs=2,
                rows=rows,
                dtype=dtype,
            )
            return (
                projection
                + embeddings
                + formula
                + Cost(
                    params=table_params,
                    params_active=active,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.channels_out <= 0:
            raise ValueError("CategorySlotFiLM requires positive channels_out.")
        if min(config.num_categories, config.num_augmentations, config.num_slots) < 1:
            raise ValueError(
                "Category, augmentation, and slot counts must be positive.",
            )
        if config.channels_out < config.num_slots + 1:
            raise ValueError(
                "CategorySlotFiLM channels_out must provide a channel for each "
                "slot and the base.",
            )
        self.channels_out = config.channels_out
        self.category_embed = Embedding.Config(
            channels_in=config.num_categories,
            channels_out=config.channels_out,
            init_weight=partial(normal, std=1.0),
        ).make()
        with torch.no_grad():
            self.category_embed.weight.zero_()
        self.augmentation_embed = Embedding.Config(
            channels_in=config.num_augmentations,
            channels_out=config.channels_out,
            init_weight=partial(normal, std=1.0),
        ).make()
        with torch.no_grad():
            self.augmentation_embed.weight.zero_()
        self.num_slots = config.num_slots
        self.slot_dim = config.channels_out // (config.num_slots + 1)
        self.base_dim = config.channels_out - config.num_slots * self.slot_dim
        # The single source head axis affects parameter rank and optimizer role.
        self.slot_base = nn.Parameter(torch.zeros(1, self.base_dim))
        self.slot_embeddings = nn.Parameter(
            torch.zeros(1, config.num_slots, self.slot_dim),
        )
        self.film_gamma = Linear.Config(
            channels_in=config.channels_out * 2,
            channels_out=config.channels_out,
            bias=True,
            init_weight=kaiming_uniform,
            init_bias=None,
        ).make()
        self.film_beta = Linear.Config(
            channels_in=config.channels_out * 2,
            channels_out=config.channels_out,
            bias=True,
            init_weight=kaiming_uniform,
            init_bias=None,
        ).make()

    def reset_parameters(self) -> None:
        """Reset every child while restoring the zero semantic tables."""
        self.category_embed.reset_parameters()
        self.augmentation_embed.reset_parameters()
        with torch.no_grad():
            self.category_embed.weight.zero_()
            self.augmentation_embed.weight.zero_()
            self.slot_base.zero_()
            self.slot_embeddings.zero_()
        self.film_gamma.reset_parameters()
        self.film_beta.reset_parameters()

    @override
    def forward(
        self,
        category_indices: Tensor,
        augmentation_indices: Tensor,
        slot_permutation: Tensor,
    ) -> Tensor:
        """Return each semantic task vector after slot permutation and FiLM.

        Args:
          category_indices: ``[B]`` zero-based semantic category IDs.
          augmentation_indices: ``[B]`` zero-based augmentation IDs.
          slot_permutation: ``[B, num_slots]`` zero-based slot indices.

        Returns:
          task: ``[B, channels_out]`` composed semantic task vectors.

        """
        batch_size = category_indices.shape[0]
        category = self.category_embed(category_indices.long()).unsqueeze(1)
        augmentation = self.augmentation_embed(augmentation_indices.long()).unsqueeze(1)
        slots = self.slot_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)
        slot_indices = (
            slot_permutation.long()
            .unsqueeze(1)
            .unsqueeze(-1)
            .expand(
                -1,
                1,
                -1,
                self.slot_dim,
            )
        )
        permuted_slots = torch.gather(slots, 2, slot_indices).flatten(start_dim=2)
        base = self.slot_base.unsqueeze(0).expand(batch_size, -1, -1)
        slot_vector = torch.cat([base, permuted_slots], dim=-1)
        aug = torch.cat([augmentation, slot_vector], dim=-1)
        gamma = self.film_gamma(aug.squeeze(1)).unsqueeze(1)
        beta = self.film_beta(aug.squeeze(1)).unsqueeze(1)
        return ((1 + gamma) * category + beta).squeeze(1)


class SigmoidResidualGate(nn.Module):
    """Gate an instance residual using its semantic task representation."""

    class Config(Fig["SigmoidResidualGate"]):
        """Width of the task representation used by the gate."""

        channels_out: int = -1
        """Task-vector width; inherited from the prefix."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the gate projection and sigmoid.

            Args:
              seq_len: Ignored; one gate is computed per puzzle.
              batch_size: Puzzle rows in the complete batch.
              dtype: Activation dtype used by the gate projection.
              **kwargs: Unread cost bus entries.

            Returns:
              cost: FLOPs, logical bytes, and owned parameters.

            """
            del seq_len, kwargs
            width = self.channels_out
            return matmul_cost(
                channels_in=width,
                channels_out=width,
                bias=True,
                rows=batch_size,
                dtype=dtype,
            ) + elementwise_cost(
                primal=batch_size * width,
                adjoint=2 * batch_size * width,
                channels=width,
                rows=batch_size,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.channels_out = config.channels_out
        self.linear = Linear.Config(
            channels_in=config.channels_out,
            channels_out=config.channels_out,
            bias=True,
            init_weight=kaiming_uniform,
            init_bias=None,
        ).make()
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 2.0)

    def reset_parameters(self) -> None:
        """Reset the linear, then restore its initially open sigmoid gate."""
        self.linear.reset_parameters()
        with torch.no_grad():
            self.linear.weight.zero_()
            if self.linear.bias is not None:
                self.linear.bias.fill_(2.0)

    @override
    def forward(self, task: Tensor) -> Tensor:
        """Return a sigmoid gate for each task vector.

        Args:
          task: ``[B, channels_out]`` semantic task vectors.

        Returns:
          gate: ``[B, channels_out]`` multiplicative residual gates.

        """
        return torch.sigmoid(self.linear(task))
