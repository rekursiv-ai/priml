"""SR-DiT: the SpeedrunDiT latent flow-matching transformer, natively in Priml.

A SiT-style diffusion transformer over INVAE latents with four additions the
upstream recipe measures as load-bearing, each of which is a slot here rather
than a flag: 2D rotary positions, SPRINT sparse-dense residual fusion, value
residual learning, and a diffused class token carried alongside the patches.

Construction order is a contract here, not a style. ``reset_parameters``
re-draws every linear layer after the whole tree exists, so what a run starts
from depends on how many random numbers each constructor consumed before it:
reordering two submodules shifts every later draw. That is also why the leaves
on this path are torch's own rather than their Priml wrappers, which draw
different counts.

References:
  https://github.com/SwayStar123/SpeedrunDiT
    The reference, pinned at c24c2ff25699cce63174ca56c2afcfeeb225e367.
  https://arxiv.org/abs/2512.12386
    Bhanded 2025, "Speedrunning ImageNet Diffusion."
  https://arxiv.org/abs/2401.08740
    Ma et al. 2024, "SiT: Exploring Flow and Diffusion-based Generative Models."

"""

from __future__ import annotations

from dataclasses import KW_ONLY, field
from typing import NamedTuple, Protocol, Self, cast, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn

import torch

from priml.cost import (
    Cost,
    cost,
    elementwise_cost,
    map_cost,
    matmul_cost,
    set_cost,
    traffic,
)
from priml.math.custom_types import TensorFn
from priml.model.attention.kernel import SdpaFused
from priml.model.custom_types import (
    ActivationFn,
    AttentionKernel,
    ChannelsIn,
    TensorModule,
    propagate_attr,
)
from priml.model.norm import RMSNorm


__all__ = [
    "AdaLNModulation",
    "DiTBlock",
    "FeedForward",
    "FinalLayer",
    "LabelEmbedder",
    "PatchEmbed",
    "SelfAttention",
    "SpeedrunDiT",
    "SprintRouting",
    "TimestepEmbedder",
    "ValueBlend",
    "ValueResidual",
    "VisionRoPE",
    "gelu_tanh",
    "modulate",
    "sincos_position_table",
]


@set_cost(map_cost(primal=8, adjoint=12))
def gelu_tanh(x: Tensor) -> Tensor:
    """Apply GELU under its tanh approximation.

    Priml carries no GELU: ``priml.model.swiglu`` registers ``relu``,
    ``sigmoid``, ``silu`` and ``relu_squared``, and ``cost_test`` asserts that a
    bare ``functional.gelu`` has no cost. So the activation is defined here,
    costed, and passed as a value into the feed-forward's ``activation`` slot.

    Args:
      x: Pre-activation values, any shape.

    Returns:
      activated: ``gelu(x, approximate="tanh")``, elementwise.

    """
    return nn.functional.gelu(x, approximate="tanh")


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply an adaLN scale-shift over the token axis.

    Args:
      x: Normalized activations, ``[batch, tokens, channels]``.
      shift: Additive term, ``[batch, channels]``.
      scale: Multiplicative offset about one, ``[batch, channels]``.

    Returns:
      modulated: ``x * (1 + scale) + shift``, broadcast over tokens.

    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def sincos_position_table(channels: int, grid: int, *, lead: int = 1) -> Tensor:
    """Build the frozen 2D sin-cos position table, with zeroed lead rows.

    Computed in float64 and rounded once to float32, which is what the
    reference does by building it in NumPy; the intermediate width is visible
    here rather than implied by a dtype three call frames away.

    Note the ordering differs from :func:`TimestepEmbedder.frequencies`: this
    table concatenates ``sin`` then ``cos``, the timestep embedding ``cos`` then
    ``sin``. They are not interchangeable and must not be unified.

    Args:
      channels: Embedding width; must be divisible by four.
      grid: Side length of the square patch grid.
      lead: Leading rows zeroed for non-spatial tokens (the class token).

    Returns:
      table: ``[lead + grid * grid, channels]`` float32 positions.

    Raises:
      ValueError: If ``channels`` is not divisible by four.

    """
    if channels % 4:
        raise ValueError(f"channels must be divisible by four; got {channels}.")
    half = channels // 2
    omega = torch.arange(half // 2, dtype=torch.float64) / (half / 2.0)
    omega = 1.0 / 10_000**omega
    steps = torch.arange(grid, dtype=torch.float64)
    # meshgrid's width axis varies fastest, so the first half of the channels
    # encodes the COLUMN and the second half the row. Swapping them still
    # trains, and silently stops matching a reference checkpoint.
    cols, rows = torch.meshgrid(steps, steps, indexing="xy")
    parts = [
        torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        for out in (
            torch.outer(cols.reshape(-1), omega),
            torch.outer(rows.reshape(-1), omega),
        )
    ]
    table = torch.cat(parts, dim=1)
    return torch.cat([table.new_zeros(lead, channels), table]).float()


class PatchEmbed(nn.Module):
    """Project image-shaped latents to a token sequence with a strided conv."""

    class Config(Fig["PatchEmbed"], kw_only=False):
        """Configuration for PatchEmbed."""

        channels_in: int = -1
        """Latent channels entering the patch projection."""

        channels_out: int = -1
        """Token width leaving the patch projection."""

        _: KW_ONLY

        image_size: int = 16
        """Side length of the square latent grid."""

        patch_size: int = 1
        """Side length of one square patch."""

        bias: bool = True
        """Whether the projection carries a bias."""

        @property
        def grid(self) -> int:
            """Patches along one side.

            Returns:
              grid: ``image_size // patch_size``.

            """
            return self.image_size // self.patch_size

        @property
        def num_patches(self) -> int:
            """Tokens produced by one image.

            Returns:
              count: ``grid ** 2``.

            """
            return self.grid**2

        @override
        def finalize(self) -> Self:
            if self.image_size % self.patch_size:
                raise ValueError(
                    f"image_size {self.image_size} must be divisible by "
                    f"patch_size {self.patch_size}.",
                )
            return super().finalize()

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one patch projection.

            A stride-equal-kernel convolution touches each input element once,
            so it prices exactly as the equivalent per-patch matmul.

            Args:
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            del kwargs
            return matmul_cost(
                channels_in=self.channels_in * self.patch_size**2,
                channels_out=self.channels_out,
                bias=self.bias,
                rows=batch_size * self.num_patches,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.grid = config.grid
        self.num_patches = config.num_patches
        self.proj = nn.Conv2d(
            config.channels_in,
            config.channels_out,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=config.bias,
        )

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        """Patchify and flatten to a token sequence.

        Args:
          x: Latents, ``[batch, channels_in, image_size, image_size]``.
          **kwargs: The open bus, unread here.

        Returns:
          tokens: ``[batch, num_patches, channels_out]``.

        """
        del kwargs
        return self.proj(x).flatten(2).transpose(1, 2)


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
        half = self.channels_frequency // 2
        # float32 regardless of the activation dtype: the frequencies span four
        # decades, and drawing them in bf16 collapses the highest ones onto each
        # other. The cast back to t.dtype happens once, at the end.
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half,
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.channels_frequency % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], -1)
        return embedding

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
    def forward(self, labels: Tensor, **kwargs: object) -> Tensor:
        """Embed labels, dropping some while training.

        Args:
          labels: Class indices, ``[batch]``.
          **kwargs: The open bus, unread here.

        Returns:
          conditioning: ``[batch, channels_out]``.

        """
        del kwargs
        if self.training and self.dropout > 0:
            labels = self.token_drop(labels)
        return self.embedding_table(labels)


class AdaLNModulation(nn.Module):
    """Project conditioning to adaLN shift, scale and gate triples.

    Zero-initialized, which is what makes each gated sublayer an identity at
    step zero rather than a random perturbation.
    """

    class Config(Fig["AdaLNModulation"], kw_only=False):
        """Configuration for AdaLNModulation."""

        channels_in: int = -1
        """Residual width being modulated."""

        _: KW_ONLY

        cond_dim: int = -1
        """Conditioning width entering the projection."""

        num_groups: int = 6
        """Modulation triples produced; two per gated sublayer."""

        activation: Makeable[nn.Module] | None = None
        """Activation before the projection; ``None`` uses ``SiLU``."""

        def cost(
            self,
            *,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one modulation projection.

            Args:
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            del kwargs
            return matmul_cost(
                channels_in=self.cond_dim,
                channels_out=self.num_groups * self.channels_in,
                bias=True,
                rows=batch_size,
                dtype=dtype,
            ) + elementwise_cost(
                primal=5,
                adjoint=5,
                channels=self.cond_dim,
                rows=batch_size,
                dtype=dtype,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_groups = config.num_groups
        activation = (
            nn.SiLU() if config.activation is None else config.activation.make()
        )
        # Not ``priml.model.AdaLNZero``, which computes this exact projection:
        # its chunks are ordered (scale, shift, gate) where these rows are
        # (shift, scale, gate), so one weight tensor means different things in
        # the two. Its zero-init also draws nothing where this draws 6H*H + 6H.
        self.modulation = nn.Sequential(
            activation,
            nn.Linear(config.cond_dim, config.num_groups * config.channels_in, True),
        )

    @override
    def forward(self, c: Tensor, **kwargs: object) -> tuple[Tensor, ...]:
        """Project conditioning to modulation parameters.

        Args:
          c: Conditioning, ``[batch, cond_dim]``.
          **kwargs: The open bus, unread here.

        Returns:
          groups: ``num_groups`` tensors of ``[batch, channels_in]``.

        """
        del kwargs
        return self.modulation(c).chunk(self.num_groups, dim=-1)

    @property
    def proj(self) -> nn.Linear:
        """The projection initialization zeroes.

        Returns:
          proj: The last layer of ``modulation``.

        """
        return _linear(self.modulation[-1])


def _linear(module: nn.Module) -> nn.Linear:
    """Narrow a container's member to the linear layer it is built as."""
    assert isinstance(module, nn.Linear)
    return module


class FeedForward(nn.Module):
    """Two-layer position-wise feed-forward network."""

    class Config(Fig["FeedForward"], kw_only=False):
        """Configuration for FeedForward."""

        channels_in: int = -1
        """Input and output width."""

        _: KW_ONLY

        channels_hidden: int = -1
        """Inner width; ``-1`` derives it from ``expansion``."""

        expansion: float = 4.0
        """Inner-width multiple used when ``channels_hidden`` is ``-1``."""

        bias: bool = True
        """Whether both projections carry biases."""

        activation: ActivationFn = gelu_tanh
        """Elementwise nonlinearity between the projections.

        Must carry a ``@set_cost`` registration, since the parent costs it."""

        @override
        def finalize(self) -> Self:
            if self.channels_hidden == -1:
                self.channels_hidden = int(self.channels_in * self.expansion)
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one feed-forward pass.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            del kwargs
            rows = seq_len * batch_size
            return (
                matmul_cost(
                    channels_in=self.channels_in,
                    channels_out=self.channels_hidden,
                    bias=self.bias,
                    rows=rows,
                    dtype=dtype,
                )
                + matmul_cost(
                    channels_in=self.channels_hidden,
                    channels_out=self.channels_in,
                    bias=self.bias,
                    rows=rows,
                    dtype=dtype,
                )
                + cost(
                    self.activation,
                    channels=self.channels_hidden * rows,
                    dtype=dtype,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.channels_in, config.channels_hidden, config.bias)
        self.act = _activation(config.activation)
        self.fc2 = nn.Linear(config.channels_hidden, config.channels_in, config.bias)

    @override
    def forward(self, x: Tensor, **kwargs: object) -> Tensor:
        """Apply the feed-forward network.

        Args:
          x: Activations, ``[..., channels_in]``.
          **kwargs: The open bus, unread here.

        Returns:
          activations: Same shape as ``x``.

        """
        del kwargs
        return self.fc2(self.act(self.fc1(x)))


def _activation(activation: ActivationFn) -> TensorFn:
    """Build an activation from a config, or pass a callable through."""
    if isinstance(activation, Makeable):
        # ``Makeable`` is runtime-checkable, so isinstance erases its type
        # parameter: ``make`` reads as returning ``object`` without the cast.
        return cast(TensorFn, activation.make())
    return activation


class ValueBlend(Protocol):
    """Blends one layer's attention values with the first layer's."""

    def __call__(self, v: Tensor, first: Tensor, /) -> Tensor:
        """Apply to the input."""
        ...


class ValueResidual(nn.Module):
    """Blend a layer's values toward the first layer's, by a learned scalar.

    References:
      https://arxiv.org/abs/2410.17897
        Zhou et al. 2024, "Value Residual Learning For Alleviating Attention
        Concentration In Transformers."

    """

    class Config(Fig["ValueResidual"], kw_only=False):
        """Configuration for ValueResidual."""

        initial: float = 0.5
        """Starting mixing weight on the first layer's values."""

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            heads: int,
            channels_head: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one value blend.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              heads: Attention heads.
              channels_head: Width of one head.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            del kwargs
            return elementwise_cost(
                primal=3,
                adjoint=4,
                channels=channels_head,
                rows=seq_len * batch_size * heads,
                dtype=dtype,
                inputs=2,
                params=1,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(config.initial))

    @override
    def forward(self, v: Tensor, first: Tensor, **kwargs: object) -> Tensor:
        """Blend values toward the first layer's.

        Args:
          v: This layer's values, ``[batch, heads, tokens, channels_head]``.
          first: First layer's values, same shape.
          **kwargs: The open bus, unread here.

        Returns:
          blended: ``weight * first + (1 - weight) * v``.

        """
        del kwargs
        return self.weight * first + (1.0 - self.weight) * v


class VisionRoPE(nn.Module):
    """Axial 2D rotary positions over a square patch grid.

    Rotates the full head width: the row phases occupy the first half of the
    head channels and the column phases the second, and within each half the
    frequencies are duplicated adjacently so that :func:`rotate_pairs` pairs
    channel ``2i`` with ``2i + 1``. That interleaved convention is not the
    split-halves one, and the two are not interchangeable.

    References:
      https://arxiv.org/abs/2303.11331
        Fang et al. 2023, "EVA-02: A Visual Representation for Neon Genesis."

    """

    class Config(Fig["VisionRoPE"], kw_only=False):
        """Configuration for VisionRoPE."""

        channels_head: int = -1
        """Head width; the whole width is rotated, so this must be a multiple
        of four."""

        _: KW_ONLY

        grid: int = -1
        """Side length of the square patch grid."""

        theta: float = 10_000.0
        """Base of the geometric frequency ladder."""

        @override
        def finalize(self) -> Self:
            if self.channels_head % 4:
                raise ValueError(
                    "channels_head must be divisible by four for axial 2D "
                    f"rotation; got {self.channels_head}.",
                )
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            heads: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost rotating queries and keys.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              heads: Attention heads.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            del kwargs
            return elementwise_cost(
                primal=3,
                adjoint=3,
                channels=self.channels_head,
                rows=seq_len * batch_size * heads * 2,
                dtype=dtype,
                inputs=2,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.channels_head = config.channels_head
        self.grid = config.grid
        quarter = config.channels_head // 4
        freqs = 1.0 / (
            config.theta ** (torch.arange(0, quarter, dtype=torch.float32) / quarter)
        )
        steps = torch.arange(config.grid, dtype=torch.float32)
        base = torch.outer(steps, freqs).repeat_interleave(2, dim=-1)
        rows = base[:, None, :].expand(config.grid, config.grid, 2 * quarter)
        cols = base[None, :, :].expand(config.grid, config.grid, 2 * quarter)
        angles = torch.cat([rows, cols], dim=-1).reshape(-1, config.channels_head)
        self.cos = nn.Buffer(angles.cos())
        self.sin = nn.Buffer(angles.sin())

    @classmethod
    def rotate_pairs(cls, x: Tensor) -> Tensor:
        """Rotate adjacent channel pairs by a quarter turn.

        Args:
          x: Activations whose last dim is even.

        Returns:
          rotated: ``(-x1, x0)`` interleaved back into the last dimension.

        """
        pairs = x.unflatten(-1, (-1, 2))
        first, second = pairs.unbind(dim=-1)
        return torch.stack((-second, first), dim=-1).flatten(-2)

    @override
    def forward(self, x: Tensor, positions: Tensor, **kwargs: object) -> Tensor:
        """Rotate the trailing tokens of ``x`` by their grid positions.

        Tokens beyond the ``positions`` length are LEADING and are left
        unrotated -- that is what keeps the class token out of the spatial
        geometry when the full sequence is passed.

        Args:
          x: Queries or keys, ``[batch, heads, tokens, channels_head]``.
          positions: Flattened grid indices, ``[batch, rotated]``.
          **kwargs: The open bus, unread here.

        Returns:
          rotated: Same shape as ``x``.

        """
        del kwargs
        lead = x.shape[-2] - positions.shape[-1]
        if lead < 0:
            raise ValueError(
                f"positions describe {positions.shape[-1]} tokens but the "
                f"sequence holds {x.shape[-2]}.",
            )
        tail = x[:, :, lead:, :]
        cos = self.cos.to(dtype=x.dtype, device=x.device)[positions].unsqueeze(1)
        sin = self.sin.to(dtype=x.dtype, device=x.device)[positions].unsqueeze(1)
        rotated = tail * cos + self.rotate_pairs(tail) * sin
        if lead == 0:
            return rotated
        return torch.cat([x[:, :, :lead, :], rotated], dim=-2)


class SelfAttention(nn.Module):
    """Bidirectional multi-head self-attention with rotary positions."""

    class Config(Fig["SelfAttention"], kw_only=False):
        """Configuration for SelfAttention."""

        channels_in: int = -1
        """Residual width entering and leaving attention."""

        _: KW_ONLY

        heads: int = 12
        """Attention heads."""

        channels_head: int = -1
        """Width of one head; ``-1`` derives it from ``channels_in``."""

        bias: bool = True
        """Whether the fused projection carries a bias."""

        dropout: float = 0.0
        """Attention dropout probability, applied only while training."""

        norm_qk: Makeable[TensorModule] | None = field(
            default_factory=lambda: RMSNorm.Config(eps=None, elementwise_affine=True),
        )
        """Per-head query and key normalization; ``None`` leaves them raw.

        Two independent modules are built, matching the reference, so a norm
        carrying parameters learns a separate scale for queries and keys.
        ``eps=None`` is not "no epsilon": torch substitutes the dtype's own,
        which is what the reference's bare ``nn.RMSNorm(head_dim)`` uses."""

        attn_kernel: Makeable[AttentionKernel] = field(default_factory=SdpaFused.Config)
        """The attention kernel, over priml's ``[..., tokens, heads, channels]``."""

        value_residual: Makeable[ValueBlend] | None = None
        """Blend toward the first layer's values; ``None`` disables it."""

        @override
        def finalize(self) -> Self:
            if self.channels_head == -1:
                if self.channels_in % self.heads:
                    raise ValueError(
                        f"channels_in {self.channels_in} must be divisible by "
                        f"heads {self.heads}.",
                    )
                self.channels_head = self.channels_in // self.heads
            propagate_attr(
                self.norm_qk,
                "channels_in",
                self.channels_head,
                protocol=ChannelsIn,
            )
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one attention block.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            rows = seq_len * batch_size
            bus = {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "heads": self.heads,
                # The shared kernel's cost reads priml's kernel spelling.
                "num_heads": self.heads,
                "channels_head": self.channels_head,
                "dtype": dtype,
                **kwargs,
            }
            total = (
                matmul_cost(
                    channels_in=self.channels_in,
                    channels_out=3 * self.heads * self.channels_head,
                    bias=self.bias,
                    rows=rows,
                    dtype=dtype,
                )
                + matmul_cost(
                    channels_in=self.heads * self.channels_head,
                    channels_out=self.channels_in,
                    bias=True,
                    rows=rows,
                    dtype=dtype,
                )
                + cost(self.attn_kernel, **bus)
            )
            for part in (self.norm_qk, self.norm_qk, self.value_residual):
                if part is not None:
                    total = total + cost(part, **bus)
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.heads = config.heads
        self.channels_head = config.channels_head
        self.dropout = config.dropout
        inner = config.heads * config.channels_head
        # Registered FIRST: the reference's blend weight is the attention
        # module's own parameter, which ``named_parameters`` yields before any
        # child's, and the gradient clip reduces its norm in that order. It
        # draws no randomness, so moving it shifts no initialization.
        self.value_residual = (
            None if config.value_residual is None else config.value_residual.make()
        )
        self.qkv = nn.Linear(config.channels_in, inner * 3, bias=config.bias)
        self.q_norm = nn.Identity() if config.norm_qk is None else config.norm_qk.make()
        self.k_norm = nn.Identity() if config.norm_qk is None else config.norm_qk.make()
        self.proj = nn.Linear(inner, config.channels_in, bias=True)
        self.attn_kernel = config.attn_kernel.make()

    @override
    def forward(
        self,
        x: Tensor,
        *,
        rope: VisionRoPE | None = None,
        positions: Tensor | None = None,
        first_values: Tensor | None = None,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        """Attend over the sequence.

        ``rope`` arrives as an argument rather than a child because one table
        serves every block; owning a copy per block would put the same
        constant in the checkpoint once per layer.

        Args:
          x: Activations, ``[batch, tokens, channels_in]``.
          rope: Shared rotary table, or ``None`` to skip rotation.
          positions: Flattened grid indices for the rotated tail.
          first_values: First layer's values for the residual blend.
          **kwargs: The open bus, unread here.

        Returns:
          attended: ``[batch, tokens, channels_in]``.
          values: This layer's raw values, before any norm or blend.

        """
        del kwargs
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            tokens,
            3,
            self.heads,
            self.channels_head,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Captured before the norms and before the blend, so every consumer
        # receives the same tensor the first layer produced.
        raw_values = v
        q, k = self.q_norm(q), self.k_norm(k)
        if self.value_residual is not None and first_values is not None:
            v = self.value_residual(v, first_values)
        if rope is not None and positions is not None:
            q = rope(q, positions)
            k = rope(k, positions)
        # Stride views both ways: the kernel's own transpose undoes these, so
        # SDPA receives exactly the head-major tensors built above.
        attended = self.attn_kernel(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.reshape(batch, tokens, channels)
        return self.proj(attended), raw_values


class DiTBlock(nn.Module):
    """Transformer block gated by adaLN-Zero conditioning."""

    class Config(Fig["DiTBlock"], kw_only=False):
        """Configuration for DiTBlock."""

        channels_in: int = -1
        """Residual width."""

        _: KW_ONLY

        cond_dim: int = -1
        """Conditioning width feeding the modulation."""

        attn: SelfAttention.Config = field(default_factory=SelfAttention.Config)
        """Self-attention sublayer."""

        ffn: FeedForward.Config = field(default_factory=FeedForward.Config)
        """Feed-forward sublayer."""

        norm1: Makeable[TensorModule] = field(
            default_factory=lambda: RMSNorm.Config(eps=1e-6, elementwise_affine=False),
        )
        """Normalization before attention."""

        norm2: Makeable[TensorModule] = field(
            default_factory=lambda: RMSNorm.Config(eps=1e-6, elementwise_affine=False),
        )
        """Normalization before the feed-forward."""

        modulation: AdaLNModulation.Config = field(
            default_factory=AdaLNModulation.Config,
        )
        """Conditioning projection producing six modulation groups."""

        @override
        def finalize(self) -> Self:
            self.attn.channels_in = self.channels_in
            self.ffn.channels_in = self.channels_in
            self.modulation.channels_in = self.channels_in
            self.modulation.cond_dim = self.cond_dim
            for norm in (self.norm1, self.norm2):
                propagate_attr(
                    norm,
                    "channels_in",
                    self.channels_in,
                    protocol=ChannelsIn,
                )
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one block.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            bus = {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "dtype": dtype,
                **kwargs,
            }
            rows = seq_len * batch_size
            # Two gated residual adds and two scale-shifts.
            glue = elementwise_cost(
                primal=6,
                adjoint=6,
                channels=self.channels_in,
                rows=rows,
                dtype=dtype,
                inputs=2,
            )
            return (
                cost(self.attn, **bus)
                + cost(self.ffn, **bus)
                + cost(self.norm1, **bus)
                + cost(self.norm2, **bus)
                + cost(self.modulation, **bus)
                + glue
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm1 = config.norm1.make()
        self.attn = config.attn.make()
        self.norm2 = config.norm2.make()
        self.ffn = config.ffn.make()
        self.modulation = config.modulation.make()

    @override
    def forward(
        self,
        x: Tensor,
        c: Tensor,
        *,
        rope: VisionRoPE | None = None,
        positions: Tensor | None = None,
        first_values: Tensor | None = None,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        """Apply one modulated block.

        Args:
          x: Activations, ``[batch, tokens, channels_in]``.
          c: Conditioning, ``[batch, cond_dim]``.
          rope: Shared rotary table, forwarded to attention.
          positions: Flattened grid indices for the rotated tail.
          first_values: First layer's values for the residual blend.
          **kwargs: The open bus, unread here.

        Returns:
          activations: Same shape as ``x``.
          values: This layer's raw attention values.

        """
        del kwargs
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            self.modulation(c)
        )
        attended, values = self.attn(
            modulate(self.norm1(x), shift_attn, scale_attn),
            rope=rope,
            positions=positions,
            first_values=first_values,
        )
        x = x + gate_attn.unsqueeze(1) * attended
        x = x + gate_ffn.unsqueeze(1) * self.ffn(
            modulate(self.norm2(x), shift_ffn, scale_ffn),
        )
        return x, values


class FinalLayer(nn.Module):
    """Modulated readout to patch values and a class-token velocity."""

    class Config(Fig["FinalLayer"], kw_only=False):
        """Configuration for FinalLayer."""

        channels_in: int = -1
        """Width entering the readout."""

        channels_out: int = -1
        """Values emitted per patch token."""

        _: KW_ONLY

        cond_dim: int = -1
        """Conditioning width feeding the modulation."""

        channels_cls: int = -1
        """Width of the class-token readout."""

        norm: Makeable[TensorModule] = field(
            default_factory=lambda: RMSNorm.Config(eps=1e-6, elementwise_affine=False),
        )
        """Normalization before the readout."""

        modulation: AdaLNModulation.Config = field(
            default_factory=lambda: AdaLNModulation.Config(num_groups=2),
        )
        """Conditioning projection producing shift and scale."""

        @override
        def finalize(self) -> Self:
            self.modulation.channels_in = self.channels_in
            self.modulation.cond_dim = self.cond_dim
            propagate_attr(
                self.norm,
                "channels_in",
                self.channels_in,
                protocol=ChannelsIn,
            )
            return super().finalize()

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one readout.

            Args:
              seq_len: Tokens per sequence, including the class token.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            bus = {
                "seq_len": seq_len,
                "batch_size": batch_size,
                "dtype": dtype,
                **kwargs,
            }
            return (
                matmul_cost(
                    channels_in=self.channels_in,
                    channels_out=self.channels_out,
                    bias=True,
                    rows=batch_size * (seq_len - 1),
                    dtype=dtype,
                )
                + matmul_cost(
                    channels_in=self.channels_in,
                    channels_out=self.channels_cls,
                    bias=True,
                    rows=batch_size,
                    dtype=dtype,
                )
                + cost(self.norm, **bus)
                + cost(self.modulation, **bus)
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_final = config.norm.make()
        self.linear = nn.Linear(config.channels_in, config.channels_out, bias=True)
        self.linear_cls = nn.Linear(config.channels_in, config.channels_cls, bias=True)
        self.modulation = config.modulation.make()

    @override
    def forward(self, x: Tensor, c: Tensor, **kwargs: object) -> tuple[Tensor, Tensor]:
        """Read out patch values and the class-token velocity.

        The modulation covers the whole sequence including the class slot, and
        the split happens after it, so the class token sees the same scale and
        shift the patches do.

        Args:
          x: Activations, ``[batch, 1 + patches, channels_in]``.
          c: Conditioning, ``[batch, cond_dim]``.
          **kwargs: The open bus, unread here.

        Returns:
          patches: ``[batch, patches, channels_out]``.
          cls: ``[batch, channels_cls]``.

        """
        del kwargs
        shift, scale = self.modulation(c)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x[:, 1:]), self.linear_cls(x[:, 0])


class SprintRouting(nn.Module):
    """Sparse-dense residual fusion: run the trunk on a token subset.

    Encoder and decoder stages see every token; the middle stage sees a random
    subset, and the dropped slots are refilled with a learned mask token before
    the two paths are concatenated and projected back to the residual width.

    References:
      https://arxiv.org/abs/2512.12386
        Bhanded 2025, "Speedrunning ImageNet Diffusion."

    """

    class Result(NamedTuple):
        """A routing plan and the tensors it produced."""

        tokens: Tensor
        """Tokens entering the middle stage."""

        keep: Tensor | None
        """Indices kept, ``[batch, kept]``; ``None`` when nothing was dropped."""

    class Config(Fig["SprintRouting"], kw_only=False):
        """Configuration for SprintRouting."""

        channels_in: int = -1
        """Residual width."""

        _: KW_ONLY

        drop_ratio: float = 0.75
        """Fraction of tokens withheld from the middle stage while training."""

        path_drop_prob: float = 0.05
        """Probability of discarding the sparse path entirely for a step."""

        num_encoder_layers: int = 2
        """Dense layers before the sparse stage."""

        num_decoder_layers: int = 2
        """Dense layers after the fusion."""

        @override
        def finalize(self) -> Self:
            if (
                math.isnan(self.drop_ratio)
                or self.drop_ratio < 0.0
                or self.drop_ratio >= 1.0
            ):
                raise ValueError(
                    f"drop_ratio must be in [0, 1); got {self.drop_ratio}.",
                )
            if (
                math.isnan(self.path_drop_prob)
                or self.path_drop_prob < 0.0
                or self.path_drop_prob > 1.0
            ):
                raise ValueError(
                    f"path_drop_prob must be in [0, 1]; got {self.path_drop_prob}.",
                )
            return super().finalize()

        def kept(self, seq_len: int) -> int:
            """Tokens surviving the drop for a given sequence length.

            Args:
              seq_len: Tokens per sequence.

            Returns:
              kept: Token count entering the middle stage.

            """
            if self.drop_ratio <= 0.0 or seq_len <= 1:
                return seq_len
            return max(1, min(seq_len, int(seq_len * (1.0 - self.drop_ratio))))

        def cost(
            self,
            *,
            seq_len: int,
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost the fusion projection.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, unread here.

            Returns:
              cost: Whole-invocation cost of this module.

            """
            del kwargs
            # Plus the mask token, which this module owns and every refill reads.
            return matmul_cost(
                channels_in=2 * self.channels_in,
                channels_out=self.channels_in,
                bias=True,
                rows=seq_len * batch_size,
                dtype=dtype,
            ) + Cost(params=self.channels_in, params_active=self.channels_in)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.drop_ratio = config.drop_ratio
        self.path_drop_prob = config.path_drop_prob
        self.num_encoder_layers = config.num_encoder_layers
        self.num_decoder_layers = config.num_decoder_layers
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.channels_in))
        self.fusion_proj = nn.Linear(2 * config.channels_in, config.channels_in, True)

    def plan(self, x: Tensor) -> Result:
        """Choose the token subset entering the middle stage.

        Args:
          x: Dense activations, ``[batch, tokens, channels_in]``.

        Returns:
          result: The sparse tokens and the indices kept.

        """
        batch, tokens, channels = x.shape
        if not self.training or self.drop_ratio <= 0.0 or tokens <= 1:
            return SprintRouting.Result(x, None)
        num_keep = max(1, int(tokens * (1.0 - self.drop_ratio)))
        if num_keep >= tokens:
            return SprintRouting.Result(x, None)
        noise = torch.rand(batch, tokens, device=x.device)
        keep = torch.argsort(noise, dim=1)[:, :num_keep]
        gathered = x.gather(1, keep.unsqueeze(-1).expand(-1, -1, channels))
        return SprintRouting.Result(gathered, keep)

    def scatter(self, sparse: Tensor, keep: Tensor | None, tokens: int) -> Tensor:
        """Refill dropped slots with the mask token.

        Args:
          sparse: Middle-stage output, ``[batch, kept, channels_in]``.
          keep: Indices kept, or ``None`` when nothing was dropped.
          tokens: Full sequence length.

        Returns:
          dense: ``[batch, tokens, channels_in]``.

        """
        if keep is None:
            return sparse
        batch, _, channels = sparse.shape
        dense = self.mask_token.expand(batch, tokens, channels).clone()
        return dense.scatter(1, keep.unsqueeze(-1).expand(-1, -1, channels), sparse)

    def drop_path(self, dense: Tensor, *, uncond: bool) -> Tensor:
        """Discard the sparse path, stochastically or on request.

        The multiply by zero is deliberate: it keeps the sparse trunk in the
        autograd graph so its parameters still receive a gradient of zero
        rather than dropping out of the step entirely.

        Args:
          dense: Refilled sparse-path output.
          uncond: Force the drop, as the unconditional branch does at sampling.

        Returns:
          dense: Either the input or the broadcast mask token.

        """
        drop = uncond
        if self.training and self.path_drop_prob > 0.0:
            decision = torch.rand(1, device=dense.device)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(decision, src=0)
            drop = bool(decision.item() < self.path_drop_prob)
        if not drop:
            return dense
        return dense * 0.0 + self.mask_token.expand_as(dense)

    @override
    def forward(self, dense: Tensor, sparse: Tensor, **kwargs: object) -> Tensor:
        """Fuse the dense and sparse paths.

        Args:
          dense: Encoder output, ``[batch, tokens, channels_in]``.
          sparse: Refilled middle-stage output, same shape.
          **kwargs: The open bus, unread here.

        Returns:
          fused: ``[batch, tokens, channels_in]``.

        """
        del kwargs
        return self.fusion_proj(torch.cat([dense, sparse], dim=-1))


class SpeedrunDiT(nn.Module):
    """Latent flow-matching transformer with a diffused class token.

    Example:
      cfg = SpeedrunDiT.Config()
      model = cfg.make()
      velocity, projections, cls_velocity = model(latents, time, label, cls)

    """

    class Output(NamedTuple):
        """One forward pass."""

        velocity: Tensor
        """Predicted latent velocity, ``[batch, channels_in, size, size]``."""

        projections: list[Tensor]
        """Per-target alignment projections of the encoder tokens."""

        cls_velocity: Tensor
        """Predicted class-token velocity, ``[batch, channels_cls]``."""

    class Config(Fig["SpeedrunDiT"], kw_only=False):
        """Configuration for SpeedrunDiT."""

        channels_in: int = 32
        """Latent channels; the readout emits the same count."""

        channels_hidden: int = 768
        """Residual width of the trunk."""

        _: KW_ONLY

        image_size: int = 16
        """Side length of the square latent grid."""

        patch_size: int = 1
        """Side length of one square patch."""

        num_layers: int = 12
        """Blocks in the trunk."""

        heads: int = 12
        """Attention heads per block."""

        num_classes: int = 1000
        """Real classes; the null class for guidance is appended beyond them."""

        projector_dims: tuple[int, ...] = (768,)
        """Width of each alignment target; one projector is built per entry."""

        projector_hidden: int = 2048
        """Inner width of each alignment projector."""

        block: DiTBlock.Config = field(default_factory=DiTBlock.Config)
        """Block template, copied once per layer by ``finalize``."""

        time_embedder: TimestepEmbedder.Config = field(
            default_factory=TimestepEmbedder.Config,
        )
        """Flow-time conditioning branch."""

        label_embedder: LabelEmbedder.Config = field(
            default_factory=LabelEmbedder.Config,
        )
        """Class conditioning branch."""

        patch_embedder: PatchEmbed.Config = field(default_factory=PatchEmbed.Config)
        """Latent-to-token projection."""

        readout: FinalLayer.Config = field(default_factory=FinalLayer.Config)
        """Modulated readout to patch values and the class velocity."""

        norm_cls: Makeable[TensorModule] = field(
            default_factory=lambda: RMSNorm.Config(eps=1e-6, elementwise_affine=True),
        )
        """Normalization of the projected class token before it joins the
        sequence."""

        rope: VisionRoPE.Config | None = field(default_factory=VisionRoPE.Config)
        """Rotary positions shared by every block; ``None`` disables them."""

        value_residual: ValueResidual.Config | None = field(
            default_factory=ValueResidual.Config,
        )
        """Value-residual template; the first block never receives one, having
        no earlier layer to blend toward."""

        sprint: SprintRouting.Config | None = field(
            default_factory=SprintRouting.Config,
        )
        """Sparse-dense routing; ``None`` runs every block densely."""

        @property
        def channels_cls(self) -> int:
            """Width of the diffused class token.

            Returns:
              channels: The first alignment target's width, which the class
                stream shares so one projector serves both.

            """
            return self.projector_dims[0]

        @property
        def num_patches(self) -> int:
            """Patch tokens produced by one latent.

            Returns:
              count: ``(image_size // patch_size) ** 2``.

            """
            return (self.image_size // self.patch_size) ** 2

        @property
        def seq_len(self) -> int:
            """Tokens entering the trunk, including the class token.

            Returns:
              length: ``1 + num_patches``.

            """
            return 1 + self.num_patches

        @override
        def finalize(self) -> Self:
            if not self.projector_dims:
                raise ValueError("projector_dims must name at least one target.")
            if self.channels_hidden % self.heads:
                raise ValueError(
                    f"channels_hidden {self.channels_hidden} must be divisible "
                    f"by heads {self.heads}.",
                )
            if self.sprint is not None:
                dense = self.sprint.num_encoder_layers + self.sprint.num_decoder_layers
                if dense > self.num_layers:
                    raise ValueError(
                        f"sprint reserves {dense} dense layers but the trunk "
                        f"holds {self.num_layers}.",
                    )
                self.sprint.channels_in = self.channels_hidden

            self.patch_embedder.channels_in = self.channels_in
            self.patch_embedder.channels_out = self.channels_hidden
            self.patch_embedder.image_size = self.image_size
            self.patch_embedder.patch_size = self.patch_size

            self.time_embedder.channels_out = self.channels_hidden
            self.label_embedder.channels_in = self.num_classes
            self.label_embedder.channels_out = self.channels_hidden

            if self.rope is not None:
                self.rope.channels_head = self.channels_hidden // self.heads
                self.rope.grid = self.image_size // self.patch_size

            self.block.channels_in = self.channels_hidden
            self.block.cond_dim = self.channels_hidden
            self.block.attn.heads = self.heads

            # The trunk hands the readout its own width: the reference exposes
            # a separate decoder width, but no projection sits between them and
            # every published size leaves the two equal.
            self.readout.channels_in = self.channels_hidden
            # A velocity lives in the space it moves, so the readout emits
            # ``channels_in`` per patch position.
            self.readout.channels_out = self.patch_size**2 * self.channels_in
            self.readout.cond_dim = self.channels_hidden
            self.readout.channels_cls = self.channels_cls
            propagate_attr(
                self.norm_cls,
                "channels_in",
                self.channels_hidden,
                protocol=ChannelsIn,
            )
            return super().finalize()

        def layers(self) -> list[DiTBlock.Config]:
            """Build the per-layer block configs.

            The first block carries no value residual: the blend targets the
            first layer's own values, which do not exist while it runs.

            Returns:
              blocks: One finalized-ready config per layer.

            """
            blocks: list[DiTBlock.Config] = []
            for index in range(self.num_layers):
                block = self.block.copy_tree()
                # Copied per slot, not shared: a single config assigned into
                # every block would let one block's finalize mutate the rest.
                block.attn.value_residual = (
                    None
                    if index == 0 or self.value_residual is None
                    else self.value_residual.copy_tree()
                )
                blocks.append(block)
            return blocks

        def cost(
            self,
            *,
            batch_size: int = 1,
            dtype: torch.dtype | None = None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward pass.

            The middle stage is priced at its sparse length, which is what
            SPRINT exists to buy; an eval pass runs it dense and costs more.

            Args:
              batch_size: Sequences per step.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus, forwarded to every child.

            Returns:
              cost: Whole-invocation cost of this model.

            """
            dense_len = self.seq_len
            bus = {"batch_size": batch_size, "dtype": dtype, **kwargs}
            blocks = self.layers()
            if self.sprint is None:
                lengths = [dense_len] * self.num_layers
            else:
                sparse_len = self.sprint.kept(dense_len)
                head = self.sprint.num_encoder_layers
                tail = self.num_layers - self.sprint.num_decoder_layers
                lengths = [
                    dense_len if index < head or index >= tail else sparse_len
                    for index in range(self.num_layers)
                ]
            total = (
                cost(self.patch_embedder, **bus)
                + cost(self.time_embedder, **bus)
                + cost(self.label_embedder, **bus)
                + cost(self.norm_cls, seq_len=1, **bus)
                + cost(self.readout, seq_len=dense_len, **bus)
                + matmul_cost(
                    channels_in=self.channels_cls,
                    channels_out=self.channels_hidden,
                    bias=True,
                    rows=batch_size,
                    dtype=dtype,
                )
            )
            for block, length in zip(blocks, lengths, strict=True):
                built = block.copy_tree().finalize()
                total = total + cost(built, seq_len=length, **bus)
                if self.rope is not None:
                    # One shared table, but every block pays to rotate.
                    total = total + cost(
                        self.rope,
                        seq_len=length,
                        heads=self.heads,
                        **bus,
                    )
            if self.sprint is not None:
                total = total + cost(self.sprint, seq_len=dense_len, **bus)
            for width in self.projector_dims:
                total = (
                    total
                    + matmul_cost(
                        channels_in=self.channels_hidden,
                        channels_out=self.projector_hidden,
                        bias=True,
                        rows=batch_size * dense_len,
                        dtype=dtype,
                    )
                    + matmul_cost(
                        channels_in=self.projector_hidden,
                        channels_out=self.projector_hidden,
                        bias=True,
                        rows=batch_size * dense_len,
                        dtype=dtype,
                    )
                    + matmul_cost(
                        channels_in=self.projector_hidden,
                        channels_out=width,
                        bias=True,
                        rows=batch_size * dense_len,
                        dtype=dtype,
                    )
                )
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.channels_out = config.channels_in
        self.num_patches = config.num_patches
        self.grid = config.image_size // config.patch_size
        self.projector_dims = list(config.projector_dims)

        # Construction order fixes the global-RNG draw order, so a seeded init
        # is reproducible: routing -> patches -> time -> labels -> blocks ->
        # projectors -> readout. ``priml.model.Linear`` cannot substitute for
        # ``nn.Linear`` anywhere below -- its ``init_bias`` is ``zeros_``,
        # which draws nothing where torch draws ``bias.numel()``, shifting
        # every later draw. ``RMSNorm`` can: unaffine it holds no parameter.
        self.sprint = None if config.sprint is None else config.sprint.make()
        self.x_embedder = config.patch_embedder.make()
        self.t_embedder = config.time_embedder.make()
        self.y_embedder = config.label_embedder.make()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.seq_len, config.channels_hidden),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [block.finalize().make() for block in config.layers()],
        )
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.channels_hidden, config.projector_hidden),
                    nn.SiLU(),
                    nn.Linear(config.projector_hidden, config.projector_hidden),
                    nn.SiLU(),
                    nn.Linear(config.projector_hidden, width),
                )
                for width in config.projector_dims
            ],
        )
        self.final_layer = config.readout.make()
        self.cls_projector = nn.Linear(
            config.channels_cls,
            config.channels_hidden,
            bias=True,
        )
        self.wg_norm = config.norm_cls.make()
        # One shared table, built last so it sits after every parameter-owning
        # submodule; it draws no randomness, so its position is free.
        self.rope = None if config.rope is None else config.rope.make()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every parameter in place.

        Order matters and is the reference's: a blanket Xavier pass over every
        linear layer, then the four tensors that want something else, then the
        zeroing that makes each gated sublayer an identity at step zero.
        """

        def basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic)
        table = sincos_position_table(
            self.pos_embed.shape[-1],
            self.grid,
            lead=self.pos_embed.shape[-2] - self.num_patches,
        )
        self.pos_embed.data.copy_(table.unsqueeze(0))
        # The patch projection is a convolution but initialized like the linear
        # layer it is equivalent to at stride == kernel, so its fan-in counts
        # the whole patch rather than one spatial position.
        patches = self.x_embedder.proj
        weight = patches.weight.data
        nn.init.xavier_uniform_(weight.view([weight.shape[0], -1]))
        if patches.bias is not None:
            nn.init.constant_(patches.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(_linear(self.t_embedder.mlp[0]).weight, std=0.02)
        nn.init.normal_(_linear(self.t_embedder.mlp[2]).weight, std=0.02)
        zeroed: list[nn.Linear] = []
        for block in self.blocks:
            assert isinstance(block, DiTBlock)
            zeroed.append(block.modulation.proj)
        final = self.final_layer
        zeroed += [final.modulation.proj, final.linear, final.linear_cls]
        for layer in zeroed:
            nn.init.constant_(layer.weight, 0)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def unpatchify(self, x: Tensor) -> Tensor:
        """Fold patch values back into a latent grid.

        Args:
          x: Patch values, ``[batch, patches, patch_size ** 2 * channels_out]``.

        Returns:
          latents: ``[batch, channels_out, size, size]``.

        """
        batch = x.shape[0]
        patch, channels, grid = self.patch_size, self.channels_out, self.grid
        x = x.reshape(batch, grid, grid, patch, patch, channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(batch, channels, grid * patch, grid * patch)

    def positions(self, batch: int, device: torch.device) -> Tensor:
        """Flattened grid indices for the patch tokens.

        Args:
          batch: Sequences per step.
          device: Device the indices are built on.

        Returns:
          positions: ``[batch, num_patches]``.

        """
        flat = torch.arange(self.num_patches, device=device, dtype=torch.long)
        return flat.view(1, -1).expand(batch, -1)

    @override
    def forward(
        self,
        media: Tensor,
        time: Tensor,
        label: Tensor,
        cls_token: Tensor,
        *,
        uncond: bool = False,
        **kwargs: object,
    ) -> Output:
        """Predict the latent and class-token velocities.

        Args:
          media: Noised latents, ``[batch, channels_in, size, size]``.
          time: Flow times, ``[batch]``.
          label: Class indices, ``[batch]``.
          cls_token: Noised class token, ``[batch, channels_cls]``.
          uncond: Discard the sparse path, as the unconditional branch does.
          **kwargs: The open bus, unread here.

        Returns:
          output: Velocities and the alignment projections.

        """
        del kwargs
        x = self.x_embedder(media)
        cls = self.wg_norm(self.cls_projector(cls_token)).unsqueeze(1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        batch, tokens, channels = x.shape
        positions = self.positions(batch, x.device)
        c = self.t_embedder(time) + self.y_embedder(label)

        if self.sprint is None:
            head, tail = self.config.num_layers, self.config.num_layers
        else:
            head = self.sprint.num_encoder_layers
            tail = self.config.num_layers - self.sprint.num_decoder_layers

        dense = x
        first_values: Tensor | None = None
        for block in self.blocks[:head]:
            dense, values = block(
                dense,
                c,
                rope=self.rope,
                positions=positions,
                first_values=first_values,
            )
            if first_values is None:
                first_values = values

        projections = [
            projector(dense.reshape(-1, channels)).reshape(batch, tokens, width)
            for projector, width in zip(
                self.projectors,
                self.projector_dims,
                strict=True,
            )
        ]

        if self.sprint is None:
            fused = dense
        else:
            plan = self.sprint.plan(dense)
            sparse_positions = positions
            sparse_values = first_values
            if plan.keep is not None:
                # The kept indices address the full sequence, class token
                # included, so the rotary ids and the first-layer values have
                # to be gathered with the very same plan or the sparse stage
                # attends with positions that belong to other tokens.
                sparse_positions = self._gather_positions(positions, plan.keep)
                sparse_values = self._gather_values(first_values, plan.keep)
            middle = plan.tokens
            for block in self.blocks[head:tail]:
                middle, _ = block(
                    middle,
                    c,
                    rope=self.rope,
                    positions=sparse_positions,
                    first_values=sparse_values,
                )
            refilled = self.sprint.scatter(middle, plan.keep, tokens)
            refilled = self.sprint.drop_path(refilled, uncond=uncond)
            fused = self.sprint(dense, refilled)

        decoded = fused
        for block in self.blocks[tail:]:
            decoded, _ = block(
                decoded,
                c,
                rope=self.rope,
                positions=positions,
                first_values=first_values,
            )

        patches, cls_velocity = self.final_layer(decoded, c)
        return SpeedrunDiT.Output(
            velocity=self.unpatchify(patches),
            projections=projections,
            cls_velocity=cls_velocity,
        )

    # The class token occupies slot zero and has no grid position, so the full-sequence
    # ids are the patch ids shifted by one with a leading zero; gathering that shifted
    # vector keeps every kept patch on its own position and leaves a surviving class
    # token at id zero, where the rotation is the identity.
    def _gather_positions(self, positions: Tensor, keep: Tensor) -> Tensor:
        """Select rotary ids for a kept token subset."""
        lead = positions.new_zeros(positions.shape[0], 1)
        return torch.cat([lead, positions], dim=1).gather(1, keep)

    @classmethod
    def _gather_values(cls, values: Tensor | None, keep: Tensor) -> Tensor | None:
        """Select first-layer values for a kept token subset."""
        if values is None:
            return None
        index = keep[:, None, :, None].expand(-1, values.size(1), -1, values.size(-1))
        return values.gather(2, index)
