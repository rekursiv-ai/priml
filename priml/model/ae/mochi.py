"""Mochi 1: an asymmetric causal video VAE, a small encoder and a large decoder.

Videos are ``1 + 6k`` frames. Every convolution is causal in time and pads by
replicating its edge; the encoder's stages attend causally along time, at each
position, and the decoder's last convolution of each stage unfolds channels
into frames and pixels. The encoder reads RGB with two octaves of Fourier
features, 15 channels.

Each half is published as its own float32 file, named as this module names
it. Mochi's own draw, ``noise * exp(logvar * 0.5) + mean``, is
:class:`~priml.model.ae.distribution.DiagonalGaussian`'s without the
log-variance clamp, which never binds on its moments.

Build the published model from :func:`mochi`, or :func:`mochi_comfy` for
Comfy's float16 repackaging of both halves in one file: each a
:class:`~priml.model.ae.video.CausalVideoVAE` around this module's halves.

Apache-2.0 license and attribution: ``LICENSES/mochi``.

References:
  https://github.com/genmoai/mochi/blob/15c9bad662df1fa3e7b53136423695b286484c08/src/genmo/mochi_preview/vae/models.py
  https://huggingface.co/genmo/mochi-1-preview
    Genmo Team 2024. Mochi 1.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, cast, override

import math

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, cost, map_cost, traffic
from priml.math.pixel import unpatchify
from priml.math.tensor import per_frame
from priml.model.ae.checkpoint import CheckpointFile, HubFile, read_state
from priml.model.ae.custom_types import StateDictRemap
from priml.model.ae.frozen_cost import POINTWISE, SUM
from priml.model.ae.latent_norm import StdLatents
from priml.model.ae.video import CausalConv3d, CausalVideoVAE
from priml.model.attention.kernel import SdpaFused
from priml.model.conv import Conv3d
from priml.model.custom_types import AttentionKernel
from priml.model.linear import Linear
from priml.model.norm import GroupNorm2d
from priml.model.swiglu import silu


if TYPE_CHECKING:
    from collections.abc import Mapping


MOCHI_REVISION: Final = "cd1d0fe89aa8382fcd9ae760bc3d7afe6183579f"
"""The commit of ``genmo/mochi-1-preview`` the defaults pin."""


def mochi_encoder() -> HubFile.Config:
    """Return Mochi's encoder as released, float32.

    Returns:
      source: The pinned Hub file.

    """
    return HubFile.Config(
        repo_id="genmo/mochi-1-preview",
        filename="encoder.safetensors",
        revision=MOCHI_REVISION,
        sha256="d3a8827a66b58a479d97420a9bf77e59078d88f538298469d8db28c37bd556ae",
    )


def mochi_decoder() -> HubFile.Config:
    """Return Mochi's decoder as released, float32.

    Returns:
      source: The pinned Hub file.

    """
    return HubFile.Config(
        repo_id="genmo/mochi-1-preview",
        filename="decoder.safetensors",
        revision=MOCHI_REVISION,
        sha256="641920faaf20e5404ddb5553ce3e295c21ed9b4bc5f6fe7c930811b84099cb14",
    )


def remap_mochi_state_dict(
    state: Mapping[str, Tensor],
    *,
    half: str,
) -> dict[str, Tensor]:
    """Rename one half's Mochi tensors to this module's names.

    A released half is its own file, already named as this module names it;
    Comfy's file holds both, each under its half's prefix.

    Args:
      state: One half's file, or both halves' prefixed.
      half: ``"encoder"`` or ``"decoder"``.

    Returns:
      state: This half's ``state_dict``.

    """
    prefix = f"{half}."
    if any(key.startswith(prefix) for key in state):
        return {
            key.removeprefix(prefix): t
            for key, t in state.items()
            if key.startswith(prefix)
        }
    return dict(state)


class Encoder(nn.Module):
    """Pixels in ``[-1, 1]`` to posterior moments, causal in time."""

    class Config(Fig["Encoder"]):
        """Widths, depth, attention, resampling, and weights."""

        channels_in: int = 3
        """Image channels; the first layer reads them with their Fourier features."""

        fourier_octaves: tuple[int, ...] = (6, 7)
        """Base-2 exponents of the Fourier features' frequencies, in turns."""

        channels_hidden: tuple[int, ...] = (64, 128, 256, 384)
        """Width per stage, finest first; each stage after the first downsamples."""

        blocks_per_stage: tuple[int, ...] = (3, 3, 4, 6, 3)
        """Residual blocks per group: the first stage, each downsampling, the last again."""

        attention: tuple[bool, ...] = (False, True, True, True, True)
        """Whether each group's blocks attend along time, as ``blocks_per_stage``."""

        temporal: tuple[int, ...] = (1, 2, 3)
        """Time each downsampling divides by."""

        spatial: tuple[int, ...] = (2, 2, 2)
        """Each side each downsampling divides by."""

        channels_latent: int = 12
        """Latent channels; the encoder emits twice as many, mean then log-variance."""

        channels_head: int = 32
        """Width of each attention head."""

        num_groups: int = 32
        """GroupNorm groups; must divide every stage's width."""

        attn_kernel: Makeable[AttentionKernel] = field(default_factory=SdpaFused.Config)
        """The causal attention along time."""

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=mochi_encoder,
        )
        """Weights; ``None`` keeps the random initialization."""

        remap: StateDictRemap = remap_mochi_state_dict
        """Renames ``checkpoint``'s tensors to this module's."""

        def factors(self) -> tuple[int, int]:
            """Return the ``(time, space)`` downsampling from pixels to latent."""
            return math.prod(self.temporal), math.prod(self.spatial)

        def cost(
            self,
            *,
            input_grid: tuple[int, int, int],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward on a ``(frames, height, width)`` clip.

            Args:
              input_grid: Frames, height, width of the pixels.
              batch_size: Clips in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; nothing here reads it.

            Returns:
              cost: Forward FLOPs, logical bytes, and parameter ownership.

            """
            del kwargs
            layers = _Layers(
                batch_size=batch_size,
                dtype=dtype,
                num_groups=self.num_groups,
            )
            widths, grid = self.channels_hidden, input_grid
            fourier = self.channels_in * (1 + 2 * len(self.fourier_octaves))
            total = layers.fourier(
                self.channels_in,
                len(self.fourier_octaves),
                grid=grid,
            )
            total += layers.linear(
                fourier,
                widths[0],
                grid=grid,
                bias=True,
                channels_last=False,
            )

            def group(index: int, width: int, grid: tuple[int, int, int]) -> Cost:
                count = self.blocks_per_stage[index]
                blocks = layers.blocks(width, count, grid=grid)
                if not self.attention[index]:
                    return blocks
                return blocks + layers.attention(
                    width,
                    count,
                    grid=grid,
                    channels_head=self.channels_head,
                    attn_kernel=self.attn_kernel,
                )

            total += group(0, widths[0], grid)
            for stage, (t, s) in enumerate(
                zip(self.temporal, self.spatial, strict=True),
            ):
                down, grid = layers.down(
                    widths[stage],
                    widths[stage + 1],
                    t,
                    s,
                    grid=grid,
                )
                total += down + group(stage + 1, widths[stage + 1], grid)
            width = widths[-1]
            return (
                total
                + group(-1, width, grid)
                + layers.norm(width, grid=grid)
                + cost(silu, channels=batch_size * width * math.prod(grid), dtype=dtype)
                + layers.linear(
                    width,
                    2 * self.channels_latent,
                    grid=grid,
                    bias=False,
                    channels_last=True,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        widths, groups = config.channels_hidden, config.num_groups
        stages = len(widths)
        if not (
            len(config.temporal) == len(config.spatial) == stages - 1
            and len(config.blocks_per_stage) == len(config.attention) == stages + 1
        ):
            raise ValueError(
                f"{stages} stages need {stages - 1} temporal and spatial factors and "
                f"{stages + 1} blocks_per_stage and attention entries.",
            )
        self.octaves = config.fourier_octaves
        fourier = config.channels_in * (1 + 2 * len(config.fourier_octaves))

        def blocks(width: int, group: int) -> list[nn.Module]:
            return [
                _ResBlock(
                    width,
                    num_groups=groups,
                    attn_block=_AttentionBlock(
                        width,
                        num_groups=groups,
                        channels_head=config.channels_head,
                        kernel=config.attn_kernel,
                    )
                    if config.attention[group]
                    else None,
                )
                for _ in range(config.blocks_per_stage[group])
            ]

        layers: list[nn.Module] = [_Conv1x1(fourier, widths[0]), *blocks(widths[0], 0)]
        for stage, (t, s) in enumerate(
            zip(config.temporal, config.spatial, strict=True),
        ):
            down = CausalConv3d.Config(
                widths[stage],
                widths[stage + 1],
                kernel_size=(t, s, s),
                stride=(t, s, s),
                past=t - 1,
                padding_mode="replicate",
            ).make()
            layers.append(_Downsample(down, *blocks(widths[stage + 1], stage + 1)))
        layers.extend(blocks(widths[-1], stages))
        # The release's names: ``layers``, ``output_norm``, ``output_proj``.
        self.layers = nn.Sequential(*layers)
        self.output_norm = _FrameGroupNorm(widths[-1], num_groups=groups)
        self.output_proj = _Conv1x1(widths[-1], 2 * config.channels_latent, bias=False)
        if config.checkpoint is not None:
            prefixes = ("encoder.", "layers.", "output_norm.", "output_proj.")
            state = read_state(config.checkpoint.make().path(), prefixes=prefixes)
            self.load_state_dict(config.remap(state, half="encoder"))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Map ``[N, C_in, T, H, W]`` pixels in ``[-1, 1]`` to ``[N, 2 C, t, h, w]`` moments."""
        h = self.layers(_fourier_features(x, self.octaves))
        h = functional.silu(self.output_norm(h))
        # Channels last, as the stages' attention leaves them: the layout decides
        # how torch runs the projection, and so its bits (``_Conv1x1``).
        return self.output_proj(h.contiguous(memory_format=torch.channels_last_3d))


class Decoder(nn.Module):
    """Latents to pixels in ``[-1, 1]``, causal in time."""

    class Config(Fig["Decoder"]):
        """Widths, depth, resampling, and weights; stages in the order they run."""

        channels_out: int = 3
        """Image channels."""

        channels_hidden: tuple[int, ...] = (768, 512, 256, 128)
        """Width per stage, coarsest first; the input convolution writes the first."""

        blocks_per_stage: tuple[int, ...] = (3, 6, 4, 3, 3)
        """Residual blocks per group: the first stage, each upsampling, the last again."""

        temporal: tuple[int, ...] = (3, 2, 1)
        """Time each upsampling multiplies by, less its first frame's copies."""

        spatial: tuple[int, ...] = (2, 2, 2)
        """Each side each upsampling multiplies by."""

        channels_latent: int = 12
        """Latent channels."""

        num_groups: int = 32
        """GroupNorm groups; must divide every stage's width."""

        checkpoint: Makeable[CheckpointFile] | None = field(
            default_factory=mochi_decoder,
        )
        """Weights; ``None`` keeps the random initialization."""

        remap: StateDictRemap = remap_mochi_state_dict
        """Renames ``checkpoint``'s tensors to this module's."""

        def factors(self) -> tuple[int, int]:
            """Return the ``(time, space)`` downsampling from pixels to latent."""
            return math.prod(self.temporal), math.prod(self.spatial)

        def cost(
            self,
            *,
            input_grid: tuple[int, int, int],
            batch_size: int,
            dtype: torch.dtype | None,
            **kwargs: object,
        ) -> Cost:
            """Cost one forward on a ``(frames, height, width)`` latent.

            Args:
              input_grid: Frames, height, width of the latent.
              batch_size: Latents in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              **kwargs: The open bus; nothing here reads it.

            Returns:
              cost: Forward FLOPs, logical bytes, and parameter ownership.

            """
            del kwargs
            layers = _Layers(
                batch_size=batch_size,
                dtype=dtype,
                num_groups=self.num_groups,
            )
            widths, grid = self.channels_hidden, input_grid
            total = cost(
                Conv3d.Config(
                    self.channels_latent,
                    widths[0],
                    kernel_size=1,
                    padding=0,
                    bias=True,
                ),
                input_grid=grid,
                batch_size=batch_size,
                dtype=dtype,
            ) + layers.blocks(widths[0], self.blocks_per_stage[0], grid=grid)
            for stage, (t, s) in enumerate(
                zip(self.temporal, self.spatial, strict=True),
            ):
                up, grid = layers.up(
                    widths[stage],
                    widths[stage + 1],
                    self.blocks_per_stage[stage + 1],
                    t,
                    s,
                    grid=grid,
                )
                total += up
            width = widths[-1]
            return (
                total
                + layers.blocks(width, self.blocks_per_stage[-1], grid=grid)
                + cost(silu, channels=batch_size * width * math.prod(grid), dtype=dtype)
                + layers.linear(
                    width,
                    self.channels_out,
                    grid=grid,
                    bias=True,
                    channels_last=False,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        widths, groups = config.channels_hidden, config.num_groups
        stages = len(widths)
        if not (
            len(config.temporal) == len(config.spatial) == stages - 1
            and len(config.blocks_per_stage) == stages + 1
        ):
            raise ValueError(
                f"{stages} stages need {stages - 1} temporal and spatial factors and "
                f"{stages + 1} blocks_per_stage entries.",
            )

        def blocks(width: int, group: int) -> list[nn.Module]:
            return [
                _ResBlock(width, num_groups=groups)
                for _ in range(config.blocks_per_stage[group])
            ]

        first = Conv3d.Config(
            config.channels_latent,
            widths[0],
            kernel_size=1,
            padding=0,
            bias=True,
        ).make()
        stack: list[nn.Module] = [nn.Sequential(first, *blocks(widths[0], 0))]
        for stage, (t, s) in enumerate(
            zip(config.temporal, config.spatial, strict=True),
        ):
            stack.append(
                _Upsample(
                    nn.Sequential(*blocks(widths[stage], stage + 1)),
                    widths[stage],
                    widths[stage + 1],
                    factor_t=t,
                    factor_s=s,
                ),
            )
        stack.append(nn.Sequential(*blocks(widths[-1], stages)))
        # The release's names: ``blocks``, ``output_proj``.
        self.blocks = nn.ModuleList(stack)
        self.output_proj = _Conv1x1(widths[-1], config.channels_out)
        if config.checkpoint is not None:
            prefixes = ("decoder.", "blocks.", "output_proj.")
            state = read_state(config.checkpoint.make().path(), prefixes=prefixes)
            self.load_state_dict(config.remap(state, half="decoder"))

    @override
    def forward(self, z: Tensor) -> Tensor:
        """Map ``[N, C, t, h, w]`` latents to ``[N, C_out, T, H, W]`` pixels in ``[-1, 1]``."""
        # Contiguous whatever produced it, as for ``Encoder.output_proj``; a
        # convolution keeps or drops channels-last by its thread count.
        x = z.contiguous()
        for block in self.blocks:
            x = cast(Tensor, block(x))
        return self.output_proj(functional.silu(x))


_MOCHI_MEAN: Final = [
    -0.06730895953510081, -0.038011381506090416, -0.07477820912866141,
    -0.05565264470995561, 0.012767231469026969, -0.04703542746246419,
    0.043896967884726704, -0.09346305707025976, -0.09918314763016893,
    -0.008729793427399178, -0.011931556316503654, -0.0321993391887285,
]  # fmt: skip
_MOCHI_STD: Final = [
    0.9263795028493863, 0.9248894543193766, 0.9393059390890617,
    0.959253732819592, 0.8244560132752793, 0.917259975397747,
    0.9294154431013696, 1.3720942357788521, 0.881393668867029,
    0.9168315692124348, 0.9185249279345552, 0.9274757570805041,
]  # fmt: skip


def mochi() -> CausalVideoVAE.Config:
    """Return Mochi 1's VAE: 12 channels, 6x in time, 8x in space.

    Returns:
      config: The published model, float32 weights pinned per half.

    """
    return CausalVideoVAE.Config(
        encoder=Encoder.Config(),
        decoder=Decoder.Config(),
        num_frames=163,
        image_size=(480, 848),
        latent_norm=StdLatents.Config(mean=list(_MOCHI_MEAN), std=list(_MOCHI_STD)),
    )


def mochi_comfy() -> CausalVideoVAE.Config:
    """Return :func:`mochi` read from Comfy's repackaging, rounded to float16.

    Returns:
      config: The model, its weights pinned.

    """
    config = mochi()
    assert isinstance(config.encoder, Encoder.Config)
    assert isinstance(config.decoder, Decoder.Config)
    checkpoint = HubFile.Config(
        repo_id="Comfy-Org/mochi_preview_repackaged",
        filename="split_files/vae/mochi_vae.safetensors",
        revision="ed4b8585dc4d3204397291c35e100884686178d6",
        sha256="1be451cec94b911980406169286babc5269e7cf6a94bbbbdf45e8d3f2c961083",
    )
    config.encoder.checkpoint = checkpoint
    config.decoder.checkpoint = checkpoint.copy_tree()
    return config


def _fourier_features(x: Tensor, octaves: tuple[int, ...]) -> Tensor:
    """Append ``sin`` and ``cos`` of each channel at each octave, as the reference does."""
    frequencies = torch.tensor(octaves, dtype=x.dtype, device=x.device)
    w = torch.pow(2.0, frequencies) * (2 * torch.pi)
    w = w.repeat(x.shape[1])[None, :, None, None, None]
    h = w * x.repeat_interleave(len(octaves), dim=1)
    return torch.cat([x, torch.sin(h), torch.cos(h)], dim=1)


class _Conv1x1(Linear):
    """A linear map of the channel axis, ``[N, C, ...]``, as the reference's ``Conv1x1``."""

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__(Linear.Config(channels_in, channels_out, bias=bias))

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        # The reference's view: torch runs it as one matmul when the producer left
        # the channels last, else as a ``bmm`` per slice; the two round apart.
        return super().forward(input.movedim(1, -1)).movedim(-1, 1)


class _FrameGroupNorm(GroupNorm2d):
    """An affine GroupNorm over each frame of ``[N, C, T, H, W]``."""

    def __init__(self, channels: int, *, num_groups: int) -> None:
        super().__init__(
            GroupNorm2d.Config(
                channels,
                num_groups=num_groups,
                elementwise_affine=True,
            ),
        )

    @override
    def forward(self, input: Tensor, **kwargs: object) -> Tensor:
        del kwargs
        return per_frame(super().forward, input)


class _Attention(nn.Module):
    """Causal multi-head attention along time at each position, queries and keys L2-normed."""

    def __init__(
        self,
        channels: int,
        *,
        channels_head: int,
        kernel: Makeable[AttentionKernel],
    ) -> None:
        super().__init__()
        self.channels_head = channels_head
        self.qkv = Linear.Config(channels, 3 * channels, bias=False).make()
        self.out = Linear.Config(channels, channels, bias=True).make()
        self.kernel = kernel.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        rows = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        heads = c // self.channels_head
        q, k, v = self.qkv(rows).unflatten(2, (3, heads, self.channels_head)).unbind(2)
        q, k = (
            functional.normalize(q, p=2, dim=-1),
            functional.normalize(k, p=2, dim=-1),
        )
        attended = self.kernel(q, k, v, is_causal=True, scale=self.channels_head**-0.5)
        out = self.out(attended.flatten(2))
        return out.reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2)


class _AttentionBlock(nn.Module):
    """GroupNorm, attention along time, added to the input."""

    def __init__(
        self,
        channels: int,
        *,
        num_groups: int,
        channels_head: int,
        kernel: Makeable[AttentionKernel],
    ) -> None:
        super().__init__()
        self.norm = _FrameGroupNorm(channels, num_groups=num_groups)
        self.attn = _Attention(channels, channels_head=channels_head, kernel=kernel)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return x + self.attn(self.norm(x))


class _ResBlock(nn.Module):
    """Two GroupNorm, SiLU, causal 3x3x3 layers added to the input, then attention."""

    def __init__(
        self,
        channels: int,
        *,
        num_groups: int,
        attn_block: _AttentionBlock | None = None,
    ) -> None:
        super().__init__()
        self.stack = nn.Sequential(
            _FrameGroupNorm(channels, num_groups=num_groups),
            nn.SiLU(),
            CausalConv3d.Config(channels, channels, padding_mode="replicate").make(),
            _FrameGroupNorm(channels, num_groups=num_groups),
            nn.SiLU(),
            CausalConv3d.Config(channels, channels, padding_mode="replicate").make(),
        )
        self.attn_block = nn.Identity() if attn_block is None else attn_block

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.attn_block(self.stack(x) + x)


class _Downsample(nn.Module):
    """A strided causal convolution, then blocks; named as the release's ``layers``."""

    def __init__(self, *modules: nn.Module) -> None:
        super().__init__()
        self.layers = nn.Sequential(*modules)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class _Upsample(nn.Module):
    """Blocks, a 1x1 projection, then channels unfolded into frames and pixels."""

    def __init__(
        self,
        blocks: nn.Sequential,
        channels_in: int,
        channels_out: int,
        *,
        factor_t: int,
        factor_s: int,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.proj = _Conv1x1(channels_in, channels_out * factor_t * factor_s * factor_s)
        self.patch = (factor_t, factor_s, factor_s)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = unpatchify(self.proj(self.blocks(x)), self.patch)
        # The first frame is not repeated: only one of its copies stays.
        return x[:, :, self.patch[0] - 1 :]


_SIN_COS: Final = map_cost(
    primal=4,
    adjoint=0,
    outputs=2,
    adjoint_inputs=0,
    adjoint_outputs=0,
)
"""A product with the frequency, then its sine and cosine."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _Layers:
    """Price the reference's layers at one batch size, group count, and dtype."""

    batch_size: int
    dtype: torch.dtype | None
    num_groups: int

    def linear(
        self,
        channels_in: int,
        channels_out: int,
        *,
        grid: tuple[int, int, int],
        bias: bool,
        channels_last: bool,
    ) -> Cost:
        """Cost ``_Conv1x1`` on every position of ``grid``.

        With the channels last in memory the moved view is one matrix and the
        product one matmul; otherwise torch runs a ``bmm`` per ``(batch,
        frame, row)``, reading the weight once per slice.
        """
        product = self.rows(channels_in, channels_out, rows=math.prod(grid), bias=bias)
        if channels_last:
            return product
        t, h, _ = grid
        slices = self.batch_size * t * h
        return product + traffic(
            "primal",
            kernel="matmul",
            elements=(slices - 1) * channels_in * channels_out,
            dtype=self.dtype,
        )

    def rows(
        self,
        channels_in: int,
        channels_out: int,
        *,
        rows: int,
        bias: bool,
    ) -> Cost:
        """Cost a ``Linear`` over ``rows`` contiguous rows a clip."""
        return cost(
            Linear.Config(channels_in, channels_out, bias=bias),
            seq_len=rows,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )

    def norm(self, channels: int, *, grid: tuple[int, int, int]) -> Cost:
        """Cost ``_FrameGroupNorm`` on every frame of ``grid``."""
        t, h, w = grid
        return cost(
            GroupNorm2d.Config(
                channels,
                num_groups=self.num_groups,
                elementwise_affine=True,
            ),
            seq_len=h * w,
            batch_size=self.batch_size * t,
            dtype=self.dtype,
        )

    def causal(self, channels: int, *, grid: tuple[int, int, int]) -> Cost:
        """Cost a residual block's replicating ``CausalConv3d``."""
        return cost(
            CausalConv3d.Config(channels, channels, padding_mode="replicate"),
            input_grid=grid,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )

    def blocks(self, channels: int, count: int, *, grid: tuple[int, int, int]) -> Cost:
        """Cost ``count`` ``_ResBlock`` stacks on ``grid``, their attention apart."""
        elements = self.batch_size * channels * math.prod(grid)
        block = (
            self.norm(channels, grid=grid).tile(2, copies=2)
            + cost(silu, channels=elements, dtype=self.dtype).tile(2)
            + self.causal(channels, grid=grid).tile(2, copies=2)
            + SUM(channels=elements, dtype=self.dtype)
        )
        return block.tile(count, copies=count)

    def attention(
        self,
        channels: int,
        count: int,
        *,
        grid: tuple[int, int, int],
        channels_head: int,
        attn_kernel: Makeable[AttentionKernel],
    ) -> Cost:
        """Cost ``count`` ``_AttentionBlock`` on ``grid``: causal attention along time."""
        t, h, w = grid
        elements = self.batch_size * channels * math.prod(grid)
        block = (
            self.norm(channels, grid=grid)
            # Into rows of time and back.
            + traffic(
                "primal",
                kernel="elementwise",
                elements=4 * elements,
                dtype=self.dtype,
            )
            + self.rows(channels, 3 * channels, rows=math.prod(grid), bias=False)
            + POINTWISE(channels=2 * elements, dtype=self.dtype).tile(3)
            + cost(
                attn_kernel,
                seq_len=t,
                batch_size=self.batch_size * h * w,
                dtype=self.dtype,
                num_heads=channels // channels_head,
                channels_head=channels_head,
            )
            + self.rows(channels, channels, rows=math.prod(grid), bias=True)
            + SUM(channels=elements, dtype=self.dtype)
        )
        return block.tile(count, copies=count)

    def down(
        self,
        channels_in: int,
        channels_out: int,
        factor_t: int,
        factor_s: int,
        *,
        grid: tuple[int, int, int],
    ) -> tuple[Cost, tuple[int, int, int]]:
        """Cost a downsampling convolution; return it with the grid it leaves."""
        t, h, w = grid
        return cost(
            CausalConv3d.Config(
                channels_in,
                channels_out,
                kernel_size=(factor_t, factor_s, factor_s),
                stride=(factor_t, factor_s, factor_s),
                past=factor_t - 1,
                padding_mode="replicate",
            ),
            input_grid=grid,
            batch_size=self.batch_size,
            dtype=self.dtype,
        ), (
            (t + factor_t - 1) // factor_t,
            h // factor_s,
            w // factor_s,
        )

    def up(
        self,
        channels_in: int,
        channels_out: int,
        count: int,
        factor_t: int,
        factor_s: int,
        *,
        grid: tuple[int, int, int],
    ) -> tuple[Cost, tuple[int, int, int]]:
        """Cost ``_Upsample``; return it with the grid it leaves."""
        t, h, w = grid
        expanded = channels_out * factor_t * factor_s * factor_s
        out = (factor_t * t - factor_t + 1, h * factor_s, w * factor_s)
        unfolded = self.batch_size * expanded * math.prod(grid)
        total = (
            self.blocks(channels_in, count, grid=grid)
            + self.linear(
                channels_in,
                expanded,
                grid=grid,
                bias=True,
                channels_last=False,
            )
            # The unfold's strided copy.
            + traffic(
                "primal",
                kernel="elementwise",
                elements=2 * unfolded,
                dtype=self.dtype,
            )
        )
        return total, out

    def fourier(
        self,
        channels: int,
        octaves: int,
        *,
        grid: tuple[int, int, int],
    ) -> Cost:
        """Cost ``_fourier_features``: the repeat, the products, sines, cosines, and join."""
        elements = self.batch_size * channels * math.prod(grid)
        features = elements * octaves
        return (
            traffic(
                "primal",
                kernel="elementwise",
                elements=2 * features,
                dtype=self.dtype,
            )
            + _SIN_COS(channels=features, dtype=self.dtype)
            + traffic(
                "primal",
                kernel="elementwise",
                elements=2 * (elements + 2 * features),
                dtype=self.dtype,
            )
        )
