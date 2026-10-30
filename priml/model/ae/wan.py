"""Wan: causal 3D convolutional KL autoencoders for video, Wan 2.1's and Wan 2.2's.

Videos are ``1 + 4k`` frames. Every convolution is causal in time, padded on
its past side only, and the first frame passes each temporal resampler alone, so
a video's first frame encodes exactly as the image it is. The reference encodes
in chunks of one, then four, frames and decodes one latent frame at a time,
threading each causal convolution's past through a cache; this port computes
the same convolutions over the whole clip at once, which those caches
reproduce.

Wan 2.2 folds 2x2 pixels into channels around the halves, and adds a
parameter-free shortcut around each stage: blocks averaged down, or duplicated
up.

Build a published model from :func:`wan21` or :func:`wan22`, or their Comfy
repackagings, :func:`wan21_comfy` and :func:`wan22_comfy`, the same tensors
rounded to bfloat16 and float16: each a
:class:`~priml.model.ae.video.CausalVideoVAE` around this module's halves.

Apache-2.0 license and attribution: ``LICENSES/Wan2.1``, ``LICENSES/Wan2.2``.

References:
  https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/vae.py
  https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/wan/modules/vae2_2.py
    Wan Team 2025. Wan: Open and advanced large-scale video generative
    models.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import math
import re

from configgle import Fig, Makeable
from torch import Tensor, nn
from torch.nn import functional

import torch

from priml.cost import Cost, cost, elementwise_cost, traffic
from priml.math.pixel import patchify, unpatchify
from priml.math.tensor import per_frame
from priml.model.ae.checkpoint import CheckpointFile, HubFile, read_state
from priml.model.ae.custom_types import StateDictRemap
from priml.model.ae.frozen_cost import SUM
from priml.model.ae.latent_norm import InverseStdLatents
from priml.model.ae.video import CausalConv3d, CausalVideoVAE
from priml.model.attention.kernel import SdpaFused
from priml.model.conv import Conv2d, conv_cost
from priml.model.custom_types import AttentionKernel
from priml.model.swiglu import silu


if TYPE_CHECKING:
    from collections.abc import Mapping


def wan21_vae() -> HubFile.Config:
    """Return Wan 2.1's VAE as released, float32, which Wan 2.2's A14B models share.

    Returns:
      source: The pinned Hub file.

    """
    return HubFile.Config(
        repo_id="Wan-AI/Wan2.1-T2V-14B",
        filename="Wan2.1_VAE.pth",
        revision="a064a6c71f5be440641209c07bf2a5ce7a2ff5e4",
        sha256="38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
    )


def wan22_vae() -> HubFile.Config:
    """Return Wan 2.2's VAE as released, float32, with the TI2V-5B model.

    Returns:
      source: The pinned Hub file.

    """
    return HubFile.Config(
        repo_id="Wan-AI/Wan2.2-TI2V-5B",
        filename="Wan2.2_VAE.pth",
        revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
        sha256="20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36",
    )


def remap_wan_state_dict(
    state: Mapping[str, Tensor],
    *,
    half: str,
) -> dict[str, Tensor]:
    """Rename one half's Wan tensors to this module's names.

    Wan 2.2 nests each stage (``downsamples.1.downsamples.0``), as this module
    does; Wan 2.1 lists its blocks flat (``downsamples.3``), each stage closing
    on its resampler, and is numbered into stages here. The projection between
    the halves is the checkpoint's top-level ``conv1`` or ``conv2``.

    Args:
      state: Wan tensors; keys outside ``half`` are ignored.
      half: ``"encoder"`` or ``"decoder"``.

    Returns:
      state: This half's ``state_dict``.

    """
    # The checkpoint's name for the projection between the halves, then this
    # module's, which ``kl``'s halves share.
    released, projection = {
        "encoder": ("conv1.", "quant_conv."),
        "decoder": ("conv2.", "post_quant_conv."),
    }[half]
    stages = {"encoder": "downsamples", "decoder": "upsamples"}[half]
    flat_block = re.compile(
        rf"^{stages}\.(\d+)\.(residual|shortcut|resample|time_conv)\.",
    )
    remapped: dict[str, Tensor] = {}
    flat: dict[int, dict[str, Tensor]] = {}
    for key, tensor in state.items():
        if key.startswith(released):
            remapped[projection + key.removeprefix(released)] = tensor
        elif key.startswith(f"{half}."):
            name = key.removeprefix(f"{half}.")
            match = flat_block.match(name)
            if match is None:
                remapped[name] = tensor
            else:
                flat.setdefault(int(match.group(1)), {})[name] = tensor
    stage = block = 0
    for index in sorted(flat):
        names = flat[index]
        for name, tensor in names.items():
            nested = f"{stages}.{stage}.{stages}.{block}."
            remapped[name.replace(f"{stages}.{index}.", nested, 1)] = tensor
        block += 1
        if any(".resample." in name or ".time_conv." in name for name in names):
            stage, block = stage + 1, 0
    return remapped


class StageShortcut(Protocol):
    """Maps a stage's input to its output's shape without parameters."""

    def __call__(
        self,
        x: Tensor,
        /,
        *,
        channels_out: int,
        factor_t: int,
        factor_s: int,
    ) -> Tensor:
        """Return ``x`` resampled by ``factor_t`` in time, ``factor_s`` in space."""
        ...


def average_shortcut(
    x: Tensor,
    /,
    *,
    channels_out: int,
    factor_t: int,
    factor_s: int,
) -> Tensor:
    """Average each block and channel group down to ``channels_out`` channels.

    Time is zero-padded at its front to a multiple of ``factor_t``, so the first
    frame averages with a zero frame, as the reference's first chunk does.

    Args:
      x: ``[N, C, T, H, W]`` stage input.
      channels_out: Channels out; a divisor of ``C * factor_t * factor_s**2``.
      factor_t: Time downsampling.
      factor_s: Space downsampling, per axis.

    Returns:
      shortcut: ``[N, channels_out, ceil(T / factor_t), H / factor_s, W / factor_s]``.

    """
    x = functional.pad(
        x,
        (0, 0, 0, 0, (factor_t - x.shape[2] % factor_t) % factor_t, 0),
    )
    b, c, t, h, w = x.shape
    t, h, w = t // factor_t, h // factor_s, w // factor_s
    x = x.view(b, c, t, factor_t, h, factor_s, w, factor_s)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    group = c * factor_t * factor_s * factor_s // channels_out
    return x.view(b, channels_out, group, t, h, w).mean(dim=2)


def duplicate_shortcut(
    x: Tensor,
    /,
    *,
    channels_out: int,
    factor_t: int,
    factor_s: int,
) -> Tensor:
    """Duplicate channels into ``factor_t x factor_s x factor_s`` blocks.

    The first frame's ``factor_t - 1`` leading copies are dropped, as the
    reference's first chunk drops them, so time grows to ``factor_t (T - 1) + 1``.

    Args:
      x: ``[N, C, T, H, W]`` stage input.
      channels_out: Channels out; ``channels_out * factor_t * factor_s**2`` a
        multiple of ``C``.
      factor_t: Time upsampling.
      factor_s: Space upsampling, per axis.

    Returns:
      shortcut: ``[N, channels_out, factor_t (T - 1) + 1, H factor_s, W factor_s]``.

    """
    repeats = channels_out * factor_t * factor_s * factor_s // x.shape[1]
    x = x.repeat_interleave(repeats, dim=1)
    b, _, t, h, w = x.shape
    x = x.view(b, channels_out, factor_t, factor_s, factor_s, t, h, w)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.view(b, channels_out, t * factor_t, h * factor_s, w * factor_s)
    return x[:, :, factor_t - 1 :]


class Encoder(nn.Module):
    """Pixels in ``[-1, 1]`` to posterior moments, causal in time."""

    class Config(Fig["Encoder"]):
        """Widths, depth, resampling, and weights; stages in the order they run."""

        channels_in: int = 3
        """Image channels."""

        patch_size: int = 1
        """Side of the pixel patch folded into channels first; Wan 2.2 folds 2."""

        channels_hidden: tuple[int, ...] = (96, 192, 384, 384)
        """Width per stage, finest first; the input convolution writes the first."""

        blocks_per_stage: int = 2
        """Residual blocks per stage."""

        temporal: tuple[int, ...] = (1, 2, 2)
        """Time each stage's downsampling divides by, 1 or 2; one per stage but the last."""

        channels_latent: int = 16
        """Latent channels; the encoder emits twice as many, mean then log-variance."""

        shortcut: StageShortcut | None = None
        """Added around each stage; ``None`` adds none, as Wan 2.1."""

        attn_kernel: Makeable[AttentionKernel] = field(default_factory=SdpaFused.Config)
        """The middle's per-frame self-attention."""

        checkpoint: Makeable[CheckpointFile] | None = field(default_factory=wan21_vae)
        """Weights; only ``encoder.*`` and the moments' projection are read.

        ``None`` keeps the random initialization.
        """

        remap: StateDictRemap = remap_wan_state_dict
        """Renames ``checkpoint``'s tensors to this module's."""

        def factors(self) -> tuple[int, int]:
            """Return the ``(time, space)`` downsampling from pixels to latent."""
            space = self.patch_size << (len(self.channels_hidden) - 1)
            return math.prod(self.temporal), space

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
                attn_kernel=self.attn_kernel,
            )
            p, widths = self.patch_size, self.channels_hidden
            frames, height, width = input_grid
            grid = (frames, height // p, width // p)
            total = layers.causal(self.channels_in * p * p, widths[0], grid=grid)
            if p > 1:
                total += layers.fold(self.channels_in * frames * height * width)
            width_in = widths[0]
            for stage, width_out in enumerate(widths):
                stage_in, grid_in = width_in, grid
                for _ in range(self.blocks_per_stage):
                    total += layers.residual(width_in, width_out, grid=grid)
                    width_in = width_out
                if stage != len(widths) - 1:
                    down, grid = layers.down(
                        width_out,
                        grid=grid,
                        temporal=self.temporal[stage] == 2,
                    )
                    total += down
                if self.shortcut is not None:
                    total += layers.shortcut(stage_in, grid=grid_in, grid_out=grid)
            moments = 2 * self.channels_latent
            return (
                total
                + layers.middle(width_in, grid=grid)
                + layers.head(width_in, moments, grid=grid)
                + layers.causal(
                    moments,
                    moments,
                    grid=grid,
                    kernel=(1, 1, 1),
                    past=0,
                )
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        p, widths = config.patch_size, config.channels_hidden
        _check_temporal(config.temporal, stages=len(widths) - 1)
        self.patch_size = p
        self.conv1 = CausalConv3d.Config(config.channels_in * p * p, widths[0]).make()
        self.downsamples: nn.ModuleList[_Stage] = nn.ModuleList()
        width_in = widths[0]
        for stage, width_out in enumerate(widths):
            blocks: list[nn.Module] = []
            for _ in range(config.blocks_per_stage):
                blocks.append(_Residual(width_in, width_out))
                width_in = width_out
            down = stage != len(widths) - 1
            factor_t = config.temporal[stage] if down else 1
            if down:
                blocks.append(_Down(width_out, temporal=factor_t == 2))
            self.downsamples.append(
                _Stage(
                    nn.Sequential(*blocks),
                    attribute="downsamples",
                    channels_out=width_out,
                    factor_t=factor_t,
                    factor_s=2 if down else 1,
                    shortcut=config.shortcut,
                ),
            )
        self.middle = _middle(width_in, kernel=config.attn_kernel)
        self.head = _head(width_in, 2 * config.channels_latent)
        moments = 2 * config.channels_latent
        # Named for the checkpoints' own tensors, as ``kl`` names its projections.
        self.quant_conv = CausalConv3d.Config(
            moments,
            moments,
            kernel_size=(1, 1, 1),
            past=0,
        ).make()
        if config.checkpoint is not None:
            state = read_state(
                config.checkpoint.make().path(),
                prefixes=("encoder.", "conv1."),
            )
            self.load_state_dict(config.remap(state, half="encoder"))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Map ``[N, C_in, T, H, W]`` pixels in ``[-1, 1]`` to ``[N, 2 C, t, h, w]`` moments."""
        if self.patch_size > 1:
            x = _fold(x, self.patch_size)
        x = self.conv1(x)
        for stage in self.downsamples:
            x = stage(x)
        return self.quant_conv(self.head(self.middle(x)))


class Decoder(nn.Module):
    """Latents to pixels in ``[-1, 1]``, causal in time."""

    class Config(Fig["Decoder"]):
        """Widths, depth, resampling, and weights; stages in the order they run."""

        channels_out: int = 3
        """Image channels."""

        patch_size: int = 1
        """Side of the pixel patch unfolded from channels last; Wan 2.2 unfolds 2."""

        channels_hidden: tuple[int, ...] = (384, 384, 192, 96)
        """Width per stage, coarsest first; the input convolution writes the first."""

        channels_upsampled: tuple[int, ...] = (192, 192, 96)
        """Width each stage's upsampling writes, the next stage's input; one per stage but the last."""

        blocks_per_stage: int = 3
        """Residual blocks per stage; the published decoders run one more than their encoders."""

        temporal: tuple[int, ...] = (2, 2, 1)
        """Time each stage's upsampling multiplies by, 1 or 2, less the first frame's copy; one per stage but the last."""

        channels_latent: int = 16
        """Latent channels."""

        shortcut: StageShortcut | None = None
        """Added around each upsampling stage; ``None`` adds none, as Wan 2.1."""

        attn_kernel: Makeable[AttentionKernel] = field(default_factory=SdpaFused.Config)
        """The middle's per-frame self-attention."""

        checkpoint: Makeable[CheckpointFile] | None = field(default_factory=wan21_vae)
        """Weights; only ``decoder.*`` and the latent's projection are read.

        ``None`` keeps the random initialization.
        """

        remap: StateDictRemap = remap_wan_state_dict
        """Renames ``checkpoint``'s tensors to this module's."""

        def factors(self) -> tuple[int, int]:
            """Return the ``(time, space)`` downsampling from pixels to latent."""
            space = self.patch_size << (len(self.channels_hidden) - 1)
            return math.prod(self.temporal), space

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
                attn_kernel=self.attn_kernel,
            )
            z, widths, grid = self.channels_latent, self.channels_hidden, input_grid
            total = layers.causal(z, z, grid=grid, kernel=(1, 1, 1), past=0)
            total += layers.causal(z, widths[0], grid=grid) + layers.middle(
                widths[0],
                grid=grid,
            )
            width_in = widths[0]
            for stage, width_out in enumerate(widths):
                stage_in, grid_in = width_in, grid
                for _ in range(self.blocks_per_stage):
                    total += layers.residual(width_in, width_out, grid=grid)
                    width_in = width_out
                if stage != len(widths) - 1:
                    width_in = self.channels_upsampled[stage]
                    up, grid = layers.up(
                        width_out,
                        width_in,
                        grid=grid,
                        temporal=self.temporal[stage] == 2,
                    )
                    total += up
                    if self.shortcut is not None:
                        total += layers.shortcut(stage_in, grid=grid_in, grid_out=grid)
            total += layers.head(
                width_in,
                self.channels_out * self.patch_size**2,
                grid=grid,
            )
            if self.patch_size > 1:
                total += layers.fold(
                    self.channels_out * self.patch_size**2 * math.prod(grid),
                )
            return total

    def __init__(self, config: Config) -> None:
        super().__init__()
        p, widths, z = config.patch_size, config.channels_hidden, config.channels_latent
        stages = len(widths) - 1
        _check_temporal(config.temporal, stages=stages)
        if len(config.channels_upsampled) != stages:
            raise ValueError(
                f"channels_upsampled needs one width per upsampling stage, {stages}; "
                f"got {len(config.channels_upsampled)}.",
            )
        self.patch_size = p
        # Named for the checkpoints' own tensors, as the encoder's ``quant_conv``.
        self.post_quant_conv = CausalConv3d.Config(
            z,
            z,
            kernel_size=(1, 1, 1),
            past=0,
        ).make()
        self.conv1 = CausalConv3d.Config(z, widths[0]).make()
        self.middle = _middle(widths[0], kernel=config.attn_kernel)
        self.upsamples: nn.ModuleList[_Stage] = nn.ModuleList()
        width_in = widths[0]
        for stage, width_out in enumerate(widths):
            blocks: list[nn.Module] = []
            for _ in range(config.blocks_per_stage):
                blocks.append(_Residual(width_in, width_out))
                width_in = width_out
            up = stage != len(widths) - 1
            factor_t = config.temporal[stage] if up else 1
            if up:
                width_in = config.channels_upsampled[stage]
                blocks.append(_Up(width_out, width_in, temporal=factor_t == 2))
            self.upsamples.append(
                _Stage(
                    nn.Sequential(*blocks),
                    attribute="upsamples",
                    channels_out=width_in,
                    factor_t=factor_t,
                    factor_s=2,
                    shortcut=config.shortcut if up else None,
                ),
            )
        self.head = _head(width_in, config.channels_out * p * p)
        if config.checkpoint is not None:
            state = read_state(
                config.checkpoint.make().path(),
                prefixes=("decoder.", "conv2."),
            )
            self.load_state_dict(config.remap(state, half="decoder"))

    @override
    def forward(self, z: Tensor) -> Tensor:
        """Map ``[N, C, t, h, w]`` latents to ``[N, C_out, T, H, W]`` pixels in ``[-1, 1]``."""
        x = self.middle(self.conv1(self.post_quant_conv(z)))
        for stage in self.upsamples:
            x = stage(x)
        x = self.head(x)
        return _unfold(x, self.patch_size) if self.patch_size > 1 else x


_WAN21_MEAN: Final = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]  # fmt: skip
_WAN21_STD: Final = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]  # fmt: skip
_WAN22_MEAN: Final = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]  # fmt: skip
_WAN22_STD: Final = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]  # fmt: skip


def wan21() -> CausalVideoVAE.Config:
    """Return Wan 2.1's VAE: 16 channels, 4x in time, 8x in space.

    Wan 2.2's A14B models use it too.

    Returns:
      config: The published model, float32 weights pinned.

    """
    return CausalVideoVAE.Config(
        encoder=Encoder.Config(),
        decoder=Decoder.Config(),
        num_frames=81,
        image_size=(480, 832),
        latent_norm=InverseStdLatents.Config(
            mean=list(_WAN21_MEAN),
            std=list(_WAN21_STD),
        ),
    )


def wan22() -> CausalVideoVAE.Config:
    """Return Wan 2.2's VAE: 48 channels, 4x in time, 16x in space.

    Pixels fold 2x2 into channels before an 8x encoder, and each stage adds a
    parameter-free shortcut.

    Returns:
      config: The published model, float32 weights pinned.

    """
    return CausalVideoVAE.Config(
        encoder=Encoder.Config(
            patch_size=2,
            channels_hidden=(160, 320, 640, 640),
            channels_latent=48,
            shortcut=average_shortcut,
            checkpoint=wan22_vae(),
        ),
        decoder=Decoder.Config(
            patch_size=2,
            channels_hidden=(1024, 1024, 512, 256),
            channels_upsampled=(1024, 1024, 512),
            channels_latent=48,
            shortcut=duplicate_shortcut,
            checkpoint=wan22_vae(),
        ),
        num_frames=121,
        image_size=(704, 1280),
        latent_norm=InverseStdLatents.Config(
            mean=list(_WAN22_MEAN),
            std=list(_WAN22_STD),
        ),
    )


def wan21_comfy() -> CausalVideoVAE.Config:
    """Return :func:`wan21` read from Comfy's repackaging, rounded to bfloat16.

    Returns:
      config: The model, its weights pinned.

    """
    config = wan21()
    _repoint(
        config,
        HubFile.Config(
            repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
            filename="split_files/vae/wan_2.1_vae.safetensors",
            revision="123acf1cc74bccbb9bfff8ac1ee72edc08c2341d",
            sha256="2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
        ),
    )
    return config


def wan22_comfy() -> CausalVideoVAE.Config:
    """Return :func:`wan22` read from Comfy's repackaging, rounded to float16.

    Returns:
      config: The model, its weights pinned.

    """
    config = wan22()
    _repoint(
        config,
        HubFile.Config(
            repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
            filename="split_files/vae/wan2.2_vae.safetensors",
            revision="ee6f4a40737a995bf5818954cfce6d59443b0f04",
            sha256="e40321bd36b9709991dae2530eb4ac303dd168276980d3e9bc4b6e2b75fed156",
        ),
    )
    return config


def _repoint(config: CausalVideoVAE.Config, checkpoint: HubFile.Config) -> None:
    """Point both halves at their own copies of ``checkpoint``."""
    assert isinstance(config.encoder, Encoder.Config)
    assert isinstance(config.decoder, Decoder.Config)
    config.encoder.checkpoint = checkpoint
    config.decoder.checkpoint = checkpoint.copy_tree()


def _check_temporal(temporal: tuple[int, ...], *, stages: int) -> None:
    """Raise unless ``temporal`` holds one factor, 1 or 2, per resampling stage."""
    if len(temporal) != stages or not set(temporal) <= {1, 2}:
        raise ValueError(
            f"temporal needs one factor, 1 or 2, per resampling stage, {stages}; "
            f"got {temporal}.",
        )


def _fold(x: Tensor, p: int) -> Tensor:
    """Fold ``p x p`` pixels into channels, width factor outermost, as Wan 2.2 does."""
    # ``(c r q)``, ``r`` the width factor: ``patchify`` with the sides swapped, and
    # frames moved out from between the channels and the sides.
    folded = patchify(x.transpose(1, 2).transpose(-1, -2), (p, p))
    return folded.transpose(-1, -2).transpose(1, 2)


def _unfold(x: Tensor, p: int) -> Tensor:
    """Reverse :func:`_fold`."""
    unfolded = unpatchify(x.transpose(1, 2).transpose(-1, -2), (p, p))
    return unfolded.transpose(-1, -2).transpose(1, 2)


class _RMSNorm(nn.Module):
    """The reference's channel-first RMS norm: ``normalize(x) * sqrt(C) * gamma``."""

    def __init__(self, channels: int, *, spatial_axes: int) -> None:
        super().__init__()
        self.scale = cast(float, channels**0.5)
        self.gamma = nn.Parameter(torch.ones(channels, *(1,) * spatial_axes))

    @override
    def forward(self, x: Tensor) -> Tensor:
        normalized = functional.normalize(x, dim=1) * self.scale
        # ``+ 0.0`` is the reference's absent bias; it turns -0.0 into 0.0.
        return normalized * self.gamma + 0.0


class _Residual(nn.Module):
    """Two norm, SiLU, causal 3x3x3 layers, added to the (projected) input."""

    def __init__(self, channels_in: int, channels_out: int) -> None:
        super().__init__()
        # The reference's indices: its dropout, an identity in evaluation, at 5.
        self.residual = nn.Sequential(
            _RMSNorm(channels_in, spatial_axes=3),
            nn.SiLU(),
            CausalConv3d.Config(channels_in, channels_out).make(),
            _RMSNorm(channels_out, spatial_axes=3),
            nn.SiLU(),
            nn.Identity(),
            CausalConv3d.Config(channels_out, channels_out).make(),
        )
        self.shortcut = (
            CausalConv3d.Config(
                channels_in,
                channels_out,
                kernel_size=(1, 1, 1),
                past=0,
            ).make()
            if channels_in != channels_out
            else nn.Identity()
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        h = self.shortcut(x)
        return self.residual(x) + h


class _Attention(nn.Module):
    """Single-head self-attention within each frame, added to its input."""

    def __init__(self, channels: int, *, kernel: Makeable[AttentionKernel]) -> None:
        super().__init__()
        self.norm = _RMSNorm(channels, spatial_axes=2)
        self.to_qkv = Conv2d.Config(
            channels,
            3 * channels,
            kernel_size=1,
            padding=0,
            bias=True,
        ).make()
        self.proj = Conv2d.Config(
            channels,
            channels,
            kernel_size=1,
            padding=0,
            bias=True,
        ).make()
        self.kernel = kernel.make()

    @override
    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        frames = self.norm(x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w))
        # The reference's operands, ``[N, 1, S, C]`` views of one contiguous
        # tensor, handed over in the kernel's ``[N, S, 1, C]``.
        qkv = (
            self.to_qkv(frames)
            .reshape(b * t, 1, 3 * c, -1)
            .permute(0, 1, 3, 2)
            .contiguous()
        )
        q, k, v = (part.transpose(1, 2) for part in qkv.chunk(3, dim=-1))
        attended = self.kernel(q, k, v).transpose(1, 2).squeeze(1)
        out = self.proj(attended.permute(0, 2, 1).reshape(b * t, c, h, w))
        return out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4) + x


class _Down(nn.Module):
    """Halve each frame's sides; with ``temporal``, every frame but the first pairs off."""

    def __init__(self, channels: int, *, temporal: bool) -> None:
        super().__init__()
        self.resample = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            Conv2d.Config(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=0,
                bias=True,
            ).make(),
        )
        self.time_conv = (
            CausalConv3d.Config(
                channels,
                channels,
                kernel_size=(3, 1, 1),
                stride=(2, 1, 1),
                past=0,
            ).make()
            if temporal
            else None
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = per_frame(self.resample, x)
        if self.time_conv is None or x.shape[2] == 1:
            return x
        return torch.cat([x[:, :, :1], self.time_conv(x)], dim=2)


class _Up(nn.Module):
    """Double each frame's sides; with ``temporal``, every frame but the first doubles."""

    def __init__(self, channels: int, channels_out: int, *, temporal: bool) -> None:
        super().__init__()
        self.resample = nn.Sequential(
            nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
            Conv2d.Config(
                channels,
                channels_out,
                kernel_size=3,
                padding=1,
                bias=True,
            ).make(),
        )
        self.time_conv = (
            CausalConv3d.Config(channels, 2 * channels, kernel_size=(3, 1, 1)).make()
            if temporal
            else None
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        b, c, t, h, w = x.shape
        if self.time_conv is not None and t > 1:
            # Each later frame becomes two: the convolution's halves, interleaved.
            doubled = self.time_conv(x[:, :, 1:]).reshape(b, 2, c, t - 1, h, w)
            doubled = torch.stack((doubled[:, 0], doubled[:, 1]), 3)
            x = torch.cat(
                [x[:, :, :1], doubled.reshape(b, c, 2 * (t - 1), h, w)],
                dim=2,
            )
        return per_frame(self.resample, x)


class _Stage(nn.Module):
    """Blocks then a resampler, with an optional parameter-free shortcut around them."""

    def __init__(
        self,
        blocks: nn.Sequential,
        *,
        attribute: str,
        channels_out: int,
        factor_t: int,
        factor_s: int,
        shortcut: StageShortcut | None,
    ) -> None:
        super().__init__()
        # Named as each half's checkpoint names a stage's blocks.
        self.attribute = attribute
        self.add_module(attribute, blocks)
        self.shortcut = shortcut
        self.channels_out = channels_out
        self.factor_t = factor_t
        self.factor_s = factor_s

    @override
    def forward(self, x: Tensor) -> Tensor:
        out = cast(Tensor, self.get_submodule(self.attribute)(x))
        if self.shortcut is None:
            return out
        return out + self.shortcut(
            x,
            channels_out=self.channels_out,
            factor_t=self.factor_t,
            factor_s=self.factor_s,
        )


def _middle(channels: int, *, kernel: Makeable[AttentionKernel]) -> nn.Sequential:
    """Return residual block, attention, residual block."""
    return nn.Sequential(
        _Residual(channels, channels),
        _Attention(channels, kernel=kernel),
        _Residual(channels, channels),
    )


def _head(channels: int, channels_out: int) -> nn.Sequential:
    """Return norm, SiLU, and the causal 3x3x3 convolution out."""
    return nn.Sequential(
        _RMSNorm(channels, spatial_axes=3),
        nn.SiLU(),
        CausalConv3d.Config(channels, channels_out).make(),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Layers:
    """Price the reference's layers at one batch size, attention kernel, and dtype."""

    batch_size: int
    dtype: torch.dtype | None
    attn_kernel: Makeable[AttentionKernel]

    def causal(
        self,
        channels_in: int,
        channels_out: int,
        *,
        grid: tuple[int, int, int],
        kernel: tuple[int, int, int] = (3, 3, 3),
        stride: tuple[int, int, int] = (1, 1, 1),
        past: int = 2,
    ) -> Cost:
        """Cost ``CausalConv3d`` on ``grid``."""
        return cost(
            CausalConv3d.Config(
                channels_in,
                channels_out,
                kernel_size=kernel,
                stride=stride,
                past=past,
            ),
            input_grid=grid,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )

    def frames(
        self,
        channels_in: int,
        channels_out: int,
        *,
        grid: tuple[int, int, int],
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> Cost:
        """Cost a biased 2D convolution on every frame of ``grid``."""
        t, h, w = grid
        return conv_cost(
            channels_in=channels_in,
            channels_out=channels_out,
            kernel_size=kernel_size,
            ndim=2,
            groups=1,
            bias=True,
            input_grid=(h, w),
            batch_size=self.batch_size * t,
            stride=stride,
            padding=padding,
            dtype=self.dtype,
        )

    def norm(self, channels: int, *, grid: tuple[int, ...]) -> Cost:
        """Cost ``_RMSNorm`` on ``grid``: its gamma, a square, a sum, a divide, three maps."""
        rows = self.batch_size * math.prod(grid)
        return elementwise_cost(
            primal=6 * channels * rows,
            adjoint=0,
            channels=channels,
            inputs=3,
            outputs=3,
            adjoint_inputs=0,
            adjoint_outputs=0,
            params=channels,
            rows=rows,
            dtype=self.dtype,
        )

    def norm_silu(self, channels: int, *, grid: tuple[int, int, int]) -> Cost:
        """Cost ``_RMSNorm`` then SiLU on ``grid``."""
        elements = self.batch_size * channels * math.prod(grid)
        return self.norm(channels, grid=grid) + cost(
            silu,
            channels=elements,
            dtype=self.dtype,
        )

    def residual(
        self,
        channels_in: int,
        channels_out: int,
        *,
        grid: tuple[int, int, int],
    ) -> Cost:
        """Cost ``_Residual``, its shortcut projection included."""
        block = (
            self.norm_silu(channels_in, grid=grid)
            + self.causal(channels_in, channels_out, grid=grid)
            + self.norm_silu(channels_out, grid=grid)
            + self.causal(channels_out, channels_out, grid=grid)
            + SUM(
                channels=self.batch_size * channels_out * math.prod(grid),
                dtype=self.dtype,
            )
        )
        if channels_in != channels_out:
            block += self.causal(
                channels_in,
                channels_out,
                grid=grid,
                kernel=(1, 1, 1),
                past=0,
            )
        return block

    def middle(self, channels: int, *, grid: tuple[int, int, int]) -> Cost:
        """Cost ``_middle``: block, per-frame attention, block."""
        t, h, w = grid
        elements = self.batch_size * channels * math.prod(grid)
        attention = (
            # The frames' copy out and back, and the projections' contiguous copy.
            traffic(
                "primal",
                kernel="elementwise",
                elements=4 * elements + 6 * elements,
                dtype=self.dtype,
            )
            + self.norm(channels, grid=grid)
            + self.frames(channels, 3 * channels, grid=grid, kernel_size=1)
            + cost(
                self.attn_kernel,
                seq_len=h * w,
                batch_size=self.batch_size * t,
                dtype=self.dtype,
                num_heads=1,
                channels_head=channels,
            )
            + self.frames(channels, channels, grid=grid, kernel_size=1)
            + SUM(channels=elements, dtype=self.dtype)
        )
        return (
            self.residual(channels, channels, grid=grid).tile(2, copies=2) + attention
        )

    def head(
        self,
        channels: int,
        channels_out: int,
        *,
        grid: tuple[int, int, int],
    ) -> Cost:
        """Cost ``_head``: norm, SiLU, the causal convolution out."""
        return self.norm_silu(channels, grid=grid) + self.causal(
            channels,
            channels_out,
            grid=grid,
        )

    def down(
        self,
        channels: int,
        *,
        grid: tuple[int, int, int],
        temporal: bool,
    ) -> tuple[Cost, tuple[int, int, int]]:
        """Cost ``_Down`` on ``grid``; return it with the grid it leaves."""
        t, h, w = grid
        padded = self.batch_size * channels * t * (h + 1) * (w + 1)
        total = traffic(
            "primal",
            kernel="elementwise",
            elements=self.batch_size * channels * math.prod(grid) + padded,
            dtype=self.dtype,
        ) + self.frames(
            channels,
            channels,
            grid=(t, h + 1, w + 1),
            kernel_size=3,
            stride=2,
        )
        h, w = h // 2, w // 2
        if temporal and t > 1:
            total += self.causal(
                channels,
                channels,
                grid=(t, h, w),
                kernel=(3, 1, 1),
                stride=(2, 1, 1),
                past=0,
            )
            # The first frame and the convolution's, joined.
            t = 1 + (t - 3) // 2 + 1
            total += traffic(
                "primal",
                kernel="elementwise",
                elements=2 * self.batch_size * channels * t * h * w,
                dtype=self.dtype,
            )
        return total, (t, h, w)

    def up(
        self,
        channels: int,
        channels_out: int,
        *,
        grid: tuple[int, int, int],
        temporal: bool,
    ) -> tuple[Cost, tuple[int, int, int]]:
        """Cost ``_Up`` on ``grid``; return it with the grid it leaves."""
        t, h, w = grid
        frames = 1 + 2 * (t - 1) if temporal and t > 1 else t
        # Nearest upsampling reads each frame once and writes it four times.
        total = traffic(
            "primal",
            kernel="elementwise",
            elements=5 * self.batch_size * channels * frames * h * w,
            dtype=self.dtype,
        ) + self.frames(
            channels,
            channels_out,
            grid=(frames, 2 * h, 2 * w),
            kernel_size=3,
            padding=1,
        )
        if frames != t:
            # The time convolution, then the interleave's copy and the join.
            total += self.causal(
                channels,
                2 * channels,
                grid=(t - 1, h, w),
                kernel=(3, 1, 1),
            ) + traffic(
                "primal",
                kernel="elementwise",
                elements=4 * self.batch_size * channels * frames * h * w,
                dtype=self.dtype,
            )
        return total, (frames, 2 * h, 2 * w)

    def shortcut(
        self,
        channels: int,
        *,
        grid: tuple[int, int, int],
        grid_out: tuple[int, int, int],
    ) -> Cost:
        """Cost a stage shortcut: its copies, a reduction or a repeat, and the sum."""
        moved = self.batch_size * channels * (math.prod(grid) + math.prod(grid_out))
        return traffic(
            "primal",
            kernel="elementwise",
            elements=3 * moved,
            dtype=self.dtype,
        ) + SUM(
            channels=moved,
            dtype=self.dtype,
        )

    def fold(self, elements: int) -> Cost:
        """Cost folding or unfolding pixel patches: one strided copy of ``elements`` a clip."""
        moved = 2 * self.batch_size * elements
        return traffic("primal", kernel="elementwise", elements=moved, dtype=self.dtype)
