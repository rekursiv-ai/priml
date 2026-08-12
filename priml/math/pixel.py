from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import itertools
import math

from torch import Tensor, nn

import torch

from priml import image as _image
from priml.math.basic import ceil_div
from priml.math.custom_types import Tensorable, convert_to_tensor
from priml.math.pooling import adaptive_avg_pool2d, adaptive_avg_pool3d


if TYPE_CHECKING:
    from turbojpeg import TurboJPEG

    import numpy as np


# Default values for standard_latent_height_width_dict
_DEFAULT_COMPRESSION = (16, 16)  # config-globals: ignore -- image geometry default.
_DEFAULT_RESOLUTIONS = tuple(range(180, 2_160 + 1, 180))
_DEFAULT_ASPECTS = (
    16 / 9,
    4 / 3,
    1.0,
    3 / 4,
    9 / 16,
)  # config-globals: ignore -- aspect presets.


def rgb2float(x: Tensorable) -> Tensor:
    """Convert RGB values from [0, 255] to float in [-1, 1].

    Args:
      x: Tensor with RGB values in [0, 255] range.

    Returns:
      result: Tensor with values in [-1, 1] range.

    """
    x = convert_to_tensor(x, dtype=torch.float32)
    return x / 127.5 - 1.0


def float2rgb(x: Tensorable) -> Tensor:
    """Convert float values from [-1, 1] to RGB in [0, 255].

    Args:
      x: Tensor with float values in [-1, 1] range.

    Returns:
      result: Tensor with uint8 RGB values in [0, 255] range.

    """
    x = convert_to_tensor(x, dtype=torch.float32)
    return torch.clamp(x * 127.5 + 127.5, 0.0, 255.0).to(torch.uint8)


class ImageShape(NamedTuple):
    """Image shape in (height, width) format."""

    height: int
    width: int


class VideoShape(NamedTuple):
    """Video shape in (frames, height, width) format."""

    frames: int
    height: int
    width: int


class ShapeBundle(NamedTuple):
    """Latent and pixel-space video shapes."""

    latent: VideoShape
    pixel_train: VideoShape
    pixel_full: VideoShape


def compute_video_shapes(
    nominal_resolution: int = 1_080,
    aspect: float = 16 / 9,
    duration_sec: float = 5.0,
    fps: float = 30.0,
    compression: tuple[int, int, int] = VideoShape(frames=8, height=16, width=16),
) -> ShapeBundle:
    """Compute latent and pixel resolutions from nominal video parameters.

    Args:
      nominal_resolution: Vertical resolution (e.g., 1_080 for "1080p").
      aspect: Aspect ratio (width/height, e.g., 16/9).
      duration_sec: Video duration in seconds (0 for single image).
      fps: Frames per second.
      compression: Compression factors for (frames, height, width).

    Returns:
      shapes: ShapeBundle with latent, pixel_train, and pixel_full VideoShapes.

    """
    # Geometric-mean side length: h * w == pixel_scale**2, so it is a scale
    # (not a radius). The 4/3 lifts a nominal height to this mean side.
    pixel_scale = nominal_resolution * (4 / 3)
    aspect_sqrt = aspect**0.5
    h = round(pixel_scale / aspect_sqrt)
    w = round(pixel_scale * aspect_sqrt)
    f = 1 if duration_sec == 0 else round(duration_sec * fps)

    comp_f, comp_h, comp_w = compression
    lat_f = ceil_div(f, comp_f)
    lat_h = ceil_div(h, comp_h)
    lat_w = ceil_div(w, comp_w)

    px_f = lat_f * comp_f
    px_h = lat_h * comp_h
    px_w = lat_w * comp_w

    f = 1 if duration_sec == 0 else px_f  # Images keep f=1; videos round up.

    result = ShapeBundle(
        latent=VideoShape(lat_f, lat_h, lat_w),
        pixel_train=VideoShape(px_f, px_h, px_w),
        pixel_full=VideoShape(f, h, w),
    )
    if any(s <= 0 for s in result.latent):
        raise ValueError(f"Invalid latent shape {result.latent}.")
    if any(s <= 0 for s in result.pixel_train):
        raise ValueError(f"Invalid training pixel shape {result.pixel_train}.")
    if any(s <= 0 for s in result.pixel_full):
        raise ValueError(f"Invalid inference pixel shape {result.pixel_full}.")

    return result


def reconstruction_diffs(x: Tensor, y: Tensor, amplification: float = 3) -> Tensor:
    """Compute scaled absolute difference for reconstruction visualization.

    Args:
      x: First tensor.
      y: Second tensor.
      amplification: Scaling factor for differences.

    Returns:
      result: Clamped uint8 difference tensor.

    """
    x, y = convert_to_tensor(x, y, dtype=torch.float32)
    return torch.clamp(amplification * abs(x - y), 0, 255).type(torch.uint8)


def patchify(x: Tensorable, patch_size: Iterable[int]) -> Tensor:
    """Rearrange tensor into patches.

    More ergonomic than einops. For len(patch_size)==3, equivalent to:
    einops.rearrange(x, "... c (f fp) (h hp) (w wp) -> ... (c fp hp wp) f h w",
                     fp=patch_size[0], hp=patch_size[1], wp=patch_size[2])

    Args:
      x: Input tensor.
      patch_size: Patch dimensions for each spatial axis.

    Returns:
      result: Patchified tensor with patches in channel dimension.

    """
    x = convert_to_tensor(x)
    patch_size = tuple(patch_size)
    rank = len(patch_size)
    if len(x.shape) < rank:
        raise ValueError(f"{x.shape=} needs to have at least {rank=} dimensions.")
    batch = x.shape[: -rank - 1]
    interleaved = (
        (d // p, p) for d, p in zip(x.shape[-rank:], patch_size, strict=True)
    )
    interleaved = (v for pair in interleaved for v in pair)
    out = x.reshape(*batch, -1, *interleaved)
    base = len(batch)
    perm = [
        *range(base + 1),
        *range(base + 2, out.ndim, 2),
        *range(base + 1, out.ndim, 2),
    ]
    out = torch.permute(out, dims=tuple(perm))
    out = out.reshape(*batch, -1, *out.shape[-rank:])
    return out


def unpatchify(x: Tensorable, patch_size: Iterable[int]) -> Tensor:
    """Reverse patchify operation.

    More ergonomic than einops. For len(patch_size)==3, equivalent to:
    einops.rearrange(x, "... (c fp hp wp) f h w -> ... c (f fp) (h hp) (w wp)",
                     fp=patch_size[0], hp=patch_size[1], wp=patch_size[2])

    Args:
      x: Patchified tensor.
      patch_size: Patch dimensions for each spatial axis.

    Returns:
      result: Unpatchified tensor with restored spatial dimensions.

    """
    x = convert_to_tensor(x)
    patch_size = tuple(patch_size)
    rank = len(patch_size)
    if len(x.shape) < rank:
        raise ValueError(f"{x.shape=} needs to have at least {rank=} dimensions.")
    batch = x.shape[: -rank - 1]
    c, *spatial = x.shape[-rank - 1 :]
    out = x.reshape(*batch, c // math.prod(patch_size), *patch_size, *spatial)
    base = len(batch) + 1
    axis_order = [
        *range(base),
        *[v for i in range(rank) for v in (base + rank + i, base + i)],
    ]
    out = torch.permute(out, dims=tuple(axis_order))
    restored_dims = (
        a * b
        for a, b in zip(
            out.shape[-2 * rank :: 2],
            out.shape[-2 * rank + 1 :: 2],
            strict=True,
        )
    )
    out = out.reshape(*out.shape[: -2 * rank], *restored_dims)
    return out


InterpolateMode = Literal[
    "area",
    "area-variance-preserving",
    "bicubic",
    "bilinear",
    "linear",
    "nearest",
    "nearest-exact",
    "trilinear",
]


def interpolate(
    input_: Tensorable,
    mode: InterpolateMode = "nearest",
    *,
    size: int | Sequence[int] = (),
    scale_factor: float | Sequence[int | float] = (),
    align_corners: bool = False,
    recompute_scale_factor: bool = False,
    antialias: bool = False,
    rank: int | None = None,
    channels_last: bool = False,
) -> Tensor:
    """Interpolate tensor with automatic rank inference and reshaping.

    More ergonomic than nn.functional.interpolate. Automatically handles:
    - Mode inference (e.g., "linear" becomes "bilinear" or "trilinear")
    - Input padding/reshaping based on rank
    - Size/scale_factor expansion to sequences

    Args:
      input_: Input tensor.
      mode: Interpolation mode.
      size: Output size.
      scale_factor: Scaling factor.
      align_corners: Whether to align corners.
      recompute_scale_factor: Whether to recompute scale factor.
      antialias: Whether to use antialiasing.
      rank: Spatial rank (inferred from mode/size/scale_factor if None).
      channels_last: Whether input is channels-last format.

    Returns:
      result: Interpolated tensor.

    Raises:
      NotImplementedError: For ``mode="area-variance-preserving"`` with a
        spatial rank other than 2 or 3 (only 2-D and 3-D are supported).

    """
    x = convert_to_tensor(input_)
    output_rank, mode_, size_, sf_, ac_, rank_ = _process_interpolate_args(
        mode,
        size,
        scale_factor,
        align_corners,
        rank,
    )
    if x.ndim < min(rank_, output_rank):
        raise ValueError(f"{x.ndim} smaller than {min(rank_,output_rank)=}.")

    if channels_last:
        x = x.moveaxis(-1, -rank_ - 1)

    if rank_ < output_rank:
        # Fewer input spatial dims than requested: insert unit axes to lift rank.
        x = x.reshape(
            *x.shape[:-rank_],
            *(1,) * (output_rank - rank_),
            *x.shape[-rank_:],
        )
    elif rank_ > output_rank:
        # More input spatial dims than requested: fold the extras via axis reorder.
        x = torch.permute(
            x,
            dims=tuple(
                itertools.chain(
                    range(-len(x.shape), -rank_ - 1),
                    range(-rank_, output_rank - rank_ - 1),
                    (-rank_ - 1,),
                    range(output_rank - rank_ - 1, 0),
                ),
            ),
        )

    batch_shape = x.shape[: -output_rank - 1]
    x = x.reshape(-1, *x.shape[-output_rank - 1 :])

    if mode_ == "area-variance-preserving":
        if rank_ == 2:
            f = adaptive_avg_pool2d
        elif rank_ == 3:
            f = adaptive_avg_pool3d
        else:
            raise NotImplementedError(f"{rank_=} not supported for {mode_=}.")
        if antialias or ac_ or recompute_scale_factor or (not size_ and not sf_):
            raise NotImplementedError(
                f"One or more of: {antialias=}, {ac_=}, {recompute_scale_factor=}, {size_=}, {sf_=} not supported for {mode_=}.",
            )
        if not size_:
            if sf_ is None:
                raise ValueError("scale_factor is required when size is not provided.")
            shape_slice: Sequence[int] = list(x.shape[-rank_:])
            size_ = tuple(int(o * s) for o, s in zip(shape_slice, sf_, strict=True))
        x = f(x, output_size=size_, variance_preserving=True)  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
    else:
        x = nn.functional.interpolate(
            input=x,
            size=size_,
            scale_factor=sf_,
            mode=mode_,
            align_corners=ac_,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )
    x = x.reshape(*batch_shape, *x.shape[-output_rank - 1 :])

    if rank_ > output_rank:
        x = torch.permute(
            x,
            dims=tuple(
                itertools.chain(
                    range(-len(x.shape), -rank_ - 1),
                    (-output_rank - 1,),
                    range(-rank_ - 1, -output_rank - 1),
                    range(-output_rank, 0),
                ),
            ),
        )

    if channels_last:
        x = x.moveaxis(-max(output_rank, rank_) - 1, -1)

    return x


def _process_interpolate_args(
    mode: InterpolateMode = "nearest",
    size: int | Sequence[int] = (),
    scale_factor: float | Sequence[int | float] = (),
    align_corners: bool = False,
    rank: int | None = None,
) -> tuple[
    int,
    InterpolateMode,
    tuple[int, ...] | None,
    tuple[int | float, ...] | None,
    bool | None,
    int,
]:
    # Determine output spatial rank from size/scale_factor.
    if mode.startswith("bi"):
        output_rank = 2
    elif mode.startswith("tri"):
        output_rank = 3
    elif isinstance(size, Sequence) and size:
        output_rank = len(size)
    elif isinstance(scale_factor, Sequence) and scale_factor:
        output_rank = len(scale_factor)
    elif rank is not None:
        output_rank = rank
    elif mode == "linear":
        # Fallback: "linear" alone implies 1D.
        output_rank = 1
    else:
        raise ValueError("Unable to infer the output rank.")

    # Resolve input spatial rank.
    rank_ = output_rank if rank is None else rank

    # Map short mode names to torch-expected names.
    mode_: InterpolateMode = mode
    if output_rank == 2:
        mode_ = cast(
            InterpolateMode,
            {"linear": "bilinear", "cubic": "bicubic"}.get(mode, mode),
        )
    elif output_rank == 3:
        mode_ = cast(InterpolateMode, {"linear": "trilinear"}.get(mode, mode))

    # Compute target spatial dimensions.
    # `isinstance(x, Sequence)` narrows to a bare `Sequence`, dropping the element
    # type the parameter annotation already carries; re-state it on the way out.
    size_: tuple[int, ...] | None
    if isinstance(size, Sequence):
        size_ = tuple(int(s) for s in size) if size else None
    else:
        size_ = (size,) * output_rank

    # Derive scale factors from size if not given.
    sf_: tuple[int | float, ...] | None
    if isinstance(scale_factor, Sequence):
        sf_ = tuple(float(s) for s in scale_factor) if scale_factor else None
    else:
        sf_ = (scale_factor,) * output_rank

    # Set alignment based on interpolation mode.
    ac_: bool | None = align_corners
    if (
        not align_corners
        and not mode_.endswith("linear")
        and not mode_.endswith("cubic")
    ):
        ac_ = None

    return output_rank, mode_, size_, sf_, ac_, rank_


# -- bytes-to-tensor decoders ------------------------------------------
# Thin wrappers over ``priml.image``: numpy decode → zero-copy
# ``torch.from_numpy`` → optional stride-only channels-first rearrange.


def _to_tensor(arr: np.ndarray | None, channels_first: bool) -> Tensor | None:
    """Zero-copy numpy → torch, optionally channels-first via strides.

    Returns None on None input so decoder bodies can be a single line.
    """
    if arr is None:
        return None
    tensor = torch.from_numpy(arr)
    if channels_first:
        tensor = tensor.moveaxis(-1, -3)
    return tensor


def decode_jpeg_turbojpeg(
    image_bytes: bytes,
    turbo_jpeg: TurboJPEG,
    height: int,
    width: int,
    *,
    crop: tuple[int, int] | tuple[int, int, int, int] | None = None,
    channels_first: bool = True,
) -> Tensor | None:
    """Decode JPEG → uint8 Tensor. See ``priml.image.decode_jpeg_turbojpeg``."""
    return _to_tensor(
        _image.decode_jpeg_turbojpeg(image_bytes, turbo_jpeg, height, width, crop),
        channels_first,
    )


def decode_webp_libwebp(
    image_bytes: bytes,
    height: int,
    width: int,
    crop: tuple[int, int] | tuple[int, int, int, int] | None = None,
    channels_first: bool = True,
) -> Tensor | None:
    """Decode WebP → uint8 Tensor. See ``priml.image.decode_webp_libwebp``."""
    return _to_tensor(
        _image.decode_webp_libwebp(image_bytes, height, width, crop),
        channels_first,
    )


def decode_image_pil(
    image_bytes: bytes,
    height: int,
    width: int,
    *,
    crop: tuple[int, int] | tuple[int, int, int, int] | None = None,
    channels_format: Literal["rgb", "rgba"] = "rgb",
    channels_first: bool = True,
) -> Tensor | None:
    """Decode via PIL → uint8 Tensor. See ``priml.image.decode_image_pil``."""
    return _to_tensor(
        _image.decode_image_pil(image_bytes, height, width, crop, channels_format),
        channels_first,
    )
