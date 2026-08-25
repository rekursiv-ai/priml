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


def rgb2float(
    x: Tensor,
    *,
    inplace: bool = False,
    unit_interval: bool = False,
) -> Tensor:
    """Convert RGB values from [0, 255] to float in [-1, 1].

    Equivalent to `x.to(float).sub(127.5).div(127.5)`, or with
    ``unit_interval=True`` to `x.to(float).div(255)`.

    Args:
      x: Floating-point tensor holding RGB values in [0, 255]. Cast at the
        call site; float16 is exact over all 256 levels round-tripped through
        ``float2rgb``, float8 is not. Scaled unconditionally, so a tensor
        already in [-1, 1] comes back in [-1.008, -0.992] -- guard on
        ``dtype == torch.uint8`` where the range depends on which decoder ran.
      inplace: Overwrite ``x`` rather than allocating. Unsafe when ``x``
        outlives the call, such as a pipeline sample read after ``yield``.
      unit_interval: Emit [0, 1]. Set it for a model whose preprocessing
        demands that range, or ahead of a mean/std normalize -- see Notes.

    Returns:
      result: Tensor in [-1, 1], or [0, 1] when ``unit_interval`` is set.

    Raises:
      TypeError: If ``x`` is not floating point. Torch would fail on an
        integer and silently drop the imaginary part of a complex.
      ValueError: If ``inplace`` is set for a leaf tensor requiring grad.

    Notes:
      Scaling straight from [0, 255] has half the error of the two-step route
      diffusers (``/255`` then ``2v - 1``) and torchvision (``/255`` then
      ``normalize(.5, .5)``) both take: their intermediate rounds at magnitude
      ~1, and the doubling scales that error with it. Proven in
      ``test_rgb2float_halves_the_error_of_a_unit_interval_intermediate``.

      A later rescale cancels the advantage. A mean/std normalize divides a
      [-1, 1] intermediate by ``2s`` rather than ``s``, giving the same
      relative error, and forces the published constants to be restated as
      ``2m - 1`` and ``2s``. Use ``unit_interval`` ahead of one, as both
      CIFAR-10 loaders do.

    """
    if not x.dtype.is_floating_point:
        raise TypeError(
            f"rgb2float expects floating-point input in [0, 255]; got {x.dtype}. "
            "Cast at the call site, e.g. x.to(torch.float16).",
        )

    if inplace and x.requires_grad and x.is_leaf:
        raise ValueError(
            "rgb2float cannot scale in place: x is a leaf requiring grad. "
            "Pass inplace=False, or detach the tensor first.",
        )

    # Divide rather than multiply by a reciprocal: 1/127.5, 2/255 and 1/255
    # are all unrepresentable, so the reciprocal's own error reaches every
    # element. The compiler does not strength-reduce it away.
    if unit_interval:
        return x.div_(255.0) if inplace else x / 255.0

    # Subtract first so the divide is the only rounding step, at the result's
    # magnitude. 127.5 is exactly 255/2, so the subtraction is exact.
    if inplace:
        return x.sub_(127.5).div_(127.5)

    return (x - 127.5) / 127.5


def float2rgb(
    x: Tensor,
    *,
    float_dtype: torch.dtype | None = None,
    inplace: bool = False,
    unit_interval: bool = False,
) -> Tensor:
    """Convert float values in [-1, 1] to RGB in [0, 255].

    Equivalent to `x.mul(127.5).add(127.5).round().clamp(0, 255).to(uint8)`,
    or with ``unit_interval=True`` to
    `x.mul(255).round().clamp(0, 255).to(uint8)`.

    Args:
      x: Floats in [-1, 1], or [0, 1] when ``unit_interval`` is set.
        Out-of-range values are clamped.
      float_dtype: Width the scaling runs at; the result is always uint8.
        ``None`` keeps the input's width. Widening float16 to float32 costs
        ~3.7x the runtime for at most one uint8 level on 1% of inputs, and
        nothing at bfloat16, where the 8 mantissa bits are the limit.
      inplace: Overwrite ``x`` rather than allocating the float intermediate;
        the uint8 result is fresh either way. Unsafe when ``x`` outlives the
        call.
      unit_interval: Read ``x`` as [0, 1]. Must match what the value actually
        carries -- an all-dark frame looks the same in either range, so a
        mismatch silently washes the image out rather than raising.

    Returns:
      result: uint8 tensor in [0, 255].

    Raises:
      TypeError: If ``x`` is not floating point.

    Notes:
      Quantizes directly from [-1, 1] rather than denormalizing to [0, 1]
      first, which recovers the 16 bfloat16 levels diffusers drops. Rounds as
      diffusers does; the two agree at float16 and float32.

      torchvision's ``convert_image_dtype`` truncates after scaling by
      ``256 - 1e-3``, which rounds to exactly 256.0 at bfloat16 and float16 --
      white overflows and the uint8 cast wraps to black, losing 128 of 256
      levels. Both claims proven in
      ``test_float2rgb_round_trips_where_both_baselines_lose_levels``.

    """
    if not x.dtype.is_floating_point:
        # Without this an integer tensor is read as [-1, 1]: uint8 [0, 1, 2]
        # would come back [128, 255, 255].
        raise TypeError(
            "float2rgb expects floating-point input in "
            f"{'[0, 1]' if unit_interval else '[-1, 1]'}; got {x.dtype}.",
        )

    # A width change allocates, so that buffer is ours to overwrite whatever
    # the caller asked for.
    x_ = x if float_dtype is None else x.to(float_dtype)

    # Scale then offset, inverting ``rgb2float`` step for step. The equivalent
    # ``(x + 1.0) * 127.5`` rounds near 1 and multiplies that error by 127.5,
    # breaking the round trip for 16 of 256 bfloat16 levels.
    #
    # Python floats, not ``torch.addcmul``: fusing would round once instead of
    # twice (under one uint8 level on 0.3% of inputs) but ``addcmul`` takes
    # only 0-dim tensor scalars, which are memory loads rather than folded
    # immediates -- ~2x slower at the float16/bfloat16 widths decode uses.
    scale = 255.0 if unit_interval else 127.5
    if (x_ is not x) or inplace:
        x_ = x_.mul_(scale) if unit_interval else x_.mul_(scale).add_(127.5)
    else:
        x_ = x_ * scale if unit_interval else x_ * scale + 127.5

    # ``round_`` is for arbitrary input, not the round trip: a value from
    # ``rgb2float`` is already integral, so truncation would reproduce all 256
    # levels and look correct. Generated output is off that lattice, where
    # truncation errs one-sided and darkens every image by half a level.
    #
    # Two cheaper spellings fail. ``+ 128.0`` with a truncating cast loses 63
    # bfloat16 levels -- spacing is 1.0 above 128, so the .5 is gone before
    # the cast. ``+ 0.5`` then floor passes every lattice point (each is a
    # tie) but brightens by a quarter level off it.
    #
    # ``clamp_`` is load-bearing: the uint8 cast wraps modularly, so 1.004
    # would become 0 and turn saturated highlights black.
    return x_.round_().clamp_(0.0, 255.0).to(torch.uint8)


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

    Raises:
      ValueError: If any argument is outside its domain. Checked here rather
        than at one call site: every other caller reached ``ceil_div`` with a
        zero stride, or ``aspect**0.5`` with a negative, and failed as
        ZeroDivisionError or a complex number far from the argument at fault.

    """
    if nominal_resolution <= 0:
        raise ValueError(
            f"nominal_resolution must be positive; got {nominal_resolution}."
        )
    if aspect <= 0 or not math.isfinite(aspect):
        raise ValueError(f"aspect must be finite and positive; got {aspect}.")
    if duration_sec < 0 or not math.isfinite(duration_sec):
        raise ValueError(
            f"duration_sec must be finite and non-negative; got {duration_sec}."
        )
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError(f"fps must be finite and positive; got {fps}.")
    if any(c < 1 for c in compression):
        raise ValueError(f"compression strides must be positive; got {compression}.")
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

    Raises:
      ValueError: If ``patch_size`` is empty or non-positive, if ``x`` has no
        channel axis beyond the patch rank, or if a spatial dimension is not
        divisible by its patch.

    """
    x = convert_to_tensor(x)
    patch_size = tuple(patch_size)
    _validate_patch_size(patch_size)
    rank = len(patch_size)
    # ``rank + 1``, not ``rank``: the channel axis sits left of the spatial
    # ones, and at equality ``batch`` came out empty and the reshape invented
    # a channel dimension rather than failing.
    if len(x.shape) < rank + 1:
        raise ValueError(f"{x.shape=} needs at least {rank + 1} dimensions.")
    spatial = tuple(x.shape[-rank:])
    # Checked here rather than left to the reshape below: `d // p` discards the
    # remainder, so a ragged dimension fails inside torch with a message naming
    # neither the axis nor the patch size.
    if any(d % p for d, p in zip(spatial, patch_size, strict=True)):
        raise ValueError(
            f"spatial dims {spatial} must each be divisible by {patch_size=}."
        )
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

    Raises:
      ValueError: If ``patch_size`` is empty or non-positive, if ``x`` has no
        channel axis beyond the patch rank, or if the channel count is not a
        multiple of the patch volume.

    """
    x = convert_to_tensor(x)
    patch_size = tuple(patch_size)
    _validate_patch_size(patch_size)
    rank = len(patch_size)
    # See ``patchify``: the unpack below needs the channel axis too.
    if len(x.shape) < rank + 1:
        raise ValueError(f"{x.shape=} needs at least {rank + 1} dimensions.")
    batch = x.shape[: -rank - 1]
    c, *spatial = x.shape[-rank - 1 :]
    # The channel axis carries one patch volume per output channel, so a
    # remainder here means the tensor was never patchified with this patch.
    if c % math.prod(patch_size):
        raise ValueError(
            f"channels {c} must be divisible by the patch volume "
            f"{math.prod(patch_size)} from {patch_size=}."
        )
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
        if rank_ not in (2, 3):
            raise NotImplementedError(f"{rank_=} not supported for {mode_=}.")
        if antialias or ac_ or recompute_scale_factor or (not size_ and not sf_):
            raise NotImplementedError(
                f"One or more of: {antialias=}, {ac_=}, {recompute_scale_factor=}, {size_=}, {sf_=} not supported for {mode_=}.",
            )
        if not size_:
            # Reachable only with a scale factor: the check above already
            # raised when neither was given.
            assert sf_ is not None
            shape_slice: Sequence[int] = list(x.shape[-rank_:])
            size_ = tuple(int(o * s) for o, s in zip(shape_slice, sf_, strict=True))
        # Dispatched on the LENGTH of the size tuple, which is what each
        # callee's signature names. Selecting the function by ``rank_`` and
        # passing ``size_`` separately let the two disagree, which the
        # suppression here used to hide: an explicit ``rank=2`` with a 3-tuple
        # size reached the 2-D pool and failed inside it as "Expected 3D or 4D
        # tensor, got 5D".
        if len(size_) == 2:
            x = adaptive_avg_pool2d(x, (size_[0], size_[1]), variance_preserving=True)
        elif len(size_) == 3:
            x = adaptive_avg_pool3d(
                x, (size_[0], size_[1], size_[2]), variance_preserving=True
            )
        else:
            raise NotImplementedError(f"{size_=} not supported for {mode_=}.")
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
        mode_ = cast("InterpolateMode", {"linear": "trilinear"}.get(mode, mode))

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


def _validate_patch_size(patch_size: tuple[int, ...]) -> None:
    """Reject a patch that cannot tile anything."""
    if not patch_size:
        raise ValueError("patch_size must name at least one axis.")
    if any(p < 1 for p in patch_size):
        raise ValueError(f"patch_size entries must be positive; got {patch_size}.")


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
