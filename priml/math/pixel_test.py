from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

from PIL import Image

import numpy as np
import pytest
import torch

from priml.math.pixel import (
    compute_video_shapes,
    decode_image_pil,
    decode_jpeg_turbojpeg,
    decode_webp_libwebp,
    float2rgb,
    interpolate,
    patchify,
    reconstruction_diffs,
    rgb2float,
    unpatchify,
)


def test_rgb2float():
    x = torch.tensor([0.0, 127.5, 255.0])
    result = rgb2float(x)
    # float16 by default: half the memory, and still exact over all 256
    # levels (pinned by the round-trip test below).
    assert result.dtype == torch.float16
    expected = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float16)
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


def test_float2rgb():
    x = torch.tensor([-1.0, 0.0, 1.0])
    result = float2rgb(x)
    expected = torch.tensor([0, 128, 255], dtype=torch.uint8)
    # Exact: ``atol=1`` here could not tell rounding from truncation, which is
    # the one thing this conversion has to get right.
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def test_the_pixel_pair_is_an_exact_round_trip():
    """Every uint8 level must survive uint8 -> float -> uint8, in either width.

    Truncating instead of rounding biases every value down by up to a level,
    so an encode/decode cycle darkens the image and repeated cycles drift.
    Measured before the fix: 4 of 256 levels came back one low in float16.
    """
    levels = torch.arange(256, dtype=torch.uint8)
    # bfloat16 is the one that catches a rounding shortcut: folding the half
    # into the offset (``+ 128.0`` instead of ``round()``) is exact in every
    # other width and loses 127 of these 256 in this one.
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        assert torch.equal(float2rgb(rgb2float(levels, dtype=dtype)), levels)


def test_float2rgb_partitions_the_domain_into_equal_bins():
    """Every level must claim an equal share of [-1, 1], endpoints included.

    Surjectivity alone is too weak to assert on its own: the round-trip test
    above already implies it, since each level arriving back from its own
    preimage puts it in the image. What that test cannot see is the shape of
    the preimage, because it only ever samples the 256 lattice points. This
    one sweeps the continuum, so it measures the quantization bin behind
    each level rather than one point inside it.

    Two ways that goes wrong, both of which land here:
      - Truncating instead of rounding shifts every boundary half a bin,
        starving level 255 and doubling level 0 (65 against 1, not 33/33).
      - A scale of 127 rather than 127.5 empties level 0 entirely.

    Interior bins hold one step each and the two endpoints hold half, the
    domain ending there. float16/bfloat16 are excluded: the swept grid
    collapses onto too few distinct values at those widths to measure a bin
    (bfloat16 keeps 2_304 of 16_385 samples), which is a property of the
    input grid rather than of the conversion.
    """
    grid = torch.linspace(-1.0, 1.0, steps=256 * 64 + 1, dtype=torch.float64)
    for dtype in (torch.float32, torch.float64):
        counts = torch.bincount(
            float2rgb(grid.to(dtype)).to(torch.int64),
            minlength=256,
        )
        unreachable = (counts == 0).nonzero().flatten().tolist()
        assert not unreachable, f"{dtype}: levels {unreachable} unreachable"
        assert int(counts[0]) == int(counts[255]), (
            f"{dtype}: endpoint bins {int(counts[0])} vs {int(counts[255])}"
        )
        interior = counts[1:255]
        assert int(interior.max()) - int(interior.min()) <= 1, (
            f"{dtype}: interior bins span "
            f"[{int(interior.min())}, {int(interior.max())}]"
        )


def test_float2rgb_clamp():
    # Test clamping for out-of-range values
    x = torch.tensor([-2.0, 0.0, 2.0])
    result = float2rgb(x)
    # Should clamp to [0, 255]
    assert result[0] == 0
    assert result[2] == 255


def test_compute_video_shapes():
    # 1080p 16:9 at 30fps for 3 seconds
    actual = compute_video_shapes(
        nominal_resolution=1080,
        aspect=16 / 9,
        duration_sec=3.0,
        fps=30,
    )
    assert actual.latent == (12, 68, 120)
    assert actual.pixel_train == (96, 1088, 1920)
    assert actual.pixel_full == (96, 1080, 1920)

    # 480p 4:3 at 24fps for 2 seconds
    actual = compute_video_shapes(
        nominal_resolution=480,
        aspect=4 / 3,
        duration_sec=2.0,
        fps=24,
    )
    assert actual.latent == (6, 35, 47)
    assert actual.pixel_train == (48, 560, 752)
    assert actual.pixel_full == (48, 554, 739)

    # 360p 1:1 at 15fps for 4 seconds
    actual = compute_video_shapes(
        nominal_resolution=360,
        aspect=1.0,
        duration_sec=4.0,
        fps=15,
    )
    assert actual.latent == (8, 30, 30)
    assert actual.pixel_train == (64, 480, 480)
    assert actual.pixel_full == (64, 480, 480)


def test_compute_video_shapes_image():
    # Test with duration_sec=0 for single image
    actual = compute_video_shapes(
        nominal_resolution=1080,
        aspect=16 / 9,
        duration_sec=0.0,
        fps=30,
    )
    # For images, training and inference frames should be 1
    # but latent might be different
    assert actual.pixel_train.frames == actual.latent.frames * 8
    assert actual.pixel_full.frames == 1


def test_reconstruction_diffs():
    x = torch.tensor([100.0, 150.0, 200.0])
    y = torch.tensor([110.0, 140.0, 190.0])
    result = reconstruction_diffs(x, y, amplification=3)
    expected = torch.tensor([30, 30, 30], dtype=torch.uint8)
    torch.testing.assert_close(result, expected, atol=1, rtol=0)


def test_patchify():
    x = torch.randn(1, 4, 8, 12, 24)
    patch = (2, 4, 6)
    z = patchify(x, patch)
    # c_out = 4 * 2 * 4 * 6 = 192, spatial = (4, 3, 4)
    assert z.shape == (1, 192, 4, 3, 4), z.shape
    # Verify roundtrip recovers original
    x_back = unpatchify(z, patch)
    torch.testing.assert_close(x_back, x, rtol=0, atol=0)


def test_patchify_2d():
    # Test with 2D patches
    x = torch.randn(2, 12, 8, 16)
    patch = (4, 8)
    z = patchify(x, patch)
    # c=12, patch=(4,8) -> c_out = 12*4*8 = 384
    assert z.shape == (2, 384, 2, 2)


def test_unpatchify():
    patch = (2, 3, 4)
    # Build a patchified tensor: c_patch = 5 * 2 * 3 * 4 = 120
    x = torch.randn(3, 120, 6, 5, 7)
    z = unpatchify(x, patch)
    # Spatial dims: (6*2, 5*3, 7*4) = (12, 15, 28), channels: 5
    assert z.shape == (3, 5, 12, 15, 28), z.shape
    # Verify roundtrip recovers original
    x_back = patchify(z, patch)
    torch.testing.assert_close(x_back, x, rtol=0, atol=0)


def test_unpatchify_2d():
    # Test with 2D patches
    x = torch.randn(2, 384, 2, 2)
    patch = (4, 8)
    z = unpatchify(x, patch)
    assert z.shape == (2, 12, 8, 16)


def test_patchify_unpatchify_roundtrip():
    x = torch.randn(1, 6, 12, 16, 24)
    patch = (3, 4, 8)
    z = patchify(x, patch)
    x_recovered = unpatchify(z, patch)
    torch.testing.assert_close(x, x_recovered, rtol=1e-6, atol=1e-6)


def test_interpolate_nearest():
    x = torch.randn(2, 3, 8, 8)
    result = interpolate(x, mode="nearest", scale_factor=2, rank=2)
    assert result.shape == (2, 3, 16, 16)


def test_interpolate_bilinear():
    x = torch.randn(2, 3, 8, 8)
    result = interpolate(x, mode="bilinear", size=(16, 16), align_corners=False)
    assert result.shape == (2, 3, 16, 16)


def test_interpolate_trilinear():
    x = torch.randn(2, 3, 4, 8, 8)
    result = interpolate(x, mode="trilinear", scale_factor=2, align_corners=False)
    assert result.shape == (2, 3, 8, 16, 16)


def test_interpolate_area():
    x = torch.randn(2, 3, 16, 16)
    result = interpolate(x, mode="area", size=(8, 8))
    assert result.shape == (2, 3, 8, 8)


def test_interpolate_area_variance_preserving():
    x = torch.randn(1, 2, 8, 8)
    result = interpolate(x, mode="area-variance-preserving", size=(4, 4), rank=2)
    assert result.shape == (1, 2, 4, 4)


def test_interpolate_area_variance_preserving_3d():
    x = torch.randn(1, 2, 8, 8, 8)
    result = interpolate(x, mode="area-variance-preserving", size=(4, 4, 4), rank=3)
    assert result.shape == (1, 2, 4, 4, 4)


def test_interpolate_with_rank():
    # Test interpolation with explicit rank
    x = torch.randn(2, 3, 4, 8, 8)
    result = interpolate(x, mode="linear", scale_factor=2, rank=2)
    assert result.shape == (2, 3, 4, 16, 16)


def test_interpolate_channels_last():
    # Test with channels_last format
    x = torch.randn(2, 8, 8, 3)
    result = interpolate(x, mode="nearest", scale_factor=2, rank=2, channels_last=True)
    assert result.shape == (2, 16, 16, 3)


def test_interpolate_linear_to_bilinear():
    # Test that "linear" mode gets converted to "bilinear" for 2D
    x = torch.randn(2, 3, 8, 8)
    result = interpolate(x, mode="linear", scale_factor=2, rank=2)
    assert result.shape == (2, 3, 16, 16)


def test_interpolate_linear_to_trilinear():
    # Test that "linear" mode gets converted to "trilinear" for 3D
    x = torch.randn(2, 3, 4, 8, 8)
    result = interpolate(x, mode="linear", scale_factor=2, rank=3)
    assert result.shape == (2, 3, 8, 16, 16)


def test_interpolate_cubic_to_bicubic():
    # Test that "cubic" mode gets converted to "bicubic" for 2D
    x = torch.randn(2, 3, 8, 8)
    result = interpolate(x, mode="cubic", scale_factor=2, rank=2)  # ty: ignore[invalid-argument-type]  # pyright: ignore[reportArgumentType]
    assert result.shape == (2, 3, 16, 16)


def test_patchify_insufficient_dimensions():
    """Test patchify raises ValueError when tensor has insufficient dimensions."""
    x = torch.randn(2, 3)  # Only 2 dimensions
    patch = (4, 8, 12)  # Needs 3 spatial dims plus a channel axis
    with pytest.raises(ValueError, match="needs at least"):
        patchify(x, patch)


def test_unpatchify_insufficient_dimensions():
    """Test unpatchify raises ValueError when tensor has insufficient dimensions."""
    x = torch.randn(2, 1152)  # Only 2 dimensions
    patch = (4, 8, 12)  # Needs 3 spatial dims plus a channel axis
    with pytest.raises(ValueError, match="needs at least"):
        unpatchify(x, patch)


def test_patchify_rejects_a_non_divisible_spatial_dim():
    """A spatial dim that is not a whole number of patches is named, not floored.

    ``d // p`` silently discards the remainder, and the reshape two lines later
    then fails inside torch with a message naming neither the axis nor the
    patch size. The rank guard beside it already raises a clean ValueError.
    """
    with pytest.raises(ValueError, match="divisible"):
        patchify(torch.randn(2, 3, 32, 33), (4, 4))


def test_unpatchify_rejects_channels_that_are_not_a_patch_multiple():
    """The channel axis must divide by the patch volume it will unfold into."""
    with pytest.raises(ValueError, match="divisible"):
        unpatchify(torch.randn(2, 7, 4, 4), (2, 2))


def test_interpolate_insufficient_dimensions():
    """Test interpolate raises ValueError when input has insufficient dimensions."""
    x = torch.randn(2)  # Only 1 dimension
    with pytest.raises(ValueError, match="smaller than"):
        interpolate(x, mode="bilinear", size=(4, 4))


def test_interpolate_rank_padding():
    """Test interpolate with rank < output_rank (needs padding)."""
    # 1D input (rank=1) interpolated to 2D output (output_rank=2)
    x = torch.randn(2, 3, 8)  # [batch, channels, width]
    result = interpolate(x, mode="bilinear", size=(8, 16), rank=1)
    # Should pad to [2, 3, 1, 8] then interpolate to [2, 3, 8, 16]
    assert result.shape == (2, 3, 8, 16)


def test_interpolate_area_variance_preserving_scale_factor():
    """Test area-variance-preserving mode with scale_factor."""
    x = torch.randn(1, 2, 16, 16)
    result = interpolate(x, mode="area-variance-preserving", scale_factor=0.5, rank=2)
    assert result.shape == (1, 2, 8, 8)


def test_interpolate_area_variance_preserving_scale_factor_3d():
    """Test area-variance-preserving mode with scale_factor in 3D."""
    x = torch.randn(1, 2, 8, 8, 8)
    result = interpolate(x, mode="area-variance-preserving", scale_factor=0.5, rank=3)
    assert result.shape == (1, 2, 4, 4, 4)


def test_interpolate_linear_1d():
    """Test linear mode with output_rank=1."""
    x = torch.randn(2, 3, 8)
    result = interpolate(x, mode="linear", size=16, rank=1)
    assert result.shape == (2, 3, 16)


def test_interpolate_unable_to_infer_rank():
    """Test interpolate raises ValueError when unable to infer output rank."""
    x = torch.randn(2, 3, 8, 8)
    # No explicit rank, no size/scale_factor, mode doesn't specify rank
    with pytest.raises(ValueError, match="Unable to infer the output rank"):
        interpolate(x, mode="area")


def test_interpolate_size_scalar():
    """Test interpolate with scalar size (non-sequence)."""
    x = torch.randn(2, 3, 8, 8)
    result = interpolate(x, mode="nearest", size=16, rank=2)
    assert result.shape == (2, 3, 16, 16)


def test_interpolate_rank_greater_than_output_rank():
    """Test interpolate with rank > output_rank (lines 288, 339)."""
    # 3D input (rank=3) interpolated to 2D output (output_rank=2)
    x = torch.randn(2, 3, 4, 8, 8)  # [batch, channels, depth, height, width]
    result = interpolate(x, mode="bilinear", size=(16, 16), rank=3)
    # Should interpolate spatial dims and reshape
    assert result.shape == (2, 3, 4, 16, 16)


def test_interpolate_infer_from_scale_factor():
    """Test interpolate infers output_rank from scale_factor (line 372)."""
    x = torch.randn(2, 3, 8, 8, 8)
    # scale_factor as sequence infers output_rank
    result = interpolate(x, mode="nearest", scale_factor=(2, 2, 2))
    assert result.shape == (2, 3, 16, 16, 16)


def test_interpolate_linear_1d_fallback():
    """Test interpolate with mode='linear' defaults to output_rank=1 (line 377)."""
    x = torch.randn(2, 3, 8)
    # No size/scale_factor/rank, mode='linear' should default to output_rank=1
    result = interpolate(x, mode="linear", size=16)
    assert result.shape == (2, 3, 16)


def test_interpolate_rank_greater_output_rank_channels_last():
    """Test interpolate with rank > output_rank and channels_last (line 339)."""
    # 3D input with channels last
    x = torch.randn(2, 4, 8, 8, 3)  # [batch, depth, height, width, channels]
    result = interpolate(x, mode="bilinear", size=(16, 16), rank=3, channels_last=True)
    # Should handle permutation correctly
    assert result.shape == (2, 4, 16, 16, 3)


def test_decode_jpeg_turbojpeg_success():
    """Test decode_jpeg_turbojpeg successful decode."""
    mock_turbo = MagicMock()
    mock_bgr = np.ones((100, 100, 3), dtype=np.uint8) * 128
    mock_turbo.decode.return_value = mock_bgr

    image_bytes = b"fake_jpeg_bytes"
    tensor = decode_jpeg_turbojpeg(
        image_bytes,
        mock_turbo,
        height=100,
        width=100,
    )

    assert tensor is not None
    assert tensor.shape == (3, 100, 100)
    assert tensor.dtype == torch.uint8


def test_decode_jpeg_turbojpeg_channels_last():
    """Test decode_jpeg_turbojpeg with channels_first=False."""
    mock_turbo = MagicMock()
    mock_bgr = np.ones((100, 100, 3), dtype=np.uint8) * 128
    mock_turbo.decode.return_value = mock_bgr

    image_bytes = b"fake_jpeg_bytes"
    tensor = decode_jpeg_turbojpeg(
        image_bytes,
        mock_turbo,
        height=100,
        width=100,
        channels_first=False,
    )

    assert tensor is not None
    assert tensor.shape == (100, 100, 3)
    assert tensor.dtype == torch.uint8


def test_decode_jpeg_turbojpeg_with_crop():
    """Test decode_jpeg_turbojpeg with cropping."""
    mock_turbo = MagicMock()
    mock_turbo.crop.return_value = b"cropped_jpeg_bytes"
    rng = np.random.default_rng()
    mock_bgr = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    mock_turbo.decode.return_value = mock_bgr

    image_bytes = b"fake_jpeg_bytes"
    tensor = decode_jpeg_turbojpeg(
        image_bytes,
        mock_turbo,
        height=200,
        width=400,
        crop=(100, 100),  # Center crop to 1:1 aspect ratio
    )

    assert tensor is not None
    # Verify crop was called
    mock_turbo.crop.assert_called_once()


def test_decode_jpeg_turbojpeg_bgr_to_rgb():
    """Test decode_jpeg_turbojpeg converts BGR to RGB."""
    mock_turbo = MagicMock()
    mock_bgr = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_bgr[:, :, 0] = 255  # Blue channel
    mock_bgr[:, :, 1] = 128  # Green channel
    mock_bgr[:, :, 2] = 64  # Red channel
    mock_turbo.decode.return_value = mock_bgr

    image_bytes = b"fake_jpeg_bytes"
    tensor = decode_jpeg_turbojpeg(
        image_bytes,
        mock_turbo,
        height=10,
        width=10,
        channels_first=False,
    )

    assert tensor is not None
    # Check RGB conversion: BGR -> RGB
    # Red channel (was BGR[2]) should be RGB[0]
    assert torch.all(tensor[:, :, 0] == 64)
    # Green channel should stay same
    assert torch.all(tensor[:, :, 1] == 128)
    # Blue channel (was BGR[0]) should be RGB[2]
    assert torch.all(tensor[:, :, 2] == 255)


def test_decode_jpeg_turbojpeg_error():
    """Test decode_jpeg_turbojpeg returns None on error."""
    mock_turbo = MagicMock()
    mock_turbo.decode.side_effect = Exception("Decode error")

    image_bytes = b"fake_jpeg_bytes"
    tensor = decode_jpeg_turbojpeg(
        image_bytes,
        mock_turbo,
        height=100,
        width=100,
    )

    assert tensor is None


def test_decode_webp_libwebp_error():
    """Test decode_webp_libwebp returns None on error."""
    with patch("webp.WebPDecoderConfig", side_effect=Exception("Decode error")):
        tensor = decode_webp_libwebp(b"fake_webp", height=100, width=100)
        assert tensor is None


def test_decode_image_pil_jpg():
    """Test decode_image_pil with JPEG."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    tensor = decode_image_pil(img_bytes, height=100, width=100)

    assert tensor is not None
    assert tensor.shape == (3, 100, 100)
    assert tensor.dtype == torch.uint8


def test_decode_image_pil_png():
    """Test decode_image_pil with PNG."""
    img = Image.new("RGB", (100, 100), color=(0, 255, 0))
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    tensor = decode_image_pil(img_bytes, height=100, width=100)

    assert tensor is not None
    assert tensor.shape == (3, 100, 100)
    assert tensor.dtype == torch.uint8


def test_decode_image_pil_channels_last():
    """Test decode_image_pil with channels_first=False."""
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    tensor = decode_image_pil(
        img_bytes,
        height=100,
        width=100,
        channels_first=False,
    )

    assert tensor is not None
    assert tensor.shape == (100, 100, 3)
    assert tensor.dtype == torch.uint8


def test_decode_image_pil_rgba_to_rgb():
    """Test decode_image_pil converts RGBA to RGB."""
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    tensor = decode_image_pil(
        img_bytes,
        height=100,
        width=100,
        channels_format="rgb",
    )

    assert tensor is not None
    assert tensor.shape == (3, 100, 100)


def test_decode_image_pil_rgba_output():
    """Test decode_image_pil with RGBA output."""
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    tensor = decode_image_pil(
        img_bytes,
        height=100,
        width=100,
        channels_format="rgba",
    )

    assert tensor is not None
    assert tensor.shape == (4, 100, 100)


def test_decode_image_pil_with_crop():
    """Test decode_image_pil with center crop."""
    img = Image.new("RGB", (200, 100), color=(255, 0, 0))
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    tensor = decode_image_pil(
        img_bytes,
        height=100,
        width=200,
        crop=(50, 50),
    )

    assert tensor is not None
    # Crop should be applied during decode
    assert tensor.shape[1] <= 100  # Height
    assert tensor.shape[2] <= 200  # Width


def test_decode_image_pil_draft_mode():
    """Test decode_image_pil uses draft mode for JPEG."""
    img = Image.new("RGB", (1000, 1000), color=(255, 0, 0))
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    tensor = decode_image_pil(
        img_bytes,
        height=1000,
        width=1000,
        crop=(100, 100),
    )

    assert tensor is not None
    # Draft mode + crop should produce smaller output than original
    assert tensor.shape[1] <= 1000
    assert tensor.shape[2] <= 1000


def test_decode_image_pil_error():
    """Test decode_image_pil returns None on error."""
    tensor = decode_image_pil(
        b"not_an_image",
        height=100,
        width=100,
    )

    assert tensor is None


def test_decode_image_pil_decompression_bomb():
    """Test decode_image_pil handles decompression bomb error."""
    with patch("PIL.Image.open", side_effect=Image.DecompressionBombError("Too large")):
        tensor = decode_image_pil(
            b"fake_bytes",
            height=100,
            width=100,
        )

        assert tensor is None


def test_decode_image_pil_grayscale_to_rgb():
    """Test decode_image_pil converts grayscale to RGB."""
    img = Image.new("L", (100, 100), color=128)
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    tensor = decode_image_pil(
        img_bytes,
        height=100,
        width=100,
        channels_format="rgb",
    )

    assert tensor is not None
    assert tensor.shape == (3, 100, 100)


def test_decode_image_pil_grayscale_to_rgba():
    """Test decode_image_pil converts grayscale to RGBA."""
    img = Image.new("L", (100, 100), color=128)
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    tensor = decode_image_pil(
        img_bytes,
        height=100,
        width=100,
        channels_format="rgba",
    )

    assert tensor is not None
    assert tensor.shape == (4, 100, 100)


def test_patchify_pair_requires_a_channel_axis_beyond_the_patch_rank():
    """``ndim == rank`` has no channel axis, so it must not be silently invented.

    ``patchify(zeros(4, 4), [2, 2])`` returned ``[4, 2, 2]``: ``batch`` came out
    empty and the reshape conjured a channel dimension, so the round trip
    returned ``(1, 4, 4)`` for a ``(4, 4)`` input. ``unpatchify`` shares the
    guard and failed as a bare ``RuntimeError`` from ``permute``.
    """
    with pytest.raises(ValueError, match="at least"):
        _ = patchify(torch.zeros(4, 4), [2, 2])
    with pytest.raises(ValueError, match="at least"):
        _ = unpatchify(torch.zeros(4, 4), [2, 2])

    # One more axis is the minimum, and it round-trips.
    x = torch.arange(3 * 4 * 4).reshape(3, 4, 4).float()
    torch.testing.assert_close(unpatchify(patchify(x, [2, 2]), [2, 2]), x)


def test_patchify_rejects_a_degenerate_patch():
    """A non-positive patch cannot tile, and reached ``d // p`` as a bare error.

    ``Patchify.Config`` validates this (``patchify.py:79``) but the public
    helper it wraps did not, so a direct caller got ZeroDivisionError.
    """
    for bad in ([0, 0], [-2, -2], [2, 0], []):
        with pytest.raises(ValueError, match="patch_size"):
            _ = patchify(torch.zeros(2, 3, 8, 8), bad)
        with pytest.raises(ValueError, match="patch_size"):
            _ = unpatchify(torch.zeros(2, 12, 4, 4), bad)


def test_compute_video_shapes_validates_its_own_arguments():
    """Each parameter is checked where it is used, not at one call site.

    ``CalcResizeDimensions`` validates ``compression`` on its behalf
    (``shapes.py:477``), leaving every other caller to hit ZeroDivisionError in
    ``ceil_div``; a negative aspect reached ``aspect**0.5`` and produced a
    complex number that failed much later in ``round``.
    """
    with pytest.raises(ValueError, match="compression"):
        _ = compute_video_shapes(compression=(1, 0, 16))
    with pytest.raises(ValueError, match="compression"):
        _ = compute_video_shapes(compression=(1, -16, 16))
    with pytest.raises(ValueError, match="aspect"):
        _ = compute_video_shapes(aspect=-1.0)
    with pytest.raises(ValueError, match="aspect"):
        _ = compute_video_shapes(aspect=0.0)
    with pytest.raises(ValueError, match="nominal_resolution"):
        _ = compute_video_shapes(nominal_resolution=0)
    with pytest.raises(ValueError, match="fps"):
        _ = compute_video_shapes(fps=0.0)
    with pytest.raises(ValueError, match="duration_sec"):
        _ = compute_video_shapes(duration_sec=-1.0)


def test_float2rgb_rejects_an_integer_tensor() -> None:
    """The dtype hint fires only for input carrying NO dtype.

    An integer tensor has one, so it reached the scaling and had its byte
    values treated as ``[-1, 1]`` floats: uint8 ``[0, 1, 2]`` came back
    ``[128, 255, 255]`` rather than round-tripping.
    """
    for dtype in (torch.uint8, torch.int64):
        with pytest.raises(TypeError, match="floating-point"):
            _ = float2rgb(torch.tensor([0, 1, 2], dtype=dtype))
    # A bare Python list still takes the hint and works.
    assert float2rgb([-1.0, 0.0, 1.0]).dtype == torch.uint8


def test_inplace_overwrites_only_a_caller_buffer_that_needed_no_conversion() -> None:
    """With no conversion to hide behind, ``inplace`` decides the aliasing."""
    values = [-1.0, 0.0, 0.5, 1.0]

    kept = torch.tensor(values, dtype=torch.float16)
    before = kept.clone()
    _ = rgb2float(kept, dtype=torch.float16, inplace=False)
    assert torch.equal(kept, before), "inplace=False overwrote the caller"

    donated = torch.tensor(values, dtype=torch.float16)
    result = rgb2float(donated, dtype=torch.float16, inplace=True)
    assert result.data_ptr() == donated.data_ptr()


@pytest.mark.gpu_torch_cuda
def test_a_converted_input_costs_one_buffer_not_two() -> None:
    """A converted buffer is scaled in place whether or not ``inplace`` is set.

    Marked ``gpu_torch_cuda`` because the CUDA caching allocator is the only
    counter that sees this: there is no public CPU equivalent, and the result
    cannot be compared against the converted tensor's address from outside --
    on both paths it merely fails to alias the caller's uint8 input.

    Declining to overwrite a buffer this function just minted allocates a
    second one to protect a temporary nobody else holds, which is exactly the
    peak the widening was meant to avoid.
    """
    numel = 1 << 22
    result_mib = numel * 2 / 2**20
    for inplace in (False, True):
        source = torch.randint(0, 256, (numel,), dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        scaled = rgb2float(source, dtype=torch.float16, inplace=inplace)
        torch.cuda.synchronize()
        peak_mib = (torch.cuda.max_memory_allocated() - base) / 2**20
        assert peak_mib < result_mib * 1.5, (
            f"{inplace=} peaked at {peak_mib:.1f} MiB for a "
            f"{result_mib:.1f} MiB result: the converted buffer was copied."
        )
        del scaled, source
        torch.cuda.empty_cache()


def test_inplace_on_a_grad_leaf_names_the_argument_at_fault() -> None:
    """Torch's own message names neither the function nor the parameter.

    Left to it, the failure surfaces as ``a leaf Variable that requires grad
    is being used in an in-place operation`` from inside the scaling, which
    does not say that ``inplace=`` is the argument to change.
    """
    for fn, arg in ((rgb2float, 128.0), (float2rgb, 0.5)):
        leaf = torch.tensor([arg], dtype=torch.float32, requires_grad=True)
        with pytest.raises(ValueError, match="leaf requiring grad"):
            _ = fn(leaf, dtype=torch.float32, inplace=True)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
