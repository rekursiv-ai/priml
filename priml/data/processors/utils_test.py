from __future__ import annotations

import pytest
import torch

from priml.data.processors.utils import (
    as_image_batch_tensor,
    compute_keyframes_as_progressive_bisection,
    image_batch_to_pil_list,
    preprocess_images,
    safe_aspect_ratio,
    sample_frame_indices,
)


def test_safe_aspect_ratio_normal():
    """Test safe_aspect_ratio with normal values."""
    ratio = safe_aspect_ratio(1080, 1920)
    assert ratio == pytest.approx(1920 / 1080)
    assert ratio == pytest.approx(16 / 9, rel=1e-6)


def test_safe_aspect_ratio_square():
    """Test safe_aspect_ratio with square aspect ratio."""
    ratio = safe_aspect_ratio(100, 100)
    assert ratio == pytest.approx(1.0)


def test_safe_aspect_ratio_portrait():
    """Test safe_aspect_ratio with portrait orientation."""
    ratio = safe_aspect_ratio(1920, 1080)
    assert ratio == pytest.approx(1080 / 1920)
    assert ratio == pytest.approx(9 / 16, rel=1e-6)


def test_safe_aspect_ratio_zero_height():
    """Test safe_aspect_ratio with zero height returns infinity."""
    ratio = safe_aspect_ratio(0, 1920)
    assert ratio == torch.inf


def test_safe_aspect_ratio_zero_width():
    """Test safe_aspect_ratio with zero width returns zero."""
    ratio = safe_aspect_ratio(1080, 0)
    assert ratio == 0.0


def test_safe_aspect_ratio_both_zero():
    """Test safe_aspect_ratio with both zero."""
    ratio = safe_aspect_ratio(0, 0)
    # 0 / 0 would be nan, but we check height != 0 first.
    assert ratio == torch.inf


def test_safe_aspect_ratio_float_values():
    """Test safe_aspect_ratio with float values."""
    ratio = safe_aspect_ratio(1080.5, 1920.5)
    assert ratio == pytest.approx(1920.5 / 1080.5)


def test_sample_frame_indices_single_frame():
    """Test sample_frame_indices with single frame request."""
    indices = sample_frame_indices(total_frames=10, num_frames=1)
    assert indices.shape == (1,)
    assert indices.dtype == torch.long
    # Should sample middle frame (index 5 for 10 frames)
    assert indices[0] == 5


def test_sample_frame_indices_all_frames():
    """Test sample_frame_indices requesting all frames."""
    indices = sample_frame_indices(total_frames=5, num_frames=5)
    assert indices.shape == (5,)
    torch.testing.assert_close(indices, torch.tensor([0, 1, 2, 3, 4]))


def test_sample_frame_indices_evenly_spaced():
    """Test sample_frame_indices returns evenly spaced frames."""
    indices = sample_frame_indices(total_frames=10, num_frames=3)
    assert indices.shape == (3,)
    # Should be evenly spaced: 0, 4.5, 9 -> 0, 4, 9.
    expected = torch.tensor([0, 4, 9], dtype=torch.long)
    torch.testing.assert_close(indices, expected)


def test_sample_frame_indices_more_than_available():
    """Test sample_frame_indices when requesting more frames than available."""
    indices = sample_frame_indices(total_frames=5, num_frames=10)
    # Should clamp to total_frames.
    assert indices.shape == (5,)
    torch.testing.assert_close(indices, torch.tensor([0, 1, 2, 3, 4]))


def test_sample_frame_indices_two_frames():
    """Test sample_frame_indices with two frames."""
    indices = sample_frame_indices(total_frames=10, num_frames=2)
    assert indices.shape == (2,)
    # Should be first and last: 0, 9.
    expected = torch.tensor([0, 9], dtype=torch.long)
    torch.testing.assert_close(indices, expected)


def test_sample_frame_indices_dtype():
    """Test sample_frame_indices returns long dtype."""
    indices = sample_frame_indices(total_frames=10, num_frames=5)
    assert indices.dtype == torch.long


def test_sample_frame_indices_large_video():
    """Test sample_frame_indices with large video."""
    indices = sample_frame_indices(total_frames=1000, num_frames=10)
    assert indices.shape == (10,)
    assert indices[0] == 0
    assert indices[-1] == 999
    # Check spacing is approximately uniform.
    diffs = indices[1:] - indices[:-1]
    assert torch.all(diffs > 0)  # Monotonically increasing.


def test_sample_frame_indices_edge_case_one_total_frame():
    """Test sample_frame_indices with only one total frame."""
    indices = sample_frame_indices(total_frames=1, num_frames=1)
    assert indices.shape == (1,)
    assert indices[0] == 0


def test_sample_frame_indices_edge_case_one_total_many_requested():
    """Test sample_frame_indices with one frame but many requested."""
    indices = sample_frame_indices(total_frames=1, num_frames=10)
    assert indices.shape == (1,)
    assert indices[0] == 0


def test_sample_frame_indices_middle_frame_odd_total():
    """Test middle frame calculation with odd total frames."""
    indices = sample_frame_indices(total_frames=9, num_frames=1)
    # Middle of 9 frames (0-8) is index 4.
    assert indices[0] == 4


def test_sample_frame_indices_middle_frame_even_total():
    """Test middle frame calculation with even total frames."""
    indices = sample_frame_indices(total_frames=10, num_frames=1)
    # Middle of 10 frames (0-9) is index 5 (using integer division)
    assert indices[0] == 5


def test_as_image_batch_tensor_cfhw_to_fchw():
    """Test as_image_batch_tensor converts (C, F, H, W) to (F, C, H, W)."""
    x = torch.zeros(3, 8, 224, 224)  # (C, F, H, W)
    result = as_image_batch_tensor(x)
    assert result.shape == (8, 3, 224, 224)  # (F, C, H, W)


def test_as_image_batch_tensor_image_f1():
    """Test as_image_batch_tensor with F=1 (image case)."""
    x = torch.zeros(3, 1, 224, 224)  # (C, F=1, H, W)
    result = as_image_batch_tensor(x)
    assert result.shape == (1, 3, 224, 224)  # (F=1, C, H, W)


def test_as_image_batch_tensor_batched():
    """Test as_image_batch_tensor with batched input (B, C, F, H, W)."""
    x = torch.zeros(4, 3, 8, 224, 224)  # (B, C, F, H, W)
    result = as_image_batch_tensor(x)
    assert result.shape == (4, 8, 3, 224, 224)  # (B, F, C, H, W)


def test_as_image_batch_tensor_flatten_leading_dims():
    """Test as_image_batch_tensor flattens leading dimensions beyond 5D."""
    x = torch.zeros(2, 3, 3, 8, 224, 224)  # (*, *, C, F, H, W)
    result = as_image_batch_tensor(x)
    assert result.shape == (6, 8, 3, 224, 224)  # (B=6, F, C, H, W)


def test_as_image_batch_tensor_preserves_values():
    """Test as_image_batch_tensor preserves tensor values."""
    x = torch.arange(3 * 2 * 4 * 4).reshape(3, 2, 4, 4).float()  # (C, F, H, W)
    result = as_image_batch_tensor(x)
    assert result.shape == (2, 3, 4, 4)  # (F, C, H, W)
    # Verify values preserved by checking a few elements.
    assert torch.all(result[0, 0] == x[0, 0])  # First channel, first frame.
    assert torch.all(result[1, 2] == x[2, 1])  # Third channel, second frame.


def test_as_image_batch_tensor_with_dtype():
    """Test as_image_batch_tensor respects dtype parameter."""
    x = torch.zeros(3, 1, 224, 224, dtype=torch.uint8)
    result = as_image_batch_tensor(x, dtype=torch.float32)
    assert result.dtype == torch.float32
    assert result.shape == (1, 3, 224, 224)


def test_as_image_batch_tensor_with_device():
    """Test as_image_batch_tensor respects device parameter."""
    x = torch.zeros(3, 1, 224, 224)
    result = as_image_batch_tensor(x, device="cpu")
    assert result.device.type == "cpu"
    assert result.shape == (1, 3, 224, 224)


def test_as_image_batch_tensor_too_few_dims_raises():
    """Test as_image_batch_tensor raises on tensors with < 4 dimensions."""
    x = torch.zeros(3, 224, 224)  # Only 3D.
    with pytest.raises(ValueError, match="Too few dimensions"):
        as_image_batch_tensor(x)


def test_preprocess_images_area_ignores_align_corners():
    """align_corners must be dropped for area mode (PyTorch rejects it)."""
    x = torch.rand(1, 3, 8, 8)
    # Area does not support align_corners; passing it must not raise.
    out = preprocess_images(x, size=(4, 4), mode="area", align_corners=True)
    assert out.shape == (1, 3, 4, 4)


def test_preprocess_images_nearest_ignores_align_corners():
    """align_corners must be dropped for nearest mode."""
    x = torch.rand(1, 3, 8, 8)
    out = preprocess_images(x, size=(4, 4), mode="nearest", align_corners=True)
    assert out.shape == (1, 3, 4, 4)


def test_preprocess_images_hybrid_downsample_ignores_align_corners():
    """Hybrid downsampling resolves to area, so align_corners must be dropped."""
    x = torch.rand(1, 3, 16, 16)
    out = preprocess_images(x, size=(4, 4), mode="hybrid", align_corners=True)
    assert out.shape == (1, 3, 4, 4)


def test_preprocess_images_bilinear_uses_align_corners():
    """align_corners must still take effect for bilinear (different results)."""
    x = torch.rand(1, 3, 8, 8)
    out_true = preprocess_images(x, size=(4, 4), mode="bilinear", align_corners=True)
    out_false = preprocess_images(x, size=(4, 4), mode="bilinear", align_corners=False)
    assert not torch.allclose(out_true, out_false)


@pytest.mark.parametrize(
    ("total", "wanted", "expected"),
    [
        (10, 0, []),
        (10, 1, [0]),
        (10, 2, [0, 9]),
        (10, 3, [0, 4, 9]),
        (10, 5, [0, 2, 4, 6, 9]),
        (3, 5, [0, 1, 2]),
        (2, 3, [0, 1]),
    ],
)
def test_progressive_bisection_adds_endpoints_then_splits_the_widest_gap(
    total: int,
    wanted: int,
    expected: list[int],
) -> None:
    assert compute_keyframes_as_progressive_bisection(total, wanted) == expected


def test_preprocess_images_rejects_integer_and_non_nchw_inputs() -> None:
    with pytest.raises(TypeError, match="float type"):
        preprocess_images(torch.zeros(1, 3, 4, 4, dtype=torch.uint8), size=(2, 2))
    with pytest.raises(TypeError, match="NCHW"):
        preprocess_images(torch.zeros(3, 4, 4), size=(2, 2))


def test_preprocess_images_lanczos_resizes_through_pil_and_rejects_align_corners():
    x = torch.linspace(-1, 1, 3 * 8 * 8).view(1, 3, 8, 8)
    out = preprocess_images(x, size=(4, 4), mode="lanczos", dtype=torch.float64)
    assert out.shape == (1, 3, 4, 4)
    assert out.dtype == torch.float64
    assert out.min() >= -1
    assert out.max() <= 1
    with pytest.raises(ValueError, match="align_corners is not supported"):
        preprocess_images(x, size=(4, 4), mode="lanczos", align_corners=False)


def test_preprocess_images_casts_when_no_resize_is_needed_and_normalizes() -> None:
    x = torch.full((1, 2, 4, 4), 0.5)
    out = preprocess_images(
        x,
        size=(4, 4),
        mean=[0.5, 0.25],
        std=[0.5, 0.25],
        dtype=torch.float64,
    )
    assert out.dtype == torch.float64
    assert torch.equal(out[0, 0], torch.zeros(4, 4, dtype=torch.float64))
    assert torch.equal(out[0, 1], torch.ones(4, 4, dtype=torch.float64))


def test_image_batch_to_pil_list_rejects_bad_inputs_and_maps_the_range() -> None:
    with pytest.raises(TypeError, match="float type"):
        image_batch_to_pil_list(torch.zeros(1, 3, 2, 2, dtype=torch.uint8))
    with pytest.raises(TypeError, match="NCHW"):
        image_batch_to_pil_list(torch.zeros(3, 2, 2))
    images = image_batch_to_pil_list(torch.tensor([[[[-1.0, 1.0]]]]).expand(2, 3, 1, 2))
    assert len(images) == 2
    assert images[0].getpixel((0, 0)) == (0, 0, 0)
    assert images[0].getpixel((1, 0)) == (255, 255, 255)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
