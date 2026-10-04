"""Tests for the batched crop-decode-resize stage."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from turbojpeg import TJFLAG_FASTDCT, TJPF_RGB, TJSAMP_444, TurboJPEG

import cv2
import numpy as np
import pytest
import torch

from priml.data.processors.decode_batch import DecodeCropResizeBatch, _drop_failed


if TYPE_CHECKING:
    from collections.abc import Iterator


Crop = tuple[int, int, int, int]


def _jpeg(*, height: int, width: int, seed: int = 0) -> bytes:
    """Encode noise as a 4:4:4 JPEG, so crop edges carry no chroma bleed."""
    rgb = np.random.default_rng(seed).integers(0, 256, (height, width, 3), np.uint8)
    return TurboJPEG().encode(rgb, pixel_format=TJPF_RGB, jpeg_subsample=TJSAMP_444)


def _media(count: int) -> list[bytes]:
    return [_jpeg(height=40, width=64, seed=i) for i in range(count)]


def _expected(data: bytes, crop: Crop, size: int) -> torch.Tensor:
    """Full decode, slice, INTER_AREA: what one image must become, as (3, H, W)."""
    y, x, h, w = crop
    rgb = TurboJPEG().decode(data, pixel_format=TJPF_RGB)[y : y + h, x : x + w]
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(resized).permute(2, 0, 1)


def _batch(crops: list[Crop], *, size: int = 8) -> dict[str, object]:
    """Build a batch as ``Batcher`` emits one: per-sample fields gathered into lists."""
    return {
        "media": _media(len(crops)),
        "crop": crops,
        "target_height": [size] * len(crops),
        "target_width": [size] * len(crops),
        "label": torch.arange(len(crops)),
        "_batch_size": len(crops),
        "_batched_list_fields": ["media", "crop", "target_height", "target_width"],
    }


def _run(
    config: DecodeCropResizeBatch.Config,
    batch: dict[str, object],
) -> dict[str, object]:
    return next(iter(config.make()(iter([batch]))))


def _tensor(batch: dict[str, object], key: str) -> torch.Tensor:
    value = batch[key]
    assert isinstance(value, torch.Tensor)
    return value


def test_every_image_is_its_crop_resized_into_one_uint8_tensor() -> None:
    crops: list[Crop] = [(0, 0, 40, 64), (5, 13, 17, 29), (10, 34, 30, 30)]
    image = _tensor(_run(DecodeCropResizeBatch.Config(), _batch(crops)), "image")
    assert image.shape == (3, 3, 8, 8)
    assert image.dtype == torch.uint8
    for index, (data, crop) in enumerate(zip(_media(3), crops, strict=True)):
        assert torch.equal(image[index], _expected(data, crop, 8))


def test_the_bytes_and_boxes_are_consumed() -> None:
    """Only the image and the untouched fields leave; the inputs would pin memory."""
    allocated: list[torch.Tensor] = []
    original_empty = torch.empty

    def allocate(size: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
        tensor = original_empty(size, dtype=dtype)
        allocated.append(tensor)
        return tensor

    with patch(
        "priml.data.processors.decode_batch.torch.empty",
        side_effect=allocate,
    ):
        out = _run(DecodeCropResizeBatch.Config(), _batch([(0, 0, 8, 8)]))

    image = _tensor(out, "image")
    assert (
        image.untyped_storage().data_ptr() == allocated[0].untyped_storage().data_ptr()
    )
    assert "media" not in out
    assert "crop" not in out
    assert torch.equal(_tensor(out, "label"), torch.arange(1))


def test_drop_failed_filters_list_fields_and_aligned_tensors() -> None:
    reason = "DecodeCropResizeBatch:decode_failed"
    names = ["failed", "kept"]
    unlisted = ["untouched", "list"]
    aligned = torch.tensor([10, 20])
    unaligned = torch.tensor([30])
    batch: dict[str, object] = {
        "_batched_list_fields": ["names"],
        "names": names,
        "unlisted": unlisted,
        "aligned": aligned,
        "unaligned": unaligned,
        "_batch_size": 2,
        "_filter_counts": {reason: 3},
    }

    _drop_failed(batch, keep=[1], size=2)

    assert batch["names"] == ["kept"]
    assert batch["unlisted"] is unlisted
    assert torch.equal(_tensor(batch, "aligned"), torch.tensor([20]))
    assert batch["unaligned"] is unaligned
    assert batch["_batch_size"] == 1
    assert batch["_filter_counts"] == {reason: 4}


def test_drop_failed_defaults_to_no_list_fields() -> None:
    aligned = torch.tensor([10, 20])
    unlisted = ["failed", "kept"]
    batch: dict[str, object] = {"unlisted": unlisted, "aligned": aligned}

    _drop_failed(batch, keep=[1], size=2)

    assert batch["unlisted"] is unlisted
    assert torch.equal(_tensor(batch, "aligned"), torch.tensor([20]))
    assert batch["_batch_size"] == 1
    assert batch["_filter_counts"] == {"DecodeCropResizeBatch:decode_failed": 1}


def test_flip_mirrors_each_image_with_probability_p() -> None:
    crops: list[Crop] = [(0, 0, 40, 64)] * 2
    always = DecodeCropResizeBatch.Config()
    always.flip_p = 1.0
    plain = _tensor(_run(DecodeCropResizeBatch.Config(), _batch(crops)), "image")
    with patch(
        "priml.data.processors.decode_batch.cv2.flip",
        wraps=cv2.flip,
    ) as flip:
        flipped = _tensor(_run(always, _batch(crops)), "image")

    assert flip.call_args is not None
    assert flip.call_args.args[1] == 1
    assert torch.equal(flipped, plain.flip(-1))


def test_zero_random_draw_does_not_flip_at_zero_probability() -> None:
    crop: Crop = (0, 0, 40, 64)
    with patch(
        "priml.data.processors.decode_batch.random.random",
        return_value=0.0,
    ):
        image = _tensor(
            _run(DecodeCropResizeBatch.Config(flip_p=0.0), _batch([crop])),
            "image",
        )

    assert torch.equal(image[0], _expected(_media(1)[0], crop, 8))


def test_an_upsampled_crop_is_opencv_area_too() -> None:
    """A crop smaller than the target is enlarged with the same kernel."""
    out = _run(DecodeCropResizeBatch.Config(), _batch([(0, 0, 4, 4)], size=16))
    assert torch.equal(
        _tensor(out, "image")[0],
        _expected(_media(1)[0], (0, 0, 4, 4), 16),
    )


def test_fast_dct_decodes_with_libjpeg_turbos_fast_idct() -> None:
    crop: Crop = (5, 13, 17, 29)
    config = DecodeCropResizeBatch.Config()
    config.fast_dct = True
    image = _tensor(_run(config, _batch([crop])), "image")
    y, x, h, w = crop
    rgb = TurboJPEG().decode(_media(1)[0], pixel_format=TJPF_RGB, flags=TJFLAG_FASTDCT)
    resized = cv2.resize(
        rgb[y : y + h, x : x + w],
        (8, 8),
        interpolation=cv2.INTER_AREA,
    )
    assert torch.equal(image[0], torch.from_numpy(resized).permute(2, 0, 1))


def test_decode_uses_target_sized_idct_and_resizes_non_square_output() -> None:
    processor = DecodeCropResizeBatch.Config(scale_to_target=True, fast_dct=True).make()
    region = np.arange(20 * 30 * 3, dtype=np.uint8).reshape(20, 30, 3)
    out = np.empty((5, 9, 3), dtype=np.uint8)

    with (
        patch(
            "priml.data.processors.decode_batch.decode_jpeg_turbojpeg_region",
            return_value=region,
        ) as decode,
        patch(
            "priml.data.processors.decode_batch.cv2.resize",
            wraps=cv2.resize,
        ) as resize,
    ):
        assert processor._decode_one(b"jpeg", (2, 3, 18, 26), out, False)

    decode.assert_called_once_with(
        b"jpeg",
        x=3,
        y=2,
        w=26,
        h=18,
        min_height=5,
        min_width=9,
        fast_dct=True,
    )
    resize.assert_called_once()
    assert resize.call_args is not None
    assert resize.call_args.args[1] == (9, 5)
    expected = cv2.resize(region, (9, 5), interpolation=cv2.INTER_AREA)
    assert np.array_equal(out, expected)


def test_unscaled_decode_does_not_request_idct_size_floors() -> None:
    processor = DecodeCropResizeBatch.Config().make()
    region = np.zeros((10, 14, 3), dtype=np.uint8)
    out = np.empty((5, 9, 3), dtype=np.uint8)

    with patch(
        "priml.data.processors.decode_batch.decode_jpeg_turbojpeg_region",
        return_value=region,
    ) as decode:
        assert processor._decode_one(b"jpeg", (1, 2, 8, 12), out, False)

    decode.assert_called_once_with(
        b"jpeg",
        x=2,
        y=1,
        w=12,
        h=8,
        min_height=0,
        min_width=0,
        fast_dct=False,
    )


def test_threads_produce_the_same_batch_as_one() -> None:
    crops: list[Crop] = [(i % 7, i % 11, 20, 30) for i in range(12)]
    one = DecodeCropResizeBatch.Config()
    one.num_threads = 1
    four = DecodeCropResizeBatch.Config()
    four.num_threads = 4
    assert torch.equal(
        _tensor(_run(one, _batch(crops)), "image"),
        _tensor(_run(four, _batch(crops)), "image"),
    )


def test_an_undecodable_image_is_dropped_and_counted() -> None:
    """A corrupt JPEG costs its own sample, not the batch, and is reported."""
    batch = _batch([(0, 0, 8, 8), (0, 0, 8, 8)])
    batch["media"] = [_media(1)[0], b"not a jpeg"]
    out = _run(DecodeCropResizeBatch.Config(), batch)
    assert _tensor(out, "image").shape[0] == 1
    assert torch.equal(_tensor(out, "label"), torch.arange(1))
    assert out["_batch_size"] == 1
    assert out["_filter_counts"] == {"DecodeCropResizeBatch:decode_failed": 1}


def test_a_sample_that_is_not_a_batch_passes_through() -> None:
    sample: dict[str, object] = {"key": "k"}
    assert _run(DecodeCropResizeBatch.Config(), sample) is sample


def test_a_partial_decode_batch_passes_through() -> None:
    media_only: dict[str, object] = {"media": [b"jpeg"]}
    crop_only: dict[str, object] = {"crop": [(0, 0, 8, 8)]}

    processor = DecodeCropResizeBatch.Config().make()
    results = list(processor(iter([media_only, crop_only])))

    assert results == [media_only, crop_only]
    assert results[0] is media_only
    assert results[1] is crop_only


def test_batches_decoded_in_flight_leave_in_arrival_order() -> None:
    """Batch N decodes while N+1 is pulled; the stream order must not change."""
    first, second = _batch([(0, 0, 40, 64)]), _batch([(5, 13, 17, 29)])
    passthrough: dict[str, object] = {"key": "k"}
    stage = DecodeCropResizeBatch.Config().make()
    out = list(stage(iter([first, passthrough, second])))
    assert [out[0] is first, out[1] is passthrough, out[2] is second] == [True] * 3
    media = _media(1)[0]
    assert torch.equal(_tensor(first, "image")[0], _expected(media, (0, 0, 40, 64), 8))
    assert torch.equal(
        _tensor(second, "image")[0],
        _expected(media, (5, 13, 17, 29), 8),
    )


def test_the_next_batch_is_pulled_before_the_current_one_is_yielded() -> None:
    """Pulling upstream while the pool decodes is what hides upstream's cost."""
    pulled: list[int] = []

    def upstream() -> Iterator[dict[str, object]]:
        for index in range(3):
            pulled.append(index)
            yield _batch([(0, 0, 8, 8)])

    stream = DecodeCropResizeBatch.Config().make()(upstream())
    _ = next(stream)
    assert pulled == [0, 1]


def test_a_shut_pool_ends_the_stream_instead_of_raising() -> None:
    """At interpreter exit the pool refuses work while a prefetch thread pulls."""
    stage = DecodeCropResizeBatch.Config().make()
    stage.pool.shutdown()
    assert list(stage(iter([_batch([(0, 0, 8, 8)])]))) == []


def test_configured_thread_count_sizes_the_decode_pool() -> None:
    with patch("priml.data.processors.decode_batch.ThreadPoolExecutor") as pool:
        _ = DecodeCropResizeBatch.Config(num_threads=3).make()

    pool.assert_called_once_with(3)


@pytest.mark.parametrize("threads", [0, -1])
def test_rejects_a_nonpositive_thread_count(threads: int) -> None:
    config = DecodeCropResizeBatch.Config()
    config.num_threads = threads
    with pytest.raises(ValueError, match="num_threads must be positive"):
        _ = config.make()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
