"""Tests for ``DecodeVideo``: real decode, real resize, real tar plumbing.

The decoder is exercised against an mp4 a module-scoped fixture encodes once
per session rather than a mock. A mocked decoder would pass while the ffmpeg
filter string was wrong, which is the only part with any risk in it -- the
contract
(``(C, F, H, W)`` float16 in ``[-1, 1]``, resized DURING decode) is enforced by
ffmpeg's behavior, not by our arithmetic.

Colors are chosen so the assertions can distinguish channel order: a decoder
that returns BGR instead of RGB, or that transposes H and W, fails loudly
instead of merely producing differently-shaped noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import io
import os
import subprocess
import tarfile
import tempfile

from PIL import Image

import imageio_ffmpeg
import pytest
import torch

from priml.data.processors.bytes import (
    CropDuringDecodeImage,
    DecodeVideo,
    GetBytesFromFile,
    GetBytesFromTarHandle,
    GetDimensionsFromBytes,
)
from priml.data.sources.tarhandle import TarFileHandle


if TYPE_CHECKING:
    from collections.abc import Iterator


_FRAMES = 8
_SOURCE_HEIGHT = 64
_SOURCE_WIDTH = 128


def _encode_video(path: Path, *, height: int, width: int, frames: int) -> None:
    """Write a small H.264 mp4 whose left half is red and right half is blue."""
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    # `lavfi` synthesises the clip inside ffmpeg, so the fixture needs no
    # checked-in binary and no numpy round-trip.
    filtergraph = (
        f"color=c=red:s={width // 2}x{height}:d=1[l];"
        f"color=c=blue:s={width // 2}x{height}:d=1[r];"
        f"[l][r]hstack=inputs=2"
    )
    _ = subprocess.run(  # noqa: S603 -- argv is built from literals and ints.
        [
            exe,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            filtergraph,
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="module")
def video_tar(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a tar containing one small mp4 under the key ``clip``."""
    root = tmp_path_factory.mktemp("decode-video")
    mp4 = root / "clip.mp4"
    _encode_video(mp4, height=_SOURCE_HEIGHT, width=_SOURCE_WIDTH, frames=_FRAMES)
    archive = root / "shard.tar"
    with tarfile.open(archive, "w") as tar:
        tar.add(mp4, arcname="clip.mp4")
    return archive


def _as_stream(items: list[DecodeVideo.Input]) -> Iterator[DecodeVideo.Input]:
    """Yield the samples; the processor consumes an iterator, not a list."""
    yield from items


def _as_stream_bytes(
    items: list[GetBytesFromTarHandle.Input],
) -> Iterator[GetBytesFromTarHandle.Input]:
    """Yield the samples; the processor consumes an iterator, not a list."""
    yield from items


def _run(archive: Path, **overrides: object) -> list[DecodeVideo.Output]:
    """Drive ``DecodeVideo`` over one sample read from ``archive``."""
    processor = DecodeVideo(DecodeVideo.Config())
    with tarfile.open(archive) as tar:
        sample = cast(
            DecodeVideo.Input,
            {
                "_tar_handle": tar,
                "key": "clip",
                "format": "mp4",
                "frames": _FRAMES,
                "height": _SOURCE_HEIGHT,
                "width": _SOURCE_WIDTH,
                **overrides,
            },
        )
        return list(processor(_as_stream([sample])))


@pytest.mark.cli_python_subprocess
def test_decode_video_emits_the_contract_tensor(video_tar: Path) -> None:
    """A decoded video is (C, F, H, W) float16 normalized to [-1, 1]."""
    out = _run(video_tar, target_frames=_FRAMES, target_height=32, target_width=32)

    assert len(out) == 1
    tensor = out[0].get("media_tensor")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float16
    assert tensor.shape == (3, _FRAMES, 32, 32)
    assert float(tensor.min()) >= -1.0
    assert float(tensor.max()) <= 1.0


@pytest.mark.cli_python_subprocess
def test_decode_video_resizes_to_the_requested_size(video_tar: Path) -> None:
    """The emitted spatial size is the REQUESTED one, not the source's.

    This is what proves the resize happened at all. It is a separate test from
    the contract check because a decoder that ignored the target and returned
    source-sized frames would still satisfy dtype and range.
    """
    out = _run(video_tar, target_frames=_FRAMES, target_height=16, target_width=48)

    tensor = out[0].get("media_tensor")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, _FRAMES, 16, 48)
    assert (_SOURCE_HEIGHT, _SOURCE_WIDTH) != (16, 48)


@pytest.mark.cli_python_subprocess
def test_decode_video_preserves_rgb_channel_order(video_tar: Path) -> None:
    """The left half decodes red and the right half blue, in RGB order.

    ffmpeg and OpenCV disagree on channel order, so an implementation that
    picked up a BGR path would silently swap them. Comparing the two halves
    catches that; comparing only shapes would not.
    """
    out = _run(video_tar, target_frames=_FRAMES, target_height=32, target_width=32)
    tensor = out[0].get("media_tensor")
    assert isinstance(tensor, torch.Tensor)

    left = tensor[:, 0, :, :8].mean(dim=(1, 2))
    right = tensor[:, 0, :, -8:].mean(dim=(1, 2))

    assert float(left[0]) > float(left[2]), f"left half should be red, got {left}"
    assert float(right[2]) > float(right[0]), f"right half should be blue, got {right}"


@pytest.mark.cli_python_subprocess
def test_decode_video_without_targets_keeps_source_size(video_tar: Path) -> None:
    """Omitting the target fields decodes at source resolution."""
    out = _run(video_tar)

    tensor = out[0].get("media_tensor")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, _FRAMES, _SOURCE_HEIGHT, _SOURCE_WIDTH)


@pytest.mark.cli_python_subprocess
def test_decode_video_subsamples_frames(video_tar: Path) -> None:
    """A smaller ``target_frames`` yields exactly that many frames."""
    out = _run(video_tar, target_frames=4, target_height=32, target_width=32)

    tensor = out[0].get("media_tensor")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[1] == 4


@pytest.mark.cli_python_subprocess
def test_decode_video_pads_a_short_clip_to_target_frames(video_tar: Path) -> None:
    """A clip shorter than ``target_frames`` is padded up to it.

    ``CalcResizeDimensions`` rounds the frame count up to the temporal
    compression multiple so a latent encoder can stride it evenly. Treating the
    field as a ceiling instead silently defeats that: the planner asks for 32
    and the encoder still receives 30. The last frame repeats rather than
    zero-filling, because a black tail is content the clip never had.
    """
    out = _run(video_tar, target_frames=_FRAMES + 4, target_height=32, target_width=32)

    tensor = out[0].get("media_tensor")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape[1] == _FRAMES + 4
    # The padding repeats the final decoded frame.
    torch.testing.assert_close(tensor[:, -1], tensor[:, _FRAMES - 1])


@pytest.mark.cli_python_subprocess
def test_decode_video_pads_from_a_source_of_exactly_one_frame(video_tar: Path) -> None:
    """The padding arithmetic must survive its smallest input.

    ``_decode`` breaks out of the read loop as soon as ``len(chunks) >=
    keep_frames``, so asking for one frame collects exactly one, and the
    ``chunks.extend([chunks[-1]] * (keep_frames - len(chunks)))`` that follows
    multiplies by zero. A source shorter than the target and a target of one
    are the two ends of that expression; only the long end was covered.
    """
    out = _run(video_tar, target_frames=1, target_height=16, target_width=16)

    tensor = out[0].get("media_tensor")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 1, 16, 16)
    assert torch.isfinite(tensor).all()


@pytest.mark.cli_python_subprocess
def test_decode_video_emits_the_signed_range_from_a_saturated_source(
    video_tar: Path,
) -> None:
    """Pure red and pure blue must reach both ends of [-1, 1], not one.

    The fixture's halves are saturated, so a correct decode puts each channel
    at both extremes. An unscaled decode would land in [0, 255] and a
    double-scaled one near -1 everywhere; both are invisible to a shape or
    dtype assertion, and the range is this class's stated contract.
    """
    out = _run(video_tar, target_frames=2, target_height=16, target_width=16)

    tensor = out[0].get("media_tensor")
    assert isinstance(tensor, torch.Tensor)
    assert float(tensor.min()) == pytest.approx(-1.0, abs=0.05)
    assert float(tensor.max()) == pytest.approx(1.0, abs=0.05)


def test_decode_video_passes_through_single_frame_samples(video_tar: Path) -> None:
    """A sample with frames <= 1 is not a video and is yielded untouched."""
    out = _run(video_tar, frames=1)

    assert len(out) == 1
    assert "media_tensor" not in out[0]


def test_decode_video_passes_through_partial_targets(video_tar: Path) -> None:
    """Target fields are all-or-nothing; a partial set is not a resize request."""
    out = _run(video_tar, target_height=32)

    assert len(out) == 1
    assert "media_tensor" not in out[0]


def test_decode_video_yields_unchanged_when_media_tensor_exists(
    video_tar: Path,
) -> None:
    """An already-decoded sample is not decoded twice."""
    marker = torch.zeros(1)
    out = _run(video_tar, media_tensor=marker)

    assert "media_tensor" in out[0]
    assert out[0]["media_tensor"] is marker


@pytest.mark.cli_python_subprocess
def test_decode_video_reports_missing_key_without_raising(video_tar: Path) -> None:
    """A key absent from the tar yields the sample rather than exploding."""
    out = _run(
        video_tar,
        key="does-not-exist",
        target_frames=_FRAMES,
        target_height=32,
        target_width=32,
    )

    assert len(out) == 1
    assert "media_tensor" not in out[0]


@pytest.mark.cli_python_subprocess
def test_encoded_fixture_is_readable() -> None:
    """The fixture encoder produces a file ffmpeg can read back.

    Guards the test infrastructure itself: if lavfi or libx264 were missing,
    every other test here would fail for a reason unrelated to DecodeVideo.
    """
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "probe.mp4"
        _encode_video(path, height=32, width=32, frames=4)
        reader = imageio_ffmpeg.read_frames(str(path))
        meta = next(reader)
        # Metadata is yielded first and frame bytes after it.
        assert isinstance(meta, dict)
        assert meta["size"] == (32, 32)
        assert sum(1 for _ in reader) == 4


@pytest.mark.cli_python_subprocess
def test_decode_video_media_bytes_roundtrip(video_tar: Path) -> None:
    """Bytes already in the sample are decoded without touching the tar."""
    with tarfile.open(video_tar) as tar:
        member = tar.getmember("clip.mp4")
        extracted = tar.extractfile(member)
        assert extracted is not None
        payload = extracted.read()

    processor = DecodeVideo(DecodeVideo.Config())
    sample: DecodeVideo.Input = {
        "media": payload,
        "key": "clip",
        "format": "mp4",
        "frames": _FRAMES,
        "height": _SOURCE_HEIGHT,
        "width": _SOURCE_WIDTH,
        "target_frames": _FRAMES,
        "target_height": 32,
        "target_width": 32,
    }
    out = list(processor(_as_stream([sample])))

    tensor = out[0].get("media_tensor")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, _FRAMES, 32, 32)


@pytest.mark.cli_python_subprocess
def test_decode_video_filters_a_non_positive_target(video_tar: Path) -> None:
    """A zero target is filtered rather than quietly yielding one frame.

    The decode loop stops once ``len(chunks) >= keep_frames``, which already
    holds after the first frame, so ``target_frames=0`` produced a one-frame
    tensor -- neither the empty clip asked for nor an error.
    """
    out = _run(video_tar, target_frames=0, target_height=32, target_width=32)

    assert "media_tensor" not in out[0]
    reasons = cast(dict[str, object], out[0]).get("filter_reasons", [])
    assert any("invalid_target" in str(r) for r in cast(list[object], reasons))


@pytest.mark.cli_python_subprocess
def test_decode_video_refuses_a_format_outside_known_formats(video_tar: Path) -> None:
    """A named format still has to be one the config claims to decode.

    ``_read_media`` built ``[fmt]`` from the sample without checking it, so
    ``known_formats`` was inert for every sample carrying a format -- which is
    the common case, not the corner one.
    """
    processor = DecodeVideo(DecodeVideo.Config(known_formats=["webm"]))
    with tarfile.open(video_tar) as tar:
        sample = cast(
            DecodeVideo.Input,
            {
                "_tar_handle": tar,
                "key": "clip",
                "format": "mp4",
                "frames": _FRAMES,
                "height": _SOURCE_HEIGHT,
                "width": _SOURCE_WIDTH,
            },
        )
        out = list(processor(_as_stream([sample])))

    assert "media_tensor" not in out[0]


@pytest.mark.cli_python_subprocess
def test_decode_video_checks_the_format_even_when_the_bytes_are_in_hand(
    video_tar: Path,
) -> None:
    """The allowlist must gate the decode, not merely the tar read.

    ``payload = media or self._read_media(...)`` short-circuits, so the check
    inside ``_read_media`` never ran for a sample an upstream processor had
    already read -- and reading the bytes upstream is the normal pipeline.
    """
    with tarfile.open(video_tar) as tar:
        extracted = tar.extractfile("clip.mp4")
        assert extracted is not None
        payload = extracted.read()

    processor = DecodeVideo(DecodeVideo.Config(known_formats=["webm"]))
    sample = cast(
        DecodeVideo.Input,
        {
            "key": "clip",
            "format": "mp4",
            "media": payload,
            "frames": _FRAMES,
            "height": _SOURCE_HEIGHT,
            "width": _SOURCE_WIDTH,
        },
    )
    out = list(processor(_as_stream([sample])))

    assert "media_tensor" not in out[0]


@pytest.mark.cli_python_subprocess
def test_decode_video_accepts_bytes_without_a_tar_key(video_tar: Path) -> None:
    """A key addresses a tar member; bytes already in hand need no address."""
    with tarfile.open(video_tar) as tar:
        extracted = tar.extractfile("clip.mp4")
        assert extracted is not None
        payload = extracted.read()

    sample = cast(
        DecodeVideo.Input,
        {
            "format": "mp4",
            "media": payload,
            "frames": _FRAMES,
            "height": _SOURCE_HEIGHT,
            "width": _SOURCE_WIDTH,
        },
    )
    out = list(DecodeVideo(DecodeVideo.Config())(_as_stream([sample])))

    assert "media_tensor" in out[0]
    assert out[0]["media_tensor"].shape == (3, _FRAMES, _SOURCE_HEIGHT, _SOURCE_WIDTH)


@pytest.mark.cli_python_subprocess
def test_decode_video_executable_is_instance_local(
    video_tar: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One decoder's executable must not rewrite process-global configuration."""
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    ambient = "/not/the/configured/ffmpeg"
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", ambient)
    processor = DecodeVideo(DecodeVideo.Config(ffmpeg_exe=executable))

    with tarfile.open(video_tar) as tar:
        extracted = tar.extractfile("clip.mp4")
        assert extracted is not None
        sample = cast(
            DecodeVideo.Input,
            {
                "format": "mp4",
                "media": extracted.read(),
                "frames": _FRAMES,
                "height": _SOURCE_HEIGHT,
                "width": _SOURCE_WIDTH,
            },
        )
    out = list(processor(_as_stream([sample])))

    assert os.environ["IMAGEIO_FFMPEG_EXE"] == ambient
    assert "media_tensor" in out[0]


def test_the_tar_handle_is_dropped_after_this_reader_finishes(tmp_path: Path) -> None:
    """Completed and terminal reads release the archive handle."""
    archive = tmp_path / "shard.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo(name="k0.jpg")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))

    processor = GetBytesFromTarHandle(GetBytesFromTarHandle.Config())
    with TarFileHandle(archive, use_mmap=False) as handle:
        samples = [
            {"_tar_handle": handle, "key": "k0", "media": b"already"},
            {"_tar_handle": handle, "key": "absent", "format": "jpg"},
            {"_tar_handle": handle, "key": "k0", "format": "jpg"},
        ]
        out = list(
            processor(
                _as_stream_bytes(
                    [cast(GetBytesFromTarHandle.Input, s) for s in samples],
                ),
            ),
        )

    assert all("_tar_handle" not in sample for sample in out)
    assert out[-1].get("media") == b"abc"


def test_a_rerouted_sample_keeps_the_handle_for_its_sibling(tmp_path: Path) -> None:
    """Routing cannot discard the only object a sibling reader can consume."""
    archive = tmp_path / "shard.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo(name="k0.tiff")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))

    first = GetBytesFromTarHandle(
        GetBytesFromTarHandle.Config(known_formats=["jpg"]),
    )
    second = GetBytesFromTarHandle(
        GetBytesFromTarHandle.Config(known_formats=["tiff"]),
    )
    with TarFileHandle(archive, use_mmap=False) as handle:
        sample = cast(
            GetBytesFromTarHandle.Input,
            {"_tar_handle": handle, "key": "k0", "format": "tiff"},
        )
        routed = list(first(_as_stream_bytes([sample])))
        assert routed[0].get("_tar_handle") is handle
        out = list(
            second(
                _as_stream_bytes(
                    [cast(GetBytesFromTarHandle.Input, routed[0])],
                ),
            ),
        )

    assert out[0].get("media") == b"abc"
    assert "_tar_handle" not in out[0]


def test_an_unreadable_sample_records_why_but_a_rerouted_one_does_not(
    tmp_path: Path,
) -> None:
    """Only a sample NO reader can rescue counts as filtered.

    A missing handle or key ends the sample everywhere, so it must appear in
    the filtered totals -- it returned silently, which made it
    indistinguishable from data that was never there. A format outside this
    reader's allowlist is the opposite case: the sample is intact and a
    sibling reader still claims it, so reporting it would count a live sample
    as dropped.
    """
    archive = tmp_path / "shard.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo(name="k0.jpg")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))

    processor = GetBytesFromTarHandle(GetBytesFromTarHandle.Config())
    with TarFileHandle(archive, use_mmap=False) as handle:
        samples = [
            {"_tar_handle": handle, "format": "jpg"},
            {"key": "k0", "format": "jpg"},
            {"_tar_handle": handle, "key": "k0", "format": "tiff"},
        ]
        out = list(
            processor(
                _as_stream_bytes(
                    [cast(GetBytesFromTarHandle.Input, s) for s in samples],
                ),
            ),
        )

    assert all("media" not in sample for sample in out)
    unreadable, rerouted = out[:2], out[2]
    for sample in unreadable:
        reasons = cast(dict[str, object], sample).get("filter_reasons")
        assert reasons, f"unreadable sample recorded no reason: {sample}"
    assert not cast(dict[str, object], rerouted).get("filter_reasons")
    assert "_tar_handle" in rerouted


def test_a_float_image_tensor_lands_in_the_same_range_as_a_video(
    video_tar: Path,
) -> None:
    """Both decoders must emit the same range, because one consumer reads both.

    ``DecodeVideo`` emits float16 in ``[-1, 1]``; ``CropDuringDecodeImage`` with
    a float ``dtype`` cast the raw uint8 straight across and emitted ``[0, 255]``.
    ``Interpolate`` rescales only when it sees ``uint8`` (``resize.py:104``), so
    the float16 image passed through untouched and trained at 100x scale.
    """
    del video_tar
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="JPEG")
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(dtype=torch.float16),
    )
    sample = cast(
        CropDuringDecodeImage.Input,
        {"media": buffer.getvalue(), "format": "jpg", "height": 8, "width": 8},
    )
    out = list(processor(iter([sample])))

    tensor = out[0].get("media_tensor")
    assert tensor is not None
    assert tensor.dtype == torch.float16
    assert float(tensor.min()) >= -1.0
    assert float(tensor.max()) <= 1.0


def test_an_integer_image_tensor_keeps_the_raw_byte_range(video_tar: Path) -> None:
    """The default (no ``dtype``) still hands on uint8 in ``[0, 255]``.

    Normalizing on the float path must not silently rescale the integer one,
    which ``Interpolate`` converts itself.
    """
    del video_tar
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="JPEG")
    processor = CropDuringDecodeImage(CropDuringDecodeImage.Config())
    sample = cast(
        CropDuringDecodeImage.Input,
        {"media": buffer.getvalue(), "format": "jpg", "height": 8, "width": 8},
    )
    out = list(processor(iter([sample])))

    tensor = out[0].get("media_tensor")
    assert tensor is not None
    assert tensor.dtype == torch.uint8
    assert int(tensor.max()) > 1


@pytest.mark.parametrize(
    ("scale_to_target", "expected_hw"),
    [(False, (64, 64)), (True, (16, 16))],
)
def test_scale_to_target_decodes_no_larger_than_the_target_needs(
    *,
    scale_to_target: bool,
    expected_hw: tuple[int, int],
) -> None:
    """A 64px crop bound for 12px decodes at 1/4 (16px), never below 12px."""
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (255, 0, 0)).save(buffer, format="JPEG")
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(scale_to_target=scale_to_target),
    )
    sample = cast(
        CropDuringDecodeImage.Input,
        {
            "media": buffer.getvalue(),
            "format": "jpg",
            "height": 64,
            "width": 64,
            "crop": (0, 0, 64, 64),
            "target_height": 12,
            "target_width": 12,
        },
    )
    tensor = next(processor(iter([sample]))).get("media_tensor")
    assert tensor is not None
    assert tuple(tensor.shape[-2:]) == expected_hw


def test_stale_tar_offsets_never_substitute_for_reading_the_member(
    tmp_path: Path,
) -> None:
    """Offset metadata on a sample must not be trusted to address the archive.

    ``member.offset_data`` is an offset into the DECOMPRESSED stream, so for a
    compressed archive it addresses nothing in the file on disk. Reading it
    there returned a full-length block of unrelated bytes -- measured on a
    ten-member ``.tar.gz`` -- and, being full-length, it passed every check and
    was yielded as the member's content.
    """
    archive = tmp_path / "shard.tar.gz"
    payloads = {f"k{i}": os.urandom(100_000) for i in range(10)}
    with tarfile.open(archive, "w:gz") as tar:
        for key, payload in payloads.items():
            info = tarfile.TarInfo(name=f"{key}.mp4")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    with tarfile.open(archive) as tar:
        member = tar.getmember("k0.mp4")
        offset, size = member.offset_data, member.size

    processor = GetBytesFromTarHandle(GetBytesFromTarHandle.Config())
    with TarFileHandle(archive, use_mmap=False) as handle:
        sample = cast(
            GetBytesFromTarHandle.Input,
            {
                "_tar_handle": handle,
                "key": "k0",
                "format": "mp4",
                "tar_offset": offset,
                "tar_size": size,
            },
        )
        out = list(processor(_as_stream_bytes([sample])))

    assert out[0].get("media") == payloads["k0"]


def test_image_decode_uses_the_jpeg_and_webp_fast_paths() -> None:
    """Turbojpeg and libwebp decode their own formats; PIL is the fallback.

    Each fast path crops DURING decode, which is the reason they exist, so a
    silent fall-through to PIL would still yield a correct-looking tensor
    while giving that up. Asserting pixel content rather than shape is what
    separates a working backend from one that returned nothing and let the
    cascade continue.
    """
    for fmt, encoding, options in (
        ("jpg", "JPEG", {"quality": 100}),
        ("webp", "WEBP", {"lossless": True}),
    ):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), (255, 0, 0)).save(buffer, format=encoding, **options)

        processor = CropDuringDecodeImage(CropDuringDecodeImage.Config())
        out = list(
            processor(
                iter(
                    [
                        cast(
                            CropDuringDecodeImage.Input,
                            {
                                "media": buffer.getvalue(),
                                "format": fmt,
                                "height": 16,
                                "width": 16,
                            },
                        ),
                    ],
                ),
            ),
        )

        tensor = out[0].get("media_tensor")
        assert tensor is not None
        assert tensor.shape == (3, 1, 16, 16), fmt
        # Red: channel 0 saturated, channel 2 dark. A BGR decode inverts this.
        assert int(tensor[0].min()) > 200, fmt
        assert int(tensor[2].max()) < 60, fmt


def test_image_decode_names_the_backend_that_rejected_the_bytes() -> None:
    """Each backend's failure is attributed, and the cascade keeps going.

    A truncated JPEG raises inside turbojpeg while a truncated WebP returns
    None from libwebp: two different failure shapes that must both become a
    filter reason naming the extension. Without the per-extension tag an
    operator sees only ``all_formats_failed`` on a bad shard and cannot tell a
    corrupt encoder from a misconfigured ``known_formats``.
    """
    processor = CropDuringDecodeImage(CropDuringDecodeImage.Config())

    for fmt, encoding, options in (
        ("jpg", "JPEG", {"quality": 100}),
        ("webp", "WEBP", {"lossless": True}),
    ):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), (255, 0, 0)).save(buffer, format=encoding, **options)
        # Magic bytes kept so the format is still recognised, payload cut
        # short: the corrupt-member case, which is the only one that reaches
        # the backend's own error path.
        encoded = buffer.getvalue()
        truncated = encoded[: len(encoded) // 3]

        out = list(
            processor(
                iter(
                    [
                        cast(
                            CropDuringDecodeImage.Input,
                            {
                                "media": truncated,
                                "format": fmt,
                                "height": 16,
                                "width": 16,
                            },
                        ),
                    ],
                ),
            ),
        )

        assert "media_tensor" not in out[0], fmt
        reasons = cast(list[str], cast(dict[str, object], out[0])["filter_reasons"])
        assert "decode_failed" in reasons[0], (fmt, reasons)
        # WHICH decoder failed and how, not just that one did. ``_process_image``
        # builds that detail (``jpg_OSError:...``) and the caller used to unpack
        # it into ``_``, leaving every failure -- a truncated file, an
        # unreadable codec, a wrong extension -- reported identically.
        assert fmt in reasons[0], (fmt, reasons)


def test_video_decode_returns_none_on_a_clip_it_cannot_read() -> None:
    """A file ffmpeg cannot decode filters the sample, never raises.

    ``_decode`` promises ``Tensor | None``, and its ``try`` exists to make the
    unreadable case a ``None``. The final ``.view(frames, h, w, 3)`` sits
    OUTSIDE that block, so any path reaching it with a byte count that is not
    a whole number of frames -- a truncated stream ffmpeg still yields chunks
    for -- raises ``RuntimeError`` out of a function documented not to,
    killing the run rather than dropping one sample.
    """
    processor = DecodeVideo(DecodeVideo.Config())

    # Real bytes, deliberately not a video: the decoder must answer None.
    result = processor._decode(
        b"\x00\x00\x00\x20ftypmp42" + bytes(64),
        height=8,
        width=8,
        keep_frames=2,
    )

    assert result is None
    """A sample already carrying ``media_tensor`` is not decoded twice.

    Two decoders write that field (this one and ``DecodeVideo``), so the guard
    is what keeps a pipeline listing both from re-running the second over the
    first's output.
    """
    processor = CropDuringDecodeImage(CropDuringDecodeImage.Config())
    existing = torch.zeros(3, 1, 4, 4, dtype=torch.uint8)

    out = list(
        processor(
            iter(
                [
                    cast(
                        CropDuringDecodeImage.Input,
                        {"media_tensor": existing, "media": b"ignored"},
                    ),
                ],
            ),
        ),
    )

    assert "media_tensor" in out[0]
    assert out[0]["media_tensor"] is existing


def test_tar_reader_skips_what_it_cannot_or_need_not_read(tmp_path: Path) -> None:
    """Terminal exits release the handle while routing preserves it."""
    archive = tmp_path / "shard.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo(name="k0.mp4")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"data"))

    processor = GetBytesFromTarHandle(GetBytesFromTarHandle.Config())
    with TarFileHandle(archive, use_mmap=False) as handle:
        out = list(
            processor(
                iter(
                    [
                        cast(
                            GetBytesFromTarHandle.Input,
                            {"_tar_handle": handle, "key": "k0", "media": b"already"},
                        ),
                        cast(GetBytesFromTarHandle.Input, {"_tar_handle": handle}),
                        cast(
                            GetBytesFromTarHandle.Input,
                            {"_tar_handle": handle, "key": "k0", "format": "png"},
                        ),
                        cast(
                            GetBytesFromTarHandle.Input,
                            {"_tar_handle": handle, "key": "absent", "format": "mp4"},
                        ),
                    ],
                ),
            ),
        )

    assert all("_tar_handle" not in out[index] for index in (0, 1, 3))
    assert "_tar_handle" in out[2]
    assert out[0].get("media") == b"already", "existing bytes were overwritten"
    assert "media" not in out[1], "read without a key"
    assert "media" not in out[2], "read a format it does not claim"
    assert not cast(dict[str, object], out[2]).get("filter_reasons")
    missing = cast(list[str], cast(dict[str, object], out[3])["filter_reasons"])
    assert "member_not_found" in missing[0]


def test_image_decode_falls_through_to_pil_and_reports_a_corrupt_payload() -> None:
    """The cascade tries each backend, then names what failed.

    ``_process_image`` walks ``known_formats`` and returns on the first
    backend that yields a tensor: turbojpeg for jpg, libwebp for webp, PIL for
    anything else. Listing "png" is what routes a PNG to PIL -- with only the
    two fast-path formats configured, the cascade never reaches it. Bytes that
    decode as nothing must come back as a filter reason rather than an
    exception, since a corrupt member in a shard is data, not a code bug.
    """
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(known_formats=["png"]),
    )

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 255, 0)).save(buffer, format="PNG")

    out = list(
        processor(
            iter(
                [
                    cast(
                        CropDuringDecodeImage.Input,
                        {"media": buffer.getvalue(), "height": 8, "width": 8},
                    ),
                    cast(
                        CropDuringDecodeImage.Input,
                        {"media": b"not an image", "height": 8, "width": 8},
                    ),
                ],
            ),
        ),
    )

    # "png" matches no fast path, so the cascade falls through to PIL.
    tensor = out[0].get("media_tensor")
    assert tensor is not None
    assert tensor.shape == (3, 1, 8, 8)
    assert int(tensor[1].max()) > int(tensor[0].max()), "green channel dominates"

    # Undecodable bytes are reported, and the sample survives.
    assert "media_tensor" not in out[1]
    reasons = cast(list[str], cast(dict[str, object], out[1])["filter_reasons"])
    assert "decode_failed" in reasons[0]


def test_image_decode_skips_a_format_it_does_not_claim() -> None:
    """A named format outside ``known_formats`` is another processor's job.

    Skipping without a filter reason is deliberate: the sample is not broken,
    it simply belongs to a decoder further down the pipeline, and marking it
    filtered would make a correctly-routed clip look dropped.
    """
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(known_formats=["jpg"]),
    )

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="PNG")

    out = list(
        processor(
            iter(
                [
                    cast(
                        CropDuringDecodeImage.Input,
                        {
                            "media": buffer.getvalue(),
                            "format": "png",
                            "height": 8,
                            "width": 8,
                        },
                    ),
                ],
            ),
        ),
    )

    assert "media_tensor" not in out[0]
    assert not cast(dict[str, object], out[0]).get("filter_reasons")


def test_image_decode_reports_a_sample_it_cannot_size() -> None:
    """Bytes without dimensions are reported, not silently skipped.

    ``_extract_fields`` distinguishes the two absences: no bytes at all is a
    routing outcome (an upstream reader may not have run), while bytes WITHOUT
    dimensions cannot be decoded and would otherwise vanish from the filter
    accounting.
    """
    processor = CropDuringDecodeImage(CropDuringDecodeImage.Config())

    out = list(
        processor(
            iter(
                [
                    cast(CropDuringDecodeImage.Input, {"media": b"payload"}),
                    cast(CropDuringDecodeImage.Input, {"height": 8, "width": 8}),
                ],
            ),
        ),
    )

    sized = cast(list[str], cast(dict[str, object], out[0])["filter_reasons"])
    assert "missing_dimensions" in sized[0]
    # No bytes yet: nothing to report, the reader simply has not run.
    assert not cast(dict[str, object], out[1]).get("filter_reasons")


def test_get_dimensions_remeasures_a_non_positive_cached_value() -> None:
    """A cached dimension is trusted only when it is actually usable.

    ``height > 0 < width`` is the guard, not mere presence: a zero or negative
    dimension is metadata that rode along on the sample, and taking it on
    trust skips the measurement that would have corrected it. Downstream every
    consumer divides by these, so a zero propagates as a ZeroDivisionError far
    from its origin.
    """
    buffer = io.BytesIO()
    Image.new("RGB", (12, 7), (255, 0, 0)).save(buffer, format="PNG")
    media = buffer.getvalue()

    processor = GetDimensionsFromBytes(GetDimensionsFromBytes.Config())
    out = list(
        processor(
            iter(
                [
                    cast(
                        GetDimensionsFromBytes.Input,
                        {"media": media, "height": 0, "width": 0},
                    ),
                    cast(
                        GetDimensionsFromBytes.Input,
                        {"media": media, "height": 999, "width": 999},
                    ),
                ],
            ),
        ),
    )

    # The unusable pair is re-measured from the bytes...
    assert (out[0].get("height"), out[0].get("width")) == (7, 12)
    # ...while a positive pair is taken as given, decode never being run.
    assert (out[1].get("height"), out[1].get("width")) == (999, 999)


def test_get_dimensions_passes_through_what_it_cannot_measure() -> None:
    """No bytes, or unmeasurable bytes, leave the sample untouched.

    Neither is a failure worth reporting: a sample with no ``media`` has not
    reached its reader yet, and bytes whose header does not parse are handled
    by the decoder downstream, which produces the real error with the format
    in hand. Adding a reason here would double-count both.
    """
    processor = GetDimensionsFromBytes(GetDimensionsFromBytes.Config())

    out = list(
        processor(
            iter(
                [
                    cast(GetDimensionsFromBytes.Input, {}),
                    cast(
                        GetDimensionsFromBytes.Input,
                        {"media": b"not an image header"},
                    ),
                ],
            ),
        ),
    )

    assert len(out) == 2
    for sample in out:
        assert "height" not in sample
        assert "width" not in sample
        assert not cast(dict[str, object], sample).get("filter_reasons")


def test_get_bytes_from_file_skips_a_sample_that_already_has_bytes(
    tmp_path: Path,
) -> None:
    """Existing ``media`` is never re-read from disk.

    The field is the handoff between readers -- tar, file, or an upstream
    fetch -- so re-reading would both cost an I/O and let a stale path
    overwrite bytes another reader already produced.
    """
    processor = GetBytesFromFile(GetBytesFromFile.Config())
    decoy = tmp_path / "decoy.bin"
    _ = decoy.write_bytes(b"from disk")

    out = list(
        processor(
            iter(
                [
                    cast(
                        GetBytesFromFile.Input,
                        {"file_path": str(decoy), "media": b"already here"},
                    ),
                ],
            ),
        ),
    )

    assert out[0].get("media") == b"already here"


def test_get_bytes_from_file_reports_an_unreadable_path(tmp_path: Path) -> None:
    """A missing file yields the sample with a reason, never drops it.

    Every sibling reports what it could not handle; a silently vanished sample
    is missing from the filter accounting too, so a shard that lost half its
    files would look like a shard that simply had fewer.
    """
    processor = GetBytesFromFile(GetBytesFromFile.Config())

    present = tmp_path / "present.bin"
    _ = present.write_bytes(b"payload")

    out = list(
        processor(
            iter(
                [
                    cast(GetBytesFromFile.Input, {"file_path": str(present)}),
                    cast(
                        GetBytesFromFile.Input,
                        {"file_path": str(tmp_path / "absent.bin")},
                    ),
                    cast(GetBytesFromFile.Input, {}),
                ],
            ),
        ),
    )

    assert len(out) == 3, "no sample may be dropped"
    assert out[0].get("media") == b"payload"
    assert "media" not in out[1]
    reasons = cast(list[str], cast(dict[str, object], out[1])["filter_reasons"])
    assert "GetBytesFromFile" in reasons[0]
    # A sample with no path at all is a routing outcome, not a failure.
    assert not cast(dict[str, object], out[2]).get("filter_reasons")


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
