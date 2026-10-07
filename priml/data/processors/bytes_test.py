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
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import patch

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
from priml.math.pixel import rgb2float


if TYPE_CHECKING:
    from collections.abc import Iterator


_FRAMES: Final = 8
_SOURCE_HEIGHT: Final = 64
_SOURCE_WIDTH: Final = 128


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
                "frames": 8,
                "height": 64,
                "width": 128,
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
    assert (64, 128) != (16, 48)


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
    assert tensor.shape[1] == 8 + 4
    # The padding repeats the final decoded frame.
    torch.testing.assert_close(tensor[:, -1], tensor[:, 8 - 1])


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
        "frames": 8,
        "height": 64,
        "width": 128,
        "target_frames": 8,
        "target_height": 32,
        "target_width": 32,
    }
    out = list(processor(_as_stream([sample])))

    tensor = out[0].get("media_tensor")
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, _FRAMES, 32, 32)


@pytest.mark.cli_python_subprocess
def test_decode_video_filters_a_clip_it_cannot_decode(video_tar: Path) -> None:
    sample = cast(
        DecodeVideo.Input,
        {
            "media": b"not a video",
            "format": "mp4",
            "frames": 8,
            "height": 4,
            "width": 4,
        },
    )

    out = list(DecodeVideo(DecodeVideo.Config())(_as_stream([sample])))

    del video_tar
    assert "media_tensor" not in out[0]
    assert out[0].get("filter_reasons") == ["DecodeVideo:decode_failed"]


@pytest.mark.cli_python_subprocess
def test_decode_video_filters_a_clip_truncated_mid_stream(tmp_path: Path) -> None:
    mp4 = tmp_path / "long.mp4"
    _ = subprocess.run(  # noqa: S603 -- argv is built from literals and ints.
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=s=64x64:d=8:r=16",
            "-g",
            "8",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4),
        ],
        check=True,
        capture_output=True,
    )
    payload = mp4.read_bytes()
    # A faststart file cut at 60% still decodes its first ~64 of 128 frames,
    # then ffmpeg hits the missing data and stops early.
    sample = cast(
        DecodeVideo.Input,
        {
            "media": payload[: len(payload) * 6 // 10],
            "format": "mp4",
            "frames": 128,
            "height": 64,
            "width": 64,
            "target_frames": 128,
            "target_height": 8,
            "target_width": 8,
        },
    )

    out = list(DecodeVideo(DecodeVideo.Config())(_as_stream([sample])))

    assert "media_tensor" not in out[0]
    assert out[0].get("filter_reasons") == ["DecodeVideo:decode_failed"]


@pytest.mark.cli_python_subprocess
def test_decode_video_decodes_untagged_red_as_pure_red(video_tar: Path) -> None:
    """The fixture is encoded with ffmpeg's default BT.601 matrix, untagged.

    Forcing ``in_color_matrix=bt709`` decoded it with the wrong matrix, which
    shifts pure red visibly into green and blue.
    """
    out = _run(video_tar, target_frames=1, target_height=16, target_width=16)
    tensor = out[0].get("media_tensor")
    assert isinstance(tensor, torch.Tensor)

    left = tensor[:, 0, 4:12, 1:5].float().mean(dim=(1, 2))

    torch.testing.assert_close(left, torch.tensor([1.0, -1.0, -1.0]), atol=0.04, rtol=0)


@pytest.mark.parametrize(
    ("targets", "reason"),
    [
        ({"target_frames": 4}, "DecodeVideo:partial_target"),
        ({"target_height": 4, "target_width": 4}, "DecodeVideo:partial_target"),
    ],
)
def test_decode_video_filters_a_partial_target(
    targets: dict[str, int],
    reason: str,
) -> None:
    sample = cast(
        DecodeVideo.Input,
        {"media": b"clip", "frames": 8, "height": 4, "width": 4, **targets},
    )

    out = list(DecodeVideo(DecodeVideo.Config())(_as_stream([sample])))

    assert out[0].get("filter_reasons") == [reason]


def test_decode_video_filters_a_video_without_dimensions() -> None:
    sample = cast(DecodeVideo.Input, {"media": b"clip", "frames": 8, "width": 4})

    out = list(DecodeVideo(DecodeVideo.Config())(_as_stream([sample])))

    assert out[0].get("filter_reasons") == ["DecodeVideo:missing_dimensions"]


def test_image_decoder_leaves_a_video_without_dimensions_to_the_video_decoder() -> None:
    sample = cast(CropDuringDecodeImage.Input, {"media": b"clip", "frames": 8})

    out = list(CropDuringDecodeImage(CropDuringDecodeImage.Config())(iter([sample])))

    assert out[0].get("filter_reasons") is None


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
                "frames": 8,
                "height": 64,
                "width": 128,
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
            "frames": 8,
            "height": 64,
            "width": 128,
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
            "frames": 8,
            "height": 64,
            "width": 128,
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
                "frames": 8,
                "height": 64,
                "width": 128,
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
    assert [sample.get("filter_reasons") for sample in unreadable] == [
        ["GetBytesFromTarHandle:missing: key"],
        ["GetBytesFromTarHandle:missing: _tar_handle"],
    ]
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
    assert out[0].get("tar_offset") == offset
    assert out[0].get("tar_size") == size


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
        samples: list[CropDuringDecodeImage.Input] = [
            {
                "media": buffer.getvalue(),
                "format": fmt,
                "height": 16,
                "width": 16,
            },
        ]
        out = list(
            processor(
                iter(
                    samples,
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

        samples: list[CropDuringDecodeImage.Input] = [
            {
                "media": truncated,
                "format": fmt,
                "height": 16,
                "width": 16,
            },
        ]
        out = list(
            processor(
                iter(
                    samples,
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
    existing = torch.zeros(3, 2, 4, 5, dtype=torch.uint8)

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
    assert out[0].get("filter_reasons") is None


def test_tar_reader_tries_extensions_and_records_exact_member_metadata(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "media.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("clip.mp4")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"data"))

    processor = GetBytesFromTarHandle(
        GetBytesFromTarHandle.Config(known_formats=["jpg", "mp4"]),
    )
    with TarFileHandle(archive, use_mmap=False) as handle:
        sample = cast(
            GetBytesFromTarHandle.Input,
            {"_tar_handle": handle, "key": "clip"},
        )
        result = next(processor(_as_stream_bytes([sample])))

    with tarfile.open(archive) as tar:
        member = tar.getmember("clip.mp4")
    assert result.get("media") == b"data"
    assert result.get("tar_offset") == member.offset_data
    assert result.get("tar_size") == member.size

    with TarFileHandle(archive, use_mmap=False) as handle:
        named_jpg = cast(
            GetBytesFromTarHandle.Input,
            {"_tar_handle": handle, "key": "clip", "format": "jpg"},
        )
        rejected = next(processor(_as_stream_bytes([named_jpg])))
    assert "media" not in rejected
    assert rejected["filter_reasons"] == [
        "GetBytesFromTarHandle:member_not_found: clip",
    ]


def test_tar_reader_skips_what_it_cannot_or_need_not_read(tmp_path: Path) -> None:
    """Terminal exits release the handle while routing preserves it."""
    archive = tmp_path / "shard.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo(name="k0.mp4")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"data"))

    processor = GetBytesFromTarHandle(GetBytesFromTarHandle.Config())
    with TarFileHandle(archive, use_mmap=False) as handle:
        samples: list[GetBytesFromTarHandle.Input] = [
            cast(
                GetBytesFromTarHandle.Input,
                {"_tar_handle": handle, "key": "k0", "media": b"already"},
            ),
            {"_tar_handle": handle},
            {"_tar_handle": handle, "key": "k0", "format": "png"},
            {"_tar_handle": handle, "key": "absent", "format": "mp4"},
        ]
        out = list(processor(iter(samples)))

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

    samples: list[CropDuringDecodeImage.Input] = [
        {"media": buffer.getvalue(), "height": 8, "width": 8},
        {"media": b"not an image", "height": 8, "width": 8},
    ]
    out = list(
        processor(
            iter(
                samples,
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

    samples: list[CropDuringDecodeImage.Input] = [
        {
            "media": buffer.getvalue(),
            "format": "png",
            "height": 8,
            "width": 8,
        },
    ]
    out = list(
        processor(
            iter(
                samples,
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

    samples: list[CropDuringDecodeImage.Input] = [
        {"media": b"payload"},
        {"height": 8, "width": 8},
    ]
    out = list(
        processor(
            iter(
                samples,
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
    samples: list[GetDimensionsFromBytes.Input] = [
        {"media": media, "height": 999, "width": 999},
        {"media": media, "height": 0, "width": 12},
        {"media": media, "height": 7, "width": 0},
        {"media": media, "height": 0, "width": 0},
    ]
    out = list(
        processor(
            iter(
                samples,
            ),
        ),
    )

    assert [sample.get("height") for sample in out] == [999, 7, 7, 7]
    assert [sample.get("width") for sample in out] == [999, 12, 12, 12]


def test_get_dimensions_trusts_positive_unit_dimensions() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (12, 7), (255, 0, 0)).save(buffer, format="PNG")
    media = buffer.getvalue()
    processor = GetDimensionsFromBytes(GetDimensionsFromBytes.Config())
    sample = cast(
        GetDimensionsFromBytes.Input,
        {"media": media, "height": 1, "width": 2},
    )

    result = next(processor(iter([sample])))

    assert result == {"media": media, "height": 1, "width": 2}


def test_get_dimensions_passes_through_what_it_cannot_measure() -> None:
    """No bytes, or unmeasurable bytes, leave the sample untouched.

    Neither is a failure worth reporting: a sample with no ``media`` has not
    reached its reader yet, and bytes whose header does not parse are handled
    by the decoder downstream, which produces the real error with the format
    in hand. Adding a reason here would double-count both.
    """
    processor = GetDimensionsFromBytes(GetDimensionsFromBytes.Config())

    samples: list[GetDimensionsFromBytes.Input] = [
        {},
        {"media": b"not an image header"},
    ]
    out = list(
        processor(
            iter(
                samples,
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
    absent = tmp_path / "absent.bin"
    _ = present.write_bytes(b"payload")

    samples: list[GetBytesFromFile.Input] = [
        cast(
            GetBytesFromFile.Input,
            {"file_path": str(present), "media": b"preserved"},
        ),
        {},
        {"file_path": str(present)},
        {"file_path": str(absent)},
        {},
    ]
    out = list(processor(iter(samples)))

    assert len(out) == 5, "no sample may be dropped"
    assert out[0].get("media") == b"preserved"
    assert not cast(dict[str, object], out[0]).get("filter_reasons")
    assert not cast(dict[str, object], out[1]).get("filter_reasons")
    assert out[2].get("media") == b"payload"
    assert "media" not in out[3]
    reasons = cast(list[str], cast(dict[str, object], out[3])["filter_reasons"])
    assert len(reasons) == 1
    assert reasons[0].startswith("GetBytesFromFile:unreadable: ")
    assert str(absent) in reasons[0]
    assert not cast(dict[str, object], out[4]).get("filter_reasons")


def test_decode_video_field_validation_and_handle_cleanup() -> None:
    processor = DecodeVideo(DecodeVideo.Config())
    samples: list[DecodeVideo.Input] = [
        {},
        {"frames": 2, "height": 2, "width": 2},
        {"media": b"x"},
        {"media": b"x", "frames": 1, "height": 2, "width": 2},
        {"media": b"x", "frames": 2, "height": 2, "width": 2, "target_height": 1},
        {
            "media": b"x",
            "frames": 2,
            "height": 2,
            "width": 2,
            "target_frames": 0,
            "target_height": 1,
            "target_width": 1,
        },
        {
            "media": b"x",
            "frames": 2,
            "height": 2,
            "width": 2,
            "target_frames": 2,
            "target_height": 1,
            "target_width": 1,
        },
    ]
    with patch.object(processor, "_process_video", return_value=None):
        output = list(processor(iter(samples)))
    assert len(output) == len(samples)
    assert "filter_reasons" in output[-2]
    assert "_tar_handle" not in output[-1]


def test_crop_image_backend_error_and_decode_video_helpers() -> None:
    image = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(known_formats=["jpg"], use_turbojpeg=True),
    )
    assert (
        image._extract_fields({"media": b"x", "frames": 2, "height": 2, "width": 2})
        is None
    )
    with patch(
        "priml.data.processors.bytes.decode_jpeg_turbojpeg",
        side_effect=ValueError("bad"),
    ):
        decoded = image._process_image(b"x", "jpg", 2, 2, None)
    assert decoded is not None
    assert decoded[0] is None

    processor = DecodeVideo(DecodeVideo.Config())
    assert (
        processor._process_video(
            None,
            "x",
            "avi",
            frames=2,
            height=2,
            width=2,
            target_frames=None,
            target_height=None,
            target_width=None,
        )
        is None
    )
    assert (
        processor._process_video(
            None,
            "x",
            "",
            frames=2,
            height=2,
            width=2,
            target_frames=None,
            target_height=None,
            target_width=None,
        )
        is None
    )
    with patch.object(processor, "_decode", return_value=None) as decode:
        assert (
            processor._process_video(
                None,
                "x",
                "",
                frames=2,
                height=2,
                width=2,
                target_frames=3,
                target_height=1,
                target_width=1,
                media=b"x",
            )
            is None
        )
        decode.assert_called_once()
    sample = cast(
        DecodeVideo.Input,
        {"media": b"x", "frames": 2, "height": 2, "width": 2, "_tar_handle": object()},
    )
    with patch.object(processor, "_process_video", return_value=None):
        output = next(processor(iter([sample])))
    assert "_tar_handle" not in output


def test_decode_video_read_media_and_decode_oserror() -> None:
    processor = DecodeVideo(DecodeVideo.Config())
    assert processor._read_media(None, "x", "") == b""
    with patch(
        "priml.data.processors.bytes.subprocess.Popen",
        side_effect=OSError,
    ):
        assert processor._decode(b"x", height=1, width=1, keep_frames=1) is None


def test_decode_video_reads_tar_and_handles_decode_stream_failures() -> None:
    class Tar:
        name: str | None = None

        def getmember(self, name: str) -> tarfile.TarInfo:
            if name == "x.mp4":
                return tarfile.TarInfo(name)
            raise KeyError(name)

        def extractfile(self, member: str | tarfile.TarInfo) -> io.BytesIO:
            del member
            return io.BytesIO(b"payload")

    processor = DecodeVideo(DecodeVideo.Config())
    tar = Tar()
    assert processor._read_media(tar, "x", "mp4") == b"payload"
    assert processor._read_media(tar, "missing", "mp4") == b""

    class Stdout:
        def read(self, size: int) -> bytes:
            del size
            return b"x"

        def close(self) -> None:
            pass

    class Process:
        stdout = Stdout()
        returncode = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: int = 0) -> None:
            del timeout

    with patch(
        "priml.data.processors.bytes.subprocess.Popen",
        return_value=Process(),
    ):
        assert processor._decode(b"x", height=2, width=2, keep_frames=1) is None


def test_crop_image_integer_dtype_and_unknown_format() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buffer, format="PNG")
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(dtype=torch.int16, known_formats=["png"]),
    )
    sample = cast(
        CropDuringDecodeImage.Input,
        {"media": buffer.getvalue(), "format": "png", "height": 4, "width": 4},
    )
    out = next(processor(iter([sample])))
    tensor = out.get("media_tensor")
    assert tensor is not None
    assert tensor.dtype == torch.int16


def test_image_decode_sets_frames_and_honors_device() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (6, 4), (255, 0, 0)).save(buffer, format="JPEG")
    processor = CropDuringDecodeImage(CropDuringDecodeImage.Config(device="meta"))
    sample = cast(
        CropDuringDecodeImage.Input,
        {"media": buffer.getvalue(), "format": "jpg", "height": 4, "width": 6},
    )

    result = next(processor(iter([sample])))

    tensor = result.get("media_tensor")
    assert tensor is not None
    assert tensor.device.type == "meta"
    assert tensor.shape == (3, 1, 4, 6)
    assert result.get("frames") == 1
    assert result.get("filter_reasons") is None


def test_image_decode_reports_corrupt_jpeg_with_filter_identity() -> None:
    processor = CropDuringDecodeImage(CropDuringDecodeImage.Config())
    sample = cast(
        CropDuringDecodeImage.Input,
        {"media": b"not a jpeg", "format": "jpg", "height": 4, "width": 6},
    )

    result = next(processor(iter([sample])))

    reasons = cast(list[str], cast(dict[str, object], result)["filter_reasons"])
    assert reasons == [
        "CropDuringDecodeImage:decode_failed: jpg_decode_failed",
    ]


def test_image_decode_routes_partial_dimensions_and_non_image_frames() -> None:
    processor = CropDuringDecodeImage(CropDuringDecodeImage.Config())
    samples = [
        cast(CropDuringDecodeImage.Input, {"media": b"x", "width": 6}),
        cast(CropDuringDecodeImage.Input, {"media": b"x", "height": 4}),
        cast(
            CropDuringDecodeImage.Input,
            {"media": b"x", "height": 4, "width": 6, "target_frames": 2},
        ),
    ]

    results = list(processor(iter(samples)))

    assert [result.get("filter_reasons") for result in results] == [
        ["CropDuringDecodeImage:missing_dimensions"],
        ["CropDuringDecodeImage:missing_dimensions"],
        None,
    ]
    assert all("media_tensor" not in result for result in results)


def test_image_decode_target_floor_uses_distinct_height_and_width() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (255, 0, 0)).save(buffer, format="JPEG")
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(scale_to_target=True),
    )
    sample = cast(
        CropDuringDecodeImage.Input,
        {
            "media": buffer.getvalue(),
            "format": "jpg",
            "height": 48,
            "width": 64,
            "target_height": 24,
            "target_width": 12,
            "crop": (0, 0, 48, 64),
        },
    )

    tensor = next(processor(iter([sample]))).get("media_tensor")

    assert tensor is not None
    assert tensor.shape == (3, 1, 24, 32)


def test_decode_video_routes_fields_and_continues_after_cached_samples() -> None:
    processor = DecodeVideo(DecodeVideo.Config())
    cached_handle = object()
    cached_tensor = torch.zeros(3, 2, 4, 6, dtype=torch.float16)
    cached = cast(
        DecodeVideo.Input,
        {"media_tensor": cached_tensor, "_tar_handle": cached_handle},
    )
    decode_handle = object()
    sample = cast(
        DecodeVideo.Input,
        {
            "_tar_handle": decode_handle,
            "key": "clip",
            "format": "mp4",
            "media": b"clip-bytes",
            "frames": 3,
            "height": 4,
            "width": 6,
            "target_frames": 5,
            "target_height": 8,
            "target_width": 12,
        },
    )
    with patch.object(processor, "_process_video", return_value=None) as process_video:
        output = list(processor(iter([cached, sample])))

    assert len(output) == 2
    assert output[0].get("media_tensor") is cached_tensor
    assert output[0].get("_tar_handle") is cached_handle
    assert "_tar_handle" not in output[1]
    process_video.assert_called_once_with(
        decode_handle,
        "clip",
        "mp4",
        frames=3,
        height=4,
        width=6,
        target_frames=5,
        target_height=8,
        target_width=12,
        media=b"clip-bytes",
    )


def test_decode_video_requires_each_input_and_keeps_empty_keys_empty() -> None:
    processor = DecodeVideo(DecodeVideo.Config())
    complete: DecodeVideo.Input = {
        "media": b"clip",
        "key": "",
        "format": "mp4",
        "frames": 2,
        "height": 4,
        "width": 6,
    }
    fields = processor._extract_fields(complete)

    assert fields is not None
    assert fields[1:3] == ("", "mp4")
    incomplete_samples = [
        cast(DecodeVideo.Input, {"media": b"clip", "height": 4, "width": 6}),
        cast(DecodeVideo.Input, {"media": b"clip", "frames": 2, "width": 6}),
        cast(DecodeVideo.Input, {"media": b"clip", "frames": 2, "height": 4}),
    ]
    for incomplete in incomplete_samples:
        assert processor._extract_fields(incomplete) is None

    for address in (
        {"key": "clip"},
        {"_tar_handle": object()},
        {},
    ):
        missing_address = cast(
            DecodeVideo.Input,
            {**address, "frames": 2, "height": 4, "width": 6},
        )
        assert processor._extract_fields(missing_address) is None


def test_video_processor_uses_tar_fallback_and_source_geometry(
    tmp_path: Path,
) -> None:
    height, width, frames = 4, 6, 2
    stdout_bytes = bytearray(bytes([255, 0, 0]) * height * width * frames)
    captured_payload = b""
    captured_args: list[str] = []

    class Stdout:
        def read(self, size: int) -> bytes:
            chunk = bytes(stdout_bytes[:size])
            del stdout_bytes[:size]
            return chunk

        def close(self) -> None:
            pass

    class Process:
        stdout = Stdout()
        returncode = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: int = 0) -> None:
            del timeout

    def launch(args: list[str], *, stdout: int, stderr: int) -> Process:
        del stdout, stderr
        nonlocal captured_payload
        captured_args.extend(args)
        captured_payload = Path(args[args.index("-i") + 1]).read_bytes()
        return Process()

    archive = tmp_path / "clip.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("clip.mp4")
        info.size = 10
        tar.addfile(info, io.BytesIO(b"clip-bytes"))

    processor = DecodeVideo(
        DecodeVideo.Config(known_formats=["webm", "mp4"], ffmpeg_exe="fake-ffmpeg"),
    )
    with (
        TarFileHandle(archive, use_mmap=False) as handle,
        patch(
            "priml.data.processors.bytes.subprocess.Popen",
            side_effect=launch,
        ),
    ):
        tensor = processor._process_video(
            handle,
            "clip",
            "",
            frames=frames,
            height=height,
            width=width,
            target_frames=None,
            target_height=None,
            target_width=None,
        )

    assert tensor is not None
    assert tensor.shape == (3, frames, height, width)
    assert captured_payload == b"clip-bytes"
    assert any(arg.startswith("scale=6:4:") for arg in captured_args)


@pytest.mark.parametrize(
    ("fmt", "encoding"),
    [("jpg", "JPEG"), ("webp", "WEBP"), ("png", "PNG")],
)
def test_image_decode_passes_the_requested_crop_to_each_backend(
    fmt: str,
    encoding: str,
) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (255, 0, 0)).save(
        buffer,
        format=encoding,
        **({"quality": 100} if encoding == "JPEG" else {}),
    )
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(known_formats=[fmt]),
    )
    sample = cast(
        CropDuringDecodeImage.Input,
        {
            "media": buffer.getvalue(),
            "format": fmt,
            "height": 24,
            "width": 32,
            "crop": (8, 0, 8, 16),
        },
    )

    tensor = next(processor(iter([sample]))).get("media_tensor")

    assert tensor is not None
    assert tensor.shape == (3, 1, 8, 16)


def test_image_decode_tries_the_next_format_after_a_backend_rejects_bytes() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), (0, 255, 0)).save(
        buffer,
        format="WEBP",
        lossless=True,
    )
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(known_formats=["jpg", "webp"]),
    )
    sample = cast(
        CropDuringDecodeImage.Input,
        {"media": buffer.getvalue(), "height": 12, "width": 16},
    )

    result = next(processor(iter([sample])))

    tensor = result.get("media_tensor")
    assert tensor is not None
    assert tensor.shape == (3, 1, 12, 16)
    assert int(tensor[1].min()) > 200
    assert result.get("filter_reasons") is None


def test_image_decode_reports_full_backend_exception_detail() -> None:
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(known_formats=["jpg"]),
    )
    error = "x" * 60
    with patch(
        "priml.data.processors.bytes.decode_jpeg_turbojpeg",
        side_effect=ValueError(error),
    ):
        result = processor._process_image(b"jpeg", "jpg", 4, 6, None)

    assert result == (None, f"jpg_ValueError:{error[:50]}")


def test_image_decode_names_an_empty_backend_cascade() -> None:
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(known_formats=[]),
    )

    assert processor._process_image(b"payload", "", 4, 6, None) == (
        None,
        "all_formats_failed",
    )


def test_rgba_image_uses_a_four_channel_decode() -> None:
    buffer = io.BytesIO()
    Image.new("RGBA", (8, 6), (255, 0, 0, 64)).save(
        buffer,
        format="WEBP",
        lossless=True,
    )
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(channels_format="rgba", known_formats=["webp"]),
    )
    sample = cast(
        CropDuringDecodeImage.Input,
        {"media": buffer.getvalue(), "format": "webp", "height": 6, "width": 8},
    )

    tensor = next(processor(iter([sample]))).get("media_tensor")

    assert tensor is not None
    assert tensor.shape == (4, 1, 6, 8)
    assert int(tensor[3].min()) == 64


def test_decode_video_routes_tar_media_and_requested_geometry(
    tmp_path: Path,
) -> None:
    height, width, frames = 4, 6, 3
    stdout_bytes = bytearray(bytes([255, 0, 0]) * height * width * frames)

    class Stdout:
        def read(self, size: int) -> bytes:
            chunk = bytes(stdout_bytes[:size])
            del stdout_bytes[:size]
            return chunk

        def close(self) -> None:
            pass

    class Process:
        stdout = Stdout()
        returncode = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: int = 0) -> None:
            del timeout

    archive = tmp_path / "clip.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("clip.mp4")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"video"))

    processor = DecodeVideo(DecodeVideo.Config(ffmpeg_exe="fake-ffmpeg"))
    with TarFileHandle(archive, use_mmap=False) as handle:
        sample = cast(
            DecodeVideo.Input,
            {
                "_tar_handle": handle,
                "key": "clip",
                "format": "mp4",
                "frames": 2,
                "height": 6,
                "width": 8,
                "target_frames": frames,
                "target_height": height,
                "target_width": width,
            },
        )
        with patch(
            "priml.data.processors.bytes.subprocess.Popen",
            return_value=Process(),
        ):
            result = next(processor(iter([sample])))

    tensor = result.get("media_tensor")
    assert tensor is not None
    assert tensor.shape == (3, frames, height, width)
    assert tensor.dtype == torch.float16
    torch.testing.assert_close(
        tensor[:, 0, 0, 0],
        torch.tensor([1.0, -1.0, -1.0], dtype=torch.float16),
    )
    assert "_tar_handle" not in result


@pytest.mark.parametrize("field", ["target_frames", "target_height", "target_width"])
def test_video_target_dimensions_accept_one_and_reject_zero(field: str) -> None:
    processor = DecodeVideo(DecodeVideo.Config())
    target = {"target_frames": 2, "target_height": 4, "target_width": 6}
    # Each target dimension is valid down to 1; zero is the rejected boundary.
    target[field] = 1
    sample = cast(
        DecodeVideo.Input,
        {"media": b"video", "frames": 2, "height": 4, "width": 6, **target},
    )

    fields = processor._extract_fields(sample)

    assert fields is not None
    assert fields[-3:] == (
        target["target_frames"],
        target["target_height"],
        target["target_width"],
    )

    target[field] = 0
    invalid = cast(
        DecodeVideo.Input,
        {"media": b"video", "frames": 2, "height": 4, "width": 6, **target},
    )
    assert processor._extract_fields(invalid) is None
    assert invalid.get("filter_reasons") == [
        f"DecodeVideo:invalid_target:f={target['target_frames']}_h={target['target_height']}_w={target['target_width']}",
    ]


def test_image_processor_passes_exact_floors_and_default_decode_failure() -> None:
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(scale_to_target=True),
    )
    sample = cast(
        CropDuringDecodeImage.Input,
        {"media": b"image", "format": "jpg", "height": 16, "width": 24},
    )
    with patch.object(
        processor,
        "_process_image",
        return_value=(None, None),
    ) as process_image:
        result = next(processor(iter([sample])))

    process_image.assert_called_once_with(
        b"image",
        "jpg",
        16,
        24,
        None,
        floor=(0, 0),
    )
    assert result.get("filter_reasons") == [
        "CropDuringDecodeImage:decode_failed",
    ]

    sample = cast(
        CropDuringDecodeImage.Input,
        {
            "media": b"image",
            "format": "jpg",
            "height": 16,
            "width": 24,
            "target_height": 6,
            "target_width": 10,
        },
    )
    with patch.object(
        processor,
        "_process_image",
        return_value=None,
    ) as process_image:
        result = next(processor(iter([sample])))

    process_image.assert_called_once_with(
        b"image",
        "jpg",
        16,
        24,
        None,
        floor=(6, 10),
    )
    assert result.get("filter_reasons") is None

    unscaled = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(scale_to_target=False),
    )
    sample = cast(
        CropDuringDecodeImage.Input,
        {
            "media": b"image",
            "format": "jpg",
            "height": 16,
            "width": 24,
            "target_height": 6,
            "target_width": 10,
        },
    )
    with patch.object(
        unscaled,
        "_process_image",
        return_value=None,
    ) as process_image:
        _ = next(unscaled(iter([sample])))

    process_image.assert_called_once_with(
        b"image",
        "jpg",
        16,
        24,
        None,
        floor=(0, 0),
    )


def test_image_decoder_continues_after_a_cached_sample() -> None:
    processor = CropDuringDecodeImage(CropDuringDecodeImage.Config())
    cached_tensor = torch.zeros(3, 2, 4, 6, dtype=torch.uint8)
    cached = cast(
        CropDuringDecodeImage.Input,
        {"media_tensor": cached_tensor, "media": b"ignored"},
    )
    decoded = torch.ones(3, 4, 6, dtype=torch.uint8)
    sample = cast(
        CropDuringDecodeImage.Input,
        {"media": b"image", "format": "jpg", "height": 4, "width": 6},
    )
    with patch.object(
        processor,
        "_process_image",
        return_value=(decoded, None),
    ) as call:
        results = list(processor(iter([cached, sample])))

    assert len(results) == 2
    assert results[0].get("media_tensor") is cached_tensor
    decoded_tensor = results[1].get("media_tensor")
    assert isinstance(decoded_tensor, torch.Tensor)
    assert decoded_tensor.shape == (3, 1, 4, 6)
    call.assert_called_once()


def test_image_backend_calls_receive_exact_decode_options() -> None:
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(scale_to_target=True),
    )
    expected = torch.zeros(3, 8, 12, dtype=torch.uint8)
    crop = (2, 4, 8, 12)
    with patch(
        "priml.data.processors.bytes.decode_jpeg_turbojpeg",
        return_value=expected,
    ) as decode_jpeg:
        decoded = processor._process_image(
            b"jpeg",
            "jpg",
            16,
            24,
            crop,
            floor=(6, 10),
        )

    assert decoded is not None
    assert decoded[0] is expected
    assert decoded[1] is None
    decode_jpeg.assert_called_once_with(
        b"jpeg",
        processor.turbo_jpeg,
        16,
        24,
        crop=crop,
        channels_first=True,
        min_height=6,
        min_width=10,
    )

    with patch(
        "priml.data.processors.bytes.decode_webp_libwebp",
        return_value=expected,
    ) as decode_webp:
        decoded = processor._process_image(b"webp", "webp", 16, 24, crop)

    assert decoded is not None
    assert decoded[0] is expected
    assert decoded[1] is None
    decode_webp.assert_called_once_with(
        b"webp",
        16,
        24,
        crop=crop,
        channels_first=True,
    )


def test_image_cascade_keeps_trying_after_none_and_exceptions() -> None:
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(known_formats=["jpg", "png"]),
    )
    decoded = torch.ones(3, 4, 6, dtype=torch.uint8)
    with (
        patch(
            "priml.data.processors.bytes.decode_jpeg_turbojpeg",
            side_effect=ValueError("bad jpeg"),
        ) as decode_jpeg,
        patch(
            "priml.data.processors.bytes.decode_image_pil",
            return_value=decoded,
        ) as decode_pil,
    ):
        result = processor._process_image(b"image", "", 4, 6, None)

    assert result is not None
    assert result[0] is decoded
    assert result[1] is None
    decode_jpeg.assert_called_once()
    decode_pil.assert_called_once()

    processor.known_formats = ["webp", "png"]
    with (
        patch(
            "priml.data.processors.bytes.decode_webp_libwebp",
            return_value=None,
        ),
        patch(
            "priml.data.processors.bytes.decode_image_pil",
            return_value=None,
        ),
    ):
        assert processor._process_image(b"image", "", 4, 6, None) == (
            None,
            "png_decode_failed",
        )


def test_image_decode_normalizes_in_place_and_initializes_only_enabled_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(
            dtype=torch.float16,
            known_formats=["png"],
            use_turbojpeg=False,
        ),
    )
    assert processor.turbo_jpeg is None
    assert processor.use_libwebp is True
    decoded = torch.zeros(3, 4, 6, dtype=torch.uint8)
    normalized = torch.ones(
        3,
        1,
        4,
        6,
        dtype=torch.float16,
    )  # CropDuringDecodeImage uses [channels, frames, height, width].
    conversions: list[tuple[torch.Tensor, bool]] = []

    def normalize(
        x: torch.Tensor,
        *,
        inplace: bool = False,
        unit_interval: bool = False,
    ) -> torch.Tensor:
        assert not unit_interval
        conversions.append((x, inplace))
        return normalized

    monkeypatch.setattr(
        "priml.data.processors.bytes.rgb2float",
        normalize,
    )
    with patch(
        "priml.data.processors.bytes.decode_image_pil",
        return_value=decoded,
    ):
        samples: list[CropDuringDecodeImage.Input] = [
            {
                "media": b"image",
                "format": "png",
                "height": 4,
                "width": 6,
            },
        ]
        result = next(
            processor(
                iter(
                    samples,
                ),
            ),
        )

    assert len(conversions) == 1
    source, inplace = conversions[0]
    torch.testing.assert_close(
        source,
        decoded.unsqueeze(-3).to(dtype=torch.float16),
    )
    assert inplace
    assert result.get("media_tensor") is normalized


def test_image_cascade_continues_after_backend_failures() -> None:
    processor = CropDuringDecodeImage(
        CropDuringDecodeImage.Config(known_formats=["webp", "png"]),
    )
    decoded = torch.ones(3, 4, 6, dtype=torch.uint8)
    with (
        patch(
            "priml.data.processors.bytes.decode_webp_libwebp",
            return_value=None,
        ) as decode_webp,
        patch(
            "priml.data.processors.bytes.decode_image_pil",
            return_value=decoded,
        ) as decode_pil,
    ):
        result = processor._process_image(b"image", "", 4, 6, None)

    assert result is not None
    assert result[0] is decoded
    assert result[1] is None
    decode_webp.assert_called_once()
    decode_pil.assert_called_once()

    processor.known_formats = ["png", "jpg"]
    with (
        patch(
            "priml.data.processors.bytes.decode_image_pil",
            return_value=None,
        ) as decode_pil,
        patch(
            "priml.data.processors.bytes.decode_jpeg_turbojpeg",
            return_value=decoded,
        ) as decode_jpeg,
    ):
        result = processor._process_image(b"image", "", 4, 6, None)

    assert result is not None
    assert result[0] is decoded
    decode_pil.assert_called_once()
    decode_jpeg.assert_called_once()


def test_video_processor_reads_tar_bytes_and_repeats_the_last_distinct_frame(
    tmp_path: Path,
) -> None:
    height, width, frames = 2, 3, 3
    frame_bytes = bytes([255, 0, 0]) * height * width
    frame_bytes += bytes([0, 255, 0]) * height * width
    frame_bytes += bytes([0, 0, 255]) * height * width
    remaining = bytearray(frame_bytes)
    captured_payload = b""
    terminated: list[bool] = []
    waited: list[int | None] = []

    class Stdout:
        def read(self, size: int) -> bytes:
            chunk = bytes(remaining[:size])
            del remaining[:size]
            return chunk

        def close(self) -> None:
            pass

    class Process:
        stdout = Stdout()
        returncode = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            terminated.append(True)

        def wait(self, timeout: int = 0) -> None:
            waited.append(timeout)

    def launch(args: list[str], *, stdout: int, stderr: int) -> Process:
        del stdout, stderr
        nonlocal captured_payload
        captured_payload = Path(args[args.index("-i") + 1]).read_bytes()
        return Process()

    archive = tmp_path / "video.tar"
    with tarfile.open(archive, "w") as tar:
        webm_payload = b"wrong-payload"
        webm = tarfile.TarInfo("clip.webm")
        webm.size = len(webm_payload)
        tar.addfile(webm, io.BytesIO(webm_payload))
        mp4_payload = b"clip-payload"
        mp4 = tarfile.TarInfo("clip.mp4")
        mp4.size = len(mp4_payload)
        tar.addfile(mp4, io.BytesIO(mp4_payload))

    processor = DecodeVideo(
        DecodeVideo.Config(
            ffmpeg_exe="fake-ffmpeg",
            known_formats=["webm", "mp4"],
        ),
    )
    with (
        TarFileHandle(archive, use_mmap=False) as handle,
        patch(
            "priml.data.processors.bytes.subprocess.Popen",
            side_effect=launch,
        ),
        patch(
            "priml.data.processors.bytes.rgb2float",
            wraps=rgb2float,
        ) as normalize,
    ):
        sample = cast(
            DecodeVideo.Input,
            {
                "_tar_handle": handle,
                "key": "clip",
                "format": "mp4",
                "frames": frames,
                "height": height,
                "width": width,
                "target_frames": 4,
                "target_height": height,
                "target_width": width,
            },
        )
        result = next(processor(_as_stream([sample])))

    tensor = result.get("media_tensor")
    assert tensor is not None
    assert tensor.shape == (3, 4, height, width)
    torch.testing.assert_close(
        tensor[:, 0, 0, 0],
        torch.tensor([1.0, -1.0, -1.0], dtype=torch.float16),
    )
    torch.testing.assert_close(
        tensor[:, 1, 0, 0],
        torch.tensor([-1.0, 1.0, -1.0], dtype=torch.float16),
    )
    torch.testing.assert_close(
        tensor[:, 2:, 0, 0],
        torch.tensor([[-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]], dtype=torch.float16),
    )
    normalize.assert_called_once()
    assert normalize.call_args.kwargs == {"inplace": True}
    assert captured_payload == b"clip-payload"
    assert terminated == [True]
    assert waited == [5]
    assert "_tar_handle" not in result


def test_video_decoder_invokes_ffmpeg_with_the_output_contract() -> None:
    height, width = 4, 6
    stdout_bytes = bytearray(bytes([255, 0, 0]) * height * width * 2)
    captured_args: list[str] = []
    captured_payload = b""
    captured_options: dict[str, object] = {}

    class Stdout:
        def read(self, size: int) -> bytes:
            chunk = bytes(stdout_bytes[:size])
            del stdout_bytes[:size]
            return chunk

        def close(self) -> None:
            pass

    class Process:
        stdout = Stdout()
        returncode = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout: int = 0) -> None:
            del timeout

    def launch(
        args: list[str],
        *,
        stdout: int,
        stderr: int,
    ) -> Process:
        nonlocal captured_payload
        captured_args.extend(args)
        captured_payload = Path(args[args.index("-i") + 1]).read_bytes()
        captured_options.update(stdout=stdout, stderr=stderr)
        return Process()

    processor = DecodeVideo(DecodeVideo.Config(ffmpeg_exe="chosen-ffmpeg"))
    with patch(
        "priml.data.processors.bytes.subprocess.Popen",
        side_effect=launch,
    ):
        result = processor._decode(
            b"video-payload",
            height=height,
            width=width,
            keep_frames=2,
        )

    assert result is not None
    assert result.shape == (3, 2, height, width)
    # ``-threads`` and ``-xerror`` precede ``-i``: placed after it, ``-threads``
    # applied to the output encoder, not the decoder.
    assert captured_args == [
        "chosen-ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-threads",
        "1",
        "-i",
        captured_args[7],
        "-pix_fmt",
        "rgb24",
        "-vcodec",
        "rawvideo",
        "-f",
        "image2pipe",
        "-vf",
        "scale=6:4:out_range=full",
        "-",
    ]
    assert Path(captured_args[7]).name.startswith("bytes-")
    assert Path(captured_args[7]).suffix == ".mp4"
    assert captured_payload == b"video-payload"
    assert captured_options == {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.DEVNULL,
    }


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
