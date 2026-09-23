"""Resize dimension calculation processor.

Calculates target resize dimensions for samples based on aspect-ratio bucketing.
Does not perform actual resizing - just adds target dimension fields to samples.
"""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict, cast

import subprocess
import tempfile

from configgle import Fig
from turbojpeg import TurboJPEG

import imageio_ffmpeg

from priml.data.pipeline.dataset import add_filter_reason_typed
from priml.image import get_dimensions
from priml.math.pixel import (
    decode_image_pil,
    decode_jpeg_turbojpeg,
    decode_webp_libwebp,
    rgb2float,
)


if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor

    import torch

    from priml.data.sources.tarhandle import TarFileProtocol
else:
    # Defer the ~1s torch import: only video decode builds a tensor, so the
    # image processors in this module never pay for it.
    from wrapt import lazy_import

    torch = lazy_import("torch")


__all__ = [
    "CropDuringDecodeImage",
    "DecodeVideo",
    "GetBytesFromFile",
    "GetBytesFromTarHandle",
    "GetDimensionsFromBytes",
]


class GetDimensionsFromBytes:
    """Extract image dimensions from media bytes without decoding."""

    class Config(Fig["GetDimensionsFromBytes"]):
        pass

    def __init__(self, config: Config):
        pass

    class Input(TypedDict, total=False):
        media: bytes

        height: int
        """Present means already measured; this processor then passes through."""

        width: int
        """Present means already measured; this processor then passes through."""

    # Height and width are both the fields read and the fields added, which a
    # TypedDict cannot say twice.
    Output = Input

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Extract dimensions from media bytes.

        The format is sniffed from the bytes themselves, so no ``format``
        field is consulted.

        Requires:
          - media: bytes - raw image bytes

        Adds:
          - height: int - image height
          - width: int - image width

        """
        for sample in samples:
            # Positive, not merely truthy: a negative cached dimension is
            # metadata that rode along on the sample, and taking it on trust
            # skips the measurement that would have corrected it.
            height, width = sample.get("height"), sample.get("width")
            if height is not None and width is not None and height > 0 < width:
                yield sample
                continue

            media = sample.get("media")
            if media is None:
                yield sample
                continue

            dims = get_dimensions(media)
            if dims is not None:
                sample["height"], sample["width"] = dims

            yield sample


class GetBytesFromFile:
    """Read raw bytes from file path and store in 'media' field.

    Simple processor that reads file bytes from disk and stores them
    in the sample's 'media' field. This allows downstream processors like
    CropDuringDecodeImage to decode from the 'media' bytes.
    """

    class Config(Fig["GetBytesFromFile"]):
        pass

    def __init__(self, config: Config):
        pass

    class Input(TypedDict, total=False):
        """Input required by GetBytesFromFile."""

        file_path: str
        key: str

    class Output(Input):
        """Output produced by GetBytesFromFile."""

        media: bytes

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Read bytes from file path and store in 'media' field.

        Requires:
          - file_path: str - path to file on disk

        Adds:
          - media: bytes - raw file bytes

        Skips sample if file cannot be read.
        """
        for sample in samples:
            sample = cast(GetBytesFromFile.Output, sample)
            # Skip if media already present.
            if sample.get("media"):
                yield sample
                continue

            file_path = sample.get("file_path")
            if file_path is None:
                yield sample
                continue

            try:
                with Path(file_path).open("rb") as f:
                    sample["media"] = f.read()
            except OSError as error:
                # Yielded with a reason rather than dropped: every sibling
                # reports what it could not handle, and a silently vanished
                # sample is missing from the filter accounting too.
                add_filter_reason_typed(
                    sample,
                    "GetBytesFromFile",
                    f"unreadable: {error}",
                )
            yield sample


class GetBytesFromTarHandle:
    """Read raw bytes from tar handle and store in 'media' field.

    Simple processor that extracts file bytes from a tar archive and stores them
    in the sample's 'media' field. This allows downstream processors like
    CropDuringDecodeImage and DecodeVideo to decode from the 'media' bytes instead
    of accessing the tar handle directly.

    Tries multiple format extensions if format is not specified.
    """

    class Config(Fig["GetBytesFromTarHandle"]):
        known_formats: list[str] = field(default_factory=lambda: ["jpg", "webp", "mp4"])
        """Extensions tried in order when the sample names no ``format``."""

    class Input(TypedDict, total=False):
        """Input required by GetBytesFromTarHandle."""

        _tar_handle: TarFileProtocol
        key: str

        format: str
        """Extension to read; empty tries each of ``known_formats`` in turn."""

    class Output(Input):
        """Output produced by GetBytesFromTarHandle."""

        media: bytes

        tar_offset: int
        """Offset of the member's data, recorded for downstream consumers.

        Never read back here to address the archive: tarfile reports it against
        the DECOMPRESSED stream, so for a compressed shard it points nowhere in
        the file on disk and a seek there returns unrelated bytes of the right
        length. Reading through the handle is also faster in both modes (4.6x
        mmap, 1.5x standard, median of 7 trials over 300 members)."""

        tar_size: int
        """Byte length of the member's data, paired with ``tar_offset``."""

    def __init__(self, config: Config):
        self.known_formats = config.known_formats

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Read bytes from tar handle and store in 'media' field.

        Requires:
          - _tar_handle: tarfile.TarFile - tar file handle
          - key: str - file key/filename (without extension)
          - format: str (optional) - file format/extension to try

        Adds:
          - media: bytes - raw file bytes

        Skips sample if file cannot be found or read.
        """
        for sample in samples:
            sample = cast(GetBytesFromTarHandle.Output, sample)
            if self._read_one(sample):
                sample.pop("_tar_handle", None)
            yield sample

    def _read_one(self, sample: GetBytesFromTarHandle.Output) -> bool:
        """Populate ``media`` and report whether this reader finished routing."""
        if sample.get("media"):
            return True

        # Reported, unlike the format skip below: a sample with no handle or
        # no key cannot be read by ANY reader, so it has left the pipeline and
        # must appear in the filtered totals. Returning silently made it
        # indistinguishable from data that was never there.
        tar_handle = sample.get("_tar_handle")
        key = sample.get("key")
        if tar_handle is None or key is None:
            missing = "_tar_handle" if tar_handle is None else "key"
            add_filter_reason_typed(sample, type(self).__name__, f"missing: {missing}")
            return True

        # NOT reported: a format outside this reader's allowlist is ROUTING,
        # not filtering. The sample is intact and a sibling reader that claims
        # the format still handles it, so marking it filtered would count a
        # live sample as dropped.
        fmt = sample.get("format")
        if fmt and fmt not in self.known_formats:
            return False
        extensions = [fmt] if fmt else self.known_formats

        # Try each extension (sequential search)
        for ext in extensions:
            try:
                member = tar_handle.getmember(f"{key}.{ext}")
                file_obj = tar_handle.extractfile(member)
                if file_obj is not None:
                    sample["media"] = file_obj.read()
                    # Capture tar offset metadata for O(1) server lookups.
                    sample["tar_offset"] = int(member.offset_data)
                    sample["tar_size"] = int(member.size)
                    return True
            except (KeyError, OSError):
                continue
        add_filter_reason_typed(sample, type(self).__name__, f"member_not_found: {key}")
        return True


class CropDuringDecodeImage:
    """Decode images to uint8 tensor with optional center crop during decode.

    For JPEG: Uses PyTurboJPEG for true crop-during-decode (efficient).
    For WebP: Uses libwebp for true crop-during-decode (efficient).

    If target_height/target_width are present, crops during decode to match target
    aspect ratio using the formula:
        target_aspect = target_width / target_height
        crop_height = min(actual_height, actual_width / target_aspect)
        crop_width = min(actual_width, actual_height * target_aspect)

    This preserves maximum resolution while matching target aspect ratio.
    Use Interpolate processor afterward to interpolate to exact target dimensions.
    Yields sample unchanged if format not in known_formats or target_frames != 1.
    """

    class Config(Fig["CropDuringDecodeImage"]):
        channels_format: Literal["rgb", "rgba"] = "rgb"
        """Channels to decode to. ``rgba`` forces the PIL path, since the
        turbojpeg and libwebp fast paths produce RGB only."""

        known_formats: list[str] = field(default_factory=lambda: ["jpg", "webp"])
        """Extensions tried in order when the sample names no ``format``."""

        use_turbojpeg: bool = True
        """Decode JPEG with PyTurboJPEG, which crops DURING the decode."""

        scale_to_target: bool = False
        """Let the JPEG IDCT downscale a crop by 1/2, 1/4, or 1/8 while it stays
        at least the sample's ``target_height``/``target_width``.

        The pixels a later resize would discard are never produced; the region
        snaps outward to the reduced grid, a sub-pixel shift. Off by default
        because the output then depends on the target, not only the crop."""

        use_libwebp: bool = True
        """Decode WebP with libwebp rather than falling through to PIL."""

        device: torch.device | str | None = None
        """Device the decoded tensor lands on; ``None`` leaves it on CPU."""

        dtype: torch.dtype | None = None
        """Width the decoded tensor is cast to; ``None`` keeps uint8.

        A floating-point width also rescales ``[0, 255]`` to ``[-1, 1]``, the
        range ``DecodeVideo`` emits, so one downstream consumer can read
        ``media_tensor`` from either decoder."""

    class Input(TypedDict, total=False):
        """Input required by CropDuringDecodeImage."""

        media: bytes
        """Raw image bytes, from ``GetBytesFromTarHandle``."""

        format: str
        """Extension such as ``jpg`` or ``webp``; empty tries every known one."""

        height: int
        """Source image height."""

        width: int
        """Source image width."""

        crop: tuple[int, int] | tuple[int, int, int, int]
        """``(h, w)`` for a centered crop, or ``(y, x, h, w)`` for a placed one."""

        frames: int
        """Source frame count; anything but 1 belongs to ``DecodeVideo``."""

        target_frames: int
        """Planned frame count; read only to route videos away from here."""

        target_height: int
        """Height a later resize produces; read only under ``scale_to_target``."""

        target_width: int
        """Width a later resize produces; read only under ``scale_to_target``."""

    class Output(Input):
        """Output produced by CropDuringDecodeImage."""

        media_tensor: NotRequired[Tensor]
        """``(C, F, H, W)``: uint8 in ``[0, 255]``, or the configured
        floating-point ``dtype`` in ``[-1, 1]``. An image has ``F=1``, but
        ``F=1`` is not necessarily an image.

        ``NotRequired`` because this processor ADDS the key only on a
        successful decode: an unrecognized format or a rejected byte payload
        yields the sample through untouched. Declaring it required told the
        checker the ``"media_tensor" in sample`` skip-if-already-decoded guard
        was dead, and deleting that guard would decode every sample twice."""

    def __init__(self, config: Config):
        self.channels_format: Literal["rgb", "rgba"] = config.channels_format
        self.known_formats = config.known_formats
        self.use_turbojpeg = config.use_turbojpeg
        self.scale_to_target = config.scale_to_target
        self.use_libwebp = config.use_libwebp
        self.device = config.device
        self.dtype = config.dtype
        # Initialize TurboJPEG decoder (thread-safe, can be reused)
        self.turbo_jpeg = TurboJPEG() if self.use_turbojpeg else None

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Load and convert images to a tensor, with optional crop/resize.

        Requires:
          - media: bytes - raw image bytes (from GetBytesFromTarHandle)
          - format: str (optional) - image format ("jpg" or "webp")
          - height: int - original image height
          - width: int - original image width

        Optional (for cropping):
          - crop: (h, w) for a centered crop, or (y, x, h, w) for a placed
            one. Supplied by ``SetCropFromTargetDimensions``; the target
            dimensions themselves are NOT read here.

        Adds:
          - media_tensor: Tensor - (C, F, H, W); uint8 in [0, 255] by default,
            or the configured floating-point ``dtype`` in [-1, 1]

        Yields sample unchanged if format not recognized or media not present.
        """
        for sample in samples:
            sample = cast(CropDuringDecodeImage.Output, sample)
            if "media_tensor" in sample:
                yield sample
                continue
            fields = self._extract_fields(sample)
            if fields is None:
                yield sample
                continue
            (
                media_bytes,
                fmt,
                height,
                width,
                crop,
            ) = fields

            floor = (
                (sample.get("target_height", 0), sample.get("target_width", 0))
                if self.scale_to_target
                else (0, 0)
            )
            decoded = self._process_image(
                media_bytes,
                fmt,
                height,
                width,
                crop,
                floor=floor,
            )

            if decoded is not None:
                media_tensor, error = decoded
                if media_tensor is None:
                    # The error names WHICH backend rejected the bytes and how
                    # (``jpg_OSError:...``); ``_process_image`` builds it for
                    # exactly this. Dropping it reported a truncated file, an
                    # unreadable codec, and a mislabeled extension identically,
                    # so a bad shard gave the operator nothing to act on.
                    add_filter_reason_typed(
                        sample,
                        type(self).__name__,
                        f"decode_failed: {error}" if error else "decode_failed",
                    )
                else:
                    # Add frames dimension: (C, H, W) -> (C, F=1, H, W)
                    media_tensor = media_tensor.unsqueeze(-3)
                    media_tensor = media_tensor.to(device=self.device)
                    # A float request means the sample is leaving the byte
                    # domain, so it gets the same [-1, 1] DecodeVideo emits.
                    # Downstream reads one field from both, and ``Interpolate``
                    # rescales only what is still uint8 -- so an unscaled float
                    # image reached the model at 100x the video's magnitude.
                    if self.dtype is None:
                        pass
                    elif self.dtype.is_floating_point:
                        media_tensor = rgb2float(
                            media_tensor.to(dtype=self.dtype),
                            inplace=True,
                        )
                    else:
                        media_tensor = media_tensor.to(dtype=self.dtype)
                    sample["media_tensor"] = media_tensor
                    # Set frames to 1 for images (defaults to 1 if missing from parquet)
                    sample["frames"] = 1

            yield sample

    def _extract_fields(
        self,
        sample: CropDuringDecodeImage.Input,
    ) -> (
        tuple[
            bytes,
            str | None,
            int,
            int,
            tuple[int, int] | tuple[int, int, int, int] | None,
        ]
        | None
    ):
        """Extract and validate required fields for image processing."""
        media_bytes = sample.get("media")
        frames = sample.get("frames", 1)
        height = sample.get("height")
        width = sample.get("width")
        target_frames = sample.get("target_frames", 1)
        crop = sample.get("crop")
        fmt = sample.get("format")

        # No bytes yet is a routing outcome, not a defect: an upstream reader
        # may not have run. A sample WITH bytes but no dimensions cannot be
        # decoded, so it is reported rather than passed on silently -- the
        # filter accounting is the only place a vanished sample shows up.
        if media_bytes is None:
            return None
        if height is None or width is None:
            add_filter_reason_typed(sample, type(self).__name__, "missing_dimensions")
            return None

        # Not an image: DecodeVideo handles anything with more than one frame.
        if frames != 1 or target_frames != 1:
            return None

        return (media_bytes, fmt, height, width, crop)

    # For JPEG: Uses PyTurboJPEG's crop() method for true crop-during-decode. For WebP:
    # Uses libwebp for true crop-during-decode. For other formats: Uses PIL.
    #
    # If crop is 4-tuple (y, x, h, w): explicit crop box. If crop is 2-tuple (h, w):
    # center crop to match aspect ratio. If crop is None: no crop.
    def _process_image(
        self,
        image_bytes: bytes,
        fmt: str | None,
        height: int,
        width: int,
        crop: tuple[int, int] | tuple[int, int, int, int] | None,
        *,
        floor: tuple[int, int] = (0, 0),
    ) -> tuple[Tensor, str | None] | tuple[None, str] | None:
        """Decode and process image bytes with optional crop."""
        # Determine file extensions to try.
        if fmt and fmt in self.known_formats:
            extensions = [fmt]
        elif fmt:
            # Format specified but not in our known set - skip without filtering.
            return None
        else:
            extensions = self.known_formats

        # The turbojpeg and libwebp paths decode RGB only, so asking for an
        # alpha channel has to fall through to PIL rather than silently
        # yielding three channels under a four-channel request.
        wants_alpha = self.channels_format == "rgba"

        last_error: str | None = None
        if self.use_turbojpeg and self.turbo_jpeg is None:
            raise ValueError("Expected self.turbo_jpeg is not None.")
        for ext in extensions:
            try:
                # Use PyTurboJPEG for JPEG files.
                if (
                    ext == "jpg"
                    and self.use_turbojpeg
                    and self.turbo_jpeg is not None
                    and not wants_alpha
                ):
                    tensor = decode_jpeg_turbojpeg(
                        image_bytes,
                        self.turbo_jpeg,
                        height,
                        width,
                        crop=crop,
                        channels_first=True,
                        min_height=floor[0],
                        min_width=floor[1],
                    )
                    if tensor is not None:
                        return (tensor, None)
                    last_error = f"{ext}_decode_failed"
                    continue

                # Use libwebp for WebP files.
                if ext == "webp" and self.use_libwebp and not wants_alpha:
                    tensor = decode_webp_libwebp(
                        image_bytes,
                        height,
                        width,
                        crop=crop,
                        channels_first=True,
                    )
                    if tensor is not None:
                        return (tensor, None)
                    last_error = f"{ext}_decode_failed"
                    continue

                # Fall back to PIL (auto-detects format from bytes)
                tensor = decode_image_pil(
                    image_bytes,
                    height,
                    width,
                    crop=crop,
                    channels_format=self.channels_format,
                    channels_first=True,
                )
                if tensor is not None:
                    return (tensor, None)
                last_error = f"{ext}_decode_failed"
                continue

            except (
                OSError,
                ValueError,
                RuntimeError,
            ) as e:
                last_error = f"{ext}_{type(e).__name__}:{str(e)[:50]}"
                continue

        # All formats failed - return error message.
        return (None, last_error or "all_formats_failed")


class DecodeVideo:
    """Decode videos to float16 tensor, with optional center crop and resize.

    If target_height/target_width are not present, decodes without resizing.
    Yields sample unchanged if format not in known_formats or frames <= 1.

    Why imageio-ffmpeg, and why the scale is in the filter graph
    -----------------------------------------------------------
    Four backends were benchmarked to the SAME endpoint this class produces --
    ``(C, F, H, W)`` float16 in ``[-1, 1]`` -- because stopping at uint8 flatters
    whichever backend leaves the cast unpaid. Synthetic H.264 clips at 512x512
    (64f), 1280x720 (128f) and 1920x1080 (96f), decoded to 256x256, one decode
    thread per worker, aggregate frames/sec across a process pool:

        library          1w    4w    16w     (720p source, resize during decode)
        imageio_ffmpeg   88   345   1072
        cv2              76   299    938
        pyav             66   263    849
        torchcodec       48   189    645

    Three findings drove the choice:

    1. Resize DURING decode beats resize after, for every backend, by 2.2x at
       1080p and 1.2x at 512px. Decoding 1920x1080 only to discard the pixels
       is the expensive mistake. This is the load-bearing decision; the backend
       matters far less than where the scale happens.
    2. imageio-ffmpeg's resize is nearly free (1.03x at 1080p) because ffmpeg
       scales inside its own filter graph on the same pass. cv2 decodes fastest
       of all (2425 fps native at 720p) but its per-frame ``cv2.resize`` costs
       6.9x at 720p and 8.5x at 1080p, so it loses overall. cv2's apparent win
       on 512->256 was the exact-2x ``INTER_AREA`` fast path: at 512->250 it
       drops 3.4x, so that advantage does not generalize.
    3. GPU decode is not usable here. torchcodec's NVDEC path is 11x faster
       single-stream (1933 fps at 1080p) but saturates by 8 workers, OOMs at
       16 on an idle GPU, competes with training for memory, and cannot resize
       during decode at all -- ``transforms=`` raises "Transforms are only
       supported for CPU devices". CPU scaled ~linearly to 16 workers.

    Rejected: decord (fastest per its own README, but last released 2021, ships
    a mistagged wheel that forces a reinstall on every ``uv sync``, dmlc/decord#356,
    and upstream declined to fix); ``imageio.v3.imread``, which is 5-8x slower
    than this pipe and whose FFMPEG plugin rejects ``filter_sequence``, so it
    cannot satisfy finding 1.

    The bundled ffmpeg 7.0.2 was compared against a 2026 git build across
    {512,720,1080} x {1,8,16} workers: 0.96x-1.05x, i.e. no difference. A custom
    binary buys codecs and hardware paths, not speed.
    """

    class Config(Fig["DecodeVideo"]):
        known_formats: list[str] = field(default_factory=lambda: ["mp4"])
        """Extensions tried in order when the sample names no ``format``."""

        ffmpeg_exe: str = ""
        """Ffmpeg binary; empty uses the executable bundled with imageio.

        A custom build is the only way to reach codecs the bundled binary
        omits. The value remains instance-local: constructing one decoder must
        not change which executable another decoder or library call uses.
        """

    class Input(TypedDict, total=False):
        """Input required by DecodeVideo."""

        _tar_handle: TarFileProtocol
        media: bytes
        key: str
        format: str
        """Extension such as ``mp4``; empty tries every known one."""

        frames: int
        height: int
        width: int

        target_frames: int
        """Frames to emit; shorter sources are padded by repeating the last."""

        target_height: int
        target_width: int

    class Output(Input):
        """Output produced by DecodeVideo."""

        media_tensor: NotRequired[Tensor]
        """``(C, F, H, W)`` float16 in ``[-1, 1]``; an image has ``F=1``, but
        ``F=1`` is not necessarily an image.

        ``NotRequired`` because this processor ADDS the key only on a
        successful decode: an unrecognized format or a failed ffmpeg read
        yields the sample through untouched. Declaring it required told the
        checker the ``"media_tensor" in sample`` skip-if-already-decoded guard
        was dead, and deleting that guard would decode every sample twice."""

    def __init__(self, config: Config):
        self.known_formats = config.known_formats
        self.ffmpeg_exe = config.ffmpeg_exe or imageio_ffmpeg.get_ffmpeg_exe()

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Load and convert videos to float16 tensor, with optional crop/resize.

        Requires:
          - _tar_handle plus key, OR media: the clip arrives either as an open
            tar member to read, or as bytes an upstream processor already read
          - format: str (optional) - video format ("mp4")
          - frames: int - original frame count
          - height: int - original video height
          - width: int - original video width

        Optional (for resizing):
          - target_frames: int - target frame count (if present)
          - target_height: int - target height (if present)
          - target_width: int - target width (if present)

        Adds:
          - media_tensor: Tensor - (C, F, H, W) float16 video tensor

        Yields sample unchanged if format not recognized or frames <= 1.
        """
        for sample in samples:
            sample = cast(DecodeVideo.Output, sample)

            # Skip if already processed.
            if "media_tensor" in sample:
                yield sample
                continue

            # Extract and validate required fields.
            result = self._extract_fields(sample)
            if result is None:
                yield sample
                continue

            (
                tar_handle,
                key,
                fmt,
                frames,
                height,
                width,
                target_frames,
                target_height,
                target_width,
            ) = result

            media_tensor = self._process_video(
                tar_handle,
                key,
                fmt,
                frames=frames,
                height=height,
                width=width,
                target_frames=target_frames,
                target_height=target_height,
                target_width=target_width,
                media=sample.get("media", b""),
            )

            if media_tensor is not None:
                sample["media_tensor"] = media_tensor
            # Decoded or not, the archive is no longer needed: keeping the
            # handle pins it open for as long as the sample sits in the
            # output queue, which its sibling readers already avoid.
            if "_tar_handle" in sample:
                del sample["_tar_handle"]

            yield sample

    def _extract_fields(
        self,
        sample: DecodeVideo.Input,
    ) -> (
        tuple[
            TarFileProtocol | None,
            str,
            str | None,
            int,
            int,
            int,
            int | None,
            int | None,
            int | None,
        ]
        | None
    ):
        """Extract and validate required fields for video processing."""
        tar_handle = sample.get("_tar_handle")
        key = sample.get("key")
        frames = sample.get("frames")
        height = sample.get("height")
        width = sample.get("width")
        target_frames = sample.get("target_frames")
        target_height = sample.get("target_height")
        target_width = sample.get("target_width")
        fmt = sample.get("format")

        # A key addresses a tar member. Once the bytes are already present,
        # neither the address nor an open archive is part of the decode.
        if not sample.get("media") and (tar_handle is None or key is None):
            return None
        if frames is None or height is None or width is None:
            return None

        # Only handle multi-frame videos (frames > 1)
        if frames <= 1:
            return None

        # If any resize dimension is present, all three must be present.
        resize_present = [
            target_frames is not None,
            target_height is not None,
            target_width is not None,
        ]
        if any(resize_present) and not all(resize_present):
            return None

        # A non-positive target is a planner bug, not a request for an empty
        # clip: the decode loop breaks on ``len(chunks) >= keep_frames``, which
        # is already true after one frame, so a zero target silently produced a
        # one-frame tensor rather than nothing or an error.
        if (
            (target_frames is not None and target_frames < 1)
            or (target_height is not None and target_height < 1)
            or (target_width is not None and target_width < 1)
        ):
            add_filter_reason_typed(
                sample,
                type(self).__name__,
                f"invalid_target:f={target_frames}_h={target_height}_w={target_width}",
            )
            return None

        return (
            tar_handle,
            key or "",
            fmt,
            frames,
            height,
            width,
            target_frames,
            target_height,
            target_width,
        )

    # See the class docstring for why the scale lives in ffmpeg's filter graph rather
    # than in a tensor op afterwards.
    def _process_video(
        self,
        tar_handle: TarFileProtocol | None,
        key: str,
        fmt: str | None,
        *,
        frames: int,
        height: int,
        width: int,
        target_frames: int | None,
        target_height: int | None,
        target_width: int | None,
        media: bytes = b"",
    ) -> Tensor | None:
        """Decode a video, resizing DURING decode, to the output contract."""
        # The allowlist is checked before the payload is chosen, not inside
        # ``_read_media``: ``media or _read_media(...)`` short-circuits, so a
        # sample that already carried bytes reached the decoder with a format
        # this processor does not claim to handle.
        if fmt and fmt not in self.known_formats:
            return None
        payload = media or self._read_media(tar_handle, key, fmt)
        if not payload:
            return None
        out_height = target_height if target_height is not None else height
        out_width = target_width if target_width is not None else width
        keep = target_frames if target_frames is not None else frames
        return self._decode(
            payload,
            height=out_height,
            width=out_width,
            keep_frames=keep,
        )

    def _read_media(
        self,
        tar_handle: TarFileProtocol | None,
        key: str,
        fmt: str | None,
    ) -> bytes:
        """Return the clip's bytes from the tar, or empty when unavailable."""
        if tar_handle is None:
            return b""
        extensions = [fmt] if fmt else list(self.known_formats)
        for ext in extensions:
            try:
                member = tar_handle.getmember(f"{key}.{ext}")
                file_obj = tar_handle.extractfile(member)
                if file_obj is not None:
                    return file_obj.read()
            except (KeyError, OSError):
                continue
        return b""

    def _decode(
        self,
        payload: bytes,
        *,
        height: int,
        width: int,
        keep_frames: int,
    ) -> Tensor | None:
        """Run ffmpeg over ``payload`` and assemble the contract tensor."""
        # Ffmpeg needs a seekable input for a faststart-less mp4. The subprocess
        # is owned here so its executable remains instance-local rather than
        # leaking through IMAGEIO_FFMPEG_EXE.
        with tempfile.NamedTemporaryFile(prefix="bytes-", suffix=".mp4") as handle:
            _ = handle.write(payload)
            handle.flush()
            scale = (
                f"scale={width}:{height}"
                ":in_color_matrix=bt709:out_color_matrix=bt709"
                ":in_range=tv:out_range=full"
            )
            try:
                process = subprocess.Popen(  # noqa: S603 -- Media probing invokes the trusted local decoder command.
                    [
                        self.ffmpeg_exe,
                        "-v",
                        "error",
                        "-i",
                        handle.name,
                        "-pix_fmt",
                        "rgb24",
                        "-vcodec",
                        "rawvideo",
                        "-f",
                        "image2pipe",
                        "-vf",
                        scale,
                        "-threads",
                        "1",
                        "-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                return None
            if process.stdout is None:
                raise ValueError("Expected process.stdout is not None.")
            stdout = process.stdout
            frame_bytes = height * width * 3
            chunks: list[bytes] = []
            try:
                for _ in range(keep_frames):
                    chunk = stdout.read(frame_bytes)
                    if not chunk:
                        break
                    if len(chunk) != frame_bytes:
                        return None
                    chunks.append(chunk)
            except (OSError, subprocess.SubprocessError):
                return None
            finally:
                stdout.close()
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

        if not chunks:
            return None
        # A source shorter than the request is padded by repeating its last
        # frame. ``keep_frames`` is a TARGET, not a ceiling: the planner rounds
        # it up to the temporal compression multiple so a latent encoder can
        # stride it evenly, and returning fewer frames silently defeats that.
        # Edge-repeat rather than zero-fill -- black frames are content the clip
        # never had.
        chunks.extend([chunks[-1]] * (keep_frames - len(chunks)))
        raw = bytearray(b"".join(chunks))
        # Checked rather than left to ``view``: this function promises
        # ``Tensor | None`` and its ``try`` above exists to make an unreadable
        # clip a None, but the reshape sits OUTSIDE it -- so a stream whose
        # bytes are not a whole number of frames raised ``RuntimeError`` from a
        # function documented not to, killing the run instead of dropping one
        # sample. ffmpeg is told the frame size, so a mismatch means a
        # truncated read rather than a caller error.
        if len(raw) != len(chunks) * height * width * 3:
            return None
        stacked = torch.frombuffer(raw, dtype=torch.uint8).view(
            len(chunks),
            height,
            width,
            3,
        )
        # (F, H, W, C) -> (C, F, H, W), scaled from [0, 255] to [-1, 1].
        scaled = rgb2float(stacked.to(torch.float16), inplace=True)
        return scaled.permute(3, 0, 1, 2).contiguous()
