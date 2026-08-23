from collections.abc import Generator
from typing import TypedDict

__all__ = ["count_frames_and_secs", "read_frames", "write_frames"]

# First value yielded by ``read_frames``, before any frame bytes. ``size`` is
# the OUTPUT geometry and ``source_size`` the input's, so the two differ
# whenever an output filter scales; code resizing during decode reads ``size``.
class VideoMeta(TypedDict):
    ffmpeg_version: str
    codec: str
    pix_fmt: str
    fps: float
    source_size: tuple[int, int]
    size: tuple[int, int]
    rotate: int
    duration: float

def read_frames(
    path: str,
    pix_fmt: str = ...,
    bpp: int | None = ...,
    input_params: list[str] | None = ...,
    output_params: list[str] | None = ...,
    bits_per_pixel: int | None = ...,
) -> Generator[VideoMeta | bytes]: ...

# The union above is not vagueness: the first value really is metadata and
# every later one really is frame bytes, and no generic parameter can say
# "the first yield differs". Callers narrow with ``isinstance``, which doubles
# as a runtime check that the metadata was not consumed by mistake.
def write_frames(
    path: str,
    size: tuple[int, int],
    pix_fmt_in: str = ...,
    pix_fmt_out: str = ...,
    fps: float = ...,
    quality: int = ...,
    bitrate: int | None = ...,
    codec: str | None = ...,
    macro_block_size: int = ...,
    ffmpeg_log_level: str = ...,
    ffmpeg_timeout: float | None = ...,
    input_params: list[str] | None = ...,
    output_params: list[str] | None = ...,
    audio_path: str | None = ...,
    audio_codec: str | None = ...,
) -> Generator[None, bytes]: ...
def count_frames_and_secs(path: str) -> tuple[int, float]: ...
