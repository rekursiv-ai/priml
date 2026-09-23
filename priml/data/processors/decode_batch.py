"""Decode, crop, and resize a whole batch of JPEGs into one uint8 tensor.

The per-sample stages (``CropDuringDecodeImage`` -> ``Interpolate`` ->
``Batcher``) build a dict and a tensor per image and stack at the end. On two
cores that overhead cost as much as the decode itself: 1037 img/s against
ffcv's 1553 on the same 40k ImageNet images. This stage runs after ``Batcher``
has gathered the bytes and crop boxes, and decodes each image straight into its
slot of one preallocated batch, across threads. libjpeg-turbo and OpenCV both
release the GIL, so the threads run in parallel with no process handoff:
1604 img/s, 1780 with ``scale_to_target``.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

import random

from configgle import Fig
from torch import Tensor

import torch

from priml.image import decode_jpeg_turbojpeg_region


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import cv2
    import numpy as np
else:
    # Defer the ~120 ms OpenCV import to the first decode.
    from wrapt import lazy_import

    cv2 = lazy_import("cv2")


__all__ = ["DecodeCropResizeBatch"]


class DecodeCropResizeBatch:
    """Decode each JPEG's crop and resize it into slot ``i`` of one batch tensor.

    Reads the per-sample lists ``Batcher`` gathers -- ``media`` (bytes),
    ``crop`` (``(y, x, h, w)``), ``target_height``, ``target_width`` -- and
    replaces ``media`` and ``crop`` with ``image``: ``(B, 3, H, W)`` uint8.
    Resizing is ``cv2.INTER_AREA`` both ways, as ffcv-imagenet does. A sample
    that fails to decode is dropped from every per-sample field and counted
    under ``_filter_counts``, so one corrupt file costs one sample.
    """

    class Config(Fig["DecodeCropResizeBatch"]):
        num_threads: int = 2
        """Decode threads; the codecs release the GIL, so these run in parallel."""

        flip_p: float = 0.0
        """Probability each image is mirrored left-right."""

        scale_to_target: bool = False
        """Let the IDCT downscale a crop by 1/2, 1/4, or 1/8 while it stays at
        least the target size. The region snaps outward to the reduced grid, a
        sub-pixel shift, and the full-resolution pixels are never produced."""

        fast_dct: bool = False
        """Decode with libjpeg-turbo's fast integer IDCT, as ffcv does, rather
        than the accurate one; faster, and off from exact by a level or two."""

    def __init__(self, config: Config) -> None:
        if config.num_threads <= 0:
            raise ValueError(f"num_threads must be positive; got {config.num_threads}.")
        self.flip_p = config.flip_p
        self.scale_to_target = config.scale_to_target
        self.fast_dct = config.fast_dct
        self.pool = ThreadPoolExecutor(config.num_threads)

    def __call__(
        self,
        samples: Iterator[dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        """Replace each batch's bytes and boxes with its decoded image tensor.

        Batch N decodes on the pool while this thread pulls batch N+1 from
        upstream, so file reads and box sampling cost no decode time.
        """
        pending: deque[tuple[dict[str, object], Callable[[], None] | None]] = deque()
        for batch in samples:
            try:
                finish = (
                    self._submit(batch)
                    if "media" in batch and "crop" in batch
                    else None
                )
            except _PoolClosedError:
                # A prefetch thread can still be pulling while the interpreter
                # exits, when the pool refuses work: the stream just ends.
                return
            pending.append((batch, finish))
            # Hold back only the newest in-flight batch; order is preserved.
            while pending and (len(pending) > 1 or pending[0][1] is None):
                yield _completed(*pending.popleft())
        while pending:
            yield _completed(*pending.popleft())

    def _submit(self, batch: dict[str, object]) -> Callable[[], None]:
        """Start decoding ``batch``; the returned callable waits and fills it in."""
        media = cast(list[bytes], batch.pop("media"))
        crops = cast(list[tuple[int, int, int, int]], batch.pop("crop"))
        heights = cast(list[int], batch["target_height"])
        widths = cast(list[int], batch["target_width"])
        height, width = heights[0], widths[0]
        # One draw per image, on the caller's thread: the order is fixed by the
        # batch, not by which decode thread finishes first.
        flips = [random.random() < self.flip_p for _ in media]  # noqa: S311 -- Augmentation draw.
        out = torch.empty((len(media), height, width, 3), dtype=torch.uint8)
        try:
            futures = [
                self.pool.submit(
                    self._decode_one,
                    media[index],
                    crops[index],
                    out[index].numpy(),
                    flips[index],
                )
                for index in range(len(media))
            ]
        except RuntimeError as error:
            # ``submit`` raises only once the pool, or the interpreter, shut down.
            raise _PoolClosedError from error

        def finish() -> None:
            ok = [future.result() for future in futures]
            keep = [index for index, good in enumerate(ok) if good]
            # NHWC storage viewed as NCHW is channels_last: no copy, and the
            # layout the model consumes.
            image = out if len(keep) == len(ok) else out[keep]
            batch["image"] = image.permute(0, 3, 1, 2)
            if len(keep) < len(ok):
                _drop_failed(batch, keep=keep, size=len(ok))

        return finish

    def _decode_one(
        self,
        data: bytes,
        crop: tuple[int, int, int, int],
        out: np.ndarray,
        flip: bool,
    ) -> bool:
        """Decode one crop into ``out`` (H, W, 3); return whether it decoded."""
        y, x, h, w = crop
        floor_h, floor_w = out.shape[:2] if self.scale_to_target else (0, 0)
        region = decode_jpeg_turbojpeg_region(
            data,
            x=x,
            y=y,
            w=w,
            h=h,
            min_height=floor_h,
            min_width=floor_w,
            fast_dct=self.fast_dct,
        )
        if region is None:
            return False
        _ = cv2.resize(
            region,
            (out.shape[1], out.shape[0]),
            dst=out,
            interpolation=cv2.INTER_AREA,
        )
        if flip:
            # In place; ``out[:] = out[:, ::-1]`` buffers a reversed copy.
            _ = cv2.flip(out, 1, dst=out)
        return True


class _PoolClosedError(Exception):
    """The decode pool refused a batch because it is shutting down."""


def _completed(
    batch: dict[str, object],
    finish: Callable[[], None] | None,
) -> dict[str, object]:
    if finish is not None:
        finish()
    return batch


def _drop_failed(batch: dict[str, object], *, keep: Sequence[int], size: int) -> None:
    """Remove failed samples from every per-sample field, and count them."""
    listed = set(cast(list[str], batch.get("_batched_list_fields", [])))
    for key, value in list(batch.items()):
        if key in listed and isinstance(value, list):
            batch[key] = [cast(list[object], value)[index] for index in keep]
        elif isinstance(value, Tensor) and value.ndim >= 1 and len(value) == size:
            batch[key] = value[list(keep)]
    batch["_batch_size"] = len(keep)
    counts = cast(dict[str, int], batch.setdefault("_filter_counts", {}))
    reason = f"{DecodeCropResizeBatch.__name__}:decode_failed"
    counts[reason] = counts.get(reason, 0) + size - len(keep)
