"""Tensorization and device transfer processors with stream synchronization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import field
from typing import TYPE_CHECKING, cast

from configgle import Fig
from torch import Tensor

import numpy as np
import torch


if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = [
    "AsTensor",
    "StreamSync",
]


class AsTensor:
    """Convert values to tensors and transfer to device with optimal transfers.

    Tensorizes numeric values (numbers, numeric lists, numpy arrays) using
    torch.as_tensor, then transfers to target device. Dicts are descended into,
    as are lists that mix tensors and numbers; strings and ``None`` are left
    alone, as is every ``_``-prefixed bookkeeping key (``_batch_size``) and the
    ``stream_field``. For optimal bandwidth:
    - Preserves source dtype during tensorization
    - Converts dtype on target device (e.g., uint16 → float32 on GPU)
    - Shares memory with numpy arrays when possible
    - Optional pinned memory for faster CPU→GPU transfers

    Supports async transfers via non_blocking. Can reuse existing CUDA streams.

    Example:
        # Tensorize labels and transfer images
        cfg.processors = [
            Batcher.Config(field_names=["image", "label"]),  # label is list[int]
            AsTensor.Config(device="cuda", dtype=torch.float32),  # → both on GPU
        ]

        # Pinned memory for faster transfers
        cfg.processors = [
            AsTensor.Config(device="cuda", pin_memory=True),  # Pin before transfer
        ]

        # Async transfer with stream reuse
        cfg.processors = [
            AsTensor.Config(device="cuda", non_blocking=True),  # Creates stream
            GPUProcessor.Config(),  # Reuses same stream
            StreamSync.Config(),  # Syncs once
        ]

    """

    class Config(Fig["AsTensor"]):
        """Configuration for AsTensor processor."""

        device: torch.device | str | None = None
        """Target device (e.g., 'cpu', 'cuda', 'cuda:0'). None = no transfer."""

        dtype: torch.dtype | None = None
        """Optional dtype conversion (done on target device for efficiency)."""

        include: list[str] = field(default_factory=list[str])
        """Field names to process. Empty = process all (unless exclude set)."""

        exclude: list[str] = field(default_factory=lambda: ["raw"])
        """Field names to exclude. Empty = no exclusions (unless include set)."""

        pin_memory: bool = False
        """If True, pin CPU memory before GPU transfer (faster transfers)."""

        non_blocking: bool = False
        """If True, CUDA-bound transfers are async on a stored/reused stream;
        a CPU target always copies blocking, as nothing would sync it."""

        stream_field: str = "pending_tensorizations"
        """Field name for CUDA stream. Creates if missing, reuses if present."""

    def __init__(self, config: Config) -> None:
        self.device = config.device
        self.dtype = config.dtype
        self.include = set(config.include) if config.include else None
        self.exclude = set(config.exclude) if config.exclude else None
        self.pin_memory = config.pin_memory
        self.non_blocking = config.non_blocking
        self.stream_field = config.stream_field

        if (
            self.include is not None
            and self.exclude is not None
            and self.include & self.exclude
        ):
            raise ValueError(
                f"'include' and 'exclude' cannot overlap. "
                f"Overlapping fields: {self.include & self.exclude}",
            )

    def __call__(
        self,
        samples: Iterator[dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        """Convert to tensors and transfer to device.

        Requires:
            - Sample dict with fields to tensorize/transfer

        Yields:
          sample: The input sample, its selected fields tensorized on the target
            device; with ``non_blocking`` and a CUDA target, ``stream_field``
            holds the ``torch.cuda.Stream`` the copies ran on.

        """
        for sample in samples:
            if (
                self.device is not None
                and torch.device(self.device).type.startswith("cuda")
                and self.non_blocking
            ):
                # Reuse existing stream if present, otherwise create new.
                if self.stream_field in sample:
                    stream = sample[self.stream_field]
                    if not isinstance(stream, torch.cuda.Stream):
                        raise TypeError(
                            f"Field '{self.stream_field}' exists but is not a torch.cuda.Stream "
                            f"(got {type(stream).__name__})",
                        )
                else:
                    stream = sample[self.stream_field] = torch.cuda.Stream()

                with torch.cuda.stream(stream):
                    self._apply_transfers(sample)
            else:
                # Sync transfers (ignore stream_field if present)
                self._apply_transfers(sample)
            yield sample

    def _apply_transfers(self, sample: dict[str, object]) -> None:
        """Tensorize and transfer every selected top-level field in place."""
        for key, value in sample.items():
            # ``_``-prefixed keys are pipeline bookkeeping (``_batch_size``, the
            # batched-field tag); converting them broke the Unbatcher reading
            # them next.
            if key.startswith("_") or key == self.stream_field:
                continue
            if self.include is not None and key not in self.include:
                continue
            if self.exclude is not None and key in self.exclude:
                continue
            sample[key] = self._convert(value)

    # Walked by container TYPE, not key type: a dict keyed by frame index is still a
    # dict. A numeric list becomes one tensor; anything else that is not numeric
    # (strings, ``None``) is returned unchanged.
    def _convert(self, value: object) -> object:
        """Return ``value`` with every numeric leaf a transferred tensor."""
        if isinstance(value, dict):
            return {
                k: self._convert(v)
                for k, v in cast(dict[object, object], value).items()
            }
        if isinstance(value, Tensor) or _is_numeric(value):
            return self._transfer(torch.as_tensor(value))
        if isinstance(value, list):
            return [self._convert(v) for v in cast(list[object], value)]
        if isinstance(value, tuple):
            return tuple(self._convert(v) for v in cast(tuple[object, ...], value))
        return value

    def _transfer(self, tensor: Tensor) -> Tensor:
        """Pin and move ``tensor`` as configured; ``as_tensor`` kept its dtype."""
        target = None if self.device is None else torch.device(self.device)
        to_cuda = target is not None and target.type == "cuda"
        # Only a CPU tensor can be pinned: a GPU-resident one (watermark scores
        # deliberately left for this stage) raised here.
        if self.pin_memory and to_cuda and tensor.device.type == "cpu":
            tensor = tensor.pin_memory()
        if target is None and self.dtype is None:
            return tensor
        # A non-blocking copy is only safe to read once a stream is synced;
        # only a CUDA target gets the stream ``StreamSync`` waits on, so a
        # copy back to the host blocks.
        return tensor.to(
            device=target,
            dtype=self.dtype,
            non_blocking=self.non_blocking and to_cuda,
        )


class StreamSync:
    """Synchronize CUDA streams stored in sample fields.

    Waits for async transfers initiated by AsTensor(non_blocking=True)
    to complete, then removes stream fields from samples.

    Example:
        cfg.processors = [
            AsTensor.Config(device="cuda", non_blocking=True),
            ExpensiveCPUWork.Config(),  # Overlaps!
            StreamSync.Config(),  # Block until transfer done
            GPUModel.Config(),  # Safe
        ]

        # Multiple streams
        cfg.processors = [
            AsTensor.Config(device="cuda:0", non_blocking=True, stream_field="stream_gpu0"),
            AsTensor.Config(device="cuda:1", non_blocking=True, stream_field="stream_gpu1"),
            ExpensiveCPUWork.Config(),
            StreamSync.Config(stream_fields=["stream_gpu0", "stream_gpu1"]),
        ]

    """

    class Config(Fig["StreamSync"]):
        """Configuration for StreamSync processor."""

        stream_fields: str | list[str] = "pending_tensorizations"
        """Field name(s) containing torch.cuda.Stream objects to sync."""

    def __init__(self, config: Config) -> None:
        # Normalize to list.
        self.stream_fields = (
            [config.stream_fields]
            if isinstance(config.stream_fields, str)
            else list(config.stream_fields)
        )

    def __call__(
        self,
        samples: Iterator[dict[str, object]],
    ) -> Iterator[dict[str, object]]:
        """Synchronize CUDA streams and remove stream fields.

        Requires:
            - Sample dict potentially containing stream fields

        Yields:
          sample: The input sample with its named streams synchronized and removed.

        """
        if not self.stream_fields:
            yield from samples
            return
        for sample in samples:
            for field_name in self.stream_fields:
                if field_name in sample:
                    stream = sample[field_name]
                    if not isinstance(stream, torch.cuda.Stream):
                        raise TypeError(
                            f"Field '{field_name}' exists but is not a torch.cuda.Stream "
                            f"(got {type(stream).__name__})",
                        )
                    stream.synchronize()
                    del sample[field_name]
            yield sample


def _is_numeric(value: object) -> bool:
    """Whether ``torch.as_tensor`` takes ``value``: a number, array, or numeric nest."""
    if isinstance(value, (bool, int, float, np.ndarray, np.number)):
        return True
    if isinstance(value, (list, tuple)):
        items = cast(Sequence[object], value)
        return bool(items) and all(
            not isinstance(v, np.ndarray) and _is_numeric(v) for v in items
        )
    return False
