"""Tensorization and device transfer processors with stream synchronization."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING, cast

from configgle import Fig
from torch import Tensor

import torch

from priml.lib.traverse import (
    recursively_iterate_over_object_descendants,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = [
    "AsTensor",
    "StreamSync",
]


class AsTensor:
    """Convert values to tensors and transfer to device with optimal transfers.

    Tensorizes non-tensor values (lists, ints, numpy arrays) using torch.as_tensor,
    then transfers to target device. For optimal bandwidth:
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
        """If True, use async transfers and store/reuse CUDA stream."""

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

        if self.include and self.exclude and self.include & self.exclude:
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
            - Sample dict with tensors on target device
            - If non_blocking=True, adds/reuses stream_field with torch.cuda.Stream

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
        """Apply tensorization and device transfers."""
        for path, value in recursively_iterate_over_object_descendants(sample):
            # Skip if path is empty (root object)
            if not path:
                continue

            # Skip leaves inside a sequence: the enclosing list/tuple is itself
            # tensorized as a whole, so descending into its elements would
            # re-tensorize each scalar after the parent was already converted.
            if isinstance(path[-1], int):
                continue

            # Filter by include/exclude.
            field_name = path[0]
            if self.include and field_name not in self.include:
                continue
            if self.exclude and field_name in self.exclude:
                continue
            if isinstance(value, (dict, torch.cuda.Stream)):
                continue

            # Navigate to parent container.
            parent: dict[str, object] | list[object] = sample
            for step in path[:-1]:
                if isinstance(parent, dict):
                    assert isinstance(step, str)
                    child = parent[step]
                else:
                    assert isinstance(step, int)
                    child = parent[step]
                assert isinstance(child, (dict, list))
                parent = cast(dict[str, object] | list[object], child)
            final_key = path[-1]

            # Tensorize and transfer
            # torch.as_tensor preserves dtype (optimal for uint16→float32 on GPU)
            # and shares memory with numpy arrays.
            tensor: Tensor = torch.as_tensor(value)

            # Pin memory before GPU transfer if requested.
            if (
                self.pin_memory
                and self.device is not None
                and torch.device(self.device).type.startswith("cuda")
            ):
                tensor = tensor.pin_memory()

            # Transfer to device and/or convert dtype.
            if self.device is not None or self.dtype is not None:
                transferred = tensor.to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=self.non_blocking,
                )
            else:
                transferred = tensor
            if isinstance(parent, dict):
                parent[final_key] = transferred
            else:
                assert isinstance(final_key, int)
                parent[final_key] = transferred


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
            - Sample dict with streams synchronized and removed

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
