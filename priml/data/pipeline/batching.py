"""Batching processors for data pipeline.

Processors for batching and unbatching samples in data pipelines.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import field
from typing import cast

import logging
import time

from configgle import Fig
from torch import Tensor

import numpy as np
import torch


logger = logging.getLogger(__name__)


_ShapeKey = tuple[tuple[int, ...] | None, ...]


__all__ = [
    "Batcher",
    "Unbatcher",
]


class Batcher:
    """Batch samples by media_tensor shape (F, H, W, C).

    Queues samples by their exact media_tensor dimensions, then stacks them
    into batches of shape (B, F, H, W, C) when queue reaches size.

    Samples with filter_reasons are immediately yielded without batching.
    Original samples are preserved in 'raw' field for unbatching, excluding
    batched fields to save memory.

    Pipeline usage:
        Batcher -> GPU processors -> Unbatcher

    Example:
        # Before: individual samples
        sample1 = {"key": "a", "media_tensor": tensor1}  # Shape: (1, 512, 512, 3)
        sample2 = {"key": "b", "media_tensor": tensor2}  # Shape: (1, 512, 512, 3)

        # After Batcher: batched samples
        batch = {
            "media_tensor": stacked_tensor,  # Shape: (2, 1, 512, 512, 3)
            "raw": [{"key": "a"}, {"key": "b"}],  # Batched fields excluded
        }

        # After GPU processing: batch with results
        batch = {
            "media_tensor": stacked_tensor,
            "clip_embeddings": {...},  # Batched embeddings
            "raw": [{"key": "a"}, {"key": "b"}],
        }

        # After Unbatcher: individual samples with results
        sample1 = {"key": "a", "media_tensor": tensor1, "clip_embeddings": {...}}
        sample2 = {"key": "b", "media_tensor": tensor2, "clip_embeddings": {...}}

    """

    class Config(Fig["Batcher"]):
        size: int = 32
        """Number of samples per batch."""

        field_names: list[str] = field(default_factory=lambda: ["media_tensor"])
        """Tensor field names to stack into batches."""

        drop_remainder: bool = False
        """Drop incomplete batches at end of stream."""

        device: torch.device | str | None = None
        """Move batched tensors to this device; ``None`` leaves them put."""

    Input = dict[str, object]

    Output = dict[str, object]
    """A batch: the stacked fields plus ``raw`` and the private bookkeeping keys.

    Not a ``TypedDict``: which fields get stacked is ``Config.field_names``, so
    the key set is chosen per configuration and cannot be named here.
    """

    def __init__(self, config: Config):
        self.size = config.size
        self.field_names = config.field_names
        self.drop_remainder = config.drop_remainder
        # ``is None``, not truthiness: only the unset field means "stay put".
        # A falsy-but-present value ("") is a device torch rejects, and taking
        # it for the no-op silently ran later GPU stages against CPU tensors.
        self.device = None if config.device is None else torch.device(config.device)

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Batch samples by shapes of all tensor field_names.

        Requires:
          - (tensor fields in field_names) - Tensor fields to batch

        Adds:
          - raw: list[dict] - original samples for unbatching

        Yields samples with filter_reasons immediately without batching.

        When ``drop_remainder`` is False, every shape key with a non-empty
        queue is flushed as one partial batch at end of stream; a stream with
        K distinct shapes therefore yields up to K trailing partial batches.

        """
        # Queue state is local to this call so the processor is reentrant.
        queues: dict[_ShapeKey, deque[Batcher.Input]] = defaultdict(deque)
        total_samples_per_shape: dict[_ShapeKey, int] = defaultdict(int)
        num_full_batches = 0

        for sample in samples:
            if sample.get("filter_reasons"):
                yield sample
                continue

            if not all(k in sample for k in self.field_names):
                yield sample
                continue

            # Compute shape key from all tensor field_names.
            shape_key = tuple(
                tuple(field_value.shape)
                if isinstance(field_value := sample.get(name), Tensor)
                else None
                for name in self.field_names
            )

            total_samples_per_shape[shape_key] += 1
            queue = queues[shape_key]
            queue.append(sample)

            # Yield batch when queue is full.
            if len(queue) >= self.size:
                num_full_batches += 1
                yield self._create_batch([queue.popleft() for _ in range(self.size)])

        # Flush remaining samples (unless drop_remainder=True)
        num_flushed_samples = 0
        num_flushed_batches = 0
        if not self.drop_remainder:
            for queue in list(queues.values()):
                if queue:
                    num_flushed_samples += len(queue)
                    num_flushed_batches += 1
                    yield self._create_batch(list(queue))

        logger.info(
            "batched %s samples across %s batches (= %s samples/batch)",
            num_full_batches * self.size,
            num_full_batches,
            self.size,
        )

        if not self.drop_remainder and num_flushed_samples > 0:
            avg_batch_size = (
                num_flushed_samples / num_flushed_batches
                if num_flushed_batches > 0
                else 0
            )
            logger.info(
                "flushed %s samples across %s batches (≈ %.1f samples/batch)",
                num_flushed_samples,
                num_flushed_batches,
                avg_batch_size,
            )

        # Log total samples seen per shape key.
        for shape_key, count in sorted(
            total_samples_per_shape.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            logger.info(
                "samples by shape %s: %s %.1f",
                shape_key,
                f"{count:,}",
                count / self.size,
            )

    def _create_batch(self, samples: list[Input]) -> Output:
        """Create a batched sample by stacking the given samples."""
        batch_size = len(samples)

        # Create batch with preserved raw samples (excluding batched fields)
        raw_samples = [
            {k: v for k, v in s.items() if k not in self.field_names} for s in samples
        ]
        batch: Batcher.Output = {"_batch_size": batch_size}
        if any(raw_samples):
            batch["raw"] = raw_samples

        # Names of fields stacked into per-sample (non-tensor) lists. The
        # Unbatcher splits a list field only when its name appears here, so a
        # batch-level list that merely happens to have length B is never
        # mistaken for per-sample data.
        batched_list_fields: list[str] = []

        # Stack all fields specified in field_names.
        for field_name in self.field_names:
            field_values: list[object] = []
            tensor_values: list[Tensor] = []
            has_non_tensor = False

            for sample in samples:
                field_value = sample.get(field_name)
                if field_value is not None:
                    if isinstance(field_value, Tensor):
                        if has_non_tensor:
                            raise TypeError(
                                f"Field '{field_name}' has mixed tensor/non-tensor values",
                            )
                        tensor_values.append(field_value)
                    else:
                        has_non_tensor = True
                        if tensor_values:
                            raise TypeError(
                                f"Field '{field_name}' has mixed tensor/non-tensor values",
                            )
                    field_values.append(field_value)

            if field_values:
                if tensor_values:
                    t0 = time.perf_counter()
                    # All values should be tensors, stack them on dim 0.
                    batched_tensor = torch.stack(tensor_values)
                    t1 = time.perf_counter()
                    # Move to device if specified.
                    if self.device is not None:
                        batched_tensor = batched_tensor.to(self.device)
                    t2 = time.perf_counter()

                    stack_ms = (t1 - t0) * 1000
                    to_ms = (t2 - t1) * 1000
                    if stack_ms > 100 or to_ms > 100:
                        logger.debug(
                            "Batcher: batch_size=%s, shape=%s, stack=%.1fms, to=%.1fms",
                            batch_size,
                            batched_tensor.shape,
                            stack_ms,
                            to_ms,
                        )

                    batch[field_name] = batched_tensor
                else:
                    # Not tensors, make a list.
                    batch[field_name] = field_values
                    batched_list_fields.append(field_name)

        if batched_list_fields:
            batch["_batched_list_fields"] = batched_list_fields

        return batch


class Unbatcher:
    """Unbatch samples, distributing batch results to individuals.

    Companion to Batcher. Distributes batched results from GPU processing
    back to individual samples that were preserved in 'raw' field.

    Samples with filter_reasons are passed through unchanged (they were
    never batched).

    For each field in the batch (except 'raw' and 'media_tensor'), if the
    field has a batch dimension, it gets split and distributed to individual
    samples.

    Example:
        # Input: batch with embeddings
        batch = {
            "media_tensor": stacked_tensor,  # Shape: (2, ...)
            "clip_embeddings": {0: [[emb1], [emb2]]},  # Batched
            "raw": [
                {"key": "a", "media_tensor": t1},
                {"key": "b", "media_tensor": t2},
            ],
        }

        # Output: individual samples with embeddings distributed
        [
            {"key": "a", "media_tensor": t1, "clip_embeddings": {0: [emb1]}},
            {"key": "b", "media_tensor": t2, "clip_embeddings": {0: [emb2]}},
        ]

    """

    class Config(Fig["Unbatcher"]): ...

    Input = dict[str, object]
    """One batch, as ``Batcher`` produced it -- see ``Batcher.Output``."""

    Output = dict[str, object]

    def __init__(self, config: Config): ...

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Unbatch samples, distributing batch results to individuals.

        Requires:
          - raw: list[dict] - original samples from Batcher (if batched)

        Yields:
          Individual samples with batch results merged in.

        If no 'raw' field is present, passes through unchanged (not batched).
        If 'filter_reasons' is present in the batch, copies them to each unbatched sample.

        """
        for sample in samples:
            # If no 'raw' field, this isn't a batched sample - pass through.
            if "raw" not in sample:
                yield sample
                continue

            # Unbatch and yield individual samples.
            yield from self._unbatch_sample(sample)

    def _unbatch_sample(self, sample: Output) -> Iterator[Output]:
        """Unbatch a single batched sample into individual samples."""
        raw_samples = sample.get("raw", [])
        if not isinstance(raw_samples, list):
            raise TypeError("raw must be a list of sample dictionaries")
        raw_items = cast(list[object], raw_samples)
        if not all(isinstance(raw_sample, dict) for raw_sample in raw_items):
            raise TypeError("raw must be a list of sample dictionaries")
        typed_raw_samples = [
            cast(dict[str, object], raw_sample) for raw_sample in raw_items
        ]
        # Prefer the Batcher's explicit count; fall back to the raw length so
        # batches written before _batch_size existed still unbatch correctly.
        size_value = sample.get("_batch_size", len(typed_raw_samples))
        if not isinstance(size_value, int):
            raise TypeError("_batch_size must be an integer")
        batch_filter_reasons = sample.get("filter_reasons")
        if batch_filter_reasons and not isinstance(batch_filter_reasons, Iterable):
            raise TypeError("filter_reasons must be iterable")
        # Field names the Batcher stacked into per-sample (non-tensor) lists.
        # Only these list fields are split; any other length-B list is treated
        # as batch-level metadata and replicated whole.
        batched_list_value = sample.get("_batched_list_fields", [])
        if not isinstance(batched_list_value, list):
            raise TypeError("_batched_list_fields must be a list of strings")
        typed_batched_list_value = cast(list[object], batched_list_value)
        if not all(isinstance(name, str) for name in typed_batched_list_value):
            raise TypeError("_batched_list_fields must be a list of strings")
        batched_list_fields: set[str] = {
            name for name in typed_batched_list_value if isinstance(name, str)
        }

        for idx, raw_sample in enumerate(typed_raw_samples):
            output_sample = dict(raw_sample)

            # Copy batch-level filter_reasons.
            if isinstance(batch_filter_reasons, Iterable):
                self._merge_filter_reasons(
                    output_sample,
                    cast(Iterable[object], batch_filter_reasons),
                )
            elif batch_filter_reasons:
                raise TypeError("filter_reasons must be iterable")

            # Distribute batch fields to this sample.
            reserved = ("raw", "filter_reasons", "_batch_size", "_batched_list_fields")
            for key, value in list(sample.items()):
                if key in reserved:
                    continue
                output_sample[key] = self._distribute_field(
                    value,
                    idx,
                    size_value,
                    is_batched_list=key in batched_list_fields,
                )

            yield output_sample

    def _merge_filter_reasons(
        self,
        output_sample: dict[str, object],
        batch_filter_reasons: Iterable[object],
    ) -> None:
        """Merge batch-level filter reasons into output sample."""
        existing_reasons = output_sample.get("filter_reasons", [])
        if isinstance(existing_reasons, list):
            output_sample["filter_reasons"] = existing_reasons + list(
                batch_filter_reasons,
            )
        else:
            output_sample["filter_reasons"] = list(batch_filter_reasons)

    def _distribute_field(
        self,
        value: object,
        idx: int,
        size: int,
        *,
        is_batched_list: bool,
    ) -> object:
        """Distribute a batch field value to an individual sample."""
        # Handle dict fields (e.g., embeddings keyed by frame index)
        if isinstance(value, dict):
            return self._distribute_dict_field(
                cast(dict[object, object], value),
                idx,
                size,
            )

        # A list is per-sample only when the Batcher tagged it as one; an
        # untagged length-B list is batch-level metadata, replicated whole.
        if isinstance(value, list):
            value_list = cast(list[object], value)
            return value_list[idx] if is_batched_list else value_list

        # Stacked per-sample tensors/arrays carry a leading batch axis.
        if isinstance(value, (Tensor, np.ndarray)) and self._has_batch_dimension(
            value,
            size,
        ):
            return value[idx]

        # Copy as-is (might be batch-level metadata)
        return value

    def _distribute_dict_field(
        self,
        value: dict[object, object],
        idx: int,
        size: int,
    ) -> dict[object, object]:
        """Distribute a dict field by extracting batch dimensions from values."""
        output_dict: dict[object, object] = {}
        for dict_key, dict_value in value.items():
            if isinstance(
                dict_value,
                (Tensor, np.ndarray),
            ) and self._has_batch_dimension(
                dict_value,
                size,
            ):
                output_dict[dict_key] = dict_value[idx]
            else:
                output_dict[dict_key] = dict_value
        return output_dict

    # A stacked per-sample tensor/array carries the sample's own feature dimensions
    # under the leading batch axis, so it has ``ndim >= 2``. A bare ``(B,)`` tensor is
    # treated as batch-level metadata and replicated rather than split into B scalars,
    # removing the false-positive where unrelated metadata happens to have length B.
    # Lists are decided by the Batcher's explicit ``_batched_list_fields`` tag, not by
    # this method.
    def _has_batch_dimension(self, value: object, size: int) -> bool:
        """Check whether a stacked tensor/array is per-sample and should split."""
        if not isinstance(value, (Tensor, np.ndarray)):
            return False
        # `ndarray.shape` is an untyped tuple in the stub, so compare via int().
        return value.ndim >= 2 and int(value.shape[0]) == size
