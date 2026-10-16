"""Batching processors for data pipeline.

Processors for batching and unbatching samples in data pipelines.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from dataclasses import field
from typing import Final, cast

import copy
import logging
import time

from configgle import Fig
from torch import Tensor

import numpy as np
import torch


logger = logging.getLogger(__name__)


_ShapeKey = tuple[tuple[int, ...] | None, ...]


__all__ = [
    "BATCHED_FIELDS_KEY",
    "Batcher",
    "Unbatcher",
]


BATCHED_FIELDS_KEY: Final = "_batched_fields"
"""Batch key listing the fields indexed by sample along their leading axis.

The record of which fields are per-sample: the ``Unbatcher`` splits exactly
these, and a stage that adds or drops samples (``DecodeCropResizeBatch``)
filters exactly these. A stage writing a new per-sample field appends its name.
"""


class Batcher:
    """Batch samples by the shapes of their ``field_names`` tensors.

    Queues samples by their exact tensor dimensions -- ``media_tensor`` in
    decoder layout ``(C, F, H, W)`` by default -- and stacks each full queue
    into a batch with a leading ``B`` axis.

    Samples with filter_reasons are immediately yielded without batching.
    Original samples are preserved in 'raw' field for unbatching, excluding
    batched fields to save memory.

    Pipeline usage:
        Batcher -> GPU processors -> Unbatcher

    Example:
        # Before: individual samples
        sample1 = {"key": "a", "media_tensor": tensor1}  # Shape: (3, 1, 512, 512)
        sample2 = {"key": "b", "media_tensor": tensor2}  # Shape: (3, 1, 512, 512)

        # After Batcher: batched samples
        batch = {
            "media_tensor": stacked_tensor,  # Shape: (2, 3, 1, 512, 512)
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
        if config.size < 1:
            raise ValueError(f"size must be positive; got {config.size}.")
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
          - raw: list[dict] - original samples for unbatching, always present
          - _batch_size: int - number of samples in the batch
          - BATCHED_FIELDS_KEY: list[str] - the per-sample fields, raw included

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
            avg_batch_size = num_flushed_samples / num_flushed_batches
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

        # ``raw`` is written even when every sample's fields were stacked: it
        # is what marks this dict as a batch for the Unbatcher and for batched
        # processors, and an absent one let a whole batch pass as one sample.
        batch: Batcher.Output = {
            "_batch_size": batch_size,
            "raw": [
                {k: v for k, v in s.items() if k not in self.field_names}
                for s in samples
            ],
        }
        # Every field indexed by sample, recorded rather than inferred from
        # shape: a stacked ``(B,)`` label and a batch-level ``(B,)`` vector look
        # alike, and so do a per-sample list and a batch-level one of length B.
        batched_fields = ["raw"]

        # Stack all fields specified in field_names.
        for field_name in self.field_names:
            # ``None`` is kept as a placeholder: dropping it shortened the list
            # and misaligned every later sample's value.
            field_values = [sample.get(field_name) for sample in samples]
            tensor_values = [v for v in field_values if isinstance(v, Tensor)]
            if tensor_values and len(tensor_values) != len(field_values):
                raise TypeError(
                    f"Field '{field_name}' has mixed tensor/non-tensor values",
                )
            batched_fields.append(field_name)
            if tensor_values:
                t0 = time.perf_counter()
                batched_tensor = torch.stack(tensor_values)
                t1 = time.perf_counter()
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
                batch[field_name] = field_values

        batch[BATCHED_FIELDS_KEY] = batched_fields
        return batch


class Unbatcher:
    """Unbatch samples, distributing batch results to individuals.

    Companion to Batcher. Distributes batched results from GPU processing
    back to individual samples that were preserved in 'raw' field.

    Samples with filter_reasons are passed through unchanged (they were
    never batched).

    Every field named in ``BATCHED_FIELDS_KEY`` is indexed by sample: list or
    tensor, element ``i`` goes to sample ``i``. An untagged field -- a later
    stage's output -- is split when it is a tensor/array with ``ndim >= 2``
    and a leading axis of the batch size (also per value of a dict field), and
    is otherwise replicated, each sample getting its own shallow copy.

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
          sample: One per ``raw`` entry, with the batch results merged in.

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
        raw_samples = sample["raw"]
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
        batched_value = sample.get(BATCHED_FIELDS_KEY, [])
        message = f"{BATCHED_FIELDS_KEY} must be a list of strings"
        if not isinstance(batched_value, list):
            raise TypeError(message)
        typed_batched_value = cast(list[object], batched_value)
        if not all(isinstance(name, str) for name in typed_batched_value):
            raise TypeError(message)
        batched_fields = {name for name in typed_batched_value if isinstance(name, str)}

        reserved = ("raw", "filter_reasons", "_batch_size", BATCHED_FIELDS_KEY)
        for idx, raw_sample in enumerate(typed_raw_samples):
            output_sample = dict(raw_sample)

            # Copy batch-level filter_reasons.
            if isinstance(batch_filter_reasons, Iterable):
                self._merge_filter_reasons(
                    output_sample,
                    cast(Iterable[object], batch_filter_reasons),
                )

            for key, value in list(sample.items()):
                if key in reserved:
                    continue
                output_sample[key] = self._distribute_field(
                    value,
                    idx,
                    size_value,
                    is_batched=key in batched_fields,
                )

            yield output_sample

    def _merge_filter_reasons(
        self,
        output_sample: dict[str, object],
        batch_filter_reasons: Iterable[object],
    ) -> None:
        """Merge batch-level filter reasons into output sample."""
        existing_reasons = output_sample.get("filter_reasons")
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
        is_batched: bool,
    ) -> object:
        """Distribute a batch field value to an individual sample."""
        if is_batched and isinstance(value, list):
            return cast(list[object], value)[idx]
        if is_batched and isinstance(value, (Tensor, np.ndarray)):
            return value[idx]

        # Handle dict fields (e.g., embeddings keyed by frame index)
        if isinstance(value, dict):
            return self._distribute_dict_field(
                cast(dict[object, object], value),
                idx,
                size,
            )

        # Untagged output of a later stage, with a leading batch axis.
        if isinstance(value, (Tensor, np.ndarray)) and self._has_batch_dimension(
            value,
            size,
        ):
            return value[idx]

        # Batch-level metadata. A shallow copy per sample, so mutating one
        # sample's value cannot reach its siblings.
        return copy.copy(value)

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

    # Applies only to fields the batch does not tag. A per-sample output carries the
    # sample's own feature dimensions under the leading batch axis, so it has
    # ``ndim >= 2``; a bare untagged ``(B,)`` tensor is replicated. A stage writing a
    # per-sample ``(B,)`` field must tag it in ``BATCHED_FIELDS_KEY``.
    def _has_batch_dimension(self, value: object, size: int) -> bool:
        """Check whether a stacked tensor/array is per-sample and should split."""
        if not isinstance(value, (Tensor, np.ndarray)):
            return False
        # `ndarray.shape` is an untyped tuple in the stub, so compare via int().
        return value.ndim >= 2 and int(value.shape[0]) == size
