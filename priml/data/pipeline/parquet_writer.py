"""Parquet file processing utilities.

Utilities for reading and writing parquet files in data pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import logging

from configgle import Fig


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from torch import Tensor

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
else:
    # Defer the ~0.4s torch import: Tensor is only touched in the isinstance
    # check inside _convert_to_serializable, so most parquet I/O never needs it.
    from wrapt import lazy_import

    np = lazy_import("numpy")  # ~90 ms; only _convert_to_serializable uses it.
    Tensor = lazy_import("torch", "Tensor")
    pa = lazy_import("pyarrow")  # ~150 ms; only ParquetMergeWriter uses it.
    pq = lazy_import("pyarrow.parquet")  # ~150 ms; only ParquetMergeWriter uses it.


# JSON-serializable types for parquet.
JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


logger = logging.getLogger(__name__)

__all__ = [
    "BufferProcessor",
    "ParquetMergeWriter",
]


class BufferProcessor:
    """Buffer samples in memory and provide access to accumulated results.

    Pass-through processor that yields samples unchanged while also accumulating
    them in a buffer. After the iterator completes, accumulated samples can be
    retrieved via get_buffered_results().

    Supports grouping by a key field - when the key changes, yields samples
    for the previous group. When group_by_key is None, buffers all samples
    until iterator completes.

    Use cases:
    - Batch updates: Collect pipeline results for writing back to storage
    - Multi-pass processing: Accumulate samples for subsequent processing
    - Analytics: Gather all samples for aggregation after pipeline completes
    """

    class Config(Fig["BufferProcessor"]):
        group_by_key: str | None = None
        """Field whose change flushes the buffer; ``None`` buffers everything."""

        key_field: str = "key"
        """Field holding each sample's unique id."""

    Input = dict[str, object]
    Output = Input

    def __init__(self, config: Config):
        self.group_by_key = config.group_by_key
        self.key_field = config.key_field
        self._buffer: dict[str, dict[str, object]] = {}
        self._current_group_value: object = None

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Yield samples unchanged while buffering them.

        Requires:
          - key: str - unique identifier for each sample (configurable via key_field)
          - (group_by_key field if configured) - field to group samples by

        Adds:
          - (none) - transparent pass-through with buffering

        """
        self._buffer = {}
        self._current_group_value = None

        for sample in samples:
            sample_key = sample.get(self.key_field)
            if sample_key is None:
                yield sample
                continue

            if self.group_by_key is not None:
                group_value = sample.get(self.group_by_key)
                if (
                    self._current_group_value is not None
                    and group_value != self._current_group_value
                ):
                    self._buffer = {}
                self._current_group_value = group_value

            assert isinstance(sample_key, str)
            self._buffer[sample_key] = dict(sample)
            yield sample

    def get_buffered_results(self) -> dict[str, dict[str, object]]:
        """Get buffered samples as dict keyed by sample key.

        Returns:
            buffered_results: Dict mapping sample keys to their full sample dicts.

        """
        return dict(self._buffer)


class ParquetMergeWriter:
    """Merge buffered results into existing parquet file.

    Reads an existing parquet file, merges in new field values from buffered
    results (typically from BufferProcessor), and writes atomically using
    .new suffix + rename pattern.

    Use case:
    - Backfill processing: Add computed fields (embeddings, scores) to existing metadata
    - Batch updates: Update multiple rows with new field values
    - Pipeline output: Write processed samples back to original parquet

    Example:
        buffer = BufferProcessor.Config().make()
        for sample in buffer(pipeline):
            process(sample)

        writer = ParquetMergeWriter.Config().make()
        writer.write(parquet_path, buffer.get_buffered_results())

    """

    class Config(Fig["ParquetMergeWriter"]):
        """Configuration for ParquetMergeWriter."""

        key_field: str = "key"
        """Column joining buffered results to existing parquet rows."""

    def __init__(self, config: Config):
        self.key_field = config.key_field

    @classmethod
    def cleanup_stale_temp_files(cls, directory: Path) -> int:
        """Remove orphaned ``*.parquet.new`` temp files left by crashes.

        The atomic-write path crashes between writing the temp file and the
        rename leave a stale ``.parquet.new`` orphan. Scan the directory and
        delete them so a later run does not mistake them for real shards.

        Args:
          directory: Directory to scan for orphaned temp files.

        Returns:
          removed_count: Number of stale temp files removed.

        """
        removed_count = 0
        for stale in directory.glob("*.parquet.new"):
            stale.unlink()
            logger.info("ParquetMergeWriter: Removed stale temp file %s", stale.name)
            removed_count += 1
        return removed_count

    def write(
        self,
        parquet_path: Path,
        buffered_results: dict[str, dict[str, object]],
        overwrite: bool = True,
    ) -> None:
        """Merge buffered results into existing parquet file.

        Reads the existing parquet, merges in new field values from buffered_results,
        and writes atomically using .new suffix + rename.

        Args:
            parquet_path: Path to existing parquet file to update.
            buffered_results: Dict mapping sample keys to field dicts.
            overwrite: If True, rename .new file over original. If False, leave .new file.

        """
        if len(buffered_results) == 0:
            logger.warning(
                "ParquetMergeWriter: No buffered results to write for %s",
                parquet_path.name,
            )
            return

        logger.info(
            "ParquetMergeWriter: Writing %s samples to %s",
            len(buffered_results),
            parquet_path.name,
        )

        # Recover from prior crashes before writing into this directory.
        self.cleanup_stale_temp_files(parquet_path.parent)

        table = pq.read_table(parquet_path)
        keys = cast(list[object], table.column(self.key_field).to_pylist())

        # Harvest field names from all buffered results, excluding
        # non-serializable fields.
        skip_fields = {"_tar_handle", "image", "media_tensor", "caption_tokens"}
        all_fields_set: set[str] = set()
        for sample in buffered_results.values():
            all_fields_set.update(sample.keys())
        field_names = sorted(
            f for f in all_fields_set if not f.startswith("_") and f not in skip_fields
        )

        logger.info("ParquetMergeWriter: Merging fields: %s", field_names)

        # Build new columns via native pyarrow (no pandas).
        for field_name in field_names:
            existing = (
                table.column(field_name).to_pylist()
                if field_name in table.column_names
                else [None] * len(keys)
            )
            values: list[object] = []
            for i, raw_key in enumerate(keys):
                assert isinstance(raw_key, str)
                key = raw_key
                buf = buffered_results.get(key)
                if buf is not None and field_name in buf:
                    value = self._convert_to_serializable(buf[field_name])
                    if isinstance(value, dict):
                        value = {str(k): v for k, v in value.items()} if value else None
                    values.append(value)
                else:
                    values.append(existing[i])
            table = (
                table.drop(field_name) if field_name in table.column_names else table
            )
            table = table.append_column(field_name, pa.array(values))

        # Write atomically.
        new_path = parquet_path.with_suffix(".parquet.new")
        pq.write_table(table, new_path)
        logger.info("ParquetMergeWriter: Wrote %s", new_path.name)
        if overwrite:
            new_path.rename(parquet_path)
            logger.info(
                "ParquetMergeWriter: Renamed %s -> %s",
                new_path.name,
                parquet_path.name,
            )

    def _convert_to_serializable(self, value: object) -> JsonValue:
        """Convert tensors and numpy arrays to lists for parquet serialization."""
        if isinstance(value, Tensor):
            return cast(JsonValue, value.cpu().numpy().tolist())
        if isinstance(value, np.ndarray):
            return cast(JsonValue, value.tolist())
        if isinstance(value, dict):
            return {
                k: self._convert_to_serializable(v)
                for k, v in cast(dict[str, object], value).items()
            }
        if isinstance(value, list):
            return [self._convert_to_serializable(v) for v in cast(list[object], value)]
        return cast(JsonValue, value)
