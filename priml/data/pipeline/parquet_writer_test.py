"""Tests for parquet processing utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import logging

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


if TYPE_CHECKING:
    import pytest

from priml.data.pipeline.parquet_writer import (
    BufferProcessor,
    ParquetMergeWriter,
)
from priml.data.processors.fields import FieldRenameKeys


if TYPE_CHECKING:
    from pathlib import Path


def test_field_remapper_rename():
    """Test renaming fields."""
    remapper = FieldRenameKeys.Config(
        mappings={
            "original_width": "width",
            "original_height": "height",
        },
    ).make()

    samples: list[dict[str, object]] = [
        {"original_width": 640, "original_height": 480, "keep": "value"},
    ]

    results = list(remapper(iter(samples)))
    assert len(results) == 1
    assert results[0]["width"] == 640
    assert results[0]["height"] == 480
    assert results[0]["keep"] == "value"


def test_field_remapper_missing_field():
    """Test that missing source fields are handled gracefully."""
    remapper = FieldRenameKeys.Config(
        mappings={"original_width": "width"},
    ).make()

    samples: list[dict[str, object]] = [{"other_field": 100}]

    results = list(remapper(iter(samples)))
    assert len(results) == 1
    assert "width" not in results[0]
    assert results[0]["other_field"] == 100


def test_field_copier_copies_fields():
    """Test copying fields to new names (same as rename but keeps both)."""
    remapper = FieldRenameKeys.Config(
        mappings={"source": "target"},
    ).make()

    samples: list[dict[str, object]] = [{"source": "value"}]
    results = list(remapper(iter(samples)))

    assert "target" in results[0]


def test_field_remapper_overwrite_existing():
    """Test that rename overwrites existing fields."""
    remapper = FieldRenameKeys.Config(
        mappings={"source": "target"},
    ).make()

    samples: list[dict[str, object]] = [
        {"source": "new_value", "target": "old_value"},
    ]
    results = list(remapper(iter(samples)))

    assert results[0]["target"] == "new_value"


def test_parquet_merge_writer_basic(tmp_path: Path):
    """Test basic parquet merge functionality."""
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(pa.table({"key": ["a", "b", "c"], "value": [1, 2, 3]}), parquet_path)

    buffered_results: dict[str, dict[str, object]] = {
        "a": {"new_field": 10, "other_field": "x"},
        "c": {"new_field": 30, "other_field": "z"},
    }

    writer = ParquetMergeWriter.Config().make()
    writer.write(parquet_path, buffered_results, overwrite=True)

    result = pq.read_table(parquet_path)
    assert result.column("key").to_pylist() == ["a", "b", "c"]
    assert result.column("value").to_pylist() == [1, 2, 3]
    assert result.column("new_field").to_pylist() == [10, None, 30]
    assert result.column("other_field").to_pylist() == ["x", None, "z"]


def test_parquet_merge_writer_logs_each_write_phase(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(pa.table({"key": ["a", "b"]}), parquet_path)

    ParquetMergeWriter.Config().make().write(
        parquet_path,
        {"a": {"new_field": 10}},
    )

    assert [record.getMessage() for record in caplog.records] == [
        "ParquetMergeWriter: Writing 1 samples to test.parquet",
        "ParquetMergeWriter: Merging fields: ['new_field']",
        "ParquetMergeWriter: Wrote test.parquet.new",
        "ParquetMergeWriter: Renamed test.parquet.new -> test.parquet",
    ]


def test_parquet_merge_writer_no_overwrite(tmp_path: Path):
    """Test merge without overwriting original file."""
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(pa.table({"key": ["a"], "value": [1]}), parquet_path)

    original_mtime = parquet_path.stat().st_mtime

    buffered_results: dict[str, dict[str, object]] = {"a": {"new_field": 10}}

    writer = ParquetMergeWriter.Config().make()
    writer.write(parquet_path, buffered_results, overwrite=False)

    assert parquet_path.stat().st_mtime == original_mtime

    new_path = parquet_path.with_suffix(".parquet.new")
    assert new_path.exists()

    result = pq.read_table(new_path)
    assert result.column("new_field").to_pylist() == [10]


def test_parquet_merge_writer_empty_buffered_results(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that empty buffered results is a no-op."""
    caplog.set_level(logging.WARNING)
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(pa.table({"key": ["a"], "value": [1]}), parquet_path)

    original_mtime = parquet_path.stat().st_mtime

    writer = ParquetMergeWriter.Config().make()
    writer.write(parquet_path, {}, overwrite=True)

    assert parquet_path.stat().st_mtime == original_mtime
    assert [record.getMessage() for record in caplog.records] == [
        "ParquetMergeWriter: No buffered results to write for test.parquet",
    ]


def test_parquet_merge_writer_dict_with_int_keys(tmp_path: Path):
    """Test that dicts with int keys are converted to str keys for parquet."""
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(pa.table({"key": ["a"]}), parquet_path)

    buffered_results: dict[str, dict[str, object]] = {
        "a": {"embeddings": {0: [1.0, 2.0], 1: [3.0, 4.0]}},
    }

    writer = ParquetMergeWriter.Config().make()
    writer.write(parquet_path, buffered_results, overwrite=True)

    result = pq.read_table(parquet_path)
    embeddings = cast(
        dict[str, list[float]],
        result.column("embeddings")[0].as_py(),  # pyright: ignore[reportAny] -- PyArrow's scalar stub returns Any.
    )

    assert "0" in embeddings
    assert "1" in embeddings

    np.testing.assert_array_equal(embeddings["0"], [1.0, 2.0])
    np.testing.assert_array_equal(embeddings["1"], [3.0, 4.0])


def test_parquet_merge_writer_skips_internal_fields(tmp_path: Path):
    """Test that internal/transient fields are not written to parquet."""
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(pa.table({"key": ["a"]}), parquet_path)

    buffered_results: dict[str, dict[str, object]] = {
        "a": {
            "good_field": "value",
            "_tar_handle": "should be skipped",
            "image": "should be skipped",
            "media_tensor": "should be skipped",
            "caption_tokens": "should be skipped",
        },
    }

    writer = ParquetMergeWriter.Config().make()
    writer.write(parquet_path, buffered_results, overwrite=True)

    result = pq.read_table(parquet_path)
    col_names = result.column_names

    assert "good_field" in col_names
    assert "_tar_handle" not in col_names
    assert "image" not in col_names
    assert "media_tensor" not in col_names
    assert col_names == ["key", "good_field"]


def test_parquet_merge_writer_replaces_existing_column_and_preserves_unmatched_rows(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(pa.table({"key": ["a", "b"], "score": [1, 2]}), parquet_path)

    ParquetMergeWriter.Config().make().write(parquet_path, {"a": {"score": 10}})

    result = pq.read_table(parquet_path)
    assert result.column_names == ["key", "score"]
    assert result.column("score").to_pylist() == [10, 2]


def test_parquet_merge_writer_custom_key_field(tmp_path: Path):
    """Test using a custom key field instead of 'key'."""
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(pa.table({"id": ["x", "y"], "value": [1, 2]}), parquet_path)

    buffered_results: dict[str, dict[str, object]] = {
        "x": {"new_field": 10},
        "y": {"new_field": 20},
    }

    writer = ParquetMergeWriter.Config(key_field="id").make()
    writer.write(parquet_path, buffered_results, overwrite=True)

    result = pq.read_table(parquet_path)
    assert result.column("id").to_pylist() == ["x", "y"]
    assert result.column("new_field").to_pylist() == [10, 20]


def test_cleanup_stale_temp_files_removes_orphans(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Orphaned .parquet.new temp files from a crash are removed."""
    caplog.set_level(logging.INFO)
    orphans = [tmp_path / f"{index}.parquet.new" for index in range(2)]
    for orphan in orphans:
        orphan.write_bytes(b"partial")
    keep = tmp_path / "00000000.parquet"
    keep.write_bytes(b"real")

    removed = ParquetMergeWriter.cleanup_stale_temp_files(tmp_path)

    assert removed == 2
    assert all(not orphan.exists() for orphan in orphans)
    assert keep.exists()
    assert {record.getMessage() for record in caplog.records} == {
        f"ParquetMergeWriter: Removed stale temp file {orphan.name}"
        for orphan in orphans
    }


def test_write_cleans_orphan_before_writing(tmp_path: Path):
    """A stale temp file from a prior crash is purged on the next write."""
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(pa.table({"key": ["a"], "value": [1]}), parquet_path)
    orphan = tmp_path / "stale.parquet.new"
    orphan.write_bytes(b"partial")

    writer = ParquetMergeWriter.Config().make()
    writer.write(parquet_path, {"a": {"new_field": 10}}, overwrite=True)

    assert not orphan.exists()


def test_buffer_processor_starts_with_an_empty_buffer() -> None:
    processor = BufferProcessor.Config().make()

    assert processor.get_buffered_results() == {}


def test_buffer_processor_passes_everything_through_and_keeps_keyed_samples() -> None:
    processor = BufferProcessor.Config().make()
    samples: list[dict[str, object]] = [
        {"key": "a", "v": 1},
        {"v": 2},
        {"key": "b", "v": 3},
    ]
    assert list(processor(iter(samples))) == samples
    assert processor.get_buffered_results() == {"a": samples[0], "b": samples[2]}


def test_buffer_processor_flushes_when_the_group_key_changes() -> None:
    config = BufferProcessor.Config()
    config.group_by_key = "shard"
    processor = config.make()
    samples: list[dict[str, object]] = [
        {"key": "a", "shard": 0},
        {"key": "b", "shard": 0},
        {"key": "c", "shard": 1},
    ]
    seen = [set(processor.get_buffered_results()) for _ in processor(iter(samples))]
    assert seen == [{"a"}, {"a", "b"}, {"c"}]


def test_parquet_merge_writer_serializes_tensors_arrays_and_nested_values(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "test.parquet"
    pq.write_table(pa.table({"key": ["a", "b"]}), parquet_path)
    buffered: dict[str, dict[str, object]] = {
        "a": {
            "tensor": torch.tensor([1.0, 2.0]),
            "array": np.array([3, 4]),
            "nested": {"x": torch.tensor([5])},
            "listed": [np.array([6]), [7]],
            "empty": {},
        },
    }
    ParquetMergeWriter.Config().make().write(parquet_path, buffered, overwrite=True)
    result = pq.read_table(parquet_path)
    assert result.column("tensor").to_pylist() == [[1.0, 2.0], None]
    assert result.column("array").to_pylist() == [[3, 4], None]
    assert result.column("nested").to_pylist() == [{"x": [5]}, None]
    assert result.column("listed").to_pylist() == [[[6], [7]], None]
    assert result.column("empty").to_pylist() == [None, None]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
