"""Tests for CHARM's typed parquet boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyarrow import parquet

import numpy as np
import pyarrow as pa
import pytest

from priml.baselines.arcagi.parquet import (
    iter_parquet_rows,
    parse_offline_aug,
    read_parquet_source,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_iter_parquet_rows_streams_typed_v2_rows(tmp_path: Path) -> None:
    """Stream parquet batches into typed rows without changing source order."""
    path = tmp_path / "tiny.parquet"
    row: dict[str, object] = {
        "format_version": 2,
        "puzzle_id": "task-a",
        "train": [{"input": [[1, 2]], "output": [[2, 1]]}],
        "test": [{"input": [[3]], "output": [[4]]}],
        "offline_augs": ["", "dih|1||colorperm|0312456789"],
    }
    second_row: dict[str, object] = {**row, "puzzle_id": "task-b"}
    _write_parquet(path, rows=[row, second_row])

    got = list(iter_parquet_rows(path, batch_size=1))

    assert [item["puzzle_id"] for item in got] == ["task-a", "task-b"]
    assert got[0]["train"][0]["output"] == [[2, 1]]
    assert got[0]["offline_augs"] == ["", "dih|1||colorperm|0312456789"]


@pytest.mark.parametrize(
    ("aug", "dih", "slots"),
    [
        ("", 0, tuple(range(9))),
        ("dih|7", 7, tuple(range(9))),
        ("colorperm|0312456789", 0, (2, 0, 1, 3, 4, 5, 6, 7, 8)),
        (
            "dih|3||colorperm|0312456789",
            3,
            (2, 0, 1, 3, 4, 5, 6, 7, 8),
        ),
    ],
)
def test_parse_offline_aug(
    aug: str,
    dih: int,
    slots: tuple[int, ...],
) -> None:
    """Parse each supported offline view into its dihedral and color slots."""
    assert parse_offline_aug(aug) == (dih, slots)


@pytest.mark.parametrize(
    "aug",
    ["scale|2", "dih|8", "colorperm|0123456788", "dih|1||dih|2"],
)
def test_parse_offline_aug_rejects_unsupported_or_invalid_values(aug: str) -> None:
    """Reject unsupported transform names and malformed view encodings."""
    with pytest.raises(ValueError, match=r"Unsupported|Invalid|Duplicate"):
        parse_offline_aug(aug)


def test_iter_parquet_rows_rejects_wrong_version(tmp_path: Path) -> None:
    """Reject rows whose format version is not the released v2 schema."""
    path = tmp_path / "wrong-version.parquet"
    _write_parquet(
        path,
        rows=[
            {
                "format_version": 1,
                "puzzle_id": "task-a",
                "train": [],
                "test": [],
                "offline_augs": [""],
            },
        ],
    )

    with pytest.raises(ValueError, match="format_version=1"):
        list(iter_parquet_rows(path))


def test_reader_filters_large_grids_and_keeps_missing_eval_output(
    tmp_path: Path,
) -> None:
    """Filter oversized training grids and retain valid label-free eval rows."""
    oversized = [[1] * 31]
    path = tmp_path / "filtered.parquet"
    _write_parquet(
        path,
        rows=[
            {
                "format_version": 2,
                "puzzle_id": "task",
                "train": [
                    {"input": oversized, "output": [[2]]},
                    {"input": [[3]], "output": None},
                ],
                "test": [
                    {"input": oversized, "output": [[2]]},
                    {"input": [[4]], "output": oversized},
                    {"input": [[5]], "output": None},
                ],
                "offline_augs": [""],
            },
        ],
    )

    source = read_parquet_source(path)

    assert source.row_train_counts.tolist() == [0]
    assert source.row_test_counts.tolist() == [2]
    assert source.row_test_train_counts.tolist() == [1]
    assert source.test_outputs.get(1) is None
    with pytest.raises(ValueError, match="pair kind"):
        source.get_pair(0, 3)


def test_reader_preserves_original_pair_indices_and_raw_aug_identity(
    tmp_path: Path,
) -> None:
    """Retain original task-local pair indexes and raw augmentation strings."""
    path = tmp_path / "original-indices.parquet"
    _write_parquet(
        path,
        rows=[
            {
                "format_version": 2,
                "puzzle_id": "task",
                "train": [
                    {"input": [[1] * 31], "output": [[2]]},
                    {"input": [[3]], "output": [[4]]},
                ],
                "test": [
                    {"input": [[5] * 31], "output": [[6]]},
                    {"input": [[7]], "output": [[8]]},
                ],
                "offline_augs": ["", "dih|0"],
            },
        ],
    )

    source = read_parquet_source(path)

    assert source.row_train_counts.tolist() == [1]
    assert source.row_test_counts.tolist() == [1]
    assert source.train_original_indices.tolist() == [1]
    assert source.test_original_indices.tolist() == [1]
    assert source.offline_aug_strings == ("", "dih|0")
    assert source.offline_aug_ids.tolist() == [0, 1]


def test_reader_stores_shapes_without_canvas_dependent_narrowing(
    tmp_path: Path,
) -> None:
    """Store grid dimensions without canvas-dependent integer narrowing."""
    path = tmp_path / "wide-shape.parquet"
    _write_parquet(
        path,
        rows=[
            {
                "format_version": 2,
                "puzzle_id": "wide",
                "train": [{"input": [[1] * 256], "output": [[2]]}],
                "test": [],
                "offline_augs": [""],
            },
        ],
    )

    source = read_parquet_source(path, max_grid_size=256)

    assert source.train_inputs.shapes.dtype == np.int64
    grid = source.train_inputs.get(0)
    assert grid is not None
    assert grid.shape == (1, 256)


def test_reader_preserves_evaluation_output_larger_than_training_canvas(
    tmp_path: Path,
) -> None:
    """Preserve full canonical evaluation outputs beyond model canvas bounds."""
    path = tmp_path / "wide-eval-target.parquet"
    wide_output = [[2] * 256]
    _write_parquet(
        path,
        rows=[
            {
                "format_version": 2,
                "puzzle_id": "wide-eval",
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]], "output": wide_output}],
                "offline_augs": [""],
            },
        ],
    )

    source = read_parquet_source(path, max_grid_size=30)

    assert source.row_train_counts.tolist() == [1]
    assert source.row_test_counts.tolist() == [1]
    output = source.test_outputs.get(0)
    assert output is not None
    assert output.shape == (1, 256)
    assert output.tolist() == wide_output


@pytest.mark.parametrize(
    ("grid", "message"),
    [
        ([], "positive height"),
        ([[1], [1, 2]], "rectangular"),
        ([[10]], "integer colors"),
    ],
)
def test_grid_validation_rejects_invalid_shapes_and_colors(
    tmp_path: Path,
    grid: list[list[int]],
    message: str,
) -> None:
    """Reject empty, ragged, or out-of-range ARC grids at the parquet boundary."""
    path = tmp_path / "invalid-grid.parquet"
    _write_parquet(
        path,
        rows=[
            {
                "format_version": 2,
                "puzzle_id": "task",
                "train": [{"input": grid, "output": [[1]]}],
                "test": [],
                "offline_augs": [""],
            },
        ],
    )

    with pytest.raises(ValueError, match=message):
        list(iter_parquet_rows(path))


def test_parquet_schema_duplicate_rows_and_missing_identity(tmp_path: Path) -> None:
    """Reject missing schema fields, duplicate rows, and absent base views."""
    missing = tmp_path / "missing.parquet"
    _write_parquet(missing, rows=[{"format_version": 2, "puzzle_id": "task"}])
    with pytest.raises(ValueError, match="missing columns"):
        list(iter_parquet_rows(missing))

    duplicate = tmp_path / "duplicate.parquet"
    row: dict[str, object] = {
        "format_version": 2,
        "puzzle_id": "task",
        "train": [],
        "test": [],
        "offline_augs": [""],
    }
    _write_parquet(duplicate, rows=[row, row])
    with pytest.raises(ValueError, match="duplicate v2 puzzle row"):
        read_parquet_source(duplicate)

    invalid_aug = tmp_path / "missing-identity.parquet"
    row["offline_augs"] = ["dih|1"]
    _write_parquet(invalid_aug, rows=[row])
    with pytest.raises(ValueError, match="no-augmentation entry first"):
        list(iter_parquet_rows(invalid_aug))


def _write_parquet(path: Path, *, rows: list[dict[str, object]]) -> None:
    parquet.write_table(pa.Table.from_pylist(rows), path)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
