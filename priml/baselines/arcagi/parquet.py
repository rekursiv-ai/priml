"""Typed, streaming reader for CHARM's ARC augmented parquet v2 files."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, SupportsIndex, TypedDict, cast

import operator

from numpy.typing import NDArray
from treekle import from_plain

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Iterator

    from pyarrow.parquet import ParquetFile
else:
    from wrapt import lazy_import

    ParquetFile = lazy_import("pyarrow.parquet", "ParquetFile")


class ParquetPair(TypedDict):
    """One ARC demonstration or test example."""

    input: list[list[int]]
    output: list[list[int]] | None


class ParquetRow(TypedDict):
    """One puzzle row from the augmented v2 format."""

    puzzle_id: str
    train: list[ParquetPair]
    test: list[ParquetPair]
    offline_augs: list[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class GridStore:
    """Compact flat storage for variable-size ARC grids."""

    values: NDArray[np.uint8]
    offsets: NDArray[np.int64]
    shapes: NDArray[np.int64]

    def get(self, index: int) -> NDArray[np.uint8] | None:
        """Return one grid view, or ``None`` for an absent output.

        Args:
          index: Flat grid index in this store.

        Returns:
          grid: A view into compact bytes, or ``None`` for a reserved shape.

        """
        shape = [
            _numpy_int(cast(SupportsIndex, self.shapes[index, axis]))
            for axis in range(2)
        ]
        height, width = shape
        if height == 0 or width == 0:
            return None
        start = _numpy_int(cast(SupportsIndex, self.offsets[index]))
        return self.values[start : start + height * width].reshape(height, width)


@dataclass(frozen=True, slots=True, kw_only=True)
class ParquetSourceData:
    """Base grids and compact row offsets from one physical parquet path."""

    path: Path
    puzzle_names: tuple[str, ...]
    row_train_starts: NDArray[np.int64]
    row_train_counts: NDArray[np.int64]
    row_test_starts: NDArray[np.int64]
    row_test_counts: NDArray[np.int64]
    row_test_train_starts: NDArray[np.int64]
    row_test_train_counts: NDArray[np.int64]
    row_augmentation_starts: NDArray[np.int64]
    row_augmentation_counts: NDArray[np.int64]
    offline_aug_ids: NDArray[np.uint32]
    offline_aug_strings: tuple[str, ...]
    dih_idxs: NDArray[np.uint8]
    colorperm_slots: NDArray[np.uint8]
    train_inputs: GridStore
    train_outputs: GridStore
    test_inputs: GridStore
    test_outputs: GridStore
    test_train_indices: NDArray[np.int64]
    train_original_indices: NDArray[np.uint32]
    test_original_indices: NDArray[np.uint32]

    @property
    def num_rows(self) -> int:
        """Number of parquet puzzle rows."""
        return len(self.puzzle_names)

    @property
    def num_augmentations(self) -> int:
        """Number of stored augmentation descriptors, excluding repeats."""
        return len(self.dih_idxs)

    def augmentation_name(self, index: int) -> tuple[int, tuple[int, ...]]:
        """Return a compact semantic key for one offline view.

        Args:
          index: Flat descriptor index.

        Returns:
          view: Dihedral index and nine non-background color slots.

        """
        return (
            _numpy_int(cast(SupportsIndex, self.dih_idxs[index])),
            tuple(
                _numpy_int(cast(SupportsIndex, self.colorperm_slots[index, slot]))
                for slot in range(9)
            ),
        )

    def offline_aug_name(self, index: int) -> str:
        """Return the literal raw identity string for one descriptor."""
        aug_id = _numpy_int(cast(SupportsIndex, self.offline_aug_ids[index]))
        return self.offline_aug_strings[aug_id]

    def get_pair(
        self,
        pair_index: int,
        pair_kind: int,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8] | None, str]:
        """Resolve one flattened demonstration or test pair to canonical grids.

        Args:
          pair_index: Flat pair index in the selected pair-kind store.
          pair_kind: Zero for demonstrations and one for test examples.

        Returns:
          pair: Canonical input, optional output, and canonical puzzle name.

        """
        if pair_kind == 0:
            row_index = _numpy_int(
                cast(
                    SupportsIndex,
                    np.searchsorted(
                        self.row_train_starts,
                        pair_index,
                        side="right",
                    )
                    - 1,
                ),
            )
            inputs, outputs = self.train_inputs, self.train_outputs
        elif pair_kind == 1:
            row_index = _numpy_int(
                cast(
                    SupportsIndex,
                    np.searchsorted(
                        self.row_test_starts,
                        pair_index,
                        side="right",
                    )
                    - 1,
                ),
            )
            inputs, outputs = self.test_inputs, self.test_outputs
        else:
            raise ValueError(f"Unknown pair kind {pair_kind}.")
        input_grid = inputs.get(pair_index)
        if input_grid is None:
            raise ValueError("An ARC input grid cannot be missing.")
        return input_grid, outputs.get(pair_index), self.puzzle_names[row_index]


def read_parquet_source(path: Path, *, max_grid_size: int = 30) -> ParquetSourceData:
    """Read base grids and row indexing without expanding augmented pairs.

    Args:
      path: Physical augmented v2 parquet file.
      max_grid_size: Maximum accepted grid height and width.

    Returns:
      source: Compact base grids, pair offsets, and offline-view descriptors.

    """
    puzzle_names: list[str] = []
    seen_names: set[str] = set()
    row_train_starts: list[int] = [0]
    row_train_counts: list[int] = []
    row_test_starts: list[int] = [0]
    row_test_counts: list[int] = []
    row_test_train_starts: list[int] = [0]
    row_test_train_counts: list[int] = []
    row_augmentation_starts: list[int] = [0]
    row_augmentation_counts: list[int] = []
    offline_aug_strings: list[str] = []
    offline_aug_string_ids: dict[str, int] = {}
    offline_aug_ids = array("I")
    if offline_aug_ids.itemsize != np.dtype(np.uint32).itemsize:
        raise RuntimeError("array('I') is not uint32-sized on this platform")
    dih_idxs = bytearray()
    colorperm_slots = bytearray()
    test_train_indices: list[int] = []
    train_original_indices: list[int] = []
    test_original_indices: list[int] = []
    train_inputs = _GridStoreBuilder()
    train_outputs = _GridStoreBuilder()
    test_inputs = _GridStoreBuilder()
    test_outputs = _GridStoreBuilder()

    for row in iter_parquet_rows(path):
        if row["puzzle_id"] in seen_names:
            raise ValueError(
                f"{path.name} contains duplicate v2 puzzle row for "
                f"{row['puzzle_id']!r}.",
            )
        seen_names.add(row["puzzle_id"])
        puzzle_names.append(row["puzzle_id"])
        row_augmentation_counts.append(len(row["offline_augs"]))
        for offline_aug in row["offline_augs"]:
            if offline_aug not in offline_aug_string_ids:
                offline_aug_string_ids[offline_aug] = len(offline_aug_strings)
                offline_aug_strings.append(offline_aug)
            offline_aug_ids.append(offline_aug_string_ids[offline_aug])
            dih_idx, slots = parse_offline_aug(offline_aug)
            dih_idxs.append(dih_idx)
            colorperm_slots.extend(slots)
        row_augmentation_starts.append(len(dih_idxs))

        for original_index, pair in enumerate(row["train"]):
            output = pair["output"]
            if not _grid_fits(pair["input"], max_grid_size) or output is None:
                continue
            if not _grid_fits(output, max_grid_size):
                continue
            train_inputs.append(pair["input"])
            train_outputs.append(output)
            train_original_indices.append(original_index)
        num_train_pairs = len(train_inputs.shapes) // 2
        row_train_counts.append(num_train_pairs - row_train_starts[-1])
        row_train_starts.append(num_train_pairs)

        for original_index, pair in enumerate(row["test"]):
            if not _grid_fits(pair["input"], max_grid_size):
                continue
            test_inputs.append(pair["input"])
            test_outputs.append(pair["output"])
            test_original_indices.append(original_index)
            if pair["output"] is None or _grid_fits(pair["output"], max_grid_size):
                test_train_indices.append(len(test_inputs.shapes) // 2 - 1)
        num_test_pairs = len(test_inputs.shapes) // 2
        row_test_counts.append(num_test_pairs - row_test_starts[-1])
        row_test_starts.append(num_test_pairs)
        row_test_train_counts.append(
            len(test_train_indices) - row_test_train_starts[-1],
        )
        row_test_train_starts.append(len(test_train_indices))

    return ParquetSourceData(
        path=path,
        puzzle_names=tuple(puzzle_names),
        row_train_starts=np.asarray(row_train_starts[:-1], dtype=np.int64),
        row_train_counts=np.asarray(row_train_counts, dtype=np.int64),
        row_test_starts=np.asarray(row_test_starts[:-1], dtype=np.int64),
        row_test_counts=np.asarray(row_test_counts, dtype=np.int64),
        row_test_train_starts=np.asarray(row_test_train_starts[:-1], dtype=np.int64),
        row_test_train_counts=np.asarray(row_test_train_counts, dtype=np.int64),
        row_augmentation_starts=np.asarray(
            row_augmentation_starts[:-1],
            dtype=np.int64,
        ),
        row_augmentation_counts=np.asarray(row_augmentation_counts, dtype=np.int64),
        offline_aug_ids=np.frombuffer(offline_aug_ids, dtype=np.uint32),
        offline_aug_strings=tuple(offline_aug_strings),
        dih_idxs=np.frombuffer(dih_idxs, dtype=np.uint8),
        colorperm_slots=np.frombuffer(colorperm_slots, dtype=np.uint8).reshape(-1, 9),
        train_inputs=train_inputs.finish(),
        train_outputs=train_outputs.finish(),
        test_inputs=test_inputs.finish(),
        test_outputs=test_outputs.finish(),
        test_train_indices=np.asarray(test_train_indices, dtype=np.int64),
        train_original_indices=np.asarray(train_original_indices, dtype=np.uint32),
        test_original_indices=np.asarray(test_original_indices, dtype=np.uint32),
    )


def iter_parquet_rows(
    path: Path,
    *,
    batch_size: int = 64,
) -> Iterator[ParquetRow]:
    """Yield validated v2 rows while bounding Arrow-to-Python conversion.

    Args:
      path: Physical augmented v2 parquet file.
      batch_size: Maximum Arrow rows converted into Python objects at once.

    Yields:
      row: One typed row preserving parquet order.

    """
    parquet = ParquetFile(path)
    required = {"format_version", "puzzle_id", "train", "test", "offline_augs"}
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(
            f"{path.name} is not an ARC augmented v2 parquet; missing columns: "
            f"{', '.join(missing)}.",
        )
    for batch in parquet.iter_batches(
        columns=sorted(required),
        batch_size=batch_size,
        use_threads=True,
    ):
        records = from_plain(batch.to_pylist(), list[dict[str, object]])
        for record in records:
            version = from_plain(record.get("format_version"), int)
            if version != 2:
                raise ValueError(
                    f"{path.name} row has format_version={version}; expected 2.",
                )
            puzzle_id = from_plain(record.get("puzzle_id"), str)
            train = _pairs(record.get("train"), "train")
            test = _pairs(record.get("test"), "test")
            offline_augs = from_plain(
                record.get("offline_augs"),
                list[str | None],
            )
            normalized_augs = [aug or "" for aug in offline_augs]
            if not normalized_augs or normalized_augs[0] != "":
                raise ValueError(
                    f"{path.name} puzzle {puzzle_id!r} must list the no-augmentation "
                    "entry first in offline_augs.",
                )
            for aug in normalized_augs:
                parse_offline_aug(aug)
            yield {
                "puzzle_id": puzzle_id,
                "train": train,
                "test": test,
                "offline_augs": normalized_augs,
            }


def parse_offline_aug(offline_aug: str) -> tuple[int, tuple[int, ...]]:
    """Decode one supported dihedral/color-permutation augmentation string.

    Args:
      offline_aug: Empty identity marker or a supported CHARM transform chain.

    Returns:
      view: Dihedral id and nine color slots, with the background omitted.

    """
    dih_idx = 0
    colorperm_slots = tuple(range(9))
    seen: set[str] = set()
    for part in offline_aug.split("||") if offline_aug else ():
        fields = part.split("|", maxsplit=1)
        if len(fields) != 2 or fields[0] not in {"dih", "colorperm"}:
            raise ValueError(f"Unsupported offline augmentation {offline_aug!r}.")
        name, parameter = fields
        if name in seen:
            raise ValueError(
                f"Duplicate {name!r} in offline augmentation {offline_aug!r}.",
            )
        seen.add(name)
        if name == "dih":
            if parameter not in tuple(str(index) for index in range(8)):
                raise ValueError(f"Invalid dihedral index in {offline_aug!r}.")
            dih_idx = int(parameter)
        else:
            if (
                len(parameter) != 10
                or parameter[0] != "0"
                or sorted(parameter[1:]) != list("123456789")
            ):
                raise ValueError(f"Invalid color permutation in {offline_aug!r}.")
            colorperm_slots = tuple(int(color) - 1 for color in parameter[1:])
    return dih_idx, colorperm_slots


def _pairs(value: object, name: str) -> list[ParquetPair]:
    pairs = from_plain(value, list[dict[str, object]], default=[])
    result: list[ParquetPair] = []
    for pair_value in pairs:
        grid_in = _grid(pair_value.get("input"), f"{name}.input")
        output = pair_value.get("output")
        typed_pair: ParquetPair = {
            "input": grid_in,
            "output": None if output is None else _grid(output, f"{name}.output"),
        }
        result.append(typed_pair)
    return result


def _grid(value: object, name: str) -> list[list[int]]:
    rows = from_plain(value, list[list[object]])
    result: list[list[int]] = []
    width: int | None = None
    for row in rows:
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{name} grid must be rectangular.")
        cells: list[int] = []
        for cell in row:
            if (
                isinstance(cell, bool)
                or not isinstance(cell, int)
                or cell < 0
                or cell > 9
            ):
                raise ValueError(f"{name} grid cells must be integer colors in 0..9.")
            cells.append(cell)
        result.append(cells)
    if not result or width == 0:
        raise ValueError(f"{name} grid must have positive height and width.")
    return result


class _GridStoreBuilder:
    """Append grids with lossless signed64 shapes and compact byte contents."""

    def __init__(self) -> None:
        self.values = bytearray()
        self.offsets: list[int] = [0]
        self.shapes: array[int] = array("q")

    def append(self, grid: list[list[int]] | None) -> None:
        """Append one grid, reserving a zero shape for a missing output.

        Args:
          grid: Validated grid or ``None`` for a missing output.

        """
        if grid is None:
            self.shapes.extend((0, 0))
            self.offsets.append(len(self.values))
            return
        height, width = len(grid), len(grid[0])
        self.values.extend(value for row in grid for value in row)
        self.shapes.extend((height, width))
        self.offsets.append(len(self.values))

    def finish(self) -> GridStore:
        """Freeze the compact builder as NumPy arrays.

        Returns:
          store: Flat grid bytes and compact shape/offset indexes.

        """
        return GridStore(
            values=np.frombuffer(self.values, dtype=np.uint8),
            offsets=np.asarray(self.offsets, dtype=np.int64),
            shapes=np.frombuffer(self.shapes, dtype=np.int64).reshape(-1, 2),
        )


def _grid_fits(grid: list[list[int]], maximum: int) -> bool:
    """Return whether both grid sides fit the configured square canvas."""
    return len(grid) <= maximum and len(grid[0]) <= maximum


def _numpy_int(value: SupportsIndex) -> int:
    return operator.index(value)
