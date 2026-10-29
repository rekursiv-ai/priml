"""Checks for the ragged module."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import tempfile

import numpy as np

from priml.baselines.casmi2026.scripts.ragged import save_stream


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def _load(path: Path) -> NDArray[np.generic]:
    return cast("NDArray[np.generic]", np.load(path, allow_pickle=False))


def test_vectors_and_matrices_round_trip_with_their_dtype(tmp_path: Path) -> None:
    values = np.arange(6, dtype=np.int32)
    with tempfile.TemporaryFile() as stream:
        values.tofile(stream)
        save_stream(stream, tmp_path / "vector.npy", dtype=np.int32)
        save_stream(stream, tmp_path / "matrix.npy", dtype=np.int32, width=2)
        save_stream(stream, tmp_path / "wide.npy", dtype=np.int32, width=3)
    vector = _load(tmp_path / "vector.npy")
    assert vector.dtype == np.int32
    np.testing.assert_array_equal(vector, values)
    np.testing.assert_array_equal(
        _load(tmp_path / "matrix.npy"),
        values.reshape(3, 2),
    )
    assert _load(tmp_path / "wide.npy").shape == (2, 3)


def test_an_empty_stream_saves_an_empty_array_of_the_declared_shape(
    tmp_path: Path,
) -> None:
    with tempfile.TemporaryFile() as stream:
        save_stream(stream, tmp_path / "vector.npy", dtype=np.float64)
        save_stream(stream, tmp_path / "matrix.npy", dtype=np.int8, width=2)
    vector = _load(tmp_path / "vector.npy")
    matrix = _load(tmp_path / "matrix.npy")
    assert (vector.shape, vector.dtype) == ((0,), np.float64)
    assert (matrix.shape, matrix.dtype) == ((0, 2), np.int8)


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
