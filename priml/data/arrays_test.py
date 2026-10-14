from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from priml.data.arrays import load_array


if TYPE_CHECKING:
    from pathlib import Path


def _save(tmp_path: Path, array: np.ndarray) -> Path:
    path = tmp_path / "a.npy"
    np.save(path, array)
    return path


def test_load_array_returns_the_saved_values(tmp_path: Path) -> None:
    path = _save(tmp_path, np.array([[3, 1], [4, 1], [5, 9]], dtype=np.int32))
    loaded = load_array(path, np.int32)
    assert loaded.dtype == np.int32
    assert loaded.tolist() == [[3, 1], [4, 1], [5, 9]]


def test_load_array_accepts_any_dtype_under_an_abstract_one(tmp_path: Path) -> None:
    path = _save(tmp_path, np.array([2, 7], dtype=np.uint8))
    assert load_array(path, np.integer).dtype == np.uint8


@pytest.mark.parametrize(
    ("stored", "requested"),
    [(np.int32, np.int64), (np.float32, np.integer), (np.int64, np.uint8)],
)
def test_load_array_rejects_a_dtype_the_file_does_not_hold(
    tmp_path: Path,
    stored: type[np.generic],
    requested: type[np.generic],
) -> None:
    path = _save(tmp_path, np.zeros((2, 3), dtype=stored))
    with pytest.raises(TypeError, match=rf"{path}.*{np.dtype(stored)}"):
        load_array(path, requested)


def test_load_array_rejects_an_npz_archive(tmp_path: Path) -> None:
    path = tmp_path / "a.npz"
    np.savez(path, a=np.zeros(2, dtype=np.int32))
    with pytest.raises(TypeError, match="NpzFile"):
        load_array(path, np.int32)


def test_load_array_memory_maps_on_request(tmp_path: Path) -> None:
    path = _save(tmp_path, np.arange(6, dtype=np.int64).reshape(2, 3))
    loaded = load_array(path, np.int64, mmap=True)
    assert isinstance(loaded, np.memmap)
    assert loaded[1, 2] == 5


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
