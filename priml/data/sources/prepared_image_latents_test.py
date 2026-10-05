from __future__ import annotations

from typing import TYPE_CHECKING

import json

from PIL import Image

import numpy as np
import pytest

from priml.data.sources.prepared_image_latents import read_image, read_labels
from priml.lib.custom_json import ReadError


if TYPE_CHECKING:
    from pathlib import Path


def test_read_labels_normalizes_names(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps({"labels": [["a\\b", 3], ["café", 7]]}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert read_labels(path) == {"a/b": 3, "café": 7}


def test_read_image_supports_numpy_and_pil(tmp_path: Path) -> None:
    array = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    npy = tmp_path / "x.npy"
    np.save(npy, array)
    assert np.array_equal(read_image(npy), array)
    png = tmp_path / "x.png"
    Image.fromarray(np.zeros((2, 4, 3), dtype=np.uint8)).save(png)
    assert read_image(png).shape == (3, 2, 4)


def test_read_labels_rejects_a_non_integer_label(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"labels": [["x", "unknown"]]}), encoding="utf-8")
    with pytest.raises(ReadError):
        read_labels(path)


def test_read_image_flattens_numpy_leading_dimensions(tmp_path: Path) -> None:
    array = np.arange(120, dtype=np.uint8).reshape(2, 3, 4, 5)
    path = tmp_path / "stack.npy"
    np.save(path, array)
    result = read_image(path)
    assert result.shape == (6, 4, 5)
    assert np.array_equal(result, array.reshape(6, 4, 5))


def test_read_image_converts_grayscale_to_rgb(tmp_path: Path) -> None:
    gray = np.arange(8, dtype=np.uint8).reshape(2, 4)
    path = tmp_path / "gray.png"
    Image.fromarray(gray).save(path)
    result = read_image(path)
    assert result.shape == (3, 2, 4)
    assert np.array_equal(result, np.stack((gray, gray, gray)))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
