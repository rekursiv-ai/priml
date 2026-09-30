from __future__ import annotations

from typing import TYPE_CHECKING

import json

from PIL import Image

import numpy as np

from priml.data.sources.prepared_image_latents import read_image, read_labels


if TYPE_CHECKING:
    from pathlib import Path


def test_read_labels_normalizes_names(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"labels": [["a\\b", 3]]}), encoding="utf-8")
    assert read_labels(path) == {"a/b": 3}


def test_read_image_supports_numpy_and_pil(tmp_path: Path) -> None:
    array = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    npy = tmp_path / "x.npy"
    np.save(npy, array)
    assert np.array_equal(read_image(npy), array)
    png = tmp_path / "x.png"
    Image.fromarray(np.zeros((2, 4, 3), dtype=np.uint8)).save(png)
    assert read_image(png).shape == (3, 2, 4)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
