"""Tests for label processors."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import tempfile

from priml.data.processors.labels import ImagenetSynsetToIndex


if TYPE_CHECKING:
    from priml.data.processors.custom_types import Sample


def test_imagenet_synset_to_index():
    """Test ImagenetSynsetToIndex converter."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("n01440764\n")
        f.write("n01443537\n")
        f.write("n01484850\n")
        labels_file = f.name

    try:
        config = ImagenetSynsetToIndex.Config()
        config.labels_file = labels_file
        converter = config.make()

        sample: Sample = {"label": "n01443537", "key": "test"}
        result = next(iter(converter(iter([sample]))))

        assert "label" in result
        assert result["label"] == 1
    finally:
        Path(labels_file).unlink()


def test_imagenet_synset_to_index_passthrough():
    """Test ImagenetSynsetToIndex passes through unknown/int labels."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("n01440764\n")
        labels_file = f.name

    try:
        config = ImagenetSynsetToIndex.Config()
        config.labels_file = labels_file
        converter = config.make()

        # Unknown synset.
        sample1: Sample = {"label": "n99999999", "key": "test1"}
        result1 = next(iter(converter(iter([sample1]))))
        assert "label" in result1
        assert result1["label"] == "n99999999"

        # Already int.
        sample2: Sample = {"label": 42, "key": "test2"}
        result2 = next(iter(converter(iter([sample2]))))
        assert "label" in result2
        assert result2["label"] == 42
    finally:
        Path(labels_file).unlink()


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
