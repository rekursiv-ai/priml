"""Label processing for classification tasks.

Processors for converting and transforming class labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict

from configgle import Fig


if TYPE_CHECKING:
    from collections.abc import Iterator


_CWD: Final = Path(__file__).resolve().parent


__all__ = [
    "ImagenetSynsetToIndex",
]


class ImagenetSynsetToIndex:
    """Convert ImageNet synset labels to integer class indices.

    Maps synset IDs (e.g., "n01440764") to integer indices (0-999) based on
    the standard ImageNet label ordering.

    Inspired by TensorFlow Datasets (TFDS) ImageNet implementation:
    - TFDS uses ClassLabel(names_file='labels.txt') for synset->index conversion
    - labels.txt contains 1000 synset IDs in alphabetic order
    - This processor replicates that mapping

    Reference: tensorflow_datasets/datasets/imagenet2012/imagenet_common.py:23-33
    """

    class Config(Fig["ImagenetSynsetToIndex"]):
        labels_file: Path | str = "labels_imagenet.txt"
        """Synsets in the canonical order; line number becomes the index.

        A relative path resolves beside this module, so the default names the
        shipped file without baking a checkout path into the config."""

    class Input(TypedDict, total=False):
        """Input required by ImagenetSynsetToIndex."""

        label: str | int

    Output = Input

    def __init__(self, config: Config):
        labels_path = _CWD / config.labels_file
        with labels_path.open() as f:
            synsets = [line.strip() for line in f]

        self.synset_to_idx = {synset: idx for idx, synset in enumerate(synsets)}

    def __call__(self, samples: Iterator[Input]) -> Iterator[Output]:
        """Convert synset labels to integer indices.

        Requires:
          - label: str | int - synset ID or class index

        Adds:
          - label: int - class index (0-999)

        """
        for sample in samples:
            label = sample.get("label")

            if isinstance(label, str) and label in self.synset_to_idx:
                yield {**sample, "label": self.synset_to_idx[label]}
            else:
                # Pass through if label is already int or unknown.
                yield sample
