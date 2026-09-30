"""Canonicalized ARC2 voting, checked against source-minted records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import json

from torch import Tensor

import numpy as np
import torch

from priml.baselines.arcagi1.augmentation import dihedral_transform
from priml.baselines.arcagi2.metric import PassK
from priml.baselines.arcagi2.record_test import assert_matches, reduce


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


class Voting(Protocol):
    """The metric surface the recorder drives."""

    def update(self, logits: Tensor, **batch: object) -> None:
        """Record one batch of ballots."""
        ...

    def compute(self) -> Mapping[str, object]:
        """Score the ballots."""
        ...


def write_manifest(root: Path) -> None:
    """Two tasks: one with two test outputs under eight views, one never voted."""
    identifiers = ["<blank>", *(f"multi|||t{i}|||0213456789" for i in range(8))]
    (root / "identifiers.json").write_text(json.dumps(identifiers))
    (root / "test_puzzles.json").write_text(
        json.dumps(
            {
                "multi": {
                    "test": [
                        {"input": [[1, 2]], "output": [[2], [1]]},
                        {"input": [[3]], "output": [[4]]},
                    ],
                },
                "missing": {"test": [{"input": [[5]], "output": [[6]]}]},
            },
        ),
    )


def record_votes(
    root: Path,
    build: Callable[[Path], Voting],
    dihedral: Callable[[np.ndarray, int], np.ndarray],
) -> dict[str, Tensor]:
    """Vote eight transformed views and record the inputs and every score.

    Args:
      root: Directory holding the manifest.
      build: Constructs the metric over ``root``.
      dihedral: Applies dihedral transform ``tid`` to a grid.

    Returns:
      record: Input tensors and the computed scores.

    """
    permutation = np.array([0, 2, 1, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
    # Eight views are required to cover every dihedral transform.
    media = torch.zeros(8, 2, 2, dtype=torch.int32)
    predictions = torch.zeros_like(media)
    for index in range(8):
        for storage, grid in ((media, [[1, 2]]), (predictions, [[2], [1]])):
            transformed = dihedral(permutation[np.array(grid)], index).copy()
            transformed_tensor = torch.from_numpy(transformed)
            rows, cols = transformed_tensor.shape
            storage[index, :rows, :cols] = transformed_tensor + 2
    # PassK reserves one leading halt-logit column in packed scores.
    packed = torch.cat([torch.zeros(8, 1), predictions.flatten(1).float()], dim=1)
    batch = {"media": media.flatten(1), "puzzle_identifiers": torch.arange(1, 9)}
    metric = build(root)
    metric.update(packed, **batch)
    return reduce({"packed": packed, **batch, "scores": dict(metric.compute())})


def test_reference_metric(tmp_path: Path) -> None:
    """Eight transformed votes solve one of three outputs across two tasks."""
    write_manifest(tmp_path)
    record = record_votes(
        tmp_path,
        lambda root: PassK.Config(working_dir=root).make(),
        lambda grid, tid: dihedral_transform(grid, tid=tid),
    )
    assert_matches("metric", "votes", record)
    assert record["scores/pass@1"].item() == 0.25
    assert record["scores/strict@1"].item() == 0.0
    assert torch.equal(record["scores/per_output@1"], torch.tensor(1 / 3))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
