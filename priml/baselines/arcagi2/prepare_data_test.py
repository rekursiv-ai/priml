"""Stage only admitted ARC2 source-format trees, without overwriting data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

import numpy as np
import pytest

from priml.baselines.arcagi2.scripts.prepare_data import prepare


if TYPE_CHECKING:
    from pathlib import Path


def test_stage_preserves_bytes_and_rejects_overwrite(tmp_path: Path) -> None:
    """Keep the prepared source byte-exact and reject an existing destination."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "_build_params.json").write_text(
        json.dumps(
            {
                "subsets": ["training2", "evaluation2", "concept"],
                "test_set_name": "evaluation2",
            },
        ),
    )
    (source / "identifiers.json").write_text('["<blank>", "puzzle"]')
    (source / "test_puzzles.json").write_text(
        '{"puzzle": {"test": [{"input": [[0]], "output": [[0]]}]}}',
    )
    for split in ("train", "test"):
        directory = source / split
        directory.mkdir()
        (directory / "dataset.json").write_text(
            '{"ignore_label_id": 0, "blank_identifier_id": 0}',
        )
        for name, array in {
            "inputs": np.full((1, 9), 2, dtype=np.int32),
            "labels": np.full((1, 9), 2, dtype=np.int32),
            "puzzle_identifiers": np.array([1], dtype=np.int32),
            "puzzle_indices": np.array([0, 1], dtype=np.int64),
            "group_indices": np.array([0, 1], dtype=np.int64),
        }.items():
            np.save(directory / f"all__{name}.npy", array)

    target = tmp_path / "staged"
    prepare(source, destination=target)
    for original in source.rglob("*"):
        if original.is_file():
            assert (
                target / original.relative_to(source)
            ).read_bytes() == original.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(source, destination=target)
    payload = source / "train/all__inputs.npy"
    original = payload.rename(tmp_path / "inputs.npy")
    payload.symlink_to(original)
    with pytest.raises(ValueError, match="linked"):
        prepare(source, destination=tmp_path / "linked")
    assert not (tmp_path / "linked").exists()
    (source / "_build_params.json").write_text(
        '{"subsets": ["training", "evaluation", "concept"], "test_set_name": "evaluation"}',
    )
    with pytest.raises(ValueError, match="ARC2"):
        prepare(source, destination=tmp_path / "wrong")


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
