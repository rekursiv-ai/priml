from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import torch

from priml.baselines.sudoku.search import SearchResult
from priml.baselines.sudoku.search_dump import (
    learned_hps_output_width,
    pack_search_rows,
    solution_visited_flags,
    summarize_search,
    write_member_dump,
)


if TYPE_CHECKING:
    from pathlib import Path

    from numpy.lib.npyio import NpzFile


def _result(preds: torch.Tensor, **visited: list[torch.Tensor]) -> SearchResult:
    return SearchResult(
        accepted=torch.tensor([True, False]),
        root_predictions=preds,
        final_predictions=preds,
        scores=torch.tensor([2.0, -1.0]),
        root_scores=torch.tensor([2.0, -1.0]),
        nodes=torch.tensor([0, 3]),
        depth=torch.tensor([0, -1]),
        scored=torch.tensor([True, True]),
        visited_predictions=visited.get("predictions", []),
        visited_puzzles=visited.get("puzzles", []),
    )


def test_packed_rows_round_trip_through_a_member_dump(tmp_path: Path) -> None:
    media = torch.full((2, 81), 1, dtype=torch.long)
    media[:, 0] = 2
    preds = torch.full((2, 81), 2, dtype=torch.long)
    rows = pack_search_rows(_result(preds))
    assert rows.shape == (2, learned_hps_output_width(81))
    assert rows[:, :7].tolist() == [
        [2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 3.0, -1.0, 4.0, 0.0, 0.0],
    ]
    path = tmp_path / "member.npz"
    write_member_dump(path, rows, media=media, label=preds)
    with cast("NpzFile", np.load(path)) as dump:
        assert dump["rows"].dtype == np.float32
        assert np.array_equal(dump["rows"], rows.numpy())
        assert dump["media"].dtype == dump["label"].dtype == np.uint8
        assert np.array_equal(dump["label"], preds.numpy())
    assert summarize_search(rows, preds)["accepted"] == 1.0
    with pytest.raises(ValueError, match="packed width"):
        write_member_dump(tmp_path / "bad.npz", rows[:, :-1], media=media)


def test_dump_and_summary_guards_reject_misaligned_rows(tmp_path: Path) -> None:
    width = learned_hps_output_width(81)
    rows = torch.zeros(2, width)
    media = torch.full((2, 81), 2, dtype=torch.long)
    with pytest.raises(ValueError, match="packed width"):
        summarize_search(rows[:, :-1], media)
    with pytest.raises(ValueError, match="disagree on N"):
        write_member_dump(tmp_path / "n.npz", rows[:1], media=media)
    with pytest.raises(ValueError, match="disagree on shape"):
        write_member_dump(tmp_path / "s.npz", rows, media=media, label=media[:, :3])
    unlabeled = tmp_path / "unlabeled.npz"
    write_member_dump(unlabeled, rows, media=media)
    with cast("NpzFile", np.load(unlabeled)) as dump:
        assert sorted(dump.files) == ["media", "rows"]


def test_solution_visited_flags_find_the_solution_at_any_node() -> None:
    """The root misses both puzzles; a level-1 candidate solves puzzle 0 only."""
    label = torch.stack([torch.full((81,), 3), torch.full((81,), 4)])
    label[:, 0] = -100
    wrong = torch.full((2, 81), 2)
    candidates = torch.stack([torch.full((81,), 3), torch.full((81,), 5)])
    result = _result(wrong, predictions=[candidates], puzzles=[torch.tensor([0, 1])])
    assert solution_visited_flags(result, label).tolist() == [True, False]
    rows = pack_search_rows(result, solution_visited=torch.tensor([True, False]))
    assert rows[:, 5].tolist() == [1.0, 0.0]
    assert summarize_search(rows, label)["solution_visited"] == 1.0


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
