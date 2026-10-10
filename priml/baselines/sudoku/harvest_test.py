from __future__ import annotations

from typing import TYPE_CHECKING, Final

from torch import Tensor

import pytest
import torch

from priml.baselines.sudoku import puzzle_data
from priml.baselines.sudoku.harvest import (
    Harvest,
    fixed_hps_node_count,
    make_random_unstuck_starts,
    rows_per_view,
    select_harvest_views,
)
from priml.baselines.sudoku.testing import Harness, load_golden, replay
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_harvest_plan_helpers_count_select_and_corrupt() -> None:
    grid = torch.arange(2 * 81).reshape(2, 81) % 9 + 2
    assert fixed_hps_node_count(depth=2, candidates=3, cell_attempts=2) == 24
    assert (
        rows_per_view(
            search_depth=2,
            search_candidates=3,
            search_cell_attempts=2,
            random_corruption_strengths=(2, 4),
            random_starts_per_strength=2,
        )
        == 29
    )
    with pytest.raises(ValueError, match="positive"):
        fixed_hps_node_count(depth=0, candidates=2, cell_attempts=2)
    with pytest.raises(ValueError, match="nonempty"):
        rows_per_view(
            search_depth=2,
            search_candidates=2,
            search_cell_attempts=2,
            random_corruption_strengths=(),
            random_starts_per_strength=2,
        )
    selected = select_harvest_views(
        torch.tensor([0, 2, 4, 6]),
        group_count=2,
        views_per_group=2,
        seed=3,
    )
    assert selected.flat_view_id.shape == (4,)
    originals = grid[:2].clone()
    originals[:, :4] = 1
    starts = make_random_unstuck_starts(
        originals,
        grid[:2],
        base_group_ids=torch.tensor([0, 2]),
        view_ids=torch.tensor([1, 0]),
        seed=4,
        corruption_strengths=(2,),
        starts_per_strength=2,
    )
    assert starts.current_state.shape == (4, 81)
    assert torch.equal(
        starts.current_state[:, 4:],
        grid[:2, 4:].repeat_interleave(2, dim=0),
    )
    with pytest.raises(ValueError, match="no blank"):
        make_random_unstuck_starts(
            torch.full((2, 81), 2),
            grid[:2],
            base_group_ids=torch.tensor([0, 1]),
            view_ids=torch.tensor([0, 0]),
            seed=0,
            corruption_strengths=(2,),
            starts_per_strength=2,
        )


def test_harvest_needs_a_source_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        Harvest.Config().make()


def _record_harvest(h: Harness) -> dict[str, Tensor]:
    h.harvest(h.generator("gen_a", seed=0))
    return h.dumps("harvest")


def test_harvest_rejects_an_invalid_plan_before_loading() -> None:
    plans: tuple[tuple[dict[str, object], str], ...] = (
        ({"group_count": 0}, "group_count"),
        ({"search_max_rows": 0}, "search_max_rows"),
        ({"checkpoints": ()}, "strictly increasing"),
        ({"search_budget": 1}, "cannot exhaust"),
    )
    for overrides, message in plans:
        cfg = Harvest.Config()
        cfg.harvest_source_checkpoint = "/runs/source.pt"
        for name, value in overrides.items():
            setattr(cfg, name, value)
        with pytest.raises(ValueError, match=message):
            cfg.make()


_RECORDERS: Final[dict[str, Callable[[Harness], dict[str, Tensor]]]] = {
    "harvest": _record_harvest,
}


@pytest.mark.parametrize("case", _RECORDERS)
def test_golden_replays_bit_for_bit(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner reproduces the frozen evaluation with zero mismatches."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "torchinductor"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "triton"))
    puzzle_data.dihedral_indices.cache_clear()
    report = mismatches(load_golden(case), replay(_RECORDERS[case], tmp_path))
    assert not report, f"{len(report)} mismatches:\n" + "\n".join(report)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
