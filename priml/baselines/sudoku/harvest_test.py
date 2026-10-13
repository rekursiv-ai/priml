from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final, cast, override

from torch import Tensor
from treekle import json_loads

import numpy as np
import pytest
import torch

from priml.baselines.sudoku import puzzle_data
from priml.baselines.sudoku.harvest import (
    Harvest,
    _exhaustive_pin_trace,
    _PinTrace,
    checkpoint_rollout_from_state,
    fixed_hps_node_count,
    make_random_unstuck_starts,
    rows_per_view,
    select_harvest_views,
)
from priml.baselines.sudoku.rollout import LearnedCheckpointRollout
from priml.baselines.sudoku.testing import Harness, load_golden, replay
from priml.testing.golden import mismatches


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from priml.baselines.sudoku.latent_recursion import LatentRecursion


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


def test_harvest_plan_helpers_reject_malformed_plans() -> None:
    bounds = torch.tensor([0, 2, 3])
    select = partial(select_harvest_views, group_count=2, views_per_group=1, seed=0)
    with pytest.raises(ValueError, match="nonempty boundary vector"):
        select(torch.tensor([0]))
    with pytest.raises(ValueError, match="views_per_group must be positive"):
        select(bounds, views_per_group=0)
    with pytest.raises(ValueError, match=r"must lie in \[1, 2\]"):
        select(bounds, group_count=3)
    with pytest.raises(ValueError, match="base group 1 has 1 views"):
        select(bounds, views_per_group=2)


def test_unstuck_starts_reject_a_malformed_plan() -> None:
    grid = torch.ones(2, 3, dtype=torch.int64)
    ids = torch.tensor([0, 1])
    starts = partial(
        make_random_unstuck_starts,
        base_group_ids=ids,
        seed=0,
        corruption_strengths=(1,),
        starts_per_strength=1,
    )
    with pytest.raises(ValueError, match="aligned grids"):
        starts(grid, grid[:1], view_ids=ids)
    with pytest.raises(ValueError, match="align with puzzle rows"):
        starts(grid, grid, view_ids=ids[:1])
    with pytest.raises(ValueError, match="count positive"):
        starts(grid, grid, view_ids=ids, starts_per_strength=0)
    with pytest.raises(ValueError, match="positive and nonempty"):
        starts(grid, grid, view_ids=ids, corruption_strengths=(0,))


def test_checkpoint_rollout_rejects_misaligned_inputs() -> None:
    model = cast("LatentRecursion", object())
    grid = torch.ones(2, 4, dtype=torch.int64)
    rollout = partial(checkpoint_rollout_from_state, model, {}, 2)
    with pytest.raises(ValueError, match="aligned grids"):
        rollout(grid, grid[:1], torch.arange(2), checkpoints=(1,))
    with pytest.raises(ValueError, match="one source puzzle"):
        rollout(grid, grid, torch.arange(3), checkpoints=(1,))
    with pytest.raises(ValueError, match=r"must lie in 1\.\.2"):
        rollout(grid, grid, torch.arange(2), checkpoints=(3,))


_GRID: Final = 4
"""Cells of the toy board the pin-trace tests search."""

_VOCAB: Final = 11
"""Token vocabulary: pad, blank, and the nine digits."""


def _uniform_rollout(boards: Tensor, rows: Tensor) -> LearnedCheckpointRollout:
    """Return uniform logits, so pins go to the first blank cell, digits 2 then 3."""
    del rows
    n = boards.shape[0]
    return LearnedCheckpointRollout(
        logits=torch.zeros(n, _GRID, _VOCAB),
        q_scores=torch.zeros(n, 1),
        predictions=boards.unsqueeze(1),
    )


def _pin_trace(*, budget: int = 6, checkpoint_count: int = 1) -> _PinTrace:
    return _exhaustive_pin_trace(
        _uniform_rollout,
        media=torch.ones(3, _GRID, dtype=torch.int64),
        base_logits=torch.zeros(3, _GRID, _VOCAB),
        active=torch.ones(3, dtype=torch.bool),
        depth=2,
        candidates=2,
        cell_attempts=1,
        budget=budget,
        max_rows=5,
        checkpoint_count=checkpoint_count,
    )


def test_pin_trace_expands_every_node_within_budget() -> None:
    """Depth 2, two branches: 2 level-one then 4 level-two nodes per puzzle."""
    trace = _pin_trace()
    assert (
        trace.puzzle_rows.tolist() == [0, 0, 1, 1, 2, 2] + [0] * 4 + [1] * 4 + [2] * 4
    )
    assert trace.node_index.tolist() == [0, 1] * 3 + [2, 3, 4, 5] * 3
    assert trace.boards[6:10, :2].tolist() == [[2, 2], [2, 3], [3, 2], [3, 3]]
    assert trace.boards[:, 2:].eq(1).all()


def test_pin_trace_stops_where_the_budget_ends() -> None:
    """Budget 3 holds level one but not level two; budget 1 holds nothing."""
    assert _pin_trace(budget=3).node_index.tolist() == [0, 1] * 3
    empty = _pin_trace(budget=1)
    assert empty.puzzle_rows.shape == (0,)
    assert empty.boards.shape == (0, _GRID)
    assert empty.predictions.shape == (0, 1, _GRID)


def test_pin_trace_rejects_a_rollout_with_the_wrong_checkpoint_count() -> None:
    with pytest.raises(ValueError, match="checkpoint_count"):
        _pin_trace(checkpoint_count=2)


class _FixedShardHarvest(Harvest):
    """A harvest whose every shard is one fixed array set; no model is built."""

    def __init__(self, tmp_path: Path, shard: dict[str, np.ndarray]) -> None:
        self.config = Harvest.Config(group_count=3, views_per_group=1)
        self.run_dir = tmp_path / "harvest"
        self.checkpoint_path = tmp_path / "source.pt"
        self._rows_per_view = 2
        self._selected = select_harvest_views(
            torch.tensor([0, 1, 2, 3]),
            group_count=3,
            views_per_group=1,
            seed=0,
        )
        self._shard = shard

    @override
    def _shard_arrays(
        self,
        *,
        flat_ids: Tensor,
        group_ids: Tensor,
        view_ids: Tensor,
        first_group: int,
    ) -> dict[str, np.ndarray]:
        del flat_ids, group_ids, view_ids, first_group
        return self._shard


def _offline_harvest(tmp_path: Path, shard: dict[str, np.ndarray]) -> Harvest:
    """Build a 3-group harvest whose every shard is ``shard``."""
    return _FixedShardHarvest(tmp_path, shard)


def _shard(*, solved: bool) -> dict[str, np.ndarray]:
    return {
        "original": np.ones((2, _GRID), dtype=np.uint8),
        "exact_solved": np.array([solved, False]),
        "q_halt": np.array([[1.0], [-1.0]], dtype=np.float32),
        "source_kind": np.array([0, 2], dtype=np.uint8),
    }


def test_harvest_resumes_only_onto_a_matching_shard(tmp_path: Path) -> None:
    out = _offline_harvest(tmp_path, _shard(solved=True)).run()
    manifest = out / "manifest.json"
    first = json_loads(manifest.read_text(), dict[str, object])
    assert first["row_count"] == 2
    assert first["solved_count"] == 1
    assert first["high_q_wrong_count"] == 0
    assert first["source_counts"] == {"0": 1, "2": 1}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _offline_harvest(tmp_path, _shard(solved=True)).run()
    manifest.unlink()
    _offline_harvest(tmp_path, _shard(solved=True)).run()
    assert json_loads(manifest.read_text(), dict[str, object]) == first
    manifest.unlink()
    with pytest.raises(ValueError, match="does not match the deterministic"):
        _offline_harvest(tmp_path, _shard(solved=False)).run()


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
    from priml.lib.testing.runner import test_main

    test_main(__file__)
