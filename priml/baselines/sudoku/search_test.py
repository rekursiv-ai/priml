from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import Tensor

import pytest
import torch

from priml.baselines.sudoku.puzzle_spec import SudokuSpec, sudoku_group_indices
from priml.baselines.sudoku.search import (
    HpsSearch,
    _empty_result,
    _violated_group_counts,
    accepted_grids,
    run_learned_pin_search_fast,
    run_pin_search_fast,
    run_search,
    select_pin_candidates,
)
from priml.baselines.sudoku.testing import (
    sudoku_solution,
    tiny_recursion,
    tiny_recursion_config,
)
from priml.baselines.sudoku.views import IDENTITY_DIGITS


if TYPE_CHECKING:
    from collections.abc import Callable

    from priml.baselines.sudoku import latent_recursion


def test_pin_candidates_and_validity_acceptance_on_constant_grids() -> None:
    media = torch.full((2, 81), 1, dtype=torch.long)
    media[:, 0] = 2
    logits = torch.zeros(2, 81, 11)
    logits[..., 2] = 4
    cells, _ = select_pin_candidates(logits, media, n_cells=2, n_digits=3)
    assert cells.shape == (2, 2)
    groups = sudoku_group_indices(SudokuSpec())
    preds = torch.full((2, 81), 2, dtype=torch.long)
    assert not bool(accepted_grids(preds, media, groups).any())
    assert torch.equal(_violated_group_counts(preds, groups), torch.full((2,), 27.0))


def test_pin_search_engines_respect_budget_rows_and_chunks() -> None:
    boards = torch.full((2, 81), 1, dtype=torch.long)
    boards[:, 0] = 2
    root_logits = torch.zeros(2, 81, 11)
    root_logits[..., 2] = 1
    active = torch.tensor([True, False])
    rollout_batches: list[tuple[int, tuple[int, ...]]] = []
    rollout_boards: list[Tensor] = []

    def rollout(candidate: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        rollout_batches.append((len(candidate), tuple(int(row) for row in rows)))
        rollout_boards.append(candidate.clone())
        out = torch.zeros(candidate.shape[0], 81, 11)
        out[..., 2] = 1
        return out, torch.full((candidate.shape[0],), 3.0)

    learned = run_learned_pin_search_fast(
        rollout,
        media=boards,
        base_logits=root_logits,
        active=active,
        acceptance_threshold=4.0,
        depth=2,
        candidates=2,
        cell_attempts=2,
        budget=8,
        max_rows=3,
    )
    assert torch.equal(learned.accepted, torch.tensor([False, False]))
    assert torch.equal(learned.grids, boards)
    assert torch.equal(learned.scores, torch.full((2,), float("-inf")))
    assert torch.equal(learned.nodes, torch.tensor([8, 0]))
    assert torch.equal(learned.depth, torch.tensor([-1, -1]))
    assert [len(rows) for rows in learned.visited_predictions] == [2, 4, 2]
    assert [len(rows) for rows in learned.visited_puzzles] == [2, 4, 2]
    expected_pins = boards[0].repeat(2, 1)
    expected_pins[:, 1] = torch.tensor([2, 3])
    assert torch.equal(rollout_boards[0], expected_pins)

    def accepts(predictions: Tensor, media: Tensor) -> Tensor:
        assert media.shape == (predictions.shape[0], 81)
        return torch.zeros(predictions.shape[0], dtype=torch.bool)

    found, grids, nodes, depth = run_pin_search_fast(
        rollout,
        media=boards,
        base_logits=root_logits,
        active=active,
        groups=sudoku_group_indices(SudokuSpec()),
        depth=3,
        candidates=2,
        cell_attempts=2,
        budget=12,
        max_rows=3,
        accept_fn=accepts,
    )
    assert torch.equal(found, torch.tensor([False, False]))
    assert torch.equal(grids, boards)
    assert torch.equal(nodes, torch.tensor([12, 0]))
    assert torch.equal(depth, torch.tensor([-1, -1]))
    default_found, default_grids, default_nodes, default_depth = run_pin_search_fast(
        rollout,
        media=boards,
        base_logits=root_logits,
        active=active,
        groups=sudoku_group_indices(SudokuSpec()),
        depth=3,
        candidates=2,
        cell_attempts=2,
        budget=12,
        max_rows=3,
    )
    assert torch.equal(default_found, torch.tensor([False, False]))
    assert torch.equal(default_grids, boards)
    assert torch.equal(default_nodes, torch.tensor([12, 0]))
    assert torch.equal(default_depth, torch.tensor([-1, -1]))
    assert rollout_batches
    assert all(size <= 3 for size, _ in rollout_batches)
    assert all(set(rows) <= {0} for _, rows in rollout_batches)


def test_learned_pin_search_accepts_the_first_row_over_threshold() -> None:
    boards = torch.full((3, 81), 1, dtype=torch.long)
    root_logits = torch.zeros(3, 81, 11)
    root_logits[..., 4] = 1
    active = torch.tensor([True, True, False])

    def rollout(candidate: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        logits = torch.zeros(candidate.shape[0], 81, 11)
        logits[..., 5] = 1
        # Puzzle 0 accepts on its second candidate; puzzle 1 never does.
        second = torch.arange(candidate.shape[0]) % 2 == 1
        return logits, torch.where((rows == 0) & second, 6.0, 1.0)

    result = run_learned_pin_search_fast(
        rollout,
        media=boards,
        base_logits=root_logits,
        active=active,
        acceptance_threshold=5.0,
        depth=2,
        candidates=2,
        cell_attempts=2,
        budget=6,
        max_rows=8,
    )
    assert torch.equal(result.accepted, torch.tensor([True, False, False]))
    assert torch.equal(result.depth, torch.tensor([1, -1, -1]))
    assert torch.equal(result.scores[:1], torch.tensor([6.0]))
    assert torch.equal(result.grids[0], torch.full((81,), 5))
    assert torch.equal(result.grids[1:], boards[1:])
    # Puzzle 0 stops after its accepting level; puzzle 1 spends the budget.
    assert torch.equal(result.nodes, torch.tensor([2, 6, 0]))


def test_learned_pin_search_rolls_nothing_out_with_no_active_puzzle() -> None:
    boards = torch.full((2, 81), 1, dtype=torch.long)
    calls: list[int] = []

    def rollout(candidate: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        del rows
        calls.append(candidate.shape[0])
        return torch.zeros(candidate.shape[0], 81, 11), torch.zeros(candidate.shape[0])

    result = run_learned_pin_search_fast(
        rollout,
        media=boards,
        base_logits=torch.zeros(2, 81, 11),
        active=torch.zeros(2, dtype=torch.bool),
        acceptance_threshold=0.0,
        depth=2,
        candidates=2,
        cell_attempts=2,
        budget=6,
        max_rows=8,
    )
    assert calls == []
    assert torch.equal(result.nodes, torch.zeros(2, dtype=torch.int64))
    assert result.visited_predictions == []


def test_search_config_rejects_invalid_policies() -> None:
    with pytest.raises(ValueError, match="max_act_steps"):
        HpsSearch.Config(max_act_steps=0).make()
    with pytest.raises(ValueError, match="finite"):
        HpsSearch.Config(acceptance_threshold=float("inf")).make()
    with pytest.raises(ValueError, match="root_acceptance"):
        HpsSearch.Config(root_acceptance_threshold=float("inf")).make()
    with pytest.raises(ValueError, match="within"):
        HpsSearch.Config(max_act_steps=2, acceptance_checkpoints=(3,)).make()
    with pytest.raises(ValueError, match="needs"):
        HpsSearch.Config(
            acceptance_checkpoints=(),
            require_prediction_stability=True,
        ).make()
    with pytest.raises(ValueError, match="search_budget"):
        HpsSearch.Config(search_budget=2, search_candidates=3).make()
    with pytest.raises(ValueError, match="strictly increasing"):
        HpsSearch.Config(acceptance_checkpoints=(2, 2)).make()
    with pytest.raises(ValueError, match="early_exit"):
        HpsSearch.Config(early_exit_at_q8=True, acceptance_checkpoints=(2,)).make()


def test_eval_search_run_branches_and_errors() -> None:
    model = tiny_recursion()
    config = HpsSearch.Config(
        max_act_steps=2,
        acceptance_checkpoints=(1, 2),
        search_depth=1,
        search_candidates=2,
        search_cell_attempts=2,
        search_budget=4,
        search_max_rows=4,
        acceptance_threshold=0.0,
    )
    media = torch.full((2, 81), 1, dtype=torch.long)
    batch = {"media": media, "valid_count": 2}
    learned = config.make().run(model, batch)
    assert learned.accepted.shape == (2,)
    empty = config.make().run(model, {"media": media, "valid_count": 0})
    assert torch.equal(empty.scored, torch.zeros(2, dtype=torch.bool))
    assert torch.equal(empty.accepted, torch.zeros(2, dtype=torch.bool))
    assert torch.equal(empty.root_predictions, media)
    assert torch.equal(empty.final_predictions, media)
    assert torch.equal(empty.scores, torch.zeros(2))
    assert torch.equal(empty.root_scores, torch.zeros(2))
    assert torch.equal(empty.nodes, torch.zeros(2, dtype=torch.int64))
    assert torch.equal(empty.depth, torch.full((2,), -1, dtype=torch.int64))
    assert empty.visited_predictions == []
    assert empty.visited_puzzles == []
    meta_empty = _empty_result(media.to("meta"))
    assert all(
        tensor.device.type == "meta"
        for tensor in (
            meta_empty.accepted,
            meta_empty.root_predictions,
            meta_empty.final_predictions,
            meta_empty.scores,
            meta_empty.root_scores,
            meta_empty.nodes,
            meta_empty.depth,
            meta_empty.scored,
        )
    )
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        typed_empty = _empty_result(media)
        assert typed_empty.scores.dtype == torch.float32
        assert typed_empty.root_scores.dtype == torch.float32
    finally:
        torch.set_default_dtype(previous_dtype)
    predicate = config.make().run(
        model,
        batch,
        accept_fn=lambda preds, _media: torch.zeros(preds.shape[0], dtype=torch.bool),
    )
    assert predicate.nodes.shape == (2,)


def test_a_puzzle_embedding_model_reads_the_batch_identifiers() -> None:
    model_config = tiny_recursion_config()
    model_config.num_puzzle_identifiers = 3
    model_config.puzzle_emb_len = 2
    model = model_config.make().eval()
    config = HpsSearch.Config(
        max_act_steps=2,
        acceptance_checkpoints=(1, 2),
        search_depth=1,
        search_candidates=2,
        search_cell_attempts=2,
        search_budget=4,
        search_max_rows=4,
    )
    media = torch.full((2, 81), 1, dtype=torch.long)
    with pytest.raises(KeyError, match="puzzle_identifiers"):
        run_search(model, {"media": media, "valid_count": 2}, config)
    batch = {
        "media": media,
        "valid_count": 2,
        "puzzle_identifiers": torch.tensor([1, 2], dtype=torch.int32),
    }
    result = run_search(model, batch, config)
    assert torch.equal(result.scored, torch.ones(2, dtype=torch.bool))


def test_search_without_checkpoints_rolls_the_family_forward() -> None:
    config = HpsSearch.Config(
        max_act_steps=2,
        acceptance_checkpoints=(),
        require_prediction_stability=False,
        search_depth=1,
        search_candidates=2,
        search_cell_attempts=2,
        search_budget=4,
        search_max_rows=4,
        acceptance_threshold=1e9,
    )
    media = torch.full((2, 81), 1, dtype=torch.long)
    media[:, 0] = 2
    result = config.make().run(tiny_recursion(), {"media": media, "valid_count": 2})
    assert result.accepted.shape == (2,)
    assert not bool(result.accepted.any())


def test_grid_acceptance_and_group_counts_pin_validity_rules() -> None:
    solution = torch.tensor(
        [((row * 3 + row // 3 + col) % 9) + 2 for row in range(9) for col in range(9)],
    )
    valid = solution.repeat(4, 1)
    duplicate = valid[1].clone()
    duplicate[0] = duplicate[1]
    negative = valid[2].clone()
    negative[0] = -1
    too_large = valid[3].clone()
    too_large[0] = 11
    predictions = torch.stack((valid[0], duplicate, negative, too_large))
    media = torch.zeros_like(predictions)
    media[:, 0] = valid[:, 0]
    groups = sudoku_group_indices(SudokuSpec())
    assert accepted_grids(predictions, media, groups).tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert _violated_group_counts(predictions, groups).tolist() == [
        0.0,
        3.0,
        3.0,
        3.0,
    ]


def test_pin_search_acceptance_records_exact_winning_path() -> None:
    solution = torch.tensor(
        [((row * 3 + row // 3 + col) % 9) + 2 for row in range(9) for col in range(9)],
    )
    media = torch.full((2, 81), 1, dtype=torch.int64)
    base_logits = torch.zeros(2, 81, 11)
    seen_boards: list[Tensor] = []
    seen_rows: list[Tensor] = []

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        seen_boards.append(boards.clone())
        seen_rows.append(rows.clone())
        logits = torch.zeros(len(boards), 81, 11)
        logits.scatter_(2, solution.expand(len(boards), -1).unsqueeze(-1), 1)
        return logits, torch.full((len(boards),), -99.0)

    found, grids, nodes, depth = run_pin_search_fast(
        rollout,
        media=media,
        base_logits=base_logits,
        active=torch.tensor([True, False]),
        groups=sudoku_group_indices(SudokuSpec()),
        depth=2,
        candidates=2,
        cell_attempts=2,
        budget=8,
        max_rows=3,
        accept_fn=lambda predictions, _: torch.ones(len(predictions), dtype=torch.bool),
    )
    assert found.tolist() == [True, False]
    assert torch.equal(grids[0], solution)
    assert torch.equal(grids[1], media[1])
    assert nodes.tolist() == [2, 0]
    assert depth.tolist() == [1, -1]
    assert [batch.shape for batch in seen_boards] == [(2, 81)]
    assert seen_rows[0].tolist() == [0, 0]
    assert torch.equal(seen_boards[0][:, 0], torch.tensor([2, 3]))
    assert torch.equal(seen_boards[0][:, 1:], torch.ones(2, 80, dtype=torch.int64))


def test_grid_scoring_and_acceptance_pin_token_boundaries() -> None:
    solution = torch.tensor(
        [((row * 3 + row // 3 + col) % 9) + 2 for row in range(9) for col in range(9)],
    )
    predictions = solution.repeat(5, 1)
    predictions[1, 0] = 0
    predictions[2, 0] = 1
    predictions[3] = 0
    predictions[4] = 1
    assert _violated_group_counts(
        predictions,
        sudoku_group_indices(SudokuSpec()),
    ).tolist() == [0.0, 3.0, 3.0, 27.0, 27.0]

    media = torch.zeros(2, 81, dtype=torch.int64)
    media[0, 0] = media[1, 0] = 2
    media[0, 8] = media[1, 8] = 10
    media[1, 1] = 1
    media[1, 2] = 0
    assert accepted_grids(
        solution.expand(2, -1),
        media,
        sudoku_group_indices(SudokuSpec()),
    ).tolist() == [True, True]
    media[1, 8] = 9
    assert accepted_grids(
        solution.expand(2, -1),
        media,
        sudoku_group_indices(SudokuSpec()),
    ).tolist() == [True, False]


def test_run_pin_search_fast_obeys_budget_and_skips_inactive_rows() -> None:
    media = torch.ones(2, 4, dtype=torch.long)
    logits = torch.zeros(2, 4, 11)
    logits[..., 2] = 1
    calls: list[tuple[Tensor, Tensor]] = []

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        calls.append((boards.clone(), rows.clone()))
        out = torch.zeros(len(boards), 4, 11)
        out[..., 2] = 1
        return out, torch.zeros(len(boards))

    found, grids, nodes, depth = run_pin_search_fast(
        rollout,
        media=media,
        base_logits=logits,
        active=torch.tensor([True, False]),
        groups=torch.empty(0, 9, dtype=torch.long),
        depth=3,
        candidates=2,
        cell_attempts=1,
        budget=6,
        max_rows=3,
        accept_fn=lambda preds, _: torch.zeros(len(preds), dtype=torch.bool),
    )
    assert not found.any()
    assert torch.equal(grids, media)
    assert nodes.tolist() == [6, 0]
    assert depth.tolist() == [-1, -1]
    assert [len(rows) for _, rows in calls] == [2, 3, 1]
    assert [rows.tolist() for _, rows in calls] == [[0, 0], [0, 0, 0], [0]]

    calls.clear()
    _, _, depth_limited_nodes, _ = run_pin_search_fast(
        rollout,
        media=media,
        base_logits=logits,
        active=torch.tensor([True, False]),
        groups=torch.empty(0, 9, dtype=torch.long),
        depth=1,
        candidates=2,
        cell_attempts=1,
        budget=100,
        max_rows=100,
        accept_fn=lambda preds, _: torch.zeros(len(preds), dtype=torch.bool),
    )
    assert depth_limited_nodes.tolist() == [2, 0]
    assert [len(rows) for _, rows in calls] == [2]


def test_accepted_grids_rejects_each_invalid_grid_and_preserves_valid_solution() -> (
    None
):
    solution = sudoku_solution()
    duplicate = solution.clone()
    duplicate[0] = duplicate[1]
    blank = solution.clone()
    blank[0] = 1
    media = torch.zeros(4, 81, dtype=torch.long)
    media[:, 0] = solution[0]
    media[3, 1] = solution[1]
    media[3, 2] = 1
    mismatched_given = solution.clone()
    mismatched_given[0] = 3
    predictions = torch.stack((solution, duplicate, blank, mismatched_given))

    accepted = accepted_grids(
        predictions,
        media,
        sudoku_group_indices(SudokuSpec()),
    )

    assert accepted.tolist() == [True, False, False, False]


def test_run_pin_search_fast_uses_default_acceptance_and_preserves_outputs() -> None:
    solution = sudoku_solution()
    media = torch.ones(2, 81, dtype=torch.long)
    base_logits = torch.zeros(2, 81, 11)
    base_logits[..., 2] = 1
    calls: list[tuple[Tensor, Tensor]] = []

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        calls.append((boards.clone(), rows.clone()))
        logits = torch.zeros(len(boards), 81, 11)
        logits.scatter_(2, solution.expand(len(boards), -1).unsqueeze(-1), 1)
        return logits, torch.zeros(len(boards))

    found, grids, nodes, depth = run_pin_search_fast(
        rollout,
        media=media,
        base_logits=base_logits,
        active=torch.tensor([True, False]),
        groups=sudoku_group_indices(SudokuSpec()),
        depth=2,
        candidates=2,
        cell_attempts=2,
        budget=6,
        max_rows=3,
    )

    assert found.tolist() == [True, False]
    assert torch.equal(grids[0], solution)
    assert torch.equal(grids[1], media[1])
    assert nodes.tolist() == [2, 0]
    assert depth.tolist() == [1, -1]
    assert [len(rows) for _, rows in calls] == [2]
    assert calls[0][1].tolist() == [0, 0]


def test_run_pin_search_fast_expands_breadth_first_with_exact_budget_and_chunks() -> (
    None
):
    media = torch.ones(2, 5, dtype=torch.long)
    base_logits = torch.zeros(2, 5, 11)
    base_logits[..., 2] = 1
    calls: list[tuple[Tensor, Tensor]] = []

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        calls.append((boards.clone(), rows.clone()))
        logits = torch.zeros(len(boards), 5, 11)
        logits[..., 2] = 1
        return logits, torch.zeros(len(boards))

    found, grids, nodes, depth = run_pin_search_fast(
        rollout,
        media=media,
        base_logits=base_logits,
        active=torch.tensor([True, False]),
        groups=torch.empty(0, 9, dtype=torch.long),
        depth=3,
        candidates=2,
        cell_attempts=1,
        budget=6,
        max_rows=3,
        accept_fn=lambda preds, _: torch.zeros(len(preds), dtype=torch.bool),
    )

    assert not found.any()
    assert torch.equal(grids, media)
    assert nodes.tolist() == [6, 0]
    assert depth.tolist() == [-1, -1]
    assert [len(rows) for _, rows in calls] == [2, 3, 1]
    assert [rows.tolist() for _, rows in calls] == [[0, 0], [0, 0, 0], [0]]


def test_run_pin_search_fast_uses_root_and_child_selector_widths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = select_pin_candidates
    widths: list[tuple[int, int]] = []

    def record_widths(
        logits: Tensor,
        boards: Tensor,
        *,
        n_cells: int,
        n_digits: int,
    ) -> tuple[Tensor, Tensor]:
        widths.append((n_cells, n_digits))
        return original(logits, boards, n_cells=n_cells, n_digits=n_digits)

    monkeypatch.setattr(
        "priml.baselines.sudoku.search.select_pin_candidates",
        record_widths,
    )
    # Pin-search inputs use the production batch x token-grid contract.
    media = torch.ones((1, 5), dtype=torch.int64)
    base_logits = torch.zeros((1, 5, 11))
    base_logits[..., 2] = 1

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        del rows
        return (
            base_logits.expand(len(boards), -1, -1),
            torch.zeros(len(boards)),
        )

    run_pin_search_fast(
        rollout,
        media=media,
        base_logits=base_logits,
        active=torch.ones(1, dtype=torch.bool),
        groups=torch.empty((0, 9), dtype=torch.int64),
        depth=2,
        candidates=2,
        cell_attempts=2,
        budget=100,
        max_rows=100,
        accept_fn=lambda predictions, _: torch.zeros(
            len(predictions),
            dtype=torch.bool,
        ),
    )

    assert widths == [(2, 2), (1, 2), (1, 2)]


def test_run_pin_search_fast_preserves_device_dtypes_and_conversions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = object()
    zero = cast("Callable[..., Tensor]", torch.zeros)
    full = cast("Callable[..., Tensor]", torch.full)
    arange = cast("Callable[..., Tensor]", torch.arange)
    tensor_to = cast("Callable[..., Tensor]", Tensor.to)
    zero_calls: list[tuple[object, object]] = []
    full_calls: list[tuple[object, object]] = []
    arange_calls: list[object] = []
    to_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_zeros(*args: object, **kwargs: object) -> Tensor:
        zero_calls.append((kwargs.get("dtype", missing), kwargs.get("device", missing)))
        return zero(*args, **kwargs)

    def record_full(*args: object, **kwargs: object) -> Tensor:
        full_calls.append((kwargs.get("dtype", missing), kwargs.get("device", missing)))
        return full(*args, **kwargs)

    def record_arange(*args: object, **kwargs: object) -> Tensor:
        arange_calls.append(kwargs.get("device", missing))
        return arange(*args, **kwargs)

    def record_to(
        self: Tensor,
        *args: object,
        **kwargs: object,
    ) -> Tensor:
        to_calls.append((args, kwargs))
        return tensor_to(self, *args, **kwargs)

    media = torch.ones((2, 5), dtype=torch.int64)
    media[0, 0] = 2
    media[1, 0] = 3
    base_logits = torch.zeros((2, 5, 11))
    base_logits[..., 2] = 1
    monkeypatch.setattr(torch, "zeros", record_zeros)
    monkeypatch.setattr(torch, "full", record_full)
    monkeypatch.setattr(torch, "arange", record_arange)
    monkeypatch.setattr(Tensor, "to", record_to)
    acceptance_calls = 0

    def accept_one_row(predictions: Tensor, boards: Tensor) -> Tensor:
        nonlocal acceptance_calls
        acceptance_calls += 1
        if acceptance_calls == 1:
            return boards[:, 0] == 2
        return torch.ones(predictions.shape[0], dtype=torch.bool)

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        del rows
        return base_logits[:1].expand(len(boards), -1, -1), boards[:, 0]

    found, _, _, depth_used = run_pin_search_fast(
        rollout,
        media=media,
        base_logits=base_logits,
        active=torch.ones(2, dtype=torch.bool),
        groups=torch.empty((0, 9), dtype=torch.int64),
        depth=2,
        candidates=2,
        cell_attempts=1,
        budget=10,
        max_rows=100,
        accept_fn=accept_one_row,
    )

    assert found.tolist() == [True, True]
    assert depth_used.tolist() == [1, 2]
    assert zero_calls == [
        (torch.int64, media.device),
        (torch.bool, media.device),
    ]
    assert full_calls == [(torch.int64, media.device)]
    assert arange_calls == [media.device, media.device]
    assert to_calls == [((torch.int64,), {})] * 4


def test_run_pin_search_fast_gathers_every_active_puzzle() -> None:
    media = torch.ones((2, 5), dtype=torch.int64)
    base_logits = torch.zeros((2, 5, 11))
    base_logits[..., 2] = 1

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        del rows
        logits = base_logits[:1].expand(len(boards), -1, -1)
        return logits, torch.zeros(len(boards))

    _, _, nodes, _ = run_pin_search_fast(
        rollout,
        media=media,
        base_logits=base_logits,
        active=torch.ones(2, dtype=torch.bool),
        groups=torch.empty((0, 9), dtype=torch.int64),
        depth=1,
        candidates=2,
        cell_attempts=1,
        budget=2,
        max_rows=100,
        accept_fn=lambda predictions, _: torch.zeros(
            len(predictions),
            dtype=torch.bool,
        ),
    )

    assert nodes.tolist() == [2, 2]


def test_empty_result_preserves_batch_device_and_field_contract() -> None:
    media = torch.tensor([[2, 1, 3], [1, 4, 1]])

    class ModelStub:
        puzzle_emb: None = None

    result = (
        HpsSearch.Config()
        .make()
        .run(
            cast("latent_recursion.LatentRecursion", ModelStub()),
            {"media": media, "valid_count": 0},
        )
    )

    assert result.depth.tolist() == [-1, -1]
    assert result.accepted.shape == result.scored.shape == (2,)
    assert result.accepted.dtype == result.scored.dtype == torch.bool
    assert result.root_predictions.shape == result.final_predictions.shape == (2, 3)
    assert (
        result.root_predictions.dtype == result.final_predictions.dtype == torch.int64
    )
    assert result.scores.shape == result.root_scores.shape == (2,)
    assert result.scores.dtype == result.root_scores.dtype == torch.float32
    assert result.nodes.shape == result.depth.shape == (2,)
    assert result.nodes.dtype == result.depth.dtype == torch.int64
    assert result.visited_predictions == []
    assert result.visited_puzzles == []

    meta_result = (
        HpsSearch.Config()
        .make()
        .run(
            cast("latent_recursion.LatentRecursion", ModelStub()),
            {"media": media.to("meta"), "valid_count": 0},
        )
    )
    assert all(
        tensor.device.type == "meta"
        for tensor in (
            meta_result.accepted,
            meta_result.root_predictions,
            meta_result.final_predictions,
            meta_result.scores,
            meta_result.root_scores,
            meta_result.nodes,
            meta_result.depth,
            meta_result.scored,
        )
    )


def test_group_violations_token_normalization_matches_clamp() -> None:
    predictions = torch.tensor([[-3, -1, 0, 1, 2, 10, 11, 12, 13]])
    # Sudoku group indexing is a single nine-cell row by contract.
    groups = torch.arange(9).reshape(1, 9)
    old_tokens = predictions.clamp(min=0, max=10)[:, groups]
    expected_counts = torch.nn.functional.one_hot(
        old_tokens,
        num_classes=len(IDENTITY_DIGITS) + 2,
    ).sum(dim=2)
    expected_violations = (
        (
            (expected_counts[..., 2:] > 1).any(dim=2)
            | (expected_counts[..., :2] > 0).any(dim=2)
        )
        .sum(dim=1)
        .float()
    )

    assert torch.equal(
        _violated_group_counts(predictions, groups),
        expected_violations,
    )


def test_group_violations_and_acceptance_enforce_sudoku_token_rules() -> None:
    solution = sudoku_solution()
    duplicate = solution.clone()
    duplicate[0] = duplicate[1]
    blank = solution.clone()
    blank[0] = 1
    negative = solution.clone()
    negative[0] = -1
    too_large = solution.clone()
    too_large[0] = 11
    predictions = torch.stack((solution, duplicate, blank, negative, too_large))
    groups = sudoku_group_indices(SudokuSpec())

    assert _violated_group_counts(predictions, groups).tolist() == [
        0.0,
        3.0,
        3.0,
        3.0,
        3.0,
    ]
    media = torch.zeros_like(predictions)
    media[:, 0] = solution[0]
    assert accepted_grids(predictions, media, groups).tolist() == [
        True,
        False,
        False,
        False,
        False,
    ]
    mismatched_given = media[:1].clone()
    mismatched_given[0, 0] = 3
    assert accepted_grids(
        solution.unsqueeze(0),
        mismatched_given,
        groups,
    ).tolist() == [False]
    ten_token = torch.where(solution == 10)[0][0]
    invalid_digit = solution.clone()
    invalid_digit[ten_token] = 11
    assert accepted_grids(
        invalid_digit.unsqueeze(0),
        torch.zeros_like(solution).unsqueeze(0),
        groups,
    ).tolist() == [False]
    two_token = torch.where(solution == 2)[0][0]
    alternate_two = torch.where(
        solution == 2,
        3,
        torch.where(solution == 3, 2, solution),
    )
    given_two = torch.zeros_like(solution).unsqueeze(0)
    given_two[0, two_token] = 2
    assert accepted_grids(
        alternate_two.unsqueeze(0),
        given_two,
        groups,
    ).tolist() == [
        False,
    ]
    invalid_given = torch.zeros_like(solution).unsqueeze(0)
    invalid_given[0, two_token] = 11
    assert accepted_grids(solution.unsqueeze(0), invalid_given, groups).tolist() == [
        True,
    ]
    alternate = torch.where(
        solution == 10,
        9,
        torch.where(solution == 9, 10, solution),
    )
    ten_givens = torch.zeros_like(solution).unsqueeze(0)
    ten_givens[0, solution == 10] = 10
    assert accepted_grids(alternate.unsqueeze(0), ten_givens, groups).tolist() == [
        False,
    ]


def test_run_pin_search_fast_starts_no_attempt_past_its_budget() -> None:
    """Each root attempt costs ``candidates`` nodes; the second would exceed 2."""

    def rollout(boards: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        del rows
        return torch.zeros(len(boards), 4, 11), torch.zeros(len(boards))

    _, _, nodes, _ = run_pin_search_fast(
        rollout,
        media=torch.ones(2, 4, dtype=torch.long),
        base_logits=torch.zeros(2, 4, 11),
        active=torch.tensor([True, True]),
        groups=torch.empty(0, 9, dtype=torch.long),
        depth=1,
        candidates=2,
        cell_attempts=2,
        budget=2,
        max_rows=8,
        accept_fn=lambda preds, _: torch.zeros(len(preds), dtype=torch.bool),
    )
    assert nodes.tolist() == [2, 2]


if __name__ == "__main__":
    from priml.lib.testing.runner import test_main

    test_main(__file__)
