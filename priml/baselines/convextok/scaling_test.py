"""Scaling must reproduce cuOpt's Stable3 preconditioning of an LP.

cuOpt equilibrates the constraint matrix with ten Ruiz passes and one Pock-Chambolle
pass, then divides the bounds and the objective by one plus their norms. PDLP runs
entirely in that scaled space, so every factor enters the solver's trajectory. These
tests pin the arithmetic on programs small enough to work by hand; the solver's
checkpoints against cuOpt (step size, iterates) check it at scale.
"""

from unittest.mock import Mock

import math

from _pytest.monkeypatch import MonkeyPatch
from torch import Tensor

import torch

from priml.baselines.convextok import scaling
from priml.baselines.convextok.program import LinearProgram
from priml.baselines.convextok.scaling import (
    _divide_by_root,
    bound_norm,
    scale_program,
)


def test_divide_by_root_divides_positive_unit_and_preserves_zero() -> None:
    scale = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)
    norm = torch.tensor([1.0, 4.0, 0.0], dtype=torch.float64)
    assert torch.equal(
        _divide_by_root(scale, norm),
        torch.tensor([2.0, 1.5, 4.0], dtype=torch.float64),
    )


def test_ruiz_and_pock_chambolle_use_row_and_column_scales_in_order() -> None:
    scaled = scale_program(
        _program([[2.0, 1.0], [0.0, 8.0]]),
        ruiz_iterations=2,
    )
    row_scale = torch.ones(2, dtype=torch.float64)
    column_scale = torch.ones(2, dtype=torch.float64)
    values = torch.tensor([2.0, 1.0, 8.0], dtype=torch.float64)
    rows = torch.tensor([0, 0, 1])
    columns = torch.tensor([0, 1, 1])
    for _ in range(2):
        magnitude = ((values * row_scale[rows]) * column_scale[columns]).abs()
        row_max = torch.stack([magnitude[:2].max(), magnitude[2]])
        column_max = torch.stack([magnitude[0], magnitude[1:].max()])
        row_scale = row_scale / row_max.sqrt()
        column_scale = column_scale / column_max.sqrt()
    magnitude = ((values * row_scale[rows]) * column_scale[columns]).abs()
    row_sum = torch.stack([magnitude[:2].sum(), magnitude[2]])
    column_sum = torch.stack([magnitude[0], magnitude[1:].sum()])
    expected_row = row_scale / row_sum.sqrt()
    expected_column = column_scale / column_sum.sqrt()
    assert torch.equal(scaled.row_scale, expected_row)
    assert torch.equal(scaled.column_scale, expected_column)
    expected_values = torch.tensor([2.0, 1.0, 8.0], dtype=torch.float64)
    expected_values = expected_values * expected_row[torch.tensor([0, 0, 1])]
    expected_values *= expected_column[torch.tensor([0, 1, 1])]
    assert torch.equal(scaled.program.values, expected_values)


def test_zero_column_scale_replaces_both_variable_bounds(
    monkeypatch: MonkeyPatch,
) -> None:
    def zero_scale(scale: Tensor, norm: Tensor) -> Tensor:
        del norm
        return torch.zeros_like(scale)

    monkeypatch.setattr(scaling, "_divide_by_root", zero_scale)
    program = _program(
        [[1.0, 0.0], [0.0, 1.0]],
        lower=[2.0, 3.0],
        upper=[4.0, 5.0],
    )
    scaled = scale_program(program, ruiz_iterations=0)
    assert scaled.column_scale.tolist() == [0.0, 0.0]
    assert scaled.program.lower.tolist() == [0.0, 0.0]
    assert scaled.program.upper.tolist() == [0.0, 0.0]


def test_default_ruiz_pass_count_is_ten() -> None:
    program = _program([[1.0, 2.0], [3.0, 4.0]])
    default = scale_program(program)
    ten_passes = scale_program(program, ruiz_iterations=10)
    assert torch.equal(default.row_scale, ten_passes.row_scale)
    assert torch.equal(default.column_scale, ten_passes.column_scale)
    assert torch.equal(default.program.values, ten_passes.program.values)


def test_zero_ruiz_passes_scale_every_bound_and_objective() -> None:
    program = _program(
        [[2.0, 0.0], [0.0, 8.0]],
        row_lower=[2.0, -8.0],
        row_upper=[4.0, 16.0],
        objective=[2.0, 8.0],
        lower=[4.0, 8.0],
        upper=[6.0, 16.0],
        objective_offset=7.0,
    )
    scaled = scale_program(program, ruiz_iterations=0)
    root = torch.tensor([2.0, 8.0], dtype=torch.float64).sqrt()
    expected_scale = 1 / root
    assert torch.equal(scaled.row_scale, expected_scale)
    assert torch.equal(scaled.column_scale, expected_scale)
    expected_values = torch.tensor([2.0, 8.0], dtype=torch.float64)
    expected_values = (expected_values * expected_scale) * expected_scale
    assert torch.equal(scaled.program.values, expected_values)
    row_lower = torch.tensor([2.0, -8.0], dtype=torch.float64) * expected_scale
    row_upper = torch.tensor([4.0, 16.0], dtype=torch.float64) * expected_scale
    bound_factor = 1 / (bound_norm(row_lower, row_upper).item() + 1)
    objective = torch.tensor([2.0, 8.0], dtype=torch.float64) * expected_scale
    objective_factor = 1 / ((objective * objective).sum().sqrt().item() + 1)
    assert scaled.bound_rescaling == bound_factor
    assert scaled.objective_rescaling == objective_factor
    assert torch.equal(scaled.program.row_lower, row_lower * bound_factor)
    assert torch.equal(scaled.program.row_upper, row_upper * bound_factor)
    assert torch.equal(scaled.program.objective, objective * objective_factor)
    assert torch.equal(
        scaled.program.lower,
        torch.tensor([4.0, 8.0], dtype=torch.float64) / expected_scale * bound_factor,
    )
    assert torch.equal(
        scaled.program.upper,
        torch.tensor([6.0, 16.0], dtype=torch.float64) / expected_scale * bound_factor,
    )
    assert scaled.program.objective_offset == 7.0


def test_diagonal_matrix_is_equilibrated_by_ruiz() -> None:
    # Ruiz divides row 0 and column 0 by sqrt(2), row 1 and column 1 by sqrt(8);
    # later passes and Pock-Chambolle see unit entries.
    scaled = scale_program(_program([[2.0, 0.0], [0.0, 8.0]]))
    expected = 1 / torch.tensor([2.0, 8.0], dtype=torch.float64).sqrt()
    torch.testing.assert_close(scaled.row_scale, expected, rtol=1e-15, atol=0)
    torch.testing.assert_close(scaled.column_scale, expected, rtol=1e-15, atol=0)
    torch.testing.assert_close(
        scaled.program.values,
        torch.ones(2, dtype=torch.float64),
        rtol=1e-15,
        atol=0,
    )


def test_empty_rows_and_columns_keep_unit_scale() -> None:
    scaled = scale_program(_program([[0.0, 0.0], [0.0, 4.0]]))
    assert scaled.row_scale[0].item() == 1.0
    assert scaled.column_scale[0].item() == 1.0


def test_pock_chambolle_divides_by_root_of_absolute_sums() -> None:
    # Every row and column of [[1, 1], [1, -1]] has max 1, so Ruiz leaves it; each row
    # and column sums to 2 in absolute value, so Pock-Chambolle divides by sqrt(2).
    scaled = scale_program(_program([[1.0, 1.0], [1.0, -1.0]]))
    assert torch.equal(
        scaled.row_scale,
        1 / torch.full((2,), 2.0, dtype=torch.float64).sqrt(),
    )
    assert torch.equal(
        scaled.column_scale,
        1 / torch.full((2,), 2.0, dtype=torch.float64).sqrt(),
    )


def test_scale_program_places_created_tensors_with_matrix(
    monkeypatch: MonkeyPatch,
) -> None:
    program = _program([[1.0, 2.0], [3.0, 4.0]])
    values = program.values
    arange = Mock(wraps=torch.arange)
    zeros = Mock(wraps=torch.zeros)
    ones = Mock(wraps=torch.ones)
    monkeypatch.setattr(torch, "arange", arange)
    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(torch, "ones", ones)
    scale_program(program, ruiz_iterations=0)
    assert arange.call_args.args == (program.num_rows,)
    assert arange.call_args.kwargs == {"device": values.device}
    assert zeros.call_args.args == (program.num_columns + 1,)
    assert zeros.call_args.kwargs == {"dtype": torch.int64, "device": values.device}
    assert [call.args for call in ones.call_args_list] == [
        (program.num_rows,),
        (program.num_columns,),
    ]
    assert all(
        call.kwargs == {"dtype": values.dtype, "device": values.device}
        for call in ones.call_args_list
    )


def test_transpose_keeps_rows_ordered_within_a_column() -> None:
    dense = [[float(row + 1), 0.0] for row in range(50)]
    scaled = scale_program(_program(dense))
    assert scaled.transpose.col_indices.tolist() == list(range(50))


def test_transpose_holds_the_same_entries_column_major() -> None:
    scaled = scale_program(_program([[1.0, 2.0], [3.0, 0.0]]))
    assert scaled.transpose.crow_indices.tolist() == [0, 2, 3]
    assert scaled.transpose.col_indices.tolist() == [0, 1, 0]
    assert scaled.transpose_order.tolist() == [0, 2, 1]
    torch.testing.assert_close(
        scaled.transpose.values,
        scaled.program.values[scaled.transpose_order],
        rtol=1e-15,
        atol=0,
    )


def test_bounds_and_objective_are_divided_by_one_plus_their_norms() -> None:
    program = _program(
        [[1.0, 0.0], [0.0, 1.0]],
        row_lower=[3.0, -math.inf],
        row_upper=[3.0, 4.0],
        objective=[0.0, 12.0],
        upper=[2.0, 5.0],
    )
    scaled = scale_program(program)
    # A unit diagonal is left unscaled. An equality row counts its bound once and an
    # infinite bound not at all: sqrt(3^2 + 4^2) = 5, and sqrt(12^2) = 12.
    assert scaled.bound_rescaling == 1 / 6
    assert scaled.objective_rescaling == 1 / 13
    assert scaled.program.row_lower.tolist() == [3.0 * (1 / 6), -math.inf]
    assert scaled.program.row_upper.tolist() == [3.0 * (1 / 6), 4.0 * (1 / 6)]
    assert scaled.program.upper.tolist() == [2.0 * (1 / 6), 5.0 * (1 / 6)]
    assert scaled.program.objective.tolist() == [0.0, 12.0 * (1 / 13)]


def test_bound_norm_counts_each_finite_side_of_each_row_kind() -> None:
    # Equality 2 counts once; range [-3, 5] counts both sides; [4, inf) and
    # (-inf, 6] count their finite side: sqrt(4 + 9 + 25 + 16 + 36) = sqrt(90).
    row_lower = torch.tensor([2.0, -3.0, 4.0, -math.inf], dtype=torch.float64)
    row_upper = torch.tensor([2.0, 5.0, math.inf, 6.0], dtype=torch.float64)
    expected = torch.tensor(90.0, dtype=torch.float64).sqrt()
    torch.testing.assert_close(
        bound_norm(row_lower, row_upper),
        expected,
        rtol=0,
        atol=0,
    )


def test_scaled_program_preserves_structure_and_empty_column_bounds() -> None:
    program = _program(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        row_lower=[-1.0, 2.0],
        row_upper=[3.0, 4.0],
        objective=[1.0, 2.0, 3.0],
        upper=[5.0, 7.0, 9.0],
    )
    scaled = scale_program(program)

    assert scaled.program.crow_indices is program.crow_indices
    assert scaled.program.col_indices is program.col_indices
    assert scaled.program.num_columns == 3
    assert scaled.program.objective_offset == program.objective_offset
    assert torch.equal(scaled.matrix.crow_indices, program.crow_indices)
    assert torch.equal(scaled.matrix.col_indices, program.col_indices)
    assert torch.equal(scaled.matrix.values, scaled.program.values)
    assert scaled.transpose.crow_indices.tolist() == [0, 1, 2, 2]
    assert scaled.transpose.col_indices.tolist() == [0, 1]
    assert torch.equal(scaled.transpose.values, scaled.program.values)
    assert torch.equal(scaled.transpose_order, torch.tensor([0, 1]))
    assert scaled.column_scale[2].item() == 1.0
    assert scaled.program.lower[2].item() == 0.0
    assert scaled.program.upper[2].item() == 9.0 * scaled.bound_rescaling


def _program(
    dense: list[list[float]],
    *,
    row_lower: list[float] | None = None,
    row_upper: list[float] | None = None,
    objective: list[float] | None = None,
    lower: list[float] | None = None,
    upper: list[float] | None = None,
    objective_offset: float = 0.0,
) -> LinearProgram:
    matrix = torch.tensor(dense, dtype=torch.float64)
    rows, columns = matrix.shape
    entry_rows, entry_columns = torch.nonzero(matrix, as_tuple=True)
    counts = torch.bincount(entry_rows, minlength=rows)
    return LinearProgram(
        crow_indices=torch.cat(
            [torch.zeros(1, dtype=torch.int64), counts.cumsum(0)],
        ).to(
            torch.int32,
        ),
        col_indices=entry_columns.to(torch.int32),
        values=matrix[entry_rows, entry_columns],
        num_columns=columns,
        row_lower=torch.tensor(row_lower or [0.0] * rows, dtype=torch.float64),
        row_upper=torch.tensor(row_upper or [0.0] * rows, dtype=torch.float64),
        objective=torch.tensor(objective or [0.0] * columns, dtype=torch.float64),
        lower=torch.tensor(lower or [0.0] * columns, dtype=torch.float64),
        upper=torch.tensor(upper or [1.0] * columns, dtype=torch.float64),
        objective_offset=objective_offset,
    )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
