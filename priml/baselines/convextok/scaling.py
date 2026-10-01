"""Precondition a linear program as cuOpt's PDLP (Stable3) does before iterating.

The constraint matrix gets a row scale ``r`` and a column scale ``d``: ten Ruiz passes,
each dividing every row and column by the square root of its largest absolute entry,
then one Pock-Chambolle pass dividing by the square root of the absolute sums. Entries
become ``(a * r) * d``; the transpose, which cuOpt scales separately, ``(a * d) * r``.
The objective is multiplied by ``d`` and the variable bounds divided by it, the row
bounds multiplied by ``r``. Finally the bounds are divided by one plus their Euclidean
norm and the objective by one plus its own.

Maxima are order independent and match cuOpt exactly; the sums use deterministic
segmented reductions whose order differs from cuOpt's, which moves the result by a
few units in the last place.
"""

from dataclasses import dataclass

from torch import Tensor

import torch

from priml.baselines.convextok.program import LinearProgram
from priml.baselines.convextok.spmv import CsrMatrix


@dataclass(frozen=True, slots=True, kw_only=True)
class ScaledProgram:
    """A program in PDLP's scaled space, and the factors that map solutions back.

    Attributes:
      program: The scaled program; its matrix is row-major.
      matrix: The scaled matrix, ready to multiply.
      transpose: The scaled matrix's transpose (a CSR of its columns), rounded as
        cuOpt rounds its separately scaled copy.
      transpose_order: Row-major position of each of the transpose's entries.
      row_scale: Cumulative row scale ``r``.
      column_scale: Cumulative column scale ``d``.
      bound_rescaling: Factor applied to every row and variable bound.
      objective_rescaling: Factor applied to the objective.

    """

    program: LinearProgram
    matrix: CsrMatrix
    transpose: CsrMatrix
    transpose_order: Tensor
    row_scale: Tensor
    column_scale: Tensor
    bound_rescaling: float
    objective_rescaling: float


def scale_program(
    program: LinearProgram,
    *,
    ruiz_iterations: int = 10,
) -> ScaledProgram:
    """Scale ``program`` with Ruiz, Pock-Chambolle (alpha 1) and bound/objective rescaling.

    Args:
      program: The program to scale.
      ruiz_iterations: Ruiz passes before Pock-Chambolle; Stable3 uses ten.

    Returns:
      scaled: The scaled program and its scaling factors.

    """
    crow = program.crow_indices.to(torch.int64)
    columns = program.col_indices.to(torch.int64)
    values = program.values
    rows = torch.repeat_interleave(
        torch.arange(program.num_rows, device=values.device),
        torch.diff(crow),
    )
    order = torch.argsort(columns, stable=True)
    transpose_crow = torch.zeros(
        program.num_columns + 1,
        dtype=torch.int64,
        device=values.device,
    )
    transpose_crow[1:] = torch.bincount(columns, minlength=program.num_columns).cumsum(
        0,
    )
    row_scale = torch.ones(program.num_rows, dtype=values.dtype, device=values.device)
    column_scale = torch.ones(
        program.num_columns,
        dtype=values.dtype,
        device=values.device,
    )
    for _ in range(ruiz_iterations):
        magnitude = ((values * row_scale[rows]) * column_scale[columns]).abs()
        row_max = torch.zeros_like(row_scale).scatter_reduce(0, rows, magnitude, "amax")
        column_max = torch.zeros_like(column_scale).scatter_reduce(
            0,
            columns,
            magnitude,
            "amax",
        )
        row_scale = _divide_by_root(row_scale, row_max)
        column_scale = _divide_by_root(column_scale, column_max)
    magnitude = ((values * row_scale[rows]) * column_scale[columns]).abs()
    row_sum = torch.segment_reduce(magnitude, "sum", offsets=crow)
    column_sum = torch.segment_reduce(magnitude[order], "sum", offsets=transpose_crow)
    row_scale = _divide_by_root(row_scale, row_sum)
    column_scale = _divide_by_root(column_scale, column_sum)

    row_lower = program.row_lower * row_scale
    row_upper = program.row_upper * row_scale
    objective = program.objective * column_scale
    zero = column_scale == 0
    lower = torch.where(zero, 0.0, program.lower / column_scale)
    upper = torch.where(zero, 0.0, program.upper / column_scale)
    bound_rescaling = 1.0 / (bound_norm(row_lower, row_upper).item() + 1.0)
    objective_rescaling = 1.0 / ((objective * objective).sum().sqrt().item() + 1.0)
    scaled_values = (values * row_scale[rows]) * column_scale[columns]
    return ScaledProgram(
        program=LinearProgram(
            crow_indices=program.crow_indices,
            col_indices=program.col_indices,
            values=scaled_values,
            num_columns=program.num_columns,
            row_lower=row_lower * bound_rescaling,
            row_upper=row_upper * bound_rescaling,
            objective=objective * objective_rescaling,
            lower=lower * bound_rescaling,
            upper=upper * bound_rescaling,
            objective_offset=program.objective_offset,
        ),
        matrix=CsrMatrix(program.crow_indices, program.col_indices, scaled_values),
        transpose=CsrMatrix(
            transpose_crow,
            rows[order],
            (values[order] * column_scale[columns[order]]) * row_scale[rows[order]],
        ),
        transpose_order=order,
        row_scale=row_scale,
        column_scale=column_scale,
        bound_rescaling=bound_rescaling,
        objective_rescaling=objective_rescaling,
    )


def bound_norm(row_lower: Tensor, row_upper: Tensor) -> Tensor:
    """Return the Euclidean norm cuOpt gives a program's row bounds.

    An equality row counts its bound once; infinite bounds do not count.

    Args:
      row_lower: Row lower bounds.
      row_upper: Row upper bounds.

    Returns:
      norm: Zero-dimensional tensor.

    """
    lower_squares = torch.where(
        torch.isfinite(row_lower) & (row_lower != row_upper),
        row_lower * row_lower,
        0.0,
    )
    upper_squares = torch.where(torch.isfinite(row_upper), row_upper * row_upper, 0.0)
    return (lower_squares + upper_squares).sum().sqrt()


def _divide_by_root(scale: Tensor, norm: Tensor) -> Tensor:
    """Divide ``scale`` by ``sqrt(norm)`` where ``norm`` is positive (cuOpt's bounded divide)."""
    return torch.where(norm > 0, scale / norm.sqrt(), scale)
