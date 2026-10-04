"""Assemble ConvexTok's linear program.

Each unique pretoken of ``n`` bytes contributes a path graph of ``n + 1``
vertices: byte edges between neighbours, and a token edge wherever a candidate
spells the bytes in between. One unit of flow runs from the first vertex to the
last, a token edge may only carry flow if its candidate is chosen, and the
budget caps how many candidates are chosen. The objective counts tokens,
weighted by pretoken frequency.

The layout follows upstream exactly -- variables are token edges (per pretoken,
by length then start), byte edges, then candidate indicators; rows are vertex
equalities, then one ``edge - indicator <= 0`` row per token edge, then the
budget -- because the solver's trajectory depends on it.
"""

from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import math

from torch import Tensor

import torch


@dataclass(frozen=True, slots=True, kw_only=True)
class LinearProgram:
    """``min objective . x + objective_offset`` s.t. ``row_lower <= A x <= row_upper``.

    The row-bounds CSR form the presolver and the solver share. The constraint
    matrix ``A`` is stored as CSR components; variables are bounded by
    ``lower <= x <= upper``. Infinite bounds are ``+-inf``.

    Attributes:
      crow_indices: Row start offsets into ``col_indices`` and ``values``.
      col_indices: Column of each stored entry, ascending within a row.
      values: Value of each stored entry, float64.
      num_columns: Number of variables.
      row_lower: Row lower bounds.
      row_upper: Row upper bounds.
      objective: Cost per variable.
      lower: Variable lower bounds.
      upper: Variable upper bounds.
      objective_offset: Constant added to the objective, e.g. by presolve.

    """

    crow_indices: Tensor
    col_indices: Tensor
    values: Tensor
    num_columns: int
    row_lower: Tensor
    row_upper: Tensor
    objective: Tensor
    lower: Tensor
    upper: Tensor
    objective_offset: float = 0.0

    @property
    def num_rows(self) -> int:
        """Number of constraint rows."""
        return len(self.crow_indices) - 1

    def to(self, device: torch.device | str) -> "LinearProgram":
        """Return the program with every tensor on ``device``.

        Args:
          device: Target device.

        Returns:
          program: The moved program.

        """
        return LinearProgram(
            crow_indices=self.crow_indices.to(device),
            col_indices=self.col_indices.to(device),
            values=self.values.to(device),
            num_columns=self.num_columns,
            row_lower=self.row_lower.to(device),
            row_upper=self.row_upper.to(device),
            objective=self.objective.to(device),
            lower=self.lower.to(device),
            upper=self.upper.to(device),
            objective_offset=self.objective_offset,
        )


class LpSolution(Protocol):
    """A solver's answer: the primal point, one entry per column of the program."""

    @property
    def primal(self) -> Tensor:
        """The solution's variable values."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class TokenizationProgram:
    """ConvexTok's linear program and the column layout that reads a solution.

    Attributes:
      program: The linear program.
      num_vertices: Equality rows, one per path-graph vertex.
      num_token_edges: Token-edge variables, first in column order.
      num_byte_edges: Byte-edge variables, after the token edges.
      num_tokens: Candidate indicators, last in column order.

    """

    program: LinearProgram
    num_vertices: int
    num_token_edges: int
    num_byte_edges: int
    num_tokens: int


def build_program(
    pretokens: dict[str, int],
    candidates: Sequence[str],
    *,
    budget: int,
) -> TokenizationProgram:
    """Build the flow linear program over pretokens with candidate indicators.

    Args:
      pretokens: Pretoken to frequency, in first-occurrence order.
      candidates: Candidate tokens in the order that numbers their indicators.
      budget: Most candidates the solution may choose.

    Returns:
      program: The linear program, laid out exactly as upstream lays it out.

    """
    counts, starts, lengths, token_ids = token_edges(
        pretokens,
        {token: position for position, token in enumerate(candidates)},
    )
    sizes = torch.tensor([len(piece) for piece in pretokens])
    weights = torch.tensor(list(pretokens.values()), dtype=torch.float64)
    edge_owner = torch.repeat_interleave(_tensor(counts))
    byte_owner = torch.repeat_interleave(sizes)
    vertex_start = torch.cumsum(sizes + 1, dim=0) - (sizes + 1)
    byte_offset = (
        torch.arange(len(byte_owner)) - (torch.cumsum(sizes, dim=0) - sizes)[byte_owner]
    )
    token_start = vertex_start[edge_owner] + _tensor(starts)
    shape = _Shape(
        vertices=int((sizes + 1).sum()),
        token_edges=len(edge_owner),
        byte_edges=len(byte_owner),
        tokens=len(candidates),
    )
    crow_indices, col_indices, values = _constraint_matrix(
        shape,
        token_start=token_start,
        token_end=token_start + _tensor(lengths),
        token_ids=_tensor(token_ids),
        byte_start=vertex_start[byte_owner] + byte_offset,
    )
    flow = torch.zeros(shape.vertices)
    flow[vertex_start] = 1.0
    flow[vertex_start + sizes] = -1.0
    program = LinearProgram(
        crow_indices=crow_indices,
        col_indices=col_indices,
        values=values,
        num_columns=shape.columns,
        row_lower=torch.cat(
            [
                flow,
                torch.full(
                    (shape.token_edges + 1,),
                    -math.inf,
                    dtype=torch.float64,
                ),
            ],
        ),
        row_upper=torch.cat(
            [
                flow,
                torch.zeros(shape.token_edges),
                torch.tensor([float(budget)], dtype=torch.float64),
            ],
        ),
        objective=torch.cat(
            [
                weights[edge_owner],
                weights[byte_owner],
                torch.zeros(shape.tokens),
            ],
        ),
        lower=torch.zeros(shape.columns, dtype=torch.float64),
        upper=torch.ones(shape.columns, dtype=torch.float64),
    )
    return TokenizationProgram(
        program=program,
        num_vertices=shape.vertices,
        num_token_edges=shape.token_edges,
        num_byte_edges=shape.byte_edges,
        num_tokens=shape.tokens,
    )


def token_edges(
    pretokens: Iterable[str],
    index: dict[str, int],
) -> tuple[array[int], array[int], array[int], array[int]]:
    """Find every candidate occurrence inside each pretoken, length-major.

    Args:
      pretokens: Pretokens in order.
      index: Candidate to indicator position.

    Returns:
      counts: Token edges per pretoken.
      starts: Byte offset of each edge within its pretoken.
      lengths: Byte length of each edge.
      token_ids: Indicator position of each edge's candidate.

    """
    counts, starts, lengths, token_ids = array("q"), array("q"), array("q"), array("q")
    for piece in pretokens:
        size = len(piece)
        found = len(token_ids)
        for length in range(2, size):
            for start in range(size - length + 1):
                token_id = index.get(piece[start : start + length])
                if token_id is not None:
                    starts.append(start)
                    lengths.append(length)
                    token_ids.append(token_id)
        if size >= 2:
            token_id = index.get(piece)
            if token_id is not None:
                starts.append(0)
                lengths.append(size)
                token_ids.append(token_id)
        counts.append(len(token_ids) - found)
    return counts, starts, lengths, token_ids


@dataclass(frozen=True, slots=True, kw_only=True)
class _Shape:
    vertices: int
    token_edges: int
    byte_edges: int
    tokens: int

    @property
    def columns(self) -> int:
        return self.token_edges + self.byte_edges + self.tokens

    @property
    def rows(self) -> int:
        return self.vertices + self.token_edges + 1


def _constraint_matrix(
    shape: _Shape,
    *,
    token_start: Tensor,
    token_end: Tensor,
    token_ids: Tensor,
    byte_start: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Place every entry, then sort into CSR order: by row, then by column."""
    token_columns = torch.arange(shape.token_edges)
    byte_columns = shape.token_edges + torch.arange(shape.byte_edges)
    indicator_columns = (
        shape.token_edges + shape.byte_edges + torch.arange(shape.tokens)
    )
    edge_rows = shape.vertices + token_columns
    budget_row = torch.full((shape.tokens,), shape.rows - 1)
    rows = torch.cat(
        [
            token_start,
            token_end,
            byte_start,
            byte_start + 1,
            edge_rows,
            edge_rows,
            budget_row,
        ],
    )
    columns = torch.cat(
        [
            token_columns,
            token_columns,
            byte_columns,
            byte_columns,
            token_columns,
            indicator_columns[token_ids],
            indicator_columns,
        ],
    )
    ones_token, ones_byte = torch.ones(shape.token_edges), torch.ones(shape.byte_edges)
    values = torch.cat(
        [
            ones_token,
            -ones_token,
            ones_byte,
            -ones_byte,
            ones_token,
            -ones_token,
            torch.ones(shape.tokens),
        ],
    ).to(torch.float64)
    order = torch.argsort(rows * shape.columns + columns)
    index_dtype = (
        torch.int32
        if max(len(order), shape.columns) <= torch.iinfo(torch.int32).max
        else torch.int64
    )
    crow_indices = torch.zeros(shape.rows + 1, dtype=torch.int64)
    crow_indices[1:] = torch.cumsum(torch.bincount(rows, minlength=shape.rows), dim=0)
    return crow_indices.to(index_dtype), columns[order].to(index_dtype), values[order]


def _tensor(values: array[int]) -> Tensor:
    """Copy an int64 array into a tensor without a per-element Python loop."""
    if not values:
        return torch.zeros(0, dtype=torch.int64)
    return torch.frombuffer(values, dtype=torch.int64).clone()
