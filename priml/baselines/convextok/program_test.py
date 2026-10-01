"""The linear program must match upstream ConvexTok's bit for bit.

Upstream orders variables as token edges (per pretoken, length-major), then byte
edges, then one indicator per candidate; rows as one flow-conservation row per
vertex, then one "edge needs its token" row per token edge, then the budget.
The solver's trajectory depends on this layout, so values, sparsity pattern and
ordering are all compared exactly.
"""

from pathlib import Path
from typing import Final, cast

import json

from torch import Tensor

import numpy as np
import torch

from priml.baselines.convextok.program import LinearProgram, build_program
from priml.lib.custom_json import DictCodec, IntCodec, ListCodec


_CWD: Final = Path(__file__).resolve().parent


def test_program_matches_upstream_golden() -> None:
    pretokens = _read_json("pretokens.json")
    candidates = _read_json("candidates.json")
    corpus = _read_json("corpus.json")
    built = build_program(
        dict(
            zip(
                ListCodec.coerce(pretokens.get("pretokens"), str, default=None),
                ListCodec.coerce(pretokens.get("frequencies"), int, default=None),
                strict=True,
            ),
        ),
        ListCodec.coerce(candidates.get("tokens"), str, default=None),
        budget=IntCodec.coerce(corpus.get("budget"), default=None),
    )
    program = built.program
    with _npz("program.npz") as golden:
        vertices, columns = (
            int(size) for size in torch.from_numpy(golden["A_eq_shape"])
        )
        assert [
            built.num_token_edges,
            built.num_byte_edges,
            built.num_tokens,
        ] == [int(size) for size in torch.from_numpy(golden["sizes"])]
        assert built.num_vertices == vertices
        assert program.num_columns == columns
        crow = program.crow_indices.to(torch.int64)
        col = program.col_indices.to(torch.int64)
        split = int(crow[vertices])
        _assert_block(
            crow[: vertices + 1],
            col[:split],
            program.values[:split],
            golden,
            "A_eq",
        )
        _assert_block(
            crow[vertices:] - split,
            col[split:],
            program.values[split:],
            golden,
            "A_ub",
        )
        b_eq, b_ub = torch.from_numpy(golden["b_eq"]), torch.from_numpy(golden["b_ub"])
        assert torch.equal(program.row_lower[:vertices], b_eq)
        assert torch.equal(program.row_upper, torch.cat([b_eq, b_ub]))
        assert bool(torch.isneginf(program.row_lower[vertices:]).all())
        assert torch.equal(program.objective, torch.from_numpy(golden["c"]))
        assert torch.equal(program.lower, torch.from_numpy(golden["lb"]))
        assert torch.equal(program.upper, torch.from_numpy(golden["ub"]))


def test_layout_of_a_single_pretoken() -> None:
    built = build_program({"abab": 1}, ["ab"], budget=1)
    assert (built.num_token_edges, built.num_byte_edges, built.num_tokens) == (2, 4, 1)
    expected = [
        # f0 f1  g0  g1  g2  g3  t0     (f: "ab" at 0 and 2; g: bytes)
        [1, 0, 1, 0, 0, 0, 0],  # Vertex 0: source.
        [0, 0, -1, 1, 0, 0, 0],  # Vertex 1.
        [-1, 1, 0, -1, 1, 0, 0],  # Vertex 2.
        [0, 0, 0, 0, -1, 1, 0],  # Vertex 3.
        [0, -1, 0, 0, 0, -1, 0],  # Vertex 4: sink.
        [1, 0, 0, 0, 0, 0, -1],  # f0 needs t0.
        [0, 1, 0, 0, 0, 0, -1],  # f1 needs t0.
        [0, 0, 0, 0, 0, 0, 1],  # Budget.
    ]
    assert _dense(built.program).tolist() == expected
    assert built.program.row_lower.tolist() == [1, 0, 0, 0, -1, *[float("-inf")] * 3]
    assert built.program.row_upper.tolist() == [1, 0, 0, 0, -1, 0, 0, 1]
    assert built.program.objective.tolist() == [1, 1, 1, 1, 1, 1, 0]


def test_edges_follow_length_major_order_and_candidate_ids() -> None:
    built = build_program({"abc": 2}, ["ab", "abc", "bc"], budget=3)
    token_rows = _dense(built.program)[built.num_vertices : -1]
    # Token edges in length-major order: "ab"@0, "bc"@1, "abc"@0; their tokens are
    # candidates 0, 2 and 1, whose indicators follow the three byte edges.
    assert token_rows[:, 6:].tolist() == [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]
    assert built.program.objective.tolist() == [2] * 6 + [0] * 3


def _dense(program: LinearProgram) -> Tensor:
    lengths = torch.diff(program.crow_indices.to(torch.int64))
    rows = torch.repeat_interleave(torch.arange(program.num_rows), lengths)
    dense = torch.zeros(program.num_rows, program.num_columns, dtype=torch.float64)
    dense[rows, program.col_indices.to(torch.int64)] = program.values
    return dense


def _assert_block(
    crow: Tensor,
    col: Tensor,
    values: Tensor,
    golden: np.lib.npyio.NpzFile,
    name: str,
) -> None:
    assert torch.equal(crow, torch.from_numpy(golden[f"{name}_indptr"]))
    assert torch.equal(col, torch.from_numpy(golden[f"{name}_indices"]))
    assert torch.equal(values, torch.from_numpy(golden[f"{name}_data"]))


def _npz(name: str) -> np.lib.npyio.NpzFile:
    loaded = cast(object, np.load(_CWD / "testdata" / name))
    assert isinstance(loaded, np.lib.npyio.NpzFile)
    return loaded


def _read_json(name: str) -> dict[str, object]:
    raw = cast(object, json.loads((_CWD / "testdata" / name).read_text()))
    return dict(DictCodec.coerce(raw, default=None))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
