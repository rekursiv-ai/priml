"""Schemas of the upstream ConvexTok goldens under ``testdata/``, and their readers.

Read one with ``json_loads((TESTDATA / "corpus.json").read_text(), Corpus)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict, cast

from treekle import json_loads


if TYPE_CHECKING:
    import numpy as np
    import torch

    from priml.baselines.convextok.presolver.presolve import Presolved
    from priml.baselines.convextok.program import LinearProgram, build_program
else:
    from wrapt import lazy_import

    # Every schema reader imports this module; only the program helpers need these.
    np = lazy_import("numpy")
    torch = lazy_import("torch")
    LinearProgram = lazy_import(
        "priml.baselines.convextok.program",
        "LinearProgram",
    )
    build_program = lazy_import(
        "priml.baselines.convextok.program",
        "build_program",
    )


_CWD: Final = Path(__file__).resolve().parent

TESTDATA: Final = _CWD / "testdata"


class Corpus(TypedDict):
    """``corpus.json``: the documents, the held-out split, and the LP knobs."""

    texts: list[str]
    heldout: list[str]
    vocab_size: int
    budget: int
    split_pattern: str


class Pretokens(TypedDict):
    """``pretokens.json``: upstream's pretoken table, in first-occurrence order."""

    pretokens: list[str]
    frequencies: list[int]


class Candidates(TypedDict):
    """``candidates.json``: upstream's candidate tokens and their counts."""

    tokens: list[str]
    counts: list[int]


class Vocab(TypedDict):
    """``vocab.json``: upstream's vocabulary under each rounding rule."""

    det: list[str]
    bias: list[str]
    all_ones: list[str]


class HeldoutTokens(TypedDict):
    """``heldout_tokens.json``: upstream's tokenization of each held-out text."""

    tokens: list[list[str]]


def fixture_program() -> LinearProgram:
    """Build the eight-document fixture's program from upstream's pretokens.

    Returns:
      program: The fixture corpus's ConvexTok linear program.

    """
    pretokens = json_loads((TESTDATA / "pretokens.json").read_text(), Pretokens)
    return build_program(
        dict(
            zip(pretokens["pretokens"], pretokens["frequencies"], strict=True),
        ),
        json_loads((TESTDATA / "candidates.json").read_text(), Candidates)["tokens"],
        budget=json_loads((TESTDATA / "corpus.json").read_text(), Corpus)["budget"],
    ).program


def npz(path: Path) -> np.lib.npyio.NpzFile:
    """Open a recorded ``.npz`` golden."""
    loaded = cast(object, np.load(path))
    assert isinstance(loaded, np.lib.npyio.NpzFile)
    return loaded


def input_program(golden: np.lib.npyio.NpzFile) -> LinearProgram:
    """Rebuild the program a PSLP golden recorded under its ``in_*`` keys.

    Args:
      golden: An opened ``presolve_fuzz`` golden.

    Returns:
      program: The program PSLP was given.

    """

    def tensor(key: str) -> torch.Tensor:
        return torch.from_numpy(golden[f"in_{key}"])

    return LinearProgram(
        crow_indices=tensor("crow_indices"),
        col_indices=tensor("col_indices"),
        values=tensor("values"),
        num_columns=int(golden["in_num_columns"]),
        row_lower=tensor("row_lower"),
        row_upper=tensor("row_upper"),
        objective=tensor("objective"),
        lower=tensor("lower"),
        upper=tensor("upper"),
    )


def assert_reduced_like(presolved: Presolved, golden: np.lib.npyio.NpzFile) -> None:
    """Raise unless the reduced program equals PSLP's, array for array, bitwise.

    Args:
      presolved: Our presolve's result.
      golden: PSLP's recorded reduction of the same program.

    Raises:
      ValueError: Naming the first array or scalar that differs.

    """
    reduced = presolved.program
    for got, key in (
        (reduced.crow_indices, "Ap"),
        (reduced.col_indices, "Ai"),
        (reduced.values, "Ax"),
        (reduced.row_lower, "lhs"),
        (reduced.row_upper, "rhs"),
        (reduced.objective, "c"),
        (reduced.lower, "lbs"),
        (reduced.upper, "ubs"),
    ):
        if not np.array_equal(got.numpy(), golden[key]):
            raise ValueError(key)
    if reduced.num_columns != len(golden["c"]):
        raise ValueError('Expected reduced.num_columns == len(golden["c"]).')
    if reduced.objective_offset != float(golden["offset"]):
        raise ValueError(
            'Expected reduced.objective_offset == float(golden["offset"]).',
        )
