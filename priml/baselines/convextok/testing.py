"""Schemas of the upstream ConvexTok goldens under ``testdata/``.

Read one with ``json_loads((TESTDATA / "corpus.json").read_text(), Corpus)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict


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
