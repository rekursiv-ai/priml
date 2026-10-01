"""Preparation must turn raw shards into upstream ConvexTok's tokenizer, end to end.

On the eight-document fixture corpus the prepared tokenizer's learned pieces must be
upstream's ``det`` vocabulary, and held-out text must segment into upstream's token
strings.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

import json

from configgle import Fig
from pyarrow import parquet
from tokenizers import Tokenizer, pre_tokenizers
from torch import Tensor

import pyarrow as pa
import pytest
import torch

from priml.baselines.convextok.prepare import (
    ConvexTokPreparation,
    donor_convextok16k,
)
from priml.baselines.convextok.program import LinearProgram
from priml.baselines.nanochat.scripts.prepare_data import donor_unigram16k
from priml.lib.custom_json import DictCodec, IntCodec, ListCodec


_CWD: Final = Path(__file__).resolve().parent


# The whole preparation, parquet to tokenizer.json: 0.3 s even with the kernels
# interpreted, past the unit tier's 100 ms.
@pytest.mark.compute_large_fixture
def test_fixture_corpus_yields_upstream_tokenizer(tmp_path: Path) -> None:
    corpus = _read_json("corpus.json")
    config = _fixture_config(tmp_path)
    config.make().build()

    tokenizer = Tokenizer.from_file(str(config.working_dir / "tokenizer.json"))
    learned = _learned_pieces(tokenizer)
    alphabet = set(pre_tokenizers.ByteLevel.alphabet())
    golden = set(
        ListCodec.coerce(_read_json("vocab.json").get("det"), str, default=None),
    )
    assert learned == golden - alphabet
    held_out = [
        tokenizer.encode(text, add_special_tokens=False).tokens
        for text in ListCodec.coerce(corpus.get("heldout"), str, default=None)
    ]
    assert held_out == [
        ListCodec.coerce(row, str, default=None)
        for row in ListCodec.coerce(
            _read_json("heldout_tokens.json").get("tokens"),
            object,
            default=None,
        )
    ]


# The same pipeline as above, with a stand-in solver instead of PDLP and presolve.
@pytest.mark.compute_large_fixture
def test_any_solver_fills_the_solver_slot(tmp_path: Path) -> None:
    """Every variable at 1 ties every indicator, so rounding keeps candidate order."""
    config = _fixture_config(tmp_path)
    config.presolve = None
    config.solver = _AllOnes.Config()
    config.make().build()

    tokenizer = Tokenizer.from_file(str(config.working_dir / "tokenizer.json"))
    candidates = ListCodec.coerce(
        _read_json("candidates.json").get("tokens"),
        str,
        default=None,
    )
    budget = IntCodec.coerce(_read_json("corpus.json").get("budget"), default=None)
    assert _learned_pieces(tokenizer) == set(candidates[:budget])


def test_vocabulary_must_leave_room_for_every_byte() -> None:
    config = ConvexTokPreparation.Config()
    config.vocab_size = 256
    with pytest.raises(ValueError, match="byte"):
        config.make()


def test_donor_recipe_fits_convextok_on_its_raw_shards(tmp_path: Path) -> None:
    """The baseline's recipe with ConvexTok in its tokenizer slot and its ten IDs."""
    config = donor_convextok16k()
    config.working_dir = tmp_path
    finalized = config.copy_tree().finalize()
    tokenizer = finalized.tokenizer
    assert isinstance(tokenizer, ConvexTokPreparation.Config)
    assert tokenizer.raw_dir == tmp_path / "raw"
    assert tokenizer.working_dir == tmp_path / "convextok16k"
    assert finalized.rows.tokenizer.path == tmp_path / "convextok16k/tokenizer.json"
    assert finalized.rows.tokenizer.reserved_count == 10
    unigram = donor_unigram16k()
    unigram.working_dir = tmp_path
    assert finalized.corpus == unigram.copy_tree().finalize().corpus


def _fixture_config(tmp_path: Path) -> ConvexTokPreparation.Config:
    """Write the fixture corpus as one raw shard and point a CPU preparation at it."""
    corpus = _read_json("corpus.json")
    raw = tmp_path / "raw"
    raw.mkdir()
    parquet.write_table(
        pa.table(
            {"text": ListCodec.coerce(corpus.get("texts"), str, default=None)},
        ),
        raw / "shard_00000.parquet",
    )
    config = ConvexTokPreparation.Config()
    config.raw_dir = raw
    config.shard_indices = [0]
    config.working_dir = tmp_path / "convextok"
    config.vocab_size = IntCodec.coerce(corpus.get("vocab_size"), default=None)
    config.num_workers = 1
    config.device = "cpu"
    return config


def _learned_pieces(tokenizer: Tokenizer) -> set[str | None]:
    """Return the tokenizer's pieces after its 256 bytes."""
    return {
        tokenizer.id_to_token(index) for index in range(256, tokenizer.get_vocab_size())
    }


def _read_json(name: str) -> dict[str, object]:
    raw = cast(object, json.loads((_CWD / "testdata" / name).read_text()))
    return dict(DictCodec.coerce(raw, default=None))


@dataclass(frozen=True, slots=True, kw_only=True)
class _Solution:
    primal: Tensor


class _AllOnes:
    """A stand-in solver that is not PDLP: every variable at its upper bound, 1."""

    class Config(Fig["_AllOnes"]):
        """No settings."""

    def __init__(self, config: Config) -> None:
        del config

    def __call__(self, program: LinearProgram, /) -> _Solution:
        return _Solution(primal=torch.ones(program.num_columns, dtype=torch.float64))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
