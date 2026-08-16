"""Tests for nanochat data preparation.

Every case here is about the same hazard: preparation writes the vocabulary a
score is measured through, so a mismatch it accepts becomes a number nobody can
attribute. None of these downloads anything -- the shards are staged locally and
the fit runs on a few hundred characters.
"""

from __future__ import annotations

from pathlib import Path

import json

from pyarrow import parquet

import numpy as np
import pyarrow as pa
import pytest

from priml.baselines.nanochat.data import token_bytes_fingerprint
from priml.baselines.nanochat.scripts.prepare_data import (
    BOS_TOKEN,
    RESERVED_TOKENS,
    prepare,
)


VOCAB = 300  # above the 16 reserved tokens and the 256 byte-level merges


def _write_shard(root: Path, index: int, documents: list[str]) -> None:
    """Write one parquet shard holding the given documents."""
    parquet.write_table(
        pa.table({"text": documents}),
        root / f"shard_{index:05d}.parquet",
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Two staged shards with enough text to fit a small vocabulary."""
    for index in range(2):
        _write_shard(
            tmp_path,
            index,
            [f"document {index} {word} " * 8 for word in ("alpha", "beta", "gamma")],
        )
    return tmp_path


def _prepare(corpus: Path, **overrides: int) -> Path:
    """Prepare ``corpus`` at small flags, overriding one at a time."""
    arguments: dict[str, int] = {
        "num_train_shards": 1,
        "vocab_size": VOCAB,
        "tokenizer_train_chars": 1_000,
        "tokenizer_doc_cap": 100,
    }
    arguments.update(overrides)
    return prepare(corpus, download=False, **arguments)


def test_preparation_writes_the_vocabulary_and_its_byte_table(corpus: Path) -> None:
    """The three artifacts the loader reads, and nothing else."""
    _prepare(corpus)
    directory = corpus / "tokenizer"
    assert (directory / "tokenizer.pkl").is_file()
    assert (directory / "token_bytes.npy").is_file()
    assert (directory / "tokenizer_recipe.json").is_file()


def test_the_recorded_fingerprint_matches_the_table_written(corpus: Path) -> None:
    """The recipe's digest is what makes a score attributable.

    A recorded digest that did not match its own table would leave the loader
    rejecting a vocabulary the preparer had just written.
    """
    _prepare(corpus)
    directory = corpus / "tokenizer"
    table = np.load(directory / "token_bytes.npy")
    recipe = json.loads((directory / "tokenizer_recipe.json").read_text())
    assert recipe["token_bytes_sha256"] == token_bytes_fingerprint(table)


def test_reserved_tokens_carry_no_bytes(corpus: Path) -> None:
    """They are document boundaries, not text, so they leave the denominator.

    Counting them would make the score depend on how often documents end,
    which is a property of the corpus rather than of the model.
    """
    _prepare(corpus)
    table = np.load(corpus / "tokenizer" / "token_bytes.npy")
    assert int(table[-len(RESERVED_TOKENS) :].sum()) == 0
    assert int(table[: -len(RESERVED_TOKENS)].min()) > 0


def test_the_recipe_records_what_the_vocabulary_was_fitted_on(corpus: Path) -> None:
    """A vocabulary fitted on other text IS a different tokenizer.

    Recording only its size would let a stale one be reused, and every token id
    would then mean something else.
    """
    _prepare(corpus)
    recipe = json.loads((corpus / "tokenizer" / "tokenizer_recipe.json").read_text())
    assert recipe["vocab_size"] == VOCAB
    assert recipe["train_chars"] == 1_000
    assert recipe["doc_cap"] == 100
    assert recipe["shards"] == ["shard_00000.parquet"]
    assert recipe["bos_token"] == BOS_TOKEN


def test_the_validation_shard_is_excluded_from_the_fit(corpus: Path) -> None:
    """The vocabulary must not be fitted on the text it will be scored on."""
    _prepare(corpus, num_train_shards=1)
    recipe = json.loads((corpus / "tokenizer" / "tokenizer_recipe.json").read_text())
    assert "shard_00001.parquet" not in recipe["shards"]


def test_a_vocabulary_fitted_under_other_flags_is_refitted(corpus: Path) -> None:
    """Reusing it would hand back a tokenizer that is not the one asked for.

    Refitted rather than refused: these artifacts are derived and this
    function is how they are derived, so a caller asking for a different
    vocabulary gets one instead of an instruction to delete a file.
    """
    _prepare(corpus)
    before = json.loads(
        (corpus / "tokenizer" / "tokenizer_recipe.json").read_text(),
    )
    _prepare(corpus, tokenizer_train_chars=2_000)
    after = json.loads((corpus / "tokenizer" / "tokenizer_recipe.json").read_text())
    assert before["train_chars"] != after["train_chars"]
    assert after["train_chars"] == 2_000


def test_a_vocabulary_without_its_recipe_is_refitted(corpus: Path) -> None:
    """What it was fitted on cannot be established, so it is fitted again."""
    _prepare(corpus)
    (corpus / "tokenizer" / "tokenizer_recipe.json").unlink()
    _prepare(corpus)
    assert (corpus / "tokenizer" / "tokenizer_recipe.json").is_file()
    assert (corpus / "tokenizer" / "token_bytes.npy").is_file()


def test_refitting_leaves_the_downloaded_shards_alone(corpus: Path) -> None:
    """Only the tokenizer directory is rewritten; the corpus is expensive."""
    _prepare(corpus)
    (corpus / "tokenizer" / "tokenizer_recipe.json").unlink()
    shards = sorted(corpus.glob("shard_*.parquet"))
    before = [path.read_bytes() for path in shards]
    _prepare(corpus)
    assert [path.read_bytes() for path in shards] == before


def test_an_intact_vocabulary_at_the_same_flags_is_reused(corpus: Path) -> None:
    """The check must not block the case it exists to make safe.

    Reuse is verified by CONTENT, not by mtime: the second call must return the
    identical table rather than refit and produce another one.
    """
    _prepare(corpus)
    before = np.load(corpus / "tokenizer" / "token_bytes.npy").copy()
    assert _prepare(corpus) == corpus
    assert np.array_equal(np.load(corpus / "tokenizer" / "token_bytes.npy"), before)


def test_a_corpus_short_a_shard_is_refused(corpus: Path) -> None:
    """The split after the training shards is the validation shard."""
    (corpus / "shard_00001.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="shard_00001"):
        _prepare(corpus)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("num_train_shards", 0, "num_train_shards"),
        ("vocab_size", len(RESERVED_TOKENS), "reserved"),
        ("tokenizer_train_chars", 0, "tokenizer_train_chars"),
        ("tokenizer_doc_cap", 0, "tokenizer_doc_cap"),
    ],
)
def test_an_invalid_flag_is_rejected_by_name(
    corpus: Path,
    field: str,
    value: int,
    match: str,
) -> None:
    """Each bound names its own field, before anything is downloaded or fitted."""
    with pytest.raises(ValueError, match=match):
        _prepare(corpus, **{field: value})


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
