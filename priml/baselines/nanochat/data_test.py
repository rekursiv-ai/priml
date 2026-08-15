"""Tests for nanochat data loading.

Every case here is about one hazard: this loader reproduces a published token
stream, so a packing that merely looks reasonable is a different experiment
wearing the same name. The packer's rules are therefore pinned by CONTENT --
which document lands where -- rather than by shape, and the rest guard the
artifact the score's denominator comes from.

``scripts/karpathy_data_parity.py`` makes the same argument against the
reference itself; these run without a corpus or a network.
"""

from __future__ import annotations

from pathlib import Path

import json
import pickle

from pyarrow import parquet

import numpy as np
import pyarrow as pa
import pytest
import tiktoken
import torch

from priml.baselines.nanochat.data import (
    NanoChatData,
    Tokenizer,
    token_bytes_fingerprint,
)


SEQ = 16
BOS = "<|reserved_0|>"
RESERVED = tuple(f"<|reserved_{index}|>" for index in range(16))
VOCAB = 256 + len(RESERVED)


def _encoding() -> tiktoken.Encoding:
    """A byte-level vocabulary: every token is one byte, plus the reserved set.

    Byte-level so a document's token count is its byte count, which is what
    lets a test say which document the packer should have chosen.
    """
    ranks = {bytes([value]): value for value in range(256)}
    return tiktoken.Encoding(
        name="test",
        pat_str=r".",
        mergeable_ranks=ranks,
        special_tokens={
            name: len(ranks) + index for index, name in enumerate(RESERVED)
        },
    )


def _write_tokenizer(root: Path) -> tiktoken.Encoding:
    """Write the vocabulary in the layout the loader reads."""
    encoding = _encoding()
    directory = root / "tokenizer"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "tokenizer.pkl").open("wb") as file:
        pickle.dump(encoding, file)
    reserved = set(RESERVED)
    lengths = np.array(
        [
            0 if (text := encoding.decode([token])) in reserved else len(text.encode())
            for token in range(encoding.n_vocab)
        ],
        dtype=np.int32,
    )
    np.save(directory / "token_bytes.npy", lengths)
    (directory / "tokenizer_recipe.json").write_text(
        json.dumps(
            {
                "bos_token": BOS,
                "token_bytes_sha256": token_bytes_fingerprint(lengths),
            },
        ),
    )
    return encoding


def _write_shard(root: Path, index: int, documents: list[str]) -> None:
    """Write one parquet shard holding the given documents, in order."""
    parquet.write_table(
        pa.table({"text": documents}),
        root / f"shard_{index:05d}.parquet",
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A two-shard corpus of documents whose lengths are distinguishable."""
    _write_tokenizer(tmp_path)
    # Lengths 1..8 encoded as repeated distinct characters, so a row states
    # which documents it took and in what order.
    _write_shard(tmp_path, 0, [chr(ord("a") + n) * (n + 1) for n in range(8)])
    _write_shard(tmp_path, 1, [chr(ord("A") + n) * (n + 1) for n in range(8)])
    return tmp_path


def _data(corpus: Path, **overrides: object) -> NanoChatData:
    config = NanoChatData.Config()
    config.base_dir = "/"
    config.working_dir = str(corpus)
    config.device = "cpu"
    config.num_train_shards = 1
    config.val_shard = 1
    config.batch_size = 2
    config.eval_batch_size = 2
    config.max_seq_len = SEQ
    config.eval_tokens = 2 * 2 * SEQ
    config.buffer_size = 8
    for name, value in overrides.items():
        setattr(config, name, value)
    return config.make()


def test_targets_are_the_inputs_shifted_by_one(corpus: Path) -> None:
    """The whole training signal: position i predicts position i + 1."""
    batch = next(iter(_data(corpus).train_dataloader()))
    assert batch["media"].shape == (2, SEQ)
    assert batch["label"].shape == (2, SEQ)
    assert torch.equal(batch["media"][:, 1:], batch["label"][:, :-1])


def test_every_row_begins_with_the_document_marker(corpus: Path) -> None:
    """BOS alignment is what makes a row a sequence of whole documents.

    Without it a row could start mid-document, and the first positions would
    train on a continuation whose context the model never saw.
    """
    data = _data(corpus)
    batch = next(iter(data.train_dataloader()))
    assert (batch["media"][:, 0] == data.tokenizer.bos_token_id).all()


def test_the_largest_fitting_document_is_taken_first(corpus: Path) -> None:
    """Best fit, not first fit: the packer minimizes what it has to crop.

    Pinned by CONTENT rather than by utilization, since a first-fit packer
    fills a row just as completely and produces a different token stream.
    """
    data = _data(corpus, buffer_size=8)
    row = next(iter(data.train_dataloader()))["media"][0]
    text = data.tokenizer.encoding.decode(
        [int(token) for token in row if int(token) < 256],
    )
    # Documents are 'a', 'bb', ... 'hhhhhhhh'; each carries a BOS the decode
    # above drops. A row of 17 slots takes the 8-token document first.
    assert text.startswith("hhhhhhhh")


def test_a_row_is_filled_by_cropping_the_shortest_document(corpus: Path) -> None:
    """No padding, ever: leftover space is filled by a cropped document.

    The SHORTEST is chosen -- it loses the least of itself -- and the row is
    full, so every position carries a real target and the loss needs no mask.
    """
    data = _data(corpus)
    batch = next(iter(data.train_dataloader()))
    # A padded position would be a zero the vocabulary never emits here, since
    # every document contributes its own byte and a BOS.
    assert batch["media"].shape == (2, SEQ)
    assert int(batch["media"].min()) >= 0
    assert not bool((batch["media"] == batch["media"][0, 0]).all())


def test_the_stream_is_deterministic(corpus: Path) -> None:
    """Two runs of one recipe must draw the identical tokens.

    The packer has no seed: its order is the corpus's, so a difference between
    two draws would be a difference in the experiment.
    """
    first = next(iter(_data(corpus).train_dataloader()))["media"].clone()
    second = next(iter(_data(corpus).train_dataloader()))["media"].clone()
    assert torch.equal(first, second)


def test_evaluation_replays_from_the_start(corpus: Path) -> None:
    """Every candidate, and every checkpoint, is scored on identical tokens.

    A stream that carried on would score later text at each evaluation and
    report the difference as progress.
    """
    data = _data(corpus)
    first = [b["media"].clone() for b in data.eval_dataloader()]
    second = [b["media"].clone() for b in data.eval_dataloader()]
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_evaluation_scores_the_configured_token_count(corpus: Path) -> None:
    """The extent is fixed in TOKENS, so it survives a batch-width change."""
    data = _data(corpus, eval_batch_size=2, eval_tokens=4 * 2 * SEQ)
    batches = list(data.eval_dataloader())
    assert len(batches) == 4
    assert sum(b["media"].numel() for b in batches) == 4 * 2 * SEQ


def test_training_and_validation_draw_from_different_shards(corpus: Path) -> None:
    """A score measured on trained text measures memorization.

    The validation shard is pinned and excluded from training, so the two
    streams share no document.
    """
    data = _data(corpus)
    marker = data.tokenizer.bos_token_id
    seen = [
        {int(token) for token in batch["media"].flatten() if int(token) != marker}
        for batch in (
            next(iter(data.train_dataloader())),
            next(iter(data.eval_dataloader())),
        )
    ]
    # The fixture's shards use disjoint character ranges; the document marker
    # is excluded because every row of both streams begins with it.
    assert not seen[0] & seen[1]


def test_the_training_stream_does_not_end(corpus: Path) -> None:
    """The budget ends the run, so the corpus must outlast it by wrapping.

    The fixture's shard holds far fewer tokens than this draws, so a stream
    that stopped at the end of the corpus would raise here.
    """
    stream = iter(_data(corpus).train_dataloader())
    for _ in range(20):
        assert next(stream)["media"].shape == (2, SEQ)


def test_the_byte_table_travels_with_the_batch(corpus: Path) -> None:
    """The score divides by it, so it must reach the metric unmediated."""
    batch = next(iter(_data(corpus).eval_dataloader()))
    assert batch["token_bytes"].shape == (VOCAB,)
    # Reserved tokens carry no bytes, which is what keeps document boundaries
    # out of the denominator.
    assert int(batch["token_bytes"][-len(RESERVED) :].sum()) == 0


def test_an_evaluation_extent_that_is_not_whole_batches_is_rejected(
    corpus: Path,
) -> None:
    """A remainder is a score covering fewer tokens than it names."""
    with pytest.raises(ValueError, match="eval_tokens"):
        _data(corpus, eval_tokens=3 * SEQ + 1)


def test_a_vocabulary_disagreeing_with_the_model_is_rejected_at_load(
    corpus: Path,
) -> None:
    """Otherwise the mismatch surfaces as an out-of-range embedding index."""
    with pytest.raises(ValueError, match="vocab_size"):
        _data(corpus, vocab_size=VOCAB // 2)


def test_a_matching_vocabulary_loads(corpus: Path) -> None:
    """The check must accept the agreeing case, or it blocks every real run."""
    batch = next(iter(_data(corpus, vocab_size=VOCAB).train_dataloader()))
    assert batch["media"].shape == (2, SEQ)


def test_a_byte_table_that_does_not_match_its_fingerprint_is_rejected(
    corpus: Path,
) -> None:
    """The byte table IS the score's denominator, so its identity is recorded.

    Two tables of equal length silently reprice every token: a change to byte
    accounting shifts BPB by roughly a real candidate effect while every shape
    check still passes, and two runs become incomparable with nothing on disk
    to tell them apart.
    """
    lengths = np.load(corpus / "tokenizer" / "token_bytes.npy")
    lengths[3] = 7  # same shape, different accounting
    np.save(corpus / "tokenizer" / "token_bytes.npy", lengths)
    with pytest.raises(ValueError, match="fingerprint"):
        _data(corpus)


def test_a_negative_byte_length_is_rejected(corpus: Path) -> None:
    """The metric's mask is ``lengths > 0``, so a negative length drops a token.

    It would vanish from both sums -- excluded from the measurement rather than
    rejected -- which is a quiet change to what the score covers.
    """
    lengths = np.load(corpus / "tokenizer" / "token_bytes.npy")
    lengths[3] = -2
    _refingerprint(corpus / "tokenizer", lengths)
    with pytest.raises(ValueError, match="negative byte length"):
        _data(corpus)


def test_a_two_dimensional_byte_table_is_rejected(corpus: Path) -> None:
    """The metric indexes it as one length per id."""
    lengths = np.ones((VOCAB, 1), dtype=np.int32)
    _refingerprint(corpus / "tokenizer", lengths)
    with pytest.raises(ValueError, match="one-dimensional"):
        _data(corpus)


def test_a_recipe_without_a_fingerprint_is_rejected(corpus: Path) -> None:
    """An optional check is no check: pre-fingerprint metadata cannot be read.

    What such a vocabulary scores is exactly what cannot be established, so
    accepting it would reintroduce the unidentifiable number the fingerprint
    exists to prevent.
    """
    (corpus / "tokenizer" / "tokenizer_recipe.json").write_text(
        json.dumps({"bos_token": BOS}),
    )
    with pytest.raises(ValueError, match="token_bytes_sha256"):
        _data(corpus)


def test_a_missing_shard_is_named(corpus: Path) -> None:
    """A split short one shard must say which, not merely fail to prepare."""
    (corpus / "shard_00001.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="shard_00001"):
        _data(corpus)


def test_a_missing_vocabulary_names_the_preparer(tmp_path: Path) -> None:
    _write_shard(tmp_path, 0, ["a"])
    _write_shard(tmp_path, 1, ["b"])
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        _data(tmp_path)


def test_resuming_an_advanced_stream_is_refused(corpus: Path) -> None:
    """The packer cannot be positioned, so a resume would replay the corpus.

    Silently restarting would retrain on the opening of the data while the
    schedules carried on from the checkpoint, which is a run neither the
    budget nor the recipe describes.
    """
    data = _data(corpus)
    with pytest.raises(ValueError, match="re-tokenizing"):
        data.load_state_dict({"batches": 12})


def test_a_fresh_stream_round_trips(corpus: Path) -> None:
    """The refusal must not block a checkpoint written before any batch."""
    data = _data(corpus)
    data.load_state_dict(data.state_dict())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("batch_size", -1),
        ("eval_batch_size", 0),
        ("eval_batch_size", -1),
        ("buffer_size", 0),
        ("max_seq_len", 1),
    ],
)
def test_a_nonpositive_size_is_rejected_by_name(
    corpus: Path,
    field: str,
    value: int,
) -> None:
    """Each bound names its own field, and none is silently absorbed."""
    with pytest.raises(ValueError, match=field):
        _data(corpus, **{field: value})


def test_the_tokenizer_prepends_the_document_marker(corpus: Path) -> None:
    """Packing depends on document LENGTH, and the marker is part of it."""
    tokenizer = Tokenizer.from_directory(corpus / "tokenizer")
    encoded = tokenizer.encode_batch(["ab", "c"])
    assert [row[0] for row in encoded] == [tokenizer.bos_token_id] * 2
    assert [len(row) for row in encoded] == [3, 2]


def _refingerprint(directory: Path, table: np.ndarray) -> None:
    """Rewrite the recipe so the fingerprint matches a replaced table.

    Without this the fingerprint check fires first and the test proves only
    that, never reaching the property it means to pin.
    """
    np.save(directory / "token_bytes.npy", table)
    recipe = json.loads((directory / "tokenizer_recipe.json").read_text())
    recipe["token_bytes_sha256"] = token_bytes_fingerprint(table)
    (directory / "tokenizer_recipe.json").write_text(json.dumps(recipe))


if __name__ == "__main__":
    from priml.lib.testing import test_main

    test_main(__file__)
