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

from collections.abc import Sized
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, TypedDict, cast
from unittest.mock import patch


if TYPE_CHECKING:
    from collections.abc import Iterator
    from os import PathLike

import json
import logging
import math
import pickle
import queue
import re
import threading

from pyarrow import parquet

import numpy as np
import pyarrow as pa
import pytest
import tiktoken
import torch

from priml.baselines.nanochat import data
from priml.baselines.nanochat.data import (
    NanoChatData,
    ReferenceEvaluation,
    Tokenizer,
    token_bytes_fingerprint,
)
from priml.lib.codec import ReadError, from_plain, loads
from priml.metrics.bits_per_byte import BitsPerByte


SEQ: Final = 16
BOS: Final = "<|reserved_0|>"
RESERVED: Final = tuple(f"<|reserved_{index}|>" for index in range(16))
VOCAB: Final = 256 + len(RESERVED)


class _ReferenceArchive(TypedDict):
    inputs: np.ndarray | str | int
    targets: np.ndarray | str | int
    score_mask: np.ndarray | str | int
    token_bytes: np.ndarray | str | int
    reference_bytes: np.ndarray | str | int
    literal_bytes: np.ndarray | str | int
    batch_size: np.ndarray | str | int
    vocab_size: np.ndarray | str | int
    bos_token_id: np.ndarray | str | int
    protocol: np.ndarray | str | int


def _reference_archive(values: dict[str, object]) -> _ReferenceArchive:
    """Narrow values into the fixed reference-archive schema."""

    def field(name: str) -> np.ndarray | str | int:
        value = values[name]
        if isinstance(value, (np.ndarray, str, int)):
            return value
        raise TypeError(f"Unsupported archive value: {type(value).__name__}.")

    return {
        "inputs": field("inputs"),
        "targets": field("targets"),
        "score_mask": field("score_mask"),
        "token_bytes": field("token_bytes"),
        "reference_bytes": field("reference_bytes"),
        "literal_bytes": field("literal_bytes"),
        "batch_size": field("batch_size"),
        "vocab_size": field("vocab_size"),
        "bos_token_id": field("bos_token_id"),
        "protocol": field("protocol"),
    }


def _save_reference_archive(path: Path, archive: _ReferenceArchive) -> None:
    np.savez(
        path,
        inputs=archive["inputs"],
        targets=archive["targets"],
        score_mask=archive["score_mask"],
        token_bytes=archive["token_bytes"],
        reference_bytes=archive["reference_bytes"],
        literal_bytes=archive["literal_bytes"],
        batch_size=archive["batch_size"],
        vocab_size=archive["vocab_size"],
        bos_token_id=archive["bos_token_id"],
        protocol=archive["protocol"],
    )


# Byte-level so a document's token count is its byte count, which is what lets a test
# say which document the packer should have chosen.
def _write_shard(root: Path, index: int, documents: list[str]) -> None:
    """Write one parquet shard holding the given documents, in order."""
    parquet.write_table(
        pa.table({"text": documents}),
        root / f"shard_{index:05d}.parquet",
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Return a two-shard corpus of documents whose lengths are distinguishable."""
    ranks = {bytes([value]): value for value in range(256)}
    encoding = tiktoken.Encoding(
        name="test",
        pat_str=r".",
        mergeable_ranks=ranks,
        special_tokens={
            name: len(ranks) + index
            for index, name in enumerate(
                RESERVED,
            )
        },
    )
    directory = tmp_path / "tokenizer"
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


def _object_dict(value: object) -> dict[str, object]:
    """Narrow a JSON object to a mutable string-keyed dictionary."""
    return from_plain(value, dict[str, object])


class _NumpyLoadSpy:
    """Return a typed fixture while recording NumPy load options."""

    def __init__(self, result: np.ndarray | np.lib.npyio.NpzFile) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        file: str | PathLike[str] | PathLike[bytes] | bytes,
        *,
        mmap_mode: Literal["r+", "r", "w+", "c"] | None = None,
        allow_pickle: bool = False,
    ) -> np.ndarray | np.lib.npyio.NpzFile:
        del file
        kwargs: dict[str, object] = {"allow_pickle": allow_pickle}
        if mmap_mode is not None:
            kwargs["mmap_mode"] = mmap_mode
        self.calls.append(kwargs)
        return self.result


def _patch_row_pairs(
    monkeypatch: pytest.MonkeyPatch,
    pairs: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    def row_pairs(
        stream: data.PackedTokenStream,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        del stream
        return iter(pairs)

    monkeypatch.setattr(data.PackedTokenStream, "_row_pairs", row_pairs)


def _write_reference_archive(path: Path) -> None:
    np.savez(
        path,
        protocol="karpathy-reference-bytes-v1",
        inputs=np.array([[0, 1, 2], [1, 0, 1]], dtype=np.int64),
        targets=np.array([[1, 2, 0], [2, 1, 0]], dtype=np.int64),
        score_mask=np.array([[True, False, True], [False, True, True]]),
        token_bytes=np.array([0, 1, 0], dtype=np.int64),
        reference_bytes=np.array([2, 2], dtype=np.int64),
        literal_bytes=np.array([2, 2], dtype=np.int64),
        batch_size=2,
        vocab_size=3,
        bos_token_id=2,
    )


def test_targets_are_the_inputs_shifted_by_one(corpus: Path) -> None:
    """The whole training signal: position i predicts position i + 1."""
    batch = next(iter(_data(corpus).train_dataloader()))
    assert batch["media"].shape == (2, SEQ)
    assert batch["label"].shape == (2, SEQ)
    assert batch["media"].dtype == batch["label"].dtype == torch.long
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
    # Documents are 'a', 'bb', ... 'hhhhhhhh'; each carries a BOS marker the decode
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
    # every document contributes its own byte and a BOS marker.
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


def test_explicit_training_shards_keep_validation_held_out(corpus: Path) -> None:
    _write_shard(corpus, 2, ["z"] * 128)
    data = _data(corpus, train_shard_indices=(2,))
    batch = next(iter(data.train_dataloader()))["media"]
    assert set(batch.flatten().tolist()) == {data.tokenizer.bos_token_id, ord("z")}
    assert data.val_paths == [corpus / "shard_00001.parquet"]
    with pytest.raises(
        ValueError,
        match=r"^The validation shard must be excluded from training\.$",
    ):
        _data(corpus, train_shard_indices=(0, 1))
    with pytest.raises(
        ValueError,
        match=r"^train_shard_indices names a shard twice\.$",
    ):
        _data(corpus, train_shard_indices=(0, 0))


def test_training_buffer_override_does_not_change_validation_rows(corpus: Path) -> None:
    _write_shard(corpus, 0, ["a"] * 128 + ["bbbbbbbb"] * 172)
    parent = _data(corpus, buffer_size=256)
    changed = _data(corpus, buffer_size=256, train_buffer_size=8)
    assert next(iter(parent.train_dataloader()))["media"][0, 1].item() == ord("b")
    assert next(iter(changed.train_dataloader()))["media"][0, 1].item() == ord("a")
    for first, second in zip(
        parent.eval_dataloader(),
        changed.eval_dataloader(),
        strict=True,
    ):
        torch.testing.assert_close(first["media"], second["media"])
        torch.testing.assert_close(first["label"], second["label"])


def test_the_training_stream_does_not_end(corpus: Path) -> None:
    """The budget ends the run, so the corpus must outlast it by wrapping.

    The fixture's shard holds far fewer tokens than this draws, so a stream
    that stopped at the end of the corpus would raise here.
    """
    stream = iter(_data(corpus).train_dataloader())
    for _ in range(20):
        assert next(stream)["media"].shape == (2, SEQ)


def test_empty_online_corpus_fails_instead_of_spinning(corpus: Path) -> None:
    """A complete unproductive corpus pass cannot fill even one token row."""
    _write_shard(corpus, 0, [])
    stream = iter(_data(corpus).train_dataloader())
    with pytest.raises(
        ValueError,
        match=r"^The online corpus contains no documents\.$",
    ):
        next(stream)


def test_empty_shard_beside_productive_shard_is_supported(corpus: Path) -> None:
    """Empty individual shards do not make a nonempty corpus invalid."""
    _write_shard(corpus, 0, [])
    _write_shard(corpus, 2, ["z"])
    data = _data(corpus, train_shard_indices=(0, 2))
    batch = next(iter(data.train_dataloader()))
    assert set(batch["label"].flatten().tolist()) == {256, ord("z")}


def test_document_batches_refill_in_order_and_wrap(corpus: Path) -> None:
    texts = [f"{index:03d}" for index in range(130)]
    _write_shard(corpus, 0, texts)
    documents = data._document_batches([corpus / "shard_00000.parquet"])
    assert next(documents) == texts[:128]
    assert next(documents) == texts[128:]
    assert next(documents) == texts[:128]
    assert next(documents) == texts[128:]


def test_document_batches_wrap_with_one_document(corpus: Path) -> None:
    _write_shard(corpus, 0, ["only"])
    documents = data._document_batches([corpus / "shard_00000.parquet"])

    assert next(documents) == ["only"]
    assert next(documents) == ["only"]


def test_the_byte_table_travels_with_the_batch(corpus: Path) -> None:
    """The score divides by it, so it must reach the metric unmediated."""
    batch = next(iter(_data(corpus).eval_dataloader()))
    assert batch["token_bytes"].shape == (VOCAB,)
    # Reserved tokens carry no bytes, which is what keeps document boundaries
    # out of the denominator.
    assert (
        int(
            batch["token_bytes"][-len(RESERVED) :].sum(),
        )
        == 0
    )


def test_an_evaluation_extent_that_is_not_whole_batches_is_rejected(
    corpus: Path,
) -> None:
    """A remainder is a score covering fewer tokens than it names."""
    with pytest.raises(ValueError, match="eval_tokens"):
        _data(corpus, eval_tokens=3 * SEQ + 1)


def test_zero_evaluation_extent_is_rejected(corpus: Path) -> None:
    with pytest.raises(
        ValueError,
        match=r"^eval_tokens=0 must be positive and a whole number of eval batches of 32 tokens; otherwise the reported score covers a different token count than it names\.$",
    ):
        _data(corpus, eval_tokens=0)


def test_sub_batch_evaluation_extent_uses_whole_batch_diagnostic(
    corpus: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^eval_tokens=1 must be positive and a whole number of eval batches of 32 tokens; otherwise the reported score covers a different token count than it names\.$",
    ):
        _data(corpus, eval_tokens=1)


def test_init_rejects_zero_training_buffer_with_exact_message() -> None:
    config = NanoChatData.Config()
    config.batch_size = 2
    config.eval_batch_size = 2
    config.max_seq_len = 4
    config.eval_tokens = 8
    config.buffer_size = 2
    config.train_buffer_size = 0
    config.device = "cpu"

    with pytest.raises(ValueError, match=r"^train_buffer_size must be positive\.$"):
        NanoChatData(config)


def test_train_dataloader_requests_prefetch_and_training_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = NanoChatData.Config()
    config.batch_size = 2
    config.eval_batch_size = 2
    config.max_seq_len = 4
    config.eval_tokens = 8
    config.buffer_size = 5
    config.train_buffer_size = 3
    config.device = "cpu"
    dataset = object.__new__(NanoChatData)
    dataset.config = config
    dataset.train_paths = [Path("train.parquet")]
    dataset._tokenizer = None
    dataset._prepared = None
    dataset.token_bytes = torch.ones(4, dtype=torch.int32)
    dataset.batch_size = 2
    dataset.device = torch.device("cpu")
    calls: list[dict[str, object]] = []

    class Stream:
        pass

    def make_stream(**kwargs: object) -> Stream:
        calls.append(kwargs)
        return Stream()

    monkeypatch.setattr(data, "PackedTokenStream", make_stream)
    stream = dataset.train_dataloader()

    assert isinstance(stream, Stream)
    assert calls == [
        {
            "paths": [Path("train.parquet")],
            "tokenizer": None,
            "prepared": None,
            "token_bytes": dataset.token_bytes,
            "batch_size": 2,
            "max_seq_len": 4,
            "buffer_size": 3,
            "device": torch.device("cpu"),
            "max_batches": None,
            "prefetch": True,
        },
    ]


@pytest.mark.parametrize("vocab_size", [1, VOCAB // 2])
def test_a_vocabulary_disagreeing_with_the_model_is_rejected_at_load(
    corpus: Path,
    vocab_size: int,
) -> None:
    """Otherwise the mismatch surfaces as an out-of-range embedding index."""
    with pytest.raises(ValueError, match="vocab_size"):
        _data(corpus, vocab_size=vocab_size)


def test_no_train_shards_requires_explicit_shards(corpus: Path) -> None:
    with pytest.raises(
        ValueError,
        match=r"^num_train_shards must be positive without explicit shards\.$",
    ):
        _data(corpus, num_train_shards=0)


def test_a_matching_vocabulary_loads(corpus: Path) -> None:
    """The check must accept the agreeing case, or it blocks every real run."""
    batch = next(iter(_data(corpus, vocab_size=VOCAB).train_dataloader()))
    assert batch["media"].shape == (2, SEQ)


def test_zero_vocabulary_sentinel_skips_online_check(corpus: Path) -> None:
    assert _data(corpus, vocab_size=0).tokenizer.vocab_size == VOCAB


def test_online_token_bytes_use_int32_on_the_configured_device(corpus: Path) -> None:
    path = corpus / "tokenizer" / "token_bytes.npy"
    lengths = _load_array(path).astype(np.int64)
    _refingerprint(corpus / "tokenizer", lengths)

    token_bytes = _data(corpus, device="meta").token_bytes

    assert token_bytes.dtype == torch.int32
    assert token_bytes.device.type == "meta"


def test_loading_online_data_logs_the_shard_and_vocabulary(
    corpus: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger=data.__name__):
        _data(corpus, train_shard_indices=(0,), num_train_shards=3)
    assert caplog.messages == [f"nanochat: 1 train shards, val shard 1, vocab {VOCAB}"]


def test_every_dataset_path_resolves_beneath_base_dir() -> None:
    """A path a sibling rebases and this one does not escapes the resource root."""
    config = NanoChatData.Config()
    config.base_dir = "/root"
    config.tokenizer_dir = "/datasets/vocab"
    config.reference_evaluation = ReferenceEvaluation.Config()
    final = config.copy_tree().finalize()
    assert final.tokenizer_dir == Path("/root/datasets/vocab")
    assert final.working_dir == Path("/root/datasets/nanochat")
    assert final.reference_evaluation is not None
    assert final.reference_evaluation.path == Path(
        "/root/datasets/nanochat/reference-eval/unigram.npz",
    )


def test_loading_restores_the_epoch_timer(corpus: Path) -> None:
    dataset = _data(corpus)
    dataset.load_state_dict(
        {"batches": 0, "timer_epoch": {"global_count": 3, "global_sec": 1.5}},
    )
    assert dataset.timer_epoch.state_dict() == {"global_count": 3, "global_sec": 1.5}


def test_a_byte_table_that_does_not_match_its_fingerprint_is_rejected(
    corpus: Path,
) -> None:
    """The byte table IS the score's denominator, so its identity is recorded.

    Two tables of equal length silently reprice every token: a change to byte
    accounting shifts BPB by roughly a real candidate effect while every shape
    check still passes, and two runs become incomparable with nothing on disk
    to tell them apart.
    """
    lengths = _load_array(corpus / "tokenizer" / "token_bytes.npy")
    recorded = token_bytes_fingerprint(lengths)
    lengths[3] = 7  # Same shape, different accounting.
    np.save(corpus / "tokenizer" / "token_bytes.npy", lengths)
    with pytest.raises(ValueError, match=r".+") as error:
        _data(corpus)
    observed = token_bytes_fingerprint(lengths)
    assert str(error.value) == (
        f"{corpus / 'tokenizer'} records byte-table fingerprint "
        f"{recorded} "
        f"but its table hashes to {observed}; the score's denominator changed, "
        "so a number measured here is not comparable with one measured before. "
        "Re-prepare."
    )


def test_a_negative_byte_length_is_rejected(corpus: Path) -> None:
    """The metric's mask is ``lengths > 0``, so a negative length drops a token.

    It would vanish from both sums -- excluded from the measurement rather than
    rejected -- which is a quiet change to what the score covers.
    """
    lengths = _load_array(corpus / "tokenizer" / "token_bytes.npy")
    lengths[3] = -2
    _refingerprint(corpus / "tokenizer", lengths)
    with pytest.raises(ValueError, match=r".+") as error:
        _data(corpus)
    assert str(error.value) == (
        f"{corpus / 'tokenizer'} holds a negative byte length; the score's "
        "denominator counts bytes, and a negative one would silently drop its "
        "token from the measurement."
    )


def test_a_fractional_byte_table_is_rejected(corpus: Path) -> None:
    lengths = _load_array(corpus / "tokenizer" / "token_bytes.npy").astype(
        np.float64,
    )
    lengths[3] = 1.5
    np.save(corpus / "tokenizer" / "token_bytes.npy", lengths)

    with pytest.raises(ValueError, match=r".+") as error:
        _data(corpus)
    assert str(error.value) == (
        f"{corpus}/tokenizer/token_bytes.npy has dtype float64; it must hold "
        "integers, since a fractional length would silently change the score's "
        "denominator."
    )


def test_a_byte_table_with_the_wrong_vocabulary_size_is_rejected(
    corpus: Path,
) -> None:
    lengths = _load_array(corpus / "tokenizer" / "token_bytes.npy")[:-1]
    _refingerprint(corpus / "tokenizer", lengths)

    with pytest.raises(ValueError, match=r".+") as error:
        Tokenizer.from_directory(corpus / "tokenizer")
    assert str(error.value) == (
        f"{corpus / 'tokenizer'} fitted {VOCAB} tokens but its byte table holds "
        f"{VOCAB - 1} entries."
    )


def test_a_two_dimensional_byte_table_is_rejected(corpus: Path) -> None:
    """The metric indexes it as one length per id."""
    lengths = np.ones((VOCAB, 1), dtype=np.int32)
    _refingerprint(corpus / "tokenizer", lengths)
    with pytest.raises(ValueError, match=r".+") as error:
        _data(corpus)
    assert str(error.value) == (
        f"{corpus}/tokenizer/token_bytes.npy has shape {(VOCAB, 1)}; it must be "
        "one-dimensional, one byte length per token id."
    )


@pytest.mark.parametrize("field", ["bos_token", "token_bytes_sha256"])
def test_a_recipe_missing_a_required_field_is_rejected(
    corpus: Path,
    field: str,
) -> None:
    values = {"bos_token": BOS, "token_bytes_sha256": "recorded"}
    del values[field]
    recipe_path = corpus / "tokenizer" / "tokenizer_recipe.json"
    recipe_path.write_text(json.dumps(values))

    with pytest.raises(ValueError, match=r".+") as error:
        _data(corpus)
    assert str(error.value) == (
        f"{recipe_path} declares no {field!r}; it predates the current "
        "preparer, so what it holds cannot be established. "
        "Re-prepare the vocabulary."
    )


def test_a_missing_shard_is_named(corpus: Path) -> None:
    """A split short one shard must say which, not merely fail to prepare."""
    (corpus / "shard_00001.parquet").unlink()
    with pytest.raises(FileNotFoundError) as error:
        _data(corpus)
    assert str(error.value) == (
        f"{corpus} is missing ['shard_00001.parquet']; prepare the corpus with "
        "`uv --quiet run --frozen python -m "
        "priml.baselines.nanochat.scripts.prepare_data`."
    )


def test_a_missing_vocabulary_names_the_preparer(tmp_path: Path) -> None:
    _write_shard(tmp_path, 0, ["a"])
    _write_shard(tmp_path, 1, ["b"])
    with pytest.raises(FileNotFoundError) as error:
        _data(tmp_path)
    assert str(error.value) == (
        f"no prepared nanochat vocabulary at {tmp_path / 'tokenizer'}; build it "
        "with `uv --quiet run --frozen python -m "
        "priml.baselines.nanochat.scripts.prepare_data`."
    )


def test_a_missing_tokenizer_artifact_rejects_a_partial_directory(
    corpus: Path,
) -> None:
    (corpus / "tokenizer" / "tokenizer.pkl").unlink()
    with pytest.raises(FileNotFoundError) as error:
        Tokenizer.from_directory(corpus / "tokenizer")
    assert str(error.value) == (
        f"no prepared nanochat vocabulary at {corpus / 'tokenizer'}; build it "
        "with `uv --quiet run --frozen python -m "
        "priml.baselines.nanochat.scripts.prepare_data`."
    )


def test_resuming_an_advanced_stream_is_refused(corpus: Path) -> None:
    """The packer cannot be positioned, so a resume would replay the corpus.

    Silently restarting would retrain on the opening of the data while the
    schedules carried on from the checkpoint, which is a run neither the
    budget nor the recipe describes.
    """
    dataset = _data(corpus)
    with pytest.raises(ValueError, match=r".+") as error:
        dataset.load_state_dict({"batches": 12})
    assert str(error.value) == (
        "this checkpoint had served 12 batches, and the packed stream cannot be "
        "positioned without re-tokenizing the corpus up to that point; resuming "
        "would silently replay the start of the data. Start a fresh run."
    )


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
        ("train_buffer_size", 0),
        ("train_buffer_size", -1),
        ("max_seq_len", 1),
        ("num_train_shards", 0),
        ("num_train_shards", -1),
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


def test_minimum_valid_packing_sizes_are_accepted(corpus: Path) -> None:
    dataset = _data(
        corpus,
        buffer_size=1,
        train_buffer_size=1,
        max_seq_len=2,
    )
    assert next(iter(dataset.train_dataloader()))["media"].shape == (2, 2)


def test_the_tokenizer_prepends_the_document_marker(corpus: Path) -> None:
    """Packing depends on document LENGTH, and the marker is part of it."""
    tokenizer = Tokenizer.from_directory(corpus / "tokenizer")
    encoded = tokenizer.encode_batch(["ab", "c"])
    assert [row[0] for row in encoded] == [tokenizer.bos_token_id] * 2
    assert [len(row) for row in encoded] == [3, 2]
    np.testing.assert_array_equal(
        tokenizer.token_bytes,
        _load_array(corpus / "tokenizer" / "token_bytes.npy"),
    )


def test_tokenizer_forwards_the_thread_count() -> None:
    calls: list[tuple[list[str], int]] = []

    class EncodingSpy:
        n_vocab = 3

        def encode_single_token(self, token: str) -> int:
            assert token == BOS
            return 2

        def encode_ordinary_batch(
            self,
            texts: list[str],
            *,
            num_threads: int,
        ) -> list[list[int]]:
            calls.append((texts, num_threads))
            return [[0] for _ in texts]

    encoding = cast(tiktoken.Encoding, EncodingSpy())
    tokenizer = Tokenizer(
        encoding,
        bos_token=BOS,
        token_bytes=np.array([1, 1, 0]),
    )
    assert tokenizer.encode_batch(["a", "b"]) == [[2, 0], [2, 0]]
    assert calls == [(["a", "b"], 8)]


# Without this the fingerprint check fires first and the test proves only that, never
# reaching the property it means to pin.
def _refingerprint(directory: Path, table: np.ndarray) -> None:
    """Rewrite the recipe so the fingerprint matches a replaced table."""
    np.save(directory / "token_bytes.npy", table)
    recipe_path = directory / "tokenizer_recipe.json"
    recipe = _object_dict(
        from_plain(loads(recipe_path.read_text()), dict[str, object]),
    )
    recipe["token_bytes_sha256"] = token_bytes_fingerprint(table)
    recipe_path.write_text(json.dumps(recipe))


def _load_array(path: Path) -> np.ndarray:
    """Load a ``.npy`` the fixture wrote, as a mutable in-memory array."""
    array = cast(object, np.load(path))
    assert isinstance(array, np.ndarray)
    return array


@pytest.fixture
def prepared_config(tmp_path: Path) -> NanoChatData.Config:
    """Write tiny frozen arrays, including misleading obsolete evaluation rows."""
    train = tmp_path / "train"
    evaluation = tmp_path / "evaluation"
    train.mkdir()
    evaluation.mkdir()
    rows = np.array([[4, 0, 1, 2, 3], [4, 3, 2, 1, 0]], dtype=np.uint16)
    inputs = np.array([[4, 2, 0, 4], [4, 1, 2, 3]], dtype=np.uint16)
    targets = np.array([[2, 0, 4, 1], [1, 2, 3, 0]], dtype=np.uint16)
    primary = np.array([1, 2, 3, 4, 0], dtype=np.int64)
    literal = np.array([1, 1, 3, 4, 0], dtype=np.int64)
    np.save(train / "train_rows.npy", rows)
    np.save(train / "eval_x.npy", np.zeros_like(inputs))
    np.save(train / "eval_y.npy", np.zeros_like(targets))
    for name, array in (
        ("eval_x.npy", inputs),
        ("eval_y.npy", targets),
        ("token_bytes_primary.npy", primary),
        ("token_bytes_literal.npy", literal),
    ):
        np.save(evaluation / name, array)
    train_manifest = {
        "vocab_size": 5,
        "bos_id": 4,
        "train": {
            "batch_size": 1,
            "seq_len": 4,
            "total_rows": 2,
            "train_shard_indices": [0],
            "buffer_size": 8,
            "documents_per_refill": 128,
        },
    }
    eval_manifest = {
        "protocol": "standard-packed-shard7-tokensub-v1",
        "vocab_size": 5,
        "bos_token_id": 4,
        "eval_batch_size": 1,
        "max_seq_len": 4,
        "rows": 2,
        "batches": 2,
        "physical_positions": 8,
        "val_shard": 1,
        "buffer_size": 1000,
        "documents_per_refill": 128,
        "byte_tables": {
            "primary": {
                "file": "token_bytes_primary.npy",
                "total_on_eval_y": SEQ,
            },
            "literal": {
                "file": "token_bytes_literal.npy",
                "total_on_eval_y": 14,
            },
            "scored_positions": 7,
        },
    }
    train_path = train / "PREPARED_MANIFEST.json"
    eval_path = evaluation / "packed-eval-manifest.json"
    train_path.write_text(json.dumps(train_manifest))
    eval_path.write_text(json.dumps(eval_manifest))
    config = NanoChatData.Config()
    config.base_dir = "/"
    config.device = "cpu"
    config.batch_size = config.eval_batch_size = 1
    config.max_seq_len = 4
    config.vocab_size = 5
    config.eval_tokens = 8
    config.num_train_shards = 1
    config.val_shard = 1
    config.train_buffer_size = 8
    config.prepared_train_manifest = train_path
    config.prepared_eval_manifest = eval_path
    return config


def test_prepared_manifest_accepts_zero_vocab_sentinel(
    prepared_config: NanoChatData.Config,
) -> None:
    prepared_config.vocab_size = 0
    prepared = prepared_config.make()._prepared
    assert prepared is not None
    assert prepared.vocab_size == 5


def test_prepared_batches_preserve_int64_and_exact_exhaustion_error(
    prepared_config: NanoChatData.Config,
) -> None:
    prepared = prepared_config.make()._prepared
    assert prepared is not None
    batches = prepared.batches(batch_size=2, training=True)
    inputs, targets = next(batches)
    assert inputs.shape == targets.shape == (2, 4)
    assert inputs.dtype == targets.dtype == torch.int64
    with pytest.raises(
        RuntimeError,
        match=r"^PREPARED_EXHAUSTED: no wrapping or online fallback\.$",
    ):
        next(batches)
    inputs, targets = next(prepared.batches(batch_size=2, training=False))
    assert inputs.shape == targets.shape == (2, 4)
    assert inputs.dtype == targets.dtype == torch.int64


def test_prepared_datasets_have_no_online_tokenizer_or_shards(
    prepared_config: NanoChatData.Config,
) -> None:
    dataset = prepared_config.make()
    assert dataset.train_paths == []
    assert dataset.val_paths == []
    with pytest.raises(
        RuntimeError,
        match=r"^Prepared token rows have no online tokenizer\.$",
    ):
        _ = dataset.tokenizer


def test_prepared_training_preserves_order_and_refuses_to_wrap(
    prepared_config: NanoChatData.Config,
) -> None:
    data = prepared_config.make()
    stream = iter(data.train_dataloader())
    first = next(stream)
    assert first["media"].tolist() == [[4, 0, 1, 2]]
    assert first["label"].tolist() == [[0, 1, 2, 3]]
    assert next(stream)["label"].tolist() == [[3, 2, 1, 0]]
    assert data.state_dict()["batches"] == 2
    with pytest.raises(RuntimeError, match="PREPARED_EXHAUSTED"):
        next(stream)
    with pytest.raises(ValueError, match="fresh run"):
        data.load_state_dict(data.state_dict())


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_non_cuda_batches_use_the_configured_device(
    prepared_config: NanoChatData.Config,
    device: str,
) -> None:
    """Device placement does not depend on CUDA's pinned-memory capability."""
    prepared_config.device = device
    data = prepared_config.make()
    for batches in (data.train_dataloader(), data.eval_dataloader()):
        batch = next(iter(batches))
        for tensor in (batch["media"], batch["label"], batch["token_bytes"]):
            assert tensor.device.type == device
        assert batch["media"].dtype == torch.long
        assert batch["label"].dtype == torch.long
        assert batch["token_bytes"].dtype == torch.int32
        assert batch["valid_count"] == 1


@pytest.mark.parametrize("prepared", [False, True])
def test_host_packing_ignores_the_ambient_default_device(
    corpus: Path,
    prepared_config: NanoChatData.Config,
    prepared: bool,
) -> None:
    """Configured CPU batches stay on CPU even inside a different device context."""
    dataset = prepared_config.make() if prepared else _data(corpus)
    for stream in (dataset.train_dataloader(), dataset.eval_dataloader()):
        with torch.device("meta"):
            batch = next(iter(stream))
        for tensor in (batch["media"], batch["label"], batch["token_bytes"]):
            assert tensor.device.type == "cpu"


def test_prepared_evaluation_replays_packed_rows_and_primary_byte_rule(
    prepared_config: NanoChatData.Config,
) -> None:
    data = prepared_config.make()
    for _ in range(2):
        metric = BitsPerByte.Config().make()
        seen: list[list[int]] = []
        for batch in data.eval_dataloader():
            seen.extend(
                from_plain(row, list[int])
                for row in from_plain(batch["label"].tolist(), list[object])
            )
            metric.update(torch.ones_like(batch["label"], dtype=torch.float32), **batch)
        assert seen == [[2, 0, 4, 1], [1, 2, 3, 0]]
        assert metric.bytes == SEQ
        assert metric.nats == 7
        assert metric.compute()["bpb"] == pytest.approx(7 / (math.log(2) * SEQ))


@pytest.mark.parametrize(
    "name",
    ["train_rows.npy", "eval_y.npy", "token_bytes_primary.npy"],
)
def test_prepared_array_geometry_is_checked(
    prepared_config: NanoChatData.Config,
    name: str,
) -> None:
    manifest = (
        prepared_config.prepared_train_manifest
        if name == "train_rows.npy"
        else prepared_config.prepared_eval_manifest
    )
    path = Path(manifest).parent / name
    array = _load_array(path)
    np.save(path, array[1:])
    with pytest.raises(ValueError, match=r"geometry|one integer per token"):
        prepared_config.make()


def test_prepared_metadata_ignores_obsolete_checksums(
    prepared_config: NanoChatData.Config,
) -> None:
    path = Path(prepared_config.prepared_eval_manifest)
    metadata = _object_dict(from_plain(loads(path.read_text()), dict[str, object]))
    metadata["eval_x_sha256"] = "obsolete"
    metadata["eval_y_sha256"] = "obsolete"
    metadata["loaded_modules_and_assets"] = {
        "tokenizer_assets": {"tokenizer.json": {"sha256": "obsolete"}},
    }
    path.write_text(json.dumps(metadata) + " \n")
    batch = next(iter(prepared_config.make().eval_dataloader()))
    assert batch["label"].tolist() == [[2, 0, 4, 1]]


def test_prepared_geometry_cannot_silently_change_the_evaluation(
    prepared_config: NanoChatData.Config,
) -> None:
    prepared_config.eval_tokens = 4
    with pytest.raises(
        ValueError,
        match=r"^Prepared geometry or evaluation protocol differs from config\.$",
    ):
        prepared_config.make()


def test_prepared_manifest_batches_match_nonunit_eval_batch_geometry(
    prepared_config: NanoChatData.Config,
) -> None:
    train_path = Path(prepared_config.prepared_train_manifest)
    evaluation_path = Path(prepared_config.prepared_eval_manifest)
    train_manifest = _object_dict(
        from_plain(loads(train_path.read_text()), dict[str, object]),
    )
    evaluation_manifest = _object_dict(
        from_plain(loads(evaluation_path.read_text()), dict[str, object]),
    )
    train = _object_dict(train_manifest["train"])
    train["seq_len"] = 2
    train_manifest["train"] = train
    evaluation_manifest["max_seq_len"] = 2
    evaluation_manifest["eval_batch_size"] = 2
    evaluation_manifest["rows"] = 4
    evaluation_manifest["batches"] = 2
    evaluation_manifest["physical_positions"] = 8
    train_path.write_text(json.dumps(train_manifest))
    evaluation_path.write_text(json.dumps(evaluation_manifest))
    np.save(
        train_path.parent / "train_rows.npy",
        np.array([[4, 0, 1], [4, 2, 3]], dtype=np.uint16),
    )
    inputs = np.array([[4, 2], [0, 4], [4, 1], [2, 3]], dtype=np.uint16)
    targets = np.array([[2, 0], [4, 1], [1, 2], [3, 0]], dtype=np.uint16)
    eval_directory = evaluation_path.parent
    np.save(eval_directory / "eval_x.npy", inputs)
    np.save(eval_directory / "eval_y.npy", targets)
    primary = _load_array(eval_directory / "token_bytes_primary.npy").astype(
        np.int64,
    )
    literal = _load_array(eval_directory / "token_bytes_literal.npy").astype(
        np.int64,
    )
    byte_tables = _object_dict(evaluation_manifest["byte_tables"])
    primary_table = _object_dict(byte_tables["primary"])
    primary_table["total_on_eval_y"] = int(primary[targets].sum(dtype=np.int64))
    byte_tables["primary"] = primary_table
    literal_table = _object_dict(byte_tables["literal"])
    literal_table["total_on_eval_y"] = int(literal[targets].sum(dtype=np.int64))
    byte_tables["literal"] = literal_table
    byte_tables["scored_positions"] = int(np.count_nonzero(primary[targets]))
    evaluation_manifest["byte_tables"] = byte_tables
    evaluation_path.write_text(json.dumps(evaluation_manifest))
    prepared_config.max_seq_len = 2
    prepared_config.eval_batch_size = 2
    prepared_config.eval_tokens = 8
    prepared = prepared_config.make()._prepared
    assert prepared is not None
    assert prepared.eval_inputs.shape == (4, 2)


def test_prepared_batch_count_must_match_evaluation_rows(
    prepared_config: NanoChatData.Config,
) -> None:
    path = Path(prepared_config.prepared_eval_manifest)
    manifest = _object_dict(from_plain(loads(path.read_text()), dict[str, object]))
    manifest["batches"] = 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(
        ValueError,
        match=r"^Prepared array geometry does not contain whole batches\.$",
    ):
        prepared_config.make()


def test_prepared_training_rows_must_contain_whole_batches(
    prepared_config: NanoChatData.Config,
) -> None:
    train_manifest_path = Path(prepared_config.prepared_train_manifest)
    manifest = _object_dict(
        from_plain(loads(train_manifest_path.read_text()), dict[str, object]),
    )
    train = _object_dict(manifest["train"])
    train["batch_size"] = 2
    train["total_rows"] = 3
    manifest["train"] = train
    train_manifest_path.write_text(json.dumps(manifest))
    rows_path = train_manifest_path.parent / "train_rows.npy"
    rows = _load_array(rows_path)
    np.save(rows_path, np.concatenate((rows, rows[:1])))
    prepared_config.batch_size = 2
    with pytest.raises(
        ValueError,
        match=r"^Prepared array geometry does not contain whole batches\.$",
    ):
        prepared_config.make()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("train_shard_indices", (2,)),
        ("val_shard", 2),
        ("buffer_size", 9),
        ("batch_size", 2),
        ("eval_batch_size", 2),
        ("max_seq_len", 3),
        ("vocab_size", 1),
        ("vocab_size", 6),
        ("num_train_shards", 2),
        ("train_buffer_size", 9),
    ],
)
def test_prepared_inputs_cannot_misreport_their_corpus_or_packing(
    prepared_config: NanoChatData.Config,
    field: str,
    value: object,
) -> None:
    setattr(prepared_config, field, value)
    if field == "max_seq_len":
        prepared_config.eval_tokens = 6
    with pytest.raises(ValueError, match="geometry"):
        prepared_config.make()


@pytest.mark.parametrize("obsolete_metadata", [False, True])
def test_archive_replay_without_checksum_pins(
    tmp_path: Path,
    obsolete_metadata: bool,
) -> None:
    """Keep BPE tensors and batch boundaries unchanged, including masked padding."""
    path = tmp_path / "rows.npz"
    inputs = np.array([[5, 1, 2], [5, 3, 5], [5, 2, 1]], dtype=np.int64)
    targets = np.array([[1, 2, 3], [3, 5, 5], [2, 1, 5]], dtype=np.int64)
    np.savez(
        path,
        allow_pickle=False,
        inputs=inputs,
        targets=targets,
        score_mask=np.not_equal(targets, 5),
        reference_bytes=np.array([3, 1, 2]),
        literal_bytes=np.array([3, 1, 2]),
        batch_size=2,
        vocab_size=6,
        bos_token_id=5,
        protocol="karpathy-reference-bytes-v1",
        token_bytes=np.array([1, 1, 1, 1, 1, 0], dtype=np.int64),
        **(
            {
                "source_sha256": "obsolete",
                "scored_bytes_sha256": "obsolete",
                "tokenizer_sha256": "obsolete",
            }
            if obsolete_metadata
            else {}
        ),
    )
    config = ReferenceEvaluation.Config()
    config.path = path
    rows = config.make()
    batches = list(rows.batches(device="cpu", vocab_size=6))
    assert len(batches) == 2
    assert torch.equal(torch.cat([b["media"] for b in batches]), torch.tensor(inputs))
    assert torch.equal(torch.cat([b["label"] for b in batches]), torch.tensor(targets))
    assert [b.get("reference_bytes") for b in batches] == [4, 2]
    assert [b.get("literal_bytes") for b in batches] == [4, 2]
    assert [b.get("evaluation_batch") for b in batches] == [0, 1]
    score_masks = [batch.get("score_mask") for batch in batches]
    assert all(isinstance(mask, torch.Tensor) for mask in score_masks)
    assert [
        mask.tolist() for mask in score_masks if isinstance(mask, torch.Tensor)
    ] == [
        [[True, True, True], [True, False, False]],
        [[True, True, False]],
    ]
    assert [b["token_bytes"].dtype for b in batches] == [torch.int32, torch.int32]
    assert [b["valid_count"] for b in batches] == [2, 1]
    assert [b.get("evaluation_batches") for b in batches] == [2, 2]
    meta_batches = list(rows.batches(device="meta", vocab_size=6))
    assert [batch["media"].device.type for batch in meta_batches] == ["meta", "meta"]
    assert [batch["label"].device.type for batch in meta_batches] == ["meta", "meta"]
    assert [batch["token_bytes"].device.type for batch in meta_batches] == [
        "meta",
        "meta",
    ]
    score_masks = [batch.get("score_mask") for batch in meta_batches]
    assert all(isinstance(mask, torch.Tensor) for mask in score_masks)
    assert [
        mask.device.type for mask in score_masks if isinstance(mask, torch.Tensor)
    ] == ["meta", "meta"]
    with pytest.raises(ValueError, match="vocabulary"):
        list(rows.batches(device="cpu", vocab_size=7))


@pytest.mark.parametrize(
    ("field", "value", "eval_tokens"),
    [("max_seq_len", 2, 8), ("eval_batch_size", 1, 16)],
)
def test_reference_geometry_must_match_model(
    corpus: Path,
    field: str,
    value: int,
    eval_tokens: int,
) -> None:
    path = corpus / "reference.npz"
    _write_reference_archive(path)
    reference = ReferenceEvaluation.Config(path=path)

    with pytest.raises(
        ValueError,
        match=r"^Reference evaluation geometry differs from the model\.$",
    ):
        _data(
            corpus,
            **{
                field: value,
                "eval_tokens": eval_tokens,
                "reference_evaluation": reference,
            },
        )


@pytest.mark.parametrize("vocab_size", [1, VOCAB])
def test_reference_vocab_mismatch_is_rejected(
    corpus: Path,
    vocab_size: int,
) -> None:
    path = corpus / "reference.npz"
    _write_reference_archive(path)
    reference = ReferenceEvaluation.Config(path=path)

    with pytest.raises(
        ValueError,
        match=r"^Reference evaluation geometry differs from the model\.$",
    ):
        _data(
            corpus,
            max_seq_len=3,
            eval_tokens=12,
            vocab_size=vocab_size,
            reference_evaluation=reference,
        )


def test_reference_zero_vocab_is_a_sentinel(corpus: Path) -> None:
    path = corpus / "reference.npz"
    _write_reference_archive(path)
    reference = ReferenceEvaluation.Config(path=path)
    dataset = _data(
        corpus,
        max_seq_len=3,
        eval_tokens=12,
        vocab_size=0,
        reference_evaluation=reference,
    )
    assert dataset._reference is not None
    assert dataset._reference.vocab_size == 3


def test_reference_loader_leaves_training_unchanged(corpus: Path) -> None:
    data = _data(corpus)
    batches = list(data.eval_dataloader())
    # The native stream reuses its buffer, so use one cloned batch as the fixture.
    inputs = batches[-1]["media"].clone().numpy()
    targets = batches[-1]["label"].clone().numpy()
    path = corpus / "reference.npz"
    table = data.tokenizer.token_bytes.astype(np.int64)
    counts = table[targets].sum(axis=1)
    np.savez(
        path,
        inputs=inputs,
        targets=targets,
        score_mask=table[targets] > 0,
        reference_bytes=counts,
        literal_bytes=counts,
        token_bytes=table,
        batch_size=2,
        vocab_size=VOCAB,
        bos_token_id=256,
        protocol="karpathy-reference-bytes-v1",
    )
    reference = ReferenceEvaluation.Config()
    reference.path = path
    changed = _data(
        corpus,
        reference_evaluation=reference,
    )
    original_train = next(iter(data.train_dataloader()))
    changed_train = next(iter(changed.train_dataloader()))
    assert torch.equal(original_train["media"], changed_train["media"])
    replay = list(changed.eval_dataloader())
    assert len(replay) == 1
    assert torch.equal(replay[0]["media"], torch.tensor(inputs))
    assert replay[0].get("reference_bytes") == int(counts.sum())
    changed_meta = _data(
        corpus,
        device="meta",
        reference_evaluation=reference,
    )
    meta_batch = next(iter(changed_meta.eval_dataloader()))
    assert meta_batch["media"].device.type == "meta"
    assert meta_batch["label"].device.type == "meta"


def test_prepared_data_requires_both_manifests(corpus: Path) -> None:
    """Require both training and evaluation descriptions before reading rows."""
    config = NanoChatData.Config()
    config.prepared_train_manifest = corpus / "missing.json"
    config.device = "cpu"
    with pytest.raises(
        ValueError,
        match=r"^Prepared data requires both manifests\.$",
    ):
        config.make()


def test_pack_row_prefers_largest_fit_and_crops_shortest() -> None:
    row = torch.empty(5, dtype=torch.long)
    buffer = [[1, 2], [3, 4, 5], [6, 7, 8, 9, 10, 11]]
    position = data._pack_row(row, buffer, position=0)
    assert position == 3
    assert row[:3].tolist() == [3, 4, 5]
    assert buffer == [[1, 2], [6, 7, 8, 9, 10, 11]]
    position = data._pack_row(row, buffer, position=position)
    assert position == 5
    assert row.tolist() == [3, 4, 5, 1, 2]
    assert buffer == [[6, 7, 8, 9, 10, 11]]


def test_pack_row_keeps_first_of_equal_largest_fits() -> None:
    row = torch.full((4,), -1, dtype=torch.long)
    buffer = [[1, 2, 3], [4, 5, 6]]

    position = data._pack_row(row, buffer, position=0)

    assert position == 3
    assert row[:3].tolist() == [1, 2, 3]
    assert buffer == [[4, 5, 6]]


def test_pack_row_takes_a_document_that_exactly_fits() -> None:
    row = torch.empty(4, dtype=torch.int64)
    buffer = [[9, 8, 7, 6], [1, 2], [3, 4, 5]]
    position = data._pack_row(row, buffer, position=0)
    assert position == row.numel()
    assert row.tolist() == [9, 8, 7, 6]
    assert buffer == [[1, 2], [3, 4, 5]]


def test_pack_row_crops_a_shortest_document_later_in_buffer() -> None:
    row = torch.full((5,), -1, dtype=torch.long)
    buffer = [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]]

    position = data._pack_row(row, buffer, position=0)

    assert position == 5
    assert row.tolist() == [8, 9, 10, 11, 12]
    assert buffer == [[1, 2, 3, 4, 5, 6, 7]]


def test_pack_row_uses_one_token_fit_ahead_of_empty_document() -> None:
    row = torch.full((2,), -1, dtype=torch.long)
    buffer = [[], [1], [2, 3, 4]]

    position = data._pack_row(row, buffer, position=0)

    assert position == 1
    assert row.tolist() == [1, -1]
    assert buffer == [[], [2, 3, 4]]


def test_pack_row_crops_the_shortest_when_none_fits() -> None:
    row = torch.full((5,), -1, dtype=torch.long)
    buffer = [[1, 2, 3, 4, 5], [7, 8, 9, 10]]
    position = data._pack_row(row, buffer, position=1)
    assert position == 5
    assert row.tolist() == [-1, 7, 8, 9, 10]
    assert buffer == [[1, 2, 3, 4, 5]]


def test_pack_row_tensors_pin_dtype_and_row_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_tensor = torch.tensor
    calls: list[dict[str, object]] = []

    def record_tensor(
        values: list[int],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        calls.append({"dtype": dtype, "device": device})
        return original_tensor(values, dtype=dtype, device=device)

    monkeypatch.setattr(torch, "tensor", record_tensor)
    row = torch.empty(3, dtype=torch.long, device="meta")
    data._pack_row(row, [[1, 2, 3]], position=0)
    data._pack_row(row, [[1, 2, 3, 4]], position=0)
    assert calls == [
        {"dtype": torch.long, "device": torch.device("meta")},
        {"dtype": torch.long, "device": torch.device("meta")},
    ]


def test_prepared_array_is_a_read_only_memory_map(tmp_path: Path) -> None:
    path = tmp_path / "rows.npy"
    values = np.arange(6, dtype=np.uint16).reshape(2, 3)
    np.save(path, values)
    array = data._array(path, shape=(2, 3), dtype=np.dtype(np.uint16))
    assert isinstance(array, np.memmap)
    assert not array.flags.writeable
    np.testing.assert_array_equal(array, values)


def test_prepared_array_rejects_noncontiguous_storage(tmp_path: Path) -> None:
    path = tmp_path / "rows.npy"
    values = np.asfortranarray(np.arange(6, dtype=np.uint16).reshape(2, 3))
    np.save(path, values)
    with pytest.raises(ValueError, match="geometry/dtype"):
        data._array(path, shape=(2, 3), dtype=np.dtype(np.uint16))


def test_token_byte_fingerprint_uses_canonical_int64_bytes() -> None:
    small = np.array([1, 2, 3], dtype=np.int16)
    canonical = np.array([1, 2, 3], dtype=np.int64)
    assert token_bytes_fingerprint(small) == token_bytes_fingerprint(canonical)


def test_manifest_coercers_reject_missing_required_values() -> None:
    with pytest.raises(ReadError):
        data._integer(None)
    with pytest.raises(ReadError):
        data._mapping(None)


def test_prepared_byte_table_is_contiguous_int64(tmp_path: Path) -> None:
    np.save(tmp_path / "bytes.npy", np.array([1, 2, 3], dtype=np.int16))
    table = data._byte_table(
        tmp_path,
        metadata={"file": "bytes.npy"},
        vocab=3,
    )
    assert table.dtype == np.dtype(np.int64)
    assert table.flags.c_contiguous
    np.testing.assert_array_equal(table, np.array([1, 2, 3], dtype=np.int64))


def test_array_load_explicitly_disables_pickle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "array.npy"
    np.save(path, np.arange(6, dtype=np.uint16).reshape(2, 3))
    expected = data._array(path, shape=(2, 3), dtype=np.dtype(np.uint16))
    loader = _NumpyLoadSpy(expected)
    monkeypatch.setattr(np, "load", loader)
    array = data._array(path, shape=(2, 3), dtype=np.dtype(np.uint16))
    assert isinstance(array, np.memmap)
    assert loader.calls == [{"mmap_mode": "r", "allow_pickle": False}]


def test_byte_table_load_explicitly_disables_pickle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np.save(tmp_path / "bytes.npy", np.arange(3, dtype=np.int64))
    expected = data._byte_table(
        tmp_path,
        metadata={"file": "bytes.npy"},
        vocab=3,
    )
    loader = _NumpyLoadSpy(expected)
    monkeypatch.setattr(np, "load", loader)
    data._byte_table(tmp_path, metadata={"file": "bytes.npy"}, vocab=3)
    assert loader.calls == [{"allow_pickle": False}]


def test_stream_and_config_validation_errors(tmp_path: Path) -> None:
    config = NanoChatData.Config(batch_size=0)
    with pytest.raises(ValueError, match="batch_size"):
        config.make()
    config = NanoChatData.Config(eval_tokens=1, eval_batch_size=2, max_seq_len=3)
    with pytest.raises(ValueError, match="whole number"):
        config.make()
    with pytest.raises(FileNotFoundError, match="missing"):
        data._shard_paths(tmp_path / "absent", indices=[2])


def test_online_row_pairs_require_a_tokenizer() -> None:
    stream = data.PackedTokenStream(
        paths=[],
        tokenizer=None,
        token_bytes=torch.ones(3, dtype=torch.int32),
        batch_size=2,
        max_seq_len=3,
        buffer_size=2,
        device=torch.device("cpu"),
        max_batches=1,
    )
    with pytest.raises(ValueError, match=r".+") as error:
        next(stream._row_pairs())
    assert str(error.value) == "Expected self.tokenizer is not None."


def test_stream_length_distinguishes_bounded_from_wrapping() -> None:
    bounded = data.PackedTokenStream(
        paths=[],
        tokenizer=None,
        token_bytes=torch.ones(3, dtype=torch.int32),
        batch_size=2,
        max_seq_len=3,
        buffer_size=2,
        device=torch.device("cpu"),
        max_batches=4,
    )
    assert len(bounded) == 4
    unbounded = data.PackedTokenStream(
        paths=[],
        tokenizer=None,
        token_bytes=torch.ones(3, dtype=torch.int32),
        batch_size=2,
        max_seq_len=3,
        buffer_size=2,
        device=torch.device("cpu"),
        max_batches=None,
    )
    with pytest.raises(
        TypeError,
        match=r"^the training stream is unbounded and has no length\.$",
    ):
        len(unbounded)


def test_stream_prefetch_defaults_and_cuda_gate() -> None:
    default_stream = data.PackedTokenStream(
        paths=[],
        tokenizer=None,
        token_bytes=torch.arange(7),
        batch_size=2,
        max_seq_len=3,
        buffer_size=4,
        device=torch.device("cpu"),
        max_batches=2,
    )

    def make_stream(device: str, prefetch: bool = False) -> data.PackedTokenStream:
        return data.PackedTokenStream(
            paths=[],
            tokenizer=None,
            token_bytes=torch.arange(7),
            batch_size=2,
            max_seq_len=3,
            buffer_size=4,
            device=torch.device(device),
            max_batches=2,
            prefetch=prefetch,
        )

    cuda_default_stream = data.PackedTokenStream(
        paths=[],
        tokenizer=None,
        token_bytes=torch.arange(7),
        batch_size=2,
        max_seq_len=3,
        buffer_size=4,
        device=torch.device("cuda"),
        max_batches=2,
    )

    assert default_stream.prefetch is False
    assert cuda_default_stream.prefetch is False
    assert make_stream("cpu").prefetch is False
    assert make_stream("cuda").prefetch is False
    assert make_stream("cuda", prefetch=True).prefetch is True


def test_packed_transfer_pins_nonblocking_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = data.PackedTokenStream(
        paths=[],
        tokenizer=None,
        token_bytes=torch.arange(7),
        batch_size=2,
        max_seq_len=3,
        buffer_size=4,
        device=torch.device("meta"),
        max_batches=1,
    )
    _patch_row_pairs(
        monkeypatch,
        [(torch.ones((2, 3)), torch.zeros((2, 3)))],
    )
    original_copy = torch.Tensor.copy_
    copy_kwargs: list[dict[str, object]] = []

    def copy(
        tensor: torch.Tensor,
        source: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        copy_kwargs.append({"non_blocking": non_blocking})
        return original_copy(
            tensor,
            source,
            non_blocking=non_blocking,
        )

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    batch = next(iter(stream._packed()))

    assert batch["media"].device.type == "meta"
    assert copy_kwargs[-1] == {"non_blocking": False}
    assert stream.served == 1


def test_prefetch_worker_propagates_packing_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenEncoder:
        def encode_batch(
            self,
            texts: list[str],
            *,
            num_threads: int = 8,
        ) -> list[list[int]]:
            del texts, num_threads
            raise ValueError("packing failed")

    stream = data.PackedTokenStream(
        paths=[],
        tokenizer=BrokenEncoder(),
        token_bytes=torch.arange(7),
        batch_size=2,
        max_seq_len=3,
        buffer_size=4,
        device=torch.device("cpu"),
        max_batches=1,
        prefetch=True,
    )

    def document_batches(paths: list[Path]) -> Iterator[list[str]]:
        del paths
        return iter([["bad"]])

    monkeypatch.setattr(data, "_document_batches", document_batches)

    with pytest.raises(ValueError, match=f"^{re.escape('packing failed')}$"):
        next(stream._prefetched())


def test_prefetched_stream_preserves_batches_and_worker_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = data.PackedTokenStream(
        paths=[],
        tokenizer=None,
        token_bytes=torch.arange(7),
        batch_size=2,
        max_seq_len=3,
        buffer_size=4,
        device=torch.device("cpu"),
        max_batches=2,
    )
    stream.prefetch = True
    pair = (torch.arange(6).reshape(2, 3), torch.arange(6, 12).reshape(2, 3))
    _patch_row_pairs(monkeypatch, [pair, (pair[0] + 10, pair[1] + 10)])
    queue_factory = patch(
        "priml.baselines.nanochat.data.queue.Queue",
        wraps=queue.Queue,
    )
    thread_factory = patch(
        "priml.baselines.nanochat.data.threading.Thread",
        wraps=threading.Thread,
    )
    with queue_factory as queue_spy, thread_factory as thread_spy:
        batches = [
            (batch["media"].clone(), batch["label"].clone(), batch["token_bytes"])
            for batch in stream._prefetched()
        ]

    assert len(batches) == 2
    assert batches[0][0].tolist() == pair[0].tolist()
    assert batches[0][1].tolist() == pair[1].tolist()
    assert batches[0][2] is stream.token_bytes
    assert batches[1][0].tolist() == (pair[0] + 10).tolist()
    assert stream.served == 2
    queue_spy.assert_called_once_with(maxsize=1)
    assert thread_spy.call_args.kwargs["name"] == "nanochat-packer"
    assert thread_spy.call_args.kwargs["daemon"] is True


def test_prefetch_synchronizes_pinned_slot_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
    event_log: list[list[str]],
) -> None:
    original_empty = torch.empty

    def empty(
        *size: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        pin_memory: bool | None = None,
    ) -> torch.Tensor:
        del pin_memory
        return original_empty(size, dtype=dtype, device=device)

    def pins_host_memory(stream: data.PackedTokenStream) -> bool:
        del stream
        return True

    monkeypatch.setattr(
        data.PackedTokenStream,
        "_pins_host_memory",
        property(pins_host_memory),
    )
    monkeypatch.setattr(torch, "empty", empty)
    stream = data.PackedTokenStream(
        paths=[],
        tokenizer=None,
        token_bytes=torch.arange(7),
        batch_size=2,
        max_seq_len=3,
        buffer_size=4,
        device=torch.device("meta"),
        max_batches=3,
        prefetch=True,
    )
    pair = (torch.ones((2, 3)), torch.zeros((2, 3)))
    _patch_row_pairs(monkeypatch, [pair] * 3)

    assert len(list(stream)) == 3
    assert event_log == [
        ["synchronize", "record", "synchronize", "record"],
        ["synchronize", "record"],
    ]


@pytest.mark.usefixtures("event_log")
def test_serial_cuda_copy_uses_the_nonblocking_pinned_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_empty = torch.empty
    original_copy = torch.Tensor.copy_
    copy_modes: list[bool] = []

    def empty(
        *size: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        pin_memory: bool | None = None,
    ) -> torch.Tensor:
        del pin_memory
        return original_empty(size, dtype=dtype, device=device)

    def pins_host_memory(stream: data.PackedTokenStream) -> bool:
        del stream
        return True

    def copy(
        tensor: torch.Tensor,
        source: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        copy_modes.append(non_blocking)
        return original_copy(tensor, source, non_blocking=non_blocking)

    monkeypatch.setattr(
        data.PackedTokenStream,
        "_pins_host_memory",
        property(pins_host_memory),
    )
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    stream = data.PackedTokenStream(
        paths=[],
        tokenizer=None,
        token_bytes=torch.arange(7),
        batch_size=2,
        max_seq_len=3,
        buffer_size=4,
        device=torch.device("meta"),
        max_batches=1,
    )
    _patch_row_pairs(
        monkeypatch,
        [(torch.ones((2, 3)), torch.zeros((2, 3)))],
    )

    batch = next(iter(stream))

    assert batch["media"].device.type == "meta"
    assert copy_modes == [False, False, True]


class _EventSpy:
    """Record the fence calls a staging slot makes, in order."""

    def __init__(self, log: list[list[str]]) -> None:
        self.calls: list[str] = []
        log.append(self.calls)

    def synchronize(self) -> None:
        self.calls.append("synchronize")

    def record(self) -> None:
        self.calls.append("record")


@pytest.fixture
def event_log(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Replace ``torch.cuda.Event`` with spies that record into a fresh log."""
    log: list[list[str]] = []
    monkeypatch.setattr(torch.cuda, "Event", partial(_EventSpy, log))
    return log


def test_serial_pinned_stream_fences_the_slot_before_refilling_it(
    monkeypatch: pytest.MonkeyPatch,
    event_log: list[list[str]],
) -> None:
    """Eval takes the serial path, so it must wait on its copy like prefetch does."""
    original_empty = torch.empty

    def empty(
        *size: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        pin_memory: bool | None = None,
    ) -> torch.Tensor:
        del pin_memory
        return original_empty(size, dtype=dtype, device=device)

    def pins_host_memory(stream: data.PackedTokenStream) -> bool:
        del stream
        return True

    monkeypatch.setattr(
        data.PackedTokenStream,
        "_pins_host_memory",
        property(pins_host_memory),
    )
    monkeypatch.setattr(torch, "empty", empty)
    stream = data.PackedTokenStream(
        paths=[],
        tokenizer=None,
        token_bytes=torch.arange(7),
        batch_size=2,
        max_seq_len=3,
        buffer_size=4,
        device=torch.device("meta"),
        max_batches=3,
    )
    pair = (torch.ones((2, 3)), torch.zeros((2, 3)))
    _patch_row_pairs(monkeypatch, [pair] * 3)

    assert len(list(stream._packed())) == 3
    assert event_log == [["synchronize", "record"] * 3]


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_cuda_serial_stream_does_not_refill_a_slot_under_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delayed copy must still read the batch it was issued for."""
    stream = data.PackedTokenStream(
        paths=[],
        tokenizer=None,
        token_bytes=torch.arange(7),
        batch_size=2,
        max_seq_len=3,
        buffer_size=4,
        device=torch.device("cuda"),
        max_batches=2,
    )
    first = torch.arange(6).reshape(2, 3)
    _patch_row_pairs(
        monkeypatch,
        [(first, first + 6), (first + 100, first + 106)],
    )
    batches = iter(stream._packed())
    # Holds the device queue so the first copy runs only after the host has
    # moved on to stage the second batch.
    torch.cuda._sleep(1 << 30)
    media = next(batches)["media"].clone()
    next(batches)
    torch.cuda.synchronize()
    assert media.cpu().tolist() == first.tolist()


def test_unpinned_prefetched_copy_is_blocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a pinned slot is fenced, so only a pinned slot may copy asynchronously."""
    stream = data.PackedTokenStream(
        paths=[],
        tokenizer=None,
        token_bytes=torch.arange(7),
        batch_size=2,
        max_seq_len=3,
        buffer_size=4,
        device=torch.device("meta"),
        max_batches=1,
    )
    _patch_row_pairs(
        monkeypatch,
        [(torch.ones((2, 3)), torch.zeros((2, 3)))],
    )
    original_copy = torch.Tensor.copy_
    copy_kwargs: list[dict[str, object]] = []

    def copy(
        tensor: torch.Tensor,
        source: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        copy_kwargs.append({"non_blocking": non_blocking})
        return original_copy(
            tensor,
            source,
            non_blocking=non_blocking,
        )

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    stream.prefetch = True
    batches = list(stream._prefetched())

    assert len(batches) == 1
    assert copy_kwargs[-1] == {"non_blocking": False}
    assert batches[0]["media"].device.type == "meta"


def test_row_pair_buffer_pins_integer_dtype_and_cpu_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedEncoder:
        def encode_batch(
            self,
            texts: list[str],
            *,
            num_threads: int = 8,
        ) -> list[list[int]]:
            del num_threads
            return [[1, 2, 3, 4] for _ in texts]

    stream = data.PackedTokenStream(
        paths=[],
        tokenizer=FixedEncoder(),
        token_bytes=torch.arange(7),
        batch_size=2,
        max_seq_len=3,
        buffer_size=2,
        device=torch.device("cpu"),
        max_batches=1,
    )

    def document_batches(paths: list[Path]) -> Iterator[list[str]]:
        del paths
        return iter([["a", "b"]] * 2)

    monkeypatch.setattr(data, "_document_batches", document_batches)
    original_empty = torch.empty
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_empty(
        *size: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        pin_memory: bool | None = None,
    ) -> torch.Tensor:
        kwargs: dict[str, object] = {"dtype": dtype, "device": device}
        if pin_memory is not None:
            kwargs["pin_memory"] = pin_memory
        calls.append((size, kwargs))
        if pin_memory is None:
            return original_empty(size, dtype=dtype, device=device)
        return original_empty(
            size,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )

    monkeypatch.setattr(torch, "empty", record_empty)
    inputs, targets = next(stream._row_pairs())

    assert inputs.dtype == targets.dtype == torch.long
    assert inputs.device == targets.device == torch.device("cpu")
    assert calls == [
        ((2, 4), {"dtype": torch.long, "device": "cpu"}),
    ]


def test_row_pairs_refills_only_below_buffer_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedEncoder:
        def __init__(self) -> None:
            self.calls = 0

        def encode_batch(
            self,
            texts: list[str],
            *,
            num_threads: int = 8,
        ) -> list[list[int]]:
            del texts, num_threads
            self.calls += 1
            return [[1, 2, 3, 4]]

    encoder = FixedEncoder()
    stream = data.PackedTokenStream(
        paths=[],
        tokenizer=encoder,
        token_bytes=torch.arange(7),
        batch_size=2,
        max_seq_len=3,
        buffer_size=2,
        device=torch.device("cpu"),
        max_batches=1,
    )

    def document_batches(paths: list[Path]) -> Iterator[list[str]]:
        del paths
        return iter([["first", "second"]] * 3)

    monkeypatch.setattr(data, "_document_batches", document_batches)

    inputs, targets = next(stream._row_pairs())

    assert inputs.shape == (2, 3)
    assert targets.shape == (2, 3)
    assert encoder.calls == 3


def test_evaluation_length_and_stream_checkpoint_state(corpus: Path) -> None:
    dataset = _data(corpus)
    evaluation = dataset.eval_dataloader()
    assert isinstance(evaluation, Sized)
    assert len(evaluation) == 2
    stream = dataset.train_dataloader()
    assert next(iter(stream))["valid_count"] == 2
    assert dataset.state_dict() == {
        "batches": 1,
        "timer_epoch": {"global_count": 0, "global_sec": 0.0},
    }


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_cuda_prefetch_stream_matches_serial(corpus: Path) -> None:
    data = _data(corpus, device="cuda")
    stream = data.train_dataloader()
    batch = next(iter(stream))
    assert batch["media"].device.type == "cuda"
    assert batch["media"].shape == (2, SEQ)


def test_tokenizer_rejects_missing_recipe_and_negative_lengths(corpus: Path) -> None:
    recipe = corpus / "tokenizer" / "tokenizer_recipe.json"
    original = recipe.read_text()
    recipe.write_text(json.dumps({"bos_token": BOS}))
    with pytest.raises(ValueError, match="token_bytes_sha256"):
        Tokenizer.from_directory(corpus / "tokenizer")
    recipe.write_text(original)
    values = _load_array(corpus / "tokenizer" / "token_bytes.npy")
    values[0] = -1
    np.save(corpus / "tokenizer" / "token_bytes.npy", values)
    recipe.write_text(
        json.dumps(
            {"bos_token": BOS, "token_bytes_sha256": token_bytes_fingerprint(values)},
        ),
    )
    with pytest.raises(ValueError, match="negative"):
        Tokenizer.from_directory(corpus / "tokenizer")


def test_tokenizer_rejects_a_non_mapping_recipe(corpus: Path) -> None:
    recipe = corpus / "tokenizer" / "tokenizer_recipe.json"
    recipe.write_text("[]")
    with pytest.raises(TypeError):
        Tokenizer.from_directory(corpus / "tokenizer")


def test_tokenizer_rejects_a_null_bos_token(corpus: Path) -> None:
    recipe = corpus / "tokenizer" / "tokenizer_recipe.json"
    recipe.write_text(
        json.dumps(
            {
                "bos_token": None,
                "token_bytes_sha256": token_bytes_fingerprint(
                    _load_array(corpus / "tokenizer" / "token_bytes.npy"),
                ),
            },
        ),
    )
    with pytest.raises(ReadError):
        Tokenizer.from_directory(corpus / "tokenizer")


def test_prepared_array_helpers_reject_shape_and_dtype(tmp_path: Path) -> None:
    path = tmp_path / "array.npy"
    np.save(path, np.zeros((2, 3), dtype=np.uint16))
    with pytest.raises(ValueError, match="geometry/dtype"):
        data._array(path, shape=(3, 2), dtype=np.dtype(np.uint16))
    np.save(tmp_path / "float.npy", np.zeros(2, dtype=np.float32))
    with pytest.raises(
        ValueError,
        match=r"^Prepared byte table must contain one integer per token\.$",
    ):
        data._byte_table(tmp_path, metadata={"file": "float.npy"}, vocab=2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("inputs", np.array(1, dtype=np.int64)),
        ("inputs", np.zeros(3, dtype=np.int64)),
        ("inputs", np.zeros((0, 3), dtype=np.int64)),
        ("targets", np.zeros((3, 2), dtype=np.int64)),
        ("score_mask", np.zeros((3, 2), dtype=np.bool_)),
        ("inputs", np.zeros((2, 3), dtype=np.int32)),
        ("targets", np.zeros((2, 3), dtype=np.int32)),
        ("score_mask", np.zeros((2, 3), dtype=np.int8)),
        ("batch_size", 0),
        ("reference_bytes", np.ones(3, dtype=np.int64)),
        ("literal_bytes", np.ones(3, dtype=np.int64)),
        ("reference_bytes", np.ones(2, dtype=np.int32)),
        ("literal_bytes", np.ones(2, dtype=np.int32)),
        ("token_bytes", np.ones(5, dtype=np.int64)),
        ("token_bytes", np.ones(4, dtype=np.int32)),
        ("bos_token_id", -1),
        ("bos_token_id", 4),
        ("inputs", np.array([[-1, 2, 0], [2, 1, 0]], dtype=np.int64)),
        ("inputs", np.array([[1, 4, 0], [2, 1, 0]], dtype=np.int64)),
        ("targets", np.array([[3, -1, 1], [1, 2, 2]], dtype=np.int64)),
        ("targets", np.array([[3, 4, 1], [1, 2, 2]], dtype=np.int64)),
        ("score_mask", np.array([[True, False, False], [False, False, False]])),
        ("reference_bytes", np.array([-1, 2], dtype=np.int64)),
        ("literal_bytes", np.array([-1, 3], dtype=np.int64)),
        ("literal_bytes", np.zeros(2, dtype=np.int64)),
        ("reference_bytes", np.zeros(2, dtype=np.int64)),
    ],
)
def test_reference_validation_rejects_malformed_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "malformed.npz"
    archive: dict[str, object] = {
        "inputs": np.array([[1, 2, 0], [2, 1, 0]], dtype=np.int64),
        "targets": np.array([[3, 1, 1], [1, 2, 2]], dtype=np.int64),
        "score_mask": np.zeros((2, 3), dtype=np.bool_),
        "token_bytes": np.ones(4, dtype=np.int64),
        "reference_bytes": np.array([3, 4], dtype=np.int64),
        "literal_bytes": np.array([3, 4], dtype=np.int64),
        "batch_size": 2,
        "vocab_size": 4,
        "bos_token_id": 3,
        "protocol": "karpathy-reference-bytes-v1",
    }
    if not isinstance(value, (np.ndarray, str, int)):
        raise TypeError(f"Unsupported archive value: {type(value).__name__}.")
    archive[field] = value
    _save_reference_archive(path, _reference_archive(archive))
    with pytest.raises(
        ValueError,
        match=(
            r"^(Invalid reference evaluation array geometry or dtype|"
            r"Invalid reference targets, scoring mask, or byte counts)\.$"
        ),
    ):
        ReferenceEvaluation.Config(path=path).make()


@pytest.mark.parametrize(
    "shape",
    [(2,), (0, 2)],
)
def test_reference_validation_requires_nonempty_rank_two_inputs(
    tmp_path: Path,
    shape: tuple[int, ...],
) -> None:
    path = tmp_path / "invalid_shape.npz"
    inputs = np.ones(shape, dtype=np.int64)
    np.savez(
        path,
        protocol="karpathy-reference-bytes-v1",
        inputs=inputs,
        targets=inputs.copy(),
        score_mask=np.zeros(shape, dtype=np.bool_),
        token_bytes=np.ones(4, dtype=np.int64),
        reference_bytes=np.ones(len(inputs), dtype=np.int64),
        literal_bytes=np.ones(len(inputs), dtype=np.int64),
        batch_size=2,
        vocab_size=4,
        bos_token_id=3,
    )
    with pytest.raises(
        ValueError,
        match=r"^Invalid reference evaluation array geometry or dtype\.$",
    ):
        ReferenceEvaluation.Config(path=path).make()


@pytest.mark.parametrize("field", ["inputs", "targets"])
def test_reference_validation_rejects_each_token_array_dtype(
    tmp_path: Path,
    field: str,
) -> None:
    path = tmp_path / "bad_token_dtype.npz"
    archive: dict[str, object] = {
        "protocol": "karpathy-reference-bytes-v1",
        "inputs": np.array([[1, 2, 0], [2, 1, 0]], dtype=np.int64),
        "targets": np.array([[1, 2, 0], [2, 1, 0]], dtype=np.int64),
        "score_mask": np.zeros((2, 3), dtype=np.bool_),
        "token_bytes": np.array([1, 1, 1, 1], dtype=np.int64),
        "reference_bytes": np.array([3, 4], dtype=np.int64),
        "literal_bytes": np.array([3, 4], dtype=np.int64),
        "batch_size": 2,
        "vocab_size": 4,
        "bos_token_id": 3,
    }
    archive[field] = np.array([[1, 2, 0], [2, 1, 0]], dtype=np.int32)
    _save_reference_archive(path, _reference_archive(archive))

    with pytest.raises(
        ValueError,
        match=r"^Invalid reference evaluation array geometry or dtype\.$",
    ):
        ReferenceEvaluation.Config(path=path).make()


def test_reference_validation_accepts_singleton_batch_and_byte_totals(
    tmp_path: Path,
) -> None:
    path = tmp_path / "singleton.npz"
    np.savez(
        path,
        protocol="karpathy-reference-bytes-v1",
        inputs=np.array([[0, 1]], dtype=np.int64),
        targets=np.array([[1, 2]], dtype=np.int64),
        score_mask=np.array([[True, False]], dtype=np.bool_),
        token_bytes=np.array([0, 1, 0], dtype=np.int64),
        reference_bytes=np.array([1], dtype=np.int64),
        literal_bytes=np.array([1], dtype=np.int64),
        batch_size=1,
        vocab_size=3,
        bos_token_id=2,
    )
    evaluation = ReferenceEvaluation.Config(path=path).make()
    batches = list(evaluation.batches(device="meta", vocab_size=3))
    assert len(batches) == 1
    assert batches[0]["valid_count"] == 1
    assert batches[0].get("reference_bytes") == 1
    assert batches[0].get("literal_bytes") == 1
    assert batches[0].get("evaluation_batches") == 1
    score_mask = batches[0].get("score_mask")
    assert isinstance(score_mask, torch.Tensor)
    assert score_mask.device.type == "meta"


def test_reference_batches_reject_a_different_model_vocabulary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "reference.npz"
    np.savez(
        path,
        protocol="karpathy-reference-bytes-v1",
        inputs=np.array([[0, 1]], dtype=np.int64),
        targets=np.array([[1, 2]], dtype=np.int64),
        score_mask=np.array([[True, False]], dtype=np.bool_),
        token_bytes=np.array([0, 1, 0], dtype=np.int64),
        reference_bytes=np.array([1], dtype=np.int64),
        literal_bytes=np.array([1], dtype=np.int64),
        batch_size=1,
        vocab_size=3,
        bos_token_id=2,
    )
    evaluation = ReferenceEvaluation.Config(path=path).make()

    with pytest.raises(
        ValueError,
        match=r"^Reference evaluation vocabulary differs from model\.$",
    ):
        list(evaluation.batches(device="cpu", vocab_size=2))


def test_reference_validation_accepts_zero_tokens_and_byte_counts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "zero_values.npz"
    np.savez(
        path,
        inputs=np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64),
        targets=np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int64),
        score_mask=np.zeros((2, 3), dtype=np.bool_),
        token_bytes=np.array([0, 1, 2], dtype=np.int64),
        reference_bytes=np.array([0, 2], dtype=np.int64),
        literal_bytes=np.array([0, 2], dtype=np.int64),
        batch_size=2,
        vocab_size=3,
        bos_token_id=0,
        protocol="karpathy-reference-bytes-v1",
    )
    batches = list(
        ReferenceEvaluation.Config(path=path)
        .make()
        .batches(
            device="cpu",
            vocab_size=3,
        ),
    )
    assert len(batches) == 1
    assert batches[0]["media"].tolist() == [[0, 1, 2], [2, 0, 1]]
    assert batches[0].get("reference_bytes") == 2


def test_reference_loader_disables_pickle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "reference.npz"
    np.savez(
        path,
        protocol="karpathy-reference-bytes-v1",
        inputs=np.array([[0, 1]], dtype=np.int64),
        targets=np.array([[1, 2]], dtype=np.int64),
        score_mask=np.array([[True, False]], dtype=np.bool_),
        token_bytes=np.array([0, 1, 0], dtype=np.int64),
        reference_bytes=np.array([1], dtype=np.int64),
        literal_bytes=np.array([1], dtype=np.int64),
        batch_size=1,
        vocab_size=3,
        bos_token_id=2,
    )
    archive = np.lib.npyio.NpzFile(
        path.open("rb"),
        own_fid=True,
        allow_pickle=False,
    )
    loader = _NumpyLoadSpy(archive)
    monkeypatch.setattr(np, "load", loader)
    ReferenceEvaluation.Config(path=path).make()
    assert loader.calls == [{"allow_pickle": False}]


def test_reference_validation_rejects_bad_protocol(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        inputs=np.zeros((2, 3), dtype=np.int64),
        targets=np.zeros((2, 3), dtype=np.int64),
        score_mask=np.ones((2, 3), dtype=np.bool_),
        token_bytes=np.ones(4, dtype=np.int64),
        reference_bytes=np.ones(2, dtype=np.int64),
        literal_bytes=np.ones(2, dtype=np.int64),
        batch_size=2,
        vocab_size=4,
        bos_token_id=0,
        protocol="wrong",
    )
    with pytest.raises(
        ValueError,
        match=r"^Unsupported reference evaluation protocol\.$",
    ):
        ReferenceEvaluation.Config(path=path).make()


@pytest.mark.parametrize("prefetch", [False, True])
def test_host_buffer_allocations_pin_shape_dtype_and_device(
    prepared_config: NanoChatData.Config,
    monkeypatch: pytest.MonkeyPatch,
    prefetch: bool,
) -> None:
    prepared_config.device = "meta"
    dataset = prepared_config.make()
    stream = dataset.train_dataloader()
    stream.prefetch = prefetch
    stream.max_batches = 1
    original_empty = torch.empty
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_empty(
        *size: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        pin_memory: bool | None = None,
    ) -> torch.Tensor:
        kwargs: dict[str, object] = {"dtype": dtype, "device": device}
        if pin_memory is not None:
            kwargs["pin_memory"] = pin_memory
        calls.append((size, kwargs))
        if pin_memory is None:
            return original_empty(size, dtype=dtype, device=device)
        return original_empty(
            size,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )

    monkeypatch.setattr(torch, "empty", record_empty)
    batch = next(iter(stream))
    assert batch["media"].shape == (1, 4)
    expected = ((8,), {"dtype": torch.long, "device": "cpu", "pin_memory": False})
    assert calls[0] == expected
    if prefetch:
        assert calls[1] == expected
        assert calls[2] == ((8,), {"dtype": torch.long, "device": torch.device("meta")})
    else:
        assert calls[1] == ((8,), {"dtype": torch.long, "device": torch.device("meta")})


def test_cpu_prefetch_worker_path_replays_prepared_rows(
    prepared_config: NanoChatData.Config,
) -> None:
    dataset = prepared_config.make()
    stream = dataset.train_dataloader()
    stream.prefetch = True
    stream.max_batches = 2
    batches = [
        (
            batch["media"].clone().tolist(),
            batch["label"].clone().tolist(),
            batch["valid_count"],
        )
        for batch in stream
    ]
    assert batches == [
        ([[4, 2, 0, 4]], [[2, 0, 4, 1]], 1),
        ([[4, 1, 2, 3]], [[1, 2, 3, 0]], 1),
    ]
    assert dataset.state_dict()["batches"] == 2


@pytest.mark.parametrize(
    ("array_name", "index"),
    [("eval_x.npy", (1, 0)), ("eval_y.npy", (1, 0))],
)
def test_prepared_evaluation_rejects_out_of_vocabulary_tokens(
    prepared_config: NanoChatData.Config,
    array_name: str,
    index: tuple[int, int],
) -> None:
    path = Path(prepared_config.prepared_eval_manifest).parent / array_name
    values = _load_array(path)
    values[index] = 5
    np.save(path, values)
    with pytest.raises(
        ValueError,
        match=r"^Prepared evaluation contains an out-of-vocabulary token\.$",
    ):
        prepared_config.make()


def test_prepared_training_rejects_out_of_vocabulary_tokens(
    prepared_config: NanoChatData.Config,
) -> None:
    path = Path(prepared_config.prepared_train_manifest).parent / "train_rows.npy"
    values = _load_array(path)
    values[1, 2] = 5
    np.save(path, values)
    with pytest.raises(
        ValueError,
        match=r"^Prepared training contains an out-of-vocabulary token\.$",
    ):
        prepared_config.make()


@pytest.mark.parametrize(
    "corruption",
    ["mask_mismatch", "special_nonzero", "ordinary_zero", "literal_negative"],
)
def test_prepared_byte_table_invariants_are_checked_independently(
    prepared_config: NanoChatData.Config,
    corruption: str,
) -> None:
    directory = Path(prepared_config.prepared_eval_manifest).parent
    manifest_path = Path(prepared_config.prepared_eval_manifest)
    manifest = _object_dict(
        from_plain(loads(manifest_path.read_text()), dict[str, object]),
    )
    primary_path = directory / "token_bytes_primary.npy"
    literal_path = directory / "token_bytes_literal.npy"
    primary = _load_array(primary_path)
    literal = _load_array(literal_path)
    if corruption == "mask_mismatch":
        literal[4] = 1
    elif corruption == "special_nonzero":
        primary[4] = literal[4] = 1
    elif corruption == "ordinary_zero":
        primary[0] = literal[0] = 0
    else:
        literal[4] = -1
    np.save(primary_path, primary)
    np.save(literal_path, literal)
    targets = _load_array(directory / "eval_y.npy")
    byte_tables = _object_dict(manifest["byte_tables"])
    primary_table = _object_dict(byte_tables["primary"])
    primary_table["total_on_eval_y"] = int(primary[targets].sum(dtype=np.int64))
    byte_tables["primary"] = primary_table
    literal_table = _object_dict(byte_tables["literal"])
    literal_table["total_on_eval_y"] = int(literal[targets].sum(dtype=np.int64))
    byte_tables["literal"] = literal_table
    byte_tables["scored_positions"] = int(np.count_nonzero(primary[targets]))
    manifest["byte_tables"] = byte_tables
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(
        ValueError,
        match=r"^Prepared byte tables disagree on the scoring mask\.$",
    ):
        prepared_config.make()


def test_prepared_byte_table_rejects_negative_literal_length(
    prepared_config: NanoChatData.Config,
) -> None:
    path = (
        Path(prepared_config.prepared_eval_manifest).parent / "token_bytes_literal.npy"
    )
    values = _load_array(path)
    values[1] = -1
    np.save(path, values)
    with pytest.raises(ValueError, match="scoring mask"):
        prepared_config.make()


def test_prepared_primary_table_requires_positive_non_bos_lengths(
    prepared_config: NanoChatData.Config,
) -> None:
    path = (
        Path(prepared_config.prepared_eval_manifest).parent / "token_bytes_primary.npy"
    )
    values = _load_array(path)
    values[0] = 0
    np.save(path, values)
    with pytest.raises(ValueError, match="scoring mask"):
        prepared_config.make()


@pytest.mark.parametrize(
    ("field", "value"),
    [("primary", 15), ("literal", 15), ("scored_positions", 6)],
)
def test_prepared_scoring_totals_are_recomputed_from_evaluation_targets(
    prepared_config: NanoChatData.Config,
    field: str,
    value: int,
) -> None:
    path = (
        Path(prepared_config.prepared_eval_manifest).parent
        / "packed-eval-manifest.json"
    )
    manifest = _object_dict(from_plain(loads(path.read_text()), dict[str, object]))
    byte_tables = _object_dict(manifest["byte_tables"])
    if field == "scored_positions":
        byte_tables[field] = value
    else:
        table = _object_dict(byte_tables[field])
        table["total_on_eval_y"] = value
        byte_tables[field] = table
    manifest["byte_tables"] = byte_tables
    path.write_text(json.dumps(manifest))
    with pytest.raises(
        ValueError,
        match=r"^Prepared evaluation byte totals or scored positions differ\.$",
    ):
        prepared_config.make()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("protocol", "other"),
        ("vocab_size", 6),
        ("bos_token_id", 3),
        ("max_seq_len", 3),
        ("physical_positions", 4),
        ("val_shard", 2),
        ("buffer_size", 9),
        ("eval_batch_size", 2),
    ],
)
def test_prepared_evaluation_manifest_must_match_its_packing(
    prepared_config: NanoChatData.Config,
    field: str,
    value: object,
) -> None:
    path = (
        Path(prepared_config.prepared_eval_manifest).parent
        / "packed-eval-manifest.json"
    )
    manifest = _object_dict(from_plain(loads(path.read_text()), dict[str, object]))
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="geometry"):
        prepared_config.make()


def test_cpu_serial_stream_skips_redundant_device_copy(
    prepared_config: NanoChatData.Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = prepared_config.make().train_dataloader()
    original_copy = torch.Tensor.copy_
    copied: list[bool] = []

    def record_copy(
        self: torch.Tensor,
        source: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        copied.append(True)
        return original_copy(
            self,
            source,
            non_blocking=non_blocking,
        )

    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    next(iter(stream))
    assert len(copied) == 2


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
