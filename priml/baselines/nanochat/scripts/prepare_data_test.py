"""Tests for nanochat data preparation.

Every case here is about the same hazard: preparation writes the vocabulary a
score is measured through, so a mismatch it accepts becomes a number nobody can
attribute. None of these downloads anything -- the shards are staged locally and
the fit runs on a few hundred characters.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Event
from typing import Final
from unittest.mock import patch

import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tarfile

from numpy import array, array_equal, int64, zeros
from pyarrow import Table, parquet

import numpy as np
import pyarrow as pa
import pytest
import tiktoken
import tokenizers

from priml.baselines.nanochat.data import (
    ReferenceEvaluation,
    token_bytes_fingerprint,
)
from priml.baselines.nanochat.experiments import exp022
from priml.baselines.nanochat.scripts.prepare_data import (
    BOS_TOKEN,
    RESERVED_TOKENS,
    CorpusPreparation,
    Preparation,
    copy_input,
    donor_unigram16k,
    encode_fragment,
    fetch_file,
    interleave_rows,
    launch_training,
    pack_row,
    prepare,
    prepare_reference_rows,
    unique_rows,
)
from priml.baselines.nanochat.scripts.prepare_tokenizer import byte_alphabet
from priml.train.checkpointing import Checkpointer


VOCAB = 300  # Above the 16 reserved tokens and the 256 byte-level merges.

_CWD: Final = Path(__file__).resolve().parent


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


def test_refitting_preserves_unowned_tokenizer_files(corpus: Path) -> None:
    """A refit owns its three artifacts, not the directory containing them."""
    _prepare(corpus)
    notes = corpus / "tokenizer" / "user-notes.txt"
    notes.write_text("Retain this file across vocabulary refits.\n")
    _prepare(corpus, tokenizer_train_chars=2_000)
    assert notes.read_text() == "Retain this file across vocabulary refits.\n"


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


@pytest.mark.parametrize("save_checkpoint", [False, True])
@pytest.mark.compute_large_fixture
def test_training_handoff_preserves_config_and_uses_priml(
    tmp_path: Path, save_checkpoint: bool
) -> None:
    config = exp022()
    config.working_dir = tmp_path / "training"
    config.seed = 1102
    expected = config.copy_tree().finalize().serialize()
    with patch("subprocess.run") as process:
        launch_training(config, save_checkpoint=save_checkpoint)
    process.assert_called_once()
    command = process.call_args.args[0]
    assert command == [
        sys.executable,
        "-m",
        "priml",
        "prepared_experiment.experiment",
    ]
    assert process.call_args.kwargs["cwd"] == config.working_dir
    assert process.call_args.kwargs["check"] is True
    namespace = runpy.run_path(str(config.working_dir / "prepared_experiment.py"))
    restored = namespace["experiment"]()
    if save_checkpoint:
        checkpoint = restored.checkpointing
        assert isinstance(checkpoint, Checkpointer.Config)
        assert checkpoint.save_every == sys.maxsize
        assert checkpoint.resume is False
        restored.checkpointing = None
    assert restored.copy_tree().finalize().serialize() == expected
    with pytest.raises(FileExistsError):
        launch_training(config)


def test_recipe_configuration_does_not_write_files(tmp_path: Path) -> None:
    config = donor_unigram16k()
    config.working_dir = tmp_path / "fresh"
    text = config.pformat(hide_default_values=False)
    assert "Preparation.Config(" in text
    assert not config.working_dir.exists()


def test_recipe_configuration_does_not_read_files() -> None:
    """The bundled recipe is ordinary configuration, not a serialized asset."""
    with patch("pathlib.Path.open", side_effect=AssertionError("Unexpected file read")):
        config = donor_unigram16k()
    assert len(config.corpus.donor_source_ids) == 490
    assert config.corpus.revision == "915333b4f8b8684f39aeaafea600fea6f43fb703"
    assert config.corpus.train_shard_indices == (*range(7), *range(8, 15))
    assert config.tokenizer.vocab_size == 16_384


@pytest.mark.parametrize(
    ("name", "corpus"), [("exp018", "raw"), ("exp019", "donor-original")]
)
def test_online_milestones_use_their_prepared_corpus(
    tmp_path: Path, name: str, corpus: str
) -> None:
    """Keep the donor-corpus transition when milestone names change."""
    config = donor_unigram16k()
    config.working_dir = tmp_path / "inputs"
    training = config.make().training_config(
        name, run_directory=tmp_path / "run", seed=42
    )
    assert training.dataset.working_dir == config.working_dir / corpus
    assert training.dataset.prepared_train_manifest == ""
    assert training.dataset.val_shard == 7


@pytest.mark.compute_large_fixture
def test_preparation_builds_and_relocates(
    tmp_path: Path,
) -> None:
    config = Preparation.Config()
    config.working_dir = tmp_path / "first"
    raw = tmp_path / "source"
    raw.mkdir()
    responses: list[io.BytesIO] = []
    for index in range(3):
        path = raw / f"shard_{index:05d}.parquet"
        parquet.write_table(
            Table.from_pydict(
                {"text": [f"{index}-{row} café 🦙" for row in range(32)]}
            ),
            path,
            row_group_size=8,
        )
        responses.append(io.BytesIO(path.read_bytes()))
    config.corpus.train_shard_indices = (0, 1)
    config.corpus.val_shard = 2
    config.corpus.donor_source_ids = ("1:3", "1:5")
    config.corpus.donor_destination_shards = 1
    config.sample.train_shard_indices = (0, 1)
    config.sample.rows_per_shard = 4
    config.tokenizer.vocab_size = 272
    config.baseline.num_train_shards = 2
    config.baseline.vocab_size = 272
    config.baseline.train_chars = 100
    config.rows.train_shard_indices = (0, 1)
    config.rows.val_shard = 2
    config.rows.train_batches = 2
    config.rows.batch_size = 2
    config.rows.eval_batches = 1
    config.rows.eval_batch_size = 2
    config.rows.max_seq_len = 32
    config.rows.train_buffer_size = 2
    config.rows.buffer_size = 2
    config.rows.documents_per_refill = 2
    with patch("urllib.request.urlopen", side_effect=responses):
        config.make().run("all")
    original = config.copy_tree().finalize().rows.make().verify()
    moved = tmp_path / "moved"
    shutil.move(str(config.working_dir), moved)
    config.working_dir = moved
    relocated = config.copy_tree().finalize().rows.make().verify()
    assert (original.train_rows == relocated.train_rows).all()
    assert (original.eval_targets == relocated.eval_targets).all()
    assert (moved / "reference-eval/bpe.npz").exists()
    assert (moved / "reference-eval/unigram.npz").exists()
    preparation = config.make()
    training = preparation.training_config(
        "exp022", run_directory=tmp_path / "training", seed=1102
    )
    original_recipe = exp022()
    assert training.step.copy_tree().finalize().serialize() == (
        original_recipe.step.copy_tree().finalize().serialize()
    )
    assert (
        training.dataset.prepared_train_manifest
        == moved / "unigram16k/prepared/train/PREPARED_MANIFEST.json"
    )
    assert training.dataset.reference_evaluation is not None
    replay = training.dataset.reference_evaluation.make()
    assert isinstance(replay, ReferenceEvaluation)
    assert replay.vocab_size == config.tokenizer.vocab_size
    assert training.seed == 1102
    assert not (tmp_path / "training").exists()
    for path in moved.rglob("*MANIFEST.json"):
        assert str(tmp_path) not in path.read_text()
        assert "sha256" not in path.read_text()
    assert not list(moved.rglob("*REPRODUCTION.json"))
    exported = tmp_path / "prepared.tar"
    preparation.dump(exported)
    with tarfile.open(exported) as archive:
        assert "preparation/donor_unigram16k.json" not in archive.getnames()
        bundled = archive.extractfile("preparation/prepare_data.py")
        assert bundled is not None
        with bundled:
            assert bundled.read() == (_CWD / "prepare_data.py").read_bytes()
    with pytest.raises(FileExistsError):
        preparation.dump(exported)


def test_input_copy_preserves_bytes_and_protects_source(tmp_path: Path) -> None:
    source = tmp_path / "original.parquet"
    destination = tmp_path / "copy.parquet"
    payload = bytes(range(256))
    source.write_bytes(payload)
    copy_input(source, destination=destination)
    assert source.read_bytes() == destination.read_bytes() == payload
    with pytest.raises(ValueError, match="protected"):
        copy_input(source, destination=source)
    assert source.read_bytes() == payload


@pytest.mark.parametrize("changed", ["repository", "revision"])
def test_corpus_fetch_rejects_changed_source(tmp_path: Path, changed: str) -> None:
    config = CorpusPreparation.Config()
    config.raw_dir = tmp_path / "raw"
    config.train_shard_indices = (0,)
    config.val_shard = 1
    config.donor_source_ids = ()
    with patch(
        "urllib.request.urlopen",
        side_effect=[io.BytesIO(b"training"), io.BytesIO(b"validation")],
    ) as download:
        config.make().fetch()
        config.make().fetch()
    assert download.call_count == 2
    if changed == "repository":
        config.repository = "another/corpus"
    else:
        config.revision = "another-revision"
    with (
        patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected fetch")),
        pytest.raises(ValueError, match=r"source.*match"),
    ):
        config.make().fetch()
    assert (config.raw_dir / "shard_00000.parquet").read_bytes() == b"training"
    assert (config.raw_dir / "shard_00001.parquet").read_bytes() == b"validation"


def test_fetch_rejects_existing_file_without_source_identity(tmp_path: Path) -> None:
    destination = tmp_path / "shard.parquet"
    destination.write_bytes(b"unknown source")
    with pytest.raises(ValueError, match=r"source.*match"):
        fetch_file("https://example.com/pinned/shard.parquet", destination=destination)
    assert destination.read_bytes() == b"unknown source"


def test_fetch_receipt_failure_leaves_download_retryable(tmp_path: Path) -> None:
    destination = tmp_path / "shard.parquet"
    url = "https://example.com/pinned/shard.parquet"
    with (
        patch("urllib.request.urlopen", return_value=io.BytesIO(b"payload")),
        patch("pathlib.Path.write_text", side_effect=OSError("Receipt write failed")),
        pytest.raises(OSError, match="Receipt write failed"),
    ):
        fetch_file(url, destination=destination)
    assert not destination.exists()
    with patch("urllib.request.urlopen", return_value=io.BytesIO(b"payload")):
        fetch_file(url, destination=destination)
    assert destination.read_bytes() == b"payload"


@pytest.mark.parametrize("same_source", [True, False])
def test_concurrent_fetch_preserves_source_identity(
    tmp_path: Path, same_source: bool
) -> None:
    """Serialize one destination before checking or publishing its source identity."""
    destination = tmp_path / "shard.parquet"
    first_url = "https://example.com/first/shard.parquet"
    second_url = (
        first_url if same_source else "https://example.com/second/shard.parquet"
    )
    downloading, overlapping, release, attempted = (Event() for _ in range(4))
    response = partial(
        _held_download,
        downloading=downloading,
        overlapping=overlapping,
        release=release,
    )
    with (
        patch("urllib.request.urlopen", side_effect=response) as download,
        ThreadPoolExecutor(max_workers=2) as workers,
    ):
        first = workers.submit(fetch_file, first_url, destination=destination)
        try:
            assert downloading.wait(timeout=5), "First download never started."
            second = workers.submit(
                _fetch_when_started,
                second_url,
                destination=destination,
                attempted=attempted,
            )
            assert attempted.wait(timeout=5), "Second caller never attempted entry."
            assert not overlapping.wait(timeout=0.05), (
                "The second caller downloaded while the first owned this destination."
            )
        finally:
            release.set()
        first.result(timeout=5)
        if same_source:
            second.result(timeout=5)
        else:
            with pytest.raises(ValueError, match=r"source.*match"):
                second.result(timeout=5)
        fetch_file(first_url, destination=destination)
        assert download.call_count == 1
    assert destination.read_bytes() == first_url.encode()
    assert (
        destination.with_name(destination.name + ".source-url").read_text() == first_url
    )


def _fetch_when_started(url: str, *, destination: Path, attempted: Event) -> None:
    attempted.set()
    fetch_file(url, destination=destination)


def _held_download(
    url: str,
    *,
    timeout: int,
    downloading: Event,
    overlapping: Event,
    release: Event,
) -> io.BytesIO:
    del timeout
    if downloading.is_set():
        overlapping.set()
    downloading.set()
    assert release.wait(timeout=5), "The test did not release its HTTP response."
    return io.BytesIO(url.encode())


def test_deduplication_retains_literal_text_and_first_identity() -> None:
    seen: dict[bytes, tuple[str, str]] = {}
    first, dropped = unique_rows(iter([("0:0", "a"), ("0:1", " a")]), seen=seen)
    assert first == [("0:0", "a"), ("0:1", " a")]
    assert dropped == []
    later, dropped = unique_rows(iter([("1:0", "a"), ("1:1", "á")]), seen=seen)
    assert later == [("1:1", "á")]
    assert dropped[0]["source_id"] == "1:0"
    assert dropped[0]["first_source_id"] == "0:0"


def test_interleaving_retains_order_and_each_donor_once() -> None:
    original = [(str(i), str(i)) for i in range(5)]
    result = interleave_rows(original, additions=[("x", "x"), ("y", "y")])
    assert [key for key, _ in result] == ["0", "x", "1", "2", "y", "3", "4"]
    assert interleave_rows([], additions=[("x", "x"), ("y", "y")]) == [
        ("x", "x"),
        ("y", "y"),
    ]


def test_largest_fit_preserves_first_tie_and_remainder() -> None:
    row = zeros(7, dtype="uint16")
    buffer = [[10, 11, 12], [20, 21, 22], [30, 31]]
    lengths = list(map(len, buffer))
    position = pack_row(row, buffer=buffer, lengths=lengths, position=0)
    assert position == 3
    assert row[:3].tolist() == [10, 11, 12]
    assert buffer == [[20, 21, 22], [30, 31]]


def test_crop_discards_shortest_document_tail() -> None:
    row = zeros(2, dtype="uint16")
    buffer = [[10, 11, 12, 13], [20, 21, 22], [30, 31, 32]]
    lengths = list(map(len, buffer))
    assert pack_row(row, buffer=buffer, lengths=lengths, position=0) == 2
    assert row.tolist() == [20, 21]
    assert buffer == [[10, 11, 12, 13], [30, 31, 32]]
    assert lengths == [4, 3]


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.cli_python_subprocess
def test_cli_prints_factory_without_preparing_inputs(
    tmp_path: Path, relative: bool
) -> None:
    """Resolve the importable Config class when launched with python -m."""
    destination = tmp_path / "inputs"
    result = subprocess.run(  # noqa: S603 -- Fixed interpreter/module and test-owned arguments.
        [
            sys.executable,
            "-m",
            "priml.baselines.nanochat.scripts.prepare_data",
            "--print-config",
            "--directory",
            destination.name if relative else str(destination),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(_CWD.parents[4])},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Preparation.Config(" in result.stdout
    assert str(destination) in result.stdout
    assert not destination.exists()


def _unigram() -> tokenizers.Tokenizer:
    backend = tokenizers.Tokenizer(
        tokenizers.models.Unigram([(piece, -6.0) for piece in byte_alphabet().values()])
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = tokenizers.decoders.ByteLevel()
    return backend


def _reference() -> tiktoken.Encoding:
    return tiktoken.Encoding(
        name="tiny",
        pat_str=r".+",
        mergeable_ranks={
            **{bytes([i]): i for i in range(256)},
            b"abcd": 256,
        },
        special_tokens={"<|reserved_0|>": 257},
    )


@pytest.mark.parametrize("raw", [b"abc", "café🙂".encode(), b"\xf0\x9f", b"a\xe2"])
def test_fragment_bytes_are_exact(raw: bytes) -> None:
    """Preserve invalid trailing bytes without Unicode replacement characters."""
    backend = _unigram()
    ids = encode_fragment(raw, tokenizer=backend)
    inverse = {char: byte for byte, char in byte_alphabet().items()}
    decoded = bytearray()
    for token in ids:
        piece = backend.id_to_token(token)
        assert piece is not None
        decoded.extend(inverse[char] for char in piece)
    assert decoded == raw
    if raw in (b"abc", "café🙂".encode()):
        assert ids == backend.encode(raw.decode(), add_special_tokens=False).ids


def test_reference_replay() -> None:
    """Retain BPE tensors exactly and score the same bytes under Unigram."""
    reference = _reference()
    inputs = array([[257, 97, 98], [257, 97, 257]], dtype=int64)
    targets = array([[97, 98, 99], [97, 257, 0xE2]], dtype=int64)
    bpe = prepare_reference_rows(
        inputs,
        targets=targets,
        reference=reference,
        tokenizer=None,
        batch_size=2,
    )
    unigram = prepare_reference_rows(
        inputs,
        targets=targets,
        reference=reference,
        tokenizer=_unigram(),
        batch_size=2,
    )
    assert array_equal(bpe["inputs"], inputs)
    assert array_equal(bpe["targets"], targets)
    inverse = {char: byte for byte, char in byte_alphabet().items()}
    backend = _unigram()
    for original, replay, mask in zip(
        targets, unigram["targets"], unigram["score_mask"], strict=True
    ):
        expected = b"".join(
            reference.decode_single_token_bytes(int(token))
            for token in original
            if token < 257
        )
        pieces = [backend.id_to_token(int(token)) for token in replay[mask]]
        assert all(piece is not None for piece in pieces)
        actual = bytes(
            inverse[char] for piece in pieces if piece is not None for char in piece
        )
        assert actual == expected
    assert int(bpe["literal_bytes"].sum()) == int(unigram["literal_bytes"].sum()) == 5
    assert (
        int(bpe["reference_bytes"].sum()) == int(unigram["reference_bytes"].sum()) == 7
    )
    assert int(unigram["score_mask"].sum()) == 5
    assert len(unigram["inputs"]) == len(inputs)
    assert (unigram["targets"][unigram["score_mask"]] < 256).all()


def test_overflow_is_rejected_without_changing_context() -> None:
    with pytest.raises(ValueError, match=r"exceeds.*context"):
        prepare_reference_rows(
            array([[257, 256, 256]], dtype=int64),
            targets=array([[256, 256, 256]], dtype=int64),
            reference=_reference(),
            tokenizer=_unigram(),
            batch_size=1,
        )


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
