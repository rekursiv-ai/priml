"""Tests for nanochat data preparation.

Every case here is about the same hazard: preparation writes the vocabulary a
score is measured through, so a mismatch it accepts becomes a number nobody can
attribute. None of these downloads anything -- the shards are staged locally and
the fit runs on a few hundred characters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Event
from types import ModuleType, SimpleNamespace
from typing import Final, TypedDict, cast
from unittest.mock import Mock, call, patch
from urllib import request

import argparse
import hashlib
import importlib
import io
import json
import logging
import os
import pickle
import pickletools
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from configgle import Fig
from numpy import array, array_equal, int64, load, savez, zeros
from numpy.lib.format import read_array
from numpy.typing import NDArray
from pyarrow import Table, parquet
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors
from torch import Tensor

import numpy as np
import pyarrow as pa
import pytest
import rustbpe
import tiktoken
import tokenizers
import torch

from priml.baselines.nanochat.data import (
    NanoChatData,
    Tokenizer,
    token_bytes_fingerprint,
)
from priml.baselines.nanochat.experiments import NgramTrainLoop, exp022
from priml.baselines.nanochat.scripts import prepare_data
from priml.baselines.nanochat.scripts.prepare_data import (
    BOS_TOKEN,
    RESERVED_TOKENS,
    BpePreparation,
    CorpusPreparation,
    Preparation,
    RowPreparation,
    _differing,
    _document_batches,
    _documents,
    _donor_texts,
    _download,
    _pad_row,
    _staged,
    _token_bytes,
    _windows,
    build_reference_eval,
    copy_input,
    default_directory,
    donor_unigram16k,
    encode_fragment,
    fetch_file,
    interleave_rows,
    launch_training,
    main,
    pack_row,
    prepare,
    prepare_reference_rows,
    unique_rows,
)
from priml.baselines.nanochat.scripts.prepare_tokenizer import (
    SPLIT_PATTERN,
    ByteLevelTokenizer,
    SamplePreparation,
    UnigramPreparation,
    byte_alphabet,
)
from priml.lib.codec import ReadError, from_plain, loads
from priml.paths import validated_output_path
from priml.train.checkpointer import Checkpointer
from priml.train.tracker import TrackerList


VOCAB: Final = 300
_CWD: Final = Path(__file__).resolve().parent


def _json_object(path: Path) -> dict[str, object]:
    return from_plain(loads(path.read_text()), dict[str, object])


def _write_shard(root: Path, index: int, documents: list[str]) -> None:
    """Write one parquet shard holding the given documents."""
    parquet.write_table(
        pa.table({"text": documents}),
        root / f"shard_{index:05d}.parquet",
    )


def _load_array(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        return read_array(stream, allow_pickle=False)


def _load_archive_array(path: Path, name: str) -> np.ndarray:
    with zipfile.ZipFile(path) as archive, archive.open(f"{name}.npy") as stream:
        return read_array(stream, allow_pickle=False)


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


class _Arguments(argparse.Namespace):
    factory: str = ""
    directory: Path = Path()
    stage: str = "all"
    num_train_shards: int | None = None
    vocab_size: int | None = None
    tokenizer_train_chars: int | None = None
    tokenizer_doc_cap: int | None = None
    print_config: bool = False
    experiment: str = ""
    seed: int = 0
    train_budget_sec: float | None = None
    save_checkpoint: bool = False
    run_directory: Path = Path()
    output: Path = Path()


def test_arguments_register_all_preparation_controls() -> None:
    parser = argparse.ArgumentParser()
    prepare_data._add_arguments(parser)
    args = _Arguments()
    parser.parse_args(
        [
            "--directory",
            "inputs",
            "--stage",
            "verify",
            "--num-train-shards",
            "3",
            "--vocab-size",
            "41",
            "--tokenizer-train-chars",
            "52",
            "--tokenizer-doc-cap",
            "17",
            "--print-config",
            "--experiment",
            "exp019",
            "--seed",
            "9",
            "--train-budget-sec",
            "2.5",
            "--save-checkpoint",
            "--run-directory",
            "run",
            "--output",
            "bundle.tar",
        ],
        namespace=args,
    )
    assert args.factory == (
        "priml.baselines.nanochat.scripts.prepare_data.donor_unigram16k"
    )
    assert args.directory == Path("inputs")
    assert args.stage == "verify"
    assert args.num_train_shards == 3
    assert args.vocab_size == 41
    assert args.tokenizer_train_chars == 52
    assert args.tokenizer_doc_cap == 17
    assert args.print_config
    assert args.experiment == "exp019"
    assert args.seed == 9
    assert args.train_budget_sec == 2.5
    assert args.save_checkpoint
    assert args.run_directory == Path("run")
    assert args.output == Path("bundle.tar")


def test_argument_defaults_stage_choices_and_help_are_exact(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser(prog="prepare-data")
    prepare_data._add_arguments(parser)
    args = parser.parse_args([])
    assert vars(args) == {
        "factory": "priml.baselines.nanochat.scripts.prepare_data.donor_unigram16k",
        "directory": Path("/opt/scratch/datasets/nanochat"),
        "stage": "all",
        "num_train_shards": None,
        "vocab_size": None,
        "tokenizer_train_chars": None,
        "tokenizer_doc_cap": None,
        "print_config": False,
        "experiment": "exp022",
        "seed": 42,
        "train_budget_sec": None,
        "save_checkpoint": False,
        "run_directory": Path("/opt/scratch/runs/nanochat-reproduction"),
        "output": Path("/opt/scratch/artifacts/nanochat/prepared-inputs.tar"),
    }

    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--stage", "invalid"])
    assert error.value.code == 2
    assert "invalid choice: 'invalid'" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert (
        "--stage {all,fetch,baseline,reference,corpus,sample,tokenizer,rows,"
        "verify,dump,train}"
    ) in help_text
    assert "factory" in help_text
    assert "--directory DIRECTORY" in help_text
    assert "--run-directory RUN_DIRECTORY" in help_text
    assert "--output OUTPUT" in help_text


def test_main_print_config_does_not_prepare_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_data",
            "--print-config",
            "--directory",
            str(tmp_path / "not-created"),
        ],
    )
    assert main() == 0
    assert "Preparation.Config(" in capsys.readouterr().out
    assert not (tmp_path / "not-created").exists()


def test_default_directory_resolves_under_the_train_loop_base() -> None:
    assert default_directory() == Path("/opt/scratch/datasets/nanochat")


def test_document_fit_caps_each_row_then_stops_at_character_budget(
    corpus: Path,
) -> None:
    shards = [corpus / "shard_00000.parquet", corpus / "shard_00001.parquet"]
    assert list(_documents(shards, max_chars=7, doc_cap=4)) == [
        "docu",
        "docu",
    ]


def test_staged_shards_preserve_requested_order_and_name_missing_inputs(
    tmp_path: Path,
) -> None:
    first = tmp_path / "shard_00000.parquet"
    second = tmp_path / "shard_00001.parquet"
    third = tmp_path / "shard_00002.parquet"
    first.touch()
    second.touch()
    third.touch()
    assert _staged(tmp_path, count=3) == [first, second, third]
    with pytest.raises(FileNotFoundError, match=r"shard_00003.parquet"):
        _staged(tmp_path, count=4)


def test_bpe_preparation_build_uses_configured_recipe(tmp_path: Path) -> None:
    config = BpePreparation.Config()
    config.raw_dir = tmp_path / "raw"
    config.num_train_shards = 3
    config.vocab_size = 41
    config.train_chars = 52
    config.doc_cap = 17
    with patch("priml.baselines.nanochat.scripts.prepare_data.prepare") as fit:
        config.make().build()
    fit.assert_called_once_with(
        config.raw_dir,
        num_train_shards=3,
        vocab_size=41,
        tokenizer_train_chars=52,
        tokenizer_doc_cap=17,
        download=False,
    )


_PINNED: Final = (
    "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/"
    "915333b4f8b8684f39aeaafea600fea6f43fb703/"
)


def test_download_fetches_missing_shards_and_retains_verified(tmp_path: Path) -> None:
    existing = tmp_path / "shard_00000.parquet"
    existing.write_bytes(b"original")
    (tmp_path / "shard_00000.parquet.source-url").write_text(
        _PINNED + "shard_00000.parquet",
        encoding="utf-8",
    )
    with patch(
        "priml.baselines.nanochat.scripts.prepare_data.request.urlopen",
        return_value=io.BytesIO(b"downloaded"),
    ) as fetch:
        paths = _download(tmp_path, count=2)
    fetch.assert_called_once_with(_PINNED + "shard_00001.parquet", timeout=120)
    assert [path.name for path in paths] == [
        "shard_00000.parquet",
        "shard_00001.parquet",
    ]
    assert existing.read_bytes() == b"original"
    assert paths[1].read_bytes() == b"downloaded"


def test_download_names_the_fix_for_a_cache_without_a_receipt(tmp_path: Path) -> None:
    legacy = tmp_path / "shard_00000.parquet"
    legacy.write_bytes(b"fetched at an unknown revision")
    with pytest.raises(
        ValueError,
        match=r"no source receipt.*Delete it to re-download",
    ):
        _download(tmp_path, count=1)
    assert legacy.read_bytes() == b"fetched at an unknown revision"


def test_donor_texts_reads_only_requested_rows(corpus: Path) -> None:
    assert _donor_texts(corpus, identities={"1:1", "0:2"}) == {
        "0:2": "document 0 gamma " * 8,
        "1:1": "document 1 beta " * 8,
    }
    with pytest.raises(ValueError, match="A donor identity is absent"):
        _donor_texts(corpus, identities={"0:99"})


def test_fit_vocabulary_records_and_reuses_exact_artifacts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    shard = tmp_path / "shard_00000.parquet"
    parquet.write_table(
        pa.table({"text": ["hello world 123", "café hello", "你好 world"]}),
        shard,
    )
    output = tmp_path / "nested" / "tokenizer"
    # `rustbpe` logs its merge progress too when the root level admits INFO. Set
    # first: the last ``set_level`` also fixes the capture handler's level.
    caplog.set_level("WARNING", logger="rustbpe")
    caplog.set_level(
        "INFO",
        logger="priml.baselines.nanochat.scripts.prepare_data",
    )
    prepare_data._fit_vocabulary(
        [shard],
        out=output,
        vocab_size=280,
        train_chars=24,
        doc_cap=7,
    )

    recipe_path = output / "tokenizer_recipe.json"
    table_path = output / "token_bytes.npy"
    tokenizer_path = output / "tokenizer.pkl"
    assert {path.name for path in output.iterdir()} == {
        "tokenizer.pkl",
        "token_bytes.npy",
        "tokenizer_recipe.json",
    }
    token_bytes = np.asarray(_load_array(table_path), dtype=np.int32)
    assert token_bytes.shape == (280,)
    assert token_bytes[-len(RESERVED_TOKENS) :].tobytes() == bytes(
        np.dtype(np.int32).itemsize * len(RESERVED_TOKENS),
    )
    recipe = _json_object(recipe_path)
    assert recipe_path.read_text() == json.dumps(recipe, sort_keys=True)
    assert recipe == {
        "vocab_size": 280,
        "train_chars": 24,
        "doc_cap": 7,
        "shards": [shard.name],
        "split_pattern": SPLIT_PATTERN,
        "bos_token": BOS_TOKEN,
        "token_bytes_sha256": token_bytes_fingerprint(token_bytes),
    }
    assert tokenizer_path.stat().st_size > 0
    pickle_strings = {
        argument
        for _, argument, _ in pickletools.genops(tokenizer_path.read_bytes())
        if isinstance(argument, str)
    }
    assert "rustbpe" in pickle_strings
    with tokenizer_path.open("rb") as stream:
        assert isinstance(pickle.load(stream), tiktoken.Encoding)
    assert caplog.messages == [
        "fitting a 280-token vocabulary",
        "nanochat vocabulary fitted: 280 tokens",
    ]
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    caplog.clear()
    prepare_data._fit_vocabulary(
        [shard],
        out=output,
        vocab_size=280,
        train_chars=24,
        doc_cap=7,
    )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    assert caplog.messages == [f"nanochat vocabulary already fitted at {output}"]
    assert not list(output.glob("*.partial"))

    caplog.clear()
    prepare_data._fit_vocabulary(
        [shard],
        out=output,
        vocab_size=280,
        train_chars=25,
        doc_cap=7,
    )
    updated = _json_object(recipe_path)
    assert updated["train_chars"] == 25
    updated_token_bytes = np.asarray(_load_array(table_path), dtype=np.int32)
    assert updated["token_bytes_sha256"] == token_bytes_fingerprint(updated_token_bytes)
    assert caplog.messages[0] == (
        f"refitting the nanochat vocabulary at {output}: "
        "recipe differs {'train_chars': (24, 25)}"
    )
    assert caplog.messages[1:] == [
        "fitting a 280-token vocabulary",
        "nanochat vocabulary fitted: 280 tokens",
    ]
    assert not list(output.glob("*.partial"))


def test_fit_vocabulary_checks_its_unicode_round_trip_probe(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shard_00000.parquet"
    shard.touch()
    output = tmp_path / "tokenizer"

    class ProbeEncoding:
        n_vocab = 17

        def __init__(self, **_kwargs: object) -> None:
            self.text = ""

        def encode_ordinary(self, text: str) -> list[int]:
            self.text = text
            return [0]

        def decode(self, tokens: list[int]) -> str:
            if tokens != [0]:
                return "a"
            return (
                self.text
                if self.text == "Hello world! Numbers: 123. Unicode: 你好"
                else "changed"
            )

    trainer = Mock()
    trainer.get_mergeable_ranks.return_value = [(b"a", 0)]
    trainer.get_pattern.return_value = ".+"
    trainer_class = Mock(return_value=trainer)
    documents = Mock(return_value=iter(["sample"]))

    with (
        patch.object(rustbpe, "Tokenizer", trainer_class),
        patch.object(tiktoken, "Encoding", side_effect=ProbeEncoding),
        patch.object(prepare_data, "_documents", documents),
        patch.object(
            prepare_data,
            "_token_bytes",
            side_effect=RuntimeError("probe passed"),
        ),
        pytest.raises(RuntimeError, match=r"^probe passed$"),
    ):
        prepare_data._fit_vocabulary(
            [shard],
            out=output,
            vocab_size=17,
            train_chars=10,
            doc_cap=5,
        )

    documents.assert_called_once_with([shard], max_chars=10, doc_cap=5)
    trainer.train_from_iterator.assert_called_once_with(
        documents.return_value,
        1,
        pattern=SPLIT_PATTERN,
    )


def test_fit_vocabulary_reports_a_failed_probe_round_trip(tmp_path: Path) -> None:
    shard = tmp_path / "shard_00000.parquet"
    shard.touch()

    class InvalidEncoding:
        n_vocab = 17

        def __init__(self, **_kwargs: object) -> None:
            pass

        def encode_ordinary(self, text: str) -> list[int]:
            del text
            return [0]

        def decode(self, tokens: list[int]) -> str:
            del tokens
            return "changed"

    trainer = Mock()
    trainer.get_mergeable_ranks.return_value = [(b"a", 0)]
    trainer.get_pattern.return_value = ".+"
    with (
        patch.object(rustbpe, "Tokenizer", return_value=trainer),
        patch.object(tiktoken, "Encoding", side_effect=InvalidEncoding),
        pytest.raises(RuntimeError) as error,
    ):
        prepare_data._fit_vocabulary(
            [shard],
            out=tmp_path / "tokenizer",
            vocab_size=17,
            train_chars=10,
            doc_cap=5,
        )

    assert str(error.value) == (
        "the fitted vocabulary does not round-trip its own probe text, so "
        "it cannot be trusted to encode the corpus."
    )


def test_fit_vocabulary_rebuilds_when_the_byte_table_is_missing(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    shard = tmp_path / "shard_00000.parquet"
    parquet.write_table(pa.table({"text": ["small corpus for refitting"]}), shard)
    output = tmp_path / "tokenizer"
    caplog.set_level(
        "INFO",
        logger="priml.baselines.nanochat.scripts.prepare_data",
    )
    prepare_data._fit_vocabulary(
        [shard],
        out=output,
        vocab_size=272,
        train_chars=20,
        doc_cap=10,
    )
    (output / "token_bytes.npy").unlink()
    caplog.clear()
    prepare_data._fit_vocabulary(
        [shard],
        out=output,
        vocab_size=272,
        train_chars=20,
        doc_cap=10,
    )
    assert (output / "token_bytes.npy").is_file()
    assert "nanochat vocabulary already fitted" not in " ".join(caplog.messages)
    assert "fitting a 272-token vocabulary" in caplog.messages
    assert "nanochat vocabulary fitted: 272 tokens" in caplog.messages


class PrepareOverrides(TypedDict, total=False):
    num_train_shards: int
    vocab_size: int
    tokenizer_train_chars: int
    tokenizer_doc_cap: int


def _prepare(corpus: Path, **overrides: int) -> Path:
    """Prepare ``corpus`` at small flags, overriding one at a time."""
    arguments: dict[str, int] = {
        "num_train_shards": 1,
        "vocab_size": VOCAB,
        "tokenizer_train_chars": 1_000,
        "tokenizer_doc_cap": 100,
    }
    arguments.update(overrides)
    return prepare(
        corpus,
        num_train_shards=arguments["num_train_shards"],
        vocab_size=arguments["vocab_size"],
        tokenizer_train_chars=arguments["tokenizer_train_chars"],
        tokenizer_doc_cap=arguments["tokenizer_doc_cap"],
        download=False,
    )


def test_document_batches_respect_shards_row_groups_and_refill_size(
    tmp_path: Path,
) -> None:
    first = tmp_path / "shard_00000.parquet"
    second = tmp_path / "shard_00002.parquet"
    for path, documents in (
        (first, ["a", "b", "c", "d", "e"]),
        (second, ["f", "g"]),
    ):
        parquet.write_table(
            pa.table({"text": documents}),
            path,
            row_group_size=3,
        )
    config = RowPreparation.Config()
    config.raw_dir = tmp_path
    config.train_shard_indices = (0, 2)
    config.documents_per_refill = 2

    assert list(_document_batches(config)) == [
        ["a", "b"],
        ["c"],
        ["d", "e"],
        ["f", "g"],
    ]


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
    table: NDArray[np.int32] = cast(
        NDArray[np.int32],
        np.load(directory / "token_bytes.npy"),
    )
    recipe = _json_object(directory / "tokenizer_recipe.json")
    assert recipe["token_bytes_sha256"] == token_bytes_fingerprint(table)


def test_reserved_tokens_carry_no_bytes(corpus: Path) -> None:
    """They are document boundaries, not text, so they leave the denominator.

    Counting them would make the score depend on how often documents end,
    which is a property of the corpus rather than of the model.
    """
    _prepare(corpus)
    table: NDArray[np.int32] = cast(
        NDArray[np.int32],
        np.load(corpus / "tokenizer" / "token_bytes.npy"),
    )
    assert int(table[-len(RESERVED_TOKENS) :].sum()) == 0
    assert int(table[: -len(RESERVED_TOKENS)].min()) > 0


def test_the_recipe_records_what_the_vocabulary_was_fitted_on(corpus: Path) -> None:
    """A vocabulary fitted on other text IS a different tokenizer.

    Recording only its size would let a stale one be reused, and every token id
    would then mean something else.
    """
    _prepare(corpus)
    recipe = _json_object(corpus / "tokenizer" / "tokenizer_recipe.json")
    assert recipe["vocab_size"] == VOCAB
    assert recipe["train_chars"] == 1_000
    assert recipe["doc_cap"] == 100
    assert recipe["shards"] == ["shard_00000.parquet"]
    assert recipe["bos_token"] == BOS_TOKEN


def test_the_validation_shard_is_excluded_from_the_fit(corpus: Path) -> None:
    """The vocabulary must not be fitted on the text it will be scored on."""
    _prepare(corpus, num_train_shards=1)
    recipe = _json_object(corpus / "tokenizer" / "tokenizer_recipe.json")
    assert "shard_00001.parquet" not in cast(list[str], recipe["shards"])


def test_a_vocabulary_fitted_under_other_flags_is_refitted(corpus: Path) -> None:
    """Reusing it would hand back a tokenizer that is not the one asked for.

    Refitted rather than refused: these artifacts are derived and this
    function is how they are derived, so a caller asking for a different
    vocabulary gets one instead of an instruction to delete a file.
    """
    _prepare(corpus)
    before = _json_object(corpus / "tokenizer" / "tokenizer_recipe.json")
    _prepare(corpus, tokenizer_train_chars=2_000)
    after = _json_object(corpus / "tokenizer" / "tokenizer_recipe.json")
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
    before: NDArray[np.int32] = cast(
        NDArray[np.int32],
        np.load(corpus / "tokenizer" / "token_bytes.npy"),
    ).copy()
    assert _prepare(corpus) == corpus
    assert np.array_equal(
        cast(NDArray[np.int32], np.load(corpus / "tokenizer" / "token_bytes.npy")),
        before,
    )


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


@pytest.mark.parametrize("save_checkpoint", [None, False, True])
def test_training_handoff_preserves_config_and_uses_priml(
    tmp_path: Path,
    save_checkpoint: bool | None,
) -> None:
    config = exp022()
    config.working_dir = tmp_path / "outer" / "training"
    config.seed = 1102
    expected = config.copy_tree().finalize().serialize()
    with patch("subprocess.run") as process:
        if save_checkpoint is None:
            launch_training(config)
        else:
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
    assert process.call_args.kwargs["env"] == {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    display = config.copy_tree()
    if save_checkpoint is True:
        display.checkpointer = Checkpointer.Config()
        display.checkpointer.save_every = sys.maxsize
        display.checkpointer.resume = False
    assert (config.working_dir / "prepared_config.txt").read_text() == (
        display.pformat(hide_default_values=False) + "\n"
    )
    namespace = runpy.run_path(str(config.working_dir / "prepared_experiment.py"))
    experiment = cast(Callable[[], NgramTrainLoop.Config], namespace["experiment"])
    restored = experiment()
    if save_checkpoint is True:
        checkpoint = cast(Checkpointer.Config, restored.checkpointer)
        assert isinstance(checkpoint, Checkpointer.Config)
        assert checkpoint.save_every == sys.maxsize
        assert checkpoint.resume is False
        restored.checkpointer = None
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
    assert isinstance(config.tokenizer, UnigramPreparation.Config)
    assert config.tokenizer.vocab_size == 16_384


@pytest.mark.parametrize(
    ("name", "corpus"),
    [("exp018", "raw"), ("exp019", "donor-original")],
)
def test_online_milestones_use_their_prepared_corpus(
    tmp_path: Path,
    name: str,
    corpus: str,
) -> None:
    """Keep the donor-corpus transition when milestone names change."""
    config = donor_unigram16k()
    config.working_dir = tmp_path / "inputs"
    training = config.make().training_config(
        name,
        run_directory=tmp_path / "run",
        seed=42,
    )
    assert training.dataset.working_dir == config.working_dir / corpus
    assert training.base_dir == "/"
    assert training.working_dir == tmp_path / "run"
    assert training.seed == 42
    assert training.dataset.tokenizer_dir == config.working_dir / "raw/tokenizer"
    assert isinstance(training.tracker, TrackerList.Config)
    assert set(training.tracker.trackers) == {"metrics"}
    assert training.dataset.prepared_train_manifest == ""
    assert training.dataset.val_shard == 7


def test_the_builtin_experiment_ladder_includes_both_endpoints(
    tmp_path: Path,
) -> None:
    preparation = donor_unigram16k().make()
    first = preparation.training_config(
        "exp004",
        run_directory=tmp_path / "first",
        seed=2,
    )
    with patch.object(RowPreparation, "verify") as verify:
        last = preparation.training_config(
            "exp023",
            run_directory=tmp_path / "last",
            seed=2,
        )
    verify.assert_called_once()
    assert first.experiment_name == "exp004"
    assert last.experiment_name == "exp023"
    assert last.dataset.prepared_train_manifest == (
        preparation.config.rows.working_dir / "train/PREPARED_MANIFEST.json"
    )
    assert last.dataset.prepared_eval_manifest == (
        preparation.config.rows.working_dir / "eval/PACKED_EVAL_MANIFEST.json"
    )
    assert last.dataset.reference_evaluation is not None
    assert last.dataset.reference_evaluation.path == (
        preparation.config.working_dir / "reference-eval/unigram.npz"
    )
    for unknown in ("exp999", "default_directory", "exp022x"):
        with pytest.raises(
            ValueError,
            match=f"Unknown NanoChat experiment: {unknown}\\.",
        ) as error:
            preparation.training_config(
                unknown,
                run_directory=tmp_path / unknown,
                seed=2,
            )
        assert str(error.value) == f"Unknown NanoChat experiment: {unknown}."


def test_training_config_rejects_factory_with_wrong_config_type(
    tmp_path: Path,
) -> None:
    class FactoryModule:
        @classmethod
        def bad(cls) -> object:
            return object()

    with (
        patch.object(
            importlib,
            "import_module",
            return_value=FactoryModule(),
        ),
        pytest.raises(TypeError) as error,
    ):
        Preparation(Preparation.Config()).training_config(
            "example.factories.bad",
            run_directory=tmp_path / "run",
            seed=42,
        )

    assert str(error.value) == "The experiment must return NgramTrainLoop.Config."


def test_a_factory_outside_the_ladder_is_named_by_its_import_path(
    tmp_path: Path,
) -> None:
    """Resolve a dotted factory, keying its corpus by the experiment it returns."""
    config = donor_unigram16k()
    config.working_dir = tmp_path / "inputs"
    training = config.make().training_config(
        "priml.baselines.nanochat.experiments.exp019",
        run_directory=tmp_path / "run",
        seed=42,
    )
    assert training.experiment_name == "exp019"
    assert training.dataset.working_dir == config.working_dir / "donor-original"


def test_a_training_budget_moves_the_schedule_and_the_stop_together(
    tmp_path: Path,
) -> None:
    config = donor_unigram16k()
    config.working_dir = tmp_path / "inputs"
    preparation = config.make()
    default = preparation.training_config(
        "exp019",
        run_directory=tmp_path / "default",
        seed=42,
    )
    budgeted = preparation.training_config(
        "exp019",
        run_directory=tmp_path / "budgeted",
        seed=42,
        train_budget_sec=300.0,
    )
    assert default.max_time == default.step.train_budget_sec == 525.0
    assert budgeted.max_time == budgeted.step.train_budget_sec == 300.0


class _ShardFitting:
    """A tokenizer recipe that reads the raw shards instead of the fitting sample."""

    class Config(Fig["_ShardFitting"]):
        """Declare the fields the preparation pushes down."""

        raw_dir: Path = Path("/unset")
        """Original shards."""

        working_dir: Path = Path("/unset")
        """Destination of the tokenizer."""

        reserved_count: int = 10
        """IDs the row encoder appends after the ordinary pieces."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def build(self) -> None:
        """Write nothing; only the wiring is under test."""


def test_the_tokenizer_slot_takes_a_shard_fitted_recipe(tmp_path: Path) -> None:
    """Another fitting algorithm drops in; its reserved IDs reach the row encoder."""
    config = donor_unigram16k()
    config.working_dir = tmp_path
    config.tokenizer = _ShardFitting.Config()
    config.tokenizer_name = "shards"
    finalized = config.copy_tree().finalize()
    assert isinstance(finalized.tokenizer, _ShardFitting.Config)
    assert finalized.tokenizer.raw_dir == tmp_path / "raw"
    assert finalized.tokenizer.working_dir == tmp_path / "shards"
    assert finalized.rows.tokenizer.path == tmp_path / "shards/tokenizer.json"
    assert finalized.rows.tokenizer.reserved_count == 10


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
                {"text": [f"{index}-{row} café 🦙" for row in range(32)]},
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
    tokenizer = config.tokenizer
    assert isinstance(tokenizer, UnigramPreparation.Config)
    tokenizer.vocab_size = 272
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
    assert np.array_equal(
        cast(NDArray[np.uint16], original.train_rows),
        cast(NDArray[np.uint16], relocated.train_rows),
    )
    assert np.array_equal(
        cast(NDArray[np.uint16], original.eval_targets),
        cast(NDArray[np.uint16], relocated.eval_targets),
    )
    assert (moved / "reference-eval/bpe.npz").exists()
    assert (moved / "reference-eval/unigram.npz").exists()
    preparation = config.make()
    training = preparation.training_config(
        "exp022",
        run_directory=tmp_path / "training",
        seed=1102,
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
    assert replay.vocab_size == tokenizer.vocab_size
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


def test_reference_archive_uses_recorded_batch_size_and_both_tokenizers(
    tmp_path: Path,
) -> None:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / "PACKED_EVAL_MANIFEST.json").write_text(
        '{"eval_batch_size": 3}',
    )
    (reference_dir / "tokenizer.pkl").write_bytes(pickle.dumps(_reference()))
    np.save(reference_dir / "eval_x.npy", array([[257, 97]], dtype=int64))
    np.save(reference_dir / "eval_y.npy", array([[97, 98]], dtype=int64))
    tokenizer_path = tmp_path / "tokenizer.json"
    _unigram().save(str(tokenizer_path))
    output = tmp_path / "output"
    arrays = {"inputs": array([[1, 2]], dtype=int64)}

    with patch(
        "priml.baselines.nanochat.scripts.prepare_data.prepare_reference_rows",
        return_value=arrays,
    ) as replay:
        build_reference_eval(
            reference_dir=reference_dir,
            unigram_path=tokenizer_path,
            reserved_count=16,
            output=output,
        )

    assert replay.call_count == 2
    assert replay.call_args_list[0].kwargs["tokenizer"] is None
    assert replay.call_args_list[0].kwargs["batch_size"] == 3
    assert replay.call_args_list[0].kwargs["reserved_count"] == 16
    assert replay.call_args_list[1].kwargs["tokenizer"] is not None
    assert replay.call_args_list[1].kwargs["batch_size"] == 3
    assert np.array_equal(
        _load_archive_array(output / "bpe.npz", "inputs"),
        np.array([[1, 2]], dtype=np.int64),
    )
    assert np.array_equal(
        _load_archive_array(output / "unigram.npz", "inputs"),
        np.array([[1, 2]], dtype=np.int64),
    )
    with pytest.raises(FileExistsError):
        build_reference_eval(
            reference_dir=reference_dir,
            unigram_path=tokenizer_path,
            reserved_count=16,
            output=output,
        )


def test_input_copy_preserves_bytes_and_protects_source(tmp_path: Path) -> None:
    source = tmp_path / "original.parquet"
    destination = tmp_path / "nested" / "deeper" / "copy.parquet"
    payload = bytes(range(256))
    source.write_bytes(payload)
    with patch(
        "priml.baselines.nanochat.scripts.prepare_data.tempfile.TemporaryDirectory",
        wraps=tempfile.TemporaryDirectory,
    ) as temporary_directory:
        copy_input(source, destination=destination)
    temporary_directory.assert_called_once_with(
        dir=destination.parent,
        prefix="nanochat-copy-",
    )
    assert source.read_bytes() == destination.read_bytes() == payload
    copy_input(source, destination=destination)
    with pytest.raises(ValueError, match="protected"):
        copy_input(source, destination=source)
    assert source.read_bytes() == payload


@pytest.mark.parametrize("changed", ["repository", "revision"])
def test_corpus_fetch_rejects_changed_source(
    tmp_path: Path,
    changed: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = CorpusPreparation.Config()
    config.raw_dir = tmp_path / "raw"
    config.train_shard_indices = (0,)
    config.val_shard = 1
    config.donor_source_ids = ()
    caplog.set_level("INFO")
    with patch(
        "urllib.request.urlopen",
        side_effect=[io.BytesIO(b"training"), io.BytesIO(b"validation")],
    ) as download:
        config.make().fetch()
        config.make().fetch()
    assert [call.args[0] for call in download.call_args_list] == [
        "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/915333b4f8b8684f39aeaafea600fea6f43fb703/shard_00000.parquet",
        "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/915333b4f8b8684f39aeaafea600fea6f43fb703/shard_00001.parquet",
    ]
    assert (
        caplog.messages
        == [
            "Fetching source shard_00000.parquet",
            "Fetching source shard_00001.parquet",
        ]
        * 2
    )
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
    with pytest.raises(ValueError, match=r"no source receipt"):
        fetch_file("https://example.com/pinned/shard.parquet", destination=destination)
    assert destination.read_bytes() == b"unknown source"


def test_fetch_receipt_failure_leaves_download_retryable(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "deeper" / "shard.parquet"
    url = "https://example.com/pinned/shard.parquet"
    with (
        patch("urllib.request.urlopen", return_value=io.BytesIO(b"payload")),
        patch("pathlib.Path.write_text", side_effect=OSError("Receipt write failed")),
        pytest.raises(OSError, match="Receipt write failed"),
    ):
        fetch_file(url, destination=destination)
    assert not destination.exists()
    assert destination.with_name(destination.name + ".lock").is_file()
    with patch("urllib.request.urlopen", return_value=io.BytesIO(b"payload")):
        fetch_file(url, destination=destination)
    assert destination.read_bytes() == b"payload"


def test_fetch_stages_beside_destination_with_named_temporary_directory(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "deeper" / "shard.parquet"
    with (
        patch("urllib.request.urlopen", return_value=io.BytesIO(b"payload")),
        patch(
            "priml.baselines.nanochat.scripts.prepare_data.tempfile.TemporaryDirectory",
            wraps=tempfile.TemporaryDirectory,
        ) as temporary_directory,
    ):
        fetch_file("https://example.com/pinned/shard.parquet", destination=destination)
    temporary_directory.assert_called_once_with(
        dir=destination.parent,
        prefix="nanochat-download-",
    )
    assert destination.read_bytes() == b"payload"


@pytest.mark.parametrize("same_source", [True, False])
def test_concurrent_fetch_preserves_source_identity(
    tmp_path: Path,
    same_source: bool,
) -> None:
    """Serialize one destination before checking or publishing its source identity."""
    destination = tmp_path / "nested" / "deeper" / "shard.parquet"
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
    assert destination.with_name(destination.name + ".lock").is_file()


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
    assert timeout == 120
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


def test_largest_fit_accepts_an_exact_row_remainder() -> None:
    row = zeros(5, dtype="uint16")
    buffer = [[10, 11, 12], [20, 21]]
    lengths = [3, 2]
    position = pack_row(row, buffer=buffer, lengths=lengths, position=2)
    assert position == 5
    assert row.tolist() == [0, 0, 10, 11, 12]
    assert buffer == [[20, 21]]
    assert lengths == [2]


def test_largest_fit_selects_a_one_token_document() -> None:
    row = zeros(3, dtype="uint16")
    buffer = [[10], [20]]
    lengths = [1, 1]
    assert pack_row(row, buffer=buffer, lengths=lengths, position=0) == 1
    assert row.tolist() == [10, 0, 0]
    assert buffer == [[20]]
    assert lengths == [1]


def test_crop_discards_shortest_document_tail() -> None:
    row = zeros(2, dtype="uint16")
    buffer = [[10, 11, 12, 13], [20, 21, 22], [30, 31, 32]]
    lengths = list(map(len, buffer))
    assert pack_row(row, buffer=buffer, lengths=lengths, position=0) == 2
    assert row.tolist() == [20, 21]
    assert buffer == [[10, 11, 12, 13], [30, 31, 32]]
    assert lengths == [4, 3]


@pytest.mark.cli_python_subprocess
def test_training_documents_read_in_a_fresh_interpreter(tmp_path: Path) -> None:
    """``--stage rows`` runs alone, so it cannot lean on an earlier stage's imports."""
    _write_shard(tmp_path, 0, ["alpha", "beta"])
    program = (
        "from pathlib import Path\n"
        "from priml.baselines.nanochat.scripts import prepare_data\n"
        "config = prepare_data.RowPreparation.Config()\n"
        f"config.raw_dir = Path({str(tmp_path)!r})\n"
        "config.train_shard_indices = (0,)\n"
        "print(list(prepare_data._document_batches(config)))\n"
    )
    result = subprocess.run(  # noqa: S603 -- Fixed interpreter and test-owned program.
        [sys.executable, "-c", program],
        env={**os.environ, "PYTHONPATH": str(_CWD.parents[4])},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[['alpha', 'beta']]"


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.cli_python_subprocess
def test_cli_prints_factory_without_preparing_inputs(
    tmp_path: Path,
    relative: bool,
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
        models.Unigram([(piece, -6.0) for piece in byte_alphabet().values()]),
    )
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
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


def test_fragment_rejects_invalid_utf8_before_the_trailing_suffix() -> None:
    with pytest.raises(
        ValueError,
        match="Reference fragment has non-terminal invalid UTF-8",
    ) as error:
        encode_fragment(b"a\xffb", tokenizer=_unigram())
    assert str(error.value) == "Reference fragment has non-terminal invalid UTF-8."


def test_fragment_does_not_add_postprocessor_special_tokens() -> None:
    backend = _unigram()
    backend.add_special_tokens(["<extra>"])
    token_id = backend.token_to_id("<extra>")
    assert token_id is not None
    backend.post_processor = processors.TemplateProcessing(
        single="<extra> $A",
        special_tokens=[("<extra>", token_id)],
    )
    assert (
        encode_fragment(b"ab", tokenizer=backend)
        == backend.encode(
            "ab",
            add_special_tokens=False,
        ).ids
    )


def test_windows_pad_when_context_holds_fewer_than_one_full_target_window() -> None:
    windows = _windows([7, 8, 9], bos=7, width=3)
    assert len(windows) == 1
    assert [item.tolist() for item in windows[0]] == [
        [7, 8, 7],
        [8, 9, 7],
        [True, True, False],
    ]


def test_windows_cover_the_partial_final_window() -> None:
    windows = _windows([7, 8, 9, 10, 11], bos=7, width=3)
    assert [item.tolist() for item in windows[0]] == [
        [7, 8, 9],
        [8, 9, 10],
        [True, True, True],
    ]
    assert [item.tolist() for item in windows[1]] == [
        [8, 9, 10],
        [9, 10, 11],
        [False, False, True],
    ]


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
        reserved_count=16,
    )
    unigram = prepare_reference_rows(
        inputs,
        targets=targets,
        reference=reference,
        tokenizer=_unigram(),
        batch_size=2,
        reserved_count=16,
    )
    assert set(bpe) == {
        "inputs",
        "targets",
        "score_mask",
        "reference_bytes",
        "literal_bytes",
        "reference_rows",
        "fragment_lengths",
        "batch_size",
        "vocab_size",
        "bos_token_id",
        "incomplete_utf8_suffixes",
        "protocol",
        "token_bytes",
    }
    assert array_equal(bpe["inputs"], inputs)
    assert array_equal(bpe["targets"], targets)
    assert cast(NDArray[np.int64], bpe["inputs"]).dtype == np.dtype(int64)
    assert cast(NDArray[np.int64], bpe["targets"]).dtype == np.dtype(int64)
    assert cast(NDArray[np.bool_], bpe["score_mask"]).dtype == np.dtype(bool)
    assert array_equal(
        cast(NDArray[np.int64], bpe["token_bytes"]),
        array(
            [len(reference.decode([i]).encode()) for i in range(257)] + [0],
            dtype=int64,
        ),
    )
    assert cast(NDArray[np.bool_], bpe["score_mask"]).tolist() == [
        [True, True, True],
        [True, False, True],
    ]
    assert cast(NDArray[np.int64], bpe["reference_rows"]).tolist() == [0, 1]
    assert cast(NDArray[np.int64], bpe["reference_bytes"]).tolist() == [3, 4]
    assert cast(NDArray[np.int64], bpe["literal_bytes"]).tolist() == [3, 2]
    assert cast(NDArray[np.int64], bpe["fragment_lengths"]).tolist() == [3, 1, 1]
    assert int(bpe["batch_size"]) == 2
    assert int(bpe["vocab_size"]) == reference.n_vocab
    assert int(bpe["bos_token_id"]) == 257
    assert int(bpe["incomplete_utf8_suffixes"]) == 1
    assert str(bpe["protocol"]) == "karpathy-reference-bytes-v1"
    inverse = {char: byte for byte, char in byte_alphabet().items()}
    backend = _unigram()
    for original, replay, mask in zip(
        cast(Iterator[NDArray[np.int64]], targets),
        cast(Iterator[NDArray[np.int64]], unigram["targets"]),
        cast(Iterator[NDArray[np.bool_]], unigram["score_mask"]),
        strict=True,
    ):
        expected = b"".join(
            reference.decode_single_token_bytes(int(token))
            for token in cast(list[int], original.tolist())
            if token < 257
        )
        pieces = [
            backend.id_to_token(int(token))
            for token in cast(list[int], replay[mask].tolist())
        ]
        assert all(piece is not None for piece in pieces)
        actual = bytes(
            inverse[char] for piece in pieces if piece is not None for char in piece
        )
        assert actual == expected
    assert (
        int(cast(NDArray[np.int64], bpe["literal_bytes"]).sum())
        == int(cast(NDArray[np.int64], unigram["literal_bytes"]).sum())
        == 5
    )
    assert (
        int(cast(NDArray[np.int64], bpe["reference_bytes"]).sum())
        == int(cast(NDArray[np.int64], unigram["reference_bytes"]).sum())
        == 7
    )
    assert int(cast(NDArray[np.bool_], unigram["score_mask"]).sum()) == 5
    assert len(unigram["inputs"]) == len(inputs)
    assert bool(
        cast(
            object,
            (
                cast(NDArray[np.int64], unigram["targets"])[
                    cast(NDArray[np.bool_], unigram["score_mask"])
                ]
                < 256
            ).all(),
        ),
    )


def test_replay_fragments_start_at_each_document_boundary() -> None:
    replay = prepare_reference_rows(
        array([[257, 97, 257]], dtype=int64),
        targets=array([[97, 257, 98]], dtype=int64),
        reference=_reference(),
        tokenizer=_unigram(),
        batch_size=1,
        reserved_count=16,
    )
    assert replay["fragment_lengths"].tolist() == [1, 1]
    assert replay["literal_bytes"].tolist() == [2]
    assert replay["reference_bytes"].tolist() == [2]


def test_reference_rows_do_not_pad_without_continuations() -> None:
    replay = prepare_reference_rows(
        array([[257, 97]], dtype=int64),
        targets=array([[97, 98]], dtype=int64),
        reference=_reference(),
        tokenizer=None,
        batch_size=2,
        reserved_count=16,
    )
    assert replay["inputs"].shape == (1, 2)
    assert replay["reference_rows"].tolist() == [0]


def test_every_replay_archive_counter_has_its_declared_integer_dtype() -> None:
    with patch.object(prepare_data, "array", wraps=array) as array_factory:
        replay = prepare_reference_rows(
            array([[257, 97, 257]], dtype=int64),
            targets=array([[97, 257, 98]], dtype=int64),
            reference=_reference(),
            tokenizer=_unigram(),
            batch_size=1,
            reserved_count=16,
        )
    integer_list_count = 0
    int64_list_count = 0
    for factory_call in array_factory.call_args_list:
        raw_values: object = factory_call.args[0]
        if not isinstance(raw_values, list):
            continue
        values = from_plain(cast(list[object], raw_values), list[object])
        if values and all(isinstance(value, int) for value in values):
            integer_list_count += 1
            dtype: object = factory_call.kwargs.get("dtype")
            if dtype is int64:
                int64_list_count += 1
    assert integer_list_count == 8
    assert int64_list_count == 6
    for name in (
        "reference_bytes",
        "literal_bytes",
        "reference_rows",
        "fragment_lengths",
        "token_bytes",
    ):
        assert replay[name].dtype == np.dtype(np.int64), name
    assert replay["score_mask"].dtype == np.dtype(np.bool_)


def test_reference_row_shape_error_message_is_stable() -> None:
    with pytest.raises(
        ValueError,
        match=r"^Reference rows are not BOS-aligned shifted targets\.$",
    ) as error:
        prepare_reference_rows(
            array([[257, 97, 98]], dtype=int64),
            targets=array([[97, 98]], dtype=int64),
            reference=_reference(),
            tokenizer=None,
            batch_size=1,
            reserved_count=16,
        )
    assert str(error.value) == "Reference rows are not BOS-aligned shifted targets."


def test_unnormalized_tokenizer_error_message_is_stable() -> None:
    tokenizer = _unigram()
    tokenizer.normalizer = normalizers.Lowercase()
    with pytest.raises(
        ValueError,
        match=r"^Unigram replay requires an unnormalized ordinary tokenizer\.$",
    ) as error:
        prepare_reference_rows(
            array([[257, 97]], dtype=int64),
            targets=array([[97, 98]], dtype=int64),
            reference=_reference(),
            tokenizer=tokenizer,
            batch_size=1,
            reserved_count=16,
        )
    assert str(error.value) == (
        "Unigram replay requires an unnormalized ordinary tokenizer."
    )


def test_reference_replay_pads_only_when_continuations_leave_a_partial_batch() -> None:
    backend = _unigram()
    replay = prepare_reference_rows(
        array([[257, 97, 98], [257, 256, 0]], dtype=int64),
        targets=array([[97, 98, 99], [256, 0, 0]], dtype=int64),
        reference=_reference(),
        tokenizer=backend,
        batch_size=3,
        reserved_count=16,
    )
    assert cast(NDArray[np.int64], replay["reference_rows"]).tolist() == [0, 1, 1]
    assert cast(NDArray[np.int64], replay["reference_bytes"]).tolist() == [3, 6, 0]
    assert cast(NDArray[np.int64], replay["literal_bytes"]).tolist() == [3, 6, 0]
    assert cast(NDArray[np.int64], replay["inputs"]).shape == (3, 3)


def test_reference_replay_counts_each_incomplete_utf8_fragment() -> None:
    replay = prepare_reference_rows(
        array([[257, 226, 257]], dtype=int64),
        targets=array([[226, 257, 226]], dtype=int64),
        reference=_reference(),
        tokenizer=None,
        batch_size=2,
        reserved_count=16,
    )
    assert int(replay["incomplete_utf8_suffixes"]) == 2
    assert cast(NDArray[np.int64], replay["fragment_lengths"]).tolist() == [1, 1]


@pytest.mark.parametrize("invalid", ["shape", "empty", "shift", "bos", "batch"])
def test_reference_replay_rejects_malformed_reference_rows(invalid: str) -> None:
    inputs = array([[257, 97, 98]], dtype=int64)
    targets = array([[97, 98, 99]], dtype=int64)
    batch_size = 1
    if invalid == "shape":
        targets = array([[97, 98]], dtype=int64)
    elif invalid == "empty":
        inputs = targets = array([], dtype=int64).reshape((0, 3))
    elif invalid == "shift":
        targets[0, 0] = 98
    elif invalid == "bos":
        inputs[0, 0] = 256
    else:
        batch_size = 0
    with pytest.raises(ValueError, match="BOS-aligned shifted targets"):
        prepare_reference_rows(
            inputs,
            targets=targets,
            reference=_reference(),
            tokenizer=None,
            batch_size=batch_size,
            reserved_count=16,
        )


@pytest.mark.parametrize("invalid", ["normalizer", "post_processor", "added_token"])
def test_reference_replay_rejects_nonordinary_tokenizers(invalid: str) -> None:
    backend = _unigram()
    if invalid == "normalizer":
        backend.normalizer = normalizers.Lowercase()
    elif invalid == "post_processor":
        backend.post_processor = processors.ByteLevel()
    else:
        backend.add_special_tokens(["<extra>"])
    with pytest.raises(ValueError, match="unnormalized ordinary tokenizer"):
        prepare_reference_rows(
            array([[257, 97, 98]], dtype=int64),
            targets=array([[97, 98, 99]], dtype=int64),
            reference=_reference(),
            tokenizer=backend,
            batch_size=1,
            reserved_count=16,
        )


def test_reference_replay_reserves_the_encoders_ids() -> None:
    """The replay vocabulary is the ordinary pieces plus the row encoder's reserve."""
    replay = prepare_reference_rows(
        array([[257, 97, 98]], dtype=int64),
        targets=array([[97, 98, 99]], dtype=int64),
        reference=_reference(),
        tokenizer=_unigram(),
        batch_size=1,
        reserved_count=10,
    )
    bos = _unigram().get_vocab_size()
    assert int(replay["bos_token_id"]) == bos
    assert int(replay["vocab_size"]) == bos + 10
    assert len(replay["token_bytes"]) == bos + 10
    assert (
        cast(NDArray[np.int64], replay["token_bytes"]).tolist()
        == [
            len(_unigram().decode([index], skip_special_tokens=False).encode())
            for index in range(bos)
        ]
        + [0] * 10
    )


def test_padding_marks_only_real_non_bos_targets_and_keeps_integer_arrays() -> None:
    inputs, targets, mask = _pad_row([7, 8, 9], bos=7, width=4)
    assert inputs.tolist() == [7, 8, 7, 7]
    assert targets.tolist() == [8, 9, 7, 7]
    assert mask.tolist() == [True, True, False, False]
    assert inputs.dtype == targets.dtype == int64
    assert mask.dtype == np.dtype(bool)


def test_windows_split_at_context_boundary_and_score_each_target_once() -> None:
    windows = _windows([7, 8, 9, 10, 11, 12], bos=7, width=3)
    assert len(windows) == 2
    assert [item.tolist() for item in windows[0]] == [
        [7, 8, 9],
        [8, 9, 10],
        [True, True, True],
    ]
    assert [item.tolist() for item in windows[1]] == [
        [9, 10, 11],
        [10, 11, 12],
        [False, True, True],
    ]
    assert all(inputs.dtype == targets.dtype == int64 for inputs, targets, _ in windows)


def test_overflow_continues_in_windows_that_score_each_target_once() -> None:
    """A replay longer than the context keeps every byte, each scored exactly once.

    Twelve byte tokens replay three BPE tokens in a width-3 context. Four windows
    each score three targets, seeing only the three tokens before their last one;
    the row's bytes stay on its first window, and filler rows keep whole batches.
    """
    backend = _unigram()
    replay = prepare_reference_rows(
        array([[257, 256, 256]], dtype=int64),
        targets=array([[256, 256, 256]], dtype=int64),
        reference=_reference(),
        tokenizer=backend,
        batch_size=3,
        reserved_count=16,
    )
    bos = backend.get_vocab_size()
    sequence = [bos, *backend.encode("abcd" * 3, add_special_tokens=False).ids]
    inputs = cast(NDArray[np.int64], replay["inputs"])
    targets = cast(NDArray[np.int64], replay["targets"])
    mask = cast(NDArray[np.bool_], replay["score_mask"])
    assert inputs.shape == (6, 3)
    assert cast(list[int], targets[mask].tolist()) == sequence[1:]
    windows = cast(Iterator[NDArray[np.int64]], inputs[:4])
    for window, end in zip(windows, (3, 6, 9, 12), strict=True):
        assert cast(list[int], window.tolist()) == sequence[end - 3 : end]
    assert not mask[4:].any()
    assert cast(NDArray[np.int64], replay["reference_rows"]).tolist() == [
        0,
        0,
        0,
        0,
        -1,
        -1,
    ]
    assert cast(list[int], replay["reference_bytes"].tolist()) == [12, 0, 0, 0, 0, 0]
    assert cast(list[int], replay["literal_bytes"].tolist()) == [12, 0, 0, 0, 0, 0]


def test_corpus_initialization_rejects_heldout_and_duplicate_training_shards() -> None:
    config = CorpusPreparation.Config()
    config.val_shard = 2
    config.train_shard_indices = (1, 2)
    with pytest.raises(ValueError, match="Validation must be excluded"):
        CorpusPreparation(config)

    config.val_shard = 3
    config.train_shard_indices = (1, 1)
    with pytest.raises(ValueError, match="must be unique"):
        CorpusPreparation(config)


def test_corpus_build_moves_donor_and_copies_validation(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    parquet.write_table(
        pa.table({"text": ["donor", "keep", "donor"]}),
        raw / "shard_00000.parquet",
    )
    parquet.write_table(pa.table({"text": ["heldout"]}), raw / "shard_00001.parquet")
    config = CorpusPreparation.Config()
    config.raw_dir = raw
    config.working_dir = tmp_path / "prepared"
    config.train_shard_indices = (0,)
    config.val_shard = 1
    config.donor_source_ids = ("0:0",)
    config.donor_destination_shards = 1

    config.make().build()

    train = (
        parquet.read_table(
            config.working_dir / "shard_00000.parquet",
        )
        .column("text")
        .to_pylist()
    )
    assert train == ["donor", "keep"]
    assert (config.working_dir / "shard_00001.parquet").read_bytes() == (
        raw / "shard_00001.parquet"
    ).read_bytes()


def test_preparation_run_all_uses_declared_stage_order() -> None:
    config = Preparation.Config()
    with (
        patch.object(prepare_data.CorpusPreparation, "fetch") as fetch,
        patch.object(prepare_data.BpePreparation, "build") as baseline,
        patch.object(prepare_data.CorpusPreparation, "build") as corpus,
        patch.object(SamplePreparation, "build") as sample,
        patch.object(UnigramPreparation, "build") as tokenizer,
        patch.object(prepare_data.RowPreparation, "build") as rows,
        patch.object(prepare_data.Preparation, "reference") as reference,
    ):
        prepare_data.Preparation(config).run("all")
    assert [
        fetch.call_count,
        baseline.call_count,
        corpus.call_count,
        sample.call_count,
        tokenizer.call_count,
        rows.call_count,
        reference.call_count,
    ] == [1] * 7


def test_preparation_run_dispatches_verify_only() -> None:
    config = Preparation.Config()
    with patch.object(prepare_data.RowPreparation, "verify") as verify:
        prepare_data.Preparation(config).run("verify")
    verify.assert_called_once_with()


def test_row_preparation_build_creates_layout_then_runs_each_stage(
    tmp_path: Path,
) -> None:
    config = RowPreparation.Config()
    config.working_dir = tmp_path / "missing-parent" / "prepared"
    config.tokenizer.path = tmp_path / "tokenizer.json"
    _unigram().save(str(config.tokenizer.path))
    encoder = ByteLevelTokenizer(config.tokenizer)
    with (
        patch.object(ByteLevelTokenizer.Config, "make", return_value=encoder),
        patch.object(
            prepare_data,
            "validated_output_path",
            wraps=validated_output_path,
        ) as validate,
        patch.object(
            Path,
            "mkdir",
            autospec=True,
            side_effect=Path.mkdir,
        ) as mkdir,
        patch.object(RowPreparation, "_training") as training,
        patch.object(RowPreparation, "_evaluation") as evaluation,
        patch.object(RowPreparation, "_manifests") as manifests,
        patch.object(RowPreparation, "verify") as verify,
    ):
        RowPreparation(config).build()
    validate.assert_called_once_with(
        config.working_dir,
        protected=[config.raw_dir, config.tokenizer.path],
    )
    mkdir.assert_any_call(config.working_dir, parents=True)
    training.assert_called_once_with(config.working_dir / "train", encoder=encoder)
    evaluation.assert_called_once_with(config.working_dir / "eval", encoder=encoder)
    manifests.assert_called_once_with(config.working_dir, encoder=encoder)
    verify.assert_called_once_with()
    assert (config.working_dir / "train").is_dir()
    assert (config.working_dir / "eval").is_dir()
    with pytest.raises(FileExistsError):
        RowPreparation(config).build()


def test_preparation_reference_passes_exact_stream_arguments_and_saves_outputs(
    tmp_path: Path,
) -> None:
    config = Preparation.Config()
    config.working_dir = tmp_path / "nested" / "run"
    config.corpus.raw_dir = tmp_path / "raw"
    config.corpus.val_shard = 7
    config.rows.eval_batch_size = 3
    config.rows.max_seq_len = 5
    config.rows.buffer_size = 7
    config.rows.eval_batches = 11
    config.rows.tokenizer.path = tmp_path / "unigram.json"
    config.rows.tokenizer.reserved_count = 13
    tokenizer_dir = config.corpus.raw_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    pickled_tokenizer = tokenizer_dir / "tokenizer.pkl"
    pickled_tokenizer.write_bytes(b"reference-tokenizer")
    encoder = Tokenizer.__new__(Tokenizer)
    encoder.token_bytes = np.array([3, 5, 8], dtype=np.int32)
    batch = {
        "media": torch.tensor([[1, 2, 3]], dtype=torch.int32),
        "label": torch.tensor([[2, 3, 4]], dtype=torch.int32),
    }
    stream = iter([batch])

    with (
        patch.object(
            Tokenizer,
            "from_directory",
            return_value=encoder,
        ) as load,
        patch.object(prepare_data, "PackedTokenStream", return_value=stream) as packed,
        patch.object(prepare_data, "write_mapping") as write_mapping,
        patch.object(prepare_data, "build_reference_eval") as build_eval,
    ):
        prepare_data.Preparation(config).reference()

    source = config.corpus.raw_dir / "shard_00007.parquet"
    load.assert_called_once_with(tokenizer_dir)
    arguments = packed.call_args.kwargs
    assert set(arguments) == {
        "paths",
        "tokenizer",
        "token_bytes",
        "batch_size",
        "max_seq_len",
        "buffer_size",
        "device",
        "max_batches",
    }
    assert arguments["paths"] == [source]
    assert arguments["tokenizer"] is encoder
    token_bytes = arguments["token_bytes"]
    assert isinstance(token_bytes, Tensor)
    assert torch.equal(
        token_bytes,
        torch.from_numpy(encoder.token_bytes),
    )
    assert arguments["batch_size"] == 3
    assert arguments["max_seq_len"] == 5
    assert arguments["buffer_size"] == 7
    assert arguments["device"] == torch.device("cpu")
    assert arguments["max_batches"] == 11

    reference_dir = config.working_dir / "reference-bpe"
    eval_x = _load_array(reference_dir / "eval_x.npy")
    eval_y = _load_array(reference_dir / "eval_y.npy")
    assert eval_x.dtype == np.dtype(np.uint16)
    assert np.array_equal(eval_x, np.array([[1, 2, 3]], dtype=np.uint16))
    assert eval_y.dtype == np.dtype(np.uint16)
    assert np.array_equal(eval_y, np.array([[2, 3, 4]], dtype=np.uint16))
    assert (reference_dir / "tokenizer.pkl").read_bytes() == b"reference-tokenizer"

    manifest = reference_dir / "PACKED_EVAL_MANIFEST.json"
    write_mapping.assert_called_once_with(manifest, value={"eval_batch_size": 3})
    build_eval.assert_called_once_with(
        reference_dir=reference_dir,
        unigram_path=config.rows.tokenizer.path,
        reserved_count=13,
        output=config.working_dir / "reference-eval",
    )


def test_preparation_reference_rejects_existing_output_directory(
    tmp_path: Path,
) -> None:
    config = Preparation.Config()
    config.working_dir = tmp_path / "run"
    config.corpus.raw_dir = tmp_path / "raw"
    config.corpus.val_shard = 2
    config.rows.tokenizer.path = tmp_path / "unigram.json"
    tokenizer_dir = config.corpus.raw_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (tokenizer_dir / "tokenizer.pkl").write_bytes(b"tokenizer")
    (config.working_dir / "reference-bpe").mkdir(parents=True)
    encoder = type("Encoder", (), {"token_bytes": np.array([1])})()
    with (
        patch.object(Tokenizer, "from_directory", return_value=encoder),
        patch.object(prepare_data, "PackedTokenStream", return_value=iter([])),
        patch.object(prepare_data, "build_reference_eval") as build_eval,
        pytest.raises(FileExistsError, match="reference-bpe"),
    ):
        prepare_data.Preparation(config).reference()
    build_eval.assert_not_called()


def test_row_preparation_training_writes_full_rows_and_flushes(tmp_path: Path) -> None:
    config = RowPreparation.Config()
    config.train_batches = 1
    config.batch_size = 1
    config.max_seq_len = 3
    config.train_buffer_size = 1
    config.tokenizer.path = tmp_path / "tokenizer.json"
    _unigram().save(str(config.tokenizer.path))
    encoder = ByteLevelTokenizer(config.tokenizer)
    with patch.object(prepare_data, "_document_batches", return_value=iter([["doc"]])):
        RowPreparation(config)._training(tmp_path, encoder=encoder)
    rows = _load_array(tmp_path / "train_rows.npy")
    assert np.array_equal(
        rows,
        np.array([encoder.encode_batch(["doc"])[0]], dtype=np.uint16),
    )


def test_row_preparation_evaluation_saves_batches_and_byte_tables(
    tmp_path: Path,
) -> None:
    config = RowPreparation.Config()
    config.eval_batches = 1
    config.eval_batch_size = 2
    config.max_seq_len = 3
    config.train_shard_indices = (0, 1)
    config.val_shard = 4
    config.buffer_size = 7
    config.tokenizer.path = tmp_path / "tokenizer.json"
    _unigram().save(str(config.tokenizer.path))
    encoder = ByteLevelTokenizer(config.tokenizer)
    batch = {
        "media": torch.tensor([[0, 1, 0], [0, 1, 0]]),
        "label": torch.tensor([[1, 0, 1], [1, 0, 1]]),
    }
    stream = iter([batch])
    with patch.object(prepare_data, "PackedTokenStream", return_value=stream) as packed:
        RowPreparation(config)._evaluation(tmp_path, encoder=encoder)
    arguments = packed.call_args.kwargs
    assert set(arguments) == {
        "paths",
        "tokenizer",
        "token_bytes",
        "batch_size",
        "max_seq_len",
        "buffer_size",
        "device",
        "max_batches",
    }
    assert arguments["paths"] == [config.raw_dir / "shard_00004.parquet"]
    assert arguments["tokenizer"] is encoder
    assert arguments["batch_size"] == 2
    assert arguments["max_seq_len"] == 3
    assert arguments["buffer_size"] == 7
    assert arguments["device"] == torch.device("cpu")
    assert arguments["max_batches"] == 1
    token_bytes = arguments["token_bytes"]
    assert isinstance(token_bytes, torch.Tensor)
    assert torch.equal(
        token_bytes,
        torch.from_numpy(encoder.token_bytes),
    )
    loaded_eval_x = _load_array(tmp_path / "eval_x.npy")
    loaded_eval_y = _load_array(tmp_path / "eval_y.npy")
    assert loaded_eval_x.dtype == np.dtype(np.uint16)
    assert loaded_eval_y.dtype == np.dtype(np.uint16)
    eval_x: NDArray[np.uint16] = np.asarray(loaded_eval_x, dtype=np.uint16)
    eval_y: NDArray[np.uint16] = np.asarray(loaded_eval_y, dtype=np.uint16)
    assert np.array_equal(eval_x, np.array([[0, 1, 0], [0, 1, 0]], dtype=np.uint16))
    assert np.array_equal(eval_y, np.array([[1, 0, 1], [1, 0, 1]], dtype=np.uint16))
    assert np.array_equal(
        _load_array(tmp_path / "token_bytes_primary.npy"),
        encoder.token_bytes,
    )
    assert np.array_equal(
        _load_array(tmp_path / "token_bytes_literal.npy"),
        encoder.token_bytes_literal,
    )


def test_row_preparation_manifests_capture_loader_geometry(tmp_path: Path) -> None:
    config = RowPreparation.Config()
    config.batch_size = 2
    config.eval_batch_size = 3
    config.max_seq_len = 4
    config.train_buffer_size = 5
    config.buffer_size = 6
    config.documents_per_refill = 7
    config.train_shard_indices = (1, 3)
    config.val_shard = 8
    config.eval_batches = 2
    config.tokenizer.path = tmp_path / "tokenizer.json"
    _unigram().save(str(config.tokenizer.path))
    encoder = ByteLevelTokenizer(config.tokenizer)
    train_dir, eval_dir = tmp_path / "train", tmp_path / "eval"
    train_dir.mkdir()
    eval_dir.mkdir()
    np.save(train_dir / "train_rows.npy", np.zeros((8, 4), dtype=np.uint16))
    byte_lengths: dict[int, tuple[int, int]] = {}
    for token_id in range(encoder.bos_token_id):
        piece = encoder.backend.id_to_token(token_id)
        assert piece is not None
        byte_lengths[token_id] = (
            len(encoder.backend.decode([token_id], skip_special_tokens=False).encode()),
            len(piece),
        )
    distinguishing_token = next(
        token_id
        for token_id, (primary, literal) in byte_lengths.items()
        if primary != literal
    )
    primary_bytes, literal_bytes = byte_lengths[distinguishing_token]
    eval_targets = np.full((2, 3), distinguishing_token, dtype=np.uint16)
    np.save(eval_dir / "eval_y.npy", eval_targets)
    writes: dict[str, dict[str, object]] = {}

    def record_mapping(path: Path, *, value: object) -> None:
        writes[path.name] = from_plain(value, dict[str, object])

    with (
        patch(
            "priml.baselines.nanochat.scripts.prepare_data.load",
            wraps=load,
        ) as load_mock,
        patch.object(
            prepare_data,
            "write_mapping",
            side_effect=record_mapping,
        ),
    ):
        RowPreparation(config)._manifests(tmp_path, encoder=encoder)

    assert [call.args for call in load_mock.call_args_list] == [
        (train_dir / "train_rows.npy",),
        (eval_dir / "eval_y.npy",),
    ]
    assert [call.kwargs for call in load_mock.call_args_list] == [
        {"mmap_mode": "r"},
        {"mmap_mode": "r"},
    ]
    train = writes["PREPARED_MANIFEST.json"]["train"]
    assert train == {
        "batch_size": 2,
        "seq_len": 4,
        "buffer_size": 5,
        "documents_per_refill": 7,
        "train_shard_indices": (1, 3),
        "total_rows": 8,
    }
    assert writes["PREPARED_MANIFEST.json"] == {
        "vocab_size": encoder.vocab_size,
        "bos_id": encoder.bos_token_id,
        "train": {
            "batch_size": 2,
            "seq_len": 4,
            "buffer_size": 5,
            "documents_per_refill": 7,
            "train_shard_indices": (1, 3),
            "total_rows": 8,
        },
    }
    assert writes["PACKED_EVAL_MANIFEST.json"] == {
        "protocol": "standard-packed-shard7-tokensub-v1",
        "vocab_size": encoder.vocab_size,
        "bos_token_id": encoder.bos_token_id,
        "eval_batch_size": 3,
        "max_seq_len": 4,
        "physical_positions": 6,
        "val_shard": 8,
        "buffer_size": 6,
        "rows": 2,
        "batches": 2,
        "byte_tables": {
            "scored_positions": 6 if primary_bytes else 0,
            "primary": {
                "file": "token_bytes_primary.npy",
                "total_on_eval_y": 6 * primary_bytes,
            },
            "literal": {
                "file": "token_bytes_literal.npy",
                "total_on_eval_y": 6 * literal_bytes,
            },
        },
    }


def test_row_preparation_verify_binds_reader_to_configured_paths() -> None:
    config = RowPreparation.Config()
    config.working_dir = Path("/prepared")
    config.batch_size = 2
    config.eval_batch_size = 3
    config.max_seq_len = 4
    config.train_shard_indices = (1, 3)
    config.val_shard = 8
    config.buffer_size = 6
    config.train_buffer_size = 7
    config.eval_batches = 5

    expected_eval_tokens = 5 * config.eval_batch_size * config.max_seq_len

    def identity_config(cfg: NanoChatData.Config) -> NanoChatData.Config:
        return cfg

    with patch.object(
        prepare_data,
        "PreparedTokenRows",
        side_effect=identity_config,
    ) as reader:
        result = RowPreparation(config).verify()
    assert isinstance(result, NanoChatData.Config)
    assert result.prepared_train_manifest == Path(
        "/prepared/train/PREPARED_MANIFEST.json",
    )
    assert result.prepared_eval_manifest == Path(
        "/prepared/eval/PACKED_EVAL_MANIFEST.json",
    )
    assert result.batch_size == config.batch_size
    assert result.eval_batch_size == config.eval_batch_size
    assert result.max_seq_len == config.max_seq_len
    assert result.eval_tokens == expected_eval_tokens
    assert result.train_shard_indices == config.train_shard_indices
    assert result.val_shard == config.val_shard
    assert result.buffer_size == config.buffer_size
    assert result.train_buffer_size == config.train_buffer_size
    reader.assert_called_once()


def test_row_training_rejects_exhausted_source_with_stable_error(
    tmp_path: Path,
) -> None:
    config = RowPreparation.Config()
    config.train_batches = 1
    config.batch_size = 1
    config.max_seq_len = 1
    config.train_buffer_size = 1
    config.tokenizer.path = tmp_path / "tokenizer.json"
    _unigram().save(str(config.tokenizer.path))
    encoder = ByteLevelTokenizer(config.tokenizer)

    with (
        patch.object(prepare_data, "_document_batches", return_value=iter([])),
        pytest.raises(RuntimeError) as error,
    ):
        RowPreparation(config)._training(tmp_path, encoder=encoder)

    assert str(error.value) == (
        "Training preparation exhausted the source corpus; wrapping is forbidden."
    )


def test_row_training_flushes_uint16_rows_and_logs_progress(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = RowPreparation.Config()
    config.train_batches = 6
    config.batch_size = 1
    config.max_seq_len = 1
    config.train_buffer_size = 1
    config.tokenizer.path = tmp_path / "tokenizer.json"
    _unigram().save(str(config.tokenizer.path))
    encoder = ByteLevelTokenizer(config.tokenizer)
    encoded = encoder.encode_batch(["x"])[0]
    assert prepare_data.TRAINING_PROGRESS_INTERVAL == 20_000
    monkeypatch.setattr(prepare_data, "TRAINING_PROGRESS_INTERVAL", 5)
    caplog.set_level(
        "INFO",
        logger="priml.baselines.nanochat.scripts.prepare_data",
    )

    with patch.object(
        prepare_data,
        "_document_batches",
        return_value=iter([["x"]] * 6),
    ) as batches:
        prepare_data.RowPreparation(config)._training(tmp_path, encoder=encoder)

    batches.assert_called_once_with(config)
    loaded_rows = _load_array(tmp_path / "train_rows.npy")
    assert loaded_rows.dtype == np.dtype(np.uint16)
    rows: NDArray[np.uint16] = np.asarray(loaded_rows, dtype=np.uint16)
    assert rows.dtype == np.dtype(np.uint16)
    assert rows.shape == (6, 2)
    expected_rows: NDArray[np.uint16] = np.array([encoded] * 6, dtype=np.uint16)
    assert np.array_equal(rows, expected_rows)
    assert caplog.messages == [
        "Prepared 0/6 training rows",
        "Prepared 5/6 training rows",
    ]


def test_preparation_dump_bundles_source_and_manifest(tmp_path: Path) -> None:
    config = Preparation.Config()
    config.working_dir = tmp_path
    config.rows.working_dir = tmp_path / "rows"
    config.tokenizer.working_dir = tmp_path / "tokenizer"
    config.sample.working_dir = tmp_path / "sample"
    for directory, name in (
        (config.rows.working_dir, "rows.npy"),
        (config.tokenizer.working_dir, "tokenizer.json"),
        (config.sample.working_dir, "fit.txt"),
        (tmp_path / "reference-eval", "eval.npz"),
        (tmp_path / "reference-bpe", "bpe.npz"),
        (tmp_path / "raw/tokenizer", "tokenizer.pkl"),
    ):
        directory.mkdir(parents=True)
        (directory / name).write_bytes(name.encode())
    destination = tmp_path / "bundle.tar"
    prepare_data.Preparation(config).dump(destination)
    with tarfile.open(destination) as archive:
        members = set(archive.getnames())
        assert "data/rows.npy" in members
        assert "tokenizer/tokenizer.json" in members
        assert "fitting/fit.txt" in members
        assert "reference-eval/eval.npz" in members
        assert "reference-bpe/bpe.npz" in members
        assert "raw/tokenizer/tokenizer.pkl" in members
        manifest = archive.extractfile("MANIFEST.json")
        assert manifest is not None
        assert "data/rows.npy" in manifest.read().decode()
    with pytest.raises(FileExistsError):
        prepare_data.Preparation(config).dump(destination)


def test_preparation_dump_manifest_tracks_config_and_stages_by_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Preparation.Config()
    config.working_dir = tmp_path
    config.rows.working_dir = tmp_path / "rows"
    config.tokenizer.working_dir = tmp_path / "tokenizer"
    config.sample.working_dir = tmp_path / "sample"
    for directory, name in (
        (config.rows.working_dir, "rows.npy"),
        (config.tokenizer.working_dir, "tokenizer.json"),
        (config.sample.working_dir, "fit.txt"),
        (tmp_path / "reference-eval", "eval.npz"),
        (tmp_path / "reference-bpe", "bpe.npz"),
        (tmp_path / "raw/tokenizer", "tokenizer.pkl"),
    ):
        directory.mkdir(parents=True)
        (directory / name).write_bytes(name.encode())
    changed = config.copy_tree()
    changed.rows.max_seq_len += 1
    destination = tmp_path / "missing" / "intermediate" / "bundle.tar"
    changed_destination = tmp_path / "missing" / "intermediate" / "changed.tar"
    original_pformat = Preparation.Config.pformat

    def pformat(
        self: Preparation.Config,
        *,
        hide_default_values: bool,
    ) -> str:
        assert hide_default_values is False
        return original_pformat(self, hide_default_values=hide_default_values)

    monkeypatch.setattr(Preparation.Config, "pformat", pformat)
    with patch.object(
        tempfile,
        "TemporaryDirectory",
        wraps=tempfile.TemporaryDirectory,
    ) as temporary_directory:
        Preparation(config).dump(destination)
        Preparation(changed).dump(changed_destination)

    assert [call.kwargs for call in temporary_directory.call_args_list] == [
        {"dir": destination.parent, "prefix": "nanochat-archive-"},
        {"dir": changed_destination.parent, "prefix": "nanochat-archive-"},
    ]
    with tarfile.open(destination) as archive:
        members = set(archive.getnames())
        assert "preparation/prepare_data.py" in members
        assert "reference-eval/eval.npz" in members
        manifest_member = archive.getmember("MANIFEST.json")
        assert manifest_member.mode == 0o644
        manifest_file = archive.extractfile(manifest_member)
        assert manifest_file is not None
        manifest_bytes = manifest_file.read()
        manifest = manifest_bytes.decode()
        expected_manifest = (
            json.dumps(
                {
                    "files": sorted(members - {"MANIFEST.json"}),
                    "config": config.pformat(hide_default_values=False),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        assert manifest_bytes == expected_manifest
        assert "max_seq_len" in manifest
    with tarfile.open(changed_destination) as archive:
        manifest_file = archive.extractfile("MANIFEST.json")
        assert manifest_file is not None
        changed_manifest = manifest_file.read().decode()
    assert "max_seq_len" in changed_manifest
    assert changed_manifest != manifest
    with pytest.raises(ValueError, match="protected"):
        Preparation(config).dump(config.rows.working_dir / "rows.npy")
    with pytest.raises(FileExistsError) as error:
        Preparation(config).dump(destination)
    assert error.value.args == (destination,)


def test_documents_stops_when_the_character_budget_is_reached(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shard.parquet"
    parquet.write_table(
        pa.table({"text": ["ab", "cd", "ef"]}),
        shard,
        row_group_size=1,
    )

    assert list(_documents([shard], max_chars=4, doc_cap=2)) == ["ab", "cd"]


def test_documents_counts_from_zero_for_a_small_budget(tmp_path: Path) -> None:
    shard = tmp_path / "shard.parquet"
    parquet.write_table(pa.table({"text": ["a", "b", "c"]}), shard)

    assert list(_documents([shard], max_chars=2, doc_cap=1)) == ["a", "b"]


def test_differing_reports_only_requested_values_that_changed() -> None:
    recorded: dict[str, object] = {"same": 4, "changed": 5}
    requested: dict[str, object] = {"same": 4, "changed": 6}

    assert _differing(recorded, requested) == {"changed": (5, 6)}


def test_download_logs_url_and_records_its_receipt(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    url = _PINNED + "shard_00000.parquet"
    caplog.set_level(
        "INFO",
        logger="priml.baselines.nanochat.scripts.prepare_data",
    )
    with patch(
        "priml.baselines.nanochat.scripts.prepare_data.request.urlopen",
        return_value=io.BytesIO(b"payload"),
    ):
        paths = _download(tmp_path, count=1)

    assert caplog.messages == [f"fetching {url}"]
    assert paths[0].read_bytes() == b"payload"
    assert (tmp_path / "shard_00000.parquet.source-url").read_text() == url


def test_token_bytes_uses_utf8_lengths_and_int32() -> None:
    encoding = tiktoken.Encoding(
        name="test-token-bytes",
        pat_str=r".+",
        mergeable_ranks={b"a": 0, "é".encode(): 1},
        special_tokens={
            token: 2 + index for index, token in enumerate(RESERVED_TOKENS)
        },
    )

    lengths = _token_bytes(encoding)

    assert lengths.dtype == np.dtype(np.int32)
    assert lengths.tolist() == [1, 2, *([0] * len(RESERVED_TOKENS))]


def test_donor_unigram16k_pins_source_ids_and_every_other_field() -> None:
    config = donor_unigram16k()
    ids = config.corpus.donor_source_ids

    assert len(ids) == 490
    assert (
        hashlib.sha256("\n".join(ids).encode()).hexdigest()
        == "f45772e511cc177162163fded084cdc2efcdfe7edfc194a5cfbefa8875341864"
    )

    expected = Preparation.Config()
    expected.corpus.donor_source_ids = ids
    expected.experiment_corpora = {"exp019": "donor-original"}
    assert config == expected


def test_prepare_uses_training_prefix_plus_one_heldout_shard(
    tmp_path: Path,
) -> None:
    with (
        patch.object(
            prepare_data,
            "_download",
            return_value=[Path("s0"), Path("s1"), Path("s2")],
        ) as download,
        patch.object(prepare_data, "_fit_vocabulary") as fit,
    ):
        result = prepare_data.prepare(
            tmp_path / "data",
            num_train_shards=2,
            vocab_size=17,
            tokenizer_train_chars=23,
            tokenizer_doc_cap=5,
        )
    assert result == tmp_path / "data"
    download.assert_called_once_with(result, count=3)
    fit.assert_called_once_with(
        [Path("s0"), Path("s1")],
        out=result / "tokenizer",
        vocab_size=17,
        train_chars=23,
        doc_cap=5,
    )


def test_prepare_uses_default_recipe_and_directory(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    directory = tmp_path / "missing-parent" / "prepared"
    shards = [tmp_path / f"shard_{index:05d}.parquet" for index in range(8)]
    caplog.set_level(
        "INFO",
        logger="priml.baselines.nanochat.scripts.prepare_data",
    )
    with (
        patch.object(
            prepare_data,
            "default_directory",
            return_value=directory,
        ) as default,
        patch.object(prepare_data, "_download", return_value=shards) as download,
        patch.object(prepare_data, "_fit_vocabulary") as fit,
    ):
        result = prepare_data.prepare(None)

    assert result == directory
    assert directory.is_dir()
    default.assert_called_once_with()
    download.assert_called_once_with(directory, count=8)
    fit.assert_called_once_with(
        shards[:7],
        out=directory / "tokenizer",
        vocab_size=8_192,
        train_chars=2_000_000_000,
        doc_cap=10_000,
    )
    assert caplog.messages == [f"nanochat corpus ready at {directory}"]


def test_prepare_accepts_one_character_tokenizer_budgets(tmp_path: Path) -> None:
    shards = [tmp_path / "shard_00000.parquet", tmp_path / "shard_00001.parquet"]
    with (
        patch.object(prepare_data, "_download", return_value=shards),
        patch.object(prepare_data, "_fit_vocabulary") as fit,
    ):
        result = prepare_data.prepare(
            tmp_path,
            num_train_shards=1,
            vocab_size=17,
            tokenizer_train_chars=1,
            tokenizer_doc_cap=1,
        )

    assert result == tmp_path
    fit.assert_called_once_with(
        shards[:1],
        out=tmp_path / "tokenizer",
        vocab_size=17,
        train_chars=1,
        doc_cap=1,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"num_train_shards": 0}, "num_train_shards must be positive; got 0."),
        (
            {"vocab_size": 16},
            "vocab_size must exceed the 16 reserved tokens; got 16.",
        ),
        (
            {"tokenizer_train_chars": 0},
            (
                "tokenizer_train_chars and tokenizer_doc_cap must be positive; "
                "got 0 and 10000."
            ),
        ),
        (
            {"tokenizer_doc_cap": 0},
            (
                "tokenizer_train_chars and tokenizer_doc_cap must be positive; "
                "got 2000000000 and 0."
            ),
        ),
    ],
)
def test_prepare_rejects_nonpositive_or_reserved_only_recipe(
    tmp_path: Path,
    kwargs: PrepareOverrides,
    expected_message: str,
) -> None:
    with (
        patch.object(prepare_data, "_download") as download,
        pytest.raises(
            ValueError,
            match=expected_message,
        ) as error,
    ):
        prepare_data.prepare(tmp_path, **kwargs)
    assert str(error.value) == expected_message
    download.assert_not_called()


def test_prepare_uses_staged_shards_when_download_is_disabled(tmp_path: Path) -> None:
    staged = [tmp_path / "shard_00000.parquet", tmp_path / "shard_00001.parquet"]
    with (
        patch.object(prepare_data, "_staged", return_value=staged) as get_staged,
        patch.object(prepare_data, "_download") as download,
        patch.object(prepare_data, "_fit_vocabulary") as fit,
    ):
        prepare_data.prepare(tmp_path, num_train_shards=1, download=False)
    get_staged.assert_called_once_with(tmp_path, count=2)
    download.assert_not_called()
    fit.assert_called_once_with(
        staged[:1],
        out=tmp_path / "tokenizer",
        vocab_size=8_192,
        train_chars=2_000_000_000,
        doc_cap=10_000,
    )


def test_fetch_file_requires_https_before_creating_paths(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "file.bin"
    with pytest.raises(
        ValueError,
        match=r"^Preparation downloads require HTTPS\.$",
    ) as error:
        prepare_data.fetch_file("http://example.test/file", destination=destination)
    assert str(error.value) == "Preparation downloads require HTTPS."
    assert not destination.parent.exists()


def test_fetch_file_downloads_once_and_validates_cached_source(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "file.bin"
    response = b"payload"
    with patch.object(
        request,
        "urlopen",
        return_value=io.BytesIO(response),
    ) as open_url:
        prepare_data.fetch_file("https://example.test/file", destination=destination)
        prepare_data.fetch_file("https://example.test/file", destination=destination)
    open_url.assert_called_once_with("https://example.test/file", timeout=120)
    assert destination.read_bytes() == response
    assert (
        destination.with_name("file.bin.source-url").read_text(
            encoding="utf-8",
        )
        == "https://example.test/file"
    )
    with pytest.raises(ValueError, match="Cached source does not match"):
        prepare_data.fetch_file("https://example.test/other", destination=destination)


def test_encode_fragment_round_trips_valid_and_incomplete_utf8() -> None:
    tokenizer = _unigram()
    valid = "Aé".encode()
    valid_ids = prepare_data.encode_fragment(valid, tokenizer=tokenizer)
    alphabet = byte_alphabet()
    assert "".join(tokenizer.id_to_token(item) or "" for item in valid_ids) == "".join(
        alphabet[value] for value in valid
    )

    incomplete = "A€".encode()[:-1]
    incomplete_ids = prepare_data.encode_fragment(incomplete, tokenizer=tokenizer)
    encoded_chars = "".join(
        tokenizer.id_to_token(item) or "" for item in incomplete_ids
    )
    inverse = {char: value for value, char in alphabet.items()}
    assert bytes(inverse[char] for char in encoded_chars) == incomplete


def test_encode_fragment_rejects_nonterminal_invalid_utf8() -> None:
    with pytest.raises(
        ValueError,
        match=r"^Reference fragment has non-terminal invalid UTF-8\.$",
    ):
        prepare_data.encode_fragment(b"\xffA", tokenizer=_unigram())


def test_build_reference_eval_writes_both_archives_and_protects_sources(
    tmp_path: Path,
) -> None:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / "PACKED_EVAL_MANIFEST.json").write_text(
        '{"eval_batch_size": 2}',
    )
    (reference_dir / "tokenizer.pkl").write_bytes(pickle.dumps(_reference()))
    np.save(reference_dir / "eval_x.npy", np.array([[256, 65]], dtype=np.int64))
    np.save(reference_dir / "eval_y.npy", np.array([[65, 66]], dtype=np.int64))
    unigram_path = tmp_path / "unigram.json"
    _unigram().save(str(unigram_path))
    destination = tmp_path / "nested" / "output"
    expected_inputs = np.asarray(
        _load_array(reference_dir / "eval_x.npy"),
        dtype=np.int64,
    )
    expected_targets = np.asarray(
        _load_array(reference_dir / "eval_y.npy"),
        dtype=np.int64,
    )
    results = [
        {"inputs": np.array([[256, 65]]), "targets": np.array([[65, 66]])},
        {"inputs": np.array([[2, 3]]), "targets": np.array([[3, 4]])},
    ]
    with (
        patch(
            "priml.baselines.nanochat.scripts.prepare_data.load",
            wraps=load,
        ) as load_mock,
        patch.object(prepare_data, "from_plain", wraps=from_plain) as convert_mock,
        patch(
            "priml.baselines.nanochat.scripts.prepare_data.validated_output_path",
            wraps=validated_output_path,
        ) as validate,
        patch.object(
            Path,
            "mkdir",
            autospec=True,
            side_effect=Path.mkdir,
        ) as mkdir,
        patch.object(
            prepare_data,
            "prepare_reference_rows",
            side_effect=results,
        ) as prepare_rows,
        patch(
            "priml.baselines.nanochat.scripts.prepare_data.savez",
            wraps=savez,
        ) as savez_mock,
    ):
        prepare_data.build_reference_eval(
            reference_dir=reference_dir,
            unigram_path=unigram_path,
            reserved_count=16,
            output=destination,
        )

    for index, replay_call in enumerate(prepare_rows.call_args_list):
        actual_inputs: object = replay_call.args[0]
        actual_targets: object = replay_call.kwargs["targets"]
        assert isinstance(actual_inputs, np.ndarray)
        assert isinstance(actual_targets, np.ndarray)
        assert np.array_equal(actual_inputs, expected_inputs)
        assert np.array_equal(actual_targets, expected_targets)
        assert isinstance(replay_call.kwargs["reference"], tiktoken.Encoding)
        assert replay_call.kwargs["reserved_count"] == 16
        assert replay_call.kwargs["batch_size"] == 2
        assert (replay_call.kwargs["tokenizer"] is None) is (index == 0)
    assert [call.args for call in load_mock.call_args_list] == [
        (reference_dir / "eval_x.npy",),
        (reference_dir / "eval_y.npy",),
    ]
    assert all(
        call.kwargs == {"allow_pickle": False} for call in load_mock.call_args_list
    )
    assert [
        call.args for call in convert_mock.call_args_list if call.args[1] is int
    ] == [
        (2, int),
        (2, int),
    ]
    validate.assert_called_once_with(
        destination,
        protected=[reference_dir, unigram_path],
    )
    mkdir.assert_any_call(destination, parents=True)
    assert savez_mock.call_count == 2
    assert all(
        call.kwargs["allow_pickle"] is False for call in savez_mock.call_args_list
    )
    assert np.array_equal(
        _load_archive_array(destination / "bpe.npz", "inputs"),
        np.array([[256, 65]], dtype=np.int64),
    )
    assert np.array_equal(
        _load_archive_array(destination / "unigram.npz", "targets"),
        np.array([[3, 4]], dtype=np.int64),
    )
    for protected in (reference_dir, unigram_path):
        with pytest.raises(ValueError, match="aliases protected input artifact"):
            prepare_data.build_reference_eval(
                reference_dir=reference_dir,
                unigram_path=unigram_path,
                reserved_count=16,
                output=protected,
            )
    with pytest.raises(FileExistsError):
        prepare_data.build_reference_eval(
            reference_dir=reference_dir,
            unigram_path=unigram_path,
            reserved_count=16,
            output=destination,
        )


def test_build_reference_eval_rejects_wrong_pickled_type(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / "PACKED_EVAL_MANIFEST.json").write_text(
        '{"eval_batch_size": 1}',
    )
    (reference_dir / "tokenizer.pkl").write_bytes(
        pickle.dumps("not encoding"),
    )
    with pytest.raises(TypeError) as error:
        prepare_data.build_reference_eval(
            reference_dir=reference_dir,
            unigram_path=tmp_path / "absent.json",
            reserved_count=16,
            output=tmp_path / "output",
        )
    assert str(error.value) == "Reference pickle does not contain a tiktoken encoding."


def _main_config(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> Preparation.Config:
    config = Preparation.Config()
    monkeypatch.setattr(sys, "argv", ["prepare_data", *argv])
    factory = Mock(return_value=config)
    monkeypatch.setattr(
        importlib,
        "import_module",
        Mock(return_value=SimpleNamespace(donor_unigram16k=factory)),
    )
    return config


def test_main_help_contains_workflow_description(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _main_config(monkeypatch, ["--help"])
    with pytest.raises(SystemExit) as error:
        prepare_data.main()
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "Original BPE preparation:" in output
    assert "Added prepared Unigram workflow:" in output
    assert (
        "--stage {all,fetch,baseline,reference,corpus,sample,tokenizer,rows,verify,dump,train}"
        in output
    )


def test_main_print_config_uses_absolute_directory_and_skips_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _main_config(
        monkeypatch,
        ["--print-config", "--directory", str(tmp_path / "inputs" / ".." / "data")],
    )
    pprint = Mock()
    monkeypatch.setattr(Preparation.Config, "pprint", pprint)
    make = Mock(side_effect=AssertionError("print-config must not build"))
    monkeypatch.setattr(Preparation.Config, "make", make)

    assert prepare_data.main() == 0

    assert config.working_dir == (tmp_path / "inputs" / ".." / "data").absolute()
    pprint.assert_called_once_with(hide_default_values=False)
    make.assert_not_called()


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (
            [
                "--num-train-shards",
                "3",
                "--vocab-size",
                "41",
                "--tokenizer-train-chars",
                "52",
                "--tokenizer-doc-cap",
                "17",
                "--directory",
                "inputs",
            ],
            {
                "num_train_shards": 3,
                "vocab_size": 41,
                "tokenizer_train_chars": 52,
                "tokenizer_doc_cap": 17,
            },
        ),
        (
            ["--vocab-size", "41", "--directory", "inputs"],
            {
                "num_train_shards": 7,
                "vocab_size": 41,
                "tokenizer_train_chars": 2_000_000_000,
                "tokenizer_doc_cap": 10_000,
            },
        ),
    ],
)
def test_main_bpe_flags_forward_overrides_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    expected: dict[str, object],
) -> None:
    _main_config(monkeypatch, argv)
    prepare_mock = Mock()
    monkeypatch.setattr(prepare_data, "prepare", prepare_mock)

    assert prepare_data.main() == 0

    assert prepare_mock.call_args.args == (Path("inputs").absolute(),)
    assert prepare_mock.call_args.kwargs == expected


def test_main_dump_and_regular_stages_dispatch_without_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    preparation = Mock()
    _main_config(
        monkeypatch,
        ["--stage", "dump", "--directory", str(tmp_path), "--output", "bundle.tar"],
    )
    monkeypatch.setattr(
        Preparation.Config,
        "make",
        Mock(return_value=preparation),
    )
    basic_config = Mock()
    monkeypatch.setattr(logging, "basicConfig", basic_config)

    assert prepare_data.main() == 0

    basic_config.assert_called_once_with(
        level=logging.INFO,
        format="%(message)s",
    )
    preparation.dump.assert_called_once_with(Path("bundle.tar"))
    preparation.run.assert_not_called()
    preparation.training_config.assert_not_called()

    basic_config.reset_mock()
    _main_config(monkeypatch, ["--stage", "verify", "--directory", str(tmp_path)])
    monkeypatch.setattr(
        Preparation.Config,
        "make",
        Mock(return_value=preparation),
    )

    assert prepare_data.main() == 0

    basic_config.assert_called_once_with(
        level=logging.INFO,
        format="%(message)s",
    )
    preparation.run.assert_called_once_with("verify")
    preparation.training_config.assert_not_called()


def test_main_train_forwards_options_and_launches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _main_config(
        monkeypatch,
        [
            "--stage",
            "train",
            "--directory",
            str(tmp_path / "inputs"),
            "--experiment",
            "exp019",
            "--run-directory",
            str(tmp_path / "run"),
            "--seed",
            "11",
            "--train-budget-sec",
            "3.5",
            "--save-checkpoint",
        ],
    )
    preparation = Mock()
    monkeypatch.setattr(
        Preparation.Config,
        "make",
        Mock(return_value=preparation),
    )
    training = Mock()
    preparation.training_config.return_value = training
    launch = Mock()
    monkeypatch.setattr(prepare_data, "launch_training", launch)

    assert prepare_data.main() == 0

    preparation.training_config.assert_called_once_with(
        "exp019",
        run_directory=tmp_path / "run",
        seed=11,
        train_budget_sec=3.5,
    )
    training.pprint.assert_called_once_with(hide_default_values=False)
    launch.assert_called_once_with(training, save_checkpoint=True)
    preparation.run.assert_not_called()
    preparation.dump.assert_not_called()


def test_main_rejects_factory_returning_wrong_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_data"])
    monkeypatch.setattr(
        importlib,
        "import_module",
        Mock(
            return_value=SimpleNamespace(donor_unigram16k=Mock(return_value=object())),
        ),
    )

    with pytest.raises(
        TypeError,
        match=r"^The factory must return Preparation\.Config\.$",
    ):
        prepare_data.main()


def test_row_initialization_rejects_validation_shard_in_training() -> None:
    config = RowPreparation.Config()
    config.val_shard = 2
    config.train_shard_indices = (1, 2)

    with pytest.raises(ValueError, match="Validation must be excluded"):
        RowPreparation(config)


def test_reference_replay_rejects_non_matrix_rows() -> None:
    with pytest.raises(ValueError, match="BOS-aligned shifted targets"):
        prepare_reference_rows(
            array([257, 97], dtype=int64),
            targets=array([97, 98], dtype=int64),
            reference=_reference(),
            tokenizer=None,
            batch_size=1,
            reserved_count=16,
        )


def _text_column(path: Path) -> list[str]:
    values: object = parquet.read_table(path).column("text").to_pylist()
    return from_plain(values, list[str])


def test_corpus_build_deduplicates_moves_donors_and_preserves_sources(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    original = {
        0: ["donor-a", "donor-b", "donor-c", "keep-zero"],
        1: ["keep-one", "duplicate"],
        2: ["keep-two", "duplicate"],
        3: ["heldout"],
    }
    for shard, texts in original.items():
        parquet.write_table(
            pa.table({"text": texts}),
            raw / f"shard_{shard:05d}.parquet",
        )
    config = CorpusPreparation.Config()
    config.raw_dir = raw
    config.working_dir = tmp_path / "nested" / "prepared"
    config.train_shard_indices = (0, 1, 2)
    config.val_shard = 3
    config.donor_source_ids = ("0:0", "0:1", "0:2")
    config.donor_destination_shards = 2
    config.row_group_size = 1
    config.compression = "zstd"

    config.make().build()

    prepared = config.working_dir
    assert _text_column(prepared / "shard_00000.parquet") == [
        "donor-a",
        "donor-c",
        "keep-zero",
    ]
    assert _text_column(prepared / "shard_00001.parquet") == [
        "keep-one",
        "donor-b",
        "duplicate",
    ]
    assert _text_column(prepared / "shard_00002.parquet") == [
        "keep-two",
    ]
    for shard, texts in original.items():
        source = raw / f"shard_{shard:05d}.parquet"
        if shard == 3:
            assert (prepared / source.name).read_bytes() == source.read_bytes()
        else:
            assert _text_column(source) == texts


def test_corpus_build_protects_raw_corpus(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    config = CorpusPreparation.Config()
    config.raw_dir = raw
    config.working_dir = raw
    config.train_shard_indices = ()
    config.donor_source_ids = ()

    with pytest.raises(
        ValueError,
        match="output path aliases protected input artifact",
    ):
        config.make().build()
    assert list(raw.iterdir()) == []


def test_corpus_build_rejects_duplicate_donor_identities(tmp_path: Path) -> None:
    config = CorpusPreparation.Config()
    config.raw_dir = tmp_path / "raw"
    config.working_dir = tmp_path / "output"
    config.donor_source_ids = ("0:0", "0:0")

    with pytest.raises(
        ValueError,
        match=r"^The donor selection contains repeated identities\.$",
    ) as error:
        config.make().build()
    assert str(error.value) == "The donor selection contains repeated identities."
    assert not config.working_dir.exists()


def test_row_preparation_build_protects_raw_and_tokenizer_inputs(
    tmp_path: Path,
) -> None:
    config = RowPreparation.Config()
    config.raw_dir = tmp_path / "raw"
    config.tokenizer.path = config.raw_dir / "tokenizer.json"
    config.working_dir = config.tokenizer.path

    with pytest.raises(
        ValueError,
        match="output path aliases protected input artifact",
    ):
        RowPreparation(config).build()
    assert not config.working_dir.exists()


def test_row_preparation_build_refuses_existing_output(tmp_path: Path) -> None:
    config = RowPreparation.Config()
    config.working_dir = tmp_path / "prepared"
    config.working_dir.mkdir()

    with pytest.raises(FileExistsError):
        RowPreparation(config).build()


def test_fetch_cached_receipt_reads_explicit_lowercase_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "shard.parquet"
    receipt = destination.with_name(destination.name + ".source-url")
    url = "https://example.test/shard.parquet"
    destination.write_bytes(b"cached")
    receipt.write_text(url, encoding="utf-8")
    read_text = Path.read_text
    encodings: list[str | None] = []

    def record_read(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path == receipt:
            encodings.append(encoding)
        return read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "read_text", record_read)
    prepare_data.fetch_file(url, destination=destination)

    assert encodings == ["utf-8"]


def test_fetch_write_receipt_uses_explicit_lowercase_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "nested" / "shard.parquet"
    write_text = Path.write_text
    encodings: list[str | None] = []

    def record_write(
        path: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> int:
        if path.name.endswith(".source-url"):
            encodings.append(encoding)
        return write_text(path, data, encoding=encoding, errors=errors)

    def open_url(url: str | request.Request, *, timeout: float) -> io.BytesIO:
        del url
        assert timeout == 120
        return io.BytesIO(b"downloaded")

    monkeypatch.setattr(Path, "write_text", record_write)
    monkeypatch.setattr(request, "urlopen", open_url)

    prepare_data.fetch_file(
        "https://example.test/shard.parquet",
        destination=destination,
    )

    assert encodings == ["utf-8"]
    assert destination.read_bytes() == b"downloaded"


def test_fetch_cached_source_mismatch_message_is_exact(tmp_path: Path) -> None:
    destination = tmp_path / "shard.parquet"
    destination.write_bytes(b"cached")
    destination.with_name(destination.name + ".source-url").write_text(
        "https://example.test/old.parquet",
        encoding="utf-8",
    )
    url = "https://example.test/new.parquet"

    with pytest.raises(
        ValueError,
        match=r"^Cached source does not match ",
    ) as error:
        prepare_data.fetch_file(url, destination=destination)

    assert str(error.value) == (
        f"Cached source does not match {url}: {destination}. "
        "Use a fresh download directory for this source."
    )


def test_fit_vocabulary_uses_lowercase_sibling_staging_suffixes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = tmp_path / "shard_00000.parquet"
    _write_shard(tmp_path, 0, ["hello world 123", "café hello", "你好 world"])
    output = tmp_path / "tokenizer"
    with_suffix = Path.with_suffix
    calls: list[tuple[str, str]] = []

    def record_suffix(path: Path, suffix: str) -> Path:
        if path.parent == output:
            calls.append((path.name, suffix))
        return with_suffix(path, suffix)

    monkeypatch.setattr(Path, "with_suffix", record_suffix)
    prepare_data._fit_vocabulary(
        [shard],
        out=output,
        vocab_size=280,
        train_chars=24,
        doc_cap=7,
    )

    assert calls == [
        ("tokenizer.pkl", ".pkl.partial"),
        ("tokenizer_recipe.json", ".json.partial"),
    ]
    assert (output / "tokenizer.pkl").is_file()
    assert (output / "tokenizer_recipe.json").is_file()


def test_fit_vocabulary_rejects_a_non_object_receipt(tmp_path: Path) -> None:
    shard = tmp_path / "shard.parquet"
    shard.touch()
    output = tmp_path / "tokenizer"
    output.mkdir()
    pickled = output / "tokenizer.pkl"
    recipe = output / "tokenizer_recipe.json"
    byte_table = output / "token_bytes.npy"
    pickled.write_bytes(b"old vocabulary")
    recipe.write_text("[]", encoding="utf-8")
    byte_table.write_bytes(b"old byte table")
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(ReadError):
        prepare_data._fit_vocabulary(
            [shard],
            out=output,
            vocab_size=280,
            train_chars=24,
            doc_cap=7,
        )

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_fit_vocabulary_logs_exact_reason_for_missing_receipt(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    shard = tmp_path / "shard_00000.parquet"
    _write_shard(tmp_path, 0, ["hello world 123", "café hello", "你好 world"])
    output = tmp_path / "tokenizer"
    output.mkdir()
    (output / "tokenizer.pkl").write_bytes(b"stale vocabulary")
    caplog.set_level(
        logging.INFO,
        logger="priml.baselines.nanochat.scripts.prepare_data",
    )

    prepare_data._fit_vocabulary(
        [shard],
        out=output,
        vocab_size=280,
        train_chars=24,
        doc_cap=7,
    )

    assert caplog.messages[0] == (
        f"refitting the nanochat vocabulary at {output}: no recipe recorded beside it"
    )


def test_unique_rows_reports_an_exact_hash_collision_error() -> None:
    collision = hashlib.sha256()
    seen: dict[bytes, tuple[str, str]] = {}

    with (
        patch.object(hashlib, "sha256", return_value=collision),
        pytest.raises(
            ValueError,
            match=r"^SHA-256 collision between distinct literal texts\.$",
        ) as error,
    ):
        prepare_data.unique_rows(
            iter((("0:0", "first literal"), ("0:1", "different literal"))),
            seen=seen,
        )

    assert str(error.value) == "SHA-256 collision between distinct literal texts."
    assert seen == {collision.digest(): ("0:0", "first literal")}


def _configure_main(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> Preparation.Config:
    config = Preparation.Config()
    factory_module = ModuleType("nanochat")
    factory_module.__dict__["donor_unigram16k"] = lambda: config

    def import_factory(name: str) -> ModuleType:
        del name
        return factory_module

    monkeypatch.setattr(sys, "argv", ["prepare_data", *argv])
    monkeypatch.setattr(importlib, "import_module", import_factory)
    return config


def test_main_passes_the_unmodified_multiline_script_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_main(monkeypatch, ["--print-config"])
    real_parser = argparse.ArgumentParser
    descriptions: list[str] = []

    def capture_parser(
        *,
        description: str,
        formatter_class: type[argparse.HelpFormatter],
    ) -> argparse.ArgumentParser:
        descriptions.append(description)
        return real_parser(
            description=description,
            formatter_class=formatter_class,
        )

    monkeypatch.setattr(
        prepare_data,
        "argparse",
        SimpleNamespace(
            ArgumentParser=capture_parser,
            RawDescriptionHelpFormatter=argparse.RawDescriptionHelpFormatter,
        ),
    )
    prepare_data.main()

    assert descriptions == [(prepare_data.__doc__ or "").split("\n", 2)[2]]
    assert "Original BPE preparation:\n    Run once" in descriptions[0]


def test_main_bpe_override_keeps_other_vocabulary_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_main(
        monkeypatch,
        ["--tokenizer-train-chars", "52", "--directory", str(tmp_path)],
    )
    prepare = Mock()
    monkeypatch.setattr(prepare_data, "prepare", prepare)

    assert prepare_data.main() == 0

    prepare.assert_called_once_with(
        tmp_path,
        num_train_shards=7,
        vocab_size=8_192,
        tokenizer_train_chars=52,
        tokenizer_doc_cap=10_000,
    )


@pytest.mark.parametrize("conflict", [["--stage", "rows"], ["--print-config"]])
def test_main_refuses_bpe_flags_beside_a_prepared_workflow_flag(
    monkeypatch: pytest.MonkeyPatch,
    conflict: list[str],
) -> None:
    _configure_main(monkeypatch, ["--vocab-size", "41", *conflict])
    prepare = Mock()
    monkeypatch.setattr(prepare_data, "prepare", prepare)
    with pytest.raises(SystemExit) as error:
        prepare_data.main()
    assert error.value.code == 2
    prepare.assert_not_called()


def test_main_hands_prepare_the_expanded_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_main(monkeypatch, ["--vocab-size", "41", "--directory", "~/corpus"])
    prepare = Mock()
    monkeypatch.setattr(prepare_data, "prepare", prepare)
    assert prepare_data.main() == 0
    assert prepare.call_args.args == (Path("~/corpus").expanduser().absolute(),)


def test_launch_training_emits_exact_recipe_text_and_explicit_config_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = exp022()
    config.working_dir = tmp_path / "run"
    original_pformat = NgramTrainLoop.Config.pformat
    format_options: list[bool] = []

    def record_pformat(
        instance: NgramTrainLoop.Config,
        *,
        hide_default_values: bool = True,
    ) -> str:
        format_options.append(hide_default_values)
        return original_pformat(
            instance,
            hide_default_values=hide_default_values,
        )

    monkeypatch.setattr(NgramTrainLoop.Config, "pformat", record_pformat)
    monkeypatch.setattr(subprocess, "run", Mock())

    prepare_data.launch_training(config)

    recipe = (config.working_dir / "prepared_experiment.py").read_text()
    lines = recipe.splitlines()
    assert lines[:4] == [
        "from priml.baselines.nanochat.experiments import NgramTrainLoop",
        "",
        "def experiment() -> NgramTrainLoop.Config:",
        '    """Run the prepared NanoChat experiment."""',
    ]
    assert lines[4].startswith("    return NgramTrainLoop.Config.deserialize(")
    assert lines[4].endswith(")")
    assert format_options == [False]


def test_encode_fragment_skips_empty_suffix_tokenization() -> None:
    tokenizer = _unigram()
    encoded = tokenizer.encode("a", add_special_tokens=False)
    with (
        patch.object(tokenizers.Tokenizer, "encode", return_value=encoded),
        patch.object(
            tokenizers.Tokenizer,
            "id_to_token",
            return_value=byte_alphabet()[ord("a")],
        ),
        patch.object(models.Unigram, "tokenize", autospec=True) as tokenize,
    ):
        assert prepare_data.encode_fragment(b"a", tokenizer=tokenizer) == encoded.ids

    tokenize.assert_not_called()


def test_encode_fragment_rejects_a_terminal_invalid_byte() -> None:
    with pytest.raises(
        ValueError,
        match=r"^Reference fragment has non-terminal invalid UTF-8\.$",
    ) as error:
        prepare_data.encode_fragment(b"\xff", tokenizer=_unigram())

    assert str(error.value) == "Reference fragment has non-terminal invalid UTF-8."


def test_encode_fragment_reports_unknown_token_exactly() -> None:
    tokenizer = _unigram()
    encoded = tokenizer.encode("a", add_special_tokens=False)

    with (
        patch.object(tokenizers.Tokenizer, "encode", return_value=encoded),
        patch.object(tokenizers.Tokenizer, "id_to_token", return_value=None),
        pytest.raises(
            ValueError,
            match=r"^Unigram emitted an unknown token\.$",
        ) as error,
    ):
        prepare_data.encode_fragment(b"a", tokenizer=tokenizer)

    assert str(error.value) == "Unigram emitted an unknown token."


def test_encode_fragment_reports_changed_bytes_exactly() -> None:
    alphabet = byte_alphabet()
    altered = dict(alphabet)
    altered[ord("a")], altered[ord("b")] = (
        altered[ord("b")],
        altered[ord("a")],
    )
    tokenizer = _unigram()
    encoded = tokenizer.encode("a", add_special_tokens=False)

    with (
        patch.object(
            prepare_data,
            "_BYTE_VALUES",
            {char: value for value, char in altered.items()},
        ),
        patch.object(tokenizers.Tokenizer, "encode", return_value=encoded),
        patch.object(
            tokenizers.Tokenizer,
            "id_to_token",
            return_value=alphabet[ord("a")],
        ),
        pytest.raises(
            ValueError,
            match=r"^Unigram replay changed the reference-selected bytes\.$",
        ) as error,
    ):
        prepare_data.encode_fragment(b"a", tokenizer=tokenizer)

    assert str(error.value) == "Unigram replay changed the reference-selected bytes."


def test_prepare_reference_rows_handles_document_boundary_after_first_fragment() -> (
    None
):
    reference = tiktoken.Encoding(
        name="reference-row-start-index",
        pat_str=r".+",
        mergeable_ranks={b"a": 0},
        special_tokens={"<|reserved_0|>": 1},
    )
    inputs = np.array([[1, 0]], dtype=np.int64)
    targets = np.array([[0, 1]], dtype=np.int64)

    replay = prepare_reference_rows(
        inputs,
        targets=targets,
        reference=reference,
        tokenizer=None,
        batch_size=1,
        reserved_count=16,
    )

    assert np.array_equal(replay["inputs"], inputs)
    assert np.array_equal(replay["targets"], targets)
    assert replay["fragment_lengths"].tolist() == [1, 0]


def test_preparation_reference_creates_nested_output_once_and_refuses_reuse(
    tmp_path: Path,
) -> None:
    config = Preparation.Config()
    config.working_dir = tmp_path / "nested" / "run"
    config.corpus.raw_dir = tmp_path / "raw"
    config.corpus.val_shard = 7
    config.rows.tokenizer.path = tmp_path / "unigram.json"
    tokenizer_dir = config.corpus.raw_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    pickled = tokenizer_dir / "tokenizer.pkl"
    pickled.write_bytes(b"fixture")
    encoder = Tokenizer.__new__(Tokenizer)
    encoder.token_bytes = np.array([1, 2], dtype=np.int32)
    batch = {
        "media": torch.tensor([[1, 2]], dtype=torch.int32),
        "label": torch.tensor([[2, 3]], dtype=torch.int32),
    }
    preparation = Preparation(config)

    with (
        patch.object(Tokenizer, "from_directory", return_value=encoder),
        patch.object(prepare_data, "PackedTokenStream", return_value=iter([batch])),
        patch.object(prepare_data, "build_reference_eval") as build_eval,
    ):
        preparation.reference()
        output = config.working_dir / "reference-bpe"
        assert np.array_equal(
            _load_array(output / "eval_x.npy"),
            np.array([[1, 2]], dtype=np.uint16),
        )
        assert np.array_equal(
            _load_array(output / "eval_y.npy"),
            np.array([[2, 3]], dtype=np.uint16),
        )
        assert (output / "tokenizer.pkl").read_bytes() == b"fixture"
        with pytest.raises(FileExistsError):
            preparation.reference()

    build_eval.assert_called_once()


def test_corpus_build_preserves_source_safety_and_parquet_recipe(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    source_train = raw / "shard_00000.parquet"
    source_val = raw / "shard_00001.parquet"
    _write_shard(raw, 0, ["donor-a", "donor-b", "keep"])
    _write_shard(raw, 1, ["heldout"])
    source_train_bytes = source_train.read_bytes()
    source_val_bytes = source_val.read_bytes()
    output = tmp_path / "nested" / "prepared"
    config = CorpusPreparation.Config()
    config.raw_dir = raw
    config.working_dir = output
    config.train_shard_indices = (0,)
    config.val_shard = 1
    config.donor_source_ids = ("0:0", "0:1")
    config.donor_destination_shards = 1
    config.row_group_size = 2
    config.compression = "zstd"
    caplog.set_level(
        "INFO",
        logger="priml.baselines.nanochat.scripts.prepare_data",
    )

    with (
        patch.object(
            prepare_data,
            "validated_output_path",
            wraps=validated_output_path,
        ) as validate,
        patch.object(
            tempfile,
            "TemporaryDirectory",
            wraps=tempfile.TemporaryDirectory,
        ) as temporary_directory,
        patch.object(
            parquet,
            "write_table",
            wraps=parquet.write_table,
        ) as write_table,
    ):
        CorpusPreparation(config).build()
        CorpusPreparation(config).build()

    assert validate.call_args_list == [
        call(output, protected=[raw]),
        call(output / source_train.name, protected=[source_train]),
        call(output / source_val.name, protected=[source_val]),
        call(output, protected=[raw]),
        call(output / source_train.name, protected=[source_train]),
        call(output / source_val.name, protected=[source_val]),
    ]
    assert [call.kwargs for call in temporary_directory.call_args_list] == [
        {"dir": output, "prefix": "nanochat-shard-"},
        {"dir": output, "prefix": "nanochat-copy-"},
        {"dir": output, "prefix": "nanochat-shard-"},
        {"dir": output, "prefix": "nanochat-copy-"},
    ]
    assert write_table.call_count == 2
    assert all(
        write.kwargs == {"row_group_size": 2, "compression": "zstd"}
        for write in write_table.call_args_list
    )
    written = parquet.ParquetFile(output / source_train.name)
    assert written.metadata.num_row_groups == 2
    assert written.metadata.row_group(0).column(0).compression == "ZSTD"
    assert _text_column(output / source_train.name) == ["donor-a", "donor-b", "keep"]
    assert (output / source_val.name).read_bytes() == source_val_bytes
    assert source_train.read_bytes() == source_train_bytes
    assert caplog.messages == [
        "Prepared corpus shard 0: 3 rows",
        "Prepared corpus shard 0: 3 rows",
    ]


def test_corpus_build_names_a_selected_noncanonical_donor(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_shard(raw, 0, ["duplicate", "duplicate", "keep"])
    _write_shard(raw, 1, ["heldout"])
    config = CorpusPreparation.Config()
    config.raw_dir = raw
    config.working_dir = tmp_path / "prepared"
    config.train_shard_indices = (0,)
    config.val_shard = 1
    config.donor_source_ids = ("0:1",)
    config.donor_destination_shards = 1

    with pytest.raises(
        ValueError,
        match=r"^A selected donor is not a canonical training document\.$",
    ) as error:
        CorpusPreparation(config).build()

    assert str(error.value) == "A selected donor is not a canonical training document."


def test_corpus_init_errors_are_exact() -> None:
    config = CorpusPreparation.Config()
    config.val_shard = 2
    config.train_shard_indices = (1, 2)
    with pytest.raises(
        ValueError,
        match=r"^Validation must be excluded from training\.$",
    ) as error:
        CorpusPreparation(config)
    assert str(error.value) == "Validation must be excluded from training."

    config.val_shard = 3
    config.train_shard_indices = (1, 1)
    with pytest.raises(
        ValueError,
        match=r"^Training shard identities must be unique\.$",
    ) as error:
        CorpusPreparation(config)
    assert str(error.value) == "Training shard identities must be unique."


def test_row_initialization_error_text_is_exact() -> None:
    config = RowPreparation.Config()
    config.val_shard = 2
    config.train_shard_indices = (1, 2)

    with pytest.raises(
        ValueError,
        match=r"^Validation must be excluded from training\.$",
    ) as error:
        RowPreparation(config)

    assert str(error.value) == "Validation must be excluded from training."


def test_donor_texts_reports_missing_identity_exactly(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_shard(raw, 0, ["present"])

    with pytest.raises(
        ValueError,
        match=r"^A donor identity is absent from its original shard\.$",
    ) as error:
        _donor_texts(raw, identities={"0:1"})

    assert str(error.value) == "A donor identity is absent from its original shard."


def test_pack_row_uses_remaining_row_capacity() -> None:
    row = np.full(6, -1, dtype=np.int64)
    buffer = [[10, 11, 12, 13, 14], [20, 21]]
    lengths = [5, 2]

    position = pack_row(row, buffer=buffer, lengths=lengths, position=2)

    assert position == 4
    assert row.tolist() == [-1, -1, 20, 21, -1, -1]
    assert buffer == [[10, 11, 12, 13, 14]]
    assert lengths == [5]


def test_training_config_leaves_absent_reference_evaluation_unchanged(
    tmp_path: Path,
) -> None:
    config = exp022()
    config.dataset.prepared_train_manifest = tmp_path / "train.json"
    config.dataset.prepared_eval_manifest = tmp_path / "eval.json"
    config.dataset.reference_evaluation = None
    preparation = Preparation(Preparation.Config())
    factory_module = ModuleType("experiments")
    factory_module.__dict__["exp022"] = lambda: config

    with (
        patch.object(
            importlib,
            "import_module",
            return_value=factory_module,
        ),
        patch.object(RowPreparation, "verify"),
    ):
        training = preparation.training_config(
            "example.exp022",
            run_directory=tmp_path / "run",
            seed=42,
        )

    assert training.dataset.reference_evaluation is None


def _corpus_config(raw: Path, output: Path) -> CorpusPreparation.Config:
    config = CorpusPreparation.Config()
    config.raw_dir = raw
    config.working_dir = output
    config.donor_destination_shards = 2
    return config


def test_a_noncanonical_donor_publishes_nothing(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_shard(raw, 0, ["same", "keep"])
    _write_shard(raw, 1, ["same"])
    _write_shard(raw, 2, ["heldout"])
    config = _corpus_config(raw, tmp_path / "prepared")
    config.train_shard_indices = (0, 1)
    config.val_shard = 2
    config.donor_source_ids = ("1:0",)  # A duplicate of 0:0, so not canonical.
    with pytest.raises(ValueError, match="not a canonical training document"):
        config.make().build()
    assert not config.working_dir.exists()


def test_donors_reach_destinations_whose_ids_exceed_the_count(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_shard(raw, 8, ["a", "b"])
    _write_shard(raw, 9, ["c", "donor"])
    _write_shard(raw, 2, ["heldout"])
    config = _corpus_config(raw, tmp_path / "prepared")
    config.train_shard_indices = (8, 9)
    config.val_shard = 2
    config.donor_source_ids = ("9:1",)
    config.make().build()
    written = [
        *_text_column(config.working_dir / "shard_00008.parquet"),
        *_text_column(config.working_dir / "shard_00009.parquet"),
    ]
    assert sorted(written) == ["a", "b", "c", "donor"]


def test_the_corpus_split_reaches_every_stage() -> None:
    config = Preparation.Config()
    config.corpus.val_shard = 8
    config.corpus.train_shard_indices = (0, 1, 7)
    config.finalize()
    for stage in (config.sample, config.rows):
        assert stage.val_shard == 8
        assert stage.train_shard_indices == (0, 1, 7)


def test_running_one_stage_builds_only_that_stage() -> None:
    config = Preparation.Config()
    config.sample.rows_per_shard = 0  # Invalid; must not block "fetch".
    with patch.object(prepare_data.CorpusPreparation, "fetch") as fetch:
        Preparation(config).run("fetch")
    fetch.assert_called_once_with()


def test_a_corrupted_byte_table_is_refitted(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    shard = tmp_path / "shard_00000.parquet"
    parquet.write_table(pa.table({"text": ["hello world 123", "café hello"]}), shard)
    output = tmp_path / "tokenizer"
    fit = partial(
        prepare_data._fit_vocabulary,
        [shard],
        out=output,
        vocab_size=280,
        train_chars=24,
        doc_cap=7,
    )
    fit()
    table = output / "token_bytes.npy"
    original = table.read_bytes()
    np.save(table, np.zeros(280, dtype=np.int32))
    caplog.set_level("WARNING", logger="rustbpe")
    caplog.set_level("INFO", logger=prepare_data.__name__)
    fit()
    assert "byte table differs from its recorded digest" in caplog.text
    assert table.read_bytes() == original


def test_prepare_refuses_to_write_over_a_protected_root() -> None:
    with pytest.raises(ValueError, match="must not be root"):
        prepare(Path("/"), download=False)


@pytest.mark.parametrize("vocab_size", [65_537, 70_000])
def test_prepare_refuses_ids_a_uint16_row_cannot_hold(
    tmp_path: Path,
    vocab_size: int,
) -> None:
    with pytest.raises(ValueError, match="vocab_size must be at most 65536"):
        prepare(tmp_path, vocab_size=vocab_size, download=False)


def test_prepared_rows_refuse_a_vocabulary_beyond_uint16() -> None:
    prepare_data._require_uint16_ids(np.zeros(65_536))
    with pytest.raises(ValueError, match="exceeds 65536"):
        prepare_data._require_uint16_ids(np.zeros(65_537))


def test_the_source_revision_is_a_commit() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", CorpusPreparation.Config().revision)


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
