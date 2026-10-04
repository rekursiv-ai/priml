"""Verify fitting text order, deterministic windows, and byte tokenizers."""

from collections.abc import Callable, Iterable
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, cast
from unittest.mock import Mock

import logging
import re
import tempfile

from pyarrow import Table, parquet
from tokenizers import AddedToken, Tokenizer, normalizers, processors

import numpy as np
import pytest

from priml.baselines.nanochat.data import DEFAULT_ENCODE_THREADS
from priml.baselines.nanochat.scripts import prepare_tokenizer
from priml.baselines.nanochat.scripts.prepare_tokenizer import (
    ByteLevelTokenizer,
    JsonValue,
    SamplePreparation,
    UnigramPreparation,
    _pruned_piece_indices,
    byte_alphabet,
    document_rows,
    frequency_model,
    load_sample,
    read_mapping,
    sample_window,
    usage_scores,
    write_mapping,
)


def test_mapping_stages_in_destination_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "nested" / "mapping.json"
    named_temporary_file = Mock(wraps=tempfile.NamedTemporaryFile)
    monkeypatch.setattr(
        tempfile,
        "NamedTemporaryFile",
        named_temporary_file,
    )

    write_mapping(path, value={"valid": True})

    assert named_temporary_file.call_args is not None
    assert named_temporary_file.call_args.kwargs["dir"] == path.parent
    assert named_temporary_file.call_args.kwargs["delete"] is False
    assert named_temporary_file.call_args.kwargs["prefix"] == "nanochat-json-"


def test_mapping_io_and_usage_scores(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "mapping.json"
    value: JsonValue = {"z": 1, "a": [True, None]}
    write_mapping(path, value=value)
    write_mapping(path, value=value)
    assert path.read_text() == '{\n  "a": [\n    true,\n    null\n  ],\n  "z": 1\n}\n'
    assert read_mapping(path) == {"a": [True, None], "z": 1}
    path.write_text("[]")
    with pytest.raises(TypeError, match=r"cannot coerce .* to dict"):
        read_mapping(path)
    scores = usage_scores([b"a", b"b"], [0, 4])
    assert scores == [0.0, 4.0]
    assert all(type(score) is float for score in scores)
    with pytest.raises(
        ValueError,
        match=r"^Expected len\(pieces\) == len\(counts\)\.$",
    ):
        usage_scores([b"a"], [])
    for invalid_number in (float("nan"), float("inf")):
        message = "JSON numbers must be finite."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            write_mapping(path, value={"invalid": [invalid_number]})

    # The message is compared exactly with ``str``: ``match`` also reads the "when
    # serializing ..." notes Python 3.14's encoder adds, and 3.12, where the public
    # package runs, adds none.
    circular = "Circular reference detected"
    recursive_value: list[prepare_tokenizer.JsonValue] = []
    recursive_value.append(recursive_value)
    with pytest.raises(ValueError, match=f"^{circular}") as recursive_list:
        write_mapping(path, value={"recursive": recursive_value})
    assert str(recursive_list.value) == circular

    recursive_mapping: dict[str, JsonValue] = {}
    recursive_mapping["recursive"] = recursive_mapping
    with pytest.raises(ValueError, match=f"^{circular}") as recursive_dict:
        write_mapping(path, value=recursive_mapping)
    assert str(recursive_dict.value) == circular


def test_pruned_piece_indices_keep_every_byte_and_rank_learned_pieces() -> None:
    earlier_wins = [10.0] * 256 + [9.0, 1.0]
    later_wins = [10.0] * 256 + [1.0, 9.0]

    assert _pruned_piece_indices(earlier_wins, ordinary=257) == [
        *range(256),
        256,
    ]
    assert _pruned_piece_indices(later_wins, ordinary=257) == [
        *range(256),
        257,
    ]
    assert _pruned_piece_indices(later_wins, ordinary=258) == list(range(258))


def test_document_rows_preserves_source_order_and_identity(tmp_path: Path) -> None:
    path = tmp_path / "shard_00003.parquet"
    parquet.write_table(
        Table.from_pydict(
            {"ignored": [1, 2, 3], "text": ["first", "café", "last"]},
        ),
        path,
    )
    assert list(document_rows(path, shard=3)) == [
        ("3:0", "first"),
        ("3:1", "café"),
        ("3:2", "last"),
    ]


def test_document_rows_preserves_offsets_across_configured_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_sizes: list[int] = []

    class Batch:
        def __init__(self, rows: list[str]) -> None:
            self.rows = rows

        def column(self, index: int) -> SimpleNamespace:
            assert index == 0
            return SimpleNamespace(to_pylist=lambda: self.rows)

    class ParquetFile:
        def __init__(self, path: Path) -> None:
            assert path == tmp_path / "source.parquet"

        def iter_batches(
            self,
            *,
            batch_size: int,
            columns: list[str],
        ) -> list[Batch]:
            batch_sizes.append(batch_size)
            assert columns == ["text"]
            return [Batch(["a", "b"]), Batch(["c"])]

    monkeypatch.setattr(
        prepare_tokenizer,
        "parquet",
        SimpleNamespace(ParquetFile=ParquetFile),
    )

    assert list(document_rows(tmp_path / "source.parquet", shard=7)) == [
        ("7:0", "a"),
        ("7:1", "b"),
        ("7:2", "c"),
    ]
    assert batch_sizes == [1_024]


def test_sample_preserves_text_and_order_without_provenance(tmp_path: Path) -> None:
    texts = ["A café 🦙", " second document\n", "A café 🦙"]
    parquet.write_table(
        Table.from_pydict({"text": texts}),
        tmp_path / "sample.parquet",
    )
    assert load_sample(tmp_path) == texts


def test_utf8_windows_and_byte_vocabulary() -> None:
    raw = ("é🦙abc" * 9).encode()
    start, end = sample_window(raw, max_bytes=9)
    assert (start, end) == (6, 15)
    assert raw[start:end].decode().encode() == raw[start:end]
    assert sample_window(b"abc", max_bytes=3) == (0, 3)
    alphabet = byte_alphabet()
    assert len(alphabet) == len(set(alphabet.values())) == 256
    assert [
        alphabet[value] for value in (32, 33, 126, 127, 160, 161, 172, 173, 174, 255)
    ] == [
        "Ġ",
        "!",
        "~",
        "ġ",
        "ł",
        "¡",
        "¬",
        "Ń",
        "®",
        "ÿ",
    ]
    model = frequency_model(
        [b"a", b"ab", b"b", b" "],
        counts=[0, 10, 2, 0],
        split_pattern=r"\S+|\s+",
    )
    encoded = model.encode("ab a b")
    assert encoded.ids == [1, 3, 0, 3, 2]
    assert model.decode(encoded.ids) == "ab a b"
    assert model.get_vocab() == {"a": 0, "ab": 1, "b": 2, "Ġ": 3}
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter"):
        frequency_model([b"a", b"b"], counts=[1], split_pattern=r"\S+")
    weighted = frequency_model(
        [b"ab", b"a", b"b"],
        counts=[0, 3, 3],
        split_pattern=r"\S+",
    )
    assert weighted.encode("ab").ids == [1, 2]
    spaced = frequency_model(
        [b"a b", b"a", b" ", b"b"],
        counts=[100, 1, 1, 1],
        split_pattern=r".+",
    )
    assert spaced.encode("a b").ids == [0]


def test_frequency_model_scores_unseen_pieces_with_floored_counts() -> None:
    model = frequency_model(
        [b"ab", b"a", b"b", b"z"],
        counts=[1, 3, 3, 0],
        split_pattern=r".+",
    )

    assert model.encode("ab").ids == [1, 2]


def test_frequency_model_rejects_unknown_bytes_without_unknown_id() -> None:
    model = frequency_model([b"a", b"ab"], counts=[1, 2], split_pattern=r".+")
    with pytest.raises(
        Exception,
        match=r"^Encountered an unknown token but `unk_id` is missing$",
    ) as error:
        model.encode("z")
    assert type(error.value) is Exception


def _byte_model() -> Tokenizer:
    return frequency_model(
        [bytes([value]) for value in range(256)],
        counts=[1] * 256,
        split_pattern=r"\S+|\s+",
    )


def test_byte_level_tokenizer_prepends_bos_and_round_trips(tmp_path: Path) -> None:
    alphabet = byte_alphabet()
    model = _byte_model()
    path = tmp_path / "tokenizer.json"
    model.save(str(path))
    config = ByteLevelTokenizer.Config()
    config.path = path
    tokenizer = config.make()

    text = "café 🦙"
    encoded = tokenizer.encode_batch(["", text])
    ids = encoded[1]
    assert encoded[0] == [tokenizer.bos_token_id]
    assert (
        cast(
            int,
            signature(ByteLevelTokenizer.encode_batch)
            .parameters["num_threads"]
            .default,
        )
        == DEFAULT_ENCODE_THREADS
    )
    assert ids[0] == tokenizer.bos_token_id
    assert tokenizer.backend.decode(ids[1:]) == text
    assert len(alphabet) == 256
    assert tokenizer.vocab_size == 272
    assert str(tokenizer.token_bytes.dtype) == "int32"
    assert str(tokenizer.token_bytes_literal.dtype) == "int32"
    assert sum(tokenizer.token_bytes_literal[ids[1:]]) == len(text.encode())


def test_byte_level_tokenizer_rejects_missing_piece_with_exact_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    vocab_flags: list[bool] = []

    class Backend:
        def get_vocab_size(self, *, with_added_tokens: bool) -> int:
            vocab_flags.append(with_added_tokens)
            return 1

        def id_to_token(self, token_id: int) -> None:
            assert token_id == 0

    backend = Backend()

    def from_file(path: str) -> Backend:
        calls.append(path)
        return backend

    monkeypatch.setattr(
        prepare_tokenizer,
        "tokenizers",
        SimpleNamespace(Tokenizer=SimpleNamespace(from_file=from_file)),
    )

    def read_mapping(_: Path) -> dict[str, object]:
        return {}

    monkeypatch.setattr(prepare_tokenizer, "read_mapping", read_mapping)

    def zeros(size: int, *, dtype: str) -> list[int]:
        assert size == 17
        assert dtype == "int32"
        return []

    monkeypatch.setattr(prepare_tokenizer, "zeros", zeros)
    config = ByteLevelTokenizer.Config()
    config.path = tmp_path / "tokenizer.json"

    with pytest.raises(ValueError, match=r"^Expected piece is not None\.$"):
        config.make()

    assert calls == [str(config.path)]
    assert vocab_flags == [False]


def test_byte_level_tokenizer_rejects_normalization(tmp_path: Path) -> None:
    model = _byte_model()
    model.normalizer = normalizers.NFC()
    path = tmp_path / "normalized.json"
    model.save(str(path))
    config = ByteLevelTokenizer.Config()
    config.path = path

    with pytest.raises(
        ValueError,
        match=r"^Expected ordinary byte pieces without normalization or added tokens\.$",
    ):
        config.make()


def test_byte_level_tokenizer_rejects_postprocessing(tmp_path: Path) -> None:
    model = _byte_model()
    model.post_processor = processors.ByteLevel(trim_offsets=False)
    path = tmp_path / "postprocessed.json"
    model.save(str(path))
    config = ByteLevelTokenizer.Config()
    config.path = path

    with pytest.raises(
        ValueError,
        match=r"^Expected ordinary byte pieces without normalization or added tokens\.$",
    ):
        config.make()


def test_byte_level_tokenizer_rejects_added_tokens(tmp_path: Path) -> None:
    model = _byte_model()
    add_special_tokens = cast(
        Callable[[list[AddedToken]], int],
        model.add_special_tokens,
    )
    add_special_tokens([AddedToken("<special>")])
    path = tmp_path / "special.json"
    model.save(str(path))
    config = ByteLevelTokenizer.Config()
    config.path = path

    with pytest.raises(
        ValueError,
        match=r"^Expected ordinary byte pieces without normalization or added tokens\.$",
    ):
        config.make()


def test_byte_level_encode_batch_requires_aligned_literal_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _byte_model()
    path = tmp_path / "tokenizer.json"
    model.save(str(path))
    config = ByteLevelTokenizer.Config()
    config.path = path
    tokenizer = config.make()
    token_id = tokenizer.backend.encode("x", add_special_tokens=False).ids[0]

    class BadDecodeBackend:
        def encode_batch(
            self,
            texts: list[str],
            *,
            add_special_tokens: bool,
        ) -> list[SimpleNamespace]:
            assert texts == ["x"]
            assert add_special_tokens is False
            return [SimpleNamespace(ids=[token_id])]

        def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
            assert ids == [token_id]
            assert skip_special_tokens is False
            return "not x"

    monkeypatch.setattr(tokenizer, "backend", BadDecodeBackend())
    with pytest.raises(
        ValueError,
        match=r"^The tokenizer failed literal document round-trip\.$",
    ):
        tokenizer.encode_batch(["x"])

    class ShortBackend:
        def encode_batch(
            self,
            texts: list[str],
            *,
            add_special_tokens: bool,
        ) -> list[SimpleNamespace]:
            assert texts == ["x"]
            assert add_special_tokens is False
            return []

    monkeypatch.setattr(tokenizer, "backend", ShortBackend())
    with pytest.raises(
        ValueError,
        match=r"^zip\(\) argument 2 is shorter than argument 1$",
    ):
        tokenizer.encode_batch(["x"])


def test_empty_utf8_window_is_empty() -> None:
    assert sample_window(b"", max_bytes=4) == (0, 0)
    assert sample_window(b"abcd", max_bytes=4) == (0, 4)
    assert sample_window(("é🦙abc" * 3).encode(), max_bytes=8) == (2, 9)
    assert sample_window(b"a", max_bytes=4) == (0, 1)
    assert sample_window("é🦙".encode(), max_bytes=4) == (2, 6)
    for repeats in range(1, 10):
        raw = ("é🦙abc" * repeats).encode()
        start, end = sample_window(raw, max_bytes=7)
        assert start == len(raw) or raw[start] & 0xC0 != 0x80
        assert end == len(raw) or raw[end] & 0xC0 != 0x80


def test_sample_preparation_rejects_validation_shard_leakage() -> None:
    config = SamplePreparation.Config()
    config.train_shard_indices = (config.val_shard,)
    with pytest.raises(
        ValueError,
        match=r"^The fitting sample must exclude validation shard7\.$",
    ):
        config.make()


def test_sample_preparation_rejects_invalid_size_boundaries() -> None:
    rows = SamplePreparation.Config()
    rows.rows_per_shard = 0
    with pytest.raises(
        ValueError,
        match=r"^Sample size must be positive and windows at least four bytes\.$",
    ):
        rows.make()

    single_row = SamplePreparation.Config()
    single_row.rows_per_shard = 1
    single_row.make()

    window = SamplePreparation.Config()
    window.max_bytes = 3
    with pytest.raises(
        ValueError,
        match=r"^Sample size must be positive and windows at least four bytes\.$",
    ):
        window.make()


def test_unigram_preparation_requires_every_ordinary_byte() -> None:
    config = UnigramPreparation.Config()
    config.vocab_size = 257
    message = "The ordinary vocabulary must contain every byte."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        config.make()


def test_preparations_protect_their_input_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = SamplePreparation.Config()
    sample.raw_dir = tmp_path / "sample"
    sample.working_dir = sample.raw_dir
    with pytest.raises(
        ValueError,
        match=(
            "^"
            + re.escape(
                f"output path aliases protected input artifact: {sample.raw_dir.absolute()}",
            )
            + "$"
        ),
    ):
        sample.make().build()

    unigram = UnigramPreparation.Config()
    unigram.sample_dir = tmp_path / "unigram"
    unigram.working_dir = unigram.sample_dir

    def load_texts(_: Path) -> list[str]:
        return ["text"]

    monkeypatch.setattr(prepare_tokenizer, "load_sample", load_texts)
    with pytest.raises(
        ValueError,
        match=(
            "^"
            + re.escape(
                f"output path aliases protected input artifact: {unigram.sample_dir.absolute()}",
            )
            + "$"
        ),
    ):
        unigram.make().build()


def test_sample_preparation_writes_stratified_source_rows(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    parquet.write_table(
        Table.from_pydict({"text": ["zero", "one", "two", "tri", "four"]}),
        raw_dir / "shard_00000.parquet",
    )
    config = SamplePreparation.Config()
    config.raw_dir = raw_dir
    config.working_dir = tmp_path / "nested" / "sample"
    config.train_shard_indices = (0,)
    config.val_shard = 1
    config.rows_per_shard = 2
    config.max_bytes = 4

    sample_preparation = config.make()
    sample_preparation.build()

    assert load_sample(config.working_dir) == ["one", "tri"]
    with pytest.raises(FileExistsError):
        sample_preparation.build()


def test_unigram_preparation_writes_tokenizer_and_counts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    parquet.write_table(
        Table.from_pydict({"text": ["ab cab", "cab ab"]}),
        sample_dir / "sample.parquet",
    )
    config = UnigramPreparation.Config()
    config.sample_dir = sample_dir
    config.working_dir = tmp_path / "nested" / "unigram"
    config.vocab_size = 258
    config.reserved_count = 2
    config.num_passes = 1
    config.batch_size = 2
    config.num_threads = 1
    # Rustbpe logs its merge progress too when the root level admits INFO. Set
    # first: the last ``set_level`` also fixes the capture handler's level.
    caplog.set_level(logging.WARNING, logger="rustbpe")
    caplog.set_level(logging.INFO, logger=prepare_tokenizer.__name__)

    config.make().build()

    tokenizer_config = ByteLevelTokenizer.Config()
    tokenizer_config.path = config.working_dir / "tokenizer.json"
    tokenizer = tokenizer_config.make()
    ids = tokenizer.encode_batch(["ab cab"])[0]
    assert tokenizer.backend.decode(ids[1:]) == "ab cab"
    assert tokenizer.bos_token_id == 256
    assert tokenizer.vocab_size == 272
    counts = cast(
        "np.ndarray[tuple[int, ...], np.dtype[np.int64]]",
        np.load(
            config.working_dir / "counts.npy",
        ),
    )
    assert counts.dtype == np.int64
    expected_counts = [0] * 256
    expected_counts[ord(" ")] = 2
    expected_counts[ord("a")] = 4
    expected_counts[ord("b")] = 4
    expected_counts[ord("c")] = 2
    assert counts.tolist() == expected_counts
    assert caplog.messages == ["Hard-EM pass 1: 12 tokens"]
    assert (config.working_dir / "tokenizer.json").is_file()
    with pytest.raises(FileExistsError):
        config.make().build()


def test_unigram_build_batches_prunes_and_saves_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = ["a", "b", "c"]
    trainer_calls: list[tuple[list[str], int, str]] = []
    seed_batches: list[tuple[list[str], int]] = []
    seed_calls: list[dict[str, object]] = []
    model_batches: list[list[str]] = []
    fitted: list[tuple[list[bytes], list[int]]] = []

    class Saved(TypedDict):
        tokenizer_path: Path
        counts_path: Path
        final_pieces: list[bytes]
        final_counts: list[int]
        array: tuple[list[int], str]

    saved: Saved = {
        "tokenizer_path": Path(),
        "counts_path": Path(),
        "final_pieces": [],
        "final_counts": [],
        "array": ([], ""),
    }

    class Trainer:
        def train_from_iterator(
            self,
            values: Iterable[str],
            vocab_size: int,
            *,
            pattern: str,
        ) -> None:
            trainer_calls.append((list(values), vocab_size, pattern))

        def get_mergeable_ranks(self) -> list[tuple[bytes, int]]:
            return [
                *((bytes([value]), value) for value in range(256)),
                (b"ab", 256),
                (b"bc", 257),
            ]

    class Seed:
        def encode_ordinary_batch(
            self,
            batch: list[str],
            *,
            num_threads: int,
        ) -> list[list[int]]:
            seed_batches.append((batch, num_threads))
            return [[256] if text != "c" else [257] for text in batch]

    class HardEmModel:
        def __init__(self, pass_index: int) -> None:
            self.pass_index = pass_index

        def encode_batch(
            self,
            batch: list[str],
            *,
            add_special_tokens: bool,
        ) -> list[SimpleNamespace]:
            assert add_special_tokens is False
            model_batches.append(batch)
            ids = [256] if self.pass_index == 0 else [257]
            return [SimpleNamespace(ids=ids) for _ in batch]

        def save(self, path: str) -> None:
            saved["tokenizer_path"] = Path(path)

    frequency_calls = 0

    def build_model(
        pieces: list[bytes],
        *,
        counts: list[int],
        split_pattern: str,
    ) -> HardEmModel:
        nonlocal frequency_calls
        assert split_pattern == config.split_pattern
        fitted.append((pieces, counts))
        frequency_calls += 1
        if frequency_calls <= 2:
            return HardEmModel(frequency_calls - 1)
        saved["final_pieces"] = pieces
        saved["final_counts"] = counts
        return HardEmModel(-1)

    def prune(pieces: list[bytes], counts: list[int]) -> list[float]:
        assert len(pieces) == len(counts) == 258
        assert counts[-2:] == [0, 3]
        return [10.0] * 256 + [1.0, 9.0]

    def load_texts(_: Path) -> list[str]:
        return list(texts)

    monkeypatch.setattr(prepare_tokenizer, "load_sample", load_texts)
    monkeypatch.setattr(
        prepare_tokenizer,
        "rustbpe",
        SimpleNamespace(Tokenizer=Trainer),
    )

    def build_seed(**kwargs: object) -> Seed:
        seed_calls.append(kwargs)
        return Seed()

    monkeypatch.setattr(
        prepare_tokenizer,
        "tiktoken",
        SimpleNamespace(Encoding=build_seed),
    )

    def build_array(values: list[int], *, dtype: str) -> tuple[list[int], str]:
        return values, dtype

    def save_result(path: str, value: tuple[list[int], str]) -> None:
        saved.update(counts_path=Path(path), array=value)

    monkeypatch.setattr(prepare_tokenizer, "frequency_model", build_model)
    monkeypatch.setattr(prepare_tokenizer, "array", build_array)
    monkeypatch.setattr(prepare_tokenizer, "save", save_result)

    config = UnigramPreparation.Config()
    config.sample_dir = tmp_path / "sample"
    config.working_dir = tmp_path / "nested" / "unigram"
    config.vocab_size = 259
    config.reserved_count = 2
    config.overshoot_learned = 1.5
    config.num_passes = 2
    config.batch_size = 2
    config.num_threads = 3
    config.pruning = prune
    config.make().build()

    assert trainer_calls == [(texts, 258, config.split_pattern)]
    assert seed_calls[0]["name"] == "nanochat-reproduction-seed"
    assert seed_calls[0]["pat_str"] == config.split_pattern
    assert seed_calls[0]["special_tokens"] == {}
    assert seed_calls[0]["mergeable_ranks"] == {
        **{bytes([value]): value for value in range(256)},
        b"ab": 256,
        b"bc": 257,
    }
    assert seed_batches == [(["a", "b"], 3), (["c"], 3)]
    assert model_batches == [["a", "b"], ["c"], ["a", "b"], ["c"]]
    assert fitted[0][0][-2:] == [b"ab", b"bc"]
    assert fitted[0][1][-2:] == [2, 1]
    assert saved["tokenizer_path"] == config.working_dir / "tokenizer.json"
    assert saved["counts_path"] == config.working_dir / "counts.npy"
    assert len(saved["final_pieces"]) == 257
    assert saved["final_pieces"][-1] == b"bc"
    assert len(saved["final_counts"]) == 257
    assert saved["final_counts"][-1] == 3
    array_values, dtype = saved["array"]
    assert dtype == "int64"
    assert array_values[-1] == 3


def test_encode_batch_uses_literal_backend_settings_and_rejects_bad_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = frequency_model(
        [bytes([value]) for value in range(256)],
        counts=[1] * 256,
        split_pattern=r"\\S+|\\s+",
    )
    path = tmp_path / "tokenizer.json"
    model.save(str(path))
    config = ByteLevelTokenizer.Config()
    config.path = path
    tokenizer = config.make()
    backend = tokenizer.backend
    calls: list[tuple[list[str], bool]] = []

    class Backend:
        def encode_batch(
            self,
            texts: list[str],
            *,
            add_special_tokens: bool,
        ) -> list[object]:
            calls.append((texts, add_special_tokens))
            return list(
                backend.encode_batch(texts, add_special_tokens=add_special_tokens),
            )

        def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
            return backend.decode(ids, skip_special_tokens=skip_special_tokens)

    monkeypatch.setattr(tokenizer, "backend", Backend())
    texts = ["", "café 🦙"]
    rows = tokenizer.encode_batch(texts)
    assert calls == [(texts, False)]
    assert rows == [
        [tokenizer.bos_token_id],
        [tokenizer.bos_token_id, *backend.encode(texts[1]).ids],
    ]

    monkeypatch.setattr(tokenizer, "backend", backend)
    token_id = backend.encode("x", add_special_tokens=False).ids[0]
    tokenizer.token_bytes_literal[token_id] = 0
    with pytest.raises(
        ValueError,
        match=r"^Token pieces do not conserve literal UTF-8 bytes\.$",
    ):
        tokenizer.encode_batch(["x"])


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
