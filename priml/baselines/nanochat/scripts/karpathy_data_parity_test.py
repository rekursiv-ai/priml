"""Tests for exact upstream token-stream parity helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import importlib
import sys
import types

from torch import Tensor

import pytest
import torch

from priml.baselines.nanochat.scripts import karpathy_data_parity
from priml.baselines.nanochat.scripts.karpathy_data_parity import (
    compare,
    compare_stream,
)
from priml.lib.testing.cli import assert_help_without_docstring


if TYPE_CHECKING:
    from collections.abc import Iterator

    from priml.baselines.nanochat.data import NanoChatBatch


def test_load_upstream_imports_and_points_to_the_supplied_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reference"
    corpus = tmp_path / "corpus"

    def tokenizer_from_directory(path: str) -> None:
        del path

    def make_dataloader(*args: object) -> Iterator[tuple[Tensor, Tensor, int]]:
        del args
        return iter(())

    module = types.ModuleType("prepare")
    module.__dict__.update(
        {
            "__file__": str(root / "prepare.py"),
            "DATA_DIR": "original-data",
            "TOKENIZER_DIR": "original-tokenizer",
            "EVAL_TOKENS": 8,
            "Tokenizer": types.SimpleNamespace(from_directory=tokenizer_from_directory),
            "make_dataloader": make_dataloader,
        },
    )
    imports: list[str] = []

    def import_module(name: str) -> types.ModuleType:
        imports.append(name)
        return module

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    loaded = karpathy_data_parity.load_upstream(root, corpus=corpus)
    assert imports == ["prepare"]
    assert sys.path[0] == str(root)
    assert str(corpus) == loaded.DATA_DIR
    assert str(corpus / "tokenizer") == loaded.TOKENIZER_DIR


def test_load_upstream_rejects_a_module_outside_the_parity_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def import_module(name: str) -> types.ModuleType:
        return types.ModuleType(name)

    monkeypatch.setattr(importlib, "import_module", import_module)

    with pytest.raises(TypeError) as exc_info:
        karpathy_data_parity.load_upstream(tmp_path, corpus=tmp_path / "corpus")

    assert str(exc_info.value) == "prepare module does not satisfy the parity contract"


class _ConfigStub:
    def make(self) -> _ConfigStub:
        return self


class _DataStub:
    Config = _ConfigStub


class _PackedStub:
    def __iter__(self) -> Iterator[_PackedStub]:
        return iter((self,))

    def __len__(self) -> int:
        return 2


class _ParsedArgs(Protocol):
    clone: Path
    corpus: Path
    batches: int
    rows: int
    num_train_shards: int
    device: str


class _Arguments:
    clone: Path
    corpus: Path
    batches: int
    rows: int
    num_train_shards: int
    device: str

    def __init__(self) -> None:
        self.clone = Path("reference")
        self.corpus = Path("corpus")
        self.batches = 1
        self.rows = 2
        self.num_train_shards = 5
        self.device = "cpu"


class _TokenizerStub:
    def get_vocab_size(self) -> int:
        return 8

    def get_bos_token_id(self) -> int:
        return 1


class _TokenizerFactoryStub:
    def __init__(self, tokenizer: _TokenizerStub) -> None:
        self.tokenizer = tokenizer

    def from_directory(self, path: str) -> _TokenizerStub:
        if path != "corpus/tokenizer":
            raise AssertionError(path)
        return self.tokenizer


class _UpstreamPrepareStub:
    def __init__(
        self,
        tokenizer: _TokenizerStub,
        loader_calls: list[tuple[object, ...]],
        *,
        eval_tokens: int,
    ) -> None:
        self.__file__ = "reference/prepare.py"
        self.DATA_DIR = "corpus"
        self.TOKENIZER_DIR = "corpus/tokenizer"
        self.EVAL_TOKENS = eval_tokens
        self.Tokenizer = _TokenizerFactoryStub(tokenizer)
        self.loader_calls = loader_calls

    def make_dataloader(
        self,
        *args: object,
    ) -> Iterator[tuple[Tensor, Tensor, int]]:
        self.loader_calls.append(args)
        return iter(())


def test_build_ours_prepares_and_configures_the_supplied_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared: list[tuple[Path, int]] = []

    def record_prepare(corpus: Path, *, num_train_shards: int) -> None:
        prepared.append((corpus, num_train_shards))

    monkeypatch.setattr(karpathy_data_parity, "NanoChatData", _DataStub)
    monkeypatch.setattr(karpathy_data_parity, "prepare", record_prepare)
    result = karpathy_data_parity.build_ours(
        corpus=tmp_path / "corpus",
        rows=3,
        device="cpu",
        num_train_shards=5,
    )
    assert prepared == [(tmp_path / "corpus", 5)]
    assert vars(result) == {
        "base_dir": "/",
        "working_dir": str(tmp_path / "corpus"),
        "device": "cpu",
        "batch_size": 3,
        "eval_batch_size": 3,
        "num_train_shards": 5,
        "val_shard": 5,
    }


def test_parse_args_displays_exact_script_and_option_help(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["karpathy_data_parity.py", "--help"])
    parser_factory = argparse.ArgumentParser
    parsers: list[argparse.ArgumentParser] = []

    def create_parser(
        *,
        description: str,
        formatter_class: type[argparse.HelpFormatter],
    ) -> argparse.ArgumentParser:
        parser = parser_factory(
            description=description,
            formatter_class=formatter_class,
        )
        parsers.append(parser)
        return parser

    monkeypatch.setattr(
        karpathy_data_parity,
        "argparse",
        types.SimpleNamespace(
            ArgumentParser=create_parser,
            RawDescriptionHelpFormatter=argparse.RawDescriptionHelpFormatter,
        ),
    )
    with pytest.raises(SystemExit) as exit_info:
        karpathy_data_parity._parse_args()
    assert exit_info.value.code == 0
    assert (
        parsers[0].description == (karpathy_data_parity.__doc__ or "").split("\n", 2)[2]
    )
    help_text = capsys.readouterr().out
    assert (
        "Prove this package's dataloaders emit the reference's exact token stream."
        in help_text
    )
    assert next(
        line for line in help_text.splitlines() if line.startswith("  --batches")
    ).endswith("Batches per split.")
    assert next(
        line for line in help_text.splitlines() if line.startswith("  --rows")
    ).endswith("Rows per batch.")
    assert next(
        line for line in help_text.splitlines() if line.startswith("  --device")
    ).endswith("Device batches land on.")


def test_main_help_works_without_a_module_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_help_without_docstring(
        monkeypatch,
        karpathy_data_parity,
        karpathy_data_parity._parse_args,
    )


def test_parse_args_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["karpathy_data_parity.py"])
    defaults = cast(_ParsedArgs, karpathy_data_parity._parse_args())
    assert defaults.clone == Path("/opt/scratch/karpathy-autoresearch")
    assert defaults.corpus == Path("/opt/scratch/datasets/nanochat")
    assert (defaults.batches, defaults.rows, defaults.num_train_shards) == (8, 4, 7)
    assert defaults.device == "cuda"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "karpathy_data_parity.py",
            "--clone",
            "reference",
            "--corpus",
            "corpus",
            "--batches",
            "2",
            "--rows",
            "3",
            "--num-train-shards",
            "5",
            "--device",
            "cpu",
        ],
    )
    changed = cast(_ParsedArgs, karpathy_data_parity._parse_args())
    assert changed.clone == Path("reference")
    assert changed.corpus == Path("corpus")
    assert (changed.batches, changed.rows, changed.num_train_shards) == (2, 3, 5)
    assert changed.device == "cpu"


def test_compare_requires_exact_shapes_and_tokens() -> None:
    tokens = torch.tensor([[1, 2], [3, 4]])
    assert compare("train", tokens, tokens.clone()) is None
    assert compare("train", tokens, torch.tensor([[1, 2]])) == (
        "train: SHAPE (2, 2) vs (1, 2)"
    )
    assert compare("train", tokens, torch.tensor([[1, 2], [3, 5]])) == (
        "train: DIFFERS at (1, 1) (4 vs 5), 1/4 positions"
    )


def test_compare_stream_reports_media_mismatch_by_name() -> None:
    theirs = iter([(torch.tensor([[1, 2]]), torch.tensor([[3, 4]]), 0)])
    ours: list[NanoChatBatch] = [
        {
            "media": torch.tensor([[2, 1]]),
            "label": torch.tensor([[3, 4]]),
            "token_bytes": torch.ones(8),
            "valid_count": 1,
        },
    ]
    assert compare_stream(theirs, iter(ours), batches=1, tag="train") == [
        "train[1] media: DIFFERS at (0, 0) (1 vs 2), 2/2 positions",
    ]


@pytest.mark.parametrize(
    ("ours_vocab", "ours_bos", "message"),
    [
        (7, 1, "vocabularies differ: theirs 8, ours 7"),
        (8, 2, "document-start token differs: theirs 1, ours 2"),
    ],
)
def test_main_rejects_tokenizer_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    ours_vocab: int,
    ours_bos: int,
    message: str,
) -> None:
    upstream = _UpstreamPrepareStub(_TokenizerStub(), [], eval_tokens=12)

    def load_upstream(root: Path, *, corpus: Path) -> _UpstreamPrepareStub:
        del root, corpus
        return upstream

    def build_ours(**kwargs: object) -> types.SimpleNamespace:
        del kwargs
        return types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(
                vocab_size=ours_vocab,
                bos_token_id=ours_bos,
            ),
        )

    def clone_upstream(root: Path) -> Path:
        return root

    monkeypatch.setattr(karpathy_data_parity, "_parse_args", _Arguments)
    monkeypatch.setattr(karpathy_data_parity, "clone_upstream", clone_upstream)
    monkeypatch.setattr(karpathy_data_parity, "load_upstream", load_upstream)
    monkeypatch.setattr(karpathy_data_parity, "build_ours", build_ours)

    with pytest.raises(RuntimeError) as exc_info:
        karpathy_data_parity.main()

    assert str(exc_info.value) == message


def test_main_reports_identical_streams_without_network_or_disk(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tokenizer = _TokenizerStub()
    loader_calls: list[tuple[object, ...]] = []
    upstream = _UpstreamPrepareStub(tokenizer, loader_calls, eval_tokens=6)
    packed = _PackedStub
    data = types.SimpleNamespace(
        tokenizer=types.SimpleNamespace(vocab_size=8, bos_token_id=1),
        config=types.SimpleNamespace(max_seq_len=3),
        train_dataloader=lambda: iter(()),
        eval_dataloader=packed,
    )
    compared: list[tuple[str, int]] = []
    val_problems = [f"val mismatch {index}" for index in range(1, 10)]
    built_kwargs: dict[str, object] = {}

    def parse_args() -> _Arguments:
        return _Arguments()

    def identity(root: Path) -> Path:
        return root

    def load_upstream(root: Path, *, corpus: Path) -> _UpstreamPrepareStub:
        if root != Path("reference") or corpus != Path("corpus"):
            raise AssertionError((root, corpus))
        return upstream

    def build_ours(
        *,
        corpus: Path,
        rows: int,
        device: str,
        num_train_shards: int,
    ) -> types.SimpleNamespace:
        built_kwargs.update(
            {
                "corpus": corpus,
                "rows": rows,
                "device": device,
                "num_train_shards": num_train_shards,
            },
        )
        return data

    def compare_with_differences(
        theirs: Iterator[tuple[Tensor, Tensor, int]],
        ours: Iterator[NanoChatBatch],
        *,
        batches: int,
        tag: str,
    ) -> list[str]:
        assert iter(theirs) is theirs
        assert iter(ours) is ours
        compared.append((tag, batches))
        return ["train mismatch"] if tag == "train" else val_problems

    monkeypatch.setattr(karpathy_data_parity, "_parse_args", parse_args)
    monkeypatch.setattr(karpathy_data_parity, "clone_upstream", identity)
    monkeypatch.setattr(karpathy_data_parity, "load_upstream", load_upstream)
    monkeypatch.setattr(karpathy_data_parity, "build_ours", build_ours)
    monkeypatch.setattr(
        karpathy_data_parity,
        "compare_stream",
        compare_with_differences,
    )
    monkeypatch.setattr(karpathy_data_parity, "PackedTokenStream", packed)
    assert karpathy_data_parity.main() == 1
    assert built_kwargs == {
        "corpus": Path("corpus"),
        "rows": 2,
        "device": "cpu",
        "num_train_shards": 5,
    }
    assert loader_calls == [
        (tokenizer, 2, 3, "train"),
        (tokenizer, 2, 3, "val"),
    ]
    assert compared == [("train", 1), ("val", 1)]
    assert capsys.readouterr().out == (
        "upstream: reference/prepare.py\n"
        "corpus:   corpus\n"
        "vocab:    8 tokens, identical on both sides\n"
        "\n[train] 1 batches: 1 differ\n"
        "    train mismatch\n"
        "\n[val] 1 batches: 9 differ\n"
        "    val mismatch 1\n"
        "    val mismatch 2\n"
        "    val mismatch 3\n"
        "    val mismatch 4\n"
        "    val mismatch 5\n"
        "    val mismatch 6\n"
        "    val mismatch 7\n"
        "    val mismatch 8\n"
        "\neval extent: theirs 1 batches, ours 2\n"
        "\n1 batches per split: 11 DIFFERENCE(S)\n"
    )

    loader_calls.clear()
    compared.clear()
    upstream.EVAL_TOKENS = 12

    def compare_identical(
        theirs: Iterator[tuple[Tensor, Tensor, int]],
        ours: Iterator[NanoChatBatch],
        *,
        batches: int,
        tag: str,
    ) -> list[str]:
        del theirs, ours
        compared.append((tag, batches))
        return []

    monkeypatch.setattr(
        karpathy_data_parity,
        "compare_stream",
        compare_identical,
    )
    assert karpathy_data_parity.main() == 0
    assert compared == [("train", 1), ("val", 1)]
    assert capsys.readouterr().out == (
        "upstream: reference/prepare.py\n"
        "corpus:   corpus\n"
        "vocab:    8 tokens, identical on both sides\n"
        "\n[train] 1 batches: 0 differ\n"
        "\n[val] 1 batches: 0 differ\n"
        "\neval extent: 2 batches on both sides\n"
        "\n1 batches per split: BIT-IDENTICAL\n"
    )


def test_compare_stream_checks_media_and_labels_for_each_batch() -> None:
    first = torch.tensor([[1, 2]])
    second = torch.tensor([[3, 4]])
    theirs = iter([(first, second, 0), (first, second, 1)])
    ours_batches: list[NanoChatBatch] = [
        {
            "media": first,
            "label": second,
            "token_bytes": torch.ones(8),
            "valid_count": 1,
        },
        {
            "media": first,
            "label": torch.tensor([[3, 5]]),
            "token_bytes": torch.ones(8),
            "valid_count": 1,
        },
    ]
    ours = iter(ours_batches)
    assert compare_stream(theirs, ours, batches=2, tag="train") == [
        "train[2] label: DIFFERS at (0, 1) (4 vs 5), 1/2 positions",
    ]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
