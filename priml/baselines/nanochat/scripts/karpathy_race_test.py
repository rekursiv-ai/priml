"""Tests for the upstream training-race adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast
from unittest.mock import Mock, call, patch

import argparse
import builtins
import importlib
import sys
import time
import types

import pytest
import torch

from priml.baselines.nanochat.scripts import karpathy_race
from priml.lib.testing.cli import assert_help_without_docstring


class _Flags(Protocol):
    clone: Path
    corpus: Path
    rows: int
    output: Path | None


class _PrepareModule(Protocol):
    DATA_DIR: str
    TOKENIZER_DIR: str


def test_microbatch_rewrite_requires_one_assignment() -> None:
    source = "DEVICE_BATCH_SIZE = 128\nother = 1\n"
    rewritten = karpathy_race._resize_microbatch(source, rows=8)
    assert rewritten == "DEVICE_BATCH_SIZE = 8\nother = 1\n"
    with pytest.raises(RuntimeError, match="found 0"):
        karpathy_race._resize_microbatch("other = 1\n", rows=8)
    with pytest.raises(RuntimeError, match="found 2"):
        karpathy_race._resize_microbatch(
            "DEVICE_BATCH_SIZE = 128\nDEVICE_BATCH_SIZE = 64\n",
            rows=8,
        )


def test_main_help_works_without_a_module_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_help_without_docstring(monkeypatch, karpathy_race, karpathy_race.main)


def test_add_arguments_sets_machine_local_defaults() -> None:
    parser = argparse.ArgumentParser()
    karpathy_race._add_arguments(parser)

    flags = cast(_Flags, parser.parse_args([]))
    assert flags.clone == Path("/opt/scratch/karpathy-autoresearch")
    assert flags.corpus == Path("/opt/scratch/datasets/nanochat-priml")
    assert flags.rows == 32
    assert flags.output is None

    flags = cast(
        _Flags,
        parser.parse_args(
            [
                "--clone",
                "clone",
                "--corpus",
                "corpus",
                "--rows",
                "7",
                "--output",
                "result.json",
            ],
        ),
    )
    assert flags.clone == Path("clone")
    assert flags.corpus == Path("corpus")
    assert flags.rows == 7
    assert flags.output == Path("result.json")
    assert parser.format_help().count("--") == 9


def test_import_prepare_points_both_paths_at_shared_corpus(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Tokenizer:
        @classmethod
        def from_directory(cls, path: str = "original") -> Tokenizer:
            del path
            return cls()

    prepare = cast(
        _PrepareModule,
        types.SimpleNamespace(DATA_DIR="old", TOKENIZER_DIR="old", Tokenizer=Tokenizer),
    )
    import_module = Mock(return_value=prepare)
    monkeypatch.setattr(importlib, "import_module", import_module)
    corpus = tmp_path / "corpus"

    assert karpathy_race._import_prepare(corpus) is prepare
    import_module.assert_called_once_with("prepare")
    assert str(corpus) == prepare.DATA_DIR
    assert str(corpus / "tokenizer") == prepare.TOKENIZER_DIR
    assert Tokenizer.from_directory.__func__.__defaults__ == (
        str(corpus / "tokenizer"),
    )


def test_main_help_uses_the_script_description(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["race", "--help"])

    with (
        patch.object(
            argparse.ArgumentParser,
            "__init__",
            autospec=True,
            side_effect=argparse.ArgumentParser.__init__,
        ) as parser_init,
        pytest.raises(SystemExit, match="0"),
    ):
        karpathy_race.main()

    assert karpathy_race.__doc__ is not None
    assert (
        parser_init.call_args.kwargs["description"]
        == karpathy_race.__doc__.split("\n", 2)[2]
    )
    output = capsys.readouterr().out
    assert (
        "Run the reference's own train.py, on this package's portable kernel." in output
    )
    assert "The companion to a plain ``exp001`` launch" in output


def test_main_executes_reference_script_and_reports_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Tokenizer:
        @classmethod
        def from_directory(cls, path: str = "unused") -> Tokenizer:
            del path
            return cls()

        def get_vocab_size(self) -> int:
            return 17

    root = tmp_path / "clone"
    root.mkdir()
    (root / "train.py").write_text(
        "DEVICE_BATCH_SIZE = 128\n"
        "assert DEVICE_BATCH_SIZE == 32\n"
        "assert __name__ == '__main__'\n"
        f"assert __file__ == {str(root / 'train.py')!r}\n"
        "import sys\n"
        f"assert sys.path[0] == {str(root)!r}\n"
        "import kernels\n"
        "assert kernels.get_kernel('flash-attention-3').flash_attn_interface.flash_attn_func.__name__ == 'their_attention'\n"
        "val_bpb = 1.25\nstep = 3\ntotal_training_time = 4.5\n",
    )
    corpus = tmp_path / "corpus"
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(sys.modules, "kernels", types.ModuleType("original"))
    output_path = tmp_path / "nested" / "summary.json"
    if output_path.parent.exists():
        output_path.unlink(missing_ok=True)
        output_path.parent.rmdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "race",
            "--clone",
            str(root),
            "--corpus",
            str(corpus),
            "--output",
            str(output_path),
        ],
    )

    def clone_upstream(clone: Path) -> Path:
        return clone

    monkeypatch.setattr(karpathy_race, "clone_upstream", clone_upstream)
    prepare = types.SimpleNamespace(Tokenizer=Tokenizer)
    import_prepare = Mock(return_value=prepare)
    monkeypatch.setattr(karpathy_race, "_import_prepare", import_prepare)
    get_device_name = Mock(return_value="CPU test")
    monkeypatch.setattr(torch.cuda, "get_device_name", get_device_name)
    monkeypatch.setattr(time, "perf_counter", Mock(side_effect=(10.0, 13.0)))
    print_mock = Mock(wraps=builtins.print)
    monkeypatch.setattr(builtins, "print", print_mock)
    compile_mock = Mock(wraps=builtins.compile)
    monkeypatch.setattr(builtins, "compile", compile_mock)

    with patch.object(Path, "mkdir", autospec=True, side_effect=Path.mkdir) as mkdir:
        assert karpathy_race.main() == 0

    assert call(output_path.parent, parents=True, exist_ok=True) in mkdir.call_args_list
    assert compile_mock.call_args.args[1] == str(root / "train.py")
    assert print_mock.call_args_list[5] == call("device:    CPU test", flush=True)
    import_prepare.assert_called_once_with(corpus)
    assert get_device_name.call_args_list == [
        ((0,), {}),
        ((0,), {}),
    ]
    output = capsys.readouterr().out
    assert f"upstream:  {root}/train.py" in output
    assert f"corpus:    {corpus}" in output
    assert "kernel:    their_attention (SDPA; FA3 needs SM90)" in output
    assert "rows/pass: 32 (theirs is 128; 45 GiB on H100)" in output
    assert "vocab:     17" in output
    assert "device:    CPU test" in output
    assert '"device": "CPU test"' in output
    assert '"rows_per_pass": 32' in output
    assert '"steps": 3' in output
    assert '"training_seconds": 4.5' in output
    assert '"val_bpb": 1.25' in output
    assert '"wall_seconds": 3.0' in output
    assert (
        '\nRESULT: {\n  "device": "CPU test",\n  "rows_per_pass": 32,\n  "steps": 3,\n  "training_seconds": 4.5,\n  "val_bpb": 1.25,\n  "wall_seconds": 3.0\n}'
        in output
    )
    assert output_path.read_text() == (
        "{\n"
        '  "device": "CPU test",\n'
        '  "rows_per_pass": 32,\n'
        '  "steps": 3,\n'
        '  "training_seconds": 4.5,\n'
        '  "val_bpb": 1.25,\n'
        '  "wall_seconds": 3.0\n'
        "}"
    )
    assert f"wrote {output_path}" in output


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
