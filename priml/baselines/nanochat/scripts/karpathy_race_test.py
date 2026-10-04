"""Tests for the upstream training-race adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast
from unittest.mock import Mock, call, patch

import argparse
import builtins
import importlib
import subprocess
import sys
import time
import types

import pytest
import torch

from priml.baselines.nanochat.scripts import karpathy_race
from priml.model.attention.value_gated_attention import sdpa_attention


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


def test_portable_kernel_stub_matches_the_reference_contract() -> None:
    kernel = karpathy_race._kernels_stub()
    assert isinstance(kernel, types.ModuleType)
    assert kernel.__name__ == "kernels"
    attention = kernel.get_kernel("flash-attention-3").flash_attn_interface
    q = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    k = q.flip(1)
    v = q + 2
    assert torch.equal(
        attention.flash_attn_func(
            q,
            k,
            v,
            causal=True,
            window_size=(1, 0),
        ),
        sdpa_attention(q, k, v, window=1),
    )
    with pytest.raises(ValueError, match=r"^other$"):
        kernel.get_kernel("other")


def test_their_attention_rejects_unsupported_masks_exactly() -> None:
    q = torch.zeros((2, 3, 4, 5))
    with pytest.raises(ValueError, match=r"^the recipe attends causally$"):
        karpathy_race.their_attention(
            q,
            q,
            q,
            causal=False,
            window_size=(0, 0),
        )
    with pytest.raises(ValueError, match=r"^unexpected future window 1$"):
        karpathy_race.their_attention(
            q,
            q,
            q,
            causal=True,
            window_size=(0, 1),
        )


def test_main_requires_a_module_docstring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(karpathy_race, "__doc__", None)
    with pytest.raises(ValueError, match=r"^Expected __doc__ is not None\.$"):
        karpathy_race.main()


def test_clone_upstream_clones_and_checks_the_pinned_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "one" / "missing" / "clone"
    git = Mock()
    git_state = Mock(
        side_effect=[
            "b11d6f283f866eb7e10fb776a4b8553fef873fd5",
            "",
            "b11d6f283f866eb7e10fb776a4b8553fef873fd5",
            "",
        ],
    )
    monkeypatch.setattr(subprocess, "run", git)
    monkeypatch.setattr(karpathy_race, "_git", git_state)

    assert karpathy_race.clone_upstream(root) == root
    assert [call.args[0] for call in git.call_args_list] == [
        [
            "git",
            "clone",
            "--quiet",
            "https://github.com/karpathy/autoresearch.git",
            str(root),
        ],
        ["git", "checkout", "--quiet", "b11d6f283f866eb7e10fb776a4b8553fef873fd5"],
    ]
    assert git.call_args_list[0].kwargs == {"check": True}
    assert git.call_args_list[1].kwargs == {"cwd": root, "check": True}

    sibling = root.parent / "clone-again"
    assert karpathy_race.clone_upstream(sibling) == sibling
    assert git.call_args_list[2].args[0] == [
        "git",
        "clone",
        "--quiet",
        "https://github.com/karpathy/autoresearch.git",
        str(sibling),
    ]
    assert git.call_args_list[3].args[0] == [
        "git",
        "checkout",
        "--quiet",
        "b11d6f283f866eb7e10fb776a4b8553fef873fd5",
    ]
    assert git.call_args_list[2].kwargs == {"check": True}
    assert git.call_args_list[3].kwargs == {"cwd": sibling, "check": True}
    assert git_state.call_args_list == [
        ((root, "rev-parse", "HEAD"), {}),
        ((root, "status", "--porcelain"), {}),
        ((sibling, "rev-parse", "HEAD"), {}),
        ((sibling, "status", "--porcelain"), {}),
    ]


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


def test_git_returns_trimmed_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = Mock(return_value=types.SimpleNamespace(stdout=" pinned \n"))
    monkeypatch.setattr(subprocess, "run", run)

    assert karpathy_race._git(tmp_path, "rev-parse", "HEAD") == "pinned"
    assert run.call_args.args[0] == ["git", "rev-parse", "HEAD"]
    assert run.call_args.kwargs == {
        "cwd": tmp_path,
        "capture_output": True,
        "text": True,
        "check": True,
    }


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


def test_clone_upstream_rejects_wrong_revision_and_dirty_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "clone"
    (root / ".git").mkdir(parents=True)
    git = Mock(side_effect=["other"])
    monkeypatch.setattr(karpathy_race, "_git", git)
    with pytest.raises(RuntimeError, match="at other, expected pinned"):
        karpathy_race.clone_upstream(root, commit="pinned")
    assert git.call_args_list == [((root, "rev-parse", "HEAD"), {})]

    git.reset_mock(side_effect=True)
    git.side_effect = ["pinned", " M train.py"]
    with pytest.raises(RuntimeError, match="has local modifications"):
        karpathy_race.clone_upstream(root, commit="pinned")
    assert git.call_args_list == [
        ((root, "rev-parse", "HEAD"), {}),
        ((root, "status", "--porcelain"), {}),
    ]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
