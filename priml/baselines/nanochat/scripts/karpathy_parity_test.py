"""Tests for pinned upstream parity helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, override, runtime_checkable

import argparse
import ast
import builtins
import contextlib
import importlib
import subprocess
import sys
import types


if TYPE_CHECKING:
    from collections.abc import Iterator

from torch import Tensor
from torch.nn.attention import SDPBackend

import pytest
import torch

from priml.baselines.nanochat.scripts import karpathy_parity
from priml.baselines.nanochat.train_step import NanoChatTrainStep
from priml.model.attention.value_gated_attention import sdpa_attention
from priml.optimizers.composite import CompositeOptimizer
from priml.train.parallelism import NoParallel


class _StepConfig:
    def __init__(self, step: NanoChatTrainStep) -> None:
        self.step = step
        self.parallelism: NoParallel.Config | None = None

    def make(self) -> NanoChatTrainStep:
        return self.step


def test_build_ours_uses_the_portable_experiment_and_requested_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = NanoChatTrainStep.__new__(NanoChatTrainStep)
    config = _StepConfig(step)
    monkeypatch.setattr(
        karpathy_parity,
        "exp001",
        lambda: types.SimpleNamespace(step=config),
    )
    assert karpathy_parity.build_ours(device="cpu") is step
    assert isinstance(config.parallelism, NoParallel.Config)
    assert config.parallelism.device == "cpu"


def test_main_exposes_the_comparison_arguments(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    descriptions: list[str] = []
    parser_factory = argparse.ArgumentParser

    def create_parser(
        *,
        description: str,
        formatter_class: type[argparse.HelpFormatter],
    ) -> argparse.ArgumentParser:
        descriptions.append(description)
        return parser_factory(description=description, formatter_class=formatter_class)

    monkeypatch.setattr(
        karpathy_parity,
        "argparse",
        types.SimpleNamespace(
            ArgumentParser=create_parser,
            RawDescriptionHelpFormatter=argparse.RawDescriptionHelpFormatter,
        ),
    )
    monkeypatch.setattr(sys, "argv", ["karpathy_parity", "--help"])
    with pytest.raises(SystemExit) as exit_error:
        karpathy_parity.main()
    assert exit_error.value.code == 0
    docstring = karpathy_parity.__doc__
    assert docstring is not None
    assert descriptions == [docstring.split("\n", 2)[2]]
    help_text = capsys.readouterr().out
    assert "Prove" in help_text
    assert "Clones karpathy/autoresearch at the pinned commit" in help_text
    assert "--steps" in help_text
    assert "--warmup" in help_text
    assert "--budget-steps" in help_text
    assert "--eval-batches" in help_text


def test_main_passes_parsed_clone_path_to_reference_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clone = Path("chosen-reference")
    monkeypatch.setattr(sys, "argv", ["karpathy_parity", "--clone", str(clone)])
    observed: list[Path] = []

    def stop_after_clone(root: Path) -> Path:
        observed.append(root)
        raise RuntimeError("stop after parsed arguments")

    monkeypatch.setattr(karpathy_parity, "clone_upstream", stop_after_clone)
    with pytest.raises(RuntimeError, match="stop after parsed arguments"):
        karpathy_parity.main()
    assert observed == [clone]


def test_main_runs_the_requested_comparison_and_reports_eval_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    clone = Path("reference")
    corpus = Path("corpus")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "karpathy_parity",
            "--clone",
            str(clone),
            "--corpus",
            str(corpus),
            "--steps",
            "2",
            "--warmup",
            "0",
            "--budget-steps",
            "4",
            "--rows",
            "3",
            "--eval-batches",
            "5",
            "--device",
            "cpu",
        ],
    )
    clone_calls: list[Path] = []

    def clone_upstream(root: Path) -> Path:
        clone_calls.append(root)
        return root

    monkeypatch.setattr(karpathy_parity, "clone_upstream", clone_upstream)

    class Reference(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(2, 3)
            self.window_sizes = [(7, 0)]
            self.config = types.SimpleNamespace(num_layers=1)
            self.blocks = [types.SimpleNamespace(attn=types.SimpleNamespace(window=7))]
            self.zero_grad_calls: list[bool] = []

        @override
        def forward(self, tokens: Tensor, targets: Tensor) -> Tensor:
            return torch.nn.functional.cross_entropy(self.projection(tokens), targets)

        @override
        def zero_grad(self, set_to_none: bool = True) -> None:
            self.zero_grad_calls.append(set_to_none)
            super().zero_grad(set_to_none=set_to_none)

    reference = Reference()
    reference.eval()
    optimizer = torch.optim.SGD(
        [
            {"params": [reference.projection.weight], "lr": 0.1},
            {"params": [reference.projection.bias], "lr": 0.2},
        ],
    )
    optimizer.param_groups[0].update(initial_lr=0.1, kind="muon")
    optimizer.param_groups[1].update(initial_lr=0.2, kind="adamw")
    build_calls: list[tuple[Path, Path, dict[str, object], int]] = []
    schedule_calls: list[tuple[str, float | int]] = []

    def build_theirs(
        root: Path,
        *,
        corpus: Path,
        loader: dict[str, object],
        rows: int,
        rng: dict[str, object],
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer, object]:
        build_calls.append((root, corpus, rng, rows))
        rng["state"] = object()
        loader["train"] = iter(
            [
                # Two rows of Reference.projection's 2 input features.
                (torch.ones(2, 2), torch.tensor([0, 1]), object()),
                (torch.zeros(2, 2), torch.tensor([1, 2]), object()),
            ],
        )
        loader["prepare"] = object()

        def get_lr_multiplier(progress: float) -> float:
            schedule_calls.append(("lr", progress))
            return 1 + progress

        def get_muon_momentum(index: int) -> float:
            schedule_calls.append(("momentum", index))
            return 0.9 + index

        def get_weight_decay(progress: float) -> float:
            schedule_calls.append(("decay", progress))
            return 0.01 + progress

        upstream = types.SimpleNamespace(
            window_sizes=reference.window_sizes,
            get_lr_multiplier=get_lr_multiplier,
            get_muon_momentum=get_muon_momentum,
            get_weight_decay=get_weight_decay,
        )
        return reference, optimizer, upstream

    monkeypatch.setattr(karpathy_parity, "build_theirs", build_theirs)
    rng_states: list[object] = []
    monkeypatch.setattr(karpathy_parity, "set_rng_state", rng_states.append)

    class OursModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(2, 3)
            self.config = types.SimpleNamespace(num_layers=1)
            self.blocks = [types.SimpleNamespace(attn=types.SimpleNamespace(window=7))]

        @override
        def forward(self, tokens: Tensor) -> Tensor:
            return self.projection(tokens)

    ours_model = OursModel()
    ours_model.eval()
    update_calls: list[tuple[float, int]] = []
    ours = types.SimpleNamespace(
        model=ours_model,
        config=types.SimpleNamespace(train_budget_sec=8.0),
        timer_step=types.SimpleNamespace(global_count=-1),
        _apply_update=lambda: update_calls.append(
            (cast(float, ours.elapsed_sec), cast(int, ours.timer_step.global_count)),
        ),
    )
    ours.elapsed_sec = 0.0
    device_calls: list[str] = []

    def build_ours(*, device: str) -> types.SimpleNamespace:
        device_calls.append(device)
        return ours

    monkeypatch.setattr(karpathy_parity, "build_ours", build_ours)

    def name_map(model: torch.nn.Module, *, layers: int) -> dict[str, str]:
        del model, layers
        return {}

    monkeypatch.setattr(karpathy_parity, "name_map", name_map)
    compared: list[tuple[bool, str]] = []

    def compare_all(
        theirs: torch.nn.Module,
        ours: torch.nn.Module,
        mapping: dict[str, str],
        *,
        grads: bool,
        tag: str,
    ) -> list[str]:
        del theirs, ours, mapping
        compared.append((grads, tag))
        return [f"{tag} issue"]

    monkeypatch.setattr(karpathy_parity, "compare_all", compare_all)
    copied: list[tuple[torch.nn.Module, torch.nn.Module]] = []

    def copy_weights(
        theirs: torch.nn.Module,
        ours: torch.nn.Module,
        mapping: dict[str, str],
    ) -> None:
        del mapping
        copied.append((theirs, ours))

    monkeypatch.setattr(karpathy_parity, "copy_weights", copy_weights)
    loss_comparisons: list[tuple[Tensor, Tensor]] = []

    def compare(label: str, left: Tensor, right: Tensor) -> str:
        assert label == "loss"
        loss_comparisons.append((left, right))
        return "loss issue"

    monkeypatch.setattr(karpathy_parity, "compare", compare)
    state_comparisons: list[int] = []

    def compare_state(
        theirs: torch.nn.Module,
        their_optimizer: torch.optim.Optimizer,
        ours: NanoChatTrainStep,
        mapping: dict[str, str],
    ) -> list[str]:
        del theirs, their_optimizer, ours, mapping
        state_comparisons.append(len(state_comparisons))
        return ["state issue"]

    monkeypatch.setattr(karpathy_parity, "compare_state", compare_state)
    evaluations: list[tuple[object, int]] = []

    def compare_eval(
        theirs: torch.nn.Module,
        ours: NanoChatTrainStep,
        upstream: object,
        prepare: object,
        *,
        batches: int,
    ) -> int:
        del theirs, ours, upstream
        evaluations.append((prepare, batches))
        return 1

    monkeypatch.setattr(karpathy_parity, "compare_eval", compare_eval)
    autocast_options: list[dict[str, object]] = []
    sdpa_backends: list[object] = []

    def autocast(**kwargs: object) -> contextlib.AbstractContextManager[None]:
        autocast_options.append(kwargs)
        return contextlib.nullcontext()

    def sdpa_kernel(backend: object) -> contextlib.AbstractContextManager[None]:
        sdpa_backends.append(backend)
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.amp, "autocast", autocast)
    monkeypatch.setattr(karpathy_parity, "sdpa_kernel", sdpa_kernel)

    assert karpathy_parity.main() == 1

    assert clone_calls == [clone]
    assert build_calls == [(clone, corpus, build_calls[0][2], 3)]
    assert rng_states == [build_calls[0][2]["state"]]
    assert device_calls == ["cpu"]
    assert copied == [(reference, ours_model)]
    assert compared == [
        (False, "init"),
        (False, "copied"),
        (True, "grad"),
        (False, "weight"),
        (True, "grad"),
        (False, "weight"),
    ]
    assert len(loss_comparisons) == 2
    assert len(state_comparisons) == 2
    assert schedule_calls == [
        ("lr", 0),
        ("momentum", 0),
        ("decay", 0),
        ("lr", 0.25),
        ("momentum", 1),
        ("decay", 0.25),
    ]
    assert optimizer.param_groups[0]["lr"] == 0.125
    assert optimizer.param_groups[0]["momentum"] == 1.9
    assert optimizer.param_groups[0]["weight_decay"] == 0.26
    assert optimizer.param_groups[1]["lr"] == 0.25
    assert optimizer.param_groups[1]["momentum"] == 0.0
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    assert update_calls == [(0.0, 0), (2.0, 1)]
    assert reference.zero_grad_calls == [True, True]
    assert reference.training
    assert ours_model.training
    assert autocast_options == [
        {"device_type": "cpu", "dtype": torch.bfloat16},
    ]
    assert sdpa_backends == [SDPBackend.MATH]
    assert len(evaluations) == 1
    assert evaluations[0][0] is not None
    assert evaluations[0][1] == 5
    assert capsys.readouterr().out.splitlines() == [
        "",
        "[0] init from one RNG state: 1 differ",
        "    init issue",
        "[0] init weights copied: 1 differ",
        "    copied issue",
        "    windows ours=[7] theirs=[7]",
        "[1] loss DIFFERS | grads 1 differ | weights 1 differ | state 1 differ",
        "    loss issue",
        "    grad issue",
        "    weight issue",
        "    state issue",
        "[2] loss DIFFERS | grads 1 differ | weights 1 differ | state 1 differ",
        "    loss issue",
        "    grad issue",
        "    weight issue",
        "    state issue",
        "",
        "2 steps, FA3->FA2 only: 11 DIFFERENCE(S)",
    ]


def test_compare_checks_shape_dtype_and_exact_values() -> None:
    value = torch.tensor([1.0, 2.0])
    assert karpathy_parity.compare("x", value, value.clone()) is None
    assert karpathy_parity.compare("x", value, torch.tensor([1.0])) == (
        "x: SHAPE (2,) vs (1,)"
    )
    assert karpathy_parity.compare("x", value, value.to(torch.float64)) == (
        "x: DTYPE torch.float32 vs torch.float64"
    )
    assert karpathy_parity.compare("x", value, torch.tensor([1.0, 3.0])) == (
        "x: DIFFERS max_abs=1.000e+00"
    )


class _FlashAttentionFunction(Protocol):
    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> Tensor: ...


class _FlashAttentionInterface(Protocol):
    flash_attn_func: _FlashAttentionFunction


class _KernelAdapter(Protocol):
    flash_attn_interface: _FlashAttentionInterface


@runtime_checkable
class _KernelRegistry(Protocol):
    def get_kernel(self, name: str) -> _KernelAdapter: ...


def test_kernel_stub_supplies_the_portable_attention_adapter() -> None:
    kernel = karpathy_parity._kernels_stub()
    assert kernel.__name__ == "kernels"
    assert isinstance(kernel, _KernelRegistry)
    adapter = kernel.get_kernel("flash-attention-3")
    assert (
        adapter.flash_attn_interface.flash_attn_func is karpathy_parity.their_attention
    )
    with pytest.raises(ValueError, match=r"^other$"):
        kernel.get_kernel("other")


def test_progress_tracks_unbilled_prefix_and_caps_at_budget() -> None:
    progress = karpathy_parity._progress_at
    assert progress(1, warmup=2, budget_steps=4) == 0
    assert progress(3, warmup=2, budget_steps=4) == 0
    assert progress(4, warmup=2, budget_steps=4) == 0.25
    assert progress(20, warmup=2, budget_steps=4) == 1


def test_portable_attention_requires_causal_no_future_window() -> None:
    q = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5)
    k = q.flip(1)
    v = q.flip(-1)
    output = karpathy_parity.their_attention(
        q,
        k,
        v,
        causal=True,
        window_size=(1, 0),
    )
    torch.testing.assert_close(
        output,
        sdpa_attention(q, k, v, window=1),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError, match=r"^the recipe attends causally$"):
        karpathy_parity.their_attention(q, k, v, causal=False, window_size=(1, 0))
    with pytest.raises(ValueError, match="future window"):
        karpathy_parity.their_attention(q, k, v, causal=True, window_size=(1, 1))


class _ParsedArgs(Protocol):
    clone: Path
    corpus: Path
    steps: int
    warmup: int
    budget_steps: int
    rows: int
    eval_batches: int
    device: str


class _TokenizerStub:
    @classmethod
    def from_directory(cls, path: str) -> None:
        del cls, path


@runtime_checkable
class _PrepareDirectory(Protocol):
    DATA_DIR: str


class _EvaluationPrepareStub:
    def __init__(self, *, eval_tokens: int, max_seq_len: int) -> None:
        self.EVAL_TOKENS = eval_tokens
        self.MAX_SEQ_LEN = max_seq_len


def test_clone_upstream_uses_the_pinned_git_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "outer" / "nested" / "reference"
    commit = "b11d6f283f866eb7e10fb776a4b8553fef873fd5"
    calls: list[tuple[list[str], Path | None, bool, bool, bool]] = []

    def run(
        arguments: list[str],
        *,
        cwd: Path | None = None,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, cwd, check, capture_output, text))
        if arguments[1] == "clone":
            root.mkdir()
            (root / ".git").mkdir()
            output = ""
        elif arguments[1] == "checkout":
            output = ""
        elif arguments[1] == "rev-parse" and arguments[2] == "HEAD":
            output = commit
        else:
            output = ""
        return subprocess.CompletedProcess(arguments, 0, output, "")

    monkeypatch.setattr(subprocess, "run", run)
    assert karpathy_parity.clone_upstream(root) == root
    assert calls == [
        (
            [
                "git",
                "clone",
                "--quiet",
                "https://github.com/karpathy/autoresearch.git",
                str(root),
            ],
            None,
            True,
            False,
            False,
        ),
        (["git", "checkout", "--quiet", commit], root, True, False, False),
        (["git", "rev-parse", "HEAD"], root, True, True, True),
        (["git", "status", "--porcelain"], root, True, True, True),
    ]


def test_clone_upstream_allows_an_existing_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parent" / "reference"
    root.parent.mkdir()
    commit = "pinned"

    def run(
        arguments: list[str],
        *,
        cwd: Path | None = None,
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, check, capture_output, text
        if arguments[1] == "clone":
            root.mkdir()
            (root / ".git").mkdir()
            output = ""
        elif arguments[1] == "rev-parse":
            output = commit
        else:
            output = ""
        return subprocess.CompletedProcess(arguments, 0, output, "")

    monkeypatch.setattr(subprocess, "run", run)
    assert karpathy_parity.clone_upstream(root, commit=commit) == root


def test_clone_upstream_rejects_wrong_revision_and_dirty_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reference"
    (root / ".git").mkdir(parents=True)
    outputs = iter(["wrong-commit", ""])

    def git(*args: object) -> str:
        del args
        return next(outputs)

    monkeypatch.setattr(karpathy_parity, "_git", git)
    with pytest.raises(RuntimeError, match="clone is at wrong-commit"):
        karpathy_parity.clone_upstream(root, commit="expected")

    outputs = iter(["expected", " M train.py"])
    with pytest.raises(
        RuntimeError,
        match=r"clone has local modifications:\n M train.py",
    ):
        karpathy_parity.clone_upstream(root, commit="expected")


def test_load_upstream_uses_the_supplied_reference_and_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reference"
    corpus = tmp_path / "corpus"
    root.mkdir()
    (root / "train.py").write_text(
        "import prepare\nDEVICE_BATCH_SIZE = 128\n"
        "class GPT:\n    pass\n"
        "class GPTConfig:\n    pass\n"
        "model = GPT()\noptimizer = object()\nprepare.make_dataloader()\n",
    )
    constants = types.ModuleType("constants")
    prepare = types.ModuleType("prepare")
    prepare.__dict__.update(
        {
            "DATA_DIR": "original-data",
            "TOKENIZER_DIR": "original-tokenizer",
            "Tokenizer": _TokenizerStub,
            "make_dataloader": lambda: iter(()),
        },
    )
    imports: list[str] = []

    def import_module(name: str) -> types.ModuleType:
        imports.append(name)
        return {"constants": constants, "prepare": prepare}[name]

    monkeypatch.setattr(importlib, "import_module", import_module)
    captured_rngs: list[dict[str, object]] = []

    def capture_rng(
        state: dict[str, object],
    ) -> contextlib.AbstractContextManager[None]:
        captured_rngs.append(state)
        return contextlib.nullcontext()

    monkeypatch.setattr(karpathy_parity, "_capture_rng_after_seeding", capture_rng)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "prepare", prepare)
    monkeypatch.setitem(sys.modules, "train", types.ModuleType("train"))
    monkeypatch.setitem(sys.modules, "kernels", types.ModuleType("old-kernels"))
    loader: dict[str, object] = {}
    rng: dict[str, object] = {}
    loaded = karpathy_parity.load_upstream(
        root,
        corpus=corpus,
        loader=loader,
        rows=2,
        rng=rng,
    )
    assert imports == ["constants", "prepare"]
    assert sys.path[0] == str(root)
    assert constants.__dict__["TIME_BUDGET"] == 1e-9
    assert isinstance(prepare, _PrepareDirectory)
    data_dir = prepare.DATA_DIR
    assert str(corpus) == data_dir
    assert loaded.__name__ == "train"
    assert sys.modules["train"] is loaded
    kernel_module = sys.modules["kernels"]
    assert kernel_module.__name__ == "kernels"
    assert captured_rngs == [rng]
    assert loader["train"] is not None
    assert loader["prepare"] is prepare
    assert loaded.__file__ == str(root / "train.py")


def _configure_load_upstream(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    root.mkdir()
    (root / "train.py").write_text(source)
    constants = types.ModuleType("constants")
    prepare = types.ModuleType("prepare")
    prepare.__dict__.update(
        {
            "DATA_DIR": "old-data",
            "TOKENIZER_DIR": "old-tokenizer",
            "Tokenizer": _TokenizerStub,
            "make_dataloader": lambda: iter(()),
        },
    )

    def import_module(name: str) -> types.ModuleType:
        return {"constants": constants, "prepare": prepare}[name]

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "prepare", prepare)
    monkeypatch.setitem(sys.modules, "train", types.ModuleType("old-train"))
    monkeypatch.setitem(sys.modules, "kernels", types.ModuleType("old-kernels"))
    monkeypatch.setattr(
        karpathy_parity,
        "_capture_rng_after_seeding",
        contextlib.nullcontext,
    )


def test_load_upstream_reports_wrong_device_batch_assignment_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reference"
    _configure_load_upstream(root, monkeypatch, "pass\n")

    with pytest.raises(RuntimeError) as error:
        karpathy_parity.load_upstream(
            root,
            corpus=tmp_path / "corpus",
            loader={},
            rows=2,
            rng={},
        )

    assert str(error.value) == (
        "expected one DEVICE_BATCH_SIZE assignment in train.py; found 0."
    )


def test_load_upstream_reports_missing_reference_definition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reference"
    _configure_load_upstream(
        root,
        monkeypatch,
        "DEVICE_BATCH_SIZE = 8\n"
        "class GPTConfig:\n    pass\n"
        "model = object()\noptimizer = object()\n",
    )

    with pytest.raises(RuntimeError) as error:
        karpathy_parity.load_upstream(
            root,
            corpus=tmp_path / "corpus",
            loader={},
            rows=2,
            rng={},
        )

    assert str(error.value) == "train.py aborted before defining GPT"


def test_load_upstream_reports_missing_train_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reference"
    _configure_load_upstream(
        root,
        monkeypatch,
        "DEVICE_BATCH_SIZE = 8\n"
        "class GPT:\n    pass\n"
        "class GPTConfig:\n    pass\n"
        "model = GPT()\noptimizer = object()\n",
    )

    with pytest.raises(RuntimeError) as error:
        karpathy_parity.load_upstream(
            root,
            corpus=tmp_path / "corpus",
            loader={},
            rows=2,
            rng={},
        )

    assert str(error.value) == "train.py never asked for a dataloader"


def test_load_upstream_reports_the_reference_path_on_syntax_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "reference"
    root.mkdir()
    source = root / "train.py"
    source.write_text("DEVICE_BATCH_SIZE = 8\\nif (\\n")
    constants = types.ModuleType("constants")
    prepare = types.ModuleType("prepare")
    prepare.__dict__.update(
        {
            "DATA_DIR": "old-data",
            "TOKENIZER_DIR": "old-tokenizer",
            "Tokenizer": _TokenizerStub,
            "make_dataloader": lambda: iter(()),
        },
    )

    def import_module(name: str) -> types.ModuleType:
        return {"constants": constants, "prepare": prepare}[name]

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "kernels", types.ModuleType("old-kernels"))
    monkeypatch.setattr(
        karpathy_parity,
        "_capture_rng_after_seeding",
        contextlib.nullcontext,
    )
    with pytest.raises(SyntaxError) as error:
        karpathy_parity.load_upstream(
            root,
            corpus=tmp_path / "corpus",
            loader={},
            rows=2,
            rng={},
        )
    assert error.value.filename == str(source)


def test_prepare_module_rebinds_paths_and_restores_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    captured = iter(())

    def original_loader(*args: object, **kwargs: object) -> Iterator[object]:
        calls.append((args, kwargs))
        return captured

    prepare = types.ModuleType("prepare")
    prepare.__dict__.update(
        {
            "DATA_DIR": "old-data",
            "TOKENIZER_DIR": "old-tokenizer",
            "Tokenizer": _TokenizerStub,
            "make_dataloader": original_loader,
        },
    )

    def import_module(name: str) -> types.ModuleType:
        if name != "prepare":
            raise AssertionError(name)
        return prepare

    monkeypatch.setattr(importlib, "import_module", import_module)

    loader: dict[str, object] = {}
    corpus = tmp_path / "corpus"
    result = karpathy_parity._prepare_module(corpus, loader)
    assert result is prepare
    data_dir = result.DATA_DIR
    tokenizer_dir = result.TOKENIZER_DIR
    assert data_dir == str(corpus)
    assert tokenizer_dir == str(corpus / "tokenizer")
    assert _TokenizerStub.from_directory.__func__.__defaults__ == (
        str(corpus / "tokenizer"),
    )
    with pytest.raises(karpathy_parity._StopModuleScopeError):
        result.make_dataloader("train", batch_size=3)
    assert calls == [(("train",), {"batch_size": 3})]
    assert loader["train"] is captured
    assert loader["prepare"] is prepare
    assert result.make_dataloader is original_loader


def test_name_map_covers_gated_and_wrapped_parameters() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.value_embeds = torch.nn.ModuleList(
                [torch.nn.Embedding(2, 3), torch.nn.Embedding(2, 3)],
            )
            self.transformer = torch.nn.Module()
            self.transformer.h = torch.nn.ModuleList(
                [torch.nn.Module(), torch.nn.Module()],
            )
            for block in self.transformer.h:
                block.attn = torch.nn.Module()
                block.attn.ve_gate = torch.nn.Linear(3, 2, bias=False)

    mapping = karpathy_parity.name_map(Model(), layers=2)
    assert mapping["embed.inner.weight"] == "transformer.wte.weight"
    assert mapping["lm_head.inner.weight"] == "lm_head.weight"
    assert mapping["mix.running"] == "resid_lambdas"
    assert mapping["mix.original"] == "x0_lambdas"
    assert (
        mapping["blocks.1.ffn.down_proj.weight"] == "transformer.h.1.mlp.c_proj.weight"
    )
    assert mapping["value_embeds.1.inner.weight"] == "value_embeds.1.weight"
    assert (
        mapping["blocks.1.attn.value_gate.weight"]
        == "transformer.h.1.attn.ve_gate.weight"
    )


def test_name_map_preserves_compiled_parameter_prefix() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._orig_mod = torch.nn.Module()
            self._orig_mod.value_embeds = torch.nn.ModuleList(
                [torch.nn.Embedding(2, 3), torch.nn.Embedding(2, 3)],
            )
            self._orig_mod.transformer = torch.nn.Module()
            self._orig_mod.transformer.h = torch.nn.ModuleList(
                [torch.nn.Module(), torch.nn.Module()],
            )
            for block in self._orig_mod.transformer.h:
                block.attn = torch.nn.Module()
                block.attn.ve_gate = torch.nn.Linear(3, 2, bias=False)

    mapping = karpathy_parity.name_map(Model(), layers=2)
    assert mapping["blocks.0.attn.proj_q.weight"] == (
        "_orig_mod.transformer.h.0.attn.c_q.weight"
    )
    assert mapping["value_embeds.1.inner.weight"] == ("_orig_mod.value_embeds.1.weight")
    assert mapping["blocks.1.attn.value_gate.weight"] == (
        "_orig_mod.transformer.h.1.attn.ve_gate.weight"
    )


def test_copy_weights_checks_mapping_and_copies_values() -> None:
    source = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
    destination = torch.nn.Linear(2, 3, bias=False, dtype=torch.float32)
    with torch.no_grad():
        source.weight.fill_(2.5)
    karpathy_parity.copy_weights(
        source,
        destination,
        {"weight": "weight"},
    )
    torch.testing.assert_close(
        destination.weight,
        source.weight.float(),
        rtol=0,
        atol=0,
    )
    assert destination.weight.dtype == torch.float32
    with pytest.raises(RuntimeError, match="name map incomplete"):
        karpathy_parity.copy_weights(source, destination, {})
    with pytest.raises(
        RuntimeError,
        match=r"name map incomplete: unmapped=\[\] absent=\['missing.weight'\]",
    ):
        karpathy_parity.copy_weights(
            source,
            destination,
            {"weight": "missing.weight"},
        )


def test_compare_all_reports_missing_and_different_gradients() -> None:
    left = torch.nn.Linear(2, 3, bias=False)
    right = torch.nn.Linear(2, 3, bias=False)
    mapping = {"weight": "weight"}
    assert karpathy_parity.compare_all(
        left,
        right,
        mapping,
        grads=True,
        tag="grad",
    ) == ["grad weight: MISSING gradient"]
    left.weight.grad = torch.ones_like(left.weight)
    right.weight.grad = torch.zeros_like(right.weight)
    assert karpathy_parity.compare_all(
        left,
        right,
        mapping,
        grads=True,
        tag="grad",
    ) == ["grad weight: DIFFERS max_abs=1.000e+00"]


def test_build_theirs_returns_reference_model_optimizer_and_schedules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = torch.nn.Linear(2, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    upstream = types.ModuleType("train")
    upstream.__file__ = str(tmp_path / "train.py")
    upstream.__dict__["fa3"] = types.SimpleNamespace(
        flash_attn_func=test_build_theirs_returns_reference_model_optimizer_and_schedules,
    )
    observed: list[tuple[object, ...]] = []

    def load(root: Path, **kwargs: object) -> types.ModuleType:
        observed.append(
            (root, kwargs["corpus"], kwargs["loader"], kwargs["rows"], kwargs["rng"]),
        )
        return upstream

    monkeypatch.setattr(karpathy_parity, "load_upstream", load)

    schedule_roots: list[Path] = []

    def schedules(root: Path, module: types.ModuleType) -> types.ModuleType:
        schedule_roots.append(root)
        return module

    monkeypatch.setattr(karpathy_parity, "their_schedules", schedules)
    monkeypatch.setattr(upstream, "model", model, raising=False)
    monkeypatch.setattr(upstream, "optimizer", optimizer, raising=False)
    result = karpathy_parity.build_theirs(
        tmp_path,
        corpus=tmp_path / "corpus",
        loader={},
        rows=2,
        rng={},
    )
    assert result == (model, optimizer, upstream)
    assert observed == [(tmp_path, tmp_path / "corpus", {}, 2, {})]
    assert schedule_roots == [tmp_path]
    output = capsys.readouterr().out
    assert f"upstream: {upstream.__file__}" in output
    assert "kernel:" in output


def test_build_theirs_rejects_invalid_model_or_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream = types.ModuleType("train")
    upstream.__file__ = str(tmp_path / "train.py")
    upstream.__dict__["fa3"] = types.SimpleNamespace(
        flash_attn_func=test_build_theirs_rejects_invalid_model_or_optimizer,
    )

    def load(*args: object, **kwargs: object) -> types.ModuleType:
        del args, kwargs
        return upstream

    def schedules(root: Path, module: types.ModuleType) -> types.ModuleType:
        del root
        return module

    monkeypatch.setattr(karpathy_parity, "load_upstream", load)
    monkeypatch.setattr(karpathy_parity, "their_schedules", schedules)
    monkeypatch.setattr(upstream, "model", object(), raising=False)
    monkeypatch.setattr(upstream, "optimizer", object(), raising=False)
    with pytest.raises(AssertionError, match="object"):
        karpathy_parity.build_theirs(
            tmp_path,
            corpus=tmp_path,
            loader={},
            rows=2,
            rng={},
        )

    monkeypatch.setattr(upstream, "model", torch.nn.Linear(2, 3), raising=False)
    with pytest.raises(AssertionError, match="object"):
        karpathy_parity.build_theirs(
            tmp_path,
            corpus=tmp_path,
            loader={},
            rows=2,
            rng={},
        )


def test_their_schedules_executes_definitions_and_rejects_missing_functions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled_modules: list[ast.Module] = []

    def compile_module(module: ast.Module, filename: str, mode: str) -> types.CodeType:
        compiled_modules.append(module)
        return builtins.compile(module, filename, mode)

    monkeypatch.setattr(karpathy_parity, "compile", compile_module, raising=False)
    (tmp_path / "train.py").write_text(
        "DEVICE_BATCH_SIZE = 2\n"
        "def get_lr_multiplier(progress): return progress + 1\n"
        "def get_muon_momentum(step): return step + 2\n"
        "def get_weight_decay(progress): return progress + 3\n",
    )
    module = types.ModuleType("train")
    result = karpathy_parity.their_schedules(tmp_path, module)
    assert result is module
    assert result.get_lr_multiplier(2) == 3
    assert result.get_muon_momentum(2) == 4
    assert result.get_weight_decay(2) == 5
    assert {
        "get_lr_multiplier",
        "get_muon_momentum",
        "get_weight_decay",
    } <= result.__dict__.keys()
    assert len(compiled_modules) == 1
    assert {
        node.name
        for node in compiled_modules[0].body
        if isinstance(node, ast.FunctionDef)
    } == {"get_lr_multiplier", "get_muon_momentum", "get_weight_decay"}
    assert result.get_lr_multiplier.__code__.co_filename == str(tmp_path / "train.py")
    (tmp_path / "train.py").write_text(
        "def get_lr_multiplier(progress): return progress\n",
    )
    with pytest.raises(RuntimeError, match=r"train.py defines no"):
        karpathy_parity.their_schedules(tmp_path, types.ModuleType("train"))


def test_loss_adapter_registers_and_forwards_the_inner_model() -> None:
    model = torch.nn.Linear(3, 4, bias=False)
    adapter = karpathy_parity._LossAdapter(model)
    tokens = torch.randn(2, 3)
    targets = torch.tensor([1, 2])
    logits = model(tokens)
    actual = adapter(tokens, targets, reduction="none")
    expected = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert adapter.inner is model


def test_compare_all_continues_after_missing_gradients() -> None:
    left = torch.nn.Linear(2, 3)
    right = torch.nn.Linear(2, 3)
    right(torch.zeros(3, 2)).sum().backward()
    assert karpathy_parity.compare_all(
        left,
        right,
        {"weight": "weight", "bias": "bias"},
        grads=True,
        tag="grad",
    ) == [
        "grad weight: MISSING gradient",
        "grad bias: MISSING gradient",
    ]


def test_compare_state_checks_matching_optimizer_moments() -> None:
    theirs = torch.nn.Linear(2, 3, bias=False)
    ours_model = torch.nn.Linear(2, 3, bias=False)
    their_optimizer = torch.optim.Adam(theirs.parameters())
    our_member = torch.optim.Adam(ours_model.parameters())
    ours_optimizer = CompositeOptimizer([our_member])
    source_parameter = theirs.weight
    destination_parameter = ours_model.weight
    their_optimizer.state[source_parameter] = {
        "exp_avg": torch.ones_like(source_parameter),
        "exp_avg_sq": torch.ones_like(source_parameter) * 2,
        "momentum_buffer": torch.ones_like(source_parameter) * 3,
    }
    first_moment = torch.ones_like(destination_parameter)
    second_moment = torch.ones_like(destination_parameter) * 2
    momentum_buffer = torch.ones_like(destination_parameter) * 3
    our_member.state[destination_parameter] = {
        "first_moment": first_moment,
        "second_moment": second_moment,
        "momentum_buffer": momentum_buffer,
    }
    step = cast(
        NanoChatTrainStep,
        types.SimpleNamespace(model=ours_model, optimizer=ours_optimizer),
    )
    assert (
        karpathy_parity.compare_state(
            theirs,
            their_optimizer,
            step,
            {"weight": "weight"},
        )
        == []
    )
    for value in (first_moment, second_moment, momentum_buffer):
        value.zero_()
    assert karpathy_parity.compare_state(
        theirs,
        their_optimizer,
        step,
        {"weight": "weight"},
    ) == [
        "state weight[first_moment]: DIFFERS max_abs=1.000e+00",
        "state weight[second_moment]: DIFFERS max_abs=2.000e+00",
        "state weight[momentum_buffer]: DIFFERS max_abs=3.000e+00",
    ]


def test_compare_state_uses_the_reference_momentum_fallback() -> None:
    theirs = torch.nn.Linear(2, 3, bias=False)
    ours_model = torch.nn.Linear(2, 3, bias=False)
    their_optimizer = torch.optim.SGD(theirs.parameters(), lr=0.1)
    our_member = torch.optim.SGD(ours_model.parameters(), lr=0.1)
    ours_optimizer = CompositeOptimizer([our_member])
    their_optimizer.state[theirs.weight] = {
        "second_momentum_buffer": torch.full_like(theirs.weight, 4),
    }
    second_moment = torch.full_like(ours_model.weight, 4)
    our_member.state[ours_model.weight] = {"second_moment": second_moment}
    step = cast(
        NanoChatTrainStep,
        types.SimpleNamespace(model=ours_model, optimizer=ours_optimizer),
    )
    assert (
        karpathy_parity.compare_state(
            theirs,
            their_optimizer,
            step,
            {"weight": "weight"},
        )
        == []
    )
    second_moment.zero_()
    assert karpathy_parity.compare_state(
        theirs,
        their_optimizer,
        step,
        {"weight": "weight"},
    ) == ["state weight[second_moment]: DIFFERS max_abs=4.000e+00"]


def test_compare_state_ignores_absent_or_shape_mismatched_buffers() -> None:
    theirs = torch.nn.Linear(2, 3, bias=False)
    ours_model = torch.nn.Linear(2, 3, bias=False)
    their_optimizer = torch.optim.Adam(theirs.parameters())
    ours_optimizer = CompositeOptimizer(
        [torch.optim.Adam(ours_model.parameters())],
    )
    their_optimizer.state[theirs.weight] = {
        "exp_avg": torch.ones(1),
    }
    step = cast(
        NanoChatTrainStep,
        types.SimpleNamespace(model=ours_model, optimizer=ours_optimizer),
    )
    assert (
        karpathy_parity.compare_state(
            theirs,
            their_optimizer,
            step,
            {"weight": "weight"},
        )
        == []
    )
    stateless_reference = torch.nn.Linear(2, 3, bias=False)
    stateless_model = torch.nn.Linear(2, 3, bias=False)
    stateless_step = cast(
        NanoChatTrainStep,
        types.SimpleNamespace(
            model=stateless_model,
            optimizer=CompositeOptimizer(
                [torch.optim.SGD(stateless_model.parameters(), lr=0.1)],
            ),
        ),
    )
    assert (
        karpathy_parity.compare_state(
            stateless_reference,
            torch.optim.SGD(stateless_reference.parameters(), lr=0.1),
            stateless_step,
            {"weight": "weight"},
        )
        == []
    )


def test_compare_state_continues_after_shape_mismatched_buffer() -> None:
    theirs = torch.nn.Linear(2, 3, bias=False)
    ours_model = torch.nn.Linear(2, 3, bias=False)
    their_optimizer = torch.optim.Adam(theirs.parameters())
    our_member = torch.optim.Adam(ours_model.parameters())
    ours_optimizer = CompositeOptimizer([our_member])
    their_optimizer.state[theirs.weight] = {
        "exp_avg": torch.ones(1),
        "exp_avg_sq": torch.ones_like(theirs.weight) * 2,
    }
    our_member.state[ours_model.weight] = {
        "first_moment": torch.zeros_like(ours_model.weight),
        "second_moment": torch.zeros_like(ours_model.weight),
    }
    step = cast(
        NanoChatTrainStep,
        types.SimpleNamespace(model=ours_model, optimizer=ours_optimizer),
    )

    assert karpathy_parity.compare_state(
        theirs,
        their_optimizer,
        step,
        {"weight": "weight"},
    ) == ["state weight[second_moment]: DIFFERS max_abs=2.000e+00"]


def test_compare_eval_scores_both_models_and_restores_eval_tokens(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    theirs = torch.nn.Linear(3, 4)
    ours_model = torch.nn.Linear(3, 4)
    scores: list[tuple[torch.nn.Module, object, int, int]] = []
    tokenizer = object()
    prepare = _EvaluationPrepareStub(eval_tokens=99, max_seq_len=5)

    def evaluate(
        model: torch.nn.Module,
        actual_tokenizer: object,
        rows: int,
    ) -> float:
        scores.append((model, actual_tokenizer, rows, prepare.EVAL_TOKENS))
        return 1.25

    upstream = types.SimpleNamespace(
        tokenizer=tokenizer,
        DEVICE_BATCH_SIZE=3,
        evaluate_bpb=evaluate,
    )
    step = cast(
        NanoChatTrainStep,
        types.SimpleNamespace(model=ours_model),
    )
    autocast_options: list[dict[str, object]] = []

    def autocast(**kwargs: object) -> contextlib.AbstractContextManager[None]:
        autocast_options.append(kwargs)
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.amp, "autocast", autocast)
    assert (
        karpathy_parity.compare_eval(
            theirs,
            step,
            cast(karpathy_parity._ReferenceModule, upstream),
            cast(karpathy_parity._PrepareModule, prepare),
            batches=3,
        )
        == 0
    )
    assert scores == [
        (theirs, tokenizer, 3, 45),
        (scores[1][0], tokenizer, 3, 45),
    ]
    adapter = scores[1][0]
    assert isinstance(adapter, karpathy_parity._LossAdapter)
    assert adapter.inner is ours_model
    assert autocast_options == [{"device_type": "cuda", "dtype": torch.bfloat16}]
    assert prepare.EVAL_TOKENS == 99
    output = capsys.readouterr().out
    assert "their metric: theirs=1.250000000 ours=1.250000000" in output


def test_compare_eval_counts_disagreement_and_restores_after_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    theirs = torch.nn.Linear(3, 4)

    def evaluate(
        model: torch.nn.Module,
        tokenizer: object,
        rows: int,
    ) -> float:
        del tokenizer, rows
        return 1.0 if model is theirs else 2.0

    upstream = types.SimpleNamespace(
        tokenizer=object(),
        DEVICE_BATCH_SIZE=2,
        evaluate_bpb=evaluate,
    )
    prepare = _EvaluationPrepareStub(eval_tokens=20, max_seq_len=5)
    step = cast(
        NanoChatTrainStep,
        types.SimpleNamespace(model=torch.nn.Linear(3, 4)),
    )

    def autocast(**kwargs: object) -> contextlib.AbstractContextManager[None]:
        del kwargs
        return contextlib.nullcontext()

    monkeypatch.setattr(torch.amp, "autocast", autocast)
    assert (
        karpathy_parity.compare_eval(
            theirs,
            step,
            cast(karpathy_parity._ReferenceModule, upstream),
            cast(karpathy_parity._PrepareModule, prepare),
            batches=2,
        )
        == 1
    )
    assert prepare.EVAL_TOKENS == 20
    assert "bpb DIFFERS by 1.000e+00" in capsys.readouterr().out


def test_git_runs_in_the_reference_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path | None, bool, bool, bool]] = []

    def run(
        arguments: list[str],
        *,
        cwd: Path | None = None,
        capture_output: bool = False,
        text: bool = False,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, cwd, check, capture_output, text))
        return subprocess.CompletedProcess(arguments, 0, "commit\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert karpathy_parity._git(tmp_path, "rev-parse", "HEAD") == "commit"
    assert calls == [(["git", "rev-parse", "HEAD"], tmp_path, True, True, True)]


def test_argument_defaults_and_overrides() -> None:
    parser = argparse.ArgumentParser()
    karpathy_parity._add_arguments(parser)
    defaults = cast(_ParsedArgs, parser.parse_args([]))
    assert (defaults.steps, defaults.warmup, defaults.budget_steps) == (20, 11, 191)
    assert (defaults.rows, defaults.eval_batches, defaults.device) == (8, 4, "cuda")
    assert defaults.clone == Path("/opt/scratch/karpathy-autoresearch")
    assert defaults.corpus == Path("/opt/scratch/datasets/nanochat-priml")
    help_text = parser.format_help()
    assert next(
        line for line in help_text.splitlines() if line.startswith("  --steps")
    ).endswith("Optimizer steps.")
    assert next(
        line for line in help_text.splitlines() if line.startswith("  --device")
    ).endswith("Device to compare on.")
    assert "--clone" in help_text
    assert "--corpus" in help_text
    changed = cast(
        _ParsedArgs,
        parser.parse_args(
            [
                "--steps",
                "3",
                "--warmup",
                "4",
                "--budget-steps",
                "5",
                "--rows",
                "2",
                "--eval-batches",
                "6",
                "--device",
                "cpu",
                "--clone",
                "reference",
                "--corpus",
                "corpus",
            ],
        ),
    )
    assert (
        changed.steps,
        changed.warmup,
        changed.budget_steps,
        changed.rows,
        changed.eval_batches,
        changed.device,
    ) == (3, 4, 5, 2, 6, "cpu")
    assert changed.clone == Path("reference")
    assert changed.corpus == Path("corpus")


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
