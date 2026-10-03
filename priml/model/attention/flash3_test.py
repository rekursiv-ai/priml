"""Tests for the FlashAttention 3 kernel and its prepared-artifact loader."""

from __future__ import annotations

from dataclasses import field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Final, override

import sys

from configgle import Fig
from torch import Tensor, nn

import pytest
import torch

from priml.cost import Cost, cost
from priml.model.attention import flash3
from priml.model.attention.flash3 import (
    Flash3Attention,
    Flash3Interface,
    Flash3UnavailableError,
    artifact_path,
    artifact_validation_error,
    expected_receipt,
    load_flash3,
    receipt_validation_error,
    runtime_files_error,
    runtime_receipt,
)
from priml.model.attention.kernel import SdpaNaive, attention_kernel_cost
from priml.testing.cost import assert_cost_matches_torch


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


FA3_MODULES: Final = ("flash_attn_interface", "flash_attn_3", "flash_attn_3._C")
"""The modules a load imports, which a test of the loader must not leak."""

INTERFACE_SOURCE: Final = (
    "def flash_attn_func(q, k, v, *, softmax_scale, causal, window_size):\n"
    "    return q\n"
)
"""A prepared interface that satisfies ``Flash3Interface`` and needs no GPU."""


@pytest.fixture
def fake_flash3(monkeypatch: pytest.MonkeyPatch) -> _FakeInterface:
    """Build kernels on a stand-in FA3, as if on an SM90 device."""
    interface = _FakeInterface()
    monkeypatch.setattr(flash3, "load_flash3", lambda: interface)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    return interface


@pytest.fixture
def isolated_imports(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Undo the ``sys.path`` entry and the FA3 modules a successful load adds."""
    monkeypatch.setattr(sys, "path", [*sys.path])
    saved = {name: sys.modules.pop(name) for name in FA3_MODULES if name in sys.modules}
    yield
    for name in FA3_MODULES:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


@pytest.mark.parametrize("window", [-1, 0, 2, 7])
def test_the_kernel_hands_fa3_the_window_and_flattens_leading_axes(
    fake_flash3: _FakeInterface,
    window: int,
) -> None:
    q = torch.randn(3, 7, 5, 4, 6)
    k, v = torch.randn(3, 7, 5, 2, 6), torch.randn(3, 7, 5, 2, 6)
    out = Flash3Attention.Config().make()(q, k, v, window=window, scale=0.5)
    torch.testing.assert_close(out, q)
    assert fake_flash3.calls == [((21, 5, 4, 6), (21, 5, 2, 6), 0.5, (window, 0))]


@pytest.mark.usefixtures("fake_flash3")
@pytest.mark.parametrize(
    ("is_causal", "attn_mask", "dropout_p"),
    [(False, None, 0.0), (True, torch.zeros(5, 7), 0.0), (True, None, 0.1)],
)
def test_the_kernel_refuses_what_fa3_cannot_express(
    *,
    is_causal: bool,
    attn_mask: Tensor | None,
    dropout_p: float,
) -> None:
    q = torch.randn(2, 5, 3, 4)
    with pytest.raises(ValueError, match="causal"):
        Flash3Attention.Config().make()(
            q,
            q,
            q,
            is_causal=is_causal,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
        )


def test_the_kernel_rejects_an_unqualified_revision() -> None:
    with pytest.raises(ValueError, match="revision identifies"):
        Flash3Attention.Config(revision="wrong").make()


def test_the_kernel_requires_sm90(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    with pytest.raises(Flash3UnavailableError, match=r"requires SM90; .* SM89"):
        Flash3Attention.Config().make()


@pytest.mark.parametrize("window", [-1, 0, 1, 4, 8, 12])
def test_the_analytical_cost_matches_a_torch_reference(window: int) -> None:
    """Exercise the real cost body; FA3 itself never runs."""
    reference = _FlashCostReference.Config()
    reference.window = window
    seq_len = 8 if window < 0 else min(window + 1, 8)
    assert_cost_matches_torch(
        reference,
        build_input=lambda: tuple(
            torch.randn(2, seq_len, 3, 4, requires_grad=True) for _ in range(3)
        ),
        seq_len=seq_len,
        batch_size=2,
        dtype=None,
        num_heads=3,
        channels_head=4,
    )


def test_the_cost_is_whole_invocations_by_batch() -> None:
    config = Flash3Attention.Config()
    single = cost(
        config,
        seq_len=8,
        batch_size=1,
        dtype=None,
        num_heads=3,
        channels_head=4,
    )
    batched = cost(
        config,
        seq_len=8,
        batch_size=2,
        dtype=None,
        num_heads=3,
        channels_head=4,
    )
    assert batched == single.tile(2)
    assert batched == attention_kernel_cost(
        seq_len=8,
        batch_size=2,
        dtype=None,
        num_heads=3,
        channels_head=4,
    )


def test_the_artifact_path_names_the_pinned_build(tmp_path: Path) -> None:
    path = artifact_path(cache_root=tmp_path)
    assert path.parent == tmp_path
    assert path.name == (
        "3da5f873029162763568db56546fee70a779fade"
        "-torch2.9.1-cu128-cxx11-x86_64-nanochat-hdim128-bf16-local"
    )


def test_the_receipt_pins_the_build_lane() -> None:
    assert expected_receipt(
        binary_sha256="b",
        interface_sha256="i",
        config_sha256="c",
    ) == {
        "source_revision": "3da5f873029162763568db56546fee70a779fade",
        "cutlass_revision": "dc4817921edda44a549197ff3a9dcf5df0636e7b",
        "torch": "2.9.1",
        "cuda": "12.8",
        "cxx11_abi": "true",
        "build_profile": "nanochat-hdim128-bf16-local",
        "binary_sha256": "b",
        "interface_sha256": "i",
        "config_sha256": "c",
    }


def test_receipt_validation_names_each_mismatch() -> None:
    expected = expected_receipt(
        binary_sha256="b",
        interface_sha256="i",
        config_sha256="c",
    )
    error = receipt_validation_error(
        {**expected, "binary_sha256": "x", "torch": "2.9.2", "extra": "y"},
        expected={key: value for key, value in expected.items() if key != "cuda"},
    )
    assert error == (
        "torch mismatch: expected 2.9.1, receipt 2.9.2; "
        "binary_sha256 mismatch: receipt x, actual b; "
        "unexpected receipt field cuda; unexpected receipt field extra"
    )
    assert receipt_validation_error({}, expected={"torch": "2.9.1"}) == (
        "missing receipt field torch"
    )


@pytest.mark.usefixtures("isolated_imports")
def test_a_prepared_artifact_validates_and_loads(tmp_path: Path) -> None:
    path = _write_prepared_artifact(tmp_path)
    assert artifact_validation_error(path) == ""
    interface = load_flash3(cache_root=tmp_path)
    assert isinstance(interface, Flash3Interface)
    assert Path(interface.__file__).is_relative_to(path)


@pytest.mark.usefixtures("isolated_imports")
def test_an_interface_without_the_entry_point_is_refused(tmp_path: Path) -> None:
    path = _write_prepared_artifact(tmp_path, interface="SENTINEL = 'prepared'\n")
    assert artifact_validation_error(path) == ""
    with pytest.raises(TypeError, match="FA3 must provide flash_attn_func"):
        load_flash3(cache_root=tmp_path)


@pytest.mark.parametrize(
    ("file", "change", "error"),
    [
        (
            "flash_attn_3/_C.abi3.so",
            "delete",
            r"exactly one flash_attn_3/_C\*\.so; found 0",
        ),
        (
            "flash_attn_3/_C.abi3.so",
            "directory",
            r"exactly one flash_attn_3/_C\*\.so; found 0",
        ),
        (
            "flash_attn_interface.py",
            "delete",
            r"missing required runtime files: flash_attn_interface\.py",
        ),
        ("READY", "delete", "missing READY receipt"),
        ("READY", "directory", "READY receipt is not a regular file"),
        ("READY", "binary", "READY receipt is not valid UTF-8"),
        (
            "READY",
            "prepend broken",
            "malformed READY receipt line 1; expected name=value",
        ),
        (
            "READY",
            "prepend =value",
            "malformed READY receipt line 1; field name is empty",
        ),
        (
            "READY",
            "prepend source_revision=x",
            "duplicate receipt field source_revision",
        ),
        (
            "flash_attn_3/_C.abi3.so",
            "rewrite",
            "binary_sha256 mismatch: receipt [0-9a-f]{64}, actual",
        ),
        ("flash_attn_interface.py", "rewrite", "interface_sha256 mismatch"),
        ("flash_attn_config.py", "rewrite", "config_sha256 mismatch"),
    ],
)
def test_a_damaged_artifact_is_refused_with_its_reason(
    tmp_path: Path,
    file: str,
    change: str,
    error: str,
) -> None:
    path = _write_prepared_artifact(tmp_path)
    _damage(path / file, change)
    assert artifact_validation_error(path)
    with pytest.raises(Flash3UnavailableError, match=f"{error}.*prepare_flash3"):
        load_flash3(cache_root=tmp_path)


def test_unreadable_or_vanishing_files_are_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prepared_artifact(tmp_path)

    def unreadable(file: Path, **kwargs: object) -> str:
        del file, kwargs
        raise OSError("no read")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_text", unreadable)
        assert (
            artifact_validation_error(path) == "could not read READY receipt: no read"
        )
    # The extension vanishes between the file check and the hashing.
    monkeypatch.setattr(flash3, "runtime_files_error", _no_runtime_file_error)
    (path / "flash_attn_3" / "_C.abi3.so").unlink()
    assert artifact_validation_error(path) == (
        "could not hash FA3 runtime files: "
        "expected exactly one flash_attn_3/_C*.so; found 0"
    )


def test_runtime_files_report_every_missing_file(tmp_path: Path) -> None:
    (tmp_path / "flash_attn_3").mkdir()
    for name in ("_C.abi3.so", "_C.cpython-312-x86_64-linux-gnu.so"):
        (tmp_path / "flash_attn_3" / name).write_bytes(b"extension")
    assert runtime_files_error(tmp_path) == (
        "missing required runtime files: flash_attn_interface.py, "
        "flash_attn_config.py; expected exactly one flash_attn_3/_C*.so; found 2"
    )
    with pytest.raises(FileNotFoundError, match="found 2"):
        runtime_receipt(tmp_path)


@pytest.mark.usefixtures("isolated_imports")
@pytest.mark.parametrize("name", ["flash_attn_3._C", "flash_attn_interface"])
def test_a_foreign_fa3_already_imported_is_refused(tmp_path: Path, name: str) -> None:
    _write_prepared_artifact(tmp_path)
    foreign = ModuleType(name)
    foreign.__file__ = "/foreign/module.py"
    sys.modules[name] = foreign
    with pytest.raises(
        Flash3UnavailableError,
        match=rf"{name} was already imported from /foreign/module\.py, outside",
    ):
        load_flash3(cache_root=tmp_path)
    foreign.__file__ = None
    with pytest.raises(Flash3UnavailableError, match="has no file path"):
        load_flash3(cache_root=tmp_path)


@pytest.mark.gpu_flash_attention
@pytest.mark.gpu_torch_cuda
@pytest.mark.parametrize("heads_kv", [4, 2])
@pytest.mark.parametrize("window", [-1, 0, 31, 64, 255])
def test_cuda_kernel_matches_fa3s_own_autograd(heads_kv: int, window: int) -> None:
    interface = _cuda_interface()
    kernel = Flash3Attention.Config().make()
    inputs = _cuda_inputs(heads_kv)
    expected = _output_and_grads(
        lambda q, k, v: interface.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=None,
            causal=True,
            window_size=(window, 0),
        ),
        inputs,
    )
    actual = _output_and_grads(lambda q, k, v: kernel(q, k, v, window=window), inputs)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.gpu_flash_attention
@pytest.mark.gpu_torch_cuda
@pytest.mark.compute_torch_compile
@pytest.mark.parametrize("heads_kv", [4, 2])
@pytest.mark.parametrize("window", [-1, 64])
def test_cuda_fullgraph_compile_matches_eager(heads_kv: int, window: int) -> None:
    _cuda_interface()
    torch.compiler.reset()
    kernel = Flash3Attention.Config().make()
    compiled = torch.compile(kernel, fullgraph=True)
    inputs = _cuda_inputs(heads_kv)
    expected = _output_and_grads(lambda q, k, v: kernel(q, k, v, window=window), inputs)
    actual = _output_and_grads(lambda q, k, v: compiled(q, k, v, window=window), inputs)
    torch.compiler.reset()
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


class _FlashCostReference(nn.Module):
    """Price the FA3 config while a torch kernel does the work, FA3 never built."""

    class Config(Fig["_FlashCostReference"]):
        estimate: Flash3Attention.Config = field(
            default_factory=Flash3Attention.Config,
        )
        """The FA3 config whose analytical cost is under test."""

        window: int = -1
        """Previous keys admitted in addition to the current position."""

        def cost(self, **kwargs: object) -> Cost:
            """Delegate accounting without constructing FA3."""
            return cost(self.estimate, window=self.window, **kwargs)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.window = config.window
        self.reference = SdpaNaive.Config().make()

    @override
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return self.reference(q, k, v, is_causal=True, window=self.window)


class _FakeInterface:
    """A stand-in FA3 that returns the queries and records what it was handed."""

    __file__ = __file__

    def __init__(self) -> None:
        self.calls: list[
            tuple[tuple[int, ...], tuple[int, ...], float | None, tuple[int, int]]
        ] = []

    def flash_attn_func(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        softmax_scale: float | None,
        causal: bool,
        window_size: tuple[int, int],
    ) -> Tensor:
        assert causal
        assert k.shape == v.shape
        self.calls.append((tuple(q.shape), tuple(k.shape), softmax_scale, window_size))
        return q.clone()


def _write_prepared_artifact(
    cache_root: Path,
    *,
    interface: str = INTERFACE_SOURCE,
) -> Path:
    """Write runtime files and their matching READY receipt; return the artifact."""
    path = artifact_path(cache_root=cache_root)
    (path / "flash_attn_3").mkdir(parents=True)
    (path / "flash_attn_3" / "_C.abi3.so").write_bytes(b"prepared extension")
    (path / "flash_attn_interface.py").write_text(interface, encoding="utf-8")
    (path / "flash_attn_config.py").write_text("CONFIG = {}\n", encoding="utf-8")
    (path / "READY").write_text(
        "".join(f"{name}={value}\n" for name, value in runtime_receipt(path).items()),
        encoding="utf-8",
    )
    return path


def _damage(file: Path, change: str) -> None:
    """Delete, replace with a directory, garble, rewrite, or prepend to ``file``."""
    if change in {"delete", "directory"}:
        file.unlink()
        if change == "directory":
            file.mkdir()
    elif change == "binary":
        file.write_bytes(b"\xff")
    elif change == "rewrite":
        file.write_bytes(b"changed")
    else:
        line = change.removeprefix("prepend ")
        file.write_text(f"{line}\n{file.read_text(encoding='utf-8')}", encoding="utf-8")


def _no_runtime_file_error(path: Path) -> str:
    del path
    return ""


def _cuda_interface() -> Flash3Interface:
    """Return the prepared FA3, skipping where it cannot run."""
    if not torch.cuda.is_available():
        pytest.skip("Requires an SM90 CUDA device and a prepared FA3 artifact.")
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Requires an SM90 CUDA device.")
    if error := artifact_validation_error(artifact_path()):
        pytest.skip(f"Requires a prepared FA3 artifact: {error}.")
    return load_flash3()


def _cuda_inputs(heads_kv: int) -> list[Tensor]:
    """Return bf16 ``q``, ``k``, ``v`` and a cotangent at FA3's built head width."""
    generator = torch.Generator(device="cuda").manual_seed(11)
    shapes = [(3, 256, 4, 128), (3, 256, heads_kv, 128), (3, 256, heads_kv, 128)]
    return [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
        for shape in (*shapes, shapes[0])
    ]


def _output_and_grads(
    attend: Callable[[Tensor, Tensor, Tensor], Tensor],
    inputs: list[Tensor],
) -> list[Tensor]:
    """Return the output and the gradients of ``q``, ``k``, ``v`` for a cotangent."""
    *tensors, cotangent = inputs
    leaves = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    out = attend(*leaves)
    return [out.detach(), *torch.autograd.grad(out, leaves, cotangent)]


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
