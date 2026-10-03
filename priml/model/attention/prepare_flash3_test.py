"""Tests for the FlashAttention 3 build script, with the compiler stubbed out."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Barrier
from typing import TYPE_CHECKING

import platform
import shutil
import sys
import zipfile

import pytest
import torch

from priml.model.attention import prepare_flash3
from priml.model.attention.flash3 import (
    artifact_path,
    artifact_validation_error,
    cutlass_revision,
    runtime_receipt,
    source_revision,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Stub the compiler with one writing the runtime files; record each build."""
    destinations: list[Path] = []
    monkeypatch.setattr(prepare_flash3, "_validate_build_runtime", lambda: None)
    monkeypatch.setattr(
        prepare_flash3,
        "_build_flash3",
        partial(_write_runtime_files, record=destinations),
    )
    return destinations


def test_prepare_builds_once_then_reuses_the_artifact(
    builds: list[Path],
    tmp_path: Path,
) -> None:
    first = prepare_flash3.prepare_flash3(cache_root=tmp_path)
    second = prepare_flash3.prepare_flash3(cache_root=tmp_path)
    assert first == second == artifact_path(cache_root=tmp_path)
    assert len(builds) == 1
    assert artifact_validation_error(first) == ""
    assert (first / "READY").read_text(encoding="utf-8") == "".join(
        f"{name}={value}\n" for name, value in runtime_receipt(first).items()
    )


def test_concurrent_prepares_adopt_the_installed_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    builds_ready = Barrier(2)
    monkeypatch.setattr(prepare_flash3, "_validate_build_runtime", lambda: None)
    monkeypatch.setattr(
        prepare_flash3,
        "_build_flash3",
        partial(_write_runtime_files, record=[], barrier=builds_ready),
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(prepare_flash3.prepare_flash3, cache_root=tmp_path)
            for _ in range(2)
        ]
        prepared = tuple(future.result(timeout=5) for future in futures)
    assert prepared == (artifact_path(cache_root=tmp_path),) * 2
    assert artifact_validation_error(prepared[0]) == ""


@pytest.mark.usefixtures("builds")
def test_prepare_refuses_to_replace_an_invalid_artifact(tmp_path: Path) -> None:
    artifact_path(cache_root=tmp_path).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="failed validation: missing required"):
        prepare_flash3.prepare_flash3(cache_root=tmp_path)


def test_prepare_refuses_a_build_without_runtime_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(prepare_flash3, "_validate_build_runtime", lambda: None)
    monkeypatch.setattr(prepare_flash3, "_build_flash3", _build_nothing)
    with pytest.raises(RuntimeError, match="FA3 build produced invalid files"):
        prepare_flash3.prepare_flash3(cache_root=tmp_path)
    assert not artifact_path(cache_root=tmp_path).exists()


def test_main_prints_the_prepared_artifact(
    builds: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["prepare_flash3", "--cache-root", str(tmp_path)])
    assert prepare_flash3.main() == 0
    assert capsys.readouterr().out == f"{artifact_path(cache_root=tmp_path)}\n"
    assert len(builds) == 1


def test_the_build_checks_out_both_pins_and_unpacks_the_wheel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands = _stub_commands(monkeypatch, source=source_revision()).commands
    destination = tmp_path / "artifact"
    destination.mkdir()
    prepare_flash3._build_flash3(destination)
    source = str(tmp_path / "source")
    assert ["git", "-C", source, "fetch", "--depth=1", "origin", source_revision()] in (
        commands
    )
    assert ["git", "-C", source, "submodule", "update", "--init", "csrc/cutlass"] in (
        commands
    )
    assert runtime_receipt(destination)["source_revision"] == source_revision()


@pytest.mark.parametrize(
    ("source", "cutlass", "error"),
    [
        ("0" * 40, cutlass_revision(), "FA3 source checkout"),
        (source_revision(), "0" * 40, "FA3 CUTLASS checkout"),
    ],
)
def test_the_build_refuses_a_checkout_off_its_pins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
    cutlass: str,
    error: str,
) -> None:
    _stub_commands(monkeypatch, source=source, cutlass=cutlass)
    destination = tmp_path / "artifact"
    destination.mkdir()
    with pytest.raises(RuntimeError, match=f"{error} does not match the pinned"):
        prepare_flash3._build_flash3(destination)


def test_build_environment_matches_qualified_hopper_lane() -> None:
    environment = prepare_flash3._build_environment({"PATH": "/usr/bin"})
    assert environment["PATH"] == "/usr/local/cuda-12.8/bin:/usr/bin"
    assert environment["CUDA_HOME"] == "/usr/local/cuda-12.8"
    assert environment["MAX_JOBS"] == "32"
    for flag in (
        "FORCE_BUILD",
        "FORCE_CXX11_ABI",
        "OFFLINE_BUILD",
        "DISABLE_SM80",
        "DISABLE_FP16",
        "DISABLE_FP8",
        "DISABLE_SPLIT",
        "DISABLE_PAGEDKV",
        "DISABLE_APPENDKV",
        "DISABLE_SOFTCAP",
        "DISABLE_PACKGQA",
        "DISABLE_VARLEN",
        "DISABLE_CLUSTER",
        "DISABLE_HDIM64",
        "DISABLE_HDIM96",
        "DISABLE_HDIM192",
        "DISABLE_HDIM256",
        "DISABLE_HDIMDIFF64",
        "DISABLE_HDIMDIFF192",
    ):
        assert environment[f"FLASH_ATTENTION_{flag}"] == "TRUE"
    for kept in ("HDIM128", "LOCAL", "BACKWARD"):
        assert f"FLASH_ATTENTION_DISABLE_{kept}" not in environment
    assert prepare_flash3._build_environment({})["PATH"] == "/usr/local/cuda-12.8/bin"


def test_the_pinned_build_runtime_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_lane(monkeypatch, {})
    prepare_flash3._validate_build_runtime()


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"system": "Darwin"}, "must be built on x86_64 Linux"),
        (
            {"version": "2.9.10"},
            (
                r"Torch 2\.9\.1; found 2\.9\.10\. .*--isolated --project "
                r"priml/baselines/nanochat/runtime"
            ),
        ),
        ({"cuda": "12.9"}, r"CUDA 12\.8; found 12\.9"),
        ({"cxx11_abi": False}, r"C\+\+11 ABI"),
        ({"nvcc": "release 12.9"}, "nvcc 12.8"),
    ],
)
def test_any_other_build_runtime_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
    error: str,
) -> None:
    _stub_lane(monkeypatch, change)
    with pytest.raises(RuntimeError, match=error):
        prepare_flash3._validate_build_runtime()


@pytest.mark.parametrize(
    ("on_path", "provisioned", "expected"),
    [
        ("/opt/cuda/bin/nvcc", False, "/opt/cuda/bin/nvcc"),
        (None, True, "/usr/local/cuda-12.8/bin/nvcc"),
        (None, False, None),
    ],
)
def test_nvcc_comes_from_path_then_the_provisioned_toolkit(
    monkeypatch: pytest.MonkeyPatch,
    on_path: str | None,
    provisioned: bool,
    expected: str | None,
) -> None:
    def which(executable: str) -> str | None:
        assert executable == "nvcc"
        return on_path

    def is_file(path: Path) -> bool:
        return provisioned and path == Path("/usr/local/cuda-12.8/bin/nvcc")

    monkeypatch.setattr(shutil, "which", which)
    monkeypatch.setattr(Path, "is_file", is_file)
    if expected is None:
        with pytest.raises(RuntimeError, match=r"requires nvcc 12\.8"):
            prepare_flash3._nvcc_path()
    else:
        assert prepare_flash3._nvcc_path() == Path(expected)


def _stub_lane(monkeypatch: pytest.MonkeyPatch, change: Mapping[str, object]) -> None:
    """Present the qualified build lane, with ``change`` applied to it."""
    lane: dict[str, object] = {
        "system": "Linux",
        "version": "2.9.1+cu128",
        "cuda": "12.8",
        "cxx11_abi": True,
        "nvcc": "Cuda compilation tools, release 12.8, V12.8.93",
        **change,
    }
    monkeypatch.setattr(platform, "system", lambda: lane["system"])
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(torch, "__version__", lane["version"])
    monkeypatch.setattr(torch.version, "cuda", lane["cuda"])
    monkeypatch.setattr(torch, "compiled_with_cxx11_abi", lambda: lane["cxx11_abi"])
    monkeypatch.setattr(prepare_flash3, "_nvcc_path", lambda: Path("/bin/nvcc"))

    def nvcc_version(command: list[str]) -> str:
        assert command == ["/bin/nvcc", "--version"]
        return str(lane["nvcc"])

    monkeypatch.setattr(prepare_flash3, "_run_output", nvcc_version)


def _build_nothing(destination: Path) -> None:
    del destination


def _write_runtime_files(
    destination: Path,
    *,
    record: list[Path],
    barrier: Barrier | None = None,
) -> None:
    """Write what a build installs, as a stand-in for compiling it."""
    record.append(destination)
    (destination / "flash_attn_3").mkdir()
    (destination / "flash_attn_3" / "_C.abi3.so").write_bytes(b"extension")
    for name in ("flash_attn_interface.py", "flash_attn_config.py"):
        (destination / name).write_text(f"# {name}\n", encoding="utf-8")
    if barrier is not None:
        barrier.wait(timeout=5)


def _stub_commands(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source: str,
    cutlass: str = "",
) -> _Commands:
    """Route the build's commands to a recorder answering at the given heads."""
    commands = _Commands(
        heads={"source": source, "cutlass": cutlass or cutlass_revision()},
    )
    monkeypatch.setattr(prepare_flash3, "_run", commands.run)
    monkeypatch.setattr(prepare_flash3, "_run_output", commands.output)
    return commands


class _Commands:
    """Record the build's commands, answer ``rev-parse``, and write the wheel."""

    def __init__(self, *, heads: dict[str, str]) -> None:
        self.heads = heads
        self.commands: list[list[str]] = []

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        del cwd, environment
        self.commands.append(command)
        if "bdist_wheel" in command:
            _write_wheel(Path(command[-1]) / "flash_attn_3-3.0.0b1-cp39-abi3.whl")

    def output(self, command: list[str]) -> str:
        return self.heads[Path(command[2]).name]


def _write_wheel(path: Path) -> None:
    """Write a wheel holding the runtime files a build installs."""
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("flash_attn_3/_C.abi3.so", b"extension")
        wheel.writestr("flash_attn_interface.py", "# interface\n")
        wheel.writestr("flash_attn_config.py", "# config\n")


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
