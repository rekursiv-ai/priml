"""Tests for the local FlashAttention-3 preparation command."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import argparse
import errno
import os
import platform
import shutil
import subprocess
import sys

import pytest

from priml.model.attention import prepare_flash3
from priml.model.attention.flash3 import (
    cutlass_revision,
    runtime_receipt,
    source_revision,
)


class _Arguments(Protocol):
    cache_root: Path


@dataclass(kw_only=True, slots=True)
class _RuntimeVersion:
    cuda: str


@dataclass(kw_only=True, slots=True)
class _Runtime:
    __version__: str
    version: _RuntimeVersion
    compiled_with_cxx11_abi: Callable[[], bool]


def _no_run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> None:
    del command, cwd, environment


def _no_path_error(path: Path) -> str:
    del path
    return ""


def _missing_receipt(path: Path) -> str:
    del path
    return "missing READY receipt"


def _missing_extension(path: Path) -> str:
    del path
    return "missing extension"


def _no_staging(path: Path) -> None:
    del path


def test_build_environment_pins_the_profile_and_preserves_path() -> None:
    assert prepare_flash3._build_environment({"PATH": "/usr/bin"}) == {
        "PATH": "/usr/local/cuda-12.8/bin" + os.pathsep + "/usr/bin",
        "CUDA_HOME": "/usr/local/cuda-12.8",
        "MAX_JOBS": "32",
        "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
        "FLASH_ATTENTION_FORCE_CXX11_ABI": "TRUE",
        "FLASH_ATTENTION_OFFLINE_BUILD": "TRUE",
        "FLASH_ATTENTION_DISABLE_SM80": "TRUE",
        "FLASH_ATTENTION_DISABLE_FP16": "TRUE",
        "FLASH_ATTENTION_DISABLE_FP8": "TRUE",
        "FLASH_ATTENTION_DISABLE_SPLIT": "TRUE",
        "FLASH_ATTENTION_DISABLE_PAGEDKV": "TRUE",
        "FLASH_ATTENTION_DISABLE_APPENDKV": "TRUE",
        "FLASH_ATTENTION_DISABLE_SOFTCAP": "TRUE",
        "FLASH_ATTENTION_DISABLE_PACKGQA": "TRUE",
        "FLASH_ATTENTION_DISABLE_VARLEN": "TRUE",
        "FLASH_ATTENTION_DISABLE_CLUSTER": "TRUE",
        "FLASH_ATTENTION_DISABLE_HDIM64": "TRUE",
        "FLASH_ATTENTION_DISABLE_HDIM96": "TRUE",
        "FLASH_ATTENTION_DISABLE_HDIM192": "TRUE",
        "FLASH_ATTENTION_DISABLE_HDIM256": "TRUE",
        "FLASH_ATTENTION_DISABLE_HDIMDIFF64": "TRUE",
        "FLASH_ATTENTION_DISABLE_HDIMDIFF192": "TRUE",
    }


def test_build_environment_works_without_an_inherited_path() -> None:
    environment = prepare_flash3._build_environment({"PRESERVED": "yes"})
    assert environment["PATH"] == "/usr/local/cuda-12.8/bin"
    assert environment["PRESERVED"] == "yes"


def test_nvcc_path_prefers_path_then_provisioned_toolkit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def which_path(name: str) -> str | None:
        assert name == "nvcc"
        return "/custom/nvcc"

    def which_none(name: str) -> str | None:
        assert name == "nvcc"
        return None

    def is_file(path: Path) -> bool:
        del path
        return True

    monkeypatch.setattr(shutil, "which", which_path)
    assert prepare_flash3._nvcc_path() == Path("/custom/nvcc")

    monkeypatch.setattr(shutil, "which", which_none)
    monkeypatch.setattr(Path, "is_file", is_file)
    assert prepare_flash3._nvcc_path() == Path("/usr/local/cuda-12.8/bin/nvcc")


def test_nvcc_path_names_missing_toolkit(monkeypatch: pytest.MonkeyPatch) -> None:
    def which_none(name: str) -> str | None:
        assert name == "nvcc"
        return None

    def is_file(path: Path) -> bool:
        del path
        return False

    monkeypatch.setattr(shutil, "which", which_none)
    monkeypatch.setattr(Path, "is_file", is_file)
    with pytest.raises(RuntimeError) as error:
        prepare_flash3._nvcc_path()
    assert str(error.value) == (
        "FA3 source preparation requires nvcc 12.8 on PATH or at "
        "/usr/local/cuda-12.8/bin/nvcc."
    )


def test_main_rejects_missing_module_docstring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare_flash3, "__doc__", None)
    with pytest.raises(
        ValueError,
        match=r"^Expected __doc__ is not None\.$",
    ) as error:
        prepare_flash3.main()
    assert str(error.value) == "Expected __doc__ is not None."


def test_cli_parses_cache_root_and_prints_prepared_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = argparse.ArgumentParser()
    prepare_flash3._add_arguments(parser)
    default_args = cast(_Arguments, parser.parse_args([]))
    explicit_args = cast(_Arguments, parser.parse_args(["--cache-root", str(tmp_path)]))
    assert default_args.cache_root == Path("/opt/scratch/caches/nanochat/fa3")
    assert explicit_args.cache_root == tmp_path

    descriptions: list[str | None] = []
    add_arguments = prepare_flash3._add_arguments

    def capture_parser(parser: argparse.ArgumentParser) -> None:
        descriptions.append(parser.description)
        add_arguments(parser)

    monkeypatch.setattr(prepare_flash3, "_add_arguments", capture_parser)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_flash3.py", "--cache-root", str(tmp_path)],
    )

    def prepare(*, cache_root: Path) -> Path:
        return cache_root / "prepared"

    monkeypatch.setattr(prepare_flash3, "prepare_flash3", prepare)
    assert prepare_flash3.main() == 0
    assert prepare_flash3.__doc__ is not None
    assert descriptions == [prepare_flash3.__doc__.split("\n", 2)[2]]
    assert capsys.readouterr().out == f"{tmp_path / 'prepared'}\n"


def test_run_wrappers_forward_arguments_and_strip_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="  result\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    command = ["git", "status"]
    environment = {"PATH": "/bin"}
    prepare_flash3._run(command, cwd=tmp_path, environment=environment)
    assert prepare_flash3._run_output(command) == "result"
    assert calls == [
        (command, {"check": True, "cwd": tmp_path, "env": environment}),
        (command, {"check": True, "capture_output": True, "text": True}),
    ]


def test_validate_build_runtime_pins_platform_torch_abi_and_nvcc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    runtime = _Runtime(
        __version__="2.9.1+cu128+build",
        version=_RuntimeVersion(cuda="12.8"),
        compiled_with_cxx11_abi=lambda: True,
    )
    monkeypatch.setattr(prepare_flash3, "torch", runtime)
    monkeypatch.setattr(prepare_flash3, "_nvcc_path", lambda: Path("/nvcc"))
    observed: list[list[str]] = []

    def run_output(command: list[str]) -> str:
        observed.append(command)
        return "Cuda compilation tools, release 12.8, V12.8.93"

    monkeypatch.setattr(prepare_flash3, "_run_output", run_output)
    prepare_flash3._validate_build_runtime()
    assert observed == [["/nvcc", "--version"]]

    for system, machine in (("Darwin", "x86_64"), ("Linux", "aarch64")):
        monkeypatch.setattr(
            platform,
            "system",
            lambda system=system: system,
        )
        monkeypatch.setattr(
            platform,
            "machine",
            lambda machine=machine: machine,
        )
        with pytest.raises(
            RuntimeError,
            match=r"^FA3 must be built on x86_64 Linux\.$",
        ):
            prepare_flash3._validate_build_runtime()
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    runtime.__version__ = "2.9.2+cu128"
    with pytest.raises(
        RuntimeError,
        match=r"^FA3 requires Torch 2\.9\.1; found 2\.9\.2\+cu128\. Build it in the isolated runtime: ",
    ):
        prepare_flash3._validate_build_runtime()
    runtime.__version__ = "2.9.1+cu128"
    runtime.version.cuda = "12.7"
    with pytest.raises(RuntimeError, match=r"^FA3 requires CUDA 12.8; found 12.7\.$"):
        prepare_flash3._validate_build_runtime()
    runtime.version.cuda = "12.8"
    runtime.compiled_with_cxx11_abi = lambda: False
    with pytest.raises(
        RuntimeError,
        match=r"^FA3 requires the Torch C\+\+11 ABI runtime\.$",
    ):
        prepare_flash3._validate_build_runtime()
    runtime.compiled_with_cxx11_abi = lambda: True

    def nvcc_output(command: list[str]) -> str:
        del command
        return "Cuda compilation tools, release 12.7, V12.7.0"

    monkeypatch.setattr(prepare_flash3, "_run_output", nvcc_output)
    with pytest.raises(RuntimeError, match=r"^FA3 requires nvcc 12.8; found:"):
        prepare_flash3._validate_build_runtime()


def test_write_receipt_records_hashes_of_all_runtime_files(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    extension = artifact / "flash_attn_3" / "_C.abi3.so"
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"extension")
    (artifact / "flash_attn_interface.py").write_bytes(b"interface")
    (artifact / "flash_attn_config.py").write_bytes(b"config")

    prepare_flash3._write_receipt(artifact)

    expected = runtime_receipt(artifact)
    receipt_bytes = (artifact / "READY").read_bytes()
    assert receipt_bytes == "".join(
        f"{name}={value}\n" for name, value in expected.items()
    ).encode("utf-8")


def test_build_flash3_pins_checkout_and_unpacks_the_single_wheel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "build" / "artifact"
    destination.mkdir(parents=True)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    outputs = iter((source_revision(), cutlass_revision()))
    output_commands: list[list[str]] = []
    glob_calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(prepare_flash3, "_run", run)

    def run_output(command: list[str]) -> str:
        output_commands.append(command)
        return next(outputs)

    def glob(path: Path, pattern: str) -> list[Path]:
        glob_calls.append((path, pattern))
        return [tmp_path / "fa3.whl"]

    def unpack_archive(filename: str, extract_dir: Path, **kwargs: str) -> None:
        assert kwargs == {"format": "zip"}
        unpacked.append((filename, extract_dir, kwargs["format"]))

    monkeypatch.setattr(prepare_flash3, "_run_output", run_output)
    monkeypatch.setattr(Path, "glob", glob)
    unpacked: list[tuple[str, Path, str]] = []
    monkeypatch.setattr(shutil, "unpack_archive", unpack_archive)

    prepare_flash3._build_flash3(destination)

    source = destination.parent / "source"
    wheels = destination.parent / "wheels"
    assert calls == [
        (["git", "init", str(source)], {}),
        (
            [
                "git",
                "-C",
                str(source),
                "remote",
                "add",
                "origin",
                "https://github.com/varunneal/flash-attention.git",
            ],
            {},
        ),
        (
            [
                "git",
                "-C",
                str(source),
                "fetch",
                "--depth=1",
                "origin",
                source_revision(),
            ],
            {},
        ),
        (["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], {}),
        (
            ["git", "-C", str(source), "submodule", "update", "--init", "csrc/cutlass"],
            {},
        ),
        (
            [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(wheels)],
            {
                "cwd": source / "hopper",
                "environment": prepare_flash3._build_environment(os.environ),
            },
        ),
    ]
    assert output_commands == [
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        ["git", "-C", str(source / "csrc" / "cutlass"), "rev-parse", "HEAD"],
    ]
    assert glob_calls == [(wheels, "*.whl")]
    assert unpacked == [(str(tmp_path / "fa3.whl"), destination, "zip")]


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        (
            ("wrong-source", cutlass_revision()),
            "FA3 source checkout does not match the pinned revision.",
        ),
        (
            (source_revision(), "wrong-cutlass"),
            "FA3 CUTLASS checkout does not match the pinned revision.",
        ),
    ],
)
def test_build_flash3_rejects_unpinned_revisions(
    outputs: tuple[str, str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "build" / "artifact"
    destination.parent.mkdir()
    monkeypatch.setattr(prepare_flash3, "_run", _no_run)
    revisions = iter(outputs)

    def run_output(command: list[str]) -> str:
        del command
        return next(revisions)

    monkeypatch.setattr(prepare_flash3, "_run_output", run_output)
    with pytest.raises(RuntimeError) as error:
        prepare_flash3._build_flash3(destination)
    assert str(error.value) == message


@pytest.mark.parametrize("wheel_count", [0, 2])
def test_build_flash3_requires_exactly_one_wheel(
    wheel_count: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "build" / "artifact"
    destination.parent.mkdir()
    monkeypatch.setattr(prepare_flash3, "_run", _no_run)
    revisions = iter((source_revision(), cutlass_revision()))

    def run_output(command: list[str]) -> str:
        del command
        return next(revisions)

    def glob(self: Path, pattern: str) -> list[Path]:
        del self, pattern
        return wheels

    monkeypatch.setattr(prepare_flash3, "_run_output", run_output)
    wheels = [tmp_path / f"{index}.whl" for index in range(wheel_count)]
    monkeypatch.setattr(Path, "glob", glob)
    with pytest.raises(RuntimeError) as error:
        prepare_flash3._build_flash3(destination)
    assert str(error.value) == f"Expected one FA3 wheel, found {wheel_count}."


def test_prepare_flash3_rejects_an_invalid_existing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "ready"
    destination.mkdir()

    def artifact_path(*, cache_root: Path) -> Path:
        assert cache_root == tmp_path
        return destination

    def validation_error(path: Path) -> str:
        del path
        return "missing READY receipt"

    monkeypatch.setattr(prepare_flash3, "artifact_path", artifact_path)
    monkeypatch.setattr(prepare_flash3, "artifact_validation_error", validation_error)
    with pytest.raises(FileExistsError) as error:
        prepare_flash3.prepare_flash3(cache_root=tmp_path)
    assert str(error.value) == (
        f"FA3 artifact at {destination} failed validation: missing READY receipt. "
        "Remove only this content-addressed directory, then prepare again."
    )


def test_prepare_flash3_returns_a_valid_existing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "ready"

    def artifact_path(*, cache_root: Path) -> Path:
        assert cache_root == tmp_path
        return destination

    def artifact_validation_error(path: Path) -> str:
        assert path == destination
        return ""

    monkeypatch.setattr(prepare_flash3, "artifact_path", artifact_path)
    monkeypatch.setattr(
        prepare_flash3,
        "artifact_validation_error",
        artifact_validation_error,
    )
    monkeypatch.setattr(
        prepare_flash3,
        "_validate_build_runtime",
        lambda: pytest.fail("valid artifact should not rebuild"),
    )
    assert prepare_flash3.prepare_flash3(cache_root=tmp_path) == destination


def test_prepare_flash3_reports_invalid_installed_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    destination = cache_root / "artifact"
    validations = iter(("missing READY receipt", "bad installed artifact"))

    def artifact_validation_error(path: Path) -> str:
        assert path == destination
        return next(validations)

    def artifact_path(*, cache_root: Path) -> Path:
        return cache_root / destination.name

    monkeypatch.setattr(prepare_flash3, "artifact_path", artifact_path)
    monkeypatch.setattr(
        prepare_flash3,
        "artifact_validation_error",
        artifact_validation_error,
    )
    monkeypatch.setattr(prepare_flash3, "runtime_files_error", _no_path_error)
    monkeypatch.setattr(prepare_flash3, "_validate_build_runtime", lambda: None)
    monkeypatch.setattr(prepare_flash3, "_build_flash3", _no_staging)
    monkeypatch.setattr(prepare_flash3, "_write_receipt", _no_staging)
    with pytest.raises(RuntimeError) as error:
        prepare_flash3.prepare_flash3(cache_root=cache_root)
    assert str(error.value) == (
        f"Prepared FA3 artifact at {destination} failed validation: "
        "bad installed artifact."
    )


def test_prepare_flash3_reports_invalid_staged_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    destination = cache_root / "artifact"

    def artifact_path(*, cache_root: Path) -> Path:
        return cache_root / destination.name

    monkeypatch.setattr(prepare_flash3, "artifact_path", artifact_path)
    monkeypatch.setattr(
        prepare_flash3,
        "artifact_validation_error",
        _missing_receipt,
    )
    monkeypatch.setattr(prepare_flash3, "_validate_build_runtime", lambda: None)
    monkeypatch.setattr(prepare_flash3, "_build_flash3", _no_staging)
    monkeypatch.setattr(prepare_flash3, "runtime_files_error", _missing_extension)
    with pytest.raises(RuntimeError) as error:
        prepare_flash3.prepare_flash3(cache_root=cache_root)
    assert str(error.value) == "FA3 build produced invalid files: missing extension."


@pytest.mark.parametrize(
    ("errno_value", "race_validation", "expected"),
    [
        (errno.EEXIST, "", "installed"),
        (errno.EPERM, "", "raised"),
        (errno.EEXIST, "invalid race artifact", "raised"),
    ],
)
def test_prepare_flash3_handles_atomic_install_races(
    errno_value: int,
    race_validation: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    destination = cache_root / "artifact"
    validations = iter(("missing READY receipt", race_validation, ""))

    def artifact_path(*, cache_root: Path) -> Path:
        return cache_root / destination.name

    monkeypatch.setattr(prepare_flash3, "artifact_path", artifact_path)

    def artifact_validation_error(path: Path) -> str:
        assert path == destination
        return next(validations)

    monkeypatch.setattr(
        prepare_flash3,
        "artifact_validation_error",
        artifact_validation_error,
    )
    monkeypatch.setattr(prepare_flash3, "runtime_files_error", _no_path_error)
    monkeypatch.setattr(prepare_flash3, "_validate_build_runtime", lambda: None)
    monkeypatch.setattr(prepare_flash3, "_build_flash3", _no_staging)
    monkeypatch.setattr(prepare_flash3, "_write_receipt", _no_staging)

    def replace(self: Path, target: Path) -> Path:
        del self, target
        raise OSError(errno_value, "atomic install collision")

    monkeypatch.setattr(Path, "replace", replace)
    if expected == "raised":
        with pytest.raises(
            OSError,
            match=rf"^\[Errno {errno_value}\] atomic install collision$",
        ) as error:
            prepare_flash3.prepare_flash3(cache_root=cache_root)
        assert (error.value.errno, str(error.value)) == (
            errno_value,
            f"[Errno {errno_value}] atomic install collision",
        )
    else:
        assert prepare_flash3.prepare_flash3(cache_root=cache_root) == destination


@pytest.mark.parametrize("precreate_cache_root", [False, True])
def test_prepare_flash3_installs_a_built_artifact_atomically(
    precreate_cache_root: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "nested" / "cache"
    if precreate_cache_root:
        cache_root.mkdir(parents=True)
    destination = cache_root / "artifact"
    validations = iter(("missing READY receipt", ""))

    def artifact_path(*, cache_root: Path) -> Path:
        return cache_root / "artifact"

    def artifact_validation_error(path: Path) -> str:
        assert path == destination
        return next(validations)

    def runtime_files_error(path: Path) -> str:
        assert path.name == "artifact"
        return ""

    def build(staging: Path) -> None:
        assert staging.name == "artifact"
        assert staging.parent.parent == cache_root
        assert staging.parent.name.startswith(".fa3-build-")
        (staging / "built").write_text("yes", encoding="utf-8")

    def write_receipt(path: Path) -> None:
        assert path.name == "artifact"

    monkeypatch.setattr(prepare_flash3, "artifact_path", artifact_path)
    monkeypatch.setattr(
        prepare_flash3,
        "artifact_validation_error",
        artifact_validation_error,
    )
    monkeypatch.setattr(prepare_flash3, "runtime_files_error", runtime_files_error)
    monkeypatch.setattr(prepare_flash3, "_validate_build_runtime", lambda: None)
    monkeypatch.setattr(prepare_flash3, "_build_flash3", build)
    monkeypatch.setattr(prepare_flash3, "_write_receipt", write_receipt)

    assert prepare_flash3.prepare_flash3(cache_root=cache_root) == destination
    assert (destination / "built").read_text(encoding="utf-8") == "yes"


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
