#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Build the pinned FlashAttention-3 source once and install it under the cache.

Clones the pinned FA3 revision, checks the CUTLASS submodule against its pin,
builds the SM90 wheel with the nanochat hdim128/bf16 profile, and installs it
atomically under the content-addressed artifact path that
``priml.baselines.nanochat.attention.load_flash3`` reads. Needs x86_64
Linux, torch 2.9.1+cu128 with the C++11 ABI, and nvcc 12.8.

Examples:
  uv --quiet run --frozen python -m priml.baselines.nanochat.scripts.prepare_flash3

'''
# fmt: on

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import argparse
import errno
import os
import platform
import shutil
import subprocess
import sys
import tempfile

from priml.baselines.nanochat.attention import (
    artifact_path,
    artifact_validation_error,
    cutlass_revision,
    runtime_files_error,
    runtime_receipt,
    source_revision,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    import torch
else:
    from wrapt import lazy_import

    torch = lazy_import("torch")  # ~1050 ms; only the build-runtime check reads it.


def main() -> int:
    """Run the program.

    Returns:
      code: Process exit code.

    """
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n", 2)[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args())
    print(prepare_flash3(cache_root=flags.cache_root))
    return 0


class Flags(Protocol):
    """Parsed command-line flags."""

    cache_root: Path


def prepare_flash3(
    *,
    cache_root: Path = Path("/opt/scratch/caches/nanochat/fa3"),
) -> Path:
    """Build the pinned FA3 source once and atomically install it.

    Args:
      cache_root: Stable node-local cache root.

    Returns:
      path: Prepared local artifact directory.

    Raises:
      FileExistsError: An incomplete artifact already occupies the target.
      RuntimeError: The build runtime or generated artifact is invalid.

    """
    destination = artifact_path(cache_root=cache_root)
    validation_error = artifact_validation_error(destination)
    if not validation_error:
        return destination
    if destination.exists():
        raise FileExistsError(
            f"FA3 artifact at {destination} failed validation: {validation_error}. "
            "Remove only this content-addressed directory, then prepare again.",
        )

    _validate_build_runtime()
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fa3-build-", dir=cache_root) as tmp:
        staging = Path(tmp) / "artifact"
        staging.mkdir()
        _build_flash3(staging)
        if runtime_error := runtime_files_error(staging):
            raise RuntimeError(f"FA3 build produced invalid files: {runtime_error}.")
        _write_receipt(staging)
        try:
            staging.replace(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or (
                artifact_validation_error(destination)
            ):
                raise
    if validation_error := artifact_validation_error(destination):
        raise RuntimeError(
            f"Prepared FA3 artifact at {destination} failed validation: "
            f"{validation_error}.",
        )
    return destination


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register flags on ``parser``."""
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/opt/scratch/caches/nanochat/fa3"),
        help="Stable node-local cache root the artifact installs under.",
    )


def _build_flash3(destination: Path) -> None:
    build_root = destination.parent
    source = build_root / "source"
    wheels = build_root / "wheels"
    wheels.mkdir()
    _run(["git", "init", str(source)])
    _run(
        [
            "git",
            "-C",
            str(source),
            "remote",
            "add",
            "origin",
            "https://github.com/varunneal/flash-attention.git",
        ],
    )
    _run(
        [
            "git",
            "-C",
            str(source),
            "fetch",
            "--depth=1",
            "origin",
            source_revision(),
        ],
    )
    _run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"])
    _run(
        [
            "git",
            "-C",
            str(source),
            "submodule",
            "update",
            "--init",
            "csrc/cutlass",
        ],
    )
    if _run_output(["git", "-C", str(source), "rev-parse", "HEAD"]) != (
        source_revision()
    ):
        raise RuntimeError("FA3 source checkout does not match the pinned revision.")
    cutlass = source / "csrc" / "cutlass"
    if _run_output(["git", "-C", str(cutlass), "rev-parse", "HEAD"]) != (
        cutlass_revision()
    ):
        raise RuntimeError("FA3 CUTLASS checkout does not match the pinned revision.")
    _run(
        [
            sys.executable,
            "setup.py",
            "bdist_wheel",
            "--dist-dir",
            str(wheels),
        ],
        cwd=source / "hopper",
        environment=_build_environment(os.environ),
    )
    built_wheels = list(wheels.glob("*.whl"))
    if len(built_wheels) != 1:
        raise RuntimeError(f"Expected one FA3 wheel, found {len(built_wheels)}.")
    shutil.unpack_archive(str(built_wheels[0]), destination, format="zip")


def _build_environment(environment: Mapping[str, str]) -> dict[str, str]:
    cuda_home = Path("/usr/local/cuda-12.8")
    path = str(cuda_home / "bin")
    if inherited_path := environment.get("PATH"):
        path = f"{path}{os.pathsep}{inherited_path}"
    return {
        **environment,
        "PATH": path,
        "CUDA_HOME": str(cuda_home),
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


def _validate_build_runtime() -> None:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("FA3 must be built on x86_64 Linux.")
    if torch.__version__.split("+", maxsplit=1)[0] != "2.9.1":
        raise RuntimeError(f"FA3 requires Torch 2.9.1; found {torch.__version__}.")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"FA3 requires CUDA 12.8; found {torch.version.cuda}.")
    if not torch.compiled_with_cxx11_abi():
        raise RuntimeError("FA3 requires the Torch C++11 ABI runtime.")
    nvcc = _nvcc_path()
    version = _run_output([str(nvcc), "--version"])
    if "release 12.8" not in version:
        raise RuntimeError(f"FA3 requires nvcc 12.8; found:\n{version}")


def _nvcc_path() -> Path:
    """Return nvcc from PATH or the provisioned CUDA 12.8 toolkit."""
    if nvcc := shutil.which("nvcc"):
        return Path(nvcc)
    provisioned = Path("/usr/local/cuda-12.8/bin/nvcc")
    if provisioned.is_file():
        return provisioned
    raise RuntimeError(
        "FA3 source preparation requires nvcc 12.8 on PATH or at "
        "/usr/local/cuda-12.8/bin/nvcc.",
    )


def _write_receipt(path: Path) -> None:
    values = runtime_receipt(path)
    (path / "READY").write_text(
        "".join(f"{name}={value}\n" for name, value in values.items()),
        encoding="utf-8",
    )


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> None:
    subprocess.run(  # noqa: S603 -- Commands are fixed preparation steps without shell expansion or user input.
        command,
        check=True,
        cwd=cwd,
        env=environment,
    )


def _run_output(command: list[str]) -> str:
    return subprocess.run(  # noqa: S603 -- Commands are fixed probes without shell expansion or user input.
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
