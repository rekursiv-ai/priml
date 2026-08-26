"""The pinned FlashAttention-3 runtime ``exp000`` reproduces its number on.

The reference recipe measured its score with FlashAttention-3, so a rung that
claims to reproduce that number must issue the same kernel: a fused attention
reduces in a different order than a masked ``scaled_dot_product_attention``,
and no amount of matching hyperparameters closes that gap.

The kernel is built from a pinned source revision, verified against a receipt,
and installed content-addressed below ``/opt/scratch/caches/nanochat/fa3``.
Run ``python -m priml.baselines.nanochat.flash3`` once on each fresh
Hopper node; training then loads that local artifact and never reaches the
network.

FA3 requires SM90. Every other rung of the ladder resolves its backend from
what the GPU offers, so only ``exp000`` is bound to this hardware.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import errno
import hashlib
import importlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile

from configgle import Fig
from torch import Tensor

import torch


logger = logging.getLogger(__name__)


class Flash3UnavailableError(RuntimeError):
    """Raised when the pinned local FlashAttention-3 artifact is unavailable."""


class Flash3Interface(Protocol):
    """FlashAttention interface used by the NanoChat model."""

    __file__: str

    def flash_attn_func(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        causal: bool,
        window_size: tuple[int, int],
    ) -> Tensor: ...


class Flash3Attention:
    """Windowed causal attention through the pinned FlashAttention-3 kernel.

    Takes ``[B, S, heads, channels_head]`` -- the layout FA3 wants, and the one
    the model holds before it transposes for SDPA -- and expresses the window
    as a kernel argument rather than a mask. That is the whole reason this
    class exists: a mask forces the dispatcher off every flash backend, so the
    windowed layers would silently run a different kernel than the reference.

    Constructed rather than called as a function so the artifact is resolved
    ONCE, at model construction, instead of on every layer of every step.
    """

    class Config(Fig["Flash3Attention"]):
        """The pinned artifact revision."""

        revision: str = "de87b9b5af06dd9984df595bef90b2eba44b181a"
        """Qualified parity reference the local build must match.

        A literal rather than a call to :func:`hf_reference_revision`: a config
        field is the experiment's declaration of what it ran against, and one
        that reads its value from the library it is pinning would follow that
        library forward and silently stop pinning anything."""

    def __init__(self, config: Config) -> None:
        if config.revision != hf_reference_revision():
            raise ValueError(
                "revision identifies the qualified parity reference and must "
                f"remain {hf_reference_revision()}; got {config.revision}.",
            )
        capability = torch.cuda.get_device_capability()
        if capability != (9, 0):
            raise Flash3UnavailableError(
                f"The pinned FlashAttention-3 build requires SM90; this device "
                f"is SM{capability[0]}{capability[1]}. Every rung below exp000 "
                "resolves its backend from the device and runs here.",
            )
        self._flash = load_flash3()

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        window: int = -1,
        **kwargs: Any,
    ) -> Tensor:
        """Attend over the last ``window`` positions, causally.

        ``window`` defaults to ``-1``: unbounded over the causal prefix.
        Remaining keyword arguments belong to the open model message bus; this
        kernel reads only the window it understands.

        """
        del kwargs
        return self._flash.flash_attn_func(
            q,
            k,
            v,
            causal=True,
            window_size=(window, 0),
        )


def source_revision() -> str:
    """Return the immutable FA3 source revision."""
    return "3da5f873029162763568db56546fee70a779fade"


def cutlass_revision() -> str:
    """Return the CUTLASS submodule revision pinned by the FA3 source."""
    return "dc4817921edda44a549197ff3a9dcf5df0636e7b"


def hf_reference_revision() -> str:
    """Return the previously qualified HF binary revision."""
    return "de87b9b5af06dd9984df595bef90b2eba44b181a"


def artifact_path(
    *,
    cache_root: Path = Path("/opt/scratch/caches/nanochat/fa3"),
) -> Path:
    """Return the content-addressed local FA3 installation path.

    Args:
        cache_root: Stable node-local cache root.

    Returns:
        path: Installation path for the pinned source and runtime combination.

    """
    identity = (
        f"{source_revision()}-torch2.9.1-cu128-cxx11-x86_64-nanochat-hdim128-bf16-local"
    )
    return cache_root / identity


def expected_receipt(
    *,
    binary_sha256: str,
    interface_sha256: str,
    config_sha256: str,
) -> dict[str, str]:
    """Return the READY receipt contents that validate a prepared artifact.

    Args:
        binary_sha256: SHA-256 hex digest of the installed extension binary.
        interface_sha256: SHA-256 hex digest of the Python interface.
        config_sha256: SHA-256 hex digest of the generated kernel configuration.

    Returns:
        receipt: Field-to-value mapping pinned to the qualified build lane.

    """
    return {
        "source_revision": source_revision(),
        "cutlass_revision": cutlass_revision(),
        "torch": "2.9.1",
        "cuda": "12.8",
        "cxx11_abi": "true",
        "build_profile": "nanochat-hdim128-bf16-local",
        "binary_sha256": binary_sha256,
        "interface_sha256": interface_sha256,
        "config_sha256": config_sha256,
    }


def receipt_validation_error(
    receipt: Mapping[str, str],
    *,
    expected: Mapping[str, str],
) -> str:
    """Return field-level READY receipt mismatch details.

    Args:
        receipt: Parsed field values from the installed READY receipt.
        expected: Field values derived from the qualified runtime and files.

    Returns:
        error: Semicolon-delimited mismatch details, or an empty string.

    """
    errors: list[str] = []
    for name, expected_value in expected.items():
        if name not in receipt:
            errors.append(f"missing receipt field {name}")
        elif receipt[name] != expected_value:
            if name.endswith("_sha256"):
                errors.append(
                    f"{name} mismatch: receipt {receipt[name]}, actual {expected_value}"
                )
            else:
                errors.append(
                    f"{name} mismatch: expected {expected_value}, receipt {receipt[name]}"
                )
    errors.extend(
        f"unexpected receipt field {name}"
        for name in sorted(receipt.keys() - expected.keys())
    )
    return "; ".join(errors)


def is_prepared(
    *,
    cache_root: Path = Path("/opt/scratch/caches/nanochat/fa3"),
) -> bool:
    """Report whether the pinned local artifact is complete and intact.

    Args:
        cache_root: Stable node-local cache root.

    Returns:
        prepared: Whether the receipt and installed files validate.

    """
    return _validate_artifact(artifact_path(cache_root=cache_root))


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
    validation_error = _artifact_validation_error(destination)
    if not validation_error:
        return destination
    if destination.exists():
        raise FileExistsError(
            f"FA3 artifact at {destination} failed validation: {validation_error}. "
            "Remove only this content-addressed directory, then prepare again."
        )

    _validate_build_runtime()
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fa3-build-", dir=cache_root) as tmp:
        staging = Path(tmp) / "artifact"
        staging.mkdir()
        _build_flash3(staging)
        if runtime_error := _runtime_files_error(staging):
            raise RuntimeError(f"FA3 build produced invalid files: {runtime_error}.")
        _write_receipt(staging)
        try:
            staging.replace(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not (
                _validate_artifact(destination)
            ):
                raise
    if validation_error := _artifact_validation_error(destination):
        raise RuntimeError(
            f"Prepared FA3 artifact at {destination} failed validation: "
            f"{validation_error}."
        )
    return destination


def load_flash3(
    *,
    cache_root: Path = Path("/opt/scratch/caches/nanochat/fa3"),
) -> Flash3Interface:
    """Load the prepared FA3 interface without network access.

    Args:
        cache_root: Stable node-local cache root.

    Returns:
        interface: Pinned local FlashAttention-3 Python interface.

    Raises:
        Flash3UnavailableError: The prepared artifact is missing or invalid.

    """
    prepared = artifact_path(cache_root=cache_root)
    if validation_error := _artifact_validation_error(prepared):
        raise Flash3UnavailableError(
            f"Prepared FlashAttention-3 is invalid at {prepared}: "
            f"{validation_error}. "
            "Run `python -m priml.baselines.nanochat.flash3` once on this node."
        )
    if module_error := _loaded_module_error("flash_attn_3._C", prepared):
        raise Flash3UnavailableError(module_error)
    prepared_str = str(prepared)
    if prepared_str not in sys.path:
        sys.path.insert(0, prepared_str)
    interface = importlib.import_module("flash_attn_interface")
    for module_name in ("flash_attn_interface", "flash_attn_3._C"):
        if module_error := _loaded_module_error(module_name, prepared):
            raise Flash3UnavailableError(module_error)
    return cast(Flash3Interface, interface)


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
        ]
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
        ]
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
        ]
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
        "/usr/local/cuda-12.8/bin/nvcc."
    )


def _validate_artifact(path: Path) -> bool:
    return not _artifact_validation_error(path)


def _artifact_validation_error(path: Path) -> str:
    """Return artifact validation errors, or an empty string."""
    if runtime_error := _runtime_files_error(path):
        return runtime_error
    receipt_path = path / "READY"
    if not receipt_path.exists():
        return "missing READY receipt"
    if not receipt_path.is_file():
        return f"READY receipt is not a regular file: {receipt_path}"
    try:
        receipt_text = receipt_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "READY receipt is not valid UTF-8"
    except OSError as error:
        return f"could not read READY receipt: {error}"
    receipt, receipt_error = _parse_receipt(receipt_text)
    if receipt_error:
        return receipt_error
    try:
        expected = _runtime_receipt(path)
    except OSError as error:
        return f"could not hash FA3 runtime files: {error}"
    return receipt_validation_error(receipt, expected=expected)


def _runtime_files_error(path: Path) -> str:
    """Return missing or ambiguous runtime-file details."""
    missing = [
        name
        for name in ("flash_attn_interface.py", "flash_attn_config.py")
        if not (path / name).is_file()
    ]
    errors: list[str] = (
        [f"missing required runtime files: {', '.join(missing)}"] if missing else []
    )
    extension_count = sum(
        extension.is_file() for extension in (path / "flash_attn_3").glob("_C*.so")
    )
    if extension_count != 1:
        errors.append(
            f"expected exactly one flash_attn_3/_C*.so; found {extension_count}"
        )
    return "; ".join(errors)


def _parse_receipt(text: str) -> tuple[dict[str, str], str]:
    """Parse a READY receipt without accepting ambiguous duplicate fields."""
    receipt: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if "=" not in line:
            return (
                {},
                f"malformed READY receipt line {line_number}; expected name=value",
            )
        name, value = line.split("=", maxsplit=1)
        if not name:
            return (
                {},
                f"malformed READY receipt line {line_number}; field name is empty",
            )
        if name in receipt:
            return {}, f"duplicate receipt field {name}"
        receipt[name] = value
    return receipt, ""


def _extension_path(path: Path) -> Path:
    extensions = [
        extension
        for extension in (path / "flash_attn_3").glob("_C*.so")
        if extension.is_file()
    ]
    if len(extensions) != 1:
        raise FileNotFoundError(
            f"expected exactly one flash_attn_3/_C*.so; found {len(extensions)}"
        )
    return extensions[0]


def _runtime_receipt(path: Path) -> dict[str, str]:
    """Return the receipt derived from every loaded runtime file."""
    return expected_receipt(
        binary_sha256=_sha256(_extension_path(path)),
        interface_sha256=_sha256(path / "flash_attn_interface.py"),
        config_sha256=_sha256(path / "flash_attn_config.py"),
    )


def _write_receipt(path: Path) -> None:
    values = _runtime_receipt(path)
    (path / "READY").write_text(
        "".join(f"{name}={value}\n" for name, value in values.items()),
        encoding="utf-8",
    )


def _loaded_module_error(module_name: str, path: Path) -> str:
    """Return an error when a loaded FA3 module comes from outside ``path``."""
    module = sys.modules.get(module_name)
    if module is None:
        return ""
    module_path = module.__file__
    if module_path is None:
        return f"Loaded {module_name} has no file path."
    if Path(module_path).resolve().is_relative_to(path.resolve()):
        return ""
    return (
        f"A non-baseline {module_name} was already imported from {module_path}; "
        f"expected it below {path}."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> None:
    subprocess.run(  # noqa: S603 -- fixed preparer commands; no shell or user input
        command,
        check=True,
        cwd=cwd,
        env=environment,
    )


def _run_output(command: list[str]) -> str:
    return subprocess.run(  # noqa: S603 -- fixed probes; no shell or user input
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


if __name__ == "__main__":
    prepare_flash3()
