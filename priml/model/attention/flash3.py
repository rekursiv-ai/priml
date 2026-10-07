"""FlashAttention 3: causal attention through a pinned, receipt-verified SM90 build.

The kernel is a local build of one pinned source revision, not an installed
package. ``prepare_flash3`` compiles it once per node into a content-addressed
directory and writes a ``READY`` receipt holding the pins and the SHA-256 of the
extension, its Python interface and its generated config. :func:`load_flash3`
checks that receipt against the files and imports the interface; it never builds
or downloads, so an unprepared node fails at ``make()`` rather than mid-run.

The build is the profile of the baseline that first needed it: SM90, bfloat16
only, and head widths of at most 128, with sliding windows and the backward.
``prepare_flash3`` compiles out FP16, FP8, the wider heads, varlen, split-KV,
paged and appended KV caches, softcapping, PackGQA and clusters. The config
holds no dtype or head width, so ``make()`` cannot refuse inputs outside the
profile; FA3 refuses them at the first forward.

The kernel takes the ``AttentionKernel`` layout,
``[..., S, num_heads, channels_head]``, and attends causally, optionally within
``window`` previous keys. Keys and values may carry fewer heads than the
queries, a divisor of theirs, which FA3 groups. An additive mask, attention
dropout and non-causal attention have no path here; each is refused rather
than dropped.

FA3's interface registers its forward and backward as torch custom ops with fake
implementations, so unlike FA4 this kernel compiles under
``torch.compile(fullgraph=True)`` without an op of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

import hashlib
import importlib
import sys

from configgle import Fig
from torch import Tensor, nn

import torch

from priml.model.attention.kernel import attention_kernel_cost


if TYPE_CHECKING:
    from collections.abc import Mapping

    from priml.cost import Cost


class Flash3UnavailableError(RuntimeError):
    """Raised when the pinned local FlashAttention 3 build cannot run here."""


@runtime_checkable
class Flash3Interface(Protocol):
    """The entry point of FA3's ``flash_attn_interface`` this module calls."""

    __file__: str

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
        """Attend over ``[B, S, H, D]`` rows with FA3's own autograd."""
        ...


class Flash3Attention(nn.Module):
    """Causal FlashAttention 3 over dense rows, optionally within a sliding window.

    Bfloat16 only, at head widths of at most 128: the pinned build compiles
    nothing else (see the module docstring).

    The window is a kernel argument rather than a mask: a mask forces torch's
    dispatcher off every flash backend, so windowed layers would silently run a
    different kernel than the one the recipe was measured on.
    """

    class Config(Fig["Flash3Attention"]):
        """The parity reference the pinned build was qualified against."""

        revision: str = "de87b9b5af06dd9984df595bef90b2eba44b181a"
        """Qualified parity reference the local build must match.

        A literal rather than a call to :func:`hf_reference_revision`: a config
        field is the experiment's declaration of what it ran against, and one
        that reads its value from the library it is pinning would follow that
        library forward and silently stop pinning anything."""

        @classmethod
        def cost(
            cls,
            *,
            seq_len: int,
            batch_size: int = 1,
            dtype: torch.dtype | None,
            num_heads: int,
            channels_head: int,
            channels_v_head: int = -1,
            window: int = -1,
            dropout_p: float = 0.0,
            rows: int = -1,
            **kwargs: object,
        ) -> Cost:
            """Cost the kernel from the shapes its owner hands it.

            See :func:`attention_kernel_cost` for every argument.

            Args:
              seq_len: Tokens per sequence.
              batch_size: Sequences in this invocation.
              dtype: Activation dtype; ``None`` is torch's default.
              num_heads: Query heads.
              channels_head: Width of each query/key head.
              channels_v_head: Value width; -1 uses the query/key width.
              window: Previous keys each query reaches, plus itself; negative is unbounded.
              dropout_p: Attention dropout rate.
              rows: Query rows sharing K/V; negative uses the modeled key count.
              **kwargs: The rest of the owner's bus, unread.

            Returns:
              cost: Whole-invocation cost of the kernel.

            """
            del kwargs
            return attention_kernel_cost(
                seq_len=seq_len,
                batch_size=batch_size,
                dtype=dtype,
                num_heads=num_heads,
                channels_head=channels_head,
                channels_v_head=channels_v_head,
                window=window,
                dropout_p=dropout_p,
                rows=rows,
            )

    def __init__(self, config: Config) -> None:
        if config.revision != hf_reference_revision():
            raise ValueError(
                "revision identifies the qualified parity reference and must "
                f"remain {hf_reference_revision()}; got {config.revision}.",
            )
        super().__init__()
        capability = torch.cuda.get_device_capability()
        if capability != (9, 0):
            raise Flash3UnavailableError(
                "The pinned FlashAttention 3 build requires SM90; this device is "
                f"SM{capability[0]}{capability[1]}. Inject a portable kernel such "
                "as SdpaCausal instead.",
            )
        # The function, not the interface module: a module object cannot be
        # deep-copied, so holding one would break copying any model holding this.
        self._attend = load_flash3().flash_attn_func

    @override
    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        window: int = -1,
        is_causal: bool = True,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0.0,
        scale: float | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Attend causally over each row.

        Args:
          q: Queries ``[..., S, H, D]``.
          k: Keys ``[..., S, H_kv, D]``; ``H_kv`` divides ``H``.
          v: Values, shaped like ``k``.
          window: Previous keys each query reaches, plus itself; -1 for all.
          is_causal: Must be True.
          attn_mask: Must be None.
          dropout_p: Must be 0.
          scale: Logit scale; None is ``D**-0.5``.
          **kwargs: The rest of the bus, unread.

        Returns:
          out: ``[..., S, H, D]``.

        Raises:
          ValueError: Non-causal attention, a mask, or dropout was asked for.

        """
        del kwargs
        if not is_causal or attn_mask is not None or dropout_p:
            raise ValueError("Flash3Attention is causal and takes no mask or dropout.")
        out = self._attend(
            q.reshape(-1, *q.shape[-3:]),
            k.reshape(-1, *k.shape[-3:]),
            v.reshape(-1, *v.shape[-3:]),
            softmax_scale=scale,
            causal=True,
            window_size=(window, 0),
        )
        return out.view(q.shape)


def source_revision() -> str:
    """Return the pinned FA3 source revision."""
    return "3da5f873029162763568db56546fee70a779fade"


def cutlass_revision() -> str:
    """Return the CUTLASS submodule revision the FA3 source pins."""
    return "dc4817921edda44a549197ff3a9dcf5df0636e7b"


def hf_reference_revision() -> str:
    """Return the previously qualified binary revision builds are held to."""
    return "de87b9b5af06dd9984df595bef90b2eba44b181a"


def cuda_version() -> str:
    """Return the CUDA version this torch was built against.

    Returns:
      version: ``torch.version.cuda``, e.g. ``"12.8"``.

    Raises:
      Flash3UnavailableError: This torch is a CPU-only build.

    """
    if torch.version.cuda is None:
        raise Flash3UnavailableError("FA3 requires a CUDA build of torch.")
    return torch.version.cuda


# The root and the identity keep the names of the baseline that first built
# this profile; renaming either orphans every node already prepared.
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
    cuda = cuda_version().replace(".", "")
    build = "cxx11-x86_64-nanochat-hdim128-bf16-local"
    return cache_root / f"{source_revision()}-torch2.9.1-cu{cuda}-{build}"


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
        "cuda": cuda_version(),
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
                    f"{name} mismatch: receipt {receipt[name]}, actual {expected_value}",
                )
            else:
                errors.append(
                    f"{name} mismatch: expected {expected_value}, receipt {receipt[name]}",
                )
    errors.extend(
        f"unexpected receipt field {name}"
        for name in sorted(receipt.keys() - expected.keys())
    )
    return "; ".join(errors)


def artifact_validation_error(path: Path) -> str:
    """Return why a prepared artifact at ``path`` is invalid, or an empty string.

    Args:
      path: Content-addressed artifact directory.

    Returns:
      error: Semicolon-delimited details, empty when the artifact validates.

    """
    if runtime_error := runtime_files_error(path):
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
        expected = runtime_receipt(path)
    except OSError as error:
        return f"could not hash FA3 runtime files: {error}"
    return receipt_validation_error(receipt, expected=expected)


def runtime_files_error(path: Path) -> str:
    """Return missing or ambiguous runtime-file details, or an empty string.

    Args:
      path: Artifact directory holding the FA3 runtime files.

    Returns:
      error: Semicolon-delimited details, empty when every file is present.

    """
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
            f"expected exactly one flash_attn_3/_C*.so; found {extension_count}",
        )
    return "; ".join(errors)


def runtime_receipt(path: Path) -> dict[str, str]:
    """Return the READY receipt derived from the runtime files at ``path``.

    Args:
      path: Artifact directory holding the FA3 runtime files.

    Returns:
      receipt: Field-to-value mapping, hashes included.

    """
    return expected_receipt(
        binary_sha256=_sha256(_extension_path(path)),
        interface_sha256=_sha256(path / "flash_attn_interface.py"),
        config_sha256=_sha256(path / "flash_attn_config.py"),
    )


def load_flash3(
    *,
    cache_root: Path = Path("/opt/scratch/caches/nanochat/fa3"),
) -> Flash3Interface:
    """Load the prepared FA3 interface without network access.

    Args:
      cache_root: Stable node-local cache root.

    Returns:
      interface: Pinned local FlashAttention 3 Python interface.

    Raises:
      Flash3UnavailableError: The prepared artifact is missing or invalid, or
        an FA3 module was already imported from elsewhere.
      TypeError: The interface lacks the entry point this module calls.

    """
    prepared = artifact_path(cache_root=cache_root)
    if validation_error := artifact_validation_error(prepared):
        raise Flash3UnavailableError(
            f"Prepared FlashAttention 3 is invalid at {prepared}: "
            f"{validation_error}. Run `uv --quiet run --frozen --isolated --project priml/baselines/nanochat/runtime python -m priml.model.attention.prepare_flash3` once on this node.",
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
    if not isinstance(interface, Flash3Interface):
        raise TypeError("FA3 must provide flash_attn_func.")
    return interface


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
            f"expected exactly one flash_attn_3/_C*.so; found {len(extensions)}",
        )
    return extensions[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
        f"{module_name} was already imported from {module_path}, outside the "
        f"prepared artifact {path}."
    )
