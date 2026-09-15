"""Causal attention, FlashAttention backends, and fused normalization/rotary kernels.

FlashAttention-3 uses a pinned, receipt-verified SM90 build. Prepare it once with
``uv --quiet run --frozen python -m priml.baselines.nanochat.attention``;
training loads the local artifact without network access. FlashAttention-4
resolves its optional installed backend when the model is constructed.

Triton parses source annotations as device code. Keep tuple annotations quoted
and omit future annotations, which makes the formatter remove those quotes.
"""

from collections.abc import Callable, Mapping
from dataclasses import field
from functools import lru_cache, partial
from importlib import import_module
from pathlib import Path
from types import FunctionType
from typing import TYPE_CHECKING, Protocol, Self, cast, override, runtime_checkable

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

from configgle import Fig, Makeable, Makes
from torch import Tensor, nn

import torch

from priml.baselines.nanochat.ngram import (
    HashedNgramTables,
    NgramSource,
    ngram_mix,
)
from priml.model.attention.rope import rotate_conjugate
from priml.model.attention.value_gated_attention import ValueGatedAttention
from priml.model.custom_types import TensorModule, propagate_attr
from priml.model.linear import Linear
from priml.model.norm import RMSNorm
from priml.model.special import Identity


if TYPE_CHECKING:
    from triton import language
    from triton.language.extra.cuda import libdevice

    import triton
else:
    from wrapt import lazy_import

    triton = lazy_import("triton")
    language = lazy_import("triton.language")
    libdevice = lazy_import("triton.language.extra.cuda.libdevice")


class _FusedContext(Protocol):
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


class CausalAttention(ValueGatedAttention):
    """Normalize Q/K before RoPE, with optional output gates and value memories."""

    class Config(Makes["CausalAttention"], ValueGatedAttention.Config):
        head_gate: Linear.Config | None = None
        """Optional per-head output gate; absent in the ungated variant."""

        fused_qk_rope: bool = False
        """Fuse parameter-free RMSNorm and RoPE, reducing in float32.

        ``norm_qk`` must be RMSNorm with epsilon None or float32 epsilon.
        In this path, None selects the float32 accumulation dtype's epsilon.
        """

        bigram: bool = False
        """Build a gate reading the second gate-width slice of the input."""

        trigram: bool = False
        """Build a gate reading the third gate-width slice of the input."""

        norm_out: Makeable[TensorModule] = field(
            default_factory=Identity.Config,
        )
        """Per-head output normalization, preceding the head gate."""

        @override
        def finalize(self) -> Self:
            width = self.channels_in if self.channels_in > 0 else self.channels_out
            if self.head_gate is not None:
                self.head_gate.channels_in = (
                    self.gate_channels if self.gate_channels >= 0 else width
                )
                self.head_gate.channels_out = (
                    self.num_heads
                    if self.num_heads > 0
                    else width // self.channels_head
                )
            propagate_attr(self.norm_out, "channels_in", self.channels_head)
            return super().finalize()

    def __init__(self, config: Config) -> None:
        norm = config.norm_qk
        if config.fused_qk_rope and (
            not isinstance(norm, RMSNorm.Config)
            or type(norm) is not RMSNorm.Config
            or norm.elementwise_affine
            or norm.eps not in (None, torch.finfo(torch.float32).eps)
        ):
            raise ValueError(
                "fused_qk_rope requires norm_qk to be parameter-free RMSNorm "
                "with epsilon None or float32 epsilon.",
            )
        required_slices = 3 if config.trigram else 2 if config.bigram else 1
        if required_slices * config.gate_channels > config.channels_in:
            raise ValueError(
                f"{required_slices} gate slices of width {config.gate_channels} "
                f"do not fit channels_in={config.channels_in}.",
            )
        super().__init__(config)
        gate = Linear.Config(
            channels_in=config.gate_channels,
            channels_out=config.num_heads,
            bias=False,
            init_weight=nn.init.zeros_,
        )
        self.bigram_gate = gate.make() if config.bigram else None
        self.trigram_gate = gate.make() if config.trigram else None
        self.head_gate = (
            config.head_gate.make() if config.head_gate is not None else None
        )
        self.norm_out = config.norm_out.make()
        self.fused_qk_rope = config.fused_qk_rope

    @override
    def reset_parameters(self) -> None:
        super().reset_parameters()
        for gate in (self.bigram_gate, self.trigram_gate, self.head_gate):
            if gate is not None:
                gate.reset_parameters()
        self.norm_out.reset_parameters()

    @override
    def forward(
        self,
        x: Tensor,
        *,
        cos_sin: tuple[Tensor, Tensor],
        value_embedding: Tensor | None = None,
        window: int | None = None,
        **kwargs: object,
    ) -> Tensor:
        """Apply causal attention with normalized rotary queries and keys.

        Args:
          x: Residual stream with channels last.
          cos_sin: Rotary factors by position and half-channel.
          value_embedding: Optional token-specific attention values.
          window: History distance; None uses the configured attention window.
          **kwargs: Memory values and unconsumed messages for the attention kernel.

        Returns:
          output: Projected attention output with the shape of ``x``.

        """
        bigram_value = kwargs.pop("bigram_value", None)
        trigram_value = kwargs.pop("trigram_value", None)
        assert bigram_value is None or isinstance(bigram_value, Tensor)
        assert trigram_value is None or isinstance(trigram_value, Tensor)
        cfg = self.config
        shape = (*x.shape[:-1], cfg.num_heads, cfg.channels_head)
        q, k = self.proj_q(x).view(shape), self.proj_k(x).view(shape)
        cos, sin = cos_sin
        if self.fused_qk_rope:
            q, k = fused_qk_norm_rope(q, k, cos, sin)
        else:
            q, k = (
                rotate_conjugate(self.norm_q(q), cos=cos, sin=sin),
                rotate_conjugate(self.norm_k(k), cos=cos, sin=sin),
            )
        v = self.proj_v(x).view(shape)
        for index, (value, gate) in enumerate(
            (
                (value_embedding, self.value_gate),
                (bigram_value, self.bigram_gate),
                (trigram_value, self.trigram_gate),
            ),
        ):
            if value is not None:
                assert gate is not None
                start = index * cfg.gate_channels
                weight = 2 * torch.sigmoid(
                    gate(x[..., start : start + cfg.gate_channels]),
                )
                v = v + weight.unsqueeze(-1) * value.view(shape)
        fused_tables = kwargs.pop("fused_tables", None)
        if fused_tables is not None:
            assert isinstance(fused_tables, list)
            logits: list[Tensor] = []
            weights: list[Tensor] = []
            indices: list[Tensor] = []
            sinks: list[Tensor] = []
            for source in cast(list[object], fused_tables):
                assert isinstance(source, NgramSource)
                gate_index, table, hashed = source
                assert isinstance(table, HashedNgramTables)
                gate = self.bigram_gate if gate_index == 1 else self.trigram_gate
                assert gate is not None
                start = gate_index * cfg.gate_channels
                logits.append(gate(x[..., start : start + cfg.gate_channels]))
                weights.extend(part.weight for part in table.tables)
                indices.extend(hashed)
                sinks.extend(table.gradient_sinks)
            bitmaps: list[Tensor] = []
            for source in cast(list[object], fused_tables):
                assert isinstance(source, NgramSource)
                (_gate_index, table, _hashed) = source
                assert isinstance(table, HashedNgramTables)
                bitmaps.extend(table.gradient_bitmaps)
            v = ngram_mix(v.contiguous(), logits, weights, indices, sinks, bitmaps)
        out = self.norm_out(
            self.attention(
                q, k, v, window=cfg.window if window is None else window, **kwargs
            ),
        )
        if self.head_gate is not None:
            head_weights = 2 * torch.sigmoid(
                self.head_gate(x[..., : cfg.gate_channels])
            )
            out = out * head_weights.unsqueeze(-1)
        return self.proj_out(out.contiguous().flatten(-2))


@torch.library.custom_op("priml_nanochat::qk_norm_rope", mutates_args=())
def fused_qk_norm_rope(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> "tuple[Tensor, Tensor]":
    """Normalize Q/K in FP32 and rotate, rounding when storing the result.

    Args:
      q: Queries shaped ``[batch, tokens, heads, channels]``.
      k: Keys with the same shape; half the channel width must be a power of two.
      cos: Rotary cosine factors by position and half-channel.
      sin: Rotary sine factors matching ``cos``.

    Returns:
      queries: Contiguous normalized and rotated queries in the input dtype.
      keys: Contiguous normalized and rotated keys in the input dtype.

    """
    assert q.shape == k.shape
    assert q.ndim == 4
    half = q.shape[-1] // 2
    assert q.shape[-1] % 2 == 0
    assert half > 0
    assert half & (half - 1) == 0
    if q.is_cuda:
        return _qk_forward_cuda(q, k, cos, sin)
    return (
        _qk_reference(q, cos, sin).contiguous(),
        _qk_reference(k, cos, sin).contiguous(),
    )


def _qk_reference(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    co = cos.reshape(-1, half)[: x.shape[1]].float()[None, :, None, :]
    si = sin.reshape(-1, half)[: x.shape[1]].float()[None, :, None, :]
    a, b = x[..., :half].float(), x[..., half:].float()
    inv = torch.rsqrt(
        (a * a + b * b).sum(-1, keepdim=True) / x.shape[-1] + 1.1920928955078125e-7
    )
    a, b = a * inv, b * inv
    return torch.cat((a * co + b * si, b * co - a * si), dim=-1).to(x.dtype)


@torch.library.custom_op("priml_nanochat::qk_backward", mutates_args=())
def _qk_backward(
    dq: Tensor, dk: Tensor, q: Tensor, k: Tensor, rotation: list[Tensor]
) -> "tuple[Tensor, Tensor]":
    cos, sin = rotation
    if q.is_cuda:
        return _qk_backward_cuda(dq, dk, q, k, cos=cos, sin=sin)
    # CPU cat can preserve channels-last cotangents. Normalize inside the opaque
    # operator so AOT cannot elide the copy using its declared contiguous strides.
    return (
        _qk_backward_reference(dq, q, cos, sin).contiguous(),
        _qk_backward_reference(dk, k, cos, sin).contiguous(),
    )


def _qk_backward_reference(dy: Tensor, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    co = cos.reshape(-1, half)[: x.shape[1]].float()[None, :, None, :]
    si = sin.reshape(-1, half)[: x.shape[1]].float()[None, :, None, :]
    a, b = x[..., :half].float(), x[..., half:].float()
    da, db = dy[..., :half].float(), dy[..., half:].float()
    inv = torch.rsqrt(
        (a * a + b * b).sum(-1, keepdim=True) / x.shape[-1] + 1.1920928955078125e-7
    )
    a, b = a * inv, b * inv
    ga, gb = da * co - db * si, da * si + db * co
    dot = (a * ga + b * gb).sum(-1, keepdim=True) / x.shape[-1]
    return torch.cat(((ga - a * dot) * inv, (gb - b * dot) * inv), dim=-1).to(x.dtype)


def _qk_setup(
    ctx: _FusedContext, inputs: tuple[Tensor, Tensor, Tensor, Tensor], output: object
) -> None:
    del output
    ctx.save_for_backward(*inputs)


def _register_qk_autograd[FunctionT: Callable[..., object]](
    function: FunctionT,
) -> FunctionT:
    fused_qk_norm_rope.register_autograd(function, setup_context=_qk_setup)
    return function


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
    ) -> Tensor:
        """Flash attn func."""
        ...


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
                f"is SM{capability[0]}{capability[1]}. Use exp001 for the "
                "portable PyTorch attention backend.",
            )
        self._flash = load_flash3()

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        window: int = -1,
        **kwargs: object,
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
    """Return the immutable FA3 source revision.

    Returns:
      result: The str.

    """
    return "3da5f873029162763568db56546fee70a779fade"


def cutlass_revision() -> str:
    """Return the CUTLASS submodule revision pinned by the FA3 source.

    Returns:
      result: The str.

    """
    return "dc4817921edda44a549197ff3a9dcf5df0636e7b"


def hf_reference_revision() -> str:
    """Return the previously qualified HF binary revision.

    Returns:
      result: The str.

    """
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
            "Remove only this content-addressed directory, then prepare again.",
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
            f"{validation_error}.",
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
            "Run `uv --quiet run --frozen python -m priml.baselines.nanochat.attention` once on this node.",
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


@runtime_checkable
class _Flash4Interface(Protocol):
    def flash_attn_func(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        causal: bool,
        window_size: tuple[int | None, int | None],
        return_lse: bool,
    ) -> tuple[Tensor, Tensor | None]: ...

    def _flash_attn_bwd(  # noqa: PLR0917 -- The upstream positional-only boundary has six tensors.
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        out: Tensor,
        grad_out: Tensor,
        lse: Tensor,
        /,
        *,
        causal: bool,
        window_size_left: int | None,
        window_size_right: int,
    ) -> tuple[Tensor, Tensor, Tensor]: ...


class _Flash4Context(Protocol):
    window: int
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


class Flash4Attention:
    """Load FA4 at make time, before compiling the enclosing model."""

    class Config(Fig["Flash4Attention"]):
        """Select the native CuTe FA4 dispatcher."""

    def __init__(self, config: Config) -> None:
        del config
        self._flash4_forward = _make_flash4_ops()

    def __call__(
        self, q: Tensor, k: Tensor, v: Tensor, *, window: int = -1, **kwargs: object
    ) -> Tensor:
        """Apply causal FA4 attention without changing the projection layout.

        Args:
            q: Queries shaped ``[B, S, H, D]``.
            k: Keys with the same shape, dtype, and device.
            v: Values with the same shape, dtype, and device.
            window: Inclusive history distance; negative means unbounded.
            **kwargs: Unconsumed model messages.

        Returns:
            out: Attention output shaped ``[B, S, H, D]``.

        """
        del kwargs
        if window >= q.shape[1]:
            window = -1
        return self._flash4_forward(q, k, v, window)[0]


@lru_cache(maxsize=1)
def _make_flash4_ops() -> Callable[
    [Tensor, Tensor, Tensor, int], tuple[Tensor, Tensor]
]:
    module = import_module("flash_attn.cute.interface")
    if not isinstance(module, _Flash4Interface):
        raise TypeError("FA4 must provide flash_attn_func and _flash_attn_bwd")
    forward = torch.library.custom_op(
        "priml_nanochat::flash4_forward",
        partial(_flash4_forward, module),
        mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v, int window) -> (Tensor, Tensor)",
    )
    backward = torch.library.custom_op(
        "priml_nanochat::flash4_backward",
        partial(_flash4_backward_kernel, module),
        mutates_args=(),
        schema="(Tensor[] saved, Tensor grad_out, int window) -> (Tensor, Tensor, Tensor)",
    )
    forward.register_fake(_flash4_forward_fake)
    backward.register_fake(_flash4_backward_fake)
    forward.register_autograd(
        partial(_flash4_backward, kernel=backward),
        setup_context=_flash4_setup_context,
    )
    return forward


def _flash4_forward(
    module: _Flash4Interface,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window: int,
) -> tuple[Tensor, Tensor]:
    out, lse = module.flash_attn_func(
        q,
        k,
        v,
        causal=True,
        window_size=(None, None) if window < 0 else (window, 0),
        return_lse=True,
    )
    assert lse is not None
    return out, lse


def _flash4_forward_fake(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window: int,
) -> tuple[Tensor, Tensor]:
    del k, v, window
    return torch.empty_like(q, memory_format=torch.contiguous_format), q.new_empty(
        (q.shape[0], q.shape[2], q.shape[1]),
        dtype=torch.float32,
    )


def _flash4_setup_context(
    ctx: _Flash4Context,
    inputs: tuple[Tensor, Tensor, Tensor, int],
    output: tuple[Tensor, Tensor],
) -> None:
    q, k, v, ctx.window = inputs
    ctx.save_for_backward(q, k, v, *output)


def _flash4_backward(
    ctx: _Flash4Context,
    grad_out: Tensor,
    grad_lse: Tensor | None,
    *,
    kernel: Callable[[list[Tensor], Tensor, int], tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, None]:
    del grad_lse
    return (*kernel(list(ctx.saved_tensors), grad_out, ctx.window), None)


def _flash4_backward_kernel(
    module: _Flash4Interface,
    saved: list[Tensor],
    grad_out: Tensor,
    window: int,
) -> tuple[Tensor, Tensor, Tensor]:
    q, k, v, out, lse = saved
    return module._flash_attn_bwd(  # noqa: SLF001 -- FA4 exposes backward only through this entry point.
        q,
        k,
        v,
        out,
        grad_out,
        lse,
        causal=True,
        window_size_left=None if window < 0 else window,
        window_size_right=0,
    )


def _flash4_backward_fake(
    saved: list[Tensor],
    grad_out: Tensor,
    window: int,
) -> tuple[Tensor, Tensor, Tensor]:
    del grad_out, window
    q, k, v, _, _ = saved
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


@fused_qk_norm_rope.register_fake
def _qk_fake(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> "tuple[Tensor, Tensor]":
    del cos, sin
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
    )


@_qk_backward.register_fake
def _qk_backward_fake(
    dq: Tensor, dk: Tensor, q: Tensor, k: Tensor, rotation: list[Tensor]
) -> "tuple[Tensor, Tensor]":
    del dq, dk, rotation
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
    )


@_register_qk_autograd
def _qk_autograd(
    ctx: _FusedContext, dq: Tensor, dk: Tensor
) -> "tuple[Tensor, Tensor, None, None]":
    q, k, cos, sin = ctx.saved_tensors
    a, b = _qk_backward(dq, dk, q, k, [cos, sin])
    return a, b, None, None


def _qk_forward_cuda(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> "tuple[Tensor, Tensor]":
    """Launch fused query/key RMS normalization and rotary embeddings."""
    (q, k) = (q.contiguous(), k.contiguous())
    (b, t, h, d) = q.shape
    rows = b * t * h

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(rows, meta["block"]),)

    (qo, ko) = (torch.empty_like(q), torch.empty_like(k))
    _compiled_qk_forward()[grid](
        buffers=(q, k, cos.contiguous(), sin.contiguous(), qo, ko),
        n_rows=rows,
        geometry=(h, t),
        constants=(1.1920928955078125e-07, d // 2, rows % 32 == 0),
    )
    return (qo, ko)


def _qk_backward_cuda(
    dq: Tensor, dk: Tensor, q: Tensor, k: Tensor, *, cos: Tensor, sin: Tensor
) -> "tuple[Tensor, Tensor]":
    """Launch inverse rotation and recomputed RMS normalization gradients."""
    (q, k) = (q.contiguous(), k.contiguous())
    (b, t, h, d) = q.shape
    rows = b * t * h

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(rows, meta["block"]),)

    (qo, ko) = (torch.empty_like(q), torch.empty_like(k))
    _compiled_qk_backward()[grid](
        buffers=(
            dq.contiguous(),
            dk.contiguous(),
            q,
            k,
            cos.contiguous(),
            sin.contiguous(),
            qo,
            ko,
        ),
        n_rows=rows,
        geometry=(h, t),
        constants=(1.1920928955078125e-07, d // 2, rows % 32 == 0),
    )
    return (qo, ko)


def _jit_kernel(function: Callable[..., None]) -> "triton.JITFunction[..., None]":
    """Bind concrete language modules before Triton hashes and compiles the function."""
    assert isinstance(function, FunctionType)
    bound = FunctionType(
        function.__code__,
        function.__globals__
        | {
            "language": import_module("triton.language"),
            "libdevice": import_module("triton.language.extra.cuda.libdevice"),
        },
        function.__name__,
        function.__defaults__,
    )
    bound.__annotations__ = function.__annotations__
    return triton.jit(bound)


@lru_cache(maxsize=1)
def _compiled_qk_forward() -> "triton.Autotuner":
    return triton.autotune(
        configs=[
            triton.Config({"block": b}, num_warps=w, num_stages=1)
            for b in (4, 8, 16, 32)
            for w in (2, 4, 8)
        ],
        key=["n_rows"],
    )(_jit_kernel(_qk_norm_rope_fwd_kernel))


def _qk_norm_rope_fwd_kernel(
    buffers: "tuple[language.tensor, ...]",
    n_rows: int,
    geometry: "tuple[int, int]",
    constants: "tuple[float, language.constexpr, bool]",
    block: "language.constexpr",
) -> None:
    """Normalize and rotate query/key rows, loading each rotary half once."""
    q_ptr, k_ptr, cos_ptr, sin_ptr, qo_ptr, ko_ptr = buffers
    n_head, seq_len = geometry
    eps = constants[0]
    half: language.constexpr = constants[1]
    even: language.constexpr = constants[2]
    pid = language.program_id(0)
    rows = pid * block + language.arange(0, block)
    j = language.arange(0, half)
    base = rows[:, None] * (2 * half) + j[None, :]
    if even:
        q1 = language.load(q_ptr + base).to(language.float32)
        q2 = language.load(q_ptr + base + half).to(language.float32)
        k1 = language.load(k_ptr + base).to(language.float32)
        k2 = language.load(k_ptr + base + half).to(language.float32)
    else:
        m = (rows < n_rows)[:, None]
        q1 = language.load(q_ptr + base, mask=m, other=0.0).to(language.float32)
        q2 = language.load(q_ptr + base + half, mask=m, other=0.0).to(language.float32)
        k1 = language.load(k_ptr + base, mask=m, other=0.0).to(language.float32)
        k2 = language.load(k_ptr + base + half, mask=m, other=0.0).to(language.float32)
    inv_d = 1.0 / (2 * half)
    rq = libdevice.rsqrt(language.sum(q1 * q1 + q2 * q2, 1) * inv_d + eps)[:, None]
    rk = libdevice.rsqrt(language.sum(k1 * k1 + k2 * k2, 1) * inv_d + eps)[:, None]
    pos = rows // n_head % seq_len
    cs_off = pos[:, None] * half + j[None, :]
    if even:
        co = language.load(cos_ptr + cs_off).to(language.float32)
        si = language.load(sin_ptr + cs_off).to(language.float32)
    else:
        m = (rows < n_rows)[:, None]
        co = language.load(cos_ptr + cs_off, mask=m, other=0.0).to(language.float32)
        si = language.load(sin_ptr + cs_off, mask=m, other=0.0).to(language.float32)
    (q1n, q2n) = (q1 * rq, q2 * rq)
    (k1n, k2n) = (k1 * rk, k2 * rk)
    qo1 = q1n * co + q2n * si
    qo2 = q2n * co - q1n * si
    ko1 = k1n * co + k2n * si
    ko2 = k2n * co - k1n * si
    if even:
        language.store(qo_ptr + base, qo1)
        language.store(qo_ptr + base + half, qo2)
        language.store(ko_ptr + base, ko1)
        language.store(ko_ptr + base + half, ko2)
    else:
        m = (rows < n_rows)[:, None]
        language.store(qo_ptr + base, qo1, mask=m)
        language.store(qo_ptr + base + half, qo2, mask=m)
        language.store(ko_ptr + base, ko1, mask=m)
        language.store(ko_ptr + base + half, ko2, mask=m)


@lru_cache(maxsize=1)
def _compiled_qk_backward() -> "triton.Autotuner":
    return triton.autotune(
        configs=[
            triton.Config({"block": b}, num_warps=w, num_stages=1)
            for b in (4, 8, 16, 32)
            for w in (2, 4, 8)
        ],
        key=["n_rows"],
    )(_jit_kernel(_qk_norm_rope_bwd_kernel))


def _qk_norm_rope_bwd_kernel(
    buffers: "tuple[language.tensor, ...]",
    n_rows: int,
    geometry: "tuple[int, int]",
    constants: "tuple[float, language.constexpr, bool]",
    block: "language.constexpr",
) -> None:
    """Undo rotation and differentiate RMS normalization using saved projections."""
    dq_ptr, dk_ptr, q_ptr, k_ptr, cos_ptr, sin_ptr, dqo_ptr, dko_ptr = buffers
    n_head, seq_len = geometry
    eps = constants[0]
    half: language.constexpr = constants[1]
    even: language.constexpr = constants[2]
    pid = language.program_id(0)
    rows = pid * block + language.arange(0, block)
    j = language.arange(0, half)
    base = rows[:, None] * (2 * half) + j[None, :]
    if even:
        q1 = language.load(q_ptr + base).to(language.float32)
        q2 = language.load(q_ptr + base + half).to(language.float32)
        k1 = language.load(k_ptr + base).to(language.float32)
        k2 = language.load(k_ptr + base + half).to(language.float32)
        dq1 = language.load(dq_ptr + base).to(language.float32)
        dq2 = language.load(dq_ptr + base + half).to(language.float32)
        dk1 = language.load(dk_ptr + base).to(language.float32)
        dk2 = language.load(dk_ptr + base + half).to(language.float32)
    else:
        m = (rows < n_rows)[:, None]
        q1 = language.load(q_ptr + base, mask=m, other=0.0).to(language.float32)
        q2 = language.load(q_ptr + base + half, mask=m, other=0.0).to(language.float32)
        k1 = language.load(k_ptr + base, mask=m, other=0.0).to(language.float32)
        k2 = language.load(k_ptr + base + half, mask=m, other=0.0).to(language.float32)
        dq1 = language.load(dq_ptr + base, mask=m, other=0.0).to(language.float32)
        dq2 = language.load(dq_ptr + base + half, mask=m, other=0.0).to(
            language.float32
        )
        dk1 = language.load(dk_ptr + base, mask=m, other=0.0).to(language.float32)
        dk2 = language.load(dk_ptr + base + half, mask=m, other=0.0).to(
            language.float32
        )
    pos = rows // n_head % seq_len
    cs_off = pos[:, None] * half + j[None, :]
    if even:
        co = language.load(cos_ptr + cs_off).to(language.float32)
        si = language.load(sin_ptr + cs_off).to(language.float32)
    else:
        m = (rows < n_rows)[:, None]
        co = language.load(cos_ptr + cs_off, mask=m, other=0.0).to(language.float32)
        si = language.load(sin_ptr + cs_off, mask=m, other=0.0).to(language.float32)
    inv_d = 1.0 / (2 * half)
    rq = libdevice.rsqrt(language.sum(q1 * q1 + q2 * q2, 1) * inv_d + eps)[:, None]
    rk = libdevice.rsqrt(language.sum(k1 * k1 + k2 * k2, 1) * inv_d + eps)[:, None]
    (qh1, qh2) = (q1 * rq, q2 * rq)
    (kh1, kh2) = (k1 * rk, k2 * rk)
    gq1 = dq1 * co - dq2 * si
    gq2 = dq1 * si + dq2 * co
    gk1 = dk1 * co - dk2 * si
    gk2 = dk1 * si + dk2 * co
    sq = language.sum(qh1 * gq1 + qh2 * gq2, 1)[:, None]
    sk = language.sum(kh1 * gk1 + kh2 * gk2, 1)[:, None]
    dqr1 = (gq1 - qh1 * inv_d * sq) * rq
    dqr2 = (gq2 - qh2 * inv_d * sq) * rq
    dkr1 = (gk1 - kh1 * inv_d * sk) * rk
    dkr2 = (gk2 - kh2 * inv_d * sk) * rk
    if even:
        language.store(dqo_ptr + base, dqr1)
        language.store(dqo_ptr + base + half, dqr2)
        language.store(dko_ptr + base, dkr1)
        language.store(dko_ptr + base + half, dkr2)
    else:
        m = (rows < n_rows)[:, None]
        language.store(dqo_ptr + base, dqr1, mask=m)
        language.store(dqo_ptr + base + half, dqr2, mask=m)
        language.store(dko_ptr + base, dkr1, mask=m)
        language.store(dko_ptr + base + half, dkr2, mask=m)


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
            f"expected exactly one flash_attn_3/_C*.so; found {extension_count}",
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
            f"expected exactly one flash_attn_3/_C*.so; found {len(extensions)}",
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
    prepare_flash3()
