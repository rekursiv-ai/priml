"""Bit-for-bit golden-file unit-test harness.

Pattern (see write-code skill rationale):

1. **Build** a module at minimum width: 1 layer, hidden=8, smallest seq_len.
2. **Randomize** every parameter with seeded ``torch.randn`` so structurally-zero
   inits (q-head bias, etc.) don't hide a regression.
3. **Snapshot** pre-run state, input, output, and changed post-run state to
   ``<test_file_dir>/testdata/``.
4. **Assert** on subsequent runs that loading the golden state and applying it
   to the same input reproduces every stored tensor bit.

Regenerate (after an intentional numeric change)::

    BFB_REGENERATE=1 uv --quiet run --frozen pytest <test_file>

Cross-architecture portability (the whole point):
  A float32 CPU kernel's last mantissa bit depends on the host's vector width
  (AVX2 vs AVX-512) and its parallel-reduction order -- a transcendental uses a
  different polynomial per width, a reduction sums in a different order per
  thread count, a matmul picks a different kernel per microarchitecture. So a
  golden minted on one host fails ``torch.equal`` on another even when the code
  is correct.

  ``host_agnostic_numerics`` removes this by computing every float32 *arithmetic*
  op in float64 and downcasting the result back to float32. The DOWNCAST is the
  mechanism, not a formality: a float64 kernel is itself an approximation and
  hosts disagree there too (measured, torch 2.11: ``sigmoid`` lands 2 float64
  ULP from the exact answer, ``tanh`` and ``rsqrt`` 1). Two float64 ULP is
  ~2**-51, about 2**-28 of ONE float32 ULP, so rounding to float32 absorbs the
  disagreement and every host lands on the same float32 bit -- measured 0 of
  4096 wrong for every op probed. It follows that a golden's floating comparand
  must be float32; ``_assert_portable_output_dtype`` also rejects complex
  outputs. A runner returning the float64 scratch keeps that host's own libm error (one
  did, off by 1 ULP between an Intel laptop and an AMD server). Only pure
  data-movement ops
  (views, reshapes, gathers) and correctly-rounded elementwise ops -- whose
  float32 result is already host-independent -- stay in float32; they are the
  ``_EXACT_F32_OPS`` allowlist. Every other float32 op is upcast by default, so
  a newly-introduced transcendental cannot silently leak: forgetting to list it
  upcasts it anyway. The allowlist is closed (IEEE-754 fixes which ops are
  correctly-rounded; views do no arithmetic) and guarded by a unit test that
  proves each entry is genuinely host-independent.

  Because matmul is upcast like everything else, the golden needs no MKL BLAS
  pin and no host-class gating of its own -- x86 (any width), ARM, Apple
  silicon, and OpenBLAS builds all reproduce, PROVIDED the comparand is float32
  (measured across Intel and AMD, and across 1/2/4/8/64 math threads: identical
  bits). Float64 GEMM is not itself invariant; it is the rounding that makes
  the result so -- and the rounding absorbs a float64 difference only until the
  exact value sits near a float32 boundary, which is why the priml conftest
  still pins ``MKL_CBWR``. Removing that pin was measured inert on AMD, where
  MKL takes a generic path, and broke goldens on Intel.

Determinism is required: the harness enables deterministic Torch algorithms
and seeds the CPU default generator before any tensor allocation.

Usage::

    from priml.testing.bfb import assert_bfb_against_golden

    def test_mymodule_bfb() -> None:
        cfg = MyModule.Config(channels=8, num_layers=1)
        assert_bfb_against_golden(
            golden_dir=Path(__file__).parent.resolve() / "testdata",
            golden_name="mymodule_min",
            build_module=lambda: cfg.make(),
            build_input=lambda: torch.randn(2, 4, 8),
            seed=0,
        )

The test fails if shapes, dtypes, stored bits, state keys, or post-run state
values differ from the golden.

Cross-implementation parity (loop vs HuggingFace, rewrite vs reference):
  A test that asserts ``torch.equal`` on the float32 outputs of TWO DIFFERENT
  computation paths -- not a golden round-trip, but e.g. our module vs HF's --
  faces the SAME host-dependence this harness exists to remove, and one extra
  trap. Both paths must run their arithmetic in the SAME order, or a float32
  matmul/softmax/reduction lands on a different last bit on a different host
  (AVX2 vs AVX-512, thread count, AMD vs Intel), and the golden minted on one
  host fails ``torch.equal`` on another. Such a test is typically
  ``@pytest.mark.cli_python_subprocess``, so it is deselected by default and can pass on
  the author's Intel box while silently never running on the AMD host where it
  would fail -- the failure only surfaces when someone forces integration marks
  on a different machine.

  Two ways to make such a comparison portable:
    1. Wrap BOTH paths in ``host_agnostic_numerics()`` (below). This upcasts
       every float32 arithmetic op to float64, so both reduce identically.
       This is the default choice for two paths YOU control.
    2. When one path is a third-party model (HuggingFace), the dispatch-mode
       upcast does NOT reach inside its FUSED kernels -- notably HF's default
       attention, ``F.scaled_dot_product_attention`` (SDPA), whose fp32
       accumulation order differs from a manual matmul+softmax. Upcasting your
       side alone then still diverges. Fix by forcing BOTH sides onto the SAME
       UNFUSED kernel: build HF with ``attn_implementation="eager"`` (plain
       matmul+softmax) and use loop's ``SdpaNaive`` attention kernel. Both then
       run the identical op sequence and ``torch.equal`` holds cross-platform.
       See ``priml/model/transformer/qwen3_hf_test.py`` for the canonical example.

  Do NOT "fix" such a test by loosening ``torch.equal`` to ``allclose`` with a
  tolerance -- that hides the very reduction-order regression the bit-for-bit
  check exists to catch. Align the kernels instead.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Final,
    NotRequired,
    TypedDict,
    cast,
    overload,
    override,
)

import os
import tempfile

from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode

import torch


if TYPE_CHECKING:
    from torch._ops import OpOverload


_ENV_REGENERATE: Final = "BFB_REGENERATE"


class _MissingGoldenError(AssertionError):
    """A missing BFB golden was minted and requires review."""


@dataclass(frozen=True, kw_only=True, slots=True)
class _TorchProcessState:
    """Process-global Torch state temporarily changed by a BFB assertion."""

    algorithms_enabled: bool
    warn_only_enabled: bool
    rng_state: Tensor


# The allowlist of float32 aten ops that run NATIVELY (not upcast) because their
# result is already bit-identical across x86 vector widths, ARM, and thread
# counts. Every float32 op *not* on this allowlist is upcast to float64 by
# default (see ``_Float64Compute``). Each entry is tagged with its category --
# the single source of truth the ``bfb_test.py`` guard derives from, so the
# allowlist and its proof obligations cannot drift:
#
#   "arith"    -- correctly-rounded elementwise arithmetic (IEEE-754 mandates
#                 one result); the guard requires a float64-recompute probe.
#   "compare"  -- boolean / index output, no rounding; exact by construction.
#   "movement" -- views, reshapes, copies, gather: move float32 bytes without
#                 arithmetic, so they cannot diverge; exact by construction.
#
# Matched by aten overloadpacket name (overload-independent). NOT on the list,
# and therefore upcast: every transcendental and reduction (host-dependent last
# bit), fused multiply-add (``addcmul_``/``addcdiv_``, whose ``a + b*c`` rounds
# differently than the float64 path), matmul (``mm``/``bmm``/..., whose float32
# kernel and reduction order are microarchitecture-dependent -- folding it into
# the upcast is what removes the MKL BLAS pin), and accumulating index ops
# (``scatter_add_``/``index_add_``/``embedding_dense_backward``, and
# ``index_put_`` with ``accumulate=True``), which sum float32 in a host-dependent
# order.
_EXACT_F32_OPS: Final[dict[str, str]] = {
    "add": "arith",
    "add_": "arith",
    "sub": "arith",
    "sub_": "arith",
    "mul": "arith",
    "mul_": "arith",
    "div": "arith",
    "div_": "arith",
    "neg": "arith",
    "abs": "arith",
    "clamp": "arith",
    "clamp_min": "arith",
    "clamp_max": "arith",
    "sign": "arith",
    "maximum": "arith",
    "minimum": "arith",
    "ge": "compare",
    "gt": "compare",
    "lt": "compare",
    "le": "compare",
    "eq": "compare",
    "ne": "compare",
    "where": "compare",
    "argmax": "compare",
    "argmin": "compare",
    "isnan": "compare",
    "isinf": "compare",
    "t": "movement",
    "transpose": "movement",
    "view": "movement",
    "_unsafe_view": "movement",
    "reshape": "movement",
    "_reshape_alias": "movement",
    "unsqueeze": "movement",
    "squeeze": "movement",
    "permute": "movement",
    "expand": "movement",
    "as_strided": "movement",
    "select": "movement",
    "slice": "movement",
    "narrow": "movement",
    "split": "movement",
    "split_with_sizes": "movement",
    "unbind": "movement",
    "chunk": "movement",
    "cat": "movement",
    "stack": "movement",
    "flatten": "movement",
    "clone": "movement",
    "contiguous": "movement",
    "copy_": "movement",
    "detach": "movement",
    "_to_copy": "movement",
    "to": "movement",
    "fill_": "movement",
    "zero_": "movement",
    "empty_like": "movement",
    "zeros_like": "movement",
    "ones_like": "movement",
    "new_empty": "movement",
    "new_empty_strided": "movement",
    "new_zeros": "movement",
    "new_ones": "movement",
    "embedding": "movement",
    "index": "movement",
    "index_select": "movement",
    "gather": "movement",
    "masked_fill": "movement",
    "masked_fill_": "movement",
    "masked_select": "movement",
    "select_backward": "movement",
    "slice_backward": "movement",
}


def _is_narrow_float(dtype: torch.dtype) -> bool:
    """Whether a dtype should be widened before the op runs.

    Named for the upcast's question -- "is this narrower than the scratch
    width" -- so float64 is False here because it IS the scratch, not because
    it is host-independent (it is not; see the module docstring). A caller
    asking whether a dtype is portable wants
    :func:`_assert_portable_output_dtype`, which admits float32 alone.

    bfloat16 and float16 count because a mixed-precision recipe COMPUTES in
    them -- an autocast forward, or an optimizer that orthogonalizes in half
    precision -- so a golden that left them native would be minted to one
    machine.
    """
    return dtype.is_floating_point and dtype != torch.float64


def _floating_dtypes(value: object) -> set[torch.dtype]:
    """The float dtypes appearing in a tensor / list / tuple.

    ``_foreach_*`` ops (e.g. ``_foreach_norm`` behind ``clip_grad_norm_``)
    receive a ``list[Tensor]`` rather than a bare tensor, so a direct
    ``isinstance(a, Tensor)`` check misses them and the upcast silently does
    not apply.
    """
    if isinstance(value, Tensor):
        return {value.dtype} if value.dtype.is_floating_point else set()
    if isinstance(value, (list, tuple)):
        found: set[torch.dtype] = set()
        for item in cast("list[object] | tuple[object, ...]", value):
            found |= _floating_dtypes(item)
        return found
    return set()


def _result_dtype(dtypes: set[torch.dtype]) -> torch.dtype:
    """The dtype the op would have produced natively.

    Torch promotes mixed inputs, so a bfloat16 tensor meeting a float32 one
    yields float32. Reproducing that promotion here is what lets the result be
    narrowed back to the width the unwrapped computation would have held --
    narrowing everything to float32 instead would silently widen a half
    precision graph and change every value downstream of it.
    """
    ordered = sorted(dtypes, key=str)
    result = ordered[0]
    for dtype in ordered[1:]:
        result = torch.promote_types(result, dtype)
    return result


def _upcast(value: object) -> object:
    if isinstance(value, Tensor) and _is_narrow_float(value.dtype):
        return value.double()
    if isinstance(value, list):
        return [_upcast(v) for v in cast(list[object], value)]
    if isinstance(value, tuple):
        return tuple(_upcast(v) for v in cast(tuple[object, ...], value))
    return value


def _downcast_f64(value: object, target: torch.dtype = torch.float32) -> object:
    if isinstance(value, Tensor) and value.dtype == torch.float64:
        return value.to(target)
    if isinstance(value, list):
        return [_downcast_f64(v, target) for v in cast(list[object], value)]
    if isinstance(value, tuple):
        return tuple(_downcast_f64(v, target) for v in cast(tuple[object, ...], value))
    return value


def _downcast_result(
    result: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    fallback: torch.dtype,
) -> object:
    """Downcast foreach results per element and ordinary results globally."""
    if not isinstance(result, list):
        return _downcast_f64(result, fallback)
    typed_result = cast(list[object], result)
    sequences: list[list[object] | tuple[object, ...]] = []
    shared_dtypes: set[torch.dtype] = set()
    for value in (*args, *kwargs.values()):
        if isinstance(value, list):
            sequence = cast(list[object], value)
        elif isinstance(value, tuple):
            sequence = cast(tuple[object, ...], value)
        else:
            shared_dtypes |= _floating_dtypes(value)
            continue
        if len(sequence) == len(typed_result):
            sequences.append(sequence)
    downcast = list[object]()
    for index, value in enumerate(typed_result):
        dtypes = set(shared_dtypes)
        for sequence in sequences:
            dtypes |= _floating_dtypes(sequence[index])
        target = _result_dtype(dtypes) if dtypes else fallback
        downcast.append(_downcast_f64(value, target))
    return downcast


def _copy_back(original: object, computed: object) -> None:
    """Narrow ``computed`` (float64) into ``original`` in place.

    Recurses into lists/tuples for ``_foreach_*`` write targets. A tensor that
    was never upcast -- one whose dtype is not a narrow float -- was written by
    the op itself, so it is skipped. The narrowing ``copy_`` is
    IEEE-correctly-rounded, hence host-independent.
    """
    if isinstance(original, Tensor):
        if _is_narrow_float(original.dtype) and isinstance(computed, Tensor):
            if original.shape != computed.shape:
                original.resize_(computed.shape)
            original.copy_(computed)
        return
    if isinstance(original, (list, tuple)) and isinstance(computed, (list, tuple)):
        for o, c in zip(
            cast("list[object] | tuple[object, ...]", original),
            cast("list[object] | tuple[object, ...]", computed),
            strict=True,
        ):
            _copy_back(o, c)


def _write_back(
    func: OpOverload[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    up_args: tuple[object, ...],
    up_kwargs: dict[str, object],
    *,
    result: object,
    target: torch.dtype,
) -> object:
    """Restore an in-place / ``out=`` / foreach op's mutation onto the originals.

    The op ran on the float64 upcast copies in ``up_args``/``up_kwargs``, so its
    computed values live there, not in the caller's float32 originals. For every
    write argument (``alias_info.is_write``), narrow its upcast copy back into the
    original (``_copy_back``, recursing through foreach ``Tensor[]``) as the side
    effect, and remember the (upcast-copy, original) pair.

    The return is then rebuilt element-wise: a returned element that IS one of the
    upcast write copies is swapped to the caller's original (preserving in-place /
    ``out=`` return identity); every other element -- a freshly-computed output
    that merely happens to ride alongside the writes, e.g.
    ``_native_batch_norm_legit`` returns ``(output, save_mean, save_invstd)`` while
    writing ``running_*`` -- is downcast and kept, never dropped. ``None`` (void
    in-place, e.g. ``_foreach_*_``) passes through, which the dispatcher requires.
    """
    schema = func._schema  # noqa: SLF001 -- OpOverload exposes its schema only privately
    # Copy each mutated float64 upcast copy back into its float32 original (the
    # side effect), recording (upcast_copy -> original) so a returned element
    # that IS a write target can be swapped to the caller's original. Returns
    # that are fresh (non-write) tensors -- e.g. ``_native_batch_norm_legit``
    # returns ``(output, save_mean, save_invstd)`` while writing ``running_*`` --
    # are simply downcast, never dropped.
    upcast_to_original: list[tuple[object, object]] = []
    for i, arg in enumerate(schema.arguments):
        if arg.alias_info is None or not arg.alias_info.is_write:
            continue
        name = arg.name
        original = kwargs[name] if name in kwargs else args[i]
        computed = up_kwargs[name] if name in up_kwargs else up_args[i]
        _copy_back(original, computed)
        upcast_to_original.append((computed, original))

    def _resolve(element: object) -> object:
        for computed, original in upcast_to_original:
            if element is computed:
                return original
        return _downcast_f64(element, target)

    if result is None:
        return None
    if isinstance(result, tuple):
        return tuple(_resolve(e) for e in cast(tuple[object, ...], result))
    if isinstance(result, list):
        return [_resolve(e) for e in cast(list[object], result)]
    return _resolve(result)


def _op_name(func: OpOverload[..., object]) -> str:
    """Return an Aten overload's packet name."""
    return func.name().split("::")[-1].split(".")[0]


class _Float64Compute(TorchDispatchMode):
    """Compute every float32 arithmetic op in float64, return float32.

    Operates at the aten-dispatch layer, so it sees the aten ops the computation
    issues during forward and autograd backward. An op is upcast when it has a
    float32 argument and its overloadpacket name is NOT
    in ``_EXACT_F32_OPS``; the float32 args are widened to float64, the op runs,
    and float64 results are narrowed back to float32. Allowlisted ops (exact
    elementwise arithmetic and pure data movement) pass through untouched.

    Upcast-by-default is the completeness guarantee: a transcendental or
    reduction absent from every list is still upcast, so it cannot silently mint
    a host-dependent golden. The only unsafe act is wrongly *adding* an op to
    ``_EXACT_F32_OPS``, which the guard test in ``bfb_test.py`` catches.
    """

    @override
    def __torch_dispatch__(
        self,
        func: OpOverload[..., object],
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}
        exact = _op_name(func) in _EXACT_F32_OPS
        input_dtypes: set[torch.dtype] = set()
        for value in (*args, *kwargs.values()):
            input_dtypes |= _floating_dtypes(value)
        narrow = {dtype for dtype in input_dtypes if _is_narrow_float(dtype)}
        if exact or not narrow:
            return func(*args, **kwargs)
        explicit_dtype = kwargs.get("dtype")
        target = (
            explicit_dtype
            if isinstance(explicit_dtype, torch.dtype)
            else _result_dtype(input_dtypes)
        )
        up_args = tuple(_upcast(a) for a in args)
        up_kwargs = {k: _upcast(v) for k, v in kwargs.items()}
        if isinstance(explicit_dtype, torch.dtype) and _is_narrow_float(explicit_dtype):
            up_kwargs["dtype"] = torch.float64
        result = func(*up_args, **up_kwargs)
        if any(
            arg.alias_info is not None and arg.alias_info.is_write
            for arg in func._schema.arguments  # noqa: SLF001 -- schema is OpOverload's only write-arg source
        ):
            # In-place / ``out=`` / foreach op: it mutated the float64 copies, not
            # the caller's originals. Narrow each back and return the originals
            # in the op's own return shape.
            return _write_back(
                func,
                args,
                kwargs,
                up_args,
                up_kwargs,
                result=result,
                target=target,
            )
        return _downcast_result(result, args, kwargs, target)


@contextmanager
def host_agnostic_numerics() -> Generator[None]:
    """Force the wrapped computation onto host-independent float kernels.

    Every float32 arithmetic op -- transcendental, reduction, matmul, forward or
    backward -- runs in float64 and downcasts to float32, except the
    ``_EXACT_F32_OPS`` allowlist of already-host-independent ops. A forward,
    backward, or full optimizer step is then bit-identical under
    ``torch.equal`` across hosts of any vector width, thread count, or vendor.

    That guarantee covers the float32 RESULT and nothing wider: the float64
    values inside carry each host's own libm error and are not comparable
    across machines. A caller that keeps one -- by returning it from a golden
    runner -- keeps the error too.

    Caveat for third-party interop: this intercepts ops at the aten-dispatch
    layer, so it upcasts everything torch itself dispatches -- but it does NOT
    reach inside a fused kernel that a third-party model invokes as one opaque
    call (e.g. HuggingFace's default ``F.scaled_dot_product_attention``). To
    compare bit-for-bit against such a model, also force it onto an UNFUSED
    kernel (HF: ``attn_implementation="eager"``) so both sides issue the same
    primitive ops. See the module docstring's cross-implementation section and
    ``priml/model/transformer/qwen3_hf_test.py``.
    """
    with sdpa_kernel(SDPBackend.MATH), _Float64Compute():
        yield


def bfb_devices() -> list[str]:
    """Return the sole device supported by portable BFB goldens.

    CPU goldens are portable because ``host_agnostic_numerics`` upcasts every
    float32 arithmetic operation to float64. CUDA has no equivalent portable
    contract: kernel selection and reduction order vary across GPU models, and
    initializing CUDA cannot be undone to make an in-process test hermetic.

    Returns:
      devices: ``["cpu"]``.

    """
    return ["cpu"]


@overload
def move_to_device[ValueT](
    value: dict[str, ValueT],
    device: str,
) -> dict[str, ValueT]: ...
@overload
def move_to_device(value: list[Tensor], device: str) -> list[Tensor]: ...
@overload
def move_to_device(value: object, device: str) -> object: ...
def move_to_device(value: object, device: str) -> object:
    """Recursively move tensors in a tensor / dict / list / tuple to device.

    Non-tensor leaves pass through untouched, so a batch mixing tensors with
    scalar metadata (e.g. ``valid_count``) moves cleanly.

    Args:
      value: A tensor, or a dict / list / tuple nesting tensors.
      device: Target device string (e.g. ``"cpu"`` or ``"cuda"``).

    Returns:
      moved: ``value`` with every tensor leaf on ``device``.

    """
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        typed_value = cast(dict[str, object], value)
        return {k: move_to_device(v, device) for k, v in typed_value.items()}
    if isinstance(value, tuple):
        typed_value = cast(tuple[object, ...], value)
        return tuple(move_to_device(v, device) for v in typed_value)
    if isinstance(value, list):
        typed_value = cast(list[object], value)
        return [move_to_device(v, device) for v in typed_value]
    return value


def first_tensor(result: object) -> Tensor:
    """Extract and validate the primary tensor from a module result.

    Args:
      result: A tensor, or a tuple or list whose first element is a tensor.

    Returns:
      tensor: The result itself, or its first element.

    Raises:
      TypeError: The result or its first element is not a tensor.

    """
    if isinstance(result, (tuple, list)):
        if not result or not isinstance(result[0], Tensor):
            raise TypeError("module result must start with a Tensor")
        return result[0]
    if not isinstance(result, Tensor):
        raise TypeError("module result must be a Tensor")
    return result


def _module_device(module: nn.Module) -> str:
    """Return the module's tensor device, defaulting tensorless modules to CPU."""
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device.type
    buffer = next(module.buffers(), None)
    return buffer.device.type if buffer is not None else "cpu"


def randomize_parameters(
    module: nn.Module,
    *,
    seed: int,
    std: float = 1.0,
) -> None:
    """Replace every parameter tensor with seeded ``randn * std``.

    Operates in-place. Buffers are left alone (RoPE cos/sin, dihedral
    caches, etc. are derived from config, not learned).

    Args:
      module: Module whose parameters to randomize.
      seed: Manual seed used for the random fill.
      std: Stddev of the normal distribution; defaults to 1.0.

    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            sample = torch.randn(p.shape, generator=gen, dtype=torch.float32) * std
            p.data.copy_(sample.to(p.dtype))


def assert_bfb_against_golden[InputT](
    *,
    golden_dir: Path,
    golden_name: str,
    build_module: Callable[[], nn.Module],
    build_input: Callable[[], InputT],
    seed: int = 0,
    run: Callable[[nn.Module, InputT], Tensor] | None = None,
) -> None:
    """Assert that running ``module(input)`` reproduces the saved golden.

    First call (no golden file present, or ``BFB_REGENERATE=1`` set):
      - Builds the module, builds the input, randomizes parameters under
        ``seed``, runs ``module(input)``, and writes ``{golden_name}.pt``
        containing the pre-run state_dict, input, output, and the post-run
        state_dict when the run mutates state. ``randomize_parameters`` uses its
        own seeded generator,
        so it is independent of the global RNG ``build_input`` may consume.
      - Immediately reloads the just-written golden, reruns, and asserts
        it round-trips bit-exactly. Regeneration fails loudly otherwise
        (INF-018), so a non-reproducible golden is never committed.
      - A missing golden is recreated but still fails the test, forcing review
        before the next run accepts it. Explicit regeneration returns normally
        after the same round-trip check.

    Subsequent calls:
      - Builds the module fresh, loads the pre-run state_dict, runs the
        module on the input, asserts the output matches the golden's
        output and the post-run state_dict matches the golden's post-run
        state_dict bit-for-bit.

    The post-run state is captured unconditionally: a non-mutating
    ``forward`` may still mutate registered buffers (BatchNorm
    ``running_mean``, EMA caches), and those mutations are part of the
    bit-for-bit contract (INF-017).

    Args:
      golden_dir: Directory holding ``.pt`` golden files. Created if
        missing.
      golden_name: Base name (no extension) for the golden file.
      build_module: Callable returning a fresh module. Called once per
        test invocation.
      build_input: Callable returning the input tensor (or a dict of
        tensors / tuple of tensors). Called once per test invocation.
      seed: Manual seed for module init randomization and input
        generation.
      run: Optional callable ``(module, input) -> Tensor``. Defaults to
        ``module(input)`` for tensor inputs, ``module(**input)`` for
        dict inputs, and ``module(*input)`` for tuple inputs.

    """
    state = _capture_torch_process_state()
    try:
        golden_dir.mkdir(parents=True, exist_ok=True)
        golden_path = golden_dir / f"{golden_name}.pt"
        missing = not golden_path.exists()
        regenerate = os.environ.get(_ENV_REGENERATE, "0") == "1"
        runner = _default_runner if run is None else run

        if missing or regenerate:
            with tempfile.NamedTemporaryFile(
                dir=golden_dir,
                prefix=f".{golden_name}.",
                suffix=".pt",
                delete=False,
            ) as candidate_file:
                candidate_path = Path(candidate_file.name)
            try:
                _write_golden(
                    golden_path=candidate_path,
                    build_module=build_module,
                    build_input=build_input,
                    seed=seed,
                    run=runner,
                )
                _replay_golden(
                    golden_path=candidate_path,
                    build_module=build_module,
                    seed=seed,
                    run=runner,
                )
                candidate_path.replace(golden_path)
            finally:
                candidate_path.unlink(missing_ok=True)
            if missing:
                raise _MissingGoldenError(
                    f"Missing golden regenerated at {golden_path}; inspect it, "
                    "then rerun the test."
                )
            return

        _replay_golden(
            golden_path=golden_path,
            build_module=build_module,
            seed=seed,
            run=runner,
        )
    finally:
        _restore_torch_process_state(state)


def _capture_torch_process_state() -> _TorchProcessState:
    """Capture every process-global setting changed by the BFB harness."""
    return _TorchProcessState(
        algorithms_enabled=torch.are_deterministic_algorithms_enabled(),
        warn_only_enabled=torch.is_deterministic_algorithms_warn_only_enabled(),
        rng_state=torch.get_rng_state(),
    )


def _restore_torch_process_state(state: _TorchProcessState) -> None:
    """Restore every process-global setting changed by the BFB harness."""
    torch.use_deterministic_algorithms(
        state.algorithms_enabled,
        warn_only=state.warn_only_enabled,
    )
    torch.set_rng_state(state.rng_state)


def _seed_bfb(seed: int) -> None:
    """Seed the CPU default generator without queuing a lazy CUDA seed."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    torch.set_rng_state(generator.get_state())


def regenerate_golden[InputT](
    *,
    golden_dir: Path,
    golden_name: str,
    build_module: Callable[[], nn.Module],
    build_input: Callable[[], InputT],
    seed: int = 0,
    run: Callable[[nn.Module, InputT], Tensor] | None = None,
) -> None:
    """Force-regenerate a golden, ignoring any existing file.

    Equivalent to setting ``BFB_REGENERATE=1`` and calling
    ``assert_bfb_against_golden`` once. The freshly written golden is
    replayed and must round-trip bit-exactly (INF-018).

    Args:
      golden_dir: Directory for the golden file.
      golden_name: Base name for the golden file.
      build_module: Module factory.
      build_input: Input factory.
      seed: Manual seed.
      run: Optional custom runner.

    """
    prior = os.environ.get(_ENV_REGENERATE)
    os.environ[_ENV_REGENERATE] = "1"
    try:
        with suppress(_MissingGoldenError):
            assert_bfb_against_golden(
                golden_dir=golden_dir,
                golden_name=golden_name,
                build_module=build_module,
                build_input=build_input,
                seed=seed,
                run=run,
            )
    finally:
        # Restore the caller's prior value rather than unconditionally
        # clearing it, so calling this inside a process launched with
        # BFB_REGENERATE=1 does not silently disable regeneration afterward.
        if prior is None:
            os.environ.pop(_ENV_REGENERATE, None)
        else:
            os.environ[_ENV_REGENERATE] = prior


class _Golden(TypedDict):
    """What a golden file stores.

    ``post_state_dict`` is absent when the run mutated nothing, which the
    replay reads as "equal to ``state_dict``".
    """

    state_dict: dict[str, Tensor]
    input: object
    output: Tensor
    seed: int
    post_state_dict: NotRequired[dict[str, Tensor]]


def _write_golden[InputT](
    *,
    golden_path: Path,
    build_module: Callable[[], nn.Module],
    build_input: Callable[[], InputT],
    seed: int,
    run: Callable[[nn.Module, InputT], Tensor],
) -> None:
    """Build, randomize, run, and snapshot pre- and post-run state."""
    torch.use_deterministic_algorithms(True)
    _seed_bfb(seed)
    module = build_module()
    device = _module_device(module)
    if device != "cpu":
        raise ValueError("The BFB harness is CPU-only.")
    inp = build_input()
    randomize_parameters(module, seed=seed)
    pre_state = _cpu_state_dict(module.state_dict())
    with host_agnostic_numerics():
        output = run(module, inp)
    _assert_portable_output_dtype(output)
    post_state = _cpu_state_dict(module.state_dict())
    payload: _Golden = {
        "state_dict": pre_state,
        "input": _to_cpu(inp),
        "output": output.detach().cpu(),
        "seed": seed,
    }
    # Absence means "unchanged", which the replay asserts against the pre-run
    # copy -- so omitting it is not a weaker check.
    if state_differs(pre_state, post_state):
        payload["post_state_dict"] = post_state
    torch.save(payload, golden_path)


def _assert_portable_output_dtype(output: Tensor) -> None:
    """Refuse a golden comparand that skipped the round back to float32.

    bfloat16 and float16 are refused too, not float64 alone: the harness
    computes in all three and only the rounding makes a value portable (see
    the module docstring). Complex outputs are unsupported. Integers carry no
    rounding and pass.

    Args:
      output: What the runner returned, and what the golden compares.

    Raises:
      TypeError: The output is complex or a float other than float32.

    """
    if output.dtype.is_complex:
        raise TypeError(
            f"bfb golden output is {output.dtype}, which is not supported; "
            "return a float32 or integer tensor."
        )
    if not output.dtype.is_floating_point or output.dtype == torch.float32:
        return
    raise TypeError(
        f"bfb golden output is {output.dtype}, which is not portable across "
        "hosts; it must be float32. host_agnostic_numerics computes in "
        "float64 and the ROUND BACK to float32 is what makes the result "
        "host-independent; returning the unrounded value stores this host's "
        "libm error. Narrow in the runner: `return value.float()`.",
    )


def state_differs(before: Mapping[str, Tensor], after: Mapping[str, Tensor]) -> bool:
    """Whether any stored tensor changed across the run.

    Public because it decides whether a golden STORES ``post_state_dict``, so
    anything reasoning about that key asks the same question. A second
    spelling of "did the state change" is how two callers drift.

    Args:
      before: State captured before the run.
      after: State captured after it.

    Returns:
      differs: Whether any entry changed.

    """
    if before.keys() != after.keys():
        return True
    return any(
        not _tensor_bits_equal(value, after[key]) for key, value in before.items()
    )


def _replay_golden[InputT](
    *,
    golden_path: Path,
    build_module: Callable[[], nn.Module],
    seed: int,
    run: Callable[[nn.Module, InputT], Tensor],
) -> None:
    """Load a golden, rerun, and assert output and post-run state match."""
    torch.use_deterministic_algorithms(True)
    _seed_bfb(seed)
    module = build_module()
    device = _module_device(module)
    if device != "cpu":
        raise ValueError("The BFB harness is CPU-only.")
    payload = cast(
        _Golden, torch.load(golden_path, weights_only=False, map_location="cpu")
    )
    module.load_state_dict(payload["state_dict"])
    inp = cast(InputT, move_to_device(payload["input"], device))
    with host_agnostic_numerics():
        output = run(module, inp)
    # Checked on replay too, not only at mint: a golden written before this
    # gate exists still carries a float64 comparand, and reporting WHY it is
    # unportable beats an opaque one-ULP mismatch on someone else's host.
    _assert_portable_output_dtype(output)
    # Absent means the run did not mutate its state, so the pre-run copy IS
    # the expectation -- a mutation introduced later then fails against it.
    _assert_equal(output, payload["output"], label="output")
    _assert_state_match(module, payload.get("post_state_dict", payload["state_dict"]))


def _default_runner(module: nn.Module, inp: object) -> Tensor:
    if isinstance(inp, dict):
        return cast(Tensor, module(**inp))
    if isinstance(inp, (list, tuple)):
        return cast(Tensor, module(*inp))
    return cast(Tensor, module(inp))


def _assert_state_match(module: nn.Module, golden: Mapping[str, Tensor]) -> None:
    live = module.state_dict()
    live_keys = set(live.keys())
    golden_keys = set(golden.keys())
    if live_keys != golden_keys:
        added = live_keys - golden_keys
        removed = golden_keys - live_keys
        raise AssertionError(
            f"state_dict keys differ: added={sorted(added)} removed={sorted(removed)}",
        )
    for k in sorted(live_keys):
        _assert_equal(live[k], golden[k], label=f"state[{k}]")


def _to_cpu(value: object) -> object:
    if torch.is_tensor(value):
        # ``.cpu()`` on an already-CPU tensor shares storage; clone so a
        # snapshot of a parameter cannot be corrupted by a later in-place
        # mutation of the live module (mutating runners, EMA buffers).
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        typed_value = cast(dict[str, object], value)
        return {k: _to_cpu(v) for k, v in typed_value.items()}
    if isinstance(value, tuple):
        typed_value = cast(tuple[object, ...], value)
        return tuple(_to_cpu(v) for v in typed_value)
    if isinstance(value, list):
        typed_value = cast(list[object], value)
        return [_to_cpu(v) for v in typed_value]
    return value


def _cpu_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        key: value.detach().to("cpu", copy=True) for key, value in state_dict.items()
    }


def _assert_equal(a: object, b: object, *, label: str) -> None:
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        if a != b:
            raise AssertionError(f"{label}: non-tensor mismatch {a!r} vs {b!r}")
        return
    if a.device != b.device:
        a = a.cpu()
        b = b.cpu()
    if a.shape != b.shape:
        raise AssertionError(
            f"{label}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}",
        )
    if a.dtype != b.dtype:
        raise AssertionError(f"{label}: dtype mismatch {a.dtype} vs {b.dtype}")
    if not _tensor_bits_equal(a, b):
        if a.dtype.is_floating_point or a.dtype.is_complex:
            max_abs_diff = f"{float((a - b).abs().max().item()):.3e}"
        else:
            values_a = cast(list[int], a.detach().cpu().reshape(-1).tolist())
            values_b = cast(list[int], b.detach().cpu().reshape(-1).tolist())
            max_abs_diff = str(
                max(
                    abs(value_a - value_b)
                    for value_a, value_b in zip(values_a, values_b, strict=True)
                )
            )
        raise AssertionError(
            f"{label}: bitwise comparison failed "
            f"(max_abs_diff={max_abs_diff}, max_ulp_diff={_max_ulp_diff(a, b)})",
        )


def _tensor_bits_equal(a: Tensor, b: Tensor) -> bool:
    """Whether two tensors have identical shapes, dtypes, and stored bits."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.device != b.device:
        a = a.cpu()
        b = b.cpu()
    a_bits = a.detach().contiguous().reshape(-1).view(torch.uint8)
    b_bits = b.detach().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(a_bits, b_bits)


def _max_ulp_diff(a: Tensor, b: Tensor) -> int | str:
    """Largest gap in representable steps, or why it could not be measured.

    The unit a bit-for-bit failure is actually measured in: 1 says the hosts
    round differently, a large count says the computation changed, and an
    absolute difference says neither on its own (1 ULP is 1e-7 near one and
    1e-45 near zero).

    Bit patterns are ordered only WITHIN a sign -- the negative half is stored
    sign-magnitude, counting away from zero -- so they are mapped to one
    monotone line first. Subtracting raw patterns instead reports ~2**31 for
    two neighbours straddling zero, which is the magnitude a total regression
    produces, from the case where the values are closest.
    """
    kind = {
        torch.float64: torch.int64,
        torch.float32: torch.int32,
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
    }.get(a.dtype)
    if kind is None:
        return "n/a"
    # NaN has no distance to anything: every comparison against it is false, so
    # a pattern subtraction returns a number that reads as real drift.
    if bool(a.isnan().any() or b.isnan().any()):
        return "nan"
    return int((_ordered(a, kind) - _ordered(b, kind)).abs().max())


def _ordered(value: Tensor, kind: torch.dtype) -> Tensor:
    """Reinterpret floats as integers that increase with the float's value.

    A negative float's pattern grows as the number falls, so the negative half
    is reflected. The result orders the whole line, which is what makes a
    subtraction count representable steps.

    Reflected about the float's OWN signed minimum, not int64's: the pattern is
    widened for the arithmetic, and reflecting about the wide minimum would
    offset the negative half by the difference between the two widths.
    """
    bits = value.detach().contiguous().view(kind)
    floor = torch.iinfo(bits.dtype).min
    wide = bits.to(torch.int64)
    return torch.where(wide < 0, floor - wide, wide)
