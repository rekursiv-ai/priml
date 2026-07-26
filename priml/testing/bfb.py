"""Bit-for-bit golden-file unit-test harness.

Pattern (see write-code skill rationale):

1. **Build** a module at minimum width: 1 layer, hidden=8, smallest seq_len.
2. **Randomize** every parameter with ``torch.randn_like`` so structurally-zero
   inits (q-head bias, etc.) don't hide a regression.
3. **Snapshot** ``(state_dict, input, output)`` to ``<test_file_dir>/goldens/``.
4. **Assert** on subsequent runs that loading the golden state and applying it
   to the same input produces a ``torch.equal`` output.

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
  op in float64 and downcasting the result back to float32. Float64 polynomials
  and reductions agree across hosts to far more than float32 precision, so the
  downcast lands on the same float32 bit everywhere. Only pure data-movement ops
  (views, reshapes, gathers) and correctly-rounded elementwise ops -- whose
  float32 result is already host-independent -- stay in float32; they are the
  ``_EXACT_F32_OPS`` allowlist. Every other float32 op is upcast by default, so
  a newly-introduced transcendental cannot silently leak: forgetting to list it
  upcasts it anyway. The allowlist is closed (IEEE-754 fixes which ops are
  correctly-rounded; views do no arithmetic) and guarded by a unit test that
  proves each entry is genuinely host-independent.

  Because matmul is upcast like everything else (float64 GEMM is both
  vector-width- and thread-count-invariant), the golden needs no MKL BLAS pin
  and no host-class gating of its own -- x86 (any width), ARM, Apple silicon,
  and OpenBLAS builds all reproduce. (The repo-root conftest still sets
  ``MKL_CBWR`` for the benefit of other, non-bfb numeric-parity tests; bfb no
  longer depends on it.)

Determinism is required: the harness enables deterministic Torch algorithms
and seeds the CPU default generator before any tensor allocation.

Usage::

    from priml.testing.bfb import assert_bfb_against_golden

    def test_mymodule_bfb() -> None:
        cfg = MyModule.Config(channels=8, num_layers=1)
        assert_bfb_against_golden(
            golden_dir=Path(__file__).parent.resolve() / "goldens",
            golden_name="mymodule_min",
            build_module=lambda: cfg.make(),
            build_input=lambda: torch.randn(2, 4, 8),
            seed=0,
        )

The test fails if either (a) shapes mismatch, (b) ``torch.equal`` returns
``False`` on the output, or (c) the state_dict keys differ from the golden.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, override

import os

from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode

import pytest
import torch


_ENV_REGENERATE = "BFB_REGENERATE"  # config-globals: ignore -- test env var name.


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
_EXACT_F32_OPS: dict[str, str] = {  # config-globals: ignore -- BFB op allowlist.
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


def _has_f32(value: object) -> bool:
    """True if value is a float32 tensor or a list/tuple containing one.

    ``_foreach_*`` ops (e.g. ``_foreach_norm`` behind ``clip_grad_norm_``)
    receive a ``list[Tensor]`` rather than a bare tensor, so a direct
    ``isinstance(a, Tensor)`` check misses them and the upcast silently does
    not apply.
    """
    if isinstance(value, Tensor):
        return value.dtype == torch.float32
    if isinstance(value, (list, tuple)):
        return any(_has_f32(v) for v in cast("list[Any] | tuple[Any, ...]", value))
    return False


def _upcast_f32(value: object) -> object:
    if isinstance(value, Tensor) and value.dtype == torch.float32:
        return value.double()
    if isinstance(value, list):
        return [_upcast_f32(v) for v in cast("list[Any]", value)]
    if isinstance(value, tuple):
        return tuple(_upcast_f32(v) for v in cast("tuple[Any, ...]", value))
    return value


def _downcast_f64(value: object) -> object:
    if isinstance(value, Tensor) and value.dtype == torch.float64:
        return value.float()
    if isinstance(value, list):
        return [_downcast_f64(v) for v in cast("list[Any]", value)]
    if isinstance(value, tuple):
        return tuple(_downcast_f64(v) for v in cast("tuple[Any, ...]", value))
    return value


def _copy_back(original: object, computed: object) -> None:
    """Narrow ``computed`` (float64) into ``original`` (float32) in place.

    Recurses into lists/tuples for ``_foreach_*`` write targets. A non-float32
    original was never upcast, so the op already wrote it -- skip. The
    float64->float32 ``copy_`` is IEEE-correctly-rounded, hence host-independent.
    """
    if isinstance(original, Tensor):
        if original.dtype == torch.float32 and isinstance(computed, Tensor):
            original.copy_(computed)
        return
    if isinstance(original, (list, tuple)) and isinstance(computed, (list, tuple)):
        for o, c in zip(
            cast("list[Any] | tuple[Any, ...]", original),
            cast("list[Any] | tuple[Any, ...]", computed),
            strict=True,
        ):
            _copy_back(o, c)


def _write_back(
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    up_args: tuple[Any, ...],
    up_kwargs: dict[str, Any],
    *,
    result: object,
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
    upcast_to_original: list[tuple[Any, Any]] = []
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
        return _downcast_f64(element)

    if result is None:
        return None
    if isinstance(result, tuple):
        return tuple(_resolve(e) for e in cast("tuple[Any, ...]", result))
    if isinstance(result, list):
        return [_resolve(e) for e in cast("list[Any]", result)]
    return _resolve(result)


class _Float64Compute(TorchDispatchMode):
    """Compute every float32 arithmetic op in float64, return float32.

    Operates at the aten-dispatch layer, so it sees every op the computation
    issues -- forward, autograd backward, and fused-op internals alike. An op
    is upcast when it has a float32 argument and its overloadpacket name is NOT
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
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        kwargs = kwargs or {}
        exact = func.overloadpacket.__name__ in _EXACT_F32_OPS
        f32_args = any(_has_f32(a) for a in args) or any(
            _has_f32(v) for v in kwargs.values()
        )
        if exact or not f32_args:
            return func(*args, **kwargs)
        up_args = tuple(_upcast_f32(a) for a in args)
        up_kwargs = {k: _upcast_f32(v) for k, v in kwargs.items()}
        result = func(*up_args, **up_kwargs)
        if any(
            arg.alias_info is not None and arg.alias_info.is_write
            for arg in func._schema.arguments  # noqa: SLF001 -- schema is OpOverload's only write-arg source
        ):
            # In-place / ``out=`` / foreach op: it mutated the float64 copies, not
            # the caller's float32 originals. Narrow each back and return the
            # originals in the op's own return shape.
            return _write_back(func, args, kwargs, up_args, up_kwargs, result=result)
        return _downcast_f64(result)


@contextmanager
def host_agnostic_numerics() -> Generator[None]:
    """Force the wrapped computation onto host-independent float kernels.

    Every float32 arithmetic op -- transcendental, reduction, matmul, forward or
    backward -- runs in float64 and downcasts to float32, except the
    ``_EXACT_F32_OPS`` allowlist of already-host-independent ops. A forward,
    backward, or full optimizer step is then bit-identical across hosts of any
    vector width or thread count under ``torch.equal``.
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


def move_to_device(value: Any, device: str) -> Any:
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
        typed_value = cast("dict[str, Any]", value)
        return {k: move_to_device(v, device) for k, v in typed_value.items()}
    if isinstance(value, tuple):
        typed_value = cast("tuple[Any, ...]", value)
        return tuple(move_to_device(v, device) for v in typed_value)
    if isinstance(value, list):
        typed_value = cast("list[Any]", value)
        return [move_to_device(v, device) for v in typed_value]
    return value


def _module_device(module: nn.Module) -> str:
    """Return the module's compute device type (``"cpu"`` or ``"cuda"``)."""
    return next(module.parameters()).device.type


def _sdpa_fingerprint(device: str) -> dict[str, bool | str]:
    """Capture the attention-kernel fingerprint for the golden's device.

    A CPU golden computes entirely on CPU; ``F.scaled_dot_product_attention``
    dispatches to the device-independent CPU math kernel, so the result does
    not depend on whether the host has a GPU. The CPU fingerprint is therefore
    just ``{"device": "cpu"}`` -- it never consults ``torch.cuda``, so a CPU
    golden is portable to every host regardless of GPU presence, absence, or a
    wedged driver.

    A CUDA golden's result does depend on which attention backend
    ``enable_determinism`` selected (flash vs mem-efficient vs math sum in
    different reduction orders), so its fingerprint records those flags and a
    replay under a different backend is caught.

    Args:
      device: The golden's compute device type (``"cpu"`` or ``"cuda"``).

    Returns:
      fingerprint: ``{"device": "cpu"}`` for CPU; the CUDA backend flags for
        CUDA.

    """
    if device != "cuda":
        return {"device": "cpu"}
    return {
        "device": "cuda",
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
    }


def randomize_parameters(
    module: nn.Module,
    *,
    seed: int,
    std: float = 1.0,
) -> None:
    """Replace every parameter tensor with ``randn_like * std``.

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


def assert_bfb_against_golden(
    *,
    golden_dir: Path,
    golden_name: str,
    build_module: Callable[[], nn.Module],
    build_input: Callable[[], Any],
    seed: int = 0,
    run: Callable[[nn.Module, Any], Tensor] | None = None,
) -> None:
    """Assert that running ``module(input)`` reproduces the saved golden.

    First call (no golden file present, or ``BFB_REGENERATE=1`` set):
      - Builds the module, builds the input, randomizes parameters under
        ``seed``, runs ``module(input)``, and writes ``{golden_name}.pt``
        containing the pre-run state_dict, input, output, and the POST-run
        state_dict. ``randomize_parameters`` uses its own seeded generator,
        so it is independent of the global RNG ``build_input`` may consume.
      - Immediately reloads the just-written golden, reruns, and asserts
        it round-trips bit-exactly. Regeneration fails loudly otherwise
        (INF-018), so a non-reproducible golden is never committed.

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
        regenerate = (
            os.environ.get(_ENV_REGENERATE, "0") == "1" or not golden_path.exists()
        )

        if regenerate:
            _write_golden(
                golden_path=golden_path,
                build_module=build_module,
                build_input=build_input,
                seed=seed,
                run=run or _default_runner,
            )
            _replay_golden(
                golden_path=golden_path,
                build_module=build_module,
                seed=seed,
                run=run or _default_runner,
            )
            return

        _replay_golden(
            golden_path=golden_path,
            build_module=build_module,
            seed=seed,
            run=run or _default_runner,
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


def regenerate_golden(
    *,
    golden_dir: Path,
    golden_name: str,
    build_module: Callable[[], nn.Module],
    build_input: Callable[[], Any],
    seed: int = 0,
    run: Callable[[nn.Module, Any], Tensor] | None = None,
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


def _write_golden(
    *,
    golden_path: Path,
    build_module: Callable[[], nn.Module],
    build_input: Callable[[], Any],
    seed: int,
    run: Callable[[nn.Module, Any], Tensor],
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
    post_state = _cpu_state_dict(module.state_dict())
    torch.save(
        {
            "state_dict": pre_state,
            "post_state_dict": post_state,
            "input": _to_cpu(inp),
            "output": output.detach().cpu(),
            "seed": seed,
            "sdpa_backend": _sdpa_fingerprint(_module_device(module)),
        },
        golden_path,
    )


def _replay_golden(
    *,
    golden_path: Path,
    build_module: Callable[[], nn.Module],
    seed: int,
    run: Callable[[nn.Module, Any], Tensor],
) -> None:
    """Load a golden, rerun, and assert output and post-run state match."""
    torch.use_deterministic_algorithms(True)
    _seed_bfb(seed)
    module = build_module()
    device = _module_device(module)
    if device != "cpu":
        raise ValueError("The BFB harness is CPU-only.")
    payload = torch.load(golden_path, weights_only=False, map_location="cpu")
    _assert_sdpa_backend_match(payload.get("sdpa_backend"), device=device)
    module.load_state_dict(payload["state_dict"])
    inp = move_to_device(payload["input"], device)
    with host_agnostic_numerics():
        output = run(module, inp)
    _assert_state_match(module, payload["post_state_dict"])
    _assert_equal(output, payload["output"], label="output")


def _default_runner(module: nn.Module, inp: Any) -> Tensor:
    if isinstance(inp, dict):
        return module(**inp)
    if isinstance(inp, (list, tuple)):
        return module(*inp)
    return module(inp)


def _assert_sdpa_backend_match(
    golden: dict[str, bool | str] | None, *, device: str
) -> None:
    """Skip or fail when the replay SDPA backend differs from the golden's.

    A None fingerprint means a pre-fingerprint golden; skip the check so
    legacy goldens still load (they are caught by the output assertion).

    A CPU golden replayed on CPU is always reproducible -- the CPU attention
    kernel is device-independent and does not consult ``torch.cuda`` -- so a
    CPU golden never skips. It runs and compares on any host regardless of GPU
    presence, absence, or a wedged driver.

    A differing ``device`` means the golden's kernel cannot be reproduced on
    this run's device class (e.g. a CUDA golden replayed on CPU), so the
    bit-for-bit comparison is meaningless here -- skip rather than fail. A
    same-device CUDA sub-backend mismatch is drift within one device class,
    which IS reproducible and fixable on this host, so keep failing loudly.

    Args:
      golden: The fingerprint recorded in the golden, or None for legacy
        goldens.
      device: This run's compute device type (``"cpu"`` or ``"cuda"``).

    Raises:
      AssertionError: If the CUDA backend differs within the same device
        class, which would otherwise surface as an opaque last-bit mismatch.

    """
    if golden is None:
        return
    live = _sdpa_fingerprint(device)
    if live == golden:
        return
    if live.get("device") != golden.get("device"):
        pytest.skip(
            "SDPA backend unreproducible on this run's device class: "
            f"golden={golden} replay={live} (a golden replayed on a different "
            "device than it was minted on). Regenerate on this device with "
            "BFB_REGENERATE=1, or replay on the golden's device.",
        )
    raise AssertionError(
        "SDPA backend mismatch between golden write and replay: "
        f"golden={golden} replay={live}. The golden was snapshotted under a "
        "different attention kernel on the same device class (e.g. flash "
        "enabled at write, disabled at replay). Regenerate the golden on the "
        "replay host with BFB_REGENERATE=1, or pin a backend-independent "
        "kernel.",
    )


def _assert_state_match(module: nn.Module, golden: dict[str, Any]) -> None:
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


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        # ``.cpu()`` on an already-CPU tensor shares storage; clone so a
        # snapshot of a parameter cannot be corrupted by a later in-place
        # mutation of the live module (mutating runners, EMA buffers).
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        typed_value = cast("dict[str, Any]", value)
        return {k: _to_cpu(v) for k, v in typed_value.items()}
    if isinstance(value, tuple):
        typed_value = cast("tuple[Any, ...]", value)
        return tuple(_to_cpu(v) for v in typed_value)
    if isinstance(value, list):
        typed_value = cast("list[Any]", value)
        return [_to_cpu(v) for v in typed_value]
    return value


def _cpu_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {k: _to_cpu(v) for k, v in state_dict.items()}


def _assert_equal(a: Any, b: Any, *, label: str) -> None:
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
    if not torch.equal(a, b):
        d = (a.float() - b.float()).abs().max().item()
        raise AssertionError(
            f"{label}: torch.equal failed (max_abs_diff={d:.3e})",
        )
