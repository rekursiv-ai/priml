"""Bit-for-bit golden-file unit-test harness.

Pattern:

1. **Build** a module at minimum size: every dim at least 2, pairwise distinct.
2. **Randomize** every parameter with seeded ``torch.randn`` so structurally-zero
   inits (q-head bias, etc.) don't hide a regression.
3. **Snapshot** to ``<test_file_dir>/testdata/`` the pre-run state, input,
   output, and every post-run tensor the run changed, all whole.
4. **Assert** on subsequent runs that loading the golden state and applying it
   to the same input reproduces the output and post-run state bit for bit.

Regenerate (after an intentional numeric change)::

    uv --quiet run --frozen pytest <test_file> --regenerate-b4b

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
  disagreement in the measured probes -- 0 of 4096 wrong for every op probed.
  This is an empirical portability policy: severe cancellation or a value at
  a rounding boundary can expose differences even after widening. Biased
  convolutions and affine matmuls therefore separate the dot product, scaling,
  and bias addition before rounding. A golden's floating comparand
  must be float32; ``_assert_portable_output_dtype`` also rejects complex
  outputs. A runner returning the float64 scratch keeps that host's own libm
  error (one did, off by 1 ULP between an Intel laptop and an AMD server). Only
  pure data-movement ops (views, reshapes, gathers) and correctly-rounded
  elementwise ops -- whose float32 result is already host-independent -- stay
  in float32; they are the
  ``_EXACT_F32_OPS`` allowlist. Every other float32 op is upcast by default, so
  a newly-introduced transcendental cannot silently leak: forgetting to list it
  upcasts it anyway. The allowlist is closed (IEEE-754 fixes which ops are
  correctly-rounded; views do no arithmetic) and guarded by a unit test that
  proves each entry is genuinely host-independent.

  Because matmul is upcast like everything else, the golden needs no MKL BLAS
  pin and no host-class gating of its own -- x86 (any width), ARM, Apple
  silicon, and OpenBLAS builds reproduce the exercised float32 comparands
  (measured across Intel and AMD, and across 1/2/4/8/64 math threads: identical
  bits). Float64 GEMM is not itself invariant; it is the rounding that makes
  the result so -- and the rounding absorbs a float64 difference only until the
  exact value sits near a float32 boundary, which is why the priml conftest
  still pins ``MKL_CBWR``. Removing that pin was measured inert on AMD, where
  MKL takes a generic path, and broke goldens on Intel.

Determinism is required: the harness enables deterministic Torch algorithms
and seeds the CPU default generator before any tensor allocation. Every step --
building the module, randomizing it, building the input, and running -- sits
inside one ``host_agnostic_numerics``, so an initializer or input drawn with
``randn`` is as portable as the forward.

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

import tempfile

from torch import Tensor, nn
from torch._decomp import decomposition_table
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND, elementwise_dtypes
from torch._prims_common.wrappers import out_wrapper
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils._python_dispatch import TorchDispatchMode

import torch

from priml.lib.codec import from_plain
from priml.testing import regenerate
from priml.testing.golden import pack, tensor_bits_equal, unpack


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence

    from torch._ops import OpOverload


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
#
# Absence from this list upcasts an op only when it HAS a float32 argument. The
# random factories (``rand``/``randn``/``normal``) have none -- their dtype is a
# kwarg or the process default -- so they are named in ``_FLOAT_FACTORIES`` and
# widened by that dtype instead. Sampling is arithmetic: a float32 ``randn`` of
# 16+ elements runs torch's vectorized Box-Muller, whose ``log``/``cos`` come
# from SLEEF under AVX2 and from libm elsewhere. Measured, x86 mint vs aarch64
# replay with the identical generator state: up to 6 ULP on the draws. The
# float64 fill uses libm ``double`` on every host, and the round to float32
# absorbs its last-bit error like any other upcast op. In-place samplers
# (``normal_``/``uniform_``/``bernoulli_``) already carry a tensor argument and
# need no naming.
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
    "set_": "movement",
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

# These arithmetic factories can have no tensor operand to identify their dtype.
# Integer arange stays native; uniform rand already produces identical bytes.
_FLOAT_FACTORIES: Final = frozenset(
    {"randn", "normal", "linspace", "logspace", "arange"},
)

# Namespaces of distributed collective ops (``dist.broadcast``/``dist.all_reduce``
# and their functional-collective form); see ``_Float64Compute.__torch_dispatch__``.
_COLLECTIVE_NAMESPACES: Final = frozenset({"c10d", "_c10d_functional"})


@contextmanager
def host_agnostic_numerics() -> Generator[None]:
    """Force the wrapped computation onto host-independent float kernels.

    Every float32 arithmetic op -- transcendental, reduction, matmul, forward or
    backward -- runs in float64 and downcasts to float32, except the
    ``_EXACT_F32_OPS`` allowlist of already-host-independent ops. Known fused
    operations are decomposed before rounding. Portability is checked against
    shared goldens on each supported host; widening alone does not prove
    equivalence for arbitrary cancellation or rounding-boundary inputs.

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

    Yields:
      item: Nothing; used as a context manager for computation.

    """
    with sdpa_kernel(SDPBackend.MATH), _Float64Compute():
        yield


@contextmanager
def portable_half_precision() -> Generator[None]:
    """Run bf16/f16 CPU arithmetic on torch's native kernels, not oneDNN's.

    oneDNN refuses the half-precision BACKWARD of a convolution on a CPU
    whose ISA it does not cover (``DNNL does not support bf16/f16 backward on
    the platform with avx2_vnni_2``), so a parity test comparing gradients
    fails on that host and passes on an AVX-512 one. Torch's fallback kernels
    run everywhere; the two sides of a comparison both take them here, so
    the test measures the model and not the host.

    Yields:
      item: Nothing; used as a context manager for computation.

    """
    # Only the enable bit: ``torch.backends.mkldnn.flags`` also rewrites the
    # oneDNN TF32 setting, which warns on every CPU-only build and the suite
    # runs with warnings as errors.
    (enabled,) = torch.backends.mkldnn.set_flags(False, _fp32_precision=None)[:1]
    try:
        yield
    finally:
        torch.backends.mkldnn.set_flags(enabled, _fp32_precision=None)


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
def move_to_device(value: Tensor, device: str) -> Tensor: ...
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
            p.data.copy_(sample)


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

    First call (no golden file present, or ``--regenerate-b4b`` passed):
      - Builds the module, builds the input, randomizes parameters under
        ``seed``, runs ``module(input)``, and writes ``{golden_name}.pt``
        containing the pre-run state_dict, input, output, and every post-run
        tensor the run changed, all whole. ``randomize_parameters`` uses its own
        seeded generator, so it is independent of the global RNG ``build_input``
        may consume.
      - Immediately reloads the just-written golden, reruns, and asserts
        it round-trips bit-exactly. Regeneration fails loudly otherwise, so a
        non-reproducible golden is never committed.
      - A missing golden is recreated but still fails the test, forcing review
        before the next run accepts it. Explicit regeneration returns normally
        after the same round-trip check.

    Subsequent calls:
      - Builds the module fresh, loads the pre-run state_dict, runs the
        module on the input, and asserts the output and every post-run
        tensor match the golden bit-for-bit.

    The post-run state is captured unconditionally: a non-mutating
    ``forward`` may still mutate registered buffers (BatchNorm
    ``running_mean``, EMA caches), and those mutations are part of the
    bit-for-bit contract.

    Args:
      golden_dir: Directory holding ``.pt`` golden files. Created if
        missing.
      golden_name: Base name (no extension) for the golden file.
      build_module: Callable returning a fresh module. Called once per replay,
        and once more to capture a new golden.
      build_input: Callable returning the input tensor (or a dict of
        tensors / tuple of tensors). Called once per replay, and once more
        to capture a new golden.
      seed: Manual seed for module init randomization and input
        generation.
      run: Optional callable ``(module, input) -> Tensor``. Defaults to
        ``module(input)`` for tensor inputs, ``module(**input)`` for
        dict inputs, and ``module(*input)`` for tuple inputs.

    """
    state = _capture_torch_process_state()
    # Everything the golden stores or replays is computed in here, not only the
    # run: an initializer, an unsaved buffer, and the input each draw or compute
    # through vector-ISA-dependent kernels when left native.
    try:
        with host_agnostic_numerics():
            _assert_bfb(
                golden_dir=golden_dir,
                golden_name=golden_name,
                build_module=build_module,
                build_input=build_input,
                seed=seed,
                run=_default_runner if run is None else run,
            )
    finally:
        _restore_torch_process_state(state)


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

    Equivalent to ``assert_bfb_against_golden`` under ``--regenerate-b4b``.
    The freshly written golden is replayed and must round-trip bit-exactly.

    Args:
      golden_dir: Directory for the golden file.
      golden_name: Base name for the golden file.
      build_module: Module factory.
      build_input: Input factory.
      seed: Manual seed.
      run: Optional custom runner.

    """
    with regenerate.forced(), suppress(_MissingGoldenError):
        assert_bfb_against_golden(
            golden_dir=golden_dir,
            golden_name=golden_name,
            build_module=build_module,
            build_input=build_input,
            seed=seed,
            run=run,
        )


class _Golden(TypedDict):
    """What a golden file stores.

    ``post_state`` holds only the tensors the run changed; every other entry's
    expectation is its ``state_dict`` value. It is absent when nothing changed.
    """

    state_dict: dict[str, Tensor]
    input: object
    output: Tensor
    seed: int
    post_state: NotRequired[dict[str, Tensor]]


# Every comparand is stored whole: a mismatch then names the element and its ULP
# distance, and the expected values can be read and diffed. Never a digest -- see the
# bit-for-bit skill. ``torch.save`` frames every tensor as its own ~300-byte storage,
# so each record is packed: one flat tensor per dtype plus an index.
def save_golden(path: Path, payload: _Golden) -> None:
    """Write a golden with each record packed.

    Args:
      path: Destination ``.pt``.
      payload: The golden, records as plain name-to-tensor mappings.

    """
    stored: dict[str, object] = {**payload, "state_dict": pack(payload["state_dict"])}
    if "post_state" in payload:
        stored["post_state"] = pack(payload["post_state"])
    torch.save(stored, path)


def load_golden(path: Path) -> _Golden:
    """Read a golden written by :func:`save_golden`, records unpacked.

    Args:
      path: Source ``.pt``.

    Returns:
      payload: The golden, records as plain name-to-tensor mappings.

    """
    payload = cast(_Golden, torch.load(path, weights_only=False))
    payload["state_dict"] = unpack(payload["state_dict"])
    if "post_state" in payload:
        payload["post_state"] = unpack(payload["post_state"])
    return payload


def stale_post_states(paths: Iterable[Path]) -> list[Path]:
    """Return the bfb goldens whose post-run state repeats an unchanged tensor.

    Replay reads an absent post-run entry as "equal to the pre-run state", so a
    stored copy of an unchanged tensor asserts nothing and only costs bytes.

    Args:
      paths: Candidate ``.pt`` files; ones that are not bfb goldens are skipped.

    Returns:
      stale: The goldens storing at least one unchanged post-run tensor.

    """
    stale: list[Path] = []
    for path in paths:
        loaded = cast(object, torch.load(path, weights_only=False))
        if not isinstance(loaded, dict):
            continue
        typed_loaded = cast(dict[str, object], loaded)
        raw = from_plain(typed_loaded, dict[str, object])
        if "post_state" not in raw or "state_dict" not in raw:
            continue
        payload = load_golden(path)
        if "post_state" not in payload:
            raise ValueError('Expected "post_state" in payload.')
        post = payload["post_state"]
        if len(changed_state(payload["state_dict"], post)) < len(post):
            stale.append(path)
    return stale


def changed_state(
    before: Mapping[str, Tensor],
    after: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Return the entries of ``after`` that are new or differ bitwise from ``before``.

    Args:
      before: State captured before the run.
      after: State captured after it.

    Returns:
      changed: Detached CPU copies of the changed entries, keyed like ``after``.

    """
    return {
        key: value.detach().cpu().clone()
        for key, value in after.items()
        if key not in before or not tensor_bits_equal(before[key], value)
    }


# bfloat16 and float16 are refused too, not float64 alone: the harness computes in all
# three and only the rounding makes a value portable (see the module docstring). Complex
# outputs are unsupported. Integers carry no rounding and pass.
def _assert_portable_output_dtype(output: Tensor) -> None:
    """Refuse a golden comparand that skipped the round back to float32."""
    if output.dtype.is_complex:
        raise TypeError(
            f"bfb golden output is {output.dtype}, which is not supported; "
            "return a float32 or integer tensor.",
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


def _replay_golden[InputT](
    *,
    golden_path: Path,
    build_module: Callable[[], nn.Module],
    build_input: Callable[[], InputT],
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
    payload = load_golden(golden_path)
    # Construction consumes RNG in the same order as minting, even though the
    # runner uses the saved input. Also detect an input builder that has drifted.
    _assert_same_input(_to_cpu(build_input()), payload["input"], label="input")
    module.load_state_dict(payload["state_dict"])
    inp = cast(InputT, move_to_device(payload["input"], device))
    output = run(module, inp)
    # Checked on replay too, not only at mint: a runner changed to return float64
    # after the golden was minted is reported by cause, not as a last-bit
    # mismatch on someone else's host.
    _assert_portable_output_dtype(output)
    _assert_equal(output, payload["output"], label="output")
    # An entry the run did not change is expected to equal its pre-run copy, so a
    # mutation introduced later fails against it.
    expected = {**payload["state_dict"], **payload.get("post_state", {})}
    _assert_portable_state_changes(
        changed_state(payload["state_dict"], module.state_dict()),
    )
    _assert_state_match(module, expected)


# A float64 buffer can hold exact float32 values without retaining any extra arithmetic
# precision. Require a lossless round-trip, including every bit, so existing buffers
# storing these values need no schema change.
def _assert_portable_state_changes(state: Mapping[str, Tensor]) -> None:
    """Reject changed state retaining precision beyond float32 or complex values."""
    for key, value in state.items():
        unrounded = value.dtype == torch.float64 and not tensor_bits_equal(
            value,
            value.float().double(),
        )
        if unrounded or value.dtype.is_complex:
            raise TypeError(
                f"bfb golden state[{key}] is {value.dtype}, which is not portable; "
                "narrow computed state before recording it.",
            )


def _default_runner(module: nn.Module, inp: object) -> Tensor:
    if isinstance(inp, dict):
        result = cast(object, module(**inp))
    elif isinstance(inp, (list, tuple)):
        result = cast(object, module(*inp))
    else:
        result = cast(object, module(inp))
    if not isinstance(result, Tensor):
        raise TypeError("The default runner requires a module that returns a Tensor.")
    return result


def _assert_state_match(module: nn.Module, expected: Mapping[str, Tensor]) -> None:
    live = module.state_dict()
    live_keys = set(live.keys())
    golden_keys = set(expected.keys())
    if live_keys != golden_keys:
        added = live_keys - golden_keys
        removed = golden_keys - live_keys
        raise AssertionError(
            f"state_dict keys differ: added={sorted(added)} removed={sorted(removed)}",
        )
    for k in sorted(live_keys):
        _assert_equal(live[k].detach(), expected[k], label=f"state[{k}]")


def _to_cpu(value: object) -> object:
    """Snapshot every tensor on CPU, sharing storage exactly where the input did."""
    leaves: dict[int, Tensor] = {}
    _ = _map_tensors(value, lambda leaf: leaves.setdefault(id(leaf), leaf))
    snapshots = _compact_copies(leaves.values())
    return _map_tensors(value, lambda leaf: snapshots[id(leaf)])


def _map_tensors(value: object, fn: Callable[[Tensor], Tensor]) -> object:
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, dict):
        typed_value = cast(dict[str, object], value)
        return {k: _map_tensors(v, fn) for k, v in typed_value.items()}
    if isinstance(value, tuple):
        typed_value = cast(tuple[object, ...], value)
        return tuple(_map_tensors(v, fn) for v in typed_value)
    if isinstance(value, list):
        typed_value = cast(list[object], value)
        return [_map_tensors(v, fn) for v in typed_value]
    return value


# Views of one storage keep sharing it, so a runner that mutates one input sees the
# change in the others at replay as it did at mint. Only the span they cover is
# copied: a golden serializes whole storages, so a slice of a large tensor would
# otherwise carry its base into the file.
def _compact_copies(tensors: Iterable[Tensor]) -> dict[int, Tensor]:
    """Copy tensors to CPU, each shared storage cut to the bytes its views use."""
    groups: dict[tuple[torch.device, int], list[tuple[int, Tensor]]] = {}
    for tensor in tensors:
        # A lazy conjugate or negation bit is not in the storage bytes.
        resolved = tensor.detach().resolve_conj().resolve_neg()
        key = (resolved.device, resolved.untyped_storage().data_ptr())
        groups.setdefault(key, []).append((id(tensor), resolved))
    copies: dict[int, Tensor] = {}
    for group in groups.values():
        spans = [_byte_span(member) for _, member in group]
        # Starting on the widest element keeps every member's offset whole.
        width = max(member.element_size() for _, member in group)
        start = min(low for low, _ in spans) // width * width
        end = max(high for _, high in spans)
        source = group[0][1]
        storage = (
            torch.empty(0, dtype=torch.uint8, device=source.device)
            .set_(source.untyped_storage(), start, (end - start,), (1,))
            .to("cpu", copy=True)
            .untyped_storage()
        )
        for index, (key, member) in enumerate(group):
            low, _ = spans[index]
            copies[key] = torch.empty(0, dtype=member.dtype).set_(
                storage,
                (low - start) // member.element_size(),
                member.shape,
                member.stride(),
            )
    return copies


def _byte_span(tensor: Tensor) -> tuple[int, int]:
    """Return the storage bytes ``[start, end)`` holding a tensor's elements."""
    start = int(tensor.storage_offset()) * tensor.element_size()
    if not tensor.numel():
        return start, start
    last = sum(
        (tensor.shape[index] - 1) * tensor.stride()[index]
        for index in range(tensor.ndim)
    )
    return start, start + (last + 1) * tensor.element_size()


def _cpu_state_dict(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _assert_same_input(live: object, stored: object, *, label: str) -> None:
    """Assert a rebuilt input equals the recorded one, recursing into containers."""
    live_type: type = type(live)
    stored_type: type = type(stored)
    if live_type != stored_type:
        raise AssertionError(f"{label}: type mismatch {live_type} vs {stored_type}")
    if live_type is dict:
        live_map = cast(dict[str, object], live)
        stored_map = cast(dict[str, object], stored)
        if live_map.keys() != stored_map.keys():
            raise AssertionError(
                f"{label}: keys differ {sorted(live_map)} vs {sorted(stored_map)}",
            )
        for key, value in live_map.items():
            _assert_same_input(value, stored_map[key], label=f"{label}[{key!r}]")
    elif live_type in {list, tuple}:
        live_seq = cast("Sequence[object]", live)
        stored_seq = cast("Sequence[object]", stored)
        if len(live_seq) != len(stored_seq):
            raise AssertionError(
                f"{label}: length {len(live_seq)} vs {len(stored_seq)}",
            )
        for index in range(len(live_seq)):
            _assert_same_input(
                live_seq[index],
                stored_seq[index],
                label=f"{label}[{index}]",
            )
    else:
        _assert_equal(live, stored, label=label)


def _ints(values: object) -> list[int]:
    if not isinstance(values, (list, tuple)):
        return []
    typed_values = cast(list[object] | tuple[object, ...], values)
    return [
        from_plain(value, int)
        for value in typed_values
        if isinstance(value, int) and not isinstance(value, bool)
    ]


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
    if not tensor_bits_equal(a, b):
        if a.dtype.is_floating_point or a.dtype.is_complex:
            max_abs_diff = f"{float((a - b).abs().max().item()):.3e}"
        elif a.dtype == torch.bool:
            max_abs_diff = "1"
        else:
            # Python ints: the int64 extremes differ by 2**64 - 1, which a
            # tensor subtraction would overflow.
            values_a = _ints(a.detach().cpu().reshape(-1).tolist())
            values_b = _ints(b.detach().cpu().reshape(-1).tolist())
            max_abs_diff = str(
                max(
                    abs(values_a[index] - values_b[index])
                    for index in range(len(values_a))
                ),
            )
        raise AssertionError(
            f"{label}: bitwise comparison failed "
            f"(max_abs_diff={max_abs_diff}, max_ulp_diff={_max_ulp_diff(a, b)})",
        )


# The unit a bit-for-bit failure is actually measured in: 1 says the hosts round
# differently, a large count says the computation changed, and an absolute difference
# says neither on its own (1 ULP is 1e-7 near one and 1e-45 near zero).
#
# Bit patterns are ordered only WITHIN a sign -- the negative half is stored sign-
# magnitude, counting away from zero -- so they are mapped to one monotone line first.
# Subtracting raw patterns instead reports ~2**31 for two neighbours straddling zero,
# which is the magnitude a total regression produces, from the case where the values are
# closest.
def _max_ulp_diff(a: Tensor, b: Tensor) -> int | str:
    """Largest gap in representable steps, or why it could not be measured."""
    kind = {
        torch.float64: torch.int64,
        torch.float32: torch.int32,
        torch.float16: torch.int16,
        torch.bfloat16: torch.int16,
    }.get(a.dtype)
    if kind is None:
        return "n/a"
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    # NaN has no distance to anything: every comparison against it is false, so
    # a pattern subtraction returns a number that reads as real drift.
    if bool(a.isnan().any() or b.isnan().any()):
        return "nan"
    if kind == torch.int64:
        # Distances across signs can exceed signed int64, even though each
        # ordered endpoint fits. Widen the subtraction to Python integers.
        values_a = _ints(_ordered(a, kind).reshape(-1).tolist())
        values_b = _ints(_ordered(b, kind).reshape(-1).tolist())
        return max(
            (abs(values_a[index] - values_b[index]) for index in range(len(values_a))),
            default=0,
        )
    return int((_ordered(a, kind) - _ordered(b, kind)).abs().max())


# A negative float's pattern grows as the number falls, so the negative half is
# reflected. The result orders the whole line, which is what makes a subtraction count
# representable steps.
#
# Reflected about the float's OWN signed minimum, not int64's: the pattern is widened
# for the arithmetic, and reflecting about the wide minimum would offset the negative
# half by the difference between the two widths.
def _ordered(value: Tensor, kind: torch.dtype) -> Tensor:
    """Reinterpret floats as integers that increase with the float's value."""
    bits = value.detach().contiguous().view(kind)
    floor = torch.iinfo(bits.dtype).min
    wide = bits.to(torch.int64)
    return torch.where(wide < 0, floor - wide, wide)


class _MissingGoldenError(AssertionError):
    """A missing BFB golden was minted and requires review."""


# Named for the upcast's question -- "is this narrower than the scratch width" -- so
# float64 is False here because it IS the scratch, not because it is host-independent
# (it is not; see the module docstring). A caller asking whether a dtype is portable
# wants :func:`_assert_portable_output_dtype`, which admits float32 alone.
#
# bfloat16 and float16 count because a mixed-precision recipe COMPUTES in them -- an
# autocast forward, or an optimizer that orthogonalizes in half precision -- so a golden
# that left them native would be minted to one machine.
def _is_narrow_float(dtype: torch.dtype) -> bool:
    """Whether a dtype should be widened before the op runs."""
    return dtype.is_floating_point and dtype != torch.float64


# ``_foreach_*`` ops (e.g. ``_foreach_norm`` behind ``clip_grad_norm_``) receive a
# ``list[Tensor]`` rather than a bare tensor, so a direct ``isinstance(a, Tensor)``
# check misses them and the upcast silently does not apply.
def _floating_tensors(value: object) -> list[Tensor]:
    """Return the floating tensors appearing in a tensor / list / tuple."""
    if isinstance(value, Tensor):
        return [value] if value.dtype.is_floating_point else []
    if isinstance(value, (list, tuple)):
        found: list[Tensor] = []
        for item in cast(list[object] | tuple[object, ...], value):
            found.extend(_floating_tensors(item))
        return found
    return []


# Torch promotes mixed inputs, so a bfloat16 tensor meeting a float32 one yields
# float32. Reproducing that promotion here is what lets the result be narrowed back to
# the width the unwrapped computation would have held -- narrowing everything to float32
# instead would silently widen a half precision graph and change every value downstream
# of it.
def _result_dtype(tensors: Sequence[Tensor]) -> torch.dtype:
    """Return the dtype the op would have produced natively."""
    # A zero-dimensional float64 tensor does not widen a float32 vector.
    # Dtype-only promotion loses that distinction.
    return elementwise_dtypes(
        *tensors,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )[1]


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
    shared_tensors: list[Tensor] = []
    for value in (*args, *kwargs.values()):
        if isinstance(value, list):
            sequence = cast(list[object], value)
        elif isinstance(value, tuple):
            sequence = cast(tuple[object, ...], value)
        else:
            shared_tensors.extend(_floating_tensors(value))
            continue
        if len(sequence) == len(typed_result):
            sequences.append(sequence)
    downcast = list[object]()
    for index, value in enumerate(typed_result):
        tensors = list(shared_tensors)
        for sequence in sequences:
            tensors.extend(_floating_tensors(sequence[index]))
        target = _result_dtype(tensors) if tensors else fallback
        downcast.append(_downcast_f64(value, target))
    return downcast


# Recurses into lists/tuples for ``_foreach_*`` write targets. A tensor that was never
# upcast -- one whose dtype is not a narrow float -- was written by the op itself, so it
# is skipped. The narrowing ``copy_`` is IEEE-correctly-rounded, hence host-independent.
def _copy_back(original: object, computed: object) -> None:
    """Narrow ``computed`` (float64) into ``original`` in place."""
    if isinstance(original, Tensor):
        if _is_narrow_float(original.dtype) and isinstance(computed, Tensor):
            if original.shape != computed.shape:
                original.resize_(computed.shape)
            original.copy_(computed)
        return
    if isinstance(original, (list, tuple)) and isinstance(computed, (list, tuple)):
        for o, c in zip(
            cast(list[object] | tuple[object, ...], original),
            cast(list[object] | tuple[object, ...], computed),
            strict=True,
        ):
            _copy_back(o, c)


# The op ran on the float64 upcast copies in ``up_args``/``up_kwargs``, so its computed
# values live there, not in the caller's float32 originals. For every write argument
# (``alias_info.is_write``), narrow its upcast copy back into the original
# (``_copy_back``, recursing through foreach ``Tensor[]``) as the side effect, and
# remember the (upcast-copy, original) pair.
#
# The return is then rebuilt element-wise: a returned element that IS one of the upcast
# write copies is swapped to the caller's original (preserving in-place / ``out=``
# return identity); every other element -- a freshly-computed output that merely happens
# to ride alongside the writes, e.g. ``_native_batch_norm_legit`` returns ``(output,
# save_mean, save_invstd)`` while writing ``running_*`` -- is downcast and kept, never
# dropped. ``None`` (void in-place, e.g. ``_foreach_*_``) passes through, which the
# dispatcher requires.
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
    """Restore an in-place / ``out=`` / foreach op's mutation onto the originals."""
    schema = func._schema  # noqa: SLF001 -- The harness reads the op schema to find write arguments.
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

    if result is None:
        return None
    if isinstance(result, tuple):
        return tuple(
            _resolve_output(e, upcast_to_original, target)
            for e in cast(tuple[object, ...], result)
        )
    if isinstance(result, list):
        return [
            _resolve_output(e, upcast_to_original, target)
            for e in cast(list[object], result)
        ]
    return _resolve_output(result, upcast_to_original, target)


def _resolve_output(
    element: object,
    upcast_to_original: Sequence[tuple[object, object]],
    target: torch.dtype,
) -> object:
    """Map an in-place output back to its original tensor, else downcast it."""
    for computed, original in upcast_to_original:
        if element is computed:
            return original
    return _downcast_f64(element, target)


def _op_name(func: OpOverload[..., object]) -> str:
    """Return an Aten overload's packet name."""
    return func.name().split("::")[-1].split(".")[0]


# A view moves no bytes, so it is exact, and its result must alias its input: upcast,
# it returns a float64 copy narrowed back, and every write through it lands on that
# copy. ``x[:, :n]`` over a whole dimension issues ``alias``, not ``slice``, so a
# rollout of one buffer stored each step into a copy and kept zeros (measured). The
# schema names every view, the allowlist's and the 13 it lacked (``alias``,
# ``diagonal``, ``unfold``, ``real``, ...), so no future view op can fall through.
def _is_view(func: OpOverload[..., object]) -> bool:
    """Whether every return aliases an input without writing it."""
    returns = func._schema.returns  # noqa: SLF001 -- The harness reads the op schema to find views.
    return bool(returns) and all(
        value.alias_info is not None and not value.alias_info.is_write
        for value in returns
    )


# ``add``/``sub`` with ``alpha != 1`` compute ``a + alpha * b``: vectorized kernels
# fuse it into one FMA rounding, the scalar kernel rounds twice, so the float32
# result depends on the host's vector ISA. Measured: 271-367 of 4096 differ between
# ATEN_CPU_CAPABILITY=default and avx2/avx512, enough to fork Adam's first moment.
def _scales_operand(
    func: OpOverload[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> bool:
    """Whether an allowlisted op carries a non-unit ``alpha`` multiplier."""
    for index, argument in enumerate(func._schema.arguments):  # noqa: SLF001 -- The harness reads the op schema to find the multiplier argument.
        if argument.name != "alpha":
            continue
        alpha = kwargs.get("alpha", args[index] if index < len(args) else 1)
        return alpha != 1
    return False


# Upcasting is not enough for a fused multiply-add: its float64 kernel is itself
# ISA-dependent (vectorized FMA vs. the scalar kernel's two roundings), and under
# cancellation the gap reaches ~1e5 float64 ULP -- measured on lerp, addcmul,
# addcdiv, scaled add, layer_norm, and the softmax backwards -- which flips the
# float32 round. Their decompositions issue separate mul/add/sub, each correctly
# rounded on every host.
_UNFUSED_OPS: Final = frozenset(
    {
        "add",
        "add_",
        "sub",
        "sub_",
        "lerp",
        "lerp_",
        "addcmul",
        "addcmul_",
        "addcdiv",
        "addcdiv_",
        "addmm",
        "baddbmm",
        "addmv",
        "_addmm_activation",
        "native_layer_norm",
        "native_layer_norm_backward",
        "_log_softmax_backward_data",
        "_softmax_backward_data",
    },
)


# With a contraction of one, x86's matrix kernel stores the product itself, so
# ``-1.0 * 0.0`` stays ``-0.0``; aarch64's adds it to a zero accumulator and stores
# ``+0.0``. A single row or column takes x86's vector kernel, which also adds to
# ``+0.0``, so those shapes run the kernel as usual.
def _one_term_mm(
    func: OpOverload[..., object],
    args: tuple[object, ...],
) -> Tensor | None:
    """Return a one-term ``mm`` as its plain product, or ``None`` for any other op."""
    if func.namespace != "aten" or _op_name(func) != "mm" or len(args) != 2:
        return None
    left, right = args
    if not isinstance(left, Tensor) or not isinstance(right, Tensor):
        return None
    if left.shape[1] != 1 or left.shape[0] < 2 or right.shape[1] < 2:
        return None
    return left * right


def _run_unfused(
    func: OpOverload[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Run ``func``, through its decomposition when its kernel may fuse an FMA."""
    if func.namespace != "aten":
        return func(*args, **kwargs)
    name = _op_name(func)
    base = name.removesuffix("_")
    if base in {"addmm", "baddbmm", "addbmm", "addmv", "_addmm_activation"}:
        matrix_names = (
            ("mat", "vec")
            if base == "addmv"
            else ("batch1", "batch2")
            if base in {"addbmm", "baddbmm"}
            else ("mat1", "mat2")
        )
        operands = [
            cast(Tensor, args[index] if index < len(args) else kwargs[argument])
            for index, argument in enumerate(("self", *matrix_names))
        ]
        bias, left, right = operands
        dimensions = 3 if base in {"addbmm", "baddbmm"} else 2
        right_dimensions = 1 if base == "addmv" else dimensions
        # The unfused path skips the native dtype check, so mixed operands -- an
        # integer matrix under a widened float bias -- go to the kernel that
        # rejects them, as do integers, whose native arithmetic is exact.
        if (
            any(value.dtype != bias.dtype for value in operands)
            or not bias.dtype.is_floating_point
            or left.ndim != dimensions
            or right.ndim != right_dimensions
        ):
            return func(*args, **kwargs)
        shape = (left.shape[-2],)
        if base != "addmv":
            shape += (right.shape[-1],)
        if base == "baddbmm":
            shape = (left.shape[0], *shape)
        # Bias may expand to the product, never expand the product itself.
        # This validation remains necessary when beta=0 ignores bias values.
        _ = bias.expand(shape)
    if name != base and base in {"addmm", "baddbmm", "addbmm", "addmv"}:
        functional = cast(
            "OpOverload[..., object]",
            getattr(torch.ops.aten, base).default,
        )
        result = _run_unfused(functional, args, kwargs)
        original = cast(Tensor, args[0] if args else kwargs["self"])
        return original.copy_(cast(Tensor, result))
    if base == "addbmm":
        return cast("Callable[..., Tensor]", _unfused_addbmm)(*args, **kwargs)
    if base in {"convolution", "_convolution"}:
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        if isinstance(bias, Tensor):
            return _unfused_convolution(func, args, kwargs, bias=bias)
    if _op_name(func) in _UNFUSED_OPS and _scales_or_fuses(func, args, kwargs):
        return decomposition_table[func](*args, **kwargs)
    return func(*args, **kwargs)


def _unfused_convolution(
    func: OpOverload[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    bias: Tensor,
) -> Tensor:
    """Add bias after the native float64 dot product, before narrowing."""
    # Native CPU convolutions initialize GEMM's accumulator with bias. Under
    # cancellation, x86 can discard that bias while ARM adds it after the dot
    # product. Upcasting alone cannot restore it. Keep these two steps separate.
    if len(args) > 2:
        result = func(*args[:2], None, *args[3:], **kwargs)
    else:
        result = func(*args, **{**kwargs, "bias": None})
    assert isinstance(result, Tensor)
    # Removing the native bias also removes its shape validation; a singleton
    # or flattened bias would otherwise be silently accepted by broadcasting.
    if bias.ndim != 1 or bias.shape[0] != result.shape[1]:
        raise RuntimeError(
            "convolution bias must be one-dimensional with one value per output channel",
        )
    return result.add_(bias.reshape(-1, *((1,) * (result.ndim - 2))))


@out_wrapper(exact_dtype=True)
def _unfused_addbmm(
    self: Tensor,
    batch1: Tensor,
    batch2: Tensor,
    beta: float = 1,
    alpha: float = 1,
) -> Tensor:
    """Reduce completed batch products before scaling and adding the bias."""
    result = alpha * torch.bmm(batch1, batch2).sum(dim=0)
    return result if beta == 0 else result + beta * self


def _scales_or_fuses(
    func: OpOverload[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> bool:
    """Whether ``func`` multiplies inside its kernel (plain add/sub do not)."""
    if _op_name(func).rstrip("_") in {"add", "sub"}:
        return _scales_operand(func, args, kwargs)
    return func in decomposition_table


class _Float64Compute(TorchDispatchMode):
    """Compute every float32 arithmetic op in float64, return float32.

    Operates at the aten-dispatch layer, so it sees the aten ops the computation
    issues during forward and autograd backward. An op is upcast when it has a
    float32 argument and its overloadpacket name is NOT
    in ``_EXACT_F32_OPS``; the float32 args are widened to float64, the op runs,
    and float64 results are narrowed back to float32. Allowlisted ops (exact
    elementwise arithmetic and pure data movement) pass through untouched, as
    does every view op by its schema (``_is_view``), so a write through a view
    reaches its base. A tensorless arithmetic factory in ``_FLOAT_FACTORIES`` is
    widened by its output dtype, since it has no argument to read the width from.

    Upcast-by-default is the completeness guarantee: a transcendental or
    reduction absent from every list is still upcast, so it cannot silently mint
    a host-dependent golden. The only unsafe act is wrongly *adding* an op to
    ``_EXACT_F32_OPS``, which the guard test in ``bfb_test.py`` catches.

    Distributed collective ops (``_COLLECTIVE_NAMESPACES``) run natively,
    never upcast: their schema declares no ``alias_info`` for the mutated
    tensor, so an upcast-then-copy-back would have no write target and the
    collective would silently become a no-op. Every other non-aten op --
    including a project's own ``torch.library`` custom op -- keeps the
    upcast.
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
        if func.namespace in _COLLECTIVE_NAMESPACES:
            return func(*args, **kwargs)
        exact = _is_view(func) or (
            func.namespace == "aten"
            and _op_name(func) in _EXACT_F32_OPS
            and not _scales_operand(
                func,
                args,
                kwargs,
            )
        )
        if exact:
            return func(*args, **kwargs)
        input_tensors: list[Tensor] = []
        for value in (*args, *kwargs.values()):
            input_tensors.extend(_floating_tensors(value))
        explicit_dtype = kwargs.get("dtype")
        # A tensorless sampler's width is its dtype kwarg or the process default;
        # see the note above ``_EXACT_F32_OPS``.
        factory = (
            func.namespace == "aten"
            and _op_name(func) in _FLOAT_FACTORIES
            and not input_tensors
        )
        factory_dtype = (
            torch.int64
            if _op_name(func) == "arange"
            and not any(isinstance(value, float) for value in (*args, *kwargs.values()))
            else torch.get_default_dtype()
        )
        target = (
            explicit_dtype
            if isinstance(explicit_dtype, torch.dtype)
            else _result_dtype(input_tensors)
            if input_tensors
            else factory_dtype
        )
        narrow = any(_is_narrow_float(value.dtype) for value in input_tensors) or (
            factory and _is_narrow_float(target)
        )
        if not narrow:
            return _run_unfused(func, args, kwargs)
        up_args = tuple(_upcast(a) for a in args)
        up_kwargs = {k: _upcast(v) for k, v in kwargs.items()}
        if factory or (
            isinstance(explicit_dtype, torch.dtype) and _is_narrow_float(explicit_dtype)
        ):
            up_kwargs["dtype"] = torch.float64
        result = _one_term_mm(func, args=up_args)
        if result is None:
            result = _run_unfused(func, up_args, up_kwargs)
        if any(
            arg.alias_info is not None and arg.alias_info.is_write
            for arg in func._schema.arguments  # noqa: SLF001 -- The harness reads the op schema to find write arguments.
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


def _module_device(module: nn.Module) -> str:
    """Return the module's tensor device, defaulting tensorless modules to CPU."""
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.device.type
    buffer = next(module.buffers(), None)
    return buffer.device.type if buffer is not None else "cpu"


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
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.set_rng_state(generator.get_state())


def _write_golden[InputT](
    *,
    golden_path: Path,
    build_module: Callable[[], nn.Module],
    build_input: Callable[[], InputT],
    seed: int,
    run: Callable[[nn.Module, InputT], Tensor],
) -> None:
    """Build, randomize, run; store the pre-run state, input, output, and changes."""
    torch.use_deterministic_algorithms(True)
    _seed_bfb(seed)
    module = build_module()
    device = _module_device(module)
    if device != "cpu":
        raise ValueError("The BFB harness is CPU-only.")
    inp = build_input()
    randomize_parameters(module, seed=seed)
    pre_state = _cpu_state_dict(module.state_dict())
    pre_input = _to_cpu(inp)
    output = run(module, inp)
    _assert_portable_output_dtype(output)
    payload: _Golden = {
        "state_dict": pre_state,
        "input": pre_input,
        "output": output.detach().to("cpu", copy=True),
        "seed": seed,
    }
    # Unchanged entries are asserted against the pre-run copy, so storing only
    # the changed ones is not a weaker check.
    if post_state := changed_state(pre_state, module.state_dict()):
        _assert_portable_state_changes(post_state)
        payload["post_state"] = post_state
    save_golden(golden_path, payload)


def _assert_bfb[InputT](
    *,
    golden_dir: Path,
    golden_name: str,
    build_module: Callable[[], nn.Module],
    build_input: Callable[[], InputT],
    seed: int,
    run: Callable[[nn.Module, InputT], Tensor],
) -> None:
    """Mint or replay a golden; the caller holds ``host_agnostic_numerics``."""
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_path = golden_dir / f"{golden_name}.pt"
    missing = not golden_path.exists()
    if missing or regenerate.b4b():
        # The candidate belongs to the destination filesystem and is cleaned
        # up on every exit. Publish only after the complete replay succeeds.
        with tempfile.TemporaryDirectory(
            dir=golden_dir,
            prefix=golden_path.name,
        ) as candidate_dir:
            candidate_path = Path(candidate_dir) / golden_path.name
            _write_golden(
                golden_path=candidate_path,
                build_module=build_module,
                build_input=build_input,
                seed=seed,
                run=run,
            )
            _replay_golden(
                golden_path=candidate_path,
                build_module=build_module,
                build_input=build_input,
                seed=seed,
                run=run,
            )
            candidate_path.replace(golden_path)
        if missing:
            raise _MissingGoldenError(
                f"Missing golden regenerated at {golden_path}; inspect it, "
                "then rerun the test.",
            )
        return

    _replay_golden(
        golden_path=golden_path,
        build_module=build_module,
        build_input=build_input,
        seed=seed,
        run=run,
    )
