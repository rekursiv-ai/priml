"""Buffer acquisition and aliasing for tensors and arrays.

Two questions about where a buffer lives, kept together because they are
the same question from opposite ends: ``convert_to_tensor`` decides whether
to WRAP the caller's memory or allocate fresh, and ``shares_storage``
detects which of those happened. Splitting them once left the aliasing
check unable to explain the conversion that produced its input.

``shares_storage`` answers for ANY pair. ``is_private_conversion`` answers
only for a pair ``convert_to_tensor`` just produced, trading generality for
a single address compare that stays traceable under ``torch.compile``.
Handed anything else -- a view and its parent, say -- it reports "private"
and a caller acting on that overwrites live data, so read its ``Warning``
before reaching for it.

Scope is buffer identity only -- not device transfer, allocator policy, or
pinning. Those have their own concerns and do not belong here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, overload

from torch import Tensor

import numpy as np
import torch


if TYPE_CHECKING:
    # Deferred to break a cycle: ``custom_types`` re-exports
    # ``convert_to_tensor`` from here, so importing the alias at runtime would
    # have each module waiting on the other. It is only ever an annotation.
    from priml.math.custom_types import Tensorable


__all__ = [
    "convert_to_tensor",
    "is_private_conversion",
    "shares_storage",
    "shares_storage_for_compile",
]


@overload
def convert_to_tensor(
    __x: Tensorable,
    /,
    *,
    dtype: torch.dtype | None = ...,
    device: torch.device | str | None = ...,
    dtype_hint: torch.dtype | None = ...,
) -> Tensor: ...


@overload
def convert_to_tensor(
    __x: Tensorable,
    __y: Tensorable,
    /,
    *xs: Tensorable,
    dtype: torch.dtype | None = ...,
    device: torch.device | str | None = ...,
    dtype_hint: torch.dtype | None = ...,
) -> tuple[Tensor, ...]: ...


def convert_to_tensor(
    *xs: Tensorable,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    dtype_hint: torch.dtype | None = None,
) -> Tensor | tuple[Tensor, ...]:
    """Convert inputs to tensors with unified dtype.

    Promotes all inputs to the highest-precedence dtype found
    among them (e.g. float64 > float32 > int64). When no input
    has an explicit dtype, torch assigns defaults then unifies.

    Returns a single Tensor when called with one argument, or a
    tuple of Tensors when called with multiple arguments.

    Args:
      xs: Values convertible to tensors.
      dtype: Force all outputs to this dtype.
      device: Place all outputs on this device.
      dtype_hint: Fallback dtype when no input carries one.

    """
    if torch.compiler.is_compiling():
        # Only the dtype default differs from the eager path below: under
        # compile the inputs cannot be re-inspected to unify bare scalars, so
        # the hint stands in. ``torch.as_tensor`` handles both kinds -- it
        # returns the identical object when a Tensor already matches, and
        # converts a Python number otherwise. Asserting Tensor here instead
        # aborted ``softcap``, ``ste_round``, ``soft_threshold``,
        # ``sinh_arcsinh`` and ``safe_pow`` under ``fullgraph=True``, since
        # each passes a float as its second argument.
        if dtype is None:
            dtype = _resolve_dtype(xs) or dtype_hint
        result = tuple(torch.as_tensor(x, dtype=dtype, device=device) for x in xs)
        return result[0] if len(xs) == 1 else result
    if dtype is None:
        dtype = _resolve_dtype(xs)
        if dtype is None and dtype_hint is None:
            # Bare scalars: let torch assign defaults, then unify.
            xs = tuple(torch.as_tensor(x, device=device) for x in xs)
            dtype = _resolve_dtype(xs)
        elif dtype is None:
            dtype = dtype_hint
    result = tuple(torch.as_tensor(x, dtype=dtype, device=device) for x in xs)
    return result[0] if len(xs) == 1 else result


def is_private_conversion(converted: Tensor, original: Tensorable) -> bool:
    """Whether ``converted`` can be overwritten without ``original`` seeing it.

    Narrow by design: correct ONLY for a pair :func:`convert_to_tensor` just
    produced, where the result is either ``original`` itself or a fresh
    allocation -- never a view of it. That invariant is what lets a single
    address comparison stand in for real aliasing analysis.

    Args:
      converted: What ``convert_to_tensor`` returned.
      original: What it was given.

    Returns:
      private: True when nothing the caller retains shares this buffer.

    Warning:
      Wrong for a pair that did NOT come from a conversion, and wrong in the
      direction that corrupts: given a slice and its parent it answers True,
      because a view reports a different ``data_ptr`` than the storage it
      shares, and a caller acting on that answer overwrites live data. There
      is no cheap runtime check for "is a view of", so nothing here can
      enforce the invariant -- the CALL SITE has to guarantee it. Reach for
      :func:`shares_storage` whenever the pair is arbitrary.

    Prefer this over :func:`shares_storage` at a conversion site, for two
    reasons that both follow from the invariant:

      - It is TRACEABLE. ``shares_storage`` walks strides and calls
        ``untyped_storage()``, which Dynamo rejects outright, so a caller
        inside a compiled region needs a ``torch.compiler.is_compiling()``
        branch -- and that branch makes eager and compiled take different
        paths through the very check that decides whether to mutate.
      - It is cheaper: one ``getattr`` and one integer compare, against two
        span walks over shape and strides.

    """
    if converted is original:
        return False
    # ``ndarray.ctypes.data`` is the array's buffer address, documented as
    # exactly ``__array_interface__['data'][0]``. Anything with no buffer -- a
    # list, a scalar -- misses the attribute and is private by construction,
    # since ``as_tensor`` had to allocate for it. Re-running ``as_tensor`` to
    # get an address to compare would rebuild the data (4.7ms per 100k list).
    #
    # ``None`` as the sentinel, not ``object()``: dynamo cannot trace a call
    # to ``object``, and would fail the whole graph rather than break out of
    # it. An address is never None, so the comparison is unchanged.
    address = getattr(getattr(original, "ctypes", None), "data", None)
    return address is None or converted.data_ptr() != address


def shares_storage(x: Tensorable, y: Tensorable) -> bool:
    """Whether ``x`` and ``y`` are backed by the same memory.

    Symmetric, and either argument may be a Tensor, an ndarray, or something
    with no buffer at all. The common use is deciding whether an in-place
    write is safe: a caller that guesses wrong either wastes an allocation or
    silently corrupts data someone else still holds.

    Args:
      x: First value.
      y: Second value.

    Returns:
      shared: True when writing through one can be observed through the other.

    Raises:
      NotImplementedError: If both values are strided such that their spans
        overlap but no byte need be common -- ``p[::2]`` against ``p[1::2]``.
        Answering that exactly is NP-complete, so neither True nor False can
        be returned honestly and the caller is told rather than guessed at.

    Compares the BYTE RANGE each value occupies. Addresses are globally
    unique, so two live values overlap iff they are in one allocation and
    their spans intersect -- ``p[0:2]`` and ``p[4:6]`` share a buffer but no
    byte, and are correctly reported unshared. A range is built from the
    element pointer plus the strides, so it holds for views, transposes and
    negative strides alike; comparing base addresses alone would call every
    slice of one tensor shared, and comparing element pointers alone would
    call a view and its own parent unshared.

    A value with no buffer -- a list, a scalar, an empty or meta tensor --
    falls back to object identity, which is the only way such a value can
    alias: two names for one list do see each other's writes.

    Not traceable: Dynamo models a tensor by shape and dtype rather than by
    location, so ``data_ptr()`` becomes a ``DataPtrVariable`` that refuses
    ``+`` and ``<``, and ``untyped_storage()`` is rejected outright. Call this
    OUTSIDE the compiled region; inside one, use
    :func:`shares_storage_for_compile`.

    Alternatives considered:
      ``Tensor._base``: ``None`` both for a zero-copy ``as_tensor(ndarray)``
        and for a tensor built from a list, so it cannot separate them.
      ``np.shares_memory``: decides the interleaved case this raises on, but
        needs a ``.numpy()`` round-trip that fails on bfloat16, grad-requiring
        and non-CPU tensors, and can hang.

    References:
      https://stackoverflow.com/questions/60587536
        Comparing ``a.ctypes.data`` against ``b.data_ptr()`` is "a completely
        valid way to access and compare the pointers. The array interface is
        designed to allow sharing data buffers."
      https://stackoverflow.com/questions/66783542
        ``data_ptr()`` "returns the pointer to the first element of the
        tensor, whereas [storage] seems to point to the memory address of the
        underlying data (not the sliced view)".
      https://numpy.org/doc/stable/reference/generated/numpy.shares_memory.html
        "NP-complete, and runtime may increase exponentially."
      https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html
        "If you have code that torch.compile has trouble tracing through ...
        you can consider wrapping the problematic code in a custom op."

    """
    x_start, x_end, x_device = _byte_span(x)
    y_start, y_end, y_device = _byte_span(y)
    if x_device != y_device or x_start >= y_end or y_start >= x_end:
        return False
    # Equal first bytes settle it whatever the strides: each value occupies
    # its own start, so that byte is common. Without this the identical pair
    # ``p[::2]`` against ``p[::2]`` would raise as undecidable.
    if x_start == y_start:
        return True
    # The ranges overlap, which settles it unless BOTH sides leave gaps: a
    # value covering every byte between its ends must really touch the other.
    # Two interleaved runs need not share one byte, and deciding that is the
    # NP-complete problem ``np.shares_memory`` solves.
    if _is_dense(x) or _is_dense(y):
        return True
    raise NotImplementedError(
        "shares_storage cannot decide two interleaved strided values: their "
        "byte ranges overlap but they may touch no common element. Deciding "
        "this exactly is NP-complete; compare the dense parents instead, or "
        "use np.shares_memory on CPU arrays.",
    )


@torch.library.custom_op("priml::shares_storage", mutates_args=())
def shares_storage_for_compile(x: Tensor, y: Tensor) -> Tensor:
    """:func:`shares_storage` as a 0-dim bool tensor, for a compiled region.

    Two functions rather than one because the return types genuinely differ
    and no single signature covers both: an ``@overload`` dispatches on
    PARAMETERS, and both spellings take the same ``(Tensor, Tensor)``. Eager
    callers are the common case and should not pay ``bool(...)`` at every
    site, so :func:`shares_storage` keeps the plain answer.

    Args:
      x: First tensor.
      y: Second tensor.

    Returns:
      shared: 0-dim bool tensor, True when the two overlap in memory.

    Raises:
      NotImplementedError: Same interleaved-stride case as
        :func:`shares_storage`. It fires while tracing, since the op body runs
        eagerly on the real tensors.

    Branch on the result with ``torch.cond``. Reading it with ``bool`` or a
    plain ``if`` is a data-dependent jump, which breaks the graph and forfeits
    the only reason to reach for this instead of :func:`shares_storage`.

    """
    # The op body runs eagerly on the real tensors even while tracing, so it
    # can just call the plain function. On the input's device: a custom op
    # returning a tensor elsewhere fails its own schema check once inductor
    # places the result.
    return torch.tensor(
        shares_storage(x, y),
        dtype=torch.bool,
        device=x.device,
    )


@shares_storage_for_compile.register_fake
def _(x: Tensor, y: Tensor) -> Tensor:
    # Tracing sees only shape, dtype and device, so this value is never read;
    # it exists so dynamo can build the graph without running the real op.
    del y
    return torch.empty((), dtype=torch.bool, device=x.device)


def _is_dense(x: Tensorable) -> bool:
    """Whether ``x`` occupies every byte of its span, rather than skipping."""
    start, end, _ = _byte_span(x)
    if isinstance(x, Tensor):
        return x.numel() * x.element_size() == end - start
    if isinstance(x, np.ndarray):
        return x.nbytes == end - start
    # A bufferless value spans one synthetic byte and fills it.
    return True


def _byte_span(x: Tensorable) -> tuple[int, int, torch.device | None]:
    """Half-open byte range ``x`` occupies, and the device holding it.

    The device rides along because each one allocates from its own pointers:
    a CUDA pointer and a CPU pointer can be the same integer while naming
    unrelated memory.
    """
    if isinstance(x, Tensor):
        # Zero-size and meta tensors report address 0, which would put every
        # one of them at the same place; fall through to identity instead.
        if x.numel() and x.device.type != "meta":
            low, high = _span_from_strides(
                x.data_ptr(),
                x.shape,
                x.stride(),
                stride_scale=x.element_size(),
                item_size=x.element_size(),
            )
            return (low, high, x.device)
    elif isinstance(x, np.ndarray) and x.size:
        # ``ndarray.strides`` is already in BYTES, unlike torch's element
        # counts, so the stride scale is 1 while the trailing element is still
        # ``itemsize`` wide. An array is always host memory, spelled as a
        # device so a CPU tensor wrapping it matches.
        low, high = _span_from_strides(
            x.ctypes.data,
            x.shape,
            x.strides,
            stride_scale=1,
            item_size=x.itemsize,
        )
        return (low, high, torch.device("cpu"))
    # No buffer: a list, a scalar, an empty. Such a value aliases only itself,
    # so a 1-byte span at ``id`` under no device reduces overlap to identity.
    return (id(x), id(x) + 1, None)


def _span_from_strides(
    pointer: int,
    shape: Sequence[int],
    strides: Sequence[int],
    *,
    stride_scale: int,
    item_size: int,
) -> tuple[int, int]:
    """Bounds of a strided layout, walking each axis from its first element.

    ``stride_scale`` converts a stride to bytes -- the element width for
    torch, 1 for numpy, whose strides are already bytes. It is separate from
    ``item_size`` because the trailing element is ``item_size`` wide in both
    cases; folding the two together shortened every numpy span by
    ``itemsize - 1`` bytes.
    """
    low = high = pointer
    for size, stride in zip(shape, strides, strict=True):
        # A negative stride runs backwards from the element pointer, so it
        # extends the low bound rather than the high one.
        #
        # Rebind rather than ``+=``: dynamo cannot trace ``iadd`` on a value
        # derived from ``data_ptr()``, and fails the whole graph rather than
        # breaking out of it.
        reach = (size - 1) * stride * stride_scale
        if reach < 0:
            low = low + reach
        else:
            high = high + reach
    return (low, high + item_size)


def _resolve_dtype(xs: tuple[Tensorable, ...]) -> torch.dtype | None:
    # Annotated rather than ``set[torch.dtype]()``: the subscript is evaluated
    # at runtime, and dynamo cannot trace it ("type object 'set' has no
    # attribute '__getitem__'"), so any compiled caller that reaches here dies.
    seen: set[torch.dtype] = set()
    for x in xs:
        dt = getattr(x, "dtype", None)
        if dt is None:
            continue
        if not isinstance(dt, torch.dtype):
            dt = _numpy_dtype_to_torch_dtype.get(dt)
            if dt is None:
                continue
        seen.add(dt)
    if not seen:
        return None
    for dtype in _dtype_coercion_precedence:
        if dtype in seen:
            return dtype
    raise ValueError(
        f"No coercible dtype among {seen}; supported dtypes (in precedence "
        f"order) are {_dtype_coercion_precedence}. Variants outside this set "
        "(e.g. complex32 or ROCm-only float8 types) must be cast explicitly.",
    )


_dtype_coercion_precedence: tuple[torch.dtype, ...] = (
    torch.complex128,
    torch.complex64,
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float8_e5m2,
    torch.float8_e4m3fn,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint64,
    torch.uint32,
    torch.uint16,
    torch.uint8,
    torch.bool,
)

_numpy_dtype_to_torch_dtype: dict[type | np.dtype[Any], torch.dtype] = {
    np.bool_: torch.bool,
    np.uint8: torch.uint8,
    np.uint16: torch.uint16,
    np.uint32: torch.uint32,
    np.uint64: torch.uint64,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
    np.dtypes.BoolDType: torch.bool,
    np.dtypes.UInt8DType: torch.uint8,
    np.dtypes.UInt16DType: torch.uint16,
    np.dtypes.UInt32DType: torch.uint32,
    np.dtypes.UInt64DType: torch.uint64,
    np.dtypes.Int8DType: torch.int8,
    np.dtypes.Int16DType: torch.int16,
    np.dtypes.Int32DType: torch.int32,
    np.dtypes.Int64DType: torch.int64,
    np.dtypes.Float16DType: torch.float16,
    np.dtypes.Float32DType: torch.float32,
    np.dtypes.Float64DType: torch.float64,
    np.dtypes.Complex64DType: torch.complex64,
    np.dtypes.Complex128DType: torch.complex128,
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("uint16"): torch.uint16,
    np.dtype("uint32"): torch.uint32,
    np.dtype("uint64"): torch.uint64,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
    np.dtypes.UShortDType: torch.uint16,
    np.dtypes.UIntDType: torch.uint32,
    np.dtypes.ULongDType: torch.uint64,
    np.dtypes.IntDType: torch.int32,
    np.dtypes.LongDType: torch.int64,
}
