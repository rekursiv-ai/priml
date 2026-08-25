# pyright: reportPrivateUsage=false
from __future__ import annotations

from unittest.mock import patch

from torch import Tensor

import numpy as np
import pytest
import torch

from priml import memory
from priml.math.custom_types import Tensorable
from priml.memory import (
    convert_to_tensor,
    is_private_conversion,
    shares_storage,
    shares_storage_for_compile,
)


def test_single_python_scalar():
    result = convert_to_tensor(42)
    assert isinstance(result, Tensor)
    assert result.item() == 42


def test_multiple_python_scalars():
    result = convert_to_tensor(1, 2.5, True)
    assert len(result) == 3
    torch.testing.assert_close(result[0], torch.tensor(1.0))
    torch.testing.assert_close(result[1], torch.tensor(2.5))
    torch.testing.assert_close(result[2], torch.tensor(1.0))


def test_numpy_array():
    x = np.array([1, 2, 3], dtype=np.int32)
    result = convert_to_tensor(x)
    assert result.dtype == torch.int32
    torch.testing.assert_close(result, torch.tensor([1, 2, 3], dtype=torch.int32))


def test_torch_tensor():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    result = convert_to_tensor(x)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, x)


@pytest.mark.parametrize(
    ("dtype1", "dtype2", "expected_dtype"),
    [
        (torch.complex128, torch.float32, torch.complex128),
        (torch.complex64, torch.float64, torch.complex64),
        (torch.float64, torch.float32, torch.float64),
        (torch.float32, torch.int64, torch.float32),
        (torch.float16, torch.int32, torch.float16),
        (torch.bfloat16, torch.int16, torch.bfloat16),
        (torch.int64, torch.int32, torch.int64),
        (torch.int32, torch.uint64, torch.int32),
        (torch.int16, torch.uint32, torch.int16),
        (torch.uint64, torch.uint32, torch.uint64),
        (torch.uint32, torch.uint16, torch.uint32),
        (torch.uint16, torch.uint8, torch.uint16),
        (torch.uint8, torch.bool, torch.uint8),
    ],
)
def test_type_coercion_precedence(
    dtype1: torch.dtype,
    dtype2: torch.dtype,
    expected_dtype: torch.dtype,
) -> None:
    x = torch.tensor([1], dtype=dtype1)
    y = torch.tensor([2], dtype=dtype2)
    result = convert_to_tensor(x, y)
    assert result[0].dtype == expected_dtype
    assert result[1].dtype == expected_dtype


def test_explicit_dtype_override():
    x = torch.tensor([1], dtype=torch.int32)
    y = torch.tensor([2.5], dtype=torch.float64)
    result = convert_to_tensor(x, y, dtype=torch.float32)
    assert result[0].dtype == torch.float32
    assert result[1].dtype == torch.float32


@pytest.mark.gpu_torch_cuda
def test_device_parameter():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x = torch.tensor([1, 2, 3])
    result = convert_to_tensor(x, device="cuda")
    assert result[0].device.type == "cuda"


def test_dtype_hint_with_python_scalars():
    result = convert_to_tensor(1, 2, 3, dtype_hint=torch.float64)
    assert result[0].dtype == torch.float64
    assert result[1].dtype == torch.float64
    assert result[2].dtype == torch.float64


def test_dtype_hint_ignored_when_dtype_present():
    x = torch.tensor([1], dtype=torch.float32)
    result = convert_to_tensor(x, dtype_hint=torch.float64)
    assert result[0].dtype == torch.float32


def test_mixed_numpy_and_torch():
    x = np.array([1, 2], dtype=np.float32)
    y = torch.tensor([3, 4], dtype=torch.float64)
    result = convert_to_tensor(x, y)
    assert result[0].dtype == torch.float64
    assert result[1].dtype == torch.float64


def test_complex_numbers():
    x = 1 + 2j
    result = convert_to_tensor(x)
    assert result.dtype == torch.complex64
    assert result.item() == (1 + 2j)


def test_bfloat16_dtype():
    x = torch.tensor([1.0], dtype=torch.bfloat16)
    y = torch.tensor([2], dtype=torch.int32)
    result = convert_to_tensor(x, y)
    assert result[0].dtype == torch.bfloat16
    assert result[1].dtype == torch.bfloat16


def test_float8_e5m2_dtype():
    if not hasattr(torch, "float8_e5m2"):
        pytest.skip("float8_e5m2 not available in this PyTorch version")
    x = torch.tensor([1.0], dtype=torch.float8_e5m2)
    y = torch.tensor([2], dtype=torch.int32)
    result = convert_to_tensor(x, y)
    assert result[0].dtype == torch.float8_e5m2
    assert result[1].dtype == torch.float8_e5m2


def test_float8_e4m3fn_dtype():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("float8_e4m3fn not available in this PyTorch version")
    x = torch.tensor([1.0], dtype=torch.float8_e4m3fn)
    y = torch.tensor([2], dtype=torch.int32)
    result = convert_to_tensor(x, y)
    assert result[0].dtype == torch.float8_e4m3fn
    assert result[1].dtype == torch.float8_e4m3fn


def test_nested_sequence():
    x = [[1, 2], [3, 4]]
    result = convert_to_tensor(x)
    assert result.shape == (2, 2)
    torch.testing.assert_close(result, torch.tensor([[1, 2], [3, 4]]))


@pytest.mark.parametrize(
    ("np_dtype", "expected_torch_dtype"),
    [
        (np.bool_, torch.bool),
        (np.uint8, torch.uint8),
        (np.uint16, torch.uint16),
        (np.uint32, torch.uint32),
        (np.uint64, torch.uint64),
        (np.int8, torch.int8),
        (np.int16, torch.int16),
        (np.int32, torch.int32),
        (np.int64, torch.int64),
        (np.float16, torch.float16),
        (np.float32, torch.float32),
        (np.float64, torch.float64),
        (np.complex64, torch.complex64),
        (np.complex128, torch.complex128),
    ],
)
def test_numpy_dtype_mapping(
    np_dtype: type[np.generic],
    expected_torch_dtype: torch.dtype,
) -> None:
    x = np.array([1, 2, 3], dtype=np_dtype)
    result = convert_to_tensor(x)
    assert result[0].dtype == expected_torch_dtype


def test_bool_dtype():
    x = torch.tensor([True, False], dtype=torch.bool)
    y = torch.tensor([1, 0], dtype=torch.int32)
    result = convert_to_tensor(x, y)
    assert result[0].dtype == torch.int32
    assert result[1].dtype == torch.int32


def test_as_tensors_while_compiling():
    """Test convert_to_tensor behavior during compilation (lines 147-149)."""
    # Simulate compilation context
    with patch.object(torch.compiler, "is_compiling", return_value=True):
        x = torch.tensor([1.0, 2.0])
        y = torch.tensor([3.0, 4.0])
        result = convert_to_tensor(x, y)
        assert len(result) == 2
        assert isinstance(result[0], Tensor)
        assert isinstance(result[1], Tensor)


def test_compile_honors_dtype():
    """Under the compile path, an explicit dtype must be applied, not dropped.

    Regression: the compile branch returned inputs unchanged, so eager and
    traced execution disagreed on output dtype. ``is_compiling`` is patched
    (rather than invoking ``torch.compile``) to exercise the branch directly.
    """
    x = torch.tensor([1.0, 2.0], dtype=torch.float32)
    eager = convert_to_tensor(x, dtype=torch.float64)
    with patch.object(torch.compiler, "is_compiling", return_value=True):
        compiled = convert_to_tensor(x, dtype=torch.float64)
    assert eager.dtype == torch.float64
    assert compiled.dtype == torch.float64
    torch.testing.assert_close(compiled, eager)


def test_compile_honors_device():
    """Under the compile path, an explicit device must be applied."""
    with patch.object(torch.compiler, "is_compiling", return_value=True):
        result = convert_to_tensor(torch.tensor([1.0, 2.0]), device="cpu")
    assert result.device.type == "cpu"


def test_compile_resolves_dtype_when_unspecified():
    """No explicit dtype under compile still unifies inputs by precedence."""
    x = torch.tensor([1], dtype=torch.int32)
    y = torch.tensor([2.0], dtype=torch.float64)
    with patch.object(torch.compiler, "is_compiling", return_value=True):
        a, b = convert_to_tensor(x, y)
    assert a.dtype == torch.float64
    assert b.dtype == torch.float64


def test_a_scalar_argument_survives_a_fullgraph_compile() -> None:
    """A Python number must convert, not abort the graph.

    The compile branch asserted every argument was already a Tensor, so every
    helper whose second argument is an ordinary float -- ``softcap(x, 1.0)``,
    ``soft_threshold``, ``ste_round``, ``sinh_arcsinh``, ``safe_pow`` -- raised
    inside dynamo under ``fullgraph=True``. Compiled for real rather than with
    a patched ``is_compiling``, because a patch cannot observe a graph break.
    """

    @torch.compile(fullgraph=True)
    def scale(x: Tensor) -> Tensor:
        a, b = convert_to_tensor(x, 2.0)
        return a * b

    x = torch.tensor([1.0, 2.0])
    torch.testing.assert_close(scale(x), torch.tensor([2.0, 4.0]))


def test_as_tensors_unrecognized_dtype():
    """``_resolve_dtype`` raises when no precedence entry matches."""
    x = torch.tensor([1.0], dtype=torch.float32)
    y = torch.tensor([2.0], dtype=torch.float64)

    original_precedence = memory._dtype_coercion_precedence
    try:
        memory._dtype_coercion_precedence = ()
        with pytest.raises(ValueError, match="No coercible dtype"):
            convert_to_tensor(x, y)
    finally:
        memory._dtype_coercion_precedence = original_precedence


def _pairs() -> list[tuple[str, Tensorable, Tensorable, bool]]:
    """Named (x, y, shared) triples; built once so views keep their parents."""
    parent = torch.arange(8, dtype=torch.float32)
    arr = np.arange(8, dtype=np.float32)
    wrapped = torch.from_numpy(arr)
    buffer = np.frombuffer(bytearray(32), dtype=np.float32)
    return [
        ("tensor with itself", parent, parent, True),
        # Every way torch spells a view. Each reports a different data_ptr
        # than its parent, so a span is what sees through them.
        ("slice with parent", parent[2:5], parent, True),
        ("reshape with parent", parent.view(2, 4), parent, True),
        ("transpose with parent", parent.view(2, 4).T, parent, True),
        ("step slice with parent", parent[::2], parent, True),
        ("unsqueeze with parent", parent.unsqueeze(0), parent, True),
        # Same allocation, no common byte. A base-address comparison calls
        # every one of these shared.
        ("disjoint slices", parent[0:2], parent[4:6], False),
        ("adjacent slices", parent[0:4], parent[4:8], False),
        ("overlapping slices", parent[0:5], parent[4:8], True),
        ("disjoint numpy slices", arr[0:2], arr[4:6], False),
        ("disjoint across containers", wrapped[0:2], arr[4:6], False),
        # numpy reverses by negating the stride and pointing at the LAST
        # element, so the span runs backwards from there; counting only the
        # forward reach would place it past the end of the buffer.
        ("negative-stride numpy", arr[::-1], arr, True),
        ("negative-stride disjoint", arr[0:2][::-1], arr[4:6], False),
        # ``torch.flip`` copies rather than restriding, so this pair really is
        # unshared -- the assertion pins the copy, not the span arithmetic.
        ("torch flip with parent", parent.flip(0), parent, False),
        # The same view kinds on the numpy side, which walks ``.base``.
        ("numpy slice with array", arr[2:5], arr, True),
        ("numpy reshape with array", arr.reshape(2, 4), arr, True),
        ("numpy transpose with array", arr.reshape(2, 4).T, arr, True),
        ("numpy step slice with array", arr[::2], arr, True),
        # Views that cross the container boundary in both directions.
        ("from_numpy with array", wrapped, arr, True),
        ("tensor slice with array", wrapped[2:5], arr, True),
        ("tensor view with array", wrapped.view(2, 4), arr, True),
        ("numpy view of tensor with tensor", parent.numpy()[2:5], parent, True),
        ("numpy view with tensor view", parent.numpy()[2:5], parent[1:6], True),
        # ``.base`` ends on a non-ndarray for these, so the walk must stop at
        # the last array rather than assume a homogeneous chain.
        ("frombuffer view with frombuffer", buffer[1:], buffer, True),
        ("asarray of memoryview with array", np.asarray(memoryview(arr)), arr, True),
        ("clone with parent", parent.clone(), parent, False),
        ("numpy copy with array", arr.copy(), arr, False),
        ("widened with array", wrapped.double(), arr, False),
        ("unrelated tensors", torch.zeros(4), torch.zeros(4), False),
        # Zero-size allocations all report address 0.
        ("unrelated empties", torch.empty(0), torch.empty(0), False),
        ("empty array with array", np.array([], dtype=np.float32), arr, False),
        ("list with tensor", [1.0, 2.0], parent, False),
        ("distinct lists", [1.0, 2.0], [1.0, 2.0], False),
        ("meta with meta", torch.zeros(3, device="meta"), torch.zeros(3), False),
        # Bufferless values alias only by being the same object, and then a
        # write through one name IS visible through the other.
        ("list with itself", shared_list := [1.0, 2.0], shared_list, True),
        ("nested inner list", [inner := [1.0]], [inner], False),
        ("inner list with itself", inner, inner, True),
        ("empty tensor with itself", empty := torch.empty(0), empty, True),
        ("meta with itself", meta := torch.zeros(3, device="meta"), meta, True),
    ]


@pytest.mark.parametrize(("name", "x", "y", "shared"), _pairs())
def test_shares_storage(name: str, x: Tensorable, y: Tensorable, shared: bool) -> None:
    """Views, wraps, copies and bufferless values, in both argument orders."""
    assert shares_storage(x, y) is shared, name
    assert shares_storage(y, x) is shared, f"{name} (reversed)"


def _leak_pair(kind: str) -> tuple[Tensorable, Tensorable]:
    """Build a (writer, observer) pair; the parent must outlive its view."""
    parent = torch.arange(8, dtype=torch.float32)
    arr = np.arange(8, dtype=np.float32)
    if kind == "tensor slice and parent":
        return (parent[2:5], parent)
    if kind == "tensor transpose and parent":
        return (parent.view(2, 4).T, parent)
    if kind == "zero-copy wrap and array":
        return (torch.from_numpy(arr), arr)
    if kind == "tensor view and its source array":
        return (torch.from_numpy(arr)[2:5], arr)
    if kind == "numpy view and its source tensor":
        return (parent.numpy()[2:5], parent)
    if kind == "frombuffer view and parent":
        buffer = np.frombuffer(bytearray(32), dtype=np.float32)
        return (buffer[1:], buffer)
    if kind == "clone and parent":
        return (parent.clone(), parent)
    return (arr.copy(), arr)


@pytest.mark.parametrize(
    "name",
    [
        "tensor slice and parent",
        "tensor transpose and parent",
        "zero-copy wrap and array",
        "tensor view and its source array",
        "numpy view and its source tensor",
        "frombuffer view and parent",
        "clone and parent",
        "numpy copy and array",
    ],
)
def test_shares_storage_predicts_whether_a_write_leaks(name: str) -> None:
    """Ground truth: the verdict must match what mutation actually does.

    Without this the predicate could only agree with itself, which is exactly
    how an address comparison that ignores view offsets passes review and then
    corrupts a caller's tensor. Covers both view kinds against both container
    kinds, since a ``.base`` walk and a storage pointer can fail separately.
    """
    x, y = _leak_pair(name)
    writer = x if isinstance(x, Tensor) else torch.from_numpy(np.asarray(x))

    if isinstance(y, Tensor):
        before = y.clone()
        writer.add_(100.0)
        leaked = not torch.equal(y, before)
    else:
        observer = np.asarray(y)
        before_np = observer.copy()
        writer.add_(100.0)
        leaked = not np.array_equal(observer, before_np)

    assert leaked is shares_storage(x, y), name


@pytest.mark.gpu_torch_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shares_storage_on_cuda_tensors() -> None:
    """Device pointers are addresses too, so the span arithmetic is unchanged.

    Worth its own test because the two rejected alternatives both fail here:
    ``np.shares_memory`` needs a ``.numpy()`` round-trip that raises on a CUDA
    tensor, and a host/device pair must never compare shared even though the
    two devices can hand out identical integers.
    """
    parent = torch.arange(8, dtype=torch.float32, device="cuda")

    assert shares_storage(parent, parent)
    assert shares_storage(parent[2:5], parent)
    assert shares_storage(parent.view(2, 4).T, parent)
    assert not shares_storage(parent[0:2], parent[4:6])
    assert not shares_storage(parent.clone(), parent)

    before = parent.clone()
    parent[2:5].add_(100.0)
    assert not torch.equal(parent, before), "the view really writes through"

    # A copy across devices shares nothing, whatever the raw integers say.
    assert not shares_storage(parent.cpu(), parent)
    assert not shares_storage(parent, np.arange(8, dtype=np.float32))

    # Asserted on the span itself rather than through ``shares_storage``: two
    # live allocations rarely land on the same integer, so a cross-device pair
    # usually passes for the wrong reason. The tagged device is what makes the
    # answer independent of that luck.
    assert (
        memory._byte_span(parent)[2]
        != (memory._byte_span(torch.arange(8, dtype=torch.float32))[2])
    ), "host and device tensors were given the same device"

    if torch.cuda.device_count() > 1:
        other = torch.arange(8, dtype=torch.float32, device="cuda:1")
        assert memory._byte_span(parent)[2] != memory._byte_span(other)[2]
        assert not shares_storage(parent, other)


def test_shares_storage_for_compile_matches_eager_under_fullgraph() -> None:
    """The custom op must agree with the eager predicate and keep one graph.

    Compiled for real rather than with a patched ``is_compiling``: a patch
    cannot observe a graph break, and the break is the thing at issue. Dynamo
    rejects ``data_ptr()`` arithmetic outright (pytorch#165408), so without the
    op wrapper ``fullgraph=True`` fails to trace at all.
    """
    parent = torch.arange(8, dtype=torch.float32)
    cases = {
        "view": (parent[2:5], parent),
        "step view": (parent[::2], parent),
        "disjoint": (parent[0:2], parent[4:6]),
        "clone": (parent.clone(), parent),
        "self": (parent, parent),
    }

    torch._dynamo.reset()
    compiled = torch.compile(shares_storage_for_compile, fullgraph=True)
    for name, (x, y) in cases.items():
        assert bool(compiled(x, y)) is shares_storage(x, y), name


def test_shares_storage_for_compile_branches_without_a_graph_break() -> None:
    """``torch.cond`` on the op keeps the whole thing in one graph.

    A plain ``if`` on the returned tensor would be a data-dependent jump, which
    breaks the graph and defeats the wrapper -- so the branch is what has to be
    asserted, not just the value.
    """

    def _bump(t: Tensor) -> Tensor:
        return t + 1.0

    def _double(t: Tensor) -> Tensor:
        return t * 2.0

    def branchy(x: Tensor, y: Tensor) -> Tensor:
        return torch.cond(shares_storage_for_compile(x, y), _bump, _double, (x,))

    parent = torch.arange(8, dtype=torch.float32)
    torch._dynamo.reset()
    explained = torch._dynamo.explain(branchy)(parent[2:5], parent)
    assert explained.graph_break_count == 0, explained.break_reasons

    torch._dynamo.reset()
    compiled = torch.compile(branchy, fullgraph=True)
    torch.testing.assert_close(
        compiled(parent[2:5], parent), torch.tensor([3.0, 4.0, 5.0])
    )
    torch.testing.assert_close(compiled(parent.clone(), parent), parent * 2.0)


def test_shares_storage_raises_on_two_interleaved_strided_views() -> None:
    """The one case a byte span cannot decide, refused rather than guessed.

    ``p[::2]`` and ``p[1::2]`` overlap as ranges yet touch no common byte.
    Returning True would be a false alarm and False would license a corrupting
    write, so neither is honest; deciding it exactly is the NP-complete problem
    ``np.shares_memory`` solves.

    Only ONE side skipping bytes stays decidable: a dense range covers every
    byte between its ends, so an overlap really is a shared byte.
    """
    parent = torch.arange(8, dtype=torch.float32)
    arr = np.arange(8, dtype=np.float32)

    with pytest.raises(NotImplementedError, match="interleaved"):
        _ = shares_storage(parent[::2], parent[1::2])
    with pytest.raises(NotImplementedError, match="interleaved"):
        _ = shares_storage(arr[::2], arr[1::2])
    # It fires inside the op too, whose body runs eagerly on real tensors.
    with pytest.raises(NotImplementedError, match="interleaved"):
        _ = shares_storage_for_compile(parent[::2], parent[1::2])

    assert shares_storage(parent[::2], parent)
    assert shares_storage(parent[1::2], parent)
    # Equal first bytes settle it whatever the strides, so an identical pair
    # must not raise.
    assert shares_storage(parent[::2], parent[::2])


def test_shares_storage_sees_through_a_view_offset() -> None:
    """The defect that motivates comparing the allocation base.

    Both element-pointer spellings disagree with reality for a view, which is
    why neither identity nor ``data_ptr`` can answer this.
    """
    parent = torch.arange(8, dtype=torch.float32)
    view = parent[2:5]

    assert view.data_ptr() != parent.data_ptr(), "element pointers differ"
    assert view.untyped_storage().data_ptr() == parent.untyped_storage().data_ptr()
    assert shares_storage(view, parent)


def test_is_private_conversion_answers_every_conversion_shape() -> None:
    """The pairs it is contracted for: same object, wrap, or fresh allocation.

    The numpy row is the whole reason this is not ``converted is not
    original``: ``as_tensor`` wraps a matching-dtype array zero-copy, so a NEW
    tensor object still holds the caller's buffer.
    """
    array = np.array([1.0, 2.0], dtype=np.float32)
    tensor = torch.tensor([1.0, 2.0])

    # Wrapped or handed back: the caller can still see writes.
    assert not is_private_conversion(convert_to_tensor(array), array)
    assert not is_private_conversion(convert_to_tensor(tensor), tensor)
    # Converted or materialized: the buffer is the function's to spend.
    assert is_private_conversion(convert_to_tensor(array, dtype=torch.float64), array)
    assert is_private_conversion(convert_to_tensor(tensor, dtype=torch.float64), tensor)
    assert is_private_conversion(convert_to_tensor([1.0, 2.0]), [1.0, 2.0])
    assert is_private_conversion(convert_to_tensor(5.0), 5.0)


def test_is_private_conversion_is_wrong_for_a_view_by_construction() -> None:
    """Pins the footgun the docstring warns about, in the corrupting direction.

    A view reports a different ``data_ptr`` than the storage it shares, so this
    calls it private and a caller acting on the answer overwrites live data.
    ``convert_to_tensor`` never returns a view, which is the invariant that
    makes the cheap check sound -- and the reason the general question needs
    ``shares_storage``. Asserting the WRONG answer keeps the boundary honest:
    widening the contract fails here rather than silently corrupting.
    """
    parent = torch.arange(8, dtype=torch.float32)
    view = parent[2:5]

    assert is_private_conversion(view, parent), "contract limit moved"
    assert shares_storage(view, parent), "the general predicate gets it right"

    before = parent.clone()
    view.add_(100.0)
    assert not torch.equal(parent, before), "the write really reaches the parent"


def test_is_private_conversion_traces_under_fullgraph_compile() -> None:
    """Traceability is the reason ``float2rgb`` uses this, not ``shares_storage``.

    ``shares_storage`` reads storage pointers and strides, which Dynamo
    rejects; a caller would need an ``is_compiling()`` branch and would then
    run different code eager versus compiled. This must stay traceable -- an
    ``object()`` sentinel here previously failed the whole graph with
    ``Dynamo does not know how to trace builtin operator 'object'``.
    """

    def convert_then_check(x: Tensor) -> Tensor:
        converted = convert_to_tensor(x, dtype=torch.float32)
        return converted * 2.0 if is_private_conversion(converted, x) else converted

    x = torch.arange(4, dtype=torch.float16)
    torch._dynamo.reset()
    explained = torch._dynamo.explain(convert_then_check)(x)
    assert explained.graph_break_count == 0, explained.break_reasons

    torch._dynamo.reset()
    compiled = torch.compile(convert_then_check, fullgraph=True)
    torch.testing.assert_close(compiled(x), convert_then_check(x))


if __name__ == "__main__":
    from priml.lib.testing.main import test_main

    test_main(__file__)
