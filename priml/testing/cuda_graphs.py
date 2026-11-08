"""CUDA graphs on the CPU: a capture records the ops it dispatches; a replay re-runs them.

``cpu_cuda_graphs(monkeypatch)`` swaps the ``torch.cuda`` calls a capturing
step makes (graphs, streams, devices, the default generators) for CPU
stand-ins, so the code that captures runs unchanged on CPU tensors. The
stand-in keeps the three properties a real capture has and an eager rerun
lacks:

- Python runs once, at capture. A replay re-runs the recorded ATen ops on the
  same tensors, so control flow, shapes and Python state are frozen at
  capture, and every result lands in the tensor the capture returned.
- A capture computes nothing: it writes no tensor and draws from no
  generator, and each result is a new, unfilled tensor until a replay fills it.
- Nothing in the step may read a device value on the host or choose a shape
  from one. Those ops raise during capture, as CUDA's capture refuses them.

The recorded op names are ``FakeGraph.ops``. Real capture -- kernels, streams,
the allocator -- is out of reach; ``priml/private/cuda_parity.py`` runs the
tests this stands in for on CUDA.
"""

from __future__ import annotations

from collections.abc import Hashable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, cast, override

import threading

from torch import Tensor
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves, tree_map_only

import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    import pytest


# Each reads a device value on the host (a sync) or sizes its output from one (a
# data-dependent shape); CUDA refuses both while a stream captures.
CAPTURE_BREAKING_OPS: frozenset[str] = frozenset(
    {
        "aten::_local_scalar_dense",
        "aten::item",
        "aten::is_nonzero",
        "aten::equal",
        "aten::nonzero",
        "aten::nonzero_numpy",
        "aten::masked_select",
        "aten::_unique2",
        "aten::unique_dim",
        "aten::unique_consecutive",
        "aten::bincount",
    },
)


# Each returns its input under another name without its schema saying so.
_ALIASING_OPS: frozenset[str] = frozenset({"aten::_unsafe_view", "aten::lift_fresh"})


class CaptureError(RuntimeError):
    """A captured step did something a CUDA capture refuses."""


class FakeStream:
    """A CUDA stream on the CPU, where work runs in program order."""

    cuda_stream = 0
    device = torch.device("cpu")

    def __init__(self, device: object = None, priority: int = 0) -> None:
        """Make a stream: the CPU has one, whatever device is asked; it reports its priority."""
        del device
        self.priority = priority

    def wait_stream(self, stream: object) -> None:
        """Order nothing: the CPU already did."""
        del stream

    def synchronize(self) -> None:
        """Wait for nothing: the CPU already did."""


@dataclass(slots=True, kw_only=True)
class _Op:
    """One recorded op: what it was called with, and the fresh tensors it made."""

    func: OpOverload[..., object]
    args: tuple[object, ...]
    kwargs: dict[str, object]
    fresh: list[tuple[int, Tensor]]


@dataclass(frozen=True, slots=True, kw_only=True)
class _Plan:
    """How recording one op treats its call, read once from its schema."""

    refused: bool
    """It reads a device value on the host or sizes its output from one."""

    indexes: bool
    """It indexes, which sizes its output by a boolean index's values."""

    view: bool
    """It changes how its input's memory is addressed, reading no value: taken
    for real, once."""

    pointwise: bool
    """It maps broadcast elements, so no value sizes its results: a call on
    floating tensors, which raises on none, is sized by running it, its result
    dropped."""

    returned: tuple[tuple[int, str], ...] | None
    """The (position, name) of each argument it returns as a result; None when
    its results are new tensors."""


@dataclass(slots=True, kw_only=True)
class FakeGraph:
    """A captured graph: the ops a capture recorded, re-run on each replay."""

    keep_graph: bool = False
    recorded: list[_Op] = field(default_factory=list)
    replays: int = 0

    @property
    def ops(self) -> list[str]:
        """The recorded ops' names, in capture order."""
        return [op.func.name() for op in self.recorded]

    def replay(self) -> None:
        """Re-run the recorded ops; copy each fresh result into the captured one."""
        if not self.recorded:
            raise CaptureError("A graph was replayed with nothing captured.")
        self.replays += 1
        with torch.no_grad():
            for op in self.recorded:
                out = op.func(*op.args, **op.kwargs)
                if not op.fresh:
                    continue
                results = [out] if isinstance(out, Tensor) else _tensors(out)
                for index, captured in op.fresh:
                    captured.copy_(results[index])

    def pool(self) -> tuple[int, int]:
        """Return a memory-pool handle; the CPU has one allocator."""
        return (0, id(self))


class _Recording(TorchDispatchMode):
    """Record every dispatched op into ``graph``, computing nothing.

    A view, or an in-place one (``t_``), is taken for real and not recorded: it
    reads no value and launches no kernel, so a CUDA graph holds no node for it,
    and the alias it leaves reads each replay's writes. An
    op whose results are its arguments (``add_``, ``copy_``) returns them
    unwritten. Any other op returns new, unfilled tensors -- the buffers a
    replay fills -- sized by a run on meta copies, or a floating pointwise
    op's by a run on its arguments, whose result is dropped.
    """

    def __init__(self, graph: FakeGraph, cuda: FakeCuda) -> None:
        super().__init__()
        self.graph = graph
        self.cuda = cuda

    @override
    def __torch_dispatch__(
        self,
        func: OpOverload[..., object],
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        del types
        kwargs = kwargs or {}
        plan = self.cuda.plans.get(func)
        if plan is None:
            plan = self.cuda.plans[func] = _plan(func)
        if plan.refused or (plan.indexes and _masks(args)):
            raise CaptureError(
                f"{func.name()} is not permitted while a stream captures.",
            )
        if plan.view:
            return func(*args, **kwargs)
        fresh: list[tuple[int, Tensor]] = []
        if plan.returned is not None:
            out = _returned(plan.returned, args, kwargs)
        else:
            key = _signature(func, args, kwargs)
            shaped = self.cuda.shapes.get(key)
            if shaped is None:
                # A meta run costs some fifty host microseconds an op; this one, a
                # few. An integer one could divide by an unfilled zero, and raise.
                shaped = self.cuda.shapes[key] = (
                    func(*args, **kwargs)
                    if plan.pointwise and _floating(args, kwargs)
                    else _shaped(func, args, kwargs)
                )
            if isinstance(shaped, Tensor):
                out = _unfilled(shaped)
                fresh = [(0, out)]
            else:
                out = tree_map_only(Tensor, _unfilled, shaped)
                fresh = list(enumerate(_tensors(out)))
        self.graph.recorded.append(
            _Op(func=func, args=args, kwargs=kwargs, fresh=fresh),
        )
        return out


@dataclass(slots=True, kw_only=True)
class FakeCuda:
    """The ``torch.cuda`` calls a capturing step makes, on the CPU."""

    graphs: list[FakeGraph] = field(default_factory=list)
    capturing: set[int] = field(default_factory=set)
    """The threads capturing now; other threads' buffers run eagerly meanwhile."""

    shapes: dict[Hashable, object] = field(default_factory=dict)
    """Each op signature's meta results: a step's ops recur across its captures."""

    plans: dict[OpOverload[..., object], _Plan] = field(default_factory=dict)
    """Each op's plan: a capture dispatches a few ops thousands of times."""

    def new_graph(self, *, keep_graph: bool = False) -> FakeGraph:
        """Return a new, empty graph."""
        self.graphs.append(FakeGraph(keep_graph=keep_graph))
        return self.graphs[-1]

    @contextmanager
    def graph(
        self,
        graph: FakeGraph,
        *,
        pool: object = None,
        stream: object = None,
        capture_error_mode: str = "global",
    ) -> Generator[None]:
        """Record the ops dispatched inside into ``graph``, computing nothing.

        Args:
          graph: The graph to record into; it must be empty.
          pool: A memory pool to share; the CPU has one allocator.
          stream: The capture stream; the CPU has one.
          capture_error_mode: CUDA's scope for refusing unsafe calls; the
            stand-in refuses them on every thread.

        Yields:
          None: Once, while the step to capture runs.

        Raises:
          CaptureError: The graph was captured into before.

        """
        del pool, stream, capture_error_mode
        if graph.recorded:
            raise CaptureError("A graph can be captured into once.")
        self.capturing.add(threading.get_ident())
        try:
            with _Recording(graph, self):
                yield
        finally:
            self.capturing.discard(threading.get_ident())


def cpu_cuda_graphs(monkeypatch: pytest.MonkeyPatch) -> FakeCuda:
    """Install CPU stand-ins for the ``torch.cuda`` graph, stream and device calls.

    The CPU's default generator becomes CUDA device 0's, and its state each
    device's, so a step that samples from "its device's default generator"
    samples from the CPU's.

    Args:
      monkeypatch: Undoes the stand-ins when the test ends.

    Returns:
      cuda: The stand-in, which lists the graphs captured.

    """
    cuda = FakeCuda()
    monkeypatch.setattr(torch.cuda, "CUDAGraph", cuda.new_graph)
    monkeypatch.setattr(torch.cuda, "graph", cuda.graph)
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", lambda: (0, 0))
    monkeypatch.setattr(torch.cuda, "Stream", FakeStream)
    monkeypatch.setattr(torch.cuda, "stream", _no_context)
    monkeypatch.setattr(torch.cuda, "current_stream", _current_stream)
    monkeypatch.setattr(torch.cuda, "default_stream", _current_stream)
    monkeypatch.setattr(torch.cuda, "device", _no_context)
    monkeypatch.setattr(torch.cuda, "synchronize", _synchronize)
    monkeypatch.setattr(torch.cuda, "default_generators", (torch.default_generator,))
    monkeypatch.setattr(torch.cuda, "get_rng_state", _get_rng_state)
    # Read on the host without dispatching an op, so the recorder cannot see them.
    for name in ("numpy", "tolist", "__array__"):
        monkeypatch.setattr(Tensor, name, _HostRead(cuda, name))
    return cuda


class _HostRead:
    """A ``Tensor`` method that reads values on the host: refused while capturing."""

    def __init__(self, cuda: FakeCuda, name: str) -> None:
        self.cuda = cuda
        self.name = name
        self.method: Callable[..., object] = getattr(Tensor, name)

    def __get__(self, tensor: Tensor | None, owner: type) -> Callable[..., object]:
        if tensor is None:
            return self.method
        if threading.get_ident() in self.cuda.capturing:
            raise CaptureError(
                f"Tensor.{self.name} is not permitted while a stream captures.",
            )
        return partial(self.method, tensor)


def _no_context(target: object = None) -> nullcontext[None]:
    """Enter a device or stream: the CPU has one of each."""
    del target
    return nullcontext()


def _current_stream(device: object = None) -> FakeStream:
    del device
    return FakeStream()


def _synchronize(device: object = None) -> None:
    """Wait for nothing: the CPU already did."""
    del device


def _get_rng_state(device: object = None) -> Tensor:
    """Return the default generator's state: the CPU's stands in for each device's."""
    del device
    return torch.default_generator.get_state()


def _plan(func: OpOverload[..., object]) -> _Plan:
    """Read from ``func``'s name and schema how recording treats its calls."""
    name = func.name()
    view = func.is_view or torch.Tag.inplace_view in func.tags or name in _ALIASING_OPS
    schema = func._schema  # noqa: SLF001 -- An op records which arguments it returns nowhere else.
    returned: tuple[tuple[int, str], ...] | None = None
    # An op returning nothing (``_foreach_add_``) returns no argument. A view's
    # results alias its input without being it.
    if not view and all(result.alias_info is not None for result in schema.returns):
        # An aliased result names the argument it is by sharing its alias set.
        named = {
            frozenset(argument.alias_info.before_set): (index, argument.name)
            for index, argument in enumerate(schema.arguments)
            if argument.alias_info is not None
        }
        returned = tuple(
            named[frozenset(result.alias_info.before_set)]
            for result in schema.returns
            if result.alias_info is not None
        )
    return _Plan(
        refused=name.partition(".")[0] in CAPTURE_BREAKING_OPS,
        indexes=name.startswith("aten::index"),
        view=view,
        # Running one draws from no generator: no aten op is tagged both
        # pointwise and nondeterministic_seeded.
        pointwise=torch.Tag.pointwise in func.tags,
        returned=returned,
    )


def _returned(
    returned: tuple[tuple[int, str], ...],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Return the arguments an op returns as its results: one, a tuple, or None."""
    results = [
        args[index] if index < len(args) else kwargs[name] for index, name in returned
    ]
    if not results:
        return None
    return results[0] if len(results) == 1 else tuple(results)


def _signature(
    func: OpOverload[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> Hashable:
    """Key an op call by what sizes its results: tensors by layout, the rest by value."""
    return (
        func,
        tuple(_layout(value) for value in args),
        tuple((name, _layout(value)) for name, value in kwargs.items()),
    )


def _layout(value: object) -> Hashable:
    if isinstance(value, Tensor):
        return (value.shape, value.stride(), value.dtype, value.requires_grad)
    if isinstance(value, (list, tuple)):
        return tuple(_layout(item) for item in cast("Sequence[object]", value))
    if isinstance(value, torch.Generator):
        return id(value)
    return cast("Hashable", value)


def _shaped(
    func: OpOverload[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    """Run ``func`` on meta copies of its arguments, for its results' shapes alone."""
    meta_args, meta_kwargs = tree_map_only(Tensor, _meta, (args, kwargs))
    if "device" in meta_kwargs:
        meta_kwargs["device"] = torch.device("meta")
    # A meta op draws nothing; a host generator would refuse the meta device.
    if "generator" in meta_kwargs:
        meta_kwargs["generator"] = None
    return func(*meta_args, **meta_kwargs)


def _floating(args: tuple[object, ...], kwargs: dict[str, object]) -> bool:
    """Whether every tensor an op call reads is floating point."""
    return all(
        t.is_floating_point()
        for t in tree_leaves((args, kwargs))
        if isinstance(t, Tensor)
    )


def _unfilled(shaped: Tensor) -> Tensor:
    """Allocate the host tensor a meta one describes, its values unwritten."""
    return torch.empty_strided(
        shaped.shape,
        shaped.stride(),
        dtype=shaped.dtype,
        device="cpu",
    )


def _meta(tensor: Tensor) -> Tensor:
    return torch.empty_strided(
        tensor.shape,
        tensor.stride(),
        dtype=tensor.dtype,
        device="meta",
        requires_grad=tensor.requires_grad,
    )


def _tensors(out: object) -> list[Tensor]:
    return [t for t in tree_leaves(out) if isinstance(t, Tensor)]


def _masks(args: tuple[object, ...]) -> bool:
    """Whether an indexing op selects by a boolean mask, sizing its output by it."""
    indices = args[1] if len(args) > 1 else None
    return any(
        isinstance(t, Tensor) and t.dtype == torch.bool for t in tree_leaves(indices)
    )
