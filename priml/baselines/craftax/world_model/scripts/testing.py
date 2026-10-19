"""Test support shared by the world-model scripts' tests: CUDA and a clock on the CPU.

The feature-gates and joint-speed scripts read ``torch.cuda`` and ``time``
directly. Their tests swap each script's ``torch`` for ``CpuTorch``, whose
``cuda`` is a ``FakeCuda``, and its ``time`` for a ``Clock``. A fake graph
re-runs the call captured into it on replay, so a graphed step is the eager
step it captured.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator


class FakeStream:
    """A CUDA stream on the CPU, which runs in program order."""

    def wait_stream(self, stream: FakeStream) -> None:
        """Order nothing."""
        del stream


class FakeGraph:
    """A CUDA graph whose replay re-runs the call captured into it."""

    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda
        self.body: Callable[[], object] | None = None

    def replay(self) -> None:
        """Count the replay, then run the captured call."""
        if self.body is None:
            raise ValueError("A graph replayed with nothing captured.")
        self.cuda.replays += 1
        self.body()


class FakeEvent:
    """A timing event that reads the replays since its start: 0.25 ms apiece."""

    def __init__(self, cuda: FakeCuda, *, enable_timing: bool) -> None:
        if not enable_timing:
            raise ValueError("A timing event needs enable_timing=True.")
        self.cuda = cuda
        self.replays = -1

    def record(self) -> None:
        """Note the replays so far."""
        self.replays = self.cuda.replays

    def elapsed_time(self, end: FakeEvent) -> float:
        """Return 0.25 ms per replay between this record and ``end``'s."""
        if self.replays < 0 or self.replays > end.replays:
            raise ValueError("Both events must be recorded, this one first.")
        return 0.25 * (end.replays - self.replays)


class FakeCuda:
    """The ``torch.cuda`` calls the scripts make, on the CPU.

    Graphs, streams and events serve the feature gates; the allocator calls,
    recorded in ``calls``, serve the joint-speed report.
    """

    def __init__(self) -> None:
        self.graphs: list[FakeGraph] = []
        self.capturing: FakeGraph | None = None
        self.replays = 0
        self.calls: list[str] = []
        # torch.cuda's class names, bound to this fake's factories.
        self.CUDAGraph = self.new_graph
        self.Stream = FakeStream
        self.Event = self.new_event

    def new_graph(self) -> FakeGraph:
        """Return a new graph."""
        self.graphs.append(FakeGraph(self))
        return self.graphs[-1]

    def new_event(self, *, enable_timing: bool) -> FakeEvent:
        """Return a timing event."""
        return FakeEvent(self, enable_timing=enable_timing)

    def current_stream(self) -> FakeStream:
        """Return the main stream."""
        return FakeStream()

    def stream(self, stream: FakeStream) -> nullcontext[None]:
        """Enter ``stream``: nothing to do on the CPU."""
        del stream
        return nullcontext()

    @contextmanager
    def graph(self, graph: FakeGraph, *, stream: FakeStream) -> Generator[None]:
        """Capture the call made inside into ``graph``.

        Args:
          graph: The graph the call made inside becomes the body of.
          stream: The capture stream; the CPU has only one.

        Yields:
          None: Once, while the call to capture runs.

        Raises:
          ValueError: The block made no call to capture.

        """
        del stream
        self.capturing = graph
        yield
        self.capturing = None
        if graph.body is None:
            raise ValueError("The capture made no engine call.")

    def synchronize(self, device: torch.device | None = None) -> None:
        """Record the wait: the CPU is already done."""
        if device not in {None, torch.device("cpu")}:
            raise ValueError(f"The CPU stands in for every device, not {device}.")
        self.calls.append("synchronize")

    def memory_allocated(self, device: torch.device) -> int:
        """Return 3 GiB resident."""
        del device
        self.calls.append("memory_allocated")
        return 3 << 30

    def reset_peak_memory_stats(self, device: torch.device) -> None:
        """Record the reset."""
        del device
        self.calls.append("reset_peak_memory_stats")

    def max_memory_allocated(self, device: torch.device) -> int:
        """Return a 5 GiB peak."""
        del device
        return 5 << 30

    def max_memory_reserved(self, device: torch.device) -> int:
        """Return 6 GiB reserved."""
        del device
        return 6 << 30

    def get_device_name(self, device: torch.device) -> str:
        """Name the device."""
        del device
        return "Fake H200"


class CpuTorch:
    """A script's ``torch`` on the CPU: CUDA faked, ``compile`` recorded."""

    def __init__(self) -> None:
        self.cuda = FakeCuda()
        self.compiled: list[dict[str, object]] = []
        # The rest of torch the scripts read, as torch's own.
        self.compiler, self.optim, self.no_grad = (
            torch.compiler,
            torch.optim,
            torch.no_grad,
        )
        self.Generator = torch.Generator
        self.arange, self.randint, self.where = torch.arange, torch.randint, torch.where
        self.full, self.tensor, self.stack = torch.full, torch.tensor, torch.stack
        self.zeros, self.zeros_like = torch.zeros, torch.zeros_like
        self.ones, self.ones_like = torch.ones, torch.ones_like
        self.rand, self.randn = torch.rand, torch.randn
        self.nan, self.bool, self.uint8 = torch.nan, torch.bool, torch.uint8
        self.int16, self.int32 = torch.int16, torch.int32

    def device(self, kind: str) -> torch.device:
        """Return the CPU for the script's ``cuda``."""
        if kind != "cuda":
            raise ValueError(f"The scripts ask for cuda, not {kind}.")
        return torch.device("cpu")

    def compile(
        self,
        function: Callable[..., object],
        **options: object,
    ) -> Callable[..., object]:
        """Record the options and return ``function`` itself."""
        self.compiled.append(options)
        return function


class Clock:
    """A ``time`` module whose ``perf_counter`` returns ``readings`` in order."""

    def __init__(self, readings: Iterable[float]) -> None:
        self._readings: Iterator[float] = iter(readings)

    def perf_counter(self) -> float:
        """Return the next reading."""
        return next(self._readings)
