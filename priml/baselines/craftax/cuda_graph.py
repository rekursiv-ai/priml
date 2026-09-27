"""Replaying a procedure's GPU work as one captured CUDA graph.

A step of this environment is thousands of kernels, each too small to occupy
the GPU, so launching them -- not running them -- is what a step costs.
Captured once and replayed as a graph, the same sequence costs one launch.

Capture imposes three rules on the procedure, which ``CudaGraphed`` relies on
rather than checks:

* It reads only tensors whose storage outlives the graph: buffers its owner
  allocated once and updates in place. A tensor REBOUND between calls is
  invisible to a replay, which still addresses the old memory. Results go the
  same way, or are tensors the procedure stores on its owner: made during
  capture, those live in the graph's own memory and every replay refills them.
* It never reads a device value on the host -- no ``.item()``, no ``bool(t)``,
  no copy from pageable memory. Capture raises on each of those.
* Its randomness comes from the generators it was registered with. A replay
  advances their state exactly as eager execution would, so it draws the same
  numbers the eager call would have drawn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class CudaGraphed:
    """A zero-argument procedure, run eagerly once and replayed as a graph after.

    The first call runs the procedure for real, which also performs every lazy
    initialization that capture must not: cuBLAS workspaces, cached index
    tensors, optimizer state. The second call captures it -- capture records
    without executing -- and replays it once, so every call has exactly one
    effect. Later calls only replay.
    """

    def __init__(
        self,
        procedure: Callable[[], None],
        *,
        generators: Sequence[torch.Generator] = (),
    ) -> None:
        """Wrap ``procedure`` without running it.

        Args:
          procedure: Updates its owner's buffers in place; see the module rules.
          generators: Every generator the procedure draws from.

        """
        self._procedure = procedure
        self._generators = tuple(generators)
        self._stream = torch.cuda.Stream()
        self._graph: torch.cuda.CUDAGraph | None = None
        self._warm = False

    def __call__(self) -> None:
        """Run the procedure once: eagerly, by capture and replay, or by replay."""
        if self._graph is None:
            if not self._warm:
                self._warm = True
                self._run_on_capture_stream()
                return
            graph = torch.cuda.CUDAGraph()
            for generator in self._generators:
                graph.register_generator_state(generator)
            with torch.cuda.graph(graph, stream=self._stream):
                self._procedure()
            self._graph = graph
        self._graph.replay()

    # The eager run shares the capture's stream so that anything it initializes per
    # stream, such as a cuBLAS workspace, is already in place when capture begins.
    def _run_on_capture_stream(self) -> None:
        """Run the procedure eagerly, ordered after and before the caller's work."""
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            self._procedure()
        torch.cuda.current_stream().wait_stream(self._stream)
